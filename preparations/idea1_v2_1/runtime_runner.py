"""Persistent E1 episode queue, bounded retries, calibration and finite mask generation."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
try:
    from .runtime_client import ROOT, SUITES, LIMITS, sha, write_json, rpc, input_digest, value_digest
    from .e1_preflight import MODELS, validate_candidates
except ImportError:
    from runtime_client import ROOT, SUITES, LIMITS, sha, write_json, rpc, input_digest, value_digest
    from e1_preflight import MODELS, validate_candidates

DEFAULT = ROOT / 'policies/realtime-vla-flash/results/idea1_v2_1_multimodel_kalman/run_20261016T045100Z'


@contextlib.contextmanager
def lock(path, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    write_json(temporary, data)
    os.replace(temporary, path)


class Ledger:
    def __init__(self, root):
        self.path = Path(root) / 'ledger.json'

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {'entries': []}

    def reserve(self, key, kind, retry=False, fix_note=''):
        with lock(self.path.with_suffix('.lock')):
            data = self.read()
            old = [r for r in data['entries'] if r['key'] == key]
            if old:
                if any(r['status'] == 'valid' for r in old):
                    raise ValueError('Valid episode, including task failure, cannot repeat')
                if len(old) != 1 or old[-1]['status'] != 'invalid' or not retry or not fix_note:
                    raise ValueError('Only one diagnosed invalid retry allowed per key')
                if sum(r['retry'] for r in data['entries']) >= 8:
                    raise ValueError('Model invalid retry budget exhausted')
            elif retry:
                raise ValueError('Cannot retry unseen key')
            token = f'{time.time_ns()}-{os.getpid()}'
            row = dict(token=token, key=key, kind=kind, status='reserved', retry=retry,
                       fix_note=fix_note, pid=os.getpid(), time=time.time())
            row['owner_start_ticks'] = Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()[19]
            data['entries'].append(row)
            atomic_json(self.path, data)
            return token

    def reconcile(self, note):
        if not note:
            raise ValueError('Explicit diagnosis required for dead reservations')
        with lock(self.path.with_suffix('.lock')):
            data = self.read()
            for row in data['entries']:
                if row['status'] != 'reserved':
                    continue
                try:
                    fields = Path(f"/proc/{row['pid']}/stat").read_text().split(') ', 1)[1].split()
                    alive = fields[0] not in ('Z', 'X') and fields[19] == row.get('owner_start_ticks')
                except FileNotFoundError:
                    alive = False
                if alive:
                    continue
                result = Path(row.get('directory', '/nonexistent')) / 'validated.json'
                if result.exists():
                    checked = json.loads(result.read_text())
                    row.update(status='valid', success=checked['success'], actions=checked['actions'], predictions=checked['predictions'])
                else:
                    row.update(status='invalid', error='Dead reservation owner')
                row['reconciliation_note'] = note
            atomic_json(self.path, data)

    def update(self, token, **fields):
        with lock(self.path.with_suffix('.lock')):
            data = self.read()
            row = next(r for r in data['entries'] if r['token'] == token)
            row.update(fields)
            atomic_json(self.path, data)


def planned(candidates):
    conditions = [('full', [])] + [(c['id'], c['mask']) for c in candidates]
    return [dict(kind='formal', condition=c, mask=m, suite=s, task=t, init=i)
            for i in range(5) for c, m in conditions for s in SUITES for t in range(10)]


def bootstrap_plan():
    reference = [dict(kind='reference', condition='official_original_full', mask=[], suite=s, task=0, init=0) for s in SUITES]
    smoke = [dict(kind='formal', condition='full', mask=[], suite=s, task=0, init=0) for s in SUITES]
    calibration = [dict(kind='calibration', condition='full', mask=[], suite=s, task=t, init=5) for s in SUITES for t in (0, 5)]
    return reference + smoke + calibration


def key(row):
    return '/'.join(map(str, (row['kind'], row['condition'], row['suite'], row['task'], row['init'])))


def init0_complete(campaign):
    for model in ('xiaomi-robotic0', 'db-cogact'):
        root = Path(campaign) / 'e1' / model
        candidates_path = root / 'candidates.json'
        if not candidates_path.exists():
            return False
        candidates = json.loads(candidates_path.read_text())
        expected = {key(row) for row in planned(candidates) if row['init'] == 0}
        valid = {r['key'] for r in Ledger(root).read()['entries'] if r['status'] == 'valid'}
        if not expected.issubset(valid):
            return False
    return True


def parity_gate(root):
    path = Path(root) / 'readiness/parity.json'
    if not path.exists() or not json.loads(path.read_text()).get('passed'):
        raise ValueError('Real native/empty/reference/fast parity must pass before environment episodes')


def compare_native_references(root):
    ledger = Ledger(root).read()['entries']
    completed = {r['key']: Path(r['directory']) for r in ledger if r['status'] == 'valid'}
    rows = []
    for suite in SUITES:
        native = completed.get(f'reference/official_original_full/{suite}/0/0')
        adapted = completed.get(f'formal/full/{suite}/0/0')
        if not native or not adapted:
            raise ValueError('Four native and four adapted Full smoke episodes required')
        left = [json.loads(s) for s in (native / 'telemetry.jsonl').read_text().splitlines()]
        right = [json.loads(s) for s in (adapted / 'telemetry.jsonl').read_text().splitlines()]
        if len(left) != len(right):
            raise ValueError(f'Native/empty Full trajectory length differs: {suite}')
        for a, b in zip(left, right):
            if not np.array_equal(a['eef_pos'], b['eef_pos']) or a['action'] != b['action']:
                raise ValueError(f'Native/empty Full measured trajectory mismatch: {suite}')
        rows.append(dict(suite=suite, steps=len(left)-1, exact_measured_xyz_and_actions=True,
                         reference=str(native), adapted=str(adapted)))
    destination = Path(root) / 'native_reference_comparison.json'
    if not destination.exists():
        write_json(destination, dict(passed=True, suites=rows))
    return rows


def validate(directory, persist=True):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    identity = json.loads((directory / 'identity.json').read_text())
    states = [json.loads(s) for s in (directory / 'telemetry.jsonl').read_text().splitlines()]
    infers = [json.loads(s) for s in (directory / 'infer.jsonl').read_text().splitlines()]
    if len(states) != result['actions'] + 1 or [r['step'] for r in states] != list(range(len(states))):
        raise ValueError('Telemetry coverage differs from executed actions')
    if not np.allclose(np.diff([r['sim_time_s'] for r in states]), .05, atol=1e-8, rtol=0):
        raise ValueError('Physical timeline differs from 20 Hz')
    offset = 0
    noise_verified = 0
    for infer in infers:
        if infer['step'] != offset or sha(infer['input_path']) != infer['input_file_sha256'] or sha(infer['prediction_path']) != infer['prediction_sha256']:
            raise ValueError('Inference identity/hash mismatch')
        with np.load(infer['prediction_path'], allow_pickle=False) as data:
            predicted = data['actions']
        with np.load(infer['input_path'], allow_pickle=False) as data:
            obs = {k: data[k] for k in data.files if k not in ('instruction', 'seed')}
            if input_digest(obs, str(data['instruction'])) != infer['input_hash']:
                raise ValueError('Recomputed raw input hash differs from server')
        if not np.array_equal(predicted, np.asarray(infer['chunk_actions'], dtype=np.float32)):
            raise ValueError('Saved predictions and inference log differ')
        if infer.get('server_prediction_path'):
            with np.load(infer['server_prediction_path'], allow_pickle=False) as data:
                if not np.array_equal(data['actions'], predicted):
                    raise ValueError('Server saved prediction differs from executed source')
        if infer.get('noise_path'):
            with np.load(infer['noise_path'], allow_pickle=False) as data:
                samples = [data[k] for k in sorted(data.files)]
                if value_digest(samples) != infer['noise_hash']:
                    raise ValueError('Actual sampled noise tensor hash differs from inference')
            noise_verified += 1
        n = infer['executed_length']
        expected = predicted[:n].copy()
        if identity['model'] == 'db-cogact':
            expected[:, -1] = np.where(expected[:, -1] < 0, -1., 1.)
        actual = np.asarray([r['action'] for r in states[offset+1:offset+n+1]])
        if n < 1 or n > result['R'] or not np.array_equal(expected, actual):
            raise ValueError('Execution differs from native decoded prefix')
        offset += n
    if offset != result['actions'] or len(infers) != result['predictions']:
        raise ValueError('Inference/action coverage mismatch')
    proc = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                           '-show_entries', 'stream=nb_read_frames,codec_name,pix_fmt,r_frame_rate', '-of', 'json',
                           str(directory / 'video.mp4')], check=True, capture_output=True, text=True)
    video = json.loads(proc.stdout)['streams'][0]
    if video['codec_name'] != 'h264' or video['pix_fmt'] != 'yuv420p' or video['r_frame_rate'] != '20/1' or int(video['nb_read_frames']) != len(states):
        raise ValueError('Video is not complete H264 yuv420p 20fps')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(directory / 'video.mp4'), '-f', 'null', '-'], check=True, capture_output=True)
    result.update(audited=True, video_decoded=True, raw_noise_recomputed_calls=noise_verified,
                  raw_noise_missing_calls=len(infers)-noise_verified)
    if persist:
        write_json(directory / 'validated.json', result)
    return result


def run_one(args, row):
    root = args.output / 'e1' / args.model
    ledger = Ledger(root)
    previous = [r for r in ledger.read()['entries'] if r['key'] == key(row)]
    if any(r['status'] == 'valid' for r in previous):
        return
    retry = bool(previous) and args.retry_invalid
    token = ledger.reserve(key(row), row['kind'], retry, args.fix_note)
    directory = root / 'episodes' / row['condition'] / row['suite'] / f"task{row['task']:02d}" / f"init{row['init']:02d}" / token
    ledger.update(token, directory=str(directory))
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [str(args.client_python), str(Path(__file__).with_name('runtime_client.py')),
           '--directory', str(directory), '--port', str(args.port), '--model', args.model,
           '--suite', row['suite'], '--task', str(row['task']), '--init', str(row['init']),
           '--kind', row['kind'], '--mask', json.dumps(row['mask'])]
    if row['kind'] == 'reference':
        cmd.append('--reference')
    child_env = os.environ.copy()
    repo = ROOT / 'policies/realtime-vla-flash'
    child_env.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(args.gpu), OMP_NUM_THREADS='1',
                     LIBERO_CONFIG_PATH=str(repo / '.libero'), PYTHONUNBUFFERED='1',
                     PYTHONPATH=':'.join([str(ROOT / 'LIBERO'), str(repo / 'packages/openpi-client/src'), child_env.get('PYTHONPATH', '')]))
    try:
        process_started = time.perf_counter()
        with (logs / (token + '.log')).open('x') as stream:
            subprocess.run(cmd, env=child_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        client_wall_s = time.perf_counter() - process_started
        validation_started = time.perf_counter()
        result = validate(directory)
        write_json(directory / 'client_cost.json', dict(client_process_wall_s=client_wall_s,
                   validation_wall_s=time.perf_counter()-validation_started,
                   scope='Client imports, simulator setup, model RPCs, simulation, telemetry and complete video encoding; excludes server startup'))
        ledger.update(token, status='valid', success=result['success'], actions=result['actions'], predictions=result['predictions'], client_process_wall_s=client_wall_s)
        print(json.dumps(dict(key=key(row), **result)), flush=True)
    except BaseException as exc:
        ledger.update(token, status='invalid', error=repr(exc))
        raise


def select_calibration(root):
    """At most ten real replans per episode, balanced over measured current phase."""
    selected = []
    for result in sorted(Path(root).glob('episodes/full/*/task*/init05/*/validated.json')):
        episode = result.parent
        states = [json.loads(s) for s in (episode / 'telemetry.jsonl').read_text().splitlines()]
        infers = [json.loads(s) for s in (episode / 'infer.jsonl').read_text().splitlines()]
        groups = {}
        for row in infers:
            phase = states[row['step']]['semantic_phase']
            groups.setdefault(phase, []).append(row)
        picks = []
        # Evenly order each phase before round robin, avoiding early-only bias.
        pools = {p: [g[i] for i in np.unique(np.linspace(0, len(g)-1, min(10, len(g))).round().astype(int))]
                 for p, g in sorted(groups.items())}
        while len(picks) < 10 and any(pools.values()):
            for phase, pool in pools.items():
                if pool and len(picks) < 10:
                    picks.append(dict(pool.pop(0), phase=phase, episode=str(episode)))
        selected.extend(picks)
    if len(list(Path(root).glob('episodes/full/*/task*/init05/*/validated.json'))) != 8:
        raise ValueError('All eight valid calibration episodes are required')
    return selected


def probes(args):
    root = args.output / 'e1' / args.model
    points = select_calibration(root)
    source = root / 'calibration/probes.json'
    if source.exists():
        return
    selection = root / 'calibration/selection.json'
    if selection.exists():
        if json.loads(selection.read_text()) != points:
            raise ValueError('Frozen calibration observation selection changed')
    else:
        write_json(selection, points)
    records = []
    depth = MODELS[args.model]['depth']
    # Saved incremental point artifacts prevent repeat calls across resume.
    for index, row in enumerate(points):
        point_path = root / f'calibration/point{index:03d}.json'
        if point_path.exists():
            records.append(json.loads(point_path.read_text()))
            continue
        with np.load(row['input_path'], allow_pickle=False) as data:
            obs = {k: data[k] for k in data.files if k not in ('instruction', 'seed')}
            instruction, seed = str(data['instruction']), int(data['seed'])
        def predict_once(label, skip, probe=False):
            destination = root / f'calibration/predictions/point{index:03d}_{label}.json'
            request_path = destination.with_suffix('.request.json')
            if destination.exists():
                return json.loads(destination.read_text())
            if request_path.exists():
                raise RuntimeError(f'Probe call outcome unresolved; preserve budget and diagnose: {request_path}')
            write_json(request_path, dict(index=index, skip=skip, seed=seed, source=row['input_path']))
            response = rpc(args.port, dict(op='predict', obs=obs, instruction=instruction, seed=seed,
                           skip=skip, reference=False, probe=probe, category='calibration'))
            write_json(destination, response)
            return response

        response = predict_once('full', [], True)
        full = np.asarray(response['actions'])
        impacts = []
        for layer in range(1, depth + 1):
            if layer <= (3 if args.model == 'xiaomi-robotic0' else 2):
                impacts.append(None)
                continue
            response_skip = predict_once(f'layer{layer:02d}', [layer])
            if response_skip['noise_hash'] != response['noise_hash'] or response_skip['input_hash'] != response['input_hash']:
                raise ValueError('Single-layer action impact must use exactly the same observation and sampled noise')
            impacts.append(float(np.sqrt(np.mean((np.asarray(response_skip['actions']) - full)**2))))
        actual_calls = 1 + sum(x is not None for x in impacts)
        record = dict(index=index, source=row, features=response.get('features', []),
                      impacts=impacts, logical_predictions=actual_calls, noise_hash=response.get('noise_hash'))
        write_json(point_path, record)
        records.append(record)
    coverage = {}
    for row in points:
        coverage[row['phase']] = coverage.get(row['phase'], 0) + 1
    write_json(source, dict(model=args.model, points=records, logical_predictions=sum(r['logical_predictions'] for r in records),
                           max_logical_predictions=80*(depth+1), real_observations=True,
                           phase_coverage=coverage, protected_layers=list(range(1, 4 if args.model == 'xiaomi-robotic0' else 3)),
                           protected_impact_status='Not measured because architecture/front layers are fixed retained',
                           selection_rule='At most ten per episode, equal phase round robin, uniformly spaced within each phase',
                           selection_sha256=sha(selection)))


def generate_masks(model, probe_path):
    data = json.loads(Path(probe_path).read_text())
    depth = MODELS[model]['depth']
    eligible = list(range(4 if model == 'xiaomi-robotic0' else 3, depth))
    impact = np.mean([[np.inf if x is None else x for x in p['impacts']] for p in data['points']], axis=0)
    features = [f for p in data['points'] for f in p.get('features', [])]
    cosine = np.zeros(depth)
    counts = np.zeros(depth)
    for feature in features:
        if feature.get('kind') != 'hidden':
            continue
        layer = feature.get('layer_1based', feature.get('layer', 0))
        value = feature.get('cosine', feature.get('input_output_cosine'))
        if isinstance(layer, int) and 1 <= layer <= depth and value is not None:
            cosine[layer-1] += float(value)
            counts[layer-1] += 1
    if not np.any(counts):
        raise ValueError('Real per-layer cosine features missing; cannot invent similarity evidence')
    cosine /= np.maximum(counts, 1)
    ranks = [sorted(eligible, key=lambda l: (-cosine[l-1], l)), sorted(eligible, key=lambda l: (impact[l-1], l))]
    rng = np.random.default_rng(1701)
    seen, candidates = set(), []
    source = dict(path=str(Path(probe_path).resolve()), sha256=sha(probe_path))

    def within_short_run_limit(layers):
        # Check the union: individually short groups can touch and form a long run.
        ordered = sorted(layers)
        return not any(ordered[i+3] - ordered[i] == 3 for i in range(len(ordered)-3))

    for budget, count in MODELS[model]['budget']:
        for algorithm in ('cosine_dispersed', 'impact_dispersed', 'seeded_random', 'adjacent_pairs', 'short_runs')[:count]:
            rank = ranks[0] if algorithm == 'cosine_dispersed' else ranks[1]
            mask = []
            if algorithm.endswith('dispersed'):
                for layer in rank:
                    if all(abs(layer - old) > 1 for old in mask):
                        mask.append(layer)
                    if len(mask) == budget:
                        break
            elif algorithm == 'seeded_random':
                mask = rng.choice(eligible, budget, replace=False).tolist()
            else:
                run = 2 if algorithm == 'adjacent_pairs' else 3
                for layer in rank:
                    group = [l for l in range(layer, layer+run) if l in eligible and l not in mask]
                    if (len(mask) + len(group) <= budget and
                        (algorithm != 'short_runs' or within_short_run_limit(mask + group))):
                        mask.extend(group)
                    if len(mask) == budget:
                        break
            for layer in rank:
                if len(mask) == budget:
                    break
                if layer not in mask and (algorithm != 'short_runs' or within_short_run_limit(mask + [layer])):
                    mask.append(layer)
            if len(mask) != budget:
                raise ValueError(f'Cannot fill {algorithm} skip budget while preserving its run constraint')
            mask = sorted(mask[:budget])
            while tuple(mask) in seen:
                proposal = sorted(rng.choice(eligible, budget, replace=False).tolist())
                # Deduplication must preserve the same constraint as construction.
                if algorithm != 'short_runs' or within_short_run_limit(proposal):
                    mask = proposal
            seen.add(tuple(mask))
            candidates.append(dict(id=f'skip{budget}_{algorithm}', mask=mask, algorithm=algorithm, seed=1701,
                                   reason='Finite heuristic from real calibration; every distinct mask is evaluated', probe_source=source))
    validate_candidates(model, depth, candidates, protected_readout=[depth])
    return candidates


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('bootstrap', 'probes', 'masks', 'formal', 'queue', 'reconcile', 'full-init0'))
    p.add_argument('--model', required=True, choices=('xiaomi-robotic0', 'db-cogact'))
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--client-python', type=Path, default=ROOT / 'policies/libero-eval-v21/.venv/bin/python')
    p.add_argument('--retry-invalid', action='store_true')
    p.add_argument('--fix-note', default='')
    p.add_argument('--limit', type=int)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    args = p.parse_args(argv)
    root = args.output / 'e1' / args.model
    root.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.shard_index < args.shard_count:
        p.error('Expected 0 <= shard-index < shard-count')
    if args.action in ('bootstrap', 'probes', 'queue') and args.shard_count != 1:
        p.error('Bootstrap and calibration have a single leader; shard only the formal queue')
    if args.action == 'reconcile':
        Ledger(root).reconcile(args.fix_note)
        return
    parity_gate(root)
    worker_lock = f'formal.shard{args.shard_index}of{args.shard_count}.lock' if args.action == 'formal' else args.action + '.lock'
    with lock(root / worker_lock, True):
        if args.action == 'full-init0':
            for row in planned([]):
                if row['init'] == 0 and row['task'] != 0:
                    run_one(args, row)
        if args.action in ('bootstrap', 'queue'):
            for row in bootstrap_plan():
                run_one(args, row)
            compare_native_references(root)
        if args.action in ('probes', 'queue'):
            probes(args)
        if args.action in ('masks', 'queue'):
            path = root / 'candidates.json'
            if not path.exists():
                write_json(path, generate_masks(args.model, root / 'calibration/probes.json'))
        if args.action in ('formal', 'queue'):
            candidates = json.loads((root / 'candidates.json').read_text())
            validate_candidates(args.model, MODELS[args.model]['depth'], candidates, [MODELS[args.model]['depth']])
            rows = planned(candidates)
            rows = rows[:args.limit] if args.limit else rows
            waiting_reported = False
            for index, row in enumerate(rows):
                if index % args.shard_count != args.shard_index:
                    continue
                while row['init'] > 0 and not init0_complete(args.output):
                    if not waiting_reported:
                        print('Waiting for both models all 1040 init0 keys before init1..4', flush=True)
                        waiting_reported = True
                    time.sleep(10)
                run_one(args, row)


if __name__ == '__main__':
    main()
