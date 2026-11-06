"""Measure D3 model latency on shared Full observations, alone on one idle GPU."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from scripts.experiment11.protocol import PROTOCOL, RESULTS, ROOT, SOURCE, SUITES, environment

os.environ['EXP11_RESULTS_ROOT'] = str(RESULTS)
os.environ['EXP11_PROTOCOL_JSON'] = json.dumps(PROTOCOL, separators=(',', ':'))

from scripts.experiment10.common import claim_gpu, dump_json, sha256
from scripts.experiment10.run_pipeline import free_gpus


BASELINE = 'Full18_D3_R5_FORMAL'
SAMPLE_PER_CONDITION = 200  # first replan point: 10 tasks × 5 inits × 4 suites


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def event(**record) -> None:
    event_path = RESULTS / 'events.jsonl'
    event_path.parent.mkdir(parents=True, exist_ok=True)
    row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **record}
    with event_path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def verify_d3_complete() -> tuple[dict, dict]:
    status_path = RESULTS / 'experiment11_closed_loop_status.json'
    status = json.loads(status_path.read_text())
    audit_path = RESULTS / 'analysis/closed_loop_d3/completion_audit.json'
    if (not status.get('complete') or status.get('active') or status.get('pending') or
            status.get('failures') or not audit_path.is_file()):
        raise RuntimeError('D3 matrix and complete D3 analysis audit are required first')
    audit = json.loads(audit_path.read_text())
    if audit.get('condition_suite_runs_validated') != 124 or audit.get('episodes_validated') != 6200:
        raise RuntimeError('D3 audit does not prove all 6200 primary episodes')
    config = json.loads((RESULTS / 'protocol/closed_loop_config_formal.json').read_text())
    if len(config['conditions']) != 31:
        raise RuntimeError('Expected Full18, original V13, and 29 candidate conditions')
    return config, audit


def latency_conditions(config: dict) -> list[dict]:
    """Time each family winner (all top-score ties) plus the two D3 controls."""
    with (RESULTS / 'analysis/closed_loop_d3/suite_results.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((RESULTS / 'protocol/candidate_manifest.json').read_text())
    selected_names = {BASELINE, 'V13_Custom_D3_R5_FORMAL'}
    for family in ('V16', 'V15', 'V14', 'V13'):
        candidates = [candidate for candidate in manifest['candidates'] if candidate['family'] == family]
        rates = {}
        for candidate in candidates:
            name = candidate['condition']['name'] + '_FORMAL'
            suite_rows = [row for row in rows if row['condition'] == name]
            if len(suite_rows) != 4 or {row['suite'] for row in suite_rows} != set(SUITES):
                raise RuntimeError(f'Missing four-suite D3 result for {candidate["variant_id"]}')
            rates[candidate['variant_id']] = sum(float(row['success_rate']) for row in suite_rows) / 4
        top = max(rates.values())
        for candidate in candidates:
            if rates[candidate['variant_id']] == top:
                selected_names.add(candidate['condition']['name'] + '_FORMAL')
    result = [row for row in config['conditions'] if row['name'] in selected_names]
    if len(result) != len(selected_names):
        raise RuntimeError('A latency tie-break condition was not found in frozen config')
    return result


def validate_attempt_id(attempt_id: str) -> str:
    if not attempt_id or attempt_id in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', attempt_id):
        raise ValueError('attempt-id must be 1..64 safe filename characters')
    return attempt_id


def attempt_dir(attempt_id: str) -> Path:
    return RESULTS / 'latency' / 'attempts' / validate_attempt_id(attempt_id)


def sample_filename(condition_name: str, suite: str | None = None) -> str:
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', condition_name)
    suffix = '' if suite is None else '__' + re.sub(r'[^A-Za-z0-9_.-]+', '_', suite)
    return f'{safe_name}{suffix}.csv'


def validate_condition_samples(condition_name: str, rows: list[dict], *,
                               expected_count: int = SAMPLE_PER_CONDITION,
                               expected_suite: str | None = None) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(f'{condition_name}: expected {expected_count} samples, got {len(rows)}')
    identities = {(row['suite'], int(row['task_id']), int(row['init_index'])) for row in rows}
    if len(identities) != expected_count:
        raise RuntimeError(f'{condition_name}: duplicate sample identities')
    if any(row['condition'] != condition_name for row in rows):
        raise RuntimeError(f'{condition_name}: per-condition output contains another condition')
    if expected_suite is not None:
        expected = {(expected_suite, task, init) for task in range(10) for init in range(5)}
        if identities != expected:
            raise RuntimeError(f'{condition_name}/{expected_suite}: sample identities do not cover task0-9/init0-4')
    elif expected_count == SAMPLE_PER_CONDITION:
        expected = {(suite, task, init) for suite in SUITES for task in range(10) for init in range(5)}
        if identities != expected:
            raise RuntimeError(f'{condition_name}: sample identities do not cover all four suites/task0-9/init0-4')


def worker(gpu_index: int, attempt_id: str, condition_name: str, suite_name: str) -> int:
    # CUDA_VISIBLE_DEVICES must be set by the parent before importing torch/Backend.
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu_index):
        raise RuntimeError('Dedicated latency worker is not bound to its leased physical GPU')
    import torch
    from scripts.experiment10.backend import Backend

    torch.cuda.init()
    config, _ = verify_d3_complete()
    conditions = [row for row in latency_conditions(config) if row['name'] == condition_name]
    if len(conditions) != 1:
        raise RuntimeError(f'Unknown/ambiguous isolated latency condition: {condition_name}')
    condition = conditions[0]
    if suite_name not in SUITES:
        raise RuntimeError(f'Unknown latency suite: {suite_name}')
    output_dir = attempt_dir(attempt_id)
    if not output_dir.is_dir():
        raise RuntimeError(f'Latency attempt directory is missing: {output_dir}')
    sample_path = output_dir / sample_filename(condition_name, suite_name)
    if sample_path.exists():
        raise FileExistsError(f'Refusing to replace per-condition output: {sample_path}')
    manifest = json.loads((RESULTS / 'protocol/candidate_manifest.json').read_text())
    condition_to_variant = {row['condition']['name'] + '_FORMAL': row['variant_id']
                            for row in manifest['candidates']}
    condition_to_variant.update({BASELINE: 'Full18', 'V13_Custom_D3_R5_FORMAL': 'V13_Custom'})
    checkpoint_hash = manifest['checkpoint_sha256']
    suite = suite_name
    entries = json.loads((SOURCE / 'observations' / suite / 'manifest.json').read_text())['observations']
    source_entries = []
    for entry in entries:
        if entry.get('task_id') in range(10):
            with np.load(entry['path'], allow_pickle=False) as data:
                source_entries.append(json.loads(str(data['metadata_json'].item())))

    samples = []
    baseline_rows = json.loads((RESULTS / 'closed_loop' / BASELINE / suite / 'run/episode_log.json').read_text())
    baseline_actions = {}
    for episode in baseline_rows:
        infer_path = RESULTS / 'closed_loop' / BASELINE / suite / 'run' / episode['infer_path']
        with infer_path.open(encoding='utf-8') as stream:
            first = json.loads(next(stream))
        baseline_actions[(int(episode['task_id']), int(episode['episode_idx']))] = (
            np.asarray(first['chunk_actions'], dtype=np.float32), int(first['infer_id']))
    for task in range(10):
        for init in range(5):
            path = RESULTS / 'full_observations' / suite / f'task{task:02d}' / f'init{init:02d}' / 'replan0000.npz'
            if not path.is_file():
                raise RuntimeError(f'Missing shared first-replan input: {path}')
            samples.append((suite, task, init, path, baseline_actions[(task, init)]))
    if len(samples) != 50:
        raise RuntimeError(f'Expected 50 shared first-replan samples for {suite}, found {len(samples)}')

    sample_rows = []
    started_all = time.monotonic()
    condition_name = condition['name']
    skip = tuple(condition['skip'])
    if condition['steps'] != 3 or condition.get('replan_steps') != 5:
        raise RuntimeError(f'Non-D3/replan5 condition in latency matrix: {condition_name}')
    backend = Backend(source_entries)
    expected_hash = checkpoint_hash
    for sample_suite, task, init, path, baseline_record in samples:
        full_actions, full_infer_id = baseline_record
        if sample_suite != suite:
            raise RuntimeError('Latency sample suite grouping mismatch')
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json'].item()))
            if (metadata.get('suite') != suite or int(metadata.get('task_id', -1)) != task or
                    int(metadata.get('episode_idx', -1)) != init or
                    int(metadata.get('infer_id', -1)) != full_infer_id or
                    metadata.get('checkpoint_sha256') != expected_hash or
                    metadata.get('skip_layers') != [] or metadata.get('denoise_steps') != 3):
                raise RuntimeError(f'Full input provenance mismatch: {path}')
            observation = {
                'observation/image': data['agent_image'],
                'observation/wrist_image': data['wrist_image'],
                'observation/state': data['robot_state'],
                'prompt': str(data['prompt'].item()),
            }
            noise = data['noise'].copy()
            input_hash = str(metadata['input_sha256'])
            noise_hash = hashlib.sha256(noise.tobytes()).hexdigest()
            if noise_hash != metadata.get('noise_sha256'):
                raise RuntimeError(f'Noise hash mismatch: {path}')
            meta = {'suite': suite, 'task_description': observation['prompt']}
            backend.prepare(observation, meta, noise)
        # A fresh process for every condition x suite prevents state from a
        # previous suite/backend from affecting parity or timing.
        _, commands, timing = backend.infer(skip, 3)
        commands = np.asarray(commands, dtype=np.float32)
        if commands.shape != (50, 7) or not np.isfinite(commands).all():
            raise RuntimeError(f'Invalid action tensor for {condition_name}/{suite}/{task}/{init}')
        parity_error = None
        if condition_name == BASELINE:
            if full_actions.shape != commands.shape:
                raise RuntimeError(f'Full18 shape parity failed at {suite}/{task}/{init}: '
                                   f'{commands.shape} != {full_actions.shape}')
            if not np.allclose(commands, full_actions, rtol=1e-4, atol=1e-5):
                delta = np.abs(commands - full_actions)
                peak = tuple(int(v) for v in np.unravel_index(int(np.argmax(delta)), delta.shape))
                raise RuntimeError(
                    f'Full18 isolated parity failed at {suite}/{task}/{init}: '
                    f'max_abs={float(delta.max()):.9g}, rmse={float(np.sqrt(np.mean((commands-full_actions)**2))):.9g}, '
                    f'peak_index={peak}, predicted={float(commands[peak]):.9g}, '
                    f'saved={float(full_actions[peak]):.9g}, full_infer_id={full_infer_id}')
            parity_error = float(np.max(np.abs(commands - full_actions)))
        cuda_ms = float(timing['cuda_policy_ms'])
        host_ms = float(timing['warm_call_ms'])
        if not (math.isfinite(cuda_ms) and cuda_ms > 0 and math.isfinite(host_ms) and host_ms > 0):
            raise RuntimeError(f'Invalid dedicated timing for {condition_name}/{suite}/{task}/{init}')
        sample_rows.append({
            'condition': condition_name, 'suite': suite, 'task_id': task, 'init_index': init,
            'variant_id': condition_to_variant[condition_name],
            'skip_layers': ','.join(map(str, skip)), 'steps': 3, 'horizon': 50, 'replan_steps': 5,
            'gpu_index': gpu_index, 'cuda_policy_ms': cuda_ms,
            'backend_infer_wall_ms': host_ms, 'input_sha256': input_hash,
            'checkpoint_sha256': expected_hash, 'full_parity_max_abs_error': parity_error,
            'timing_source': 'dedicated single-GPU Backend CUDA events; fresh process per condition and suite',
        })
    validate_condition_samples(condition_name, sample_rows, expected_count=50, expected_suite=suite)
    write_csv(sample_path, sample_rows, list(sample_rows[0]))
    event(event='isolated_latency_suite_complete', attempt_id=attempt_id, condition=condition_name,
          suite=suite, samples=len(sample_rows), wall_seconds=time.monotonic() - started_all)
    return 0


def read_condition_csv(path: Path, condition_name: str, *, suite: str | None = None) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    validate_condition_samples(condition_name, rows,
                               expected_count=50 if suite is not None else SAMPLE_PER_CONDITION,
                               expected_suite=suite)
    return rows


def write_aggregates(sample_rows: list[dict], conditions: list[dict], gpu_index: int,
                     output_dir: Path, attempt_id: str, condition_order: list[str],
                     started_all: float, checkpoint_hash: str) -> tuple[Path, Path]:
    if len(sample_rows) != SAMPLE_PER_CONDITION * len(conditions):
        raise RuntimeError(f'Latency coverage mismatch: {len(sample_rows)} total rows')
    per_condition = {}
    for condition in conditions:
        condition_name = condition['name']
        values = [row for row in sample_rows if row['condition'] == condition_name]
        validate_condition_samples(condition_name, values)
        per_condition[condition_name] = values

    sample_path = output_dir / 'isolated_d3_samples.csv'
    aggregate_path = output_dir / 'isolated_d3.csv'
    metadata_path = output_dir / 'isolated_d3_metadata.json'
    if sample_path.exists() or aggregate_path.exists() or metadata_path.exists():
        raise FileExistsError(f'Refusing to replace published D3 latency artifacts in {output_dir}')
    write_csv(sample_path, sample_rows, list(sample_rows[0]))
    aggregate_rows = []
    for condition in conditions:
        values = per_condition[condition['name']]
        cuda = np.asarray([row['cuda_policy_ms'] for row in values], dtype=float)
        host = np.asarray([row['backend_infer_wall_ms'] for row in values], dtype=float)
        aggregate_rows.append({
            'condition': condition['name'],
            'variant_id': values[0]['variant_id'],
            'skip_layers': ','.join(map(str, condition['skip'])),
            'steps': 3, 'horizon': 50, 'replan_steps': 5, 'gpu_index': gpu_index,
            'samples': len(values), 'mean_cuda_policy_ms': float(cuda.mean()),
            'p50_cuda_policy_ms': float(np.median(cuda)), 'p95_cuda_policy_ms': float(np.quantile(cuda, .95)),
            'mean_backend_infer_wall_ms': float(host.mean()),
            'p50_backend_infer_wall_ms': float(np.median(host)),
            'speedup_vs_full_p50': None,
        })
    full_p50 = next(row['p50_cuda_policy_ms'] for row in aggregate_rows if row['condition'] == BASELINE)
    for row in aggregate_rows:
        row['speedup_vs_full_p50'] = full_p50 / row['p50_cuda_policy_ms']
    write_csv(aggregate_path, aggregate_rows, list(aggregate_rows[0]))
    dump_json(metadata_path, {
        'conditions': len(conditions), 'samples_per_condition': SAMPLE_PER_CONDITION,
        'total_samples': len(sample_rows), 'gpu_index': gpu_index, 'attempt_id': attempt_id,
        'condition_order': condition_order,
        'actual_condition_order_randomized': True,
        'measurement_scope': 'D3 model/backend only, common first-replan observations, exclusive GPU, fresh Python process per condition x suite; excludes simulator/RPC/video costs.',
        'wall_seconds': time.monotonic() - started_all,
        'sample_csv_sha256': sha256(sample_path), 'aggregate_csv_sha256': sha256(aggregate_path),
        'checkpoint_sha256': checkpoint_hash,
    })
    return sample_path, aggregate_path


def manager(attempt_id: str) -> int:
    attempt_id = validate_attempt_id(attempt_id)
    status_path = RESULTS / f'experiment11_d3_latency_{attempt_id}_status.json'
    output = RESULTS / 'latency'
    output.mkdir(parents=True, exist_ok=True)
    isolated_dir = attempt_dir(attempt_id)
    if status_path.exists() or isolated_dir.exists():
        raise FileExistsError(f'Latency attempt already exists; choose a new attempt-id: {attempt_id}')
    published = [output / name for name in ('isolated_d3_samples.csv', 'isolated_d3.csv',
                                             'isolated_d3_metadata.json')]
    if any(path.exists() for path in published):
        raise FileExistsError('Published latency artifacts already exist; audit them before a new attempt')
    config, _ = verify_d3_complete()
    measured_conditions = latency_conditions(config)
    rng = np.random.default_rng(20261010)
    rng.shuffle(measured_conditions)
    measured_names = [row['name'] for row in measured_conditions]
    work_units = []
    for condition_name in measured_names:
        suite_order = list(SUITES)
        rng.shuffle(suite_order)
        work_units.extend((condition_name, suite) for suite in suite_order)
    isolated_dir.mkdir(parents=True)
    dump_json(isolated_dir / 'attempt.json', {
        'attempt_id': attempt_id, 'conditions': measured_conditions,
        'condition_order': measured_names, 'work_units': work_units,
        'samples_per_condition': SAMPLE_PER_CONDITION, 'samples_per_worker': 50,
        'process_isolation': 'one fresh worker process per condition x suite',
    })
    state = {'attempt_id': attempt_id, 'complete': False, 'state': 'waiting_for_idle_gpu',
             'active': None, 'failures': [], 'conditions': measured_names,
             'work_units': [{'condition': condition, 'suite': suite} for condition, suite in work_units],
             'completed_units': [], 'completed_conditions': [], 'gpu_index': None}
    dump_json(status_path, state)
    while True:
        available = free_gpus()
        if available:
            gpu = available[0]
            lease = claim_gpu(gpu)
            if lease is not None and gpu in free_gpus():
                break
            if lease is not None:
                lease.close()
        event(event='isolated_latency_waiting_for_idle_gpu', attempt_id=attempt_id, sleep_seconds=300)
        state['state'] = 'waiting_for_idle_gpu'
        dump_json(status_path, state)
        time.sleep(300)
    env = environment()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['EXP11_METADATA_KEY'] = 'experiment11'
    env['EXP11_SOURCE_RESULTS'] = str(SOURCE)
    state['gpu_index'] = gpu
    started_all = time.monotonic()
    try:
        for unit_index, (condition_name, suite) in enumerate(work_units):
            worker_command = [
                str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment11.measure_d3_latency',
                '--worker', '--gpu-index', str(gpu), '--attempt-id', attempt_id,
                '--condition-name', condition_name, '--suite', suite,
            ]
            log_path = isolated_dir / f'{unit_index + 1:02d}_{sample_filename(condition_name, suite)}.log'
            state['state'] = 'running_suite'
            state['active'] = {'pid': 0, 'gpu': gpu, 'condition': condition_name, 'suite': suite}
            dump_json(status_path, state)
            with log_path.open('x', encoding='utf-8') as log_stream:
                process = subprocess.Popen(worker_command, cwd=ROOT, env=env,
                                           stdout=log_stream, stderr=subprocess.STDOUT)
                state['active'] = {'pid': process.pid, 'gpu': gpu,
                                   'condition': condition_name, 'suite': suite}
                dump_json(status_path, state)
                code = process.wait()
            if code != 0:
                state['state'] = 'failed'
                state['active'] = None
                state['failures'].append({'pid': process.pid, 'code': code,
                                          'condition': condition_name, 'suite': suite,
                                          'log': str(log_path)})
                dump_json(status_path, state)
                lease.close()
                return code
            suite_rows = read_condition_csv(
                isolated_dir / sample_filename(condition_name, suite), condition_name, suite=suite)
            state['completed_units'].append({'condition': condition_name, 'suite': suite,
                                             'samples': len(suite_rows)})
            completed_suites = {row['suite'] for row in state['completed_units']
                                if row['condition'] == condition_name}
            if completed_suites == set(SUITES):
                state['completed_conditions'].append({'name': condition_name,
                                                      'samples': SAMPLE_PER_CONDITION})
            state['active'] = None
            dump_json(status_path, state)

        sample_rows = []
        for condition_name in measured_names:
            condition_rows = []
            for suite in SUITES:
                condition_rows.extend(read_condition_csv(
                    isolated_dir / sample_filename(condition_name, suite), condition_name, suite=suite))
            validate_condition_samples(condition_name, condition_rows)
            sample_rows.extend(condition_rows)
        manifest = json.loads((RESULTS / 'protocol/candidate_manifest.json').read_text())
        checkpoint_hash = manifest['checkpoint_sha256']
        attempt_samples, attempt_aggregate = write_aggregates(
            sample_rows, measured_conditions, gpu, isolated_dir, attempt_id,
            measured_names, started_all, checkpoint_hash,
        )
        # The attempt-specific files are authoritative. Publish the canonical
        # files only after all conditions and parity checks pass; link creation
        # fails rather than overwriting any previous valid result.
        for source, name in ((attempt_samples, 'isolated_d3_samples.csv'),
                             (attempt_aggregate, 'isolated_d3.csv'),
                             (isolated_dir / 'isolated_d3_metadata.json', 'isolated_d3_metadata.json')):
            os.link(source, output / name)
        state['state'] = 'complete'
        state['complete'] = True
        state['active'] = None
        state['published_artifacts'] = [str(path) for path in published]
        dump_json(status_path, state)
        lease.close()
        return 0
    except BaseException as exc:
        state['state'] = 'failed'
        state['active'] = None
        state['failures'].append({'error': repr(exc)})
        dump_json(status_path, state)
        lease.close()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Wait for one idle GPU and run isolated benchmark')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--gpu-index', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--condition-name', help=argparse.SUPPRESS)
    parser.add_argument('--suite', choices=SUITES, help=argparse.SUPPRESS)
    parser.add_argument('--attempt-id', help='Unique ID for a new isolated latency attempt')
    args = parser.parse_args()
    if args.worker:
        if args.gpu_index is None or not args.condition_name or not args.attempt_id or not args.suite:
            raise ValueError('--worker requires --gpu-index, --condition-name, --suite, and --attempt-id')
        return worker(args.gpu_index, args.attempt_id, args.condition_name, args.suite)
    config, _ = verify_d3_complete()
    selected = latency_conditions(config)
    print(json.dumps({'dry_run': not args.launch, 'conditions': len(selected),
                      'condition_names': [row['name'] for row in selected],
                      'samples_per_condition': SAMPLE_PER_CONDITION,
                      'total_input_samples': len(selected) * SAMPLE_PER_CONDITION,
                      'measurement': 'single idle GPU, isolated CUDA-event and backend timing'}, indent=2))
    if args.launch:
        if not args.attempt_id:
            raise ValueError('--launch requires a unique --attempt-id (for example retry1_process_isolated)')
        return manager(args.attempt_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
