"""Independent pi0 LRU2 execution revision, 16-call gate and two-env lifecycle."""
import argparse
from contextlib import contextmanager, ExitStack
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
from common import ROOT, atomic_json, evaluation_env, exclusive, gpu_lease, lock, prior_diagnostic_budget, sha
from runtime_data import load_observation, verify_frozen, verify_parity, verify_acceptance
from runtime_runner import CLIENT, available_port, free_gpus, model_python
from throughput_revision import read, stop_process, verify_revision as verify_base
from pi0_prompt_cache import function_identity

HERE = Path(__file__).resolve().parent
KIND = 'pi0_prompt_length_runtime_cache2'
SOURCES = ('pi0_prompt_cache.py', 'pi0_prompt_cache_server.py', 'pi0_prompt_cache_client.py', 'pi0_prompt_cache_revision.py')
MEMORY_CAP_MIB = 23*1024


def original_prepare_identity():
    project = str(ROOT/'policies/realtime-vla-flash')
    if project not in sys.path:
        sys.path.insert(0, project)
    from scripts.experiment10.backend import Backend
    return function_identity(Backend.prepare)


def requests_for(manifest):
    converted = ROOT/'shared/checkpoints/realtime-vla-flash/converted'
    prompts = {}
    for directory in ('draft_spatial', 'draft_object', 'draft', 'draft_10'):
        for prompt, entry in read(converted/directory/'manifest.json')['prompts'].items():
            prompts.setdefault(prompt, entry['prompt_len'])
    chosen = []
    for point in manifest['observations']['pi0']:
        _, instruction, _ = load_observation(point)
        length = prompts[instruction]
        if not chosen or length != chosen[0]['prompt_length']:
            chosen.append(dict(point=point, prompt_length=length))
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError('Require two frozen real observations with different native prompt lengths')
    conditions = [item for item in manifest['conditions'] if item['model'] == 'pi0' and item['schedule'].get('m') == .5]
    if len(conditions) != 2:
        raise ValueError('Require original Full/V14 u50 conditions')
    seeds = [chosen[0]['point']['seed'], chosen[0]['point']['seed'] ^ 0x10203]
    return [dict(index=index, condition=condition, seed=seed, **point)
        for index, (seed, condition, point) in enumerate((seed, condition, point)
            for seed in seeds for condition in conditions for point in chosen)]


def prepare(attempt, base_revision, revision):
    attempt, base_revision, revision = [Path(value).resolve() for value in (attempt, base_revision, revision)]
    _, manifest = verify_base(base_revision)
    if manifest != verify_frozen(attempt):
        raise ValueError('Base revision belongs to another frozen attempt')
    revision.mkdir(parents=True, exist_ok=False)
    identity = dict(schema=1, kind=KIND, source_attempt=str(attempt), base_revision=str(base_revision),
        base_manifest_sha256=sha(base_revision/'revision_manifest.json'),
        manifest_hash=manifest['manifest_hash'], manifest_file_sha256=sha(attempt/'condition_manifest.json'),
        frozen_code_hashes=manifest['code_hashes'], code_hashes={name: sha(HERE/name) for name in SOURCES},
        model='pi0', environments=2, cache_capacity=2, memory_cap_mib=MEMORY_CAP_MIB,
        original_prepare=original_prepare_identity(),
        requests=requests_for(manifest), diagnostic_predictions=16,
        reference_order=[0, 2, 4, 6, 1, 3, 5, 7], cached_order=list(range(8)),
        scientific_policy='original observation/tokenization/padding/weights/masks/noise/forward kernels; only complete runtime lifetime changes',
        benchmark_policy='not a latency benchmark; independent single-request benchmark remains separate', created=time.time())
    atomic_json(revision/'revision_manifest.json', identity)
    return identity


def verify_revision(revision):
    revision = Path(revision).resolve()
    identity = read(revision/'revision_manifest.json')
    _, manifest = verify_base(identity['base_revision'])
    if (identity.get('schema') != 1 or identity.get('kind') != KIND or identity.get('model') != 'pi0' or
            identity.get('cache_capacity') != 2 or identity.get('environments') != 2 or
            identity['base_manifest_sha256'] != sha(Path(identity['base_revision'])/'revision_manifest.json') or
            identity['manifest_file_sha256'] != sha(Path(identity['source_attempt'])/'condition_manifest.json') or
            identity['manifest_hash'] != manifest['manifest_hash'] or identity['frozen_code_hashes'] != manifest['code_hashes'] or
            set(identity['code_hashes']) != set(SOURCES) or identity['original_prepare'] != original_prepare_identity() or
            identity['memory_cap_mib'] != MEMORY_CAP_MIB or identity['requests'] != requests_for(manifest)):
        raise ValueError('Cache execution revision identity changed')
    for name, expected in identity['code_hashes'].items():
        if sha(HERE/name) != expected:
            raise ValueError(f'Cache revision source changed: {name}')
    verify_frozen(identity['source_attempt'])
    return identity, manifest


def acceptance(revision, phase, evidence=None):
    revision = Path(revision)
    path = revision/'acceptance'/f'pi0.{phase}.json'
    receipt = read(path)
    if (receipt.get('schema') != 1 or receipt.get('accepted') is not True or receipt.get('model') != 'pi0' or
            receipt.get('phase') != phase or receipt.get('reviewer_model') != 'gpt-6.1-sol' or
            receipt.get('reasoning_effort') != 'medium' or receipt.get('revision_manifest_sha256') != sha(revision/'revision_manifest.json') or
            not receipt.get('evidence')):
        raise ValueError('Independent cache execution acceptance missing/stale')
    if evidence is not None and receipt.get('phase_receipt_sha256') != sha(evidence):
        raise ValueError('Cache acceptance does not bind exact phase evidence')
    return sha(path)


def reserve_gate(revision, manifest, parity):
    revision = Path(revision)
    identity = read(revision/'revision_manifest.json')
    attempt = Path(identity['source_attempt'])
    base = attempt/'readiness/pi0/diagnostic_budget.json'
    path = base.with_name('prompt_cache_diagnostic_budget.json')
    with lock(base.with_suffix('.lock')):
        original = read(base)
        throughput = base.with_name('throughput_diagnostic_budget.json')
        earlier = read(throughput)
        cache = read(path) if path.exists() else dict(schema=1, entries=[])
        revision_hash = sha(revision/'revision_manifest.json')
        if any(item['revision_manifest_sha256'] == revision_hash for item in cache['entries']):
            raise RuntimeError('Gate already reserved; no implicit diagnostic retry')
        total = parity['reserved_calls'] + original['smoke_reserved'] + prior_diagnostic_budget(manifest, 'pi0')
        total += sum(item['reserved_predictions'] for item in earlier['entries'])
        total += sum(item['reserved_predictions'] for item in cache['entries'])
        if total+16 > 256:
            raise RuntimeError('Combined pi0 diagnostic budget would exceed 256')
        entry = dict(revision_manifest_sha256=revision_hash, reserved_predictions=16, status='reserved',
            base_total=total, original_budget_sha256=sha(base), throughput_budget_sha256=sha(throughput),
            parity_sha256=sha(attempt/'parity/pi0.json'), timestamp=time.time())
        cache['entries'].append(entry)
        atomic_json(path, cache)
        return total+16


@contextmanager
def server(revision, output, gpu, fd, reference=False):
    from runtime_client import rpc
    identity, _ = verify_revision(revision)
    output = Path(output)
    output.mkdir(parents=True)
    port, env = available_port(), evaluation_env(gpu)
    env['V22_GPU_LEASE_FD'] = str(fd)
    command = [str(model_python('pi0')), str(HERE/'pi0_prompt_cache_server.py'), '--revision', str(revision),
        '--output-dir', str(output), '--port', str(port)]
    if reference:
        command.append('--reference')
    with (output/'server.log').open('a') as stream:
        process = subprocess.Popen(command, env=env, pass_fds=(fd,), stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+900
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Cache/reference server exited during load')
                try:
                    health = rpc(port, dict(op='health'))
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server load timed out')
                    time.sleep(1)
            parity = verify_parity(identity['source_attempt'], health['protocol'], 'pi0')
            verify_acceptance(identity['source_attempt'], 'pi0', parity)
            yield process, port, health, parity
        finally:
            stop_process(process)
            deadline = time.monotonic()+30
            while gpu not in free_gpus():
                if time.monotonic() > deadline:
                    raise RuntimeError('Prior server GPU resources did not naturally release; no replacement may start')
                time.sleep(.2)


def memory_sample(gpu, path):
    value = float(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    with Path(path).open('a') as stream:
        import json
        stream.write(json.dumps(dict(timestamp=time.time(), memory_mib=value))+'\n')
    if value >= MEMORY_CAP_MIB:
        raise RuntimeError('Total GPU memory reached the cache revision cap')
    return value


def gate(revision, gpu):
    from runtime_client import rpc, check_prediction
    from runtime_data import audited_protocol
    from runtime_inference import save_prediction
    revision = Path(revision).resolve()
    identity, manifest = verify_revision(revision)
    accepted = acceptance(revision, 'source')
    session = revision/'sessions'/f'pi0_gate_{uuid.uuid4().hex}'
    session.mkdir(parents=True)
    reference_dir = revision/'gate_reference'/session.name
    with ExitStack() as stack:
        for path in [Path(identity['source_attempt'])/'leases'/f'pi0.shard{shard}.lock' for shard in (0, 1)] + [revision/'manager.lock']:
            stack.enter_context(exclusive(path))
        fd = stack.enter_context(gpu_lease(gpu, allow_inherited=False))
        if gpu not in free_gpus():
            raise RuntimeError('GPU is not free for the independent cache gate')
        peak = 0.
        for reference, output, order in ((True, reference_dir, identity['reference_order']),
                                         (False, session/'server', identity['cached_order'])):
            with server(revision, output, gpu, fd, reference) as (process, port, health, parity):
                if reference:
                    total = reserve_gate(revision, manifest, parity)
                protocol = audited_protocol(identity['source_attempt'], 'pi0', health['protocol'])
                atomic_json(output/'gate_execution_receipt.json', dict(reference=reference,
                    revision_manifest_sha256=sha(revision/'revision_manifest.json'), model='pi0', port=port,
                    server_pid=process.pid, frozen_runtime_protocol=health['protocol'], calls_before=health['calls']))
                if not reference:
                    atomic_json(session/'execution_receipt.json', dict(schema=1, model='pi0', phase='gate', environments=0, gate_request_clients=1,
                        revision_manifest_sha256=sha(revision/'revision_manifest.json'), phase_acceptance_sha256=accepted,
                        frozen_runtime_protocol=health['protocol'], server_pid=process.pid, manager_pid=os.getpid(), gpu=gpu, port=port))
                for index in order:
                    request = identity['requests'][index]
                    observation, instruction, _ = load_observation(request['point'])
                    result = rpc(port, dict(op='predict', obs=observation, instruction=instruction,
                        skip=request['condition']['skip'], schedule=request['condition']['schedule'], seed=request['seed'], category='cache_equivalence'))
                    path = (reference_dir/'outputs' if reference else session/'outputs')/f'{index}.json'
                    row = save_prediction(path, result, seed=request['seed'], condition=request['condition']['id'])
                    check_prediction(row, request['condition'], protocol, request['seed'])
                    peak = max(peak, memory_sample(gpu, session/'gpu_memory.jsonl'))
                if rpc(port, dict(op='health'))['calls']-health['calls'] != 8:
                    raise ValueError('Gate server made a different number of logical calls')
        checks = compare_gate(revision, session, reference_dir)
        ready = dict(schema=1, model='pi0', phase='gate', revision_manifest_sha256=sha(revision/'revision_manifest.json'),
            session=str(session), reference_session=str(reference_dir), actual_predictions=16,
            total_diagnostic_reserved=total, comparisons=checks, exact=True, peak_sampled_gpu_memory_mib=peak,
            memory_cap_mib=MEMORY_CAP_MIB, cached_order=identity['cached_order'], reference_order=identity['reference_order'])
        atomic_json(revision/'pi0.gate_ready.json', ready)
        budget_path = Path(identity['source_attempt'])/'readiness/pi0/prompt_cache_diagnostic_budget.json'
        with lock(budget_path.with_name('diagnostic_budget.lock')):
            budget = read(budget_path)
            item = next(item for item in budget['entries'] if item['revision_manifest_sha256'] == sha(revision/'revision_manifest.json'))
            item.update(status='gate_complete', actual_predictions=16, gate_receipt_sha256=sha(revision/'pi0.gate_ready.json'))
            atomic_json(budget_path, budget)
        return ready


def compare_gate(revision, session, reference_dir):
    identity = read(Path(revision)/'revision_manifest.json')
    checks = []
    for request in identity['requests']:
        index = request['index']
        a, b = [read(path) for path in (Path(reference_dir)/'outputs'/f'{index}.json', Path(session)/'outputs'/f'{index}.json')]
        for key in ('actions', 'input_hash', 'noise_hash', 'raw_prediction_hash', 'seed', 'loaded_code_hashes'):
            if a[key] != b[key]:
                raise ValueError(f'Cache/reference {key} differs')
        for key in ('NFE', 'actual_NFE', 'skip', 'schedule', 'cache_key'):
            if a['audit'].get(key) != b['audit'].get(key):
                raise ValueError(f'Cache/reference audit {key} differs')
        for key in ('noise_path', 'server_prediction_path'):
            with np.load(a[key], allow_pickle=False) as left, np.load(b[key], allow_pickle=False) as right:
                if left.files != right.files or any(not np.array_equal(left[name], right[name]) for name in left.files):
                    raise ValueError('Cache/reference raw/noise arrays differ')
        event = b['execution_prompt_cache']
        if (event['prompt_length'] != request['prompt_length'] or len(event['resident_lengths']) > 2 or
                event['hit'] != (index >= 2) or event['startup_forward_calls'] != (0 if index >= 2 else 4) or
                event['startup_action_NFE'] != (0 if index >= 2 else 40) or event['evictions']):
            raise ValueError('LRU A/B hit/startup/memory lifecycle gate failed')
        checks.append(dict(index=index, exact=True, max_abs=0., reference_sha256=sha(Path(reference_dir)/'outputs'/f'{index}.json'),
            cached_sha256=sha(Path(session)/'outputs'/f'{index}.json'), hit=event['hit'], startup_action_NFE=event['startup_action_NFE']))
    return checks


def run(revision, gpu, phase):
    from runtime_client import rpc
    revision = Path(revision).resolve()
    identity, manifest = verify_revision(revision)
    required = 'gate' if phase == 'pilot' else 'pilot'
    accepted = acceptance(revision, required, revision/f'pi0.{required}_ready.json')
    session = revision/'sessions'/f'pi0_{phase}_{uuid.uuid4().hex}'
    session.mkdir(parents=True)
    def stop(*_):
        atomic_json(session/'STOP', dict(reason='manager drain', timestamp=time.time()))
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with ExitStack() as stack:
        for path in [Path(identity['source_attempt'])/'leases'/f'pi0.shard{shard}.lock' for shard in (0, 1)] + [revision/'manager.lock']:
            stack.enter_context(exclusive(path))
        fd = stack.enter_context(gpu_lease(gpu, allow_inherited=False))
        if gpu not in free_gpus():
            raise RuntimeError('GPU is busy before cache execution')
        if phase == 'pilot':
            if (revision/'pi0.pilot_session.json').exists():
                raise RuntimeError('Pilot session may not be repeated within a revision')
            atomic_json(revision/'pi0.pilot_session.json', dict(session=str(session), revision_manifest_sha256=sha(revision/'revision_manifest.json')))
        with server(revision, session/'server', gpu, fd) as (process, port, health, parity):
            atomic_json(session/'execution_receipt.json', dict(schema=1, model='pi0', phase=phase, environments=2,
                revision_manifest_sha256=sha(revision/'revision_manifest.json'), phase_acceptance_sha256=accepted,
                frozen_runtime_protocol=health['protocol'], server_pid=process.pid, manager_pid=os.getpid(), gpu=gpu, port=port))
            clients = []
            start, peak, ready = time.time(), 0., False
            try:
                for slot in (0, 1):
                    stream = stack.enter_context((session/f'client{slot}.log').open('a'))
                    clients.append(subprocess.Popen([str(CLIENT), str(HERE/'pi0_prompt_cache_client.py'), '--revision', str(revision),
                        '--session', str(session), '--port', str(port), '--slot', str(slot), '--phase', phase],
                        env=evaluation_env(gpu), stdout=stream, stderr=subprocess.STDOUT))
                while any(child.poll() is None for child in clients):
                    if process.poll() is not None:
                        for child in clients:
                            stop_process(child)
                        raise RuntimeError('Cache server exited; preserve and diagnose reservations')
                    if any(child.poll() not in (None, 0) for child in clients):
                        stop()
                    if phase == 'pilot' and not ready and all((session/f'pilot{slot}.json').exists() for slot in (0, 1)):
                        atomic_json(revision/'pi0.pilot_ready.json', dict(schema=1, model='pi0', phase='pilot',
                            revision_manifest_sha256=sha(revision/'revision_manifest.json'), session=str(session),
                            results=[read(session/f'pilot{slot}.json') for slot in (0, 1)],
                            evaluation_seconds=time.time()-start, peak_sampled_gpu_memory_mib=peak, pending_medium_acceptance=True))
                        ready = True
                    try:
                        peak = max(peak, memory_sample(gpu, session/'gpu_memory.jsonl'))
                    except RuntimeError:
                        stop()
                        raise
                    time.sleep(1)
                if any(child.returncode != 0 for child in clients):
                    raise RuntimeError('Cache client failed; no automatic retry')
                summaries = [read(session/f'client{slot}_summary.json') for slot in (0, 1)]
                atomic_json(session/'complete.json', dict(schema=1, model='pi0', phase=phase, session=str(session),
                    revision_manifest_sha256=sha(revision/'revision_manifest.json'), client_summaries=summaries,
                    valid=sum(item['valid'] for item in summaries), evaluation_seconds=time.time()-start,
                    peak_sampled_gpu_memory_mib=peak, server_calls=rpc(port, dict(op='health'))['calls']))
            finally:
                if any(child.poll() is None for child in clients):
                    stop()
                    for child in clients:
                        child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'gate', 'pilot', 'run'))
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--attempt', type=Path)
    parser.add_argument('--base-revision', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.operation == 'prepare':
        if args.attempt is None or args.base_revision is None:
            parser.error('prepare requires --attempt and --base-revision')
        prepare(args.attempt, args.base_revision, args.revision)
    elif args.operation == 'gate':
        gate(args.revision, args.gpu)
    else:
        run(args.revision, args.gpu, args.operation)


if __name__ == '__main__':
    main()
