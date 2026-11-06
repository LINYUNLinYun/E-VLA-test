"""Independent sampled all-process benchmark revision; canonical scientific loop."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from common import atomic_json, gpu_lease, lock, exclusive, sha
import runtime_inference as original
from runtime_data import verify_frozen, verify_parity, verify_acceptance, load_observation
from benchmark_revision import prepare as prepare_base, verify_revision as unused_base_verify, read
from benchmark_gpu_identity import CausalIdentity
from benchmark_all_process_guard import Scanner, Monitor, abi_identity, INTERVAL, MAX_GAP

HERE = Path(__file__).resolve().parent
KIND = 'benchmark_sampled_all_process_revision'
SOURCES = ('benchmark_all_process_guard.py', 'benchmark_all_process_revision.py',
    'benchmark_all_process_analysis.py', 'final_provenance_analysis_v4.py')


def prepare(parent, revision):
    identity = prepare_base(parent, revision)
    manifest = read(Path(parent)/'condition_manifest.json')
    if len(manifest['conditions']) != 54 or any(len(points) != 32 for points in manifest['observations'].values()):
        raise ValueError('Original benchmark scope must remain 54 conditions x32 points x25 calls')
    identity.update(kind=KIND, revision_code_hashes={**identity['revision_code_hashes'], **{name:sha(HERE/name) for name in SOURCES}},
        guard=dict(interval=INTERVAL, max_gap=MAX_GAP, compute='bound causal PID only', graphics='empty', abi=abi_identity()),
        original_execute_sha256=hashlib.sha256(inspect.getsource(original._execute).encode()).hexdigest(),
        authorization='per-model/per-phase receipt; no global v1 authorization accepted',
        isolation_claim='all observed samples and per-call boundaries only; not continuous nanosecond isolation',
        observer_effect='monitor may perturb measurements; no subtraction and no zero-effect claim')
    identity['required_total_calls'] = 43200
    atomic_json(Path(revision)/'revision_manifest.json', identity)
    return identity


def verify_revision(revision):
    revision = Path(revision).resolve()
    identity = read(revision/'revision_manifest.json')
    parent = Path(identity['source_attempt'])
    manifest = verify_frozen(parent)
    verify_frozen(revision)
    if (identity.get('kind') != KIND or identity['schema'] != 1 or identity['source_manifest_sha256'] != sha(parent/'condition_manifest.json')
            or identity.get('required_total_calls') != 43200 or len(manifest['conditions']) != 54
            or any(len(points) != 32 for points in manifest['observations'].values())
            or identity['manifest_hash'] != manifest['manifest_hash'] or identity['frozen_runtime_code_hashes'] != manifest['code_hashes']
            or identity['guard'] != dict(interval=INTERVAL, max_gap=MAX_GAP, compute='bound causal PID only', graphics='empty', abi=abi_identity())
            or identity['original_execute_sha256'] != hashlib.sha256(inspect.getsource(original._execute).encode()).hexdigest()
            or set(identity['required_models']) != set(manifest['protocol_hashes'])):
        raise ValueError('Unknown/changed all-process benchmark revision')
    for name, expected in identity['revision_code_hashes'].items():
        if sha(HERE/name) != expected:
            raise ValueError('Benchmark revision source changed')
    for name, expected in identity['copied_metadata_hashes'].items():
        if sha(revision/name) != expected or sha(parent/name) != expected:
            raise ValueError('Frozen benchmark metadata changed')
    return identity


def authorize(revision, model, phase):
    identity = verify_revision(revision)
    if model not in identity['required_models'] or phase not in ('allocation_gate', 'benchmark'):
        raise ValueError('Unknown benchmark model/phase')
    path = Path(revision)/'acceptance'/f'{model}.{phase}.json'
    receipt = read(path)
    if (receipt.get('schema') != 1 or receipt.get('accepted') is not True or receipt.get('model') != model
            or receipt.get('phase') != phase or receipt.get('reviewer_model') != 'gpt-6.1-sol'
            or receipt.get('reasoning_effort') != 'medium' or not receipt.get('evidence')
            or receipt.get('revision_manifest_sha256') != sha(Path(revision)/'revision_manifest.json')
            or receipt.get('revision_code_hashes') != identity['revision_code_hashes']
            or not str(receipt.get('uuid', '')).startswith('GPU-')):
        raise ValueError('Missing/stale exact per-model phase authorization')
    if phase == 'benchmark':
        gate = Path(receipt['gate_receipt']).resolve()
        if not gate.is_relative_to(Path(revision).resolve()/'attempts'/model) or sha(gate) != receipt['gate_receipt_sha256']:
            raise ValueError('Benchmark authorization lacks its own exact allocation gate')
        proof = read(gate)
        if proof.get('passed') is not True or proof.get('model') != model or proof.get('phase') != 'allocation_gate' or proof['uuid'] != receipt['uuid'] or proof['revision_manifest_sha256'] != sha(Path(revision)/'revision_manifest.json'):
            raise ValueError('Allocation gate does not authorize this model/UUID/revision')
    return receipt, path


def publish_selection(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def point_loop(root, model, policy, monitor, seal):
    manifest = read(root/'condition_manifest.json')
    points = manifest['observations'][model]
    conds = [c for c in manifest['conditions'] if c['model'] == model]
    transport = original.benchmark_transport(policy)
    for point in points:
        if (root/'PAUSE').exists():
            raise RuntimeError('Benchmark paused; partial attempt invalid')
        obs, instruction, seed = load_observation(point)
        expected_noise = None
        for repeat in range(25):
            rotation = (point['index']+repeat)%len(conds)
            for condition in conds[rotation:]+conds[:rotation]:
                if (root/'PAUSE').exists():
                    raise RuntimeError('Benchmark paused; partial attempt invalid')
                token = f"p{point['index']:02d}_r{repeat:02d}_{condition['id']}"
                before = policy.calls.copy() if isinstance(policy.calls, dict) else policy.calls
                pre = monitor.check(token+':before')  # Outside the original measured interval.
                started = time.monotonic()
                tick = time.perf_counter()
                result = transport(obs=obs, instruction=instruction, seed=seed, skip=condition['skip'],
                    schedule=condition['schedule'], category='warmup' if repeat < 5 else 'measurement')
                elapsed = (time.perf_counter()-tick)*1000
                ended = time.monotonic()
                post = monitor.check(token+':after')  # No guard cost subtraction.
                result.setdefault('timing', {})['client_roundtrip_ms'] = elapsed
                row = original.save_prediction(root/'benchmark'/model/'samples'/f'{token}.json', result,
                    condition=condition['id'], point=point['index'], seed=seed, repeat=repeat,
                    phase='warmup' if repeat < 5 else 'measurement', independent_forward=True,
                    calls_before=before, calls_after=policy.calls,
                    all_process_guard=dict(before_sequence=pre, after_sequence=post, started=started, ended=ended))
                if expected_noise and row['noise_hash'] != expected_noise:
                    raise ValueError('Benchmark noise not paired')
                expected_noise = row['noise_hash']
        seal(f"point{point['index']:02d}")  # Memory monitor chunk persisted only outside timed calls.
    atomic_json(root/'benchmark'/model/'complete.json', dict(conditions=len(conds), points=32, warmup=5, measurement=20, exclusive=True))


def run(revision, model, gate_only=False):
    revision = Path(revision).resolve()
    if model not in verify_revision(revision)['required_models']:
        raise ValueError('Unknown benchmark model')
    phase = 'allocation_gate' if gate_only else 'benchmark'
    with exclusive(revision/'leases'/f'{model}.{phase}.lock'):
        return _run(revision, model, gate_only)


def _run(revision, model, gate_only=False):
    revision = Path(revision).resolve()
    phase = 'allocation_gate' if gate_only else 'benchmark'
    approval, approval_path = authorize(revision, model, phase)
    if not gate_only and (revision/'selected'/f'{model}.json').exists():
        raise ValueError('A completed model benchmark is already selected; never overwrite/retry implicitly')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.isdecimal():
        raise ValueError('One physical GPU index required')
    identity = verify_revision(revision)
    root = revision/'attempts'/model/uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    for name in ('condition_manifest.json', 'runtime_acceptance.json'):
        shutil.copy2(revision/name, root/name)
    for name in ('protocols', 'parity'):
        shutil.copytree(revision/name, root/name)
    chunks, binding_events, scans, monitor, scanner = [], [], [], None, None
    def seal(label, stop=False):
        response = monitor.flush(label, stop=stop)
        path = root/'monitor'/f'{len(chunks):04d}_{label}.json'
        atomic_json(path, response)
        chunks.append(dict(path=str(path), sha256=sha(path)))
        if response.get('error'):
            raise RuntimeError(response['error'])
    try:
        with gpu_lease(gpu, allow_inherited=False) as fd:
            os.environ['V22_GPU_LEASE_FD'] = str(fd)
            scanner = Scanner(gpu=gpu)
            if scanner.uuid != approval['uuid']:
                raise ValueError('Assigned GPU UUID differs from exact model authorization')
            def scan(_gpu):
                value = scanner.scan()
                value['gpu'] = gpu
                scans.append(value)
                if value['graphics'] or any(memory is None for memory in value['processes'].values()):
                    raise RuntimeError('Graphics process or unknown memory in causal binding baseline')
                return value
            import torch
            binding = CausalIdentity(gpu, binding_events.append, scanner=scan)
            binding.bind(torch)
            atomic_json(root/'binding.json', dict(events=binding_events, scans=scans))
            monitor = Monitor(binding.uuid, binding.pid, binding.local)
            monitor.check('execution_start')
            execution = dict(schema=1, model=model, phase=phase, revision_manifest_sha256=sha(revision/'revision_manifest.json'),
                authorization_path=str(approval_path), authorization_sha256=sha(approval_path), uuid=binding.uuid,
                gpu=gpu, local=binding.local, nvml_pid=binding.pid, isolation_claim=identity['isolation_claim'], observer_effect=identity['observer_effect'])
            atomic_json(root/'execution_receipt.json', execution)
            if not gate_only:
                from runtime_models import NativePolicy
                with lock(root/'leases'/f'{model}.inference.lock'):
                    policy = NativePolicy(model, root/'native'/model)
                    verify_frozen(root, policy.protocol, model)
                    parity = verify_parity(root, original.serial(policy.protocol), model)
                    verify_acceptance(root, model, parity)
                    atomic_json(root/'protocols'/f'{model}.runtime.json', original.serial(policy.protocol))
                    monitor.check('model_loaded')
                    seal('startup')
                    point_loop(root, model, policy, monitor, seal)
            seal('final', stop=True)
            monitor = None
            scanner.close()
            scanner = None
            receipt = dict(schema=1, model=model, phase=phase, passed=True, uuid=binding.uuid,
                revision_manifest_sha256=sha(revision/'revision_manifest.json'), binding_sha256=sha(root/'binding.json'),
                execution_receipt_sha256=sha(root/'execution_receipt.json'), monitor_chunks=chunks,
                observation_claim=identity['isolation_claim'])
            if not gate_only:
                receipt['scientific_complete_sha256'] = sha(root/'benchmark'/model/'complete.json')
            atomic_json(root/('gate_complete.json' if gate_only else 'revision_complete.json'), receipt)
            if not gate_only:
                publish_selection(revision/'selected'/f'{model}.json', dict(model=model, attempt=str(root),
                    revision_manifest_sha256=sha(revision/'revision_manifest.json'),
                    completion_sha256=sha(root/'revision_complete.json')))
    except BaseException as error:
        if monitor:
            try: seal('failed', stop=True)
            except BaseException as secondary:
                atomic_json(root/'monitor_incomplete.json', dict(error=repr(secondary)))
        atomic_json(root/'binding_failure_evidence.json', dict(events=binding_events, scans=scans))
        atomic_json(root/'invalid_attempt.json', dict(passed=False, model=model, phase=phase, error=repr(error), monitor_chunks=chunks))
        raise
    finally:
        if scanner:
            scanner.close()
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'gate', 'benchmark'))
    parser.add_argument('--revision', required=True)
    parser.add_argument('--parent')
    parser.add_argument('--model')
    args = parser.parse_args()
    if args.operation == 'prepare':
        if not args.parent: parser.error('prepare requires parent')
        prepare(args.parent, args.revision)
    else:
        if not args.model: parser.error('execution requires model')
        print(run(args.revision, args.model, args.operation == 'gate'))


if __name__ == '__main__':
    main()
