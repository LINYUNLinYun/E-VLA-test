"""Explicit benchmark execution revision; frozen scientific runtime is unchanged."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

from common import atomic_json, gpu_lease, sha
import runtime_inference
from runtime_data import verify_frozen
from benchmark_gpu_identity import CausalIdentity

HERE = Path(__file__).resolve().parent
REVISION_FILES = ('benchmark_revision.py', 'benchmark_gpu_identity.py', 'benchmark_revision_analysis.py')


def read(path):
    return json.loads(Path(path).read_text())


def original_function_hash():
    return hashlib.sha256(inspect.getsource(runtime_inference.assert_exclusive_gpu).encode()).hexdigest()


def prepare(parent, revision):
    parent, revision = Path(parent).resolve(), Path(revision).resolve()
    manifest = verify_frozen(parent)
    if revision.exists():
        raise ValueError('Revision must be a new directory; never resume/reuse benchmark samples')
    identity = dict(schema=1, kind='benchmark_execution_revision', source_attempt=str(parent),
                    manifest_hash=manifest['manifest_hash'],
                    source_manifest_sha256=sha(parent/'condition_manifest.json'),
                    frozen_runtime_code_hashes=manifest['code_hashes'],
                    original_exclusivity_function_sha256=original_function_hash(),
                    revision_code_hashes={name: sha(HERE/name) for name in REVISION_FILES},
                    timing_change=False, identity_method='causal CUDA allocation/release; not OS PID mapping',
                    historical_benchmark_policy='all parent benchmark samples excluded; no reuse',
                    required_models=list(manifest['protocol_hashes']), created=time.time())
    revision.mkdir(parents=True)
    copied = {}
    for name in ('condition_manifest.json', 'runtime_acceptance.json'):
        shutil.copy2(parent/name, revision/name)
        copied[name] = sha(revision/name)
    for name in ('protocols', 'parity'):
        shutil.copytree(parent/name, revision/name)
        copied.update({str(p.relative_to(revision)): sha(p) for p in (revision/name).rglob('*')
                       if p.is_file() and not p.name.endswith('.runtime.json')})
    identity['copied_metadata_hashes'] = copied
    atomic_json(revision/'revision_manifest.json', identity)
    return identity


def verify_revision(revision, require_acceptance=True):
    revision = Path(revision).resolve()
    identity = read(revision/'revision_manifest.json')
    parent = Path(identity['source_attempt']).resolve()
    manifest = verify_frozen(parent)
    verify_frozen(revision)
    if identity['kind'] != 'benchmark_execution_revision' or identity['schema'] != 1:
        raise ValueError('Unknown benchmark revision schema')
    if sha(parent/'condition_manifest.json') != identity['source_manifest_sha256'] or manifest['manifest_hash'] != identity['manifest_hash']:
        raise ValueError('Parent manifest identity changed')
    if manifest['code_hashes'] != identity['frozen_runtime_code_hashes']:
        raise ValueError('Frozen runtime identity changed')
    if original_function_hash() != identity['original_exclusivity_function_sha256']:
        raise ValueError('Original exclusivity function changed')
    for name, expected in identity['revision_code_hashes'].items():
        if sha(HERE/name) != expected:
            raise ValueError(f'Revision source changed: {name}')
    for name, expected in identity['copied_metadata_hashes'].items():
        if sha(revision/name) != expected or sha(parent/name) != expected:
            raise ValueError(f'Copied source metadata changed: {name}')
    if require_acceptance:
        acceptance = read(revision/'benchmark_revision_acceptance.json')
        if (acceptance.get('schema') != 1 or acceptance.get('accepted') is not True or
                acceptance.get('reviewer_model') != 'gpt-6.1-sol' or acceptance.get('reasoning_effort') != 'medium' or
                acceptance.get('revision_manifest_sha256') != sha(revision/'revision_manifest.json') or
                not acceptance.get('evidence')):
            raise ValueError('Independent medium benchmark revision acceptance missing/stale')
    return identity


def run(revision, model, gate_only=False):
    revision = Path(revision).resolve()
    identity = verify_revision(revision, require_acceptance=not gate_only)
    if model not in identity['required_models']:
        raise ValueError('Unknown model')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.isdecimal():
        raise ValueError('Select exactly one physical GPU index')
    destination = revision/('gates' if gate_only else 'benchmark')/model
    # A failed or interrupted revision is retained, never implicitly resumed.
    destination.mkdir(parents=True, exist_ok=False)
    log = destination/'identity_events.jsonl'
    def record(row):
        with log.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
    with gpu_lease(gpu, allow_inherited=False) as fd:
        os.environ['V22_GPU_LEASE_FD'] = str(fd)
        import torch
        binding = CausalIdentity(gpu, record)
        binding.bind(torch)
        atomic_json(destination/'execution_receipt.json', dict(
            schema=1, revision_manifest_sha256=sha(revision/'revision_manifest.json'),
            acceptance_sha256=None if gate_only else sha(revision/'benchmark_revision_acceptance.json'),
            model=model, gate_only=gate_only, gpu=gpu, uuid=binding.uuid,
            local=binding.local, nvml_pid=binding.pid, identity_method='causal allocation/release binding'))
        if gate_only:
            binding.assert_exclusive(snapshot=True)
            atomic_json(destination/'gate_complete.json', dict(identity_events_sha256=sha(log), passed=True))
            return
        runtime_inference.assert_exclusive_gpu = binding.assert_exclusive
        # Only the original benchmark branch; no parity/offline/formal entry exists here.
        runtime_inference._execute(revision, model, 'benchmark')
        if not (destination/'complete.json').exists():
            raise RuntimeError('Benchmark interrupted without completion')
        atomic_json(destination/'revision_complete.json', dict(
            identity_events_sha256=sha(log), execution_receipt_sha256=sha(destination/'execution_receipt.json'),
            complete_sha256=sha(destination/'complete.json'), exclusive=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'gate', 'benchmark'])
    parser.add_argument('--revision', required=True)
    parser.add_argument('--parent')
    parser.add_argument('--model')
    args = parser.parse_args()
    if args.operation == 'prepare':
        if not args.parent:
            parser.error('prepare requires --parent')
        prepare(args.parent, args.revision)
    else:
        if not args.model:
            parser.error('gate/benchmark requires --model')
        run(args.revision, args.model, gate_only=args.operation == 'gate')


if __name__ == '__main__':
    main()
