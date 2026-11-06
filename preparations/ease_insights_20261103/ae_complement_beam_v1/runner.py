"""Adapter for the existing retained GPU pool; no direct GPU lifecycle changes."""
import json
from pathlib import Path
import time
from preparations.idea1_v3_1.common import environment, lock
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.ease_insights_20261103.vlm_local_search_v1.runner import resolve_binding
from .common import *


def recover_case(output, cid, scene, manifest, finalize):
    path = replay_dir(output, cid, scene)
    if not path.exists():
        return
    required = ['identity.json', 'result.json', 'arrays.npz']
    require(all((path / name).exists() for name in required), 'Partial offline record requires explicit technical recovery: ' + str(path))
    if not (path / 'replay_receipt.json').exists():
        finalize(output, cid, scene, manifest)


def cross_service(output, logs, manifest_hash):
    import numpy as np
    output, logs = Path(output), Path(logs)
    with lock(output / 'cross_service.lock'):
        ref, identity = output / 'cross_service_reference.npz', output / 'cross_service_reference.json'
        require(ref.exists() == identity.exists(), 'Incomplete replay reference')
        if not ref.exists():
            ref.write_bytes((logs / 'gate_actions.npz').read_bytes())
            atomic_json(identity, dict(manifest_hash=manifest_hash, sha256=sha(ref), first_service=str(logs)))
        bound = json.loads(identity.read_text())
        require(bound['manifest_hash'] == manifest_hash and bound['sha256'] == sha(ref), 'Replay reference identity changed')
        checks = []
        with np.load(ref, allow_pickle=False) as a, np.load(logs / 'gate_actions.npz', allow_pickle=False) as b:
            require(a.files == b.files, 'Finite gate keys differ')
            for key in a.files:
                require(a[key].shape == b[key].shape and np.isfinite(b[key]).all(), 'Invalid finite gate arrays')
                checks.append(dict(key=key, max_abs=float(np.abs(a[key] - b[key]).max()), passed=bool(np.allclose(a[key], b[key], atol=1e-5, rtol=1e-5))))
        result = dict(passed=all(c['passed'] for c in checks), checks=checks, manifest_hash=manifest_hash,
                      atol=1e-5, rtol=1e-5, scope='Four initial archived snapshots; not all110 inputs or closed-loop quality')
        atomic_json(logs / 'cross_service.json', result)
        require(result['passed'], 'Cross-service replay numerical gate failed')
        return result


def run(args):
    require(not args.launch, 'Use the existing retained resource pool')
    output = Path(args.output).resolve()
    manifest = read_manifest(output / 'manifest.json')
    validate_output(output, manifest)
    planned_config(manifest, args.condition)
    stage = stage_for(manifest)
    require(args.stage == stage, 'Replay stage differs')
    scenes = cases(manifest, stage, args.shard_index, args.shards)
    logs = output / args.condition / 'runs' / f'{stage}_shard{args.shard_index:02d}of{args.shards:02d}_{time.time_ns()}'
    server = [str(MODEL_PYTHON), '-B', '-m', MODULE + '.server', '--manifest', str(output / 'manifest.json'),
              '--condition', args.condition, '--port', str(args.port), '--gpu-uuid', args.gpu_uuid, '--output', str(logs)]
    worker = [str(MODEL_PYTHON), '-B', '-m', MODULE + '.worker', '--output', str(output), '--condition', args.condition,
              '--port', str(args.port), '--stage', stage, '--shard-index', str(args.shard_index), '--shards', str(args.shards)]
    marker = output / args.condition / f'{stage}_shard{args.shard_index:02d}of{args.shards:02d}_complete.json'
    return dict(args=args, output=output, manifest=manifest, scenes=scenes, logs=logs, marker=marker,
                server=server, worker=worker)
