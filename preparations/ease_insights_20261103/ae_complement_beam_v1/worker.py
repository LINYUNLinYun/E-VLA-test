"""Commit complete archived-snapshot comparisons without creating an environment."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v2_1.runtime_client import rpc
from .common import *
from .engine import noise_hash
from .metrics import errors


def completed(output, cid, scene, manifest_hash):
    path = replay_dir(output, cid, scene) / 'replay_receipt.json'
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    require(value['valid'] is True and value['manifest_hash'] == manifest_hash and value['case'] == scene and
            value['condition'] == cid and value['physical_episodes'] == 0, 'Existing replay identity differs')
    for name, pin in value['artifacts'].items():
        require(sha(path.parent / name) == pin, 'Existing replay artifact changed')
    return True


def validate_record(output, cid, scene, manifest):
    directory = replay_dir(output, cid, scene)
    identity = json.loads((directory / 'identity.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    expected = manifest['observations'][scene['index']]
    require(identity['case'] == result['case'] == scene and identity['manifest_hash'] == result['manifest_hash'] == manifest['manifest_hash'], 'Replay identity differs')
    require(result['masks'] == [[], *MASKS] and result['noise_seeds'] == manifest['noise_seeds'] and
            result['input_sha256'] == expected['sha256'] == sha(expected['path']), 'Snapshot, noise or joint-mask assignment differs')
    require(result['generated_noise_fp32_sha256'] == [noise_hash(s) for s in manifest['noise_seeds']], 'Generated noise identity differs')
    require(result['physical_episodes'] == 0 and result['calls'] == 462 and len(result['execution_receipts']) == 154, 'Replay sample count differs')
    for mask, receipt in zip([[], *MASKS], result['execution_receipts']):
        require(receipt['actual_NFE'] == 2 and receipt['bypassed_layers_per_step'] == mask and
                receipt['executed_layers_per_step'] == [i for i in range(1, 19) if i not in mask] and
                receipt['actual_layer_calls'] == (18 - len(mask)) * 2 and receipt['original_helper_restored'], 'Observed execution differs')
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        require(arrays['raw'].shape == (3, 154, 50, 32) and np.isfinite(arrays['raw']).all(), 'Invalid complete raw action array')
        computed = errors(arrays['actions'])
        for metric in ('mse', 'nmse'):
            require(np.array_equal(computed[metric], arrays[metric]), 'Derived replay error differs')
        for metric in ('ae_ms', 'decode_ms'):
            require(arrays[metric].shape == (3, 154) and np.isfinite(arrays[metric]).all() and (arrays[metric] >= 0).all(), 'Invalid measured native component timing')
    receipt = dict(valid=True, manifest_hash=manifest['manifest_hash'], condition=cid, case=scene,
                   physical_episodes=0, observations=1, repeated_noise_seeds=3, joint_mask_samples=459, dense_D2_samples=3,
                   artifacts={name: sha(directory / name) for name in ['identity.json', 'result.json', 'arrays.npz']},
                   scope='Offline command-error screen; no rollout success, independent initial state, or speed claim')
    return receipt


def finalize(output, cid, scene, manifest):
    receipt = validate_record(output, cid, scene, manifest)
    atomic_json(replay_dir(output, cid, scene) / 'replay_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('output', 'condition', 'stage'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--shard-index', required=True, type=int)
    parser.add_argument('--shards', required=True, type=int)
    args = parser.parse_args()
    manifest = read_manifest(Path(args.output) / 'manifest.json', verify_sources=True)
    validate_output(args.output, manifest)
    planned_config(manifest, args.condition)
    protocol = rpc(args.port, dict(op='health'))['protocol']
    require(protocol['manifest_hash'] == manifest['manifest_hash'] and protocol['condition_id'] == args.condition and
            protocol['mode'] == 'offline_snapshot_replay' and protocol['real_input_parity_passed'], 'Wrong replay service')
    for scene in cases(manifest, args.stage, args.shard_index, args.shards):
        if completed(args.output, args.condition, scene, manifest['manifest_hash']):
            continue
        directory = replay_dir(args.output, args.condition, scene)
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / 'identity.json', dict(manifest_hash=manifest['manifest_hash'], condition=args.condition,
                    case=scene, worker_pid=os.getpid(), protocol_server_pid=protocol['server_pid'], physical_episodes=0))
        begin = time.monotonic()
        reply = rpc(args.port, dict(op='screen', index=scene['index']))
        arrays = reply.pop('arrays')
        require(reply.pop('ok') is True and reply['index'] == scene['index'] and reply['sample_id'] == scene['sample_id'], 'Wrong snapshot reply')
        temporary = directory / 'arrays.tmp.npz'
        np.savez_compressed(temporary, **arrays)
        temporary.replace(directory / 'arrays.npz')
        reply.update(condition=args.condition, case=scene, calls=462, wall_s=time.monotonic() - begin)
        atomic_json(directory / 'result.json', reply)
        receipt = finalize(args.output, args.condition, scene, manifest)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
