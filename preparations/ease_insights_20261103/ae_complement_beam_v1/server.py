"""One persistent native VLA service for archived snapshot replay."""
import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback
from .common import read_manifest, validate_output, planned_config, atomic_json, require, sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('manifest', 'condition', 'gpu-uuid', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid, 'Wrong assigned GPU UUID')
    manifest = read_manifest(args.manifest, verify_sources=True)
    validate_output(Path(args.manifest).parent, manifest)
    cfg = planned_config(manifest, args.condition)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from preparations.idea1_v3.safe_qk import install
    install(str(output / 'qk_receipt.json'))
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    from preparations.ease_insights_20261103.module_tradeoff_v1.backend import Backend
    from preparations.idea1_v2_1.runtime_client import recv_exact
    from .engine import gate, replay
    start = time.monotonic()
    backend = Backend()
    require(backend.protocol['checkpoint_sha256'] == manifest['checkpoints']['pi0']['sha256'], 'Wrong current checkpoint')
    norm = manifest['normalization']
    require(backend.protocol['checkpoint_assets'][norm['path']]['sha256'] == norm['sha256'], 'Loaded normalization asset differs')
    numerical = gate(backend, manifest, output)
    qk = json.loads((output / 'qk_receipt.json').read_text())
    require(qk.get('passed') is True and qk.get('unsafe_legacy_kernel_executed') is False and
            qk.get('atol') == .0009765625 and qk.get('rtol') == .0078125, 'QK gate differs')
    protocol = dict(backend.protocol, manifest_hash=manifest['manifest_hash'], condition_id=args.condition, config=cfg,
                    gpu_uuid=args.gpu_uuid, server_pid=os.getpid(), real_input_parity_passed=numerical['passed'],
                    qk_reference_passed=True, mode='offline_snapshot_replay', physical_episodes=0,
                    normalization=manifest['normalization'], normalization_actual_sha256=sha(manifest['normalization']['path']),
                    preparation_s=time.monotonic() - start,
                    input_contract='Already rotated/resized224x224 images and direct8D state; native adapter preprocessed-input branch; no second rotation or quaternion reconstruction',
                    comparison='Fresh denseAE18/D2 and all153 AE16/D2 masks; identical full VLM18 prefix and noise per snapshot/seed',
                    quality_or_speed_claim=False)
    atomic_json(output / 'protocol.json', protocol)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', args.port))
        listener.listen(4)
        print('READY', flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                try:
                    request = pickle.loads(recv_exact(connection, struct.unpack('!I', recv_exact(connection, 4))[0]))
                    if request == dict(op='health'):
                        value = dict(protocol=protocol)
                    elif set(request) == {'op', 'index'} and request['op'] == 'screen':
                        value = replay(backend, manifest, request['index'])
                    else:
                        raise ValueError('Unsupported replay request')
                    value['ok'] = True
                except Exception:
                    value = dict(ok=False, error=traceback.format_exc())
                payload = pickle.dumps(value, protocol=5)
                connection.sendall(struct.pack('!I', len(payload)) + payload)


if __name__ == '__main__':
    main()
