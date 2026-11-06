"""Serial B1 module-skip service with two isolated same-task clients."""
import argparse
import gc
import json
import os
import pickle
import socket
import struct
import time
import traceback
from pathlib import Path

from .common import OUT, read_manifest
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', default=str(OUT / 'manifest.json'))
    parser.add_argument('--force-single', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    import torch
    from preparations.idea1_v3.safe_qk import install
    from preparations.idea1_v3_4_E2_module_skip.backend import Backend
    from .policy import Policy
    from .preflight import preflight
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    install(str(out / 'qk_receipt.json'))
    backend = Backend()
    measured_protocol = manifest['cost_table']['provenance']['protocol']
    for field in ('checkpoint_sha256', 'dtype', 'attention', 'backend'):
        if field in ('checkpoint_sha256', 'dtype') or field in measured_protocol:
            if backend.protocol.get(field) != measured_protocol.get(field) or measured_protocol.get(field) is None:
                raise ValueError('service differs from measured cost hardware protocol: ' + field)
    actors = preflight(backend, out, manifest['budgets'],
                       force_single=args.force_single or (out / 'single_actor_required.json').exists())
    if actors not in (1, 2):
        raise ValueError('preflight must return one or two isolated actors')
    worker_identity = json.loads((out / 'identity.json').read_text()) if (out / 'identity.json').exists() else {}
    protocol = dict(backend.protocol, experiment='E2_mixed_cost', conditions=manifest['conditions'],
                    gpu_uuid=worker_identity.get('gpu_uuid'), gpu_index=worker_identity.get('gpu'),
                    manifest_hash=manifest['manifest_hash'], budgets={key: value['budget_hash'] for key, value in manifest['budgets'].items()},
                    environment_workers=actors, service_mode='serial B1; isolated same-task clients',
                    service_load_and_gate_s=time.perf_counter() - start, server_pid=os.getpid(),
                    schedule=['Full-D10'] + ['Fast-D2'] * 4,
                    jacobian='one Full-only positive weighted VJP; raw/error module residual gates')
    atomic(out / 'protocol.json', protocol)
    policies, group_prompt, resource_failed = {}, None, False
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', args.port))
        server.listen(16)
        print('READY', flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    size = struct.unpack('!I', recv_exact(connection, 4))[0]
                    if size > 128 * 1024 * 1024:
                        raise ValueError('RPC request exceeds limit')
                    request = pickle.loads(recv_exact(connection, size))
                    operation = request.pop('op')
                    client = request.pop('client', 'default')
                    if operation == 'health':
                        response = dict(protocol=protocol)
                    elif resource_failed:
                        raise RuntimeError('single_actor_required: restart with one environment')
                    elif operation == 'reset_all':
                        if any(policy.pending is not None for policy in policies.values()):
                            raise RuntimeError('Unacknowledged action during group reset')
                        for policy in policies.values():
                            policy.selector.clear()
                            policy.reset()
                        policies.clear()
                        gc.collect()
                        torch.cuda.empty_cache()
                        group_prompt = None
                        response = {}
                    else:
                        if client not in policies:
                            if len(policies) >= actors:
                                raise RuntimeError('client count exceeds preflight isolation capacity')
                            policies[client] = Policy(backend, manifest['budgets'])
                        policy = policies[client]
                        if operation == 'configure':
                            response = policy.configure(**request)
                        elif operation == 'predict':
                            if group_prompt is None:
                                group_prompt = request['instruction']
                            if request['instruction'] != group_prompt:
                                raise ValueError('two environments must share the same task/prompt')
                            response = policy.predict(**request)
                        elif operation == 'ack':
                            response = policy.acknowledge(**request)
                        else:
                            raise ValueError(f'unknown RPC operation {operation}')
                    response['ok'] = True
                except torch.cuda.OutOfMemoryError:
                    resource_failed = True
                    atomic(out / 'single_actor_required.json', dict(reason='runtime CUDA OOM; bounded fallback to one actor'))
                    response = dict(ok=False, error=traceback.format_exc())
                except Exception:
                    response = dict(ok=False, error=traceback.format_exc())
                payload = pickle.dumps(response, protocol=5)
                connection.sendall(struct.pack('!I', len(payload)) + payload)


if __name__ == '__main__':
    main()
