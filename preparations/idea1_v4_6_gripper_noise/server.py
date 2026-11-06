"""One eager V12+D1 service; Full is only a residual observation pass."""
import argparse
import os
import pickle
import socket
import struct
import time
import traceback
from pathlib import Path

from preparations.idea1_v2_1.runtime_client import recv_exact
from preparations.idea1_v3.io import atomic
from .policy import Policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    os.environ.update(SPEC_TRITON_DRAFT_GRAPH='0', SPEC_TRITON_VERIFY_GRAPH='0',
                      SPEC_TRITON_VERIFY_SEMANTICS_GRAPH='0')
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from preparations.idea1_v3_4_v12_compensation.backend import Backend
    from .preflight import run
    backend = Backend()
    run(backend, out)

    def forbidden(*args, **kwargs):
        raise AssertionError('Full/legacy AE and verification are forbidden in the gripper noise repair')

    backend.sample = forbidden
    backend.verify = forbidden
    policy = Policy(backend)
    protocol = dict(backend.protocol,
                    experiment='v12_d1_kalman6_gripper_noise_no_fallback',
                    server_pid=os.getpid(), load_and_preflight_s=time.perf_counter()-start,
                    full_action_expert_calls=0, time_embedding='openpi_linspace',
                    encoder_graph=False, ae_graph=False, verifier=False)
    atomic(out/'protocol.json', protocol)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', args.port))
        server.listen(8)
        print('READY', flush=True)
        while True:
            connection, _ = server.accept()
            stopping = False
            with connection:
                try:
                    request = pickle.loads(recv_exact(connection, struct.unpack('!I', recv_exact(connection, 4))[0]))
                    op = request.pop('op')
                    if op == 'health':
                        response = dict(protocol=protocol)
                    elif op == 'configure':
                        response = policy.configure(**request)
                    elif op == 'predict':
                        response = policy.predict(**request)
                    elif op == 'ack':
                        response = policy.acknowledge(**request)
                    elif op == 'stop':
                        response, stopping = {}, True
                    else:
                        raise ValueError(op)
                    response['ok'] = True
                except Exception:
                    response = dict(ok=False, error=traceback.format_exc())
                data = pickle.dumps(response, protocol=5)
                connection.sendall(struct.pack('!I', len(data))+data)
            if stopping:
                return


if __name__ == '__main__':
    main()
