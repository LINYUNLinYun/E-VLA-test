"""One persistent native service per idle GPU; only selector uses a graph."""
import argparse
import pickle
import socket
import struct
import traceback
from pathlib import Path
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact
from preparations.idea1_v3_4_selection_async.runtime import setup
from .policy import Policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    backend = setup(output)
    policy = Policy(backend)
    protocol = dict(backend.protocol, experiment='idea1_v4_1_warmstart', selector_cuda_graph=True,
        native_inference_graph=False, verify_graph=False, verification_times=[.10,.05],
        projection_seed=340610, tau=.10, environment_workers=1)
    atomic(output/'protocol.json', protocol)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', args.port))
        server.listen(8)
        print('READY', flush=True)
        while True:
            connection, _ = server.accept()
            stop = False
            with connection:
                try:
                    request = pickle.loads(recv_exact(connection, struct.unpack('!I', recv_exact(connection,4))[0]))
                    op = request.pop('op')
                    if op == 'health':
                        result = dict(protocol=protocol)
                    elif op == 'preflight':
                        from .preflight import run
                        result = run(backend, policy, output/'preflight')
                    elif op == 'replay_case':
                        from .replay import run_case
                        result = run_case(backend, **request)
                    elif op == 'configure':
                        result = policy.configure(**request)
                    elif op == 'predict':
                        result = policy.predict(**request)
                    elif op == 'ack':
                        result = policy.acknowledge(**request)
                    elif op == 'finish':
                        result = policy.finish()
                    elif op == 'reset_all':
                        policy.clear()
                        result = {}
                    elif op == 'stop':
                        policy.clear()
                        result = {}
                        stop = True
                    else:
                        raise ValueError(op)
                    result['ok'] = True
                except Exception:
                    result = dict(ok=False, error=traceback.format_exc())
                payload = pickle.dumps(result, protocol=5)
                connection.sendall(struct.pack('!I', len(payload))+payload)
            if stop:
                return


if __name__ == '__main__':
    main()
