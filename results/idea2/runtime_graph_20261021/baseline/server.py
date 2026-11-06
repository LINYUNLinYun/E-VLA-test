"""Trusted localhost pickle RPC, compatible with existing LIBERO episode client."""
import argparse
import pickle
import socket
import struct
import traceback

from .experiments import load_backend
from .policy import PipelinePolicy
from preparations.idea1_v2_1.runtime_client import recv_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=9012)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--submission-order", choices=("ae_first", "prefill_first"), default="prefill_first")
    parser.add_argument("--host-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    policy = PipelinePolicy(load_backend(args.checkpoint, args.device), submission_order=args.submission_order,
                            host_workers=args.host_workers, diagnostics=args.diagnostics)
    try:
        serve(policy, args.port)
    finally:
        policy.close()


def serve(policy, port):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(8)
        print("READY", flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    size = struct.unpack("!I", recv_exact(connection, 4))[0]
                    if size > 128 * 1024 * 1024:
                        raise ValueError("Oversized trusted-local request")
                    request = pickle.loads(recv_exact(connection, size))
                    operation = request.pop("op")
                    if operation == "health":
                        result = dict(protocol=policy.protocol)
                    elif operation == "configure":
                        result = dict(protocol=policy.configure(**request))
                    elif operation in ("reset", "drain"):
                        result = policy.reset()
                    elif operation == "shutdown":
                        result = policy.reset()
                        result["shutdown"] = True
                    elif operation == "predict":
                        result = policy.predict(**request)
                    else:
                        raise ValueError("Unknown operation")
                    result["ok"] = True
                except Exception:
                    result = dict(ok=False, error=traceback.format_exc())
                payload = pickle.dumps(result, protocol=5)
                connection.sendall(struct.pack("!I", len(payload)) + payload)
                if result.get("shutdown"):
                    return


if __name__ == "__main__":
    main()
