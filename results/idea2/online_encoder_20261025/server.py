"""Single scheduling Host; duplex request exchanges a real next observation."""
import argparse
import json
from pathlib import Path
import socket
import traceback

from .wire import send, receive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi05_libero_pytorch_idea2")
    parser.add_argument("--arm", default="sm16_encoder")
    parser.add_argument("--cleanup", required=True)
    args = parser.parse_args()
    from preparations.idea2.experiments import load_backend
    from .policy import OnlinePolicy
    from results.idea2.fresh_tail_20261025.adapter import finish_contexts
    backend = load_backend(args.checkpoint, "cuda:0", model_config="pi05_libero", num_steps=10)
    policy = OnlinePolicy(backend, args.arm)
    try:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", args.port))
            server.listen(4)
            print("READY", flush=True)
            running = True
            while running:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(600)
                    try:
                        request = receive(connection)
                        op = request.pop("op")
                        if op == "health": result = dict(protocol=policy.protocol)
                        elif op == "configure": result = dict(protocol=policy.configure(**request))
                        elif op == "reset": result = policy.reset()
                        elif op == "drain": result = policy.drain()
                        elif op == "shutdown":
                            result = policy.drain()
                            running = False
                        elif op == "predict": result = policy.predict(**request)
                        elif op == "predict_ahead":
                            def exchange(request_id):
                                send(connection, dict(ok=True, event="advance", request_id=request_id))
                                value = receive(connection)
                                if value.get("event") != "observation" or value.get("request_id") != request_id:
                                    raise ValueError("Out-of-order real observation exchange")
                                if not value.get("ok"):
                                    raise RuntimeError("Environment worker failed during its R5 advance")
                                return value
                            result = policy.predict_ahead(exchange=exchange, **request)
                        else: raise ValueError("Unknown online operation")
                        send(connection, dict(result, ok=True))
                    except Exception:
                        try:
                            send(connection, dict(ok=False, error=traceback.format_exc()))
                        except OSError:
                            # An empty readiness connection has no result reader.
                            pass
    finally:
        policy.close()
    del policy, backend
    cleanup = finish_contexts()
    Path(args.cleanup).write_text(json.dumps(cleanup, indent=2)+"\n")


if __name__ == "__main__":
    main()
