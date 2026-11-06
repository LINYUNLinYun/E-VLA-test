"""The original continuous GPU path, with explicit client alignment metadata."""
import argparse
import json
from pathlib import Path
import socket
import traceback

from results.idea2.online_encoder_20261025.wire import receive, send


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi05_libero_pytorch_idea2")
    p.add_argument("--cleanup", required=True)
    args = p.parse_args()
    from preparations.idea2.experiments import load_backend
    from results.idea2.continuous_encoder_20261026.policy import ContinuousPolicy
    from results.idea2.fresh_tail_20261025.adapter import finish_contexts

    class Policy(ContinuousPolicy):
        alignment = "off"

        @property
        def protocol(self):
            return dict(super().protocol, control_alignment=self.alignment,
                        tail_actions_used=self.alignment == "control_steps",
                        action_selection="R5 handoff; optional skip of observation-age prefix; no action padding")

        def configure(self, arm, alignment):
            if alignment not in ("off", "control_steps"):
                raise ValueError("Unknown alignment")
            self.alignment = alignment
            return super().configure(arm)

    backend = load_backend(args.checkpoint, "cuda:0", model_config="pi05_libero", num_steps=10)
    policy = Policy(backend, "sm16_encoder")
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
                    connection.settimeout(180)
                    try:
                        message = receive(connection)
                        op = message.pop("op")
                        if op == "health": result = dict(protocol=policy.protocol)
                        elif op == "configure": result = dict(protocol=policy.configure(**message))
                        elif op == "reset": result = policy.reset()
                        elif op == "drain": result = policy.drain()
                        elif op == "shutdown": result = policy.drain(); running = False
                        elif op == "predict": result = policy.predict(**message)
                        elif op == "predict_ahead":
                            def exchange(request_id):
                                send(connection, dict(ok=True, event="advance", request_id=request_id))
                                value = receive(connection)
                                if not value.get("ok") or value.get("event") != "observation" or value.get("request_id") != request_id:
                                    raise ValueError("Camera exchange crossed request identity")
                                return value
                            result = policy.predict_ahead(exchange=exchange, **message)
                        else: raise ValueError("Unknown operation")
                        send(connection, dict(result, ok=True))
                    except Exception:
                        try: send(connection, dict(ok=False, error=traceback.format_exc()))
                        except OSError: pass
    finally:
        policy.close()
    del policy, backend
    Path(args.cleanup).write_text(json.dumps(finish_contexts(), indent=2)+"\n")


if __name__ == "__main__":
    main()
