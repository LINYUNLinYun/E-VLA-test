"""Localhost RPC server hosting one static pi0.5 vision condition."""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import traceback

from preparations.idea1_v2_1.runtime_client import recv_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config)
    if not args.launch:
        print(json.dumps({"dry_run": True, "config": config}))
        return

    from experiments.pi05_vision_ab.policy_adapter import Pi05VisionPolicy

    policy = Pi05VisionPolicy(
        mode=config["mode"],
        keep_tokens=int(config["keep_tokens"]),
        condition_id=config["condition_id"],
    )
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", args.port))
        server.listen(8)
        print("READY", flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                op = None
                try:
                    size = struct.unpack("!I", recv_exact(connection, 4))[0]
                    if size > 128 * 1024 * 1024:
                        raise ValueError("oversized request")
                    request = pickle.loads(recv_exact(connection, size))
                    op = request.pop("op")
                    if op == "health":
                        result = {"protocol": policy.protocol}
                    elif op == "reset":
                        result = policy.reset()
                    elif op == "predict":
                        result = policy.predict(**request)
                    elif op == "shutdown":
                        result = {"ok": True}
                    else:
                        raise ValueError(f"unknown operation {op!r}")
                    result["ok"] = True
                except Exception:
                    result = {"ok": False, "error": traceback.format_exc()}
                payload = pickle.dumps(result, protocol=5)
                connection.sendall(struct.pack("!I", len(payload)) + payload)
                if op == "shutdown":
                    return


if __name__ == "__main__":
    main()
