"""One frozen condition per service, trusted localhost RPC, no implicit launch."""
import argparse
import json
import os
import pickle
import socket
import struct
import time
import traceback

from .common import atomic_json, read_manifest


def receive(connection, length):
    blocks = bytearray()
    while len(blocks) < length:
        piece = connection.recv(length-len(blocks))
        if not piece:
            raise ConnectionError("Incomplete localhost RPC")
        blocks.extend(piece)
    return bytes(blocks)


def rpc(port, request):
    payload = pickle.dumps(request,protocol=5)
    with socket.create_connection(("127.0.0.1",port),timeout=1800) as connection:
        connection.sendall(struct.pack("!I",len(payload))+payload)
        size = struct.unpack("!I",receive(connection,4))[0]
        result = pickle.loads(receive(connection,size))
    if not result.get("ok"):
        raise RuntimeError(result.get("error",result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--condition",required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--protocol-output")
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest,verify_sources=args.launch)
    condition = next(c for c in manifest["conditions"]+manifest["benchmark_references"] if c["id"]==args.condition)
    if not args.launch:
        print(json.dumps(dict(dry_run=True,condition=condition,port=args.port)))
        return
    from .models import load
    policy = load(condition)
    protocol = dict(policy.protocol,manifest_hash=manifest["manifest_hash"],condition_id=condition["id"],
                    server_pid=os.getpid(),
                    real_input_parity_passed=False,
                    concurrency="single model request at a time; multiple rollout clients overlap environment work")
    if args.protocol_output:
        atomic_json(args.protocol_output,protocol)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(("127.0.0.1",args.port))
        server.listen(8)
        print("READY",flush=True)
        while True:
            connection,_ = server.accept()
            with connection:
                shutdown = False
                try:
                    size = struct.unpack("!I",receive(connection,4))[0]
                    if size > 128*1024*1024:
                        raise ValueError("Oversized trusted-local request")
                    request = pickle.loads(receive(connection,size))
                    operation = request.pop("op")
                    begin = time.perf_counter()
                    if operation == "health":
                        result = dict(protocol=protocol)
                    elif operation == "predict":
                        result = policy.predict(**request)
                    elif operation == "vision":
                        result = policy.vision_benchmark(**request)
                    elif operation == "parity":
                        result = policy.parity(**request)
                        protocol["real_input_parity_passed"] = bool(result.get("passed") and result.get("noise_paired",True))
                        if args.protocol_output:
                            atomic_json(args.protocol_output,protocol)
                    elif operation == "reset":
                        result = policy.reset()
                    elif operation == "shutdown":
                        policy.reset()
                        result,shutdown = {},True
                    else:
                        raise ValueError("Unknown operation")
                    result.update(ok=True,server_dispatch_ms=(time.perf_counter()-begin)*1000,
                                  manifest_hash=manifest["manifest_hash"],condition_id=condition["id"])
                except Exception:
                    result = dict(ok=False,error=traceback.format_exc())
                payload = pickle.dumps(result,protocol=5)
                connection.sendall(struct.pack("!I",len(payload))+payload)
                if shutdown:
                    return


if __name__ == "__main__":
    main()
