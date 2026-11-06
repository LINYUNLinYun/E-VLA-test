"""Guarded E2 localhost policy service, requiring completed real-model parity."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback

from preparations.idea1_v3_1.server import receive


def gate_receipt(path):
    raw = Path(path).read_bytes()
    gate = json.loads(raw)
    if gate.get("status")!="complete" or not gate.get("parity_passed"):
        raise ValueError("Completed native raw+decoded refresh-all parity gate is required")
    for filename,expected in gate["source_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Source differs from real-model gate: {filename}")
    return gate,hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate",required=True)
    parser.add_argument("--mode",choices=("native","reuse03","d2"),required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--deadline-unix",type=float,required=True)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True,**vars(args))))
        return
    gate,identity = gate_receipt(args.gate)
    if time.time()>=args.deadline_unix:
        raise TimeoutError("DEADLINE: experiment budget expired")
    from .residual import ResidualPolicy,require_idle_gpu
    print("GPU_PREFLIGHT "+json.dumps(require_idle_gpu()),flush=True)
    policy = ResidualPolicy(args.mode)
    protocol = dict(manifest_hash=identity,condition_id=args.mode,real_input_parity_passed=True,
        H=policy.model.action_horizon,R=5,D=2 if args.mode=="d2" else 10,
        backend="native JAX",gate=args.gate,checkpoint=gate["checkpoint"]["path"],
        checkpoint_identity_artifacts=gate["checkpoint"]["identity_artifacts"],fresh_prefix=True,request_local_residual=True,
        residual_boundary="pre-final-RMSNorm",approximation="skip scanned Transformer layers at indices other than 0/3" if args.mode=="reuse03" else "none")
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(("127.0.0.1",args.port))
        server.listen(8)
        server.settimeout(5)
        print("READY "+json.dumps(protocol),flush=True)
        while time.time()<args.deadline_unix:
            try:
                connection,_ = server.accept()
            except socket.timeout:
                continue
            shutdown = False
            with connection:
                try:
                    size = struct.unpack("!I",receive(connection,4))[0]
                    if size>128*1024*1024:
                        raise ValueError("Oversized localhost request")
                    request = pickle.loads(receive(connection,size))
                    op = request.pop("op")
                    tick = time.perf_counter()
                    if op=="predict":
                        result = policy.predict(**request)
                    elif op=="health":
                        result = dict(protocol=protocol)
                    elif op=="reset":
                        result = policy.reset()
                    elif op=="shutdown":
                        result,shutdown = {},True
                    else:
                        raise ValueError("Unsupported operation")
                    result.update(ok=True,manifest_hash=identity,condition_id=args.mode,
                                  server_dispatch_ms=(time.perf_counter()-tick)*1000)
                except Exception:
                    result = dict(ok=False,failure_code="TECHNICAL",error=traceback.format_exc())
                payload = pickle.dumps(result,protocol=5)
                connection.sendall(struct.pack("!I",len(payload))+payload)
            if shutdown:
                break


if __name__ == "__main__":
    main()
