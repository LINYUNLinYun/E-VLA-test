"""Serialized trusted-local RPC for fixed zero-shot transfer modes."""
import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback

from preparations.idea1_v3_1.common import atomic_json
from preparations.idea1_v3_1.server import receive
from .transfer_policy import ROOT,TransferPolicy,json_safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--deadline-unix",type=float,required=True)
    parser.add_argument("--legacy-bank",default=str(ROOT/"results/idea1_v3_1/input_bank.pkl"))
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True,**vars(args))))
        return
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Immutable service receipt already exists")
    from .transfer_protocol import load_manifest
    manifest = load_manifest(args.protocol)
    identity = manifest["manifest_hash"]
    if time.time()>=args.deadline_unix:
        raise TimeoutError("Service deadline already elapsed")
    from .residual import require_idle_gpu
    preflight = require_idle_gpu()
    receipt = dict(status="constructing",server_pid=os.getpid(),manifest_hash=identity,
        started_unix=time.time(),deadline_unix=args.deadline_unix,gpu_preflight=preflight,
        category_counters={},legacy_gate_passed=False)
    atomic_json(output,receipt)
    try:
        policy = TransferPolicy(manifest,identity)
        receipt["legacy_gate"] = policy.legacy_preflight(args.legacy_bank)
        receipt["legacy_gate_passed"] = policy.legacy_gate_passed
        receipt["base_prompt_checks"] = policy.backend.base_prompt_checks
        if not policy.legacy_gate_passed:
            raise ArithmeticError("LEGACY_REAL_GPU_PARITY_FAILED")
        gpu_uuid = preflight["device"].split(",")[1].strip()
        policy.protocol.update(server_pid=os.getpid(),gpu_uuid=gpu_uuid,legacy_gate_passed=True)
        receipt.update(status="ready",ready_unix=time.time(),protocol=json_safe(policy.protocol))
        atomic_json(output,receipt)
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(("127.0.0.1",args.port))
            server.listen(8)
            server.settimeout(5)
            print("READY "+json.dumps(json_safe(policy.protocol)),flush=True)
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
                            raise ValueError("Oversized trusted localhost request")
                        request = pickle.loads(receive(connection,size))
                        operation = request.pop("op")
                        tick = time.perf_counter()
                        if operation=="health":
                            result = dict(protocol=policy.protocol,counters=policy.counters)
                        elif operation=="predict":
                            result = policy.predict(**request)
                        elif operation=="reset":
                            result = policy.reset()
                        elif operation=="preflight":
                            result = receipt["legacy_gate"]
                        elif operation=="shutdown":
                            result,shutdown = {},True
                        else:
                            raise ValueError("Unknown operation")
                        result = json_safe(dict(result,ok=True,manifest_hash=identity,protocol_hash=identity,
                            server_dispatch_ms=(time.perf_counter()-tick)*1000))
                    except Exception:
                        result = dict(ok=False,manifest_hash=identity,failure_code="TECHNICAL",error=traceback.format_exc())
                    # All output values are Python primitives/lists, avoiding
                    # NumPy2 pickle module names in Python3.8 CALVIN clients.
                    payload = pickle.dumps(result,protocol=4)
                    connection.sendall(struct.pack("!I",len(payload))+payload)
                receipt["category_counters"] = json_safe(policy.counters)
                atomic_json(output,receipt)
                if shutdown:
                    break
        receipt.update(status="complete",completed_unix=time.time(),category_counters=json_safe(policy.counters))
    except Exception:
        receipt.update(status="failed",completed_unix=time.time(),error=traceback.format_exc())
        raise
    finally:
        atomic_json(output,receipt)


if __name__ == "__main__":
    main()
