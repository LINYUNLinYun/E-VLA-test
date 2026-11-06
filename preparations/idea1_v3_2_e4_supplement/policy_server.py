"""Serialized localhost supplement service; distinct native and transfer families."""
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
from preparations.idea1_v3_2.transfer_protocol import load_manifest
from preparations.idea1_v3_2.transfer_policy import json_safe
from .policy_native import CorrectedTransferPolicy, NativeCalvinPolicy


def native_libero_raw(observation):
    """Actual native LIBERO/Plus only: exact old image flips and native quaternion state."""
    import numpy as np
    from preparations.idea1_v2_1.runtime_models import native_observation
    images,state = native_observation(observation)
    return dict(base_rgb=np.asarray(images[0]),wrist_rgb=np.asarray(images[1]),state8=np.asarray(state,np.float32))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family",required=True,choices=("libero_transfer","calvin_abc_d_native"))
    p.add_argument("--protocol",required=True)
    p.add_argument("--port",type=int,required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--deadline-unix",type=float,required=True)
    p.add_argument("--legacy-bank",default="results/idea1_v3_1/input_bank.pkl")
    p.add_argument("--checkpoint")
    p.add_argument("--norm-stats")
    p.add_argument("--launch",action="store_true")
    args=p.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True,**vars(args)))); return
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    manifest=load_manifest(args.protocol)
    identity=manifest["manifest_hash"]
    if time.time()>=args.deadline_unix: raise TimeoutError("Deadline elapsed")
    # Immediate physical UUID idle check before either constructor. This process
    # never assumes a container PID equals the nvidia-smi host PID.
    from preparations.idea1_v3_2.residual import require_idle_gpu
    gpu=require_idle_gpu()
    receipt=dict(status="constructing",family=args.family,manifest_hash=identity,server_pid=os.getpid(),
        gpu_preflight=gpu,started_unix=time.time(),deadline_unix=args.deadline_unix)
    atomic_json(output,receipt)
    try:
        if args.family=="libero_transfer":
            policy=CorrectedTransferPolicy(manifest,identity)
            gate=policy.legacy_preflight(args.legacy_bank)
            if not gate["passed"]: raise RuntimeError("LEGACY_NATIVE_REAL_GPU_PARITY_FAILED")
        else:
            if not args.checkpoint or not args.norm_stats: raise ValueError("Native pinned assets required")
            policy=NativeCalvinPolicy(args.checkpoint,args.norm_stats,launch=True)
            # Native gate must be supplied as a registered captured environment
            # input; no arbitrary black-image warmup is claimed as policy parity.
            gate=dict(passed=False,status="native_input_preflight_required",strict_checkpoint_keys_loaded=True)
        uuid=gpu["device"].split(",")[1].strip()
        policy.protocol.update(server_pid=os.getpid(),gpu_uuid=uuid,manifest_hash=identity,
            family=args.family,legacy_gate_passed=bool(gate["passed"]))
        receipt.update(status="ready",ready_unix=time.time(),gate=gate,protocol=policy.protocol)
        atomic_json(output,json_safe(receipt))
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(("127.0.0.1",args.port));server.listen(8);server.settimeout(3)
            print("READY "+json.dumps(json_safe(policy.protocol)),flush=True)
            while time.time()<args.deadline_unix:
                try: connection,_=server.accept()
                except socket.timeout: continue
                shutdown=False
                with connection:
                    try:
                        size=struct.unpack("!I",receive(connection,4))[0]
                        if size>128*1024*1024: raise ValueError("Oversized localhost RPC")
                        request=pickle.loads(receive(connection,size));op=request.pop("op")
                        if op=="health": result=dict(protocol=policy.protocol)
                        elif op=="shutdown": result,shutdown={},True
                        elif op=="reset":
                            result=policy.reset() if args.family=="libero_transfer" else dict(request_local=True)
                        elif op=="preflight" and args.family!="libero_transfer":
                            result=policy.preflight(**request)
                            receipt["gate"]=result
                        elif op=="predict":
                            family=request.pop("family",args.family)
                            if family!=args.family: raise ValueError("Cross-checkpoint family substitution forbidden")
                            environment=request.pop("environment",None)
                            mother_suite=request.pop("mother_suite",None)
                            observation=request.pop("observation",None)
                            if environment in ("libero","libero_plus"):
                                if observation is None:
                                    raise ValueError("Native LIBERO/Plus requires original observation, not transfer bridge raw")
                                draft_for_suite={"libero_goal":"draft_goal","libero_object":"draft_object",
                                    "libero_spatial":"draft_spatial","libero_10":"draft_10"}
                                if request.get("mode","").startswith("draft_") and draft_for_suite.get(mother_suite)!=request["mode"]:
                                    raise ValueError("Draft must use exact released mother-suite head")
                            if observation is not None:
                                if environment not in ("libero","libero_plus") or args.family!="libero_transfer":
                                    raise ValueError("Native observation route is LIBERO/Plus only")
                                if "raw" in request: raise ValueError("Ambiguous raw plus native observation")
                                request["raw"]=native_libero_raw(observation)
                            result=policy.predict(**request)
                        else: raise ValueError("Unknown operation")
                        result=json_safe(dict(result,ok=True,manifest_hash=identity,protocol_hash=identity))
                    except Exception:
                        result=dict(ok=False,failure_code="TECHNICAL",error=traceback.format_exc(),manifest_hash=identity)
                    payload=pickle.dumps(result,protocol=4)
                    connection.sendall(struct.pack("!I",len(payload))+payload)
                receipt["category_counters"]=json_safe(policy.counters)
                atomic_json(output,json_safe(receipt))
                if shutdown: break
        receipt.update(status="complete",completed_unix=time.time())
    except Exception:
        receipt.update(status="failed",error=traceback.format_exc(),completed_unix=time.time());raise
    finally: atomic_json(output,json_safe(receipt))


if __name__=="__main__": main()
