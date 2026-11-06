"""Independent registered corrected-B service; never a Plus permission proxy.

CPU contract and launch preparation only until explicit B authorization + GPU
grant. Reuses released weights/backend implementation, not Plus case identity,
model process, counters, or scientific permission. No import-time construction.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback

from preparations.idea1_v3_2.transfer_protocol import digest,sha,load_manifest
from preparations.idea1_v3_2.transfer_policy import MODES,TransferPolicy,json_safe
from preparations.idea1_v3_1.common import atomic_json
from preparations.idea1_v3_1.server import receive
from .transfer_registration import check_registration,load_registration,SCOPE


def validate_binding(registration,authorization,service_manifest,now=None):
    """A valid Plus/model reference alone never grants B quality permission."""
    check_registration(registration)
    now=time.time() if now is None else now
    model_reference=registration["policy_reference"]["manifest_hash"]
    expected=dict(scope=SCOPE,allow_quality=True,registration_hash=registration["registration_hash"],
        service_manifest_hash=service_manifest["manifest_hash"],model_reference_manifest_hash=model_reference)
    for key,value in expected.items():
        if authorization.get(key)!=value:raise PermissionError("Explicit corrected-B authorization mismatch: "+key)
    if service_manifest.get("scope")!=SCOPE or service_manifest.get("registration_hash")!=registration["registration_hash"]:
        raise PermissionError("Independent B service manifest required; Plus manifest is not B permission")
    if service_manifest.get("model_reference_manifest_hash")!=model_reference:
        raise PermissionError("Registered source model reference changed")
    if tuple(authorization.get("modes",()))!=tuple(MODES):raise PermissionError("All six B modes must be explicitly authorized")
    if set(authorization.get("environments",()))!={"calvin","simpler","pusht"}:
        raise PermissionError("B environments must be explicitly authorized; no Plus/native substitution")
    if not authorization.get("gpu_uuid","").startswith("GPU-"):
        raise PermissionError("Explicit physical B GPU UUID authorization required")
    if not isinstance(authorization.get("gpu"),int) or authorization["gpu"] not in (0,1,2,3):
        raise PermissionError("Explicit physical B GPU index required for native renderer binding")
    if float(authorization.get("deadline_unix",0))<=now:raise PermissionError("B authorization expired")
    return dict(scope=SCOPE,registration_hash=registration["registration_hash"],
        service_manifest_hash=service_manifest["manifest_hash"],model_reference_manifest_hash=model_reference,
        gpu_uuid=authorization["gpu_uuid"],deadline_unix=authorization["deadline_unix"])


class BRegisteredPolicy:
    def __init__(self,registration,authorization,service_manifest,*,backend=None,launch=False):
        binding=validate_binding(registration,authorization,service_manifest)
        if backend is None and not launch:raise RuntimeError("Explicit B model launch grant required")
        self.registration=registration;self.authorization=authorization;self.binding=binding
        self.case_lookup={case["case_id"]:case for case in registration["cases"]}
        self.namespace=SCOPE+":"+registration["registration_hash"]
        self.inner=TransferPolicy(service_manifest,service_manifest["manifest_hash"],backend=backend)
        self.protocol=dict(self.inner.protocol,family="libero_transfer",scope=SCOPE,
            authorized_scientific_registrations=[registration["registration_hash"]],
            model_reference_manifest_hash=binding["model_reference_manifest_hash"],
            counter_namespace=self.namespace,registration_hash=registration["registration_hash"],
            quality_scope_authorized=True,
            state="B client-owned corrected continuous native-quaternion lift; no server LIBERO flips",
            permission="Explicit independent corrected-B registration+parent authorization; Plus390 reference is weights-only",
            backend_reuse="Released backend implementation/weights only; separate process/counters, no Plus upstream RPC")

    def legacy_preflight(self,bank):
        result=self.inner.legacy_preflight(bank)
        self.protocol["legacy_gate_passed"]=self.inner.legacy_gate_passed
        return result

    def reset(self):return self.inner.reset()

    def predict(self,*,registration_hash,case_id,environment,mode,raw,instruction,seed,category="quality"):
        validate_binding(self.registration,self.authorization,
            dict(manifest_hash=self.binding["service_manifest_hash"],scope=SCOPE,
                registration_hash=self.binding["registration_hash"],model_reference_manifest_hash=self.binding["model_reference_manifest_hash"]))
        if registration_hash!=self.registration["registration_hash"]:
            raise PermissionError("Request scientific B registration mismatch")
        case=self.case_lookup.get(case_id)
        if case is None or case["environment"]!=environment:
            raise PermissionError("Unregistered B case/environment; Plus/native quality forbidden")
        if mode not in MODES:raise ValueError("Only frozen six source modes allowed")
        if category not in ("quality","warmup","preflight"):raise ValueError("Unknown B counter category")
        result=self.inner.predict(mode,raw,instruction,seed,category=category)
        return dict(result,registration_hash=registration_hash,counter_namespace=self.namespace,
            counters={self.namespace:result["counters"]},
            attribution=dict(scope=SCOPE,registration_hash=registration_hash,case_id=case_id,
                case_hash=digest(case),environment=environment,mode=mode,
                service_manifest_hash=self.binding["service_manifest_hash"],
                model_reference_manifest_hash=self.binding["model_reference_manifest_hash"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True);p.add_argument("--authorization",required=True)
    p.add_argument("--protocol",required=True);p.add_argument("--output",required=True)
    p.add_argument("--port",type=int,default=19145);p.add_argument("--deadline-unix",type=float,required=True)
    p.add_argument("--legacy-bank",default="results/idea1_v3_1/input_bank.pkl");p.add_argument("--launch",action="store_true")
    args=p.parse_args();registration=load_registration(args.registration,verify_sources=True)
    authorization=json.loads(Path(args.authorization).read_text());manifest=load_manifest(args.protocol)
    binding=validate_binding(registration,authorization,manifest)
    if args.deadline_unix>binding["deadline_unix"]:raise PermissionError("B execution cannot exceed explicit authorization deadline")
    if not args.launch:
        print(json.dumps(dict(cpu_contract_valid=True,model_constructed=False,gpu_used=False,binding=binding)));return
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    from preparations.idea1_v3_2.residual import require_idle_gpu
    gpu=require_idle_gpu();actual_uuid=gpu["device"].split(",")[1].strip()
    if actual_uuid!=binding["gpu_uuid"] or int(gpu["device"].split(",")[0])!=authorization["gpu"]:
        raise PermissionError("Assigned B physical GPU index/UUID differs parent authorization")
    receipt=dict(status="constructing",server_pid=os.getpid(),binding=binding,gpu_preflight=gpu,
        registration_path=args.registration,registration_sha256=sha(args.registration),
        authorization_path=args.authorization,authorization_sha256=sha(args.authorization),
        source_sha256={str(Path(__file__)):sha(__file__)},started_unix=time.time(),deadline_unix=args.deadline_unix)
    atomic_json(output,receipt)
    try:
        policy=BRegisteredPolicy(registration,authorization,manifest,launch=True)
        # Released backend initialization loads CPU assets only. Recheck after
        # that work, immediately before legacy_preflight creates CUDA runtime.
        receipt["final_pre_runtime_allocation_gpu_preflight"]=require_idle_gpu()
        final_uuid=receipt["final_pre_runtime_allocation_gpu_preflight"]["device"].split(",")[1].strip()
        if final_uuid!=binding["gpu_uuid"]:raise PermissionError("B final allocation UUID differs authorization")
        receipt["legacy_gate"]=policy.legacy_preflight(args.legacy_bank)
        if not receipt["legacy_gate"]["passed"]:raise RuntimeError("B source native six-route raw/decoded legacy parity failed")
        policy.protocol.update(server_pid=os.getpid(),gpu_uuid=actual_uuid,legacy_gate_passed=True)
        receipt.update(status="ready",protocol=policy.protocol,ready_unix=time.time());atomic_json(output,json_safe(receipt))
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(("127.0.0.1",args.port));server.listen(8);server.settimeout(3)
            print("READY corrected_B_transfer "+registration["registration_hash"],flush=True)
            while time.time()<args.deadline_unix:
                try:connection,_=server.accept()
                except socket.timeout:continue
                shutdown=False
                with connection:
                    try:
                        size=struct.unpack("!I",receive(connection,4))[0]
                        if size>128*1024*1024:raise ValueError("Oversized trusted localhost request")
                        request=pickle.loads(receive(connection,size));operation=request.pop("op")
                        if operation=="health":result=dict(protocol=policy.protocol,authorized_scientific_registrations=policy.protocol["authorized_scientific_registrations"])
                        elif operation=="predict":
                            if request.pop("family","libero_transfer")!="libero_transfer":raise PermissionError("B source family mismatch")
                            if "observation" in request:raise PermissionError("B raw client-oriented inputs only, not native Plus observations")
                            result=policy.predict(**request)
                        elif operation=="reset":
                            if request.get("registration_hash")!=registration["registration_hash"]:
                                raise PermissionError("B reset registration mismatch")
                            result=policy.reset()
                        elif operation=="shutdown":result,shutdown={},True
                        else:raise ValueError("Unknown B operation")
                        result=json_safe(dict(result,ok=True,manifest_hash=manifest["manifest_hash"],protocol_hash=manifest["manifest_hash"]))
                    except Exception:result=dict(ok=False,failure_code="B_SCOPE_OR_TECHNICAL",error=traceback.format_exc())
                    payload=pickle.dumps(result,protocol=4);connection.sendall(struct.pack("!I",len(payload))+payload)
                receipt["counter_namespace"]=policy.namespace;receipt["counters"]={policy.namespace:json_safe(policy.inner.counters)}
                atomic_json(output,json_safe(receipt))
                if shutdown:break
        receipt.update(status="complete",completed_unix=time.time())
    except Exception:receipt.update(status="failed",error=traceback.format_exc(),completed_unix=time.time());raise
    finally:atomic_json(output,json_safe(receipt))


if __name__=="__main__":main()
