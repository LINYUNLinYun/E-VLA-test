"""CPU-only independent B service identity, created BEFORE parent authorization.

No authorization file is created or implied. Authorization later references the
resulting service hash; its own bytes are logged only in the execution receipt,
avoiding a circular service-manifest/authorization hash dependency.
"""
import argparse
import json
from pathlib import Path
import time
from preparations.idea1_v3_2.transfer_protocol import digest,sha,load_manifest
from .transfer_registration import load_registration,SCOPE,MODES

ROOT=Path(__file__).resolve().parents[2]


def build(registration_path):
    registration_path=Path(registration_path).resolve()
    registration=load_registration(registration_path,verify_sources=True)
    reference_path=Path(registration["input_files"]["model_protocol_reference"]["path"])
    reference=load_manifest(reference_path)
    if reference["manifest_hash"]!=registration["policy_reference"]["manifest_hash"]:
        raise ValueError("B weights-only model reference differs registered identity")
    checkpoints=reference["checkpoint_metadata"]
    artifacts=[checkpoints["shared_base"],*checkpoints["draft_heads"].values()]
    for artifact in artifacts:
        if sha(artifact["path"])!=artifact["sha256"]:
            raise ValueError("Registered released B model artifact changed: "+artifact["path"])
    sources=dict(registration["source_files"])
    names=["preparations/idea1_v3_2/transfer_policy.py","preparations/idea1_v3_2/transfer_prompts.py",
        "preparations/idea1_v3/policy.py","preparations/idea1_v2_2/pi0_adapter.py",
        "preparations/idea1_v2_1/runtime_models.py","preparations/idea1_v2_1/runtime_client.py",
        "preparations/idea1_v3_2/transfer_protocol.py","preparations/idea1_v3_1/common.py","preparations/idea1_v3_1/server.py",
        "policies/realtime-vla-flash/scripts/experiment10/backend.py",
        "policies/realtime-vla-flash/scripts/spec/triton/pi0_spec_infer.py",
        "policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py"]
    names += [str(Path(__file__).relative_to(ROOT)),str(Path(__file__).with_name("policy_B_server.py").relative_to(ROOT))]
    sources.update({name:sha(ROOT/name) for name in names})
    receipts={str(registration_path):sha(registration_path),str(reference_path.resolve()):sha(reference_path)}
    receipts.update({str(Path(item["path"]).resolve()):item["sha256"] for item in registration["input_files"].values()})
    manifest=dict(schema=1,status="registered_CPU_only_not_authorized_or_executed",scope=SCOPE,family="libero_transfer",
        registration_hash=registration["registration_hash"],registration_path=str(registration_path),
        model_reference_manifest_hash=reference["manifest_hash"],model_reference_scope="Released weights and implementation lineage only; no Plus390 scientific permission/results",
        checkpoint_metadata=checkpoints,prompt_proof=reference["prompt_proof"],
        modes=list(MODES),H=50,R=5,denoising=registration["denoising"],case_count=300,condition_count=1800,
        source_files=sources,receipt_files=receipts,created_unix=time.time(),
        allocation="Separate B model service process only after explicit parent physicalGPU authorization; no upstream Plus proxy",
        counter_namespace="corrected_B_transfer:"+registration["registration_hash"],
        request_gate="Everypredict must carry exactregistration_hash+registeredcase_id/environment; no native Plus observation route",
        authorization_required_fields=["scope","allow_quality","registration_hash","service_manifest_hash",
            "model_reference_manifest_hash","gpu","gpu_uuid","modes","environments","deadline_unix"],
        authorization_manifest_order="Freeze this service hash first; parent separately authorizes it; execution receipt pins authorization bytes",
        quality_launch_authorized=False,model_constructed=False,gpu_used=False)
    manifest["manifest_hash"]=digest(manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--registration",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    result=build(args.registration);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),manifest_hash=result["manifest_hash"],registration_hash=result["registration_hash"],
        gpu_used=False,quality_launch_authorized=False)))


if __name__=="__main__":main()
