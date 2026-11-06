"""Native LIBERO-Plus six-route service preparation; never the cross-robot bridge."""
import argparse
import json
from pathlib import Path
import sys
import time

from preparations.idea1_v3_2.transfer_policy import cpu_preflight,MODES
from preparations.idea1_v3_2.transfer_protocol import digest,sha
from .policy_native import CorrectedTransferPolicy

ROOT=Path(__file__).resolve().parents[2]
HEADS={"libero_goal":"draft_goal","libero_object":"draft_object",
    "libero_spatial":"draft_spatial","libero_10":"draft_10"}


class NativeLiberoPlusPolicy(CorrectedTransferPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.inner.protocol.update(control_family="Released LIBERO base+four released mother-suite draft heads; native LIBERO/Plus inputs",
            state_convention="Actual simulator original quaternion -> unchanged native noncanonical quat_axisangle; no cross-robot pose/lift bridge",
            image_convention="Original simulator RGB -> exact legacy native_observation180-degree flips server-side, once",
            language="Actual request instruction native SP/full BF16 lookup, no nearest-source substitution",
            head_selection="Explicit matching_draft from mother_suite; Full/V14 share same base",
            action_convention="Unchanged released LIBERO normalization+extraDelta/AbsoluteActions and native7 controller",
            H=50,R=5,D=10)


def prepare():
    result=cpu_preflight()
    additional=[Path(__file__),Path(__file__).with_name("policy_native.py"),Path(__file__).with_name("policy_server.py"),
        ROOT/"preparations/idea1_v2_1/runtime_models.py",ROOT/"preparations/idea1_v2_1/runtime_client.py",
        ROOT/"preparations/idea1_v3_2/transfer_protocol.py",ROOT/"preparations/idea1_v3_1/common.py",
        ROOT/"preparations/idea1_v3_1/server.py"]
    result["source_sha256"].update({str(path):sha(path) for path in additional})
    result.update(kind="native_LIBERO_Plus_CPU_service_preparation",family="libero_transfer",mother_suite_heads=HEADS,
        native_input_contract="original simulator observation dictionary, not preflipped images or cross-embodiment state8 bridge",
        condition_rule="Each registered instance has Full/V14/its own mother-suite draft, not all4 drafts perinstance",
        launch_blockers=["Completed assigned-device native renderer/environment preflight receipt",
            "New frozen actual Plus/original-baseline selection manifest and new deadline",
            "Explicit idle physical GPU grant and same-input6route native legacy gate on startup"],
        model_constructed=False,gpu_used=False)
    return result


def freeze(assets_path,environment_receipt,selection_path,deadline_unix,extra_sources=()):
    assets=json.loads(Path(assets_path).read_text())
    if assets.get("status")!="complete" or assets.get("gpu_used"):
        raise ValueError("Completed CPU-only native assets preparation required")
    for path,expected in assets["source_sha256"].items():
        if sha(path)!=expected: raise ValueError("Prepared service source changed: "+path)
    environment=json.loads(Path(environment_receipt).read_text())
    if environment.get("status")!="complete" or environment.get("native_render_passed") is not True:
        raise ValueError("Completed actual native renderer receipt required; import-only receipt is insufficient")
    selection=json.loads(Path(selection_path).read_text())
    if selection.get("status")!="complete": raise ValueError("Completed actual scientific task selection required")
    cases=selection["selected"]+selection["original_baselines"]
    if not cases or deadline_unix<=time.time(): raise ValueError("Nonempty selection and fresh deadline required")
    for row in cases:
        if HEADS.get(row["mother_suite"])!=row["matching_draft"]:
            raise ValueError("Scientific selection changed mother-suite draft mapping")
    # Fixed round-robin original controls: first native task from every mother
    # suite comes before second tasks. This stage is reusable in the full40.
    groups={}
    for index,row in enumerate(selection["original_baselines"]):
        groups.setdefault(row["mother_suite"],[]).append(index)
    original_order=[indices[level] for level in range(max(map(len,groups.values())))
        for indices in groups.values() if level<len(indices)]
    sources={str(Path(path).relative_to(ROOT)):value for path,value in assets["source_sha256"].items()}
    for path,expected in environment.get("source_sha256",environment.get("source_files",{})).items():
        resolved=Path(path) if Path(path).is_absolute() else ROOT/path
        if sha(resolved)!=expected: raise ValueError("Native environment source changed: "+path)
        sources[str(resolved)]=expected
    for path in extra_sources:
        resolved=Path(path).resolve()
        if resolved.suffix!=".py" or not resolved.is_file(): raise ValueError("Extra frozen source must be existing Python code")
        sources[str(resolved)]=sha(resolved)
    manifest=dict(schema=1,kind="native_LIBERO_Plus_and_original_controls",family="libero_transfer",
        modes=list(MODES),H=50,R=5,D=10,deadline_unix=deadline_unix,
        mother_suite_heads=HEADS,case_count=len(cases),condition_count=sum(3 for row in cases),
        condition_rule="FullD10,V14D10,exactly matching mother-suite released draft per instance",
        original_execution_order=original_order,initial_original_case_indices=original_order[:len(groups)],
        staged_execution="First original baseline per mother suite (12 conditions); remaining originals/Plus require parent review, fixed full selection unchanged",
        case_selection=selection,checkpoint_metadata=assets["checkpoint_metadata"],prompt_proof=assets["prompt_proof"],
        input_contract=assets["native_input_contract"],
        source_files=sources,
        receipt_files={str(Path(path).resolve()):sha(path) for path in (assets_path,environment_receipt,selection_path)},
        created_unix=time.time(),startup_gate="six real legacy native LIBERO input/noise routes strict1e-5, raw+decoded; no quality beforegate",
        noise_protocol="request seed fixed bycase/replan, excludesmode; draft does not use noise",
        limitations="LIBERO-Plus robustness, not multi-embodiment zero-shot transfer or native target-checkpoint positive control")
    manifest["manifest_hash"]=digest(manifest)
    return manifest


def main():
    if "--serve" in sys.argv:
        # New entrypoint only; leaves the frozen service source unchanged.
        from . import policy_server
        policy_server.CorrectedTransferPolicy=NativeLiberoPlusPolicy
        sys.argv.remove("--serve")
        policy_server.main();return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--freeze",action="store_true")
    p.add_argument("--assets")
    p.add_argument("--environment-receipt")
    p.add_argument("--selection")
    p.add_argument("--deadline-unix",type=float)
    p.add_argument("--extra-source",action="append",default=[])
    args=p.parse_args();output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    if args.freeze:
        if not all((args.assets,args.environment_receipt,args.selection,args.deadline_unix)):
            raise ValueError("Freeze requires explicit new assets/environment/selection/deadline")
        result=freeze(args.assets,args.environment_receipt,args.selection,args.deadline_unix,args.extra_source)
    else: result=prepare()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),status=result.get("status"),manifest_hash=result.get("manifest_hash"))))


if __name__=="__main__": main()
