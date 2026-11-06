"""Post-hoc CPU native pilot decoded-vs-executed audit; no causal attribution."""
import argparse
import json
from pathlib import Path
import numpy as np
from preparations.idea1_v3_2.transfer_protocol import sha

ROOT=Path(__file__).resolve().parents[2]


def audit(service_path,episodes):
    service=json.loads(Path(service_path).read_text())
    receipts={str(Path(service_path)):sha(service_path),str(Path(__file__)):sha(__file__)}
    result=dict(status="complete",cpu_only=True,service_status=service["status"],
        manifest_hash=service["manifest_hash"],modes={},episodes=[],source_input_sha256=receipts,
        limitations=["Original pilot client clipped first6 commands; retain as clipped-protocol pilot, not exact native evaluator.",
            "Decoded output outside unit range is not itself an error: official CALVIN eval does not clip first6.",
            "Gripper values outside[-1,1] are thresholded, not clipped; exclude them from arm clipping rates.",
            "These counts cannot isolate causal effects of V14 skipping, client clipping, or state semantics."])
    for mode,count in service["category_counters"]["quality"].items():
        samples=count["output_elements"]
        outside=np.array(count["output_outside_unit_by_axis"])
        result["modes"][mode]=dict(requests=count["requests"],decoded_action_rows=samples,
            decoded_outside_unit_count_by_axis=outside.tolist(),decoded_outside_unit_fraction_by_axis=(outside/samples).tolist(),
            decoded_first6_outside_unit_element_fraction=float(outside[:6].sum()/(samples*6)),
            observed_input_gripper_positive=count["state_gripper_positive"],observed_input_gripper_negative=count["state_gripper_negative"],
            executed_rows=0,executed_arm_changed_rows=0,executed_arm_changed_elements=0,all_arm_changes_match_clip=True)
    for path in sorted(Path(episodes).glob("seq*/*/telemetry.jsonl")):
        mode=path.parent.name
        if mode not in result["modes"]: continue
        infer=path.with_name("infer.jsonl")
        rows=[json.loads(line) for line in infer.open()]
        by_task={}
        for row in rows: by_task.setdefault(row["subtask_index"],[]).append(row)
        executed,changed_rows,changed_elements,matches=0,0,0,True
        for line in path.open():
            telemetry=json.loads(line);command=telemetry.get("executed_command")
            if command is None: continue
            step=telemetry["subtask_step"]-1
            candidates=[r for r in by_task[telemetry["subtask_index"]] if r["subtask_step"]<=step]
            row=max(candidates,key=lambda r:r["subtask_step"])
            raw=np.asarray(row["actions"])[step-row["subtask_step"]]
            actual=np.asarray(command["native_action"])
            difference=np.abs(raw[:6]-actual[:6])>1e-7
            executed+=1;changed_rows+=int(difference.any());changed_elements+=int(difference.sum())
            matches=matches and bool(np.allclose(actual[:6],np.clip(raw[:6],-1,1),atol=1e-7,rtol=0))
        mode_result=result["modes"][mode]
        mode_result["executed_rows"]+=executed
        mode_result["executed_arm_changed_rows"]+=changed_rows
        mode_result["executed_arm_changed_elements"]+=changed_elements
        mode_result["all_arm_changes_match_clip"] &= matches
        result["episodes"].append(dict(path=str(path.parent),mode=mode,executed_rows=executed,
            changed_arm_rows=changed_rows,changed_arm_elements=changed_elements,all_match_clip=matches))
        receipts[str(path)]=sha(path);receipts[str(infer)]=sha(infer)
    for row in result["modes"].values():
        row["executed_arm_clipped_row_fraction"]=row["executed_arm_changed_rows"]/row["executed_rows"]
        row["executed_arm_clipped_element_fraction"]=row["executed_arm_changed_elements"]/(row["executed_rows"]*6)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--service",required=True);p.add_argument("--episodes",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists(): raise FileExistsError(output)
    result=audit(a.service,a.episodes);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result["modes"]))


if __name__=="__main__":main()
