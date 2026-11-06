"""CPU-only resource-quality summary and separate request-1 approximation deltas.

This deliberately does not change the strict conditioning comparison table.
Different frontiers are allowed only in the explicitly named approximation
delta, after matching both observations, noise seeds, and model protocol.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("_resource_quality_summary_helpers",HERE/"analyze_closedloop.py")
helpers=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
MODES=helpers.MODES


def load_run(path,label_prefix,expected_sms=None):
    if not (path/"plan.json").exists():
        return dict(path=str(path),label_prefix=label_prefix,status="not_started",plan=None,cases=[],episodes={})
    plan=helpers.read_json(path/"plan.json")
    execution=helpers.read_json(path/"execution.json") if (path/"execution.json").exists() else dict(status="not_started",cases=[])
    observed={(row["case"],row["mode"]):row for row in execution.get("cases",[])}
    episodes={}
    cases=[]
    for case in plan["cases"]:
        key=(case["case"],case["mode"])
        entry=observed.get(key,{})
        directory=path/case["relative_directory"]
        row={name:case[name] for name in ("case","suite","task","init","mode","lead")}
        row.update(configuration=label_prefix+"_"+case["mode"],run=str(path),gpu_uuid=plan["gpu_uuid"],
            expected_sm_counts=expected_sms,status=entry.get("status","pending"),success=None,technical_failure=False)
        worker_path=directory.parent/(directory.name+".worker_receipt.json")
        if worker_path.exists():
            worker=helpers.read_json(worker_path)
            if worker.get("status")=="technical_failure":
                row.update(status="technical_failure",technical_failure=True,error=worker.get("error",worker.get("drain_error")))
        if entry.get("status")=="technical_failure":
            row.update(status="technical_failure",technical_failure=True,error=execution.get("error"))
        if entry.get("status")=="valid" and not row["technical_failure"]:
            try:
                receipt=helpers.read_json(directory/"smoke_receipt.json")
                complete=helpers.read_json(directory/"complete.json")
                protocol=helpers.read_json(directory/"native_protocol.json")
                infers=[json.loads(line) for line in (directory/"infer.jsonl").read_text().splitlines()]
                if receipt.get("status")!="passed" or complete.get("status")!="valid" or not complete.get("valid"):
                    raise ValueError("Episode validation receipt did not pass")
                if (protocol.get("num_steps"),protocol.get("H"),protocol.get("R"))!=(10,10,5):
                    raise ValueError("Expected D10/H10/R5")
                if len(infers)!=complete["predictions"] or not infers or not infers[0]["bootstrap"]:
                    raise ValueError("Incomplete inference history")
                if any(r["next_encoder"] or r["lookahead"] for r in infers):
                    raise ValueError("Closed-loop lookahead was enabled")
                if expected_sms is not None:
                    accepted=complete.get("resource_adapter",{})
                    if accepted.get("actual_sm_counts")!=[expected_sms] or accepted.get("verified_prediction_protocols")!=len(infers):
                        raise ValueError("Missing validated per-request SM allocation")
                    for request in infers:
                        resource=request["protocol"]["resource_adapter"]
                        requested=resource.get("requested",resource)
                        if requested.get("ae_sm_min")!=expected_sms[0] or requested.get("encoder_after_step")!=1:
                            raise ValueError("Resource request changed")
                        if not resource["actual_resources"] or any(v["sm_counts"]!=expected_sms for v in resource["actual_resources"]):
                            raise ValueError("Actual resource split changed")
                steady=[r for r in infers if not r["bootstrap"] and not r["graph_setup_ms"]]
                row.update(status="valid",success=bool(complete["success"]),task_failure=not bool(complete["success"]),
                    steps=complete["steps"],requests=len(infers),steady_requests=len(steady),
                    steady_rpc_ms=helpers.stats([r["client_roundtrip_ms"] for r in steady]),
                    steady_action_ready_ms=helpers.stats([r["timing"]["action_ready_host_ms"] for r in steady]),
                    mean_new_layer_fraction=float(np.mean([f/18 for r in steady for f in r["frontiers"]])) if steady else None,
                    first_nonbootstrap_frontiers=infers[1]["frontiers"] if len(infers)>1 else None,
                    all_work_wall_s=complete["all_work_wall_s"],validation_receipt=str(directory/"smoke_receipt.json"),
                    seed_validation="Reused passed original model_seed validation",D=10,H=10,R=5)
                episodes[key]=dict(rows=infers,protocol=protocol,case=case,plan=plan,gpu_uuid=plan["gpu_uuid"])
            except (KeyError,ValueError,TypeError,FileNotFoundError) as error:
                row.update(status="technical_failure",technical_failure=True,error=repr(error))
        cases.append(row)
    complete=execution.get("status")=="passed" and execution.get("source_unchanged") is True and all(r["status"]=="valid" for r in cases)
    status="complete" if complete else "technical_failure" if execution.get("status") in ("failed","excluded_source_drift") or any(r["technical_failure"] for r in cases) else "in_progress"
    return dict(path=str(path),label_prefix=label_prefix,status=status,execution_status=execution.get("status"),
        source_unchanged=execution.get("source_unchanged"),plan=plan,cases=cases,episodes=episodes)


def model_identity(episode):
    protocol=episode["protocol"]
    return dict(checkpoint=episode["plan"]["checkpoint"],config=protocol.get("config"),
        D=protocol.get("num_steps"),H=protocol.get("H"),R=protocol.get("R"),attention=protocol.get("attention"))


def compare(reference,candidate,*,request,configuration,case):
    row=dict(configuration=configuration,case=case,reference="normal_fresh18",request=request,
             comparison="bootstrap_consistency" if request==0 else "approximation_delta",status="pending")
    if reference is None or candidate is None:
        return row
    if len(reference["rows"])<=request or len(candidate["rows"])<=request:
        row.update(status="not_available",reason="Episode has no request1")
        return row
    ref,got=reference["rows"][request],candidate["rows"][request]
    first_ref,first_got=reference["rows"][0],candidate["rows"][0]
    matches=dict(current_input=ref["input_hash"]==got["input_hash"],
        bootstrap_input=first_ref["input_hash"]==first_got["input_hash"],
        current_seed=ref["seed"]==got["seed"],bootstrap_seed=first_ref["seed"]==first_got["seed"],
        model_protocol=model_identity(reference)==model_identity(candidate),
        request_identity=ref["request"]==got["request"]==request,
        old_source=ref["old_prefix_observation_id"]==got["old_prefix_observation_id"]==0,
        new_source=ref["new_prefix_observation_id"]==got["new_prefix_observation_id"]==request,
        no_lookahead=not any(r["lookahead"] or r["next_encoder"] for r in (ref,got,first_ref,first_got)))
    row.update(matches=matches,candidate_gpu_uuid=candidate["gpu_uuid"],reference_gpu_uuid=reference["gpu_uuid"],
        current_input_hash=got["input_hash"],bootstrap_input_hash=first_got["input_hash"],seed=got["seed"],
        reference_frontiers=ref["frontiers"],candidate_frontiers=got["frontiers"],
        frontier_sequences_equal=ref["frontiers"]==got["frontiers"])
    if request==0:
        matches["full_fresh_bootstrap"]=ref["bootstrap"] and got["bootstrap"] and ref["frontiers"]==got["frontiers"]==[18]*10
    else:
        matches["first_nonbootstrap"]=not ref["bootstrap"] and not got["bootstrap"]
    if not all(matches.values()):
        row.update(status="not_comparable",reason=";".join(name for name,value in matches.items() if not value))
        return row
    row["status"]="comparable"
    decoded_ref=np.asarray(ref["chunk_actions"])
    decoded_got=np.asarray(got["chunk_actions"])
    if decoded_ref.shape!=(10,7) or decoded_got.shape!=(10,7):
        raise ValueError("Validated D10/H10/R5 response has wrong decoded action shape")
    row["decoded_full_H10"]=helpers.errors(decoded_ref,decoded_got)
    row["decoded_execution_prefix_R5"]=helpers.errors(decoded_ref[:5],decoded_got[:5])
    if request==0:
        row["normalized"]=helpers.errors(ref["normalized_actions"],got["normalized_actions"])
        row["exact_all_actions"]=row["normalized"]["exact_equal"] and row["decoded_full_H10"]["exact_equal"]
    else:
        row.update(frontiers_may_differ=True,implementation_numerical_error=False,
            reference_actual_executed_length=ref["executed_length"],candidate_actual_executed_length=got["executed_length"],
            interpretation="Approximate-algorithm difference on matched current/old observations and noise, not an implementation numerical-consistency failure")
    return row


def analyze(normal_path,sm16_path,sm32_path):
    runs=[load_run(normal_path,"normal"),load_run(sm16_path,"sm16",[16,112]),load_run(sm32_path,"sm32",[32,96])]
    normal=runs[0]
    configurations=[("normal_"+mode,normal,mode) for mode in MODES]+[("sm16_lead3",runs[1],"lead3"),("sm32_lead3",runs[2],"lead3")]
    cases=[row for run in runs for row in run["cases"]]
    summaries=[]
    frontiers=[]
    bootstrap=[]
    deltas=[]
    for name,run,mode in configurations:
        selected=[row for row in cases if row["configuration"]==name]
        valid=[row for row in selected if row["status"]=="valid"]
        records=[r for row in valid for r in run["episodes"][row["case"],mode]["rows"]]
        steady=[r for r in records if not r["bootstrap"] and not r["graph_setup_ms"]]
        summaries.append(dict(configuration=name,status=run["status"],expected_cases=8,valid_cases=len(valid),
            successes=sum(row["success"] for row in valid),task_failures=sum(row["task_failure"] for row in valid),
            technical_failures=sum(row["technical_failure"] for row in selected),
            steps=helpers.stats([row["steps"] for row in valid]),total_steps=sum(row["steps"] for row in valid),
            steady_rpc_ms=helpers.stats([r["client_roundtrip_ms"] for r in steady]),
            steady_action_ready_ms=helpers.stats([r["timing"]["action_ready_host_ms"] for r in steady]),
            steady_requests=len(steady),gpu_uuid=run["plan"]["gpu_uuid"] if run["plan"] else None,
            requested_resources=run["plan"].get("resource_adapter") if run["plan"] else None,
            next_encoder=False,all_work_wall_s=sum(row["all_work_wall_s"] for row in valid)))
        for iteration in range(10):
            values=[r["frontiers"][iteration] for r in steady]
            mean=float(np.mean(values)) if values else None
            frontiers.append(dict(configuration=name,euler_iteration=iteration+1,requests=len(values),
                mean_new_layers=mean,new_layer_fraction=mean/18 if values else None,
                min_new_layers=min(values) if values else None,max_new_layers=max(values) if values else None))
        if name!="normal_fresh18":
            for case in range(8):
                reference=normal["episodes"].get((case,"fresh18"))
                candidate=run["episodes"].get((case,mode))
                bootstrap.append(compare(reference,candidate,request=0,configuration=name,case=case))
                deltas.append(compare(reference,candidate,request=1,configuration=name,case=case))
    for summary in summaries:
        name=summary["configuration"]
        matched_bootstrap=[r for r in bootstrap if r["configuration"]==name and r["status"]=="comparable"]
        matched_delta=[r for r in deltas if r["configuration"]==name and r["status"]=="comparable"]
        summary.update(bootstrap_comparable_cases=len(matched_bootstrap),
            bootstrap_exact_action_cases=sum(r["exact_all_actions"] for r in matched_bootstrap),
            bootstrap_normalized_max_abs=max((r["normalized"]["max_abs"] for r in matched_bootstrap),default=None),
            bootstrap_decoded_max_abs=max((r["decoded_full_H10"]["max_abs"] for r in matched_bootstrap),default=None),
            approximation_delta_comparable_cases=len(matched_delta))
        for key,label in (("decoded_full_H10","H10"),("decoded_execution_prefix_R5","R5")):
            summary["approximation_delta_"+label+"_mae"]=helpers.stats([r[key]["mae"] for r in matched_delta])
            summary["approximation_delta_"+label+"_max_abs"]=max((r[key]["max_abs"] for r in matched_delta),default=None)
    excluded=HERE/"closedloop_sm32_gpu3_clean"
    exclusions=[]
    if excluded.exists() and excluded.resolve()!=sm32_path.resolve():
        exclusions.append(dict(path=str(excluded),reason="Parent-designated port19086 TIME_WAIT preflight attempt, not the actual SM32 episode run",
            execution_json_exists=(excluded/"execution.json").exists(),completed_episodes=len(list(excluded.glob("episodes/*/*/task*/init00/complete.json"))),
            action="preserved unchanged and excluded from episode counts"))
    complete=all(run["status"]=="complete" for run in runs) and all(row["valid_cases"]==8 for row in summaries)
    status="complete" if complete else "technical_failure" if any(run["status"]=="technical_failure" for run in runs) else "in_progress"
    return dict(status=status,generated_unix=time.time(),configurations=summaries,cases=cases,frontiers=frontiers,
        bootstrap_comparisons=bootstrap,approximation_delta=deltas,excluded_preflight_attempts=exclusions,
        runs=[{key:value for key,value in run.items() if key not in ("episodes","cases")} for run in runs],
        notes=["Each resource configuration uses the original eight suite/task/init cases and passed original seed/D10/H10/R5/EGL validation receipts.",
            "Actual SM allocations and encoder_after_step=1 are verified from every resource-run inference protocol. next_encoder=False everywhere.",
            "Bootstrap compares normal GPU0 fresh18 input, seed, normalized actions and decoded actions; cross-GPU differences are reported directly without assuming exact equality.",
            "approximation_delta is separate from strict conditioning comparisons: only request1 is compared, after matching current input, bootstrap/old input, both seeds, source IDs and model D/H/R/config/checkpoint; frontiers may differ intentionally.",
            "H10 delta is over all decoded [10,7] values; R5 delta is over the first five scheduled actions. Actual executed length is recorded separately in case an episode ends early.",
            "No later diverged request is compared. No gripper-flip metric or invented threshold is used.",
            "Steady RPC distributions pool free closed-loop trajectories and run on different GPUs; they are not controlled same-input hardware speedups.",
            "Task failures remain valid quality outcomes; technical failures are separate. Existing strict conditioning files and all frozen sources are untouched."])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal",default=str(HERE/"closedloop_gpu0"))
    parser.add_argument("--sm16",default=str(HERE/"closedloop_sm16_gpu3_clean"))
    parser.add_argument("--sm32",default=str(HERE/"closedloop_sm32_gpu3_clean_port19087"))
    parser.add_argument("--output",default="reports/idea2_progressive_kv_20261025")
    args=parser.parse_args()
    result=analyze(Path(args.normal).resolve(),Path(args.sm16).resolve(),Path(args.sm32).resolve())
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    temp=output/"resource_quality_summary.json.tmp"
    temp.write_text(json.dumps(result,indent=2))
    temp.replace(output/"resource_quality_summary.json")
    for key in ("configurations","cases","frontiers","bootstrap_comparisons","approximation_delta"):
        helpers.write_csv(output/f"resource_quality_{key}.csv",result[key])
    print(json.dumps(dict(status=result["status"],configurations=[{key:row[key] for key in ("configuration","status","valid_cases","successes","technical_failures")} for row in result["configurations"]],
        approximation_delta_comparable=sum(row["status"]=="comparable" for row in result["approximation_delta"])),indent=2))


if __name__=="__main__":
    main()
