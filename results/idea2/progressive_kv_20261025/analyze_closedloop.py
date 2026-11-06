"""CPU-only closed-loop summary; incomplete matrices never report completion.

Action equality is checked only for complete matching conditioning histories:
observation hashes, seeds, source identities, model protocol, and KV frontiers.
Different trajectories/frontiers are counted, not treated as numerical errors.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


MODES=("lead0","lead1","lead3","lead6","fresh18")
LEADS=dict(zip(MODES,(0,1,3,6,18)))


def read_json(path):
    return json.loads(path.read_text())


def stats(values):
    values=np.asarray(values,dtype=float)
    if not len(values):
        return dict(n=0,mean=None,p50=None,p95=None,min=None,max=None)
    return dict(n=len(values),mean=float(values.mean()),p50=float(np.percentile(values,50)),
                p95=float(np.percentile(values,95)),min=float(values.min()),max=float(values.max()))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def errors(reference,candidate):
    a,b=np.asarray(reference,np.float64),np.asarray(candidate,np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid action array in validated artifact")
    difference=np.abs(a-b)
    return dict(mae=float(difference.mean()),max_abs=float(difference.max()),exact_equal=bool(np.array_equal(a,b)))


def write_csv(path,rows):
    def flatten(row,prefix=""):
        result={}
        for key,value in row.items():
            name=prefix+key
            if isinstance(value,dict):
                result.update(flatten(value,name+"_"))
            else:
                result[name]=value
        return result
    rows=[flatten(row) for row in rows]
    fields=sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def conditioning(rows,protocol,checkpoint):
    """Cumulative history is conservative: no comparison after input divergence."""
    model=dict(checkpoint=checkpoint,config=protocol.get("config"),D=protocol.get("num_steps"),
               H=protocol.get("H"),R=protocol.get("R"),attention=protocol.get("attention"))
    inputs=fingerprint(model)
    complete=inputs
    result=[]
    for row in rows:
        inputs=fingerprint([inputs,dict(input_hash=row["input_hash"],seed=row["seed"],request=row["request"],
            step=row["step"],old_source=row["old_prefix_observation_id"],new_source=row["new_prefix_observation_id"],
            bootstrap=row["bootstrap"],next_encoder=row["next_encoder"],lookahead=row["lookahead"])])
        complete=fingerprint([complete,inputs,row["frontiers"]])
        result.append(dict(input_history=inputs,complete_conditioning=complete))
    return result


def analyze(root,max_error,mae_error):
    plan=read_json(root/"plan.json")
    execution_path=root/"execution.json"
    execution=read_json(execution_path) if execution_path.exists() else dict(status="not_started",cases=[])
    current={(row["case"],row["label"]):row for row in execution.get("cases",[])}
    case_rows=[]
    observations={}
    validation_findings=[]
    inputs_used=[]
    for case in plan["cases"]:
        key=(case["case"],case["label"])
        active=current.get(key,{})
        directory=root/case["relative_directory"]
        item={name:case[name] for name in ("case","label","mode","suite","task","init","lead")}
        item.update(directory=str(directory),status=active.get("status","pending"),success=None,
                    technical_failure=False,task_failure=False)
        if active.get("status")=="technical_failure":
            item["technical_failure"]=True
            item["error"]=active.get("error",execution.get("error"))
        worker_path=directory.parent/(directory.name+".worker_receipt.json")
        if worker_path.exists():
            worker=read_json(worker_path)
            if worker.get("status")=="technical_failure":
                item.update(status="technical_failure",technical_failure=True,error=worker.get("error",worker.get("drain_error")))
        complete_path=directory/"complete.json"
        receipt_path=directory/"smoke_receipt.json"
        if active.get("status")=="valid" and complete_path.exists() and receipt_path.exists():
            try:
                complete=read_json(complete_path)
                receipt=read_json(receipt_path)
                protocol=read_json(directory/"native_protocol.json")
                if complete.get("status")!="valid" or not complete.get("valid") or receipt.get("status")!="passed":
                    raise ValueError("Existing validation receipts did not pass")
                if (protocol.get("num_steps"),protocol.get("H"),protocol.get("R"))!=(10,10,5):
                    raise ValueError("D10/H10/R5 protocol mismatch")
                if protocol.get("mode")!=case["mode"] or protocol.get("lead")!=case["lead"] or protocol.get("next_encoder") is not False:
                    raise ValueError("Condition or next-encoder protocol mismatch")
                rows=[json.loads(line) for line in (directory/"infer.jsonl").read_text().splitlines()]
                if len(rows)!=complete["predictions"] or not rows or sum(r["bootstrap"] for r in rows)!=1:
                    raise ValueError("Incomplete inference or bootstrap count")
                if not rows[0]["bootstrap"] or rows[0]["frontiers"]!=[18]*10:
                    raise ValueError("First response is not the full current-KV bootstrap")
                if any(r["next_encoder"] or r["lookahead"] for r in rows):
                    raise ValueError("Unexpected closed-loop future observation")
                item.update(status="valid",success=bool(complete["success"]),task_failure=not bool(complete["success"]),
                    steps=complete["steps"],predictions=complete["predictions"],
                    all_work_wall_s=complete["all_work_wall_s"],client_process_wall_s=active.get("client_process_wall_s"),
                    cold_capture_requests=complete["cold_capture_requests"],
                    bootstrap_rpc_ms=rows[0]["client_roundtrip_ms"],bootstrap_ms=rows[0]["bootstrap_ms"],
                    graph_setup_ms=rows[0]["graph_setup_ms"],validation_receipt=str(receipt_path),
                    seed_validation="Reused passed original model_seed validation from smoke_receipt.json",
                    D=10,H=10,R=5,source_input_hash=rows[0]["input_hash"])
                steady=[r for r in rows if not r["bootstrap"] and r["graph_setup_ms"]==0]
                item["steady_rpc_ms"]=stats([r["client_roundtrip_ms"] for r in steady])
                item["steady_action_ready_ms"]=stats([r["timing"]["action_ready_host_ms"] for r in steady])
                item["steady_mean_new_layer_fraction"]=float(np.mean([f/18 for r in steady for f in r["frontiers"]])) if steady else None
                observations[key]=dict(rows=rows,conditioning=conditioning(rows,protocol,plan["checkpoint"]),case=case)
                inputs_used.extend([str(complete_path),str(receipt_path),str(directory/"native_protocol.json"),str(directory/"infer.jsonl")])
            except (KeyError,ValueError,TypeError) as error:
                item.update(status="technical_failure",technical_failure=True,error=repr(error))
                validation_findings.append(dict(case=key,error=repr(error)))
        case_rows.append(item)
    summaries={}
    frontiers=[]
    for mode in MODES:
        selected=[r for r in case_rows if r["mode"]==mode]
        valid=[r for r in selected if r["status"]=="valid"]
        records=[row for case in valid for row in observations[case["case"],mode]["rows"]]
        steady=[r for r in records if not r["bootstrap"] and r["graph_setup_ms"]==0]
        summaries[mode]=dict(expected_episodes=len(selected),valid_episodes=len(valid),
            success_count=sum(r["success"] for r in valid),task_failure_count=sum(r["task_failure"] for r in valid),
            technical_failure_count=sum(r["technical_failure"] for r in selected),
            pending_or_running=sum(r["status"] in ("pending","running") for r in selected),
            success_rate=sum(r["success"] for r in valid)/len(valid) if valid else None,
            executed_steps=sum(r["steps"] for r in valid),steps=stats([r["steps"] for r in valid]),
            requests=len(records),steady_requests=len(steady),
            steady_rpc_ms=stats([r["client_roundtrip_ms"] for r in steady]),
            steady_action_ready_ms=stats([r["timing"]["action_ready_host_ms"] for r in steady]),
            bootstrap_rpc_ms=stats([r["client_roundtrip_ms"] for r in records if r["bootstrap"]]),
            total_case_work_wall_s=sum(r["all_work_wall_s"] for r in valid),
            step_count_includes_valid_task_failures=True,
            latency_scope="Pooled real closed-loop requests across valid episodes; trajectories may differ; not matched-input GPU benchmark")
        for phase,phase_rows in (("bootstrap",[r for r in records if r["bootstrap"]]),("steady",steady)):
            for step in range(10):
                values=[r["frontiers"][step] for r in phase_rows]
                moments=stats(values)
                frontiers.append(dict(mode=mode,lead=LEADS[mode],phase=phase,euler_iteration=step+1,
                    requests=len(values),mean_new_layers=moments["mean"],min_new_layers=moments["min"],max_new_layers=moments["max"],
                    new_layer_fraction=moments["mean"]/18 if values else None,
                    old_layer_fraction=1-moments["mean"]/18 if values else None,
                    aggregation="Request-weighted, including valid task failures"))
    bootstrap=[]
    comparisons=[]
    all_cases=sorted({r["case"] for r in plan["cases"]})
    for case_index in all_cases:
        reference=observations.get((case_index,"fresh18"))
        for mode in MODES[:-1]:
            candidate=observations.get((case_index,mode))
            first=dict(case=case_index,mode=mode,reference_mode="fresh18",status="pending",input_match=None,seed_match=None)
            if reference is None or candidate is None:
                bootstrap.append(first)
                continue
            for index in range(min(len(reference["rows"]),len(candidate["rows"]))):
                ref,got=reference["rows"][index],candidate["rows"][index]
                rc,gc=reference["conditioning"][index],candidate["conditioning"][index]
                row=dict(case=case_index,mode=mode,reference_mode="fresh18",request=index,bootstrap=index==0,
                    input_history_match=rc["input_history"]==gc["input_history"],
                    complete_conditioning_match=rc["complete_conditioning"]==gc["complete_conditioning"],
                    current_frontiers_match=ref["frontiers"]==got["frontiers"],
                    reference_frontiers=ref["frontiers"],candidate_frontiers=got["frontiers"])
                if row["complete_conditioning_match"]:
                    row["normalized_error"]=errors(ref["normalized_actions"],got["normalized_actions"])
                    row["decoded_error"]=errors(ref["chunk_actions"],got["chunk_actions"])
                    row["tolerance_pass"]=all(row[name]["max_abs"]<=max_error and row[name]["mae"]<=mae_error for name in ("normalized_error","decoded_error"))
                    row["status"]="comparable"
                else:
                    row["status"]="not_comparable"
                    row["reason"]="different_KV_conditioning_history" if row["input_history_match"] else "observation_or_noise_history_diverged"
                comparisons.append(row)
                if index==0:
                    first.update(input_match=ref["input_hash"]==got["input_hash"],seed_match=ref["seed"]==got["seed"],
                        input_hash=ref["input_hash"],seed=ref["seed"],status="passed" if row.get("tolerance_pass") else "failed",
                        normalized_error=row.get("normalized_error"),decoded_error=row.get("decoded_error"),
                        conditioning_match=row["complete_conditioning_match"])
            bootstrap.append(first)
    expected_full=40
    valid_count=sum(r["status"]=="valid" for r in case_rows)
    failures=sum(r["technical_failure"] for r in case_rows)
    full_matrix=len(plan["cases"])==expected_full and {r["mode"] for r in plan["cases"]}==set(MODES) and len(all_cases)==8
    matched_failures=[r for r in comparisons if r["status"]=="comparable" and not r["tolerance_pass"]]
    complete=(execution.get("status")=="passed" and execution.get("source_unchanged") is True and
              valid_count==expected_full and full_matrix and len(bootstrap)==32 and
              all(r["status"]=="passed" for r in bootstrap) and not validation_findings and not matched_failures)
    status="complete" if complete else "technical_failure" if failures or execution.get("status") in ("failed","excluded_source_drift") else "validation_failed" if execution.get("status")=="passed" and (any(r["status"]=="failed" for r in bootstrap) or matched_failures or validation_findings) else "in_progress"
    return dict(status=status,generated_unix=time.time(),root=str(root),execution_status=execution.get("status"),
        full_matrix_expected_episodes=40,planned_episodes=len(plan["cases"]),valid_episodes=valid_count,
        technical_failure_count=failures,full_matrix=full_matrix,modes=summaries,cases=case_rows,frontiers=frontiers,
        bootstrap_comparisons=bootstrap,bootstrap_expected=32,
        bootstrap_comparisons_passed=sum(r["status"]=="passed" for r in bootstrap),
        conditioning_comparisons=comparisons,validation_findings=validation_findings,
        conditioning_comparison_counts=dict(comparable=sum(r["status"]=="comparable" for r in comparisons),
            input_history_diverged=sum(r.get("reason")=="observation_or_noise_history_diverged" for r in comparisons),
            different_KV_history=sum(r.get("reason")=="different_KV_conditioning_history" for r in comparisons),
            matched_tolerance_failures=len(matched_failures)),
        tolerance=dict(max_abs=max_error,mae=mae_error),source_sha256=plan["source_sha256"],
        source_unchanged_at_shutdown=execution.get("source_unchanged"),input_artifacts=inputs_used,
        notes=["Original seed and H/D/R checks are reused from passed episode validation receipts; no environment or GPU reruns.",
            "Fresh bootstrap comparisons use each of eight cases against fresh18 for all four mixed-KV modes (32 comparisons).",
            "Action differences are computed only for identical cumulative observation/noise/source/model and frontier histories; diverged trajectories or different KV schedules have no action-error metric.",
            "Valid task failure remains in success-rate/step/latency denominators. Technical failure is separate.",
            "Frontier is the new-KV layer count out of 18, aggregated separately for each of 10 Euler iterations; bootstrap is separated.",
            "next_encoder is false throughout; these closed-loop results test approximate mixed-KV quality, not unavailable lookahead scheduling."])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="results/idea2/progressive_kv_20261025/closedloop_gpu0")
    parser.add_argument("--output",default="reports/idea2_progressive_kv_20261025")
    parser.add_argument("--max-error",type=float,default=1e-5)
    parser.add_argument("--mae-error",type=float,default=1e-6)
    args=parser.parse_args()
    result=analyze(Path(args.input).resolve(),args.max_error,args.mae_error)
    dest=Path(args.output)
    dest.mkdir(parents=True,exist_ok=True)
    temporary=dest/"closedloop_summary.json.tmp"
    temporary.write_text(json.dumps(result,indent=2))
    temporary.replace(dest/"closedloop_summary.json")
    write_csv(dest/"closedloop_modes.csv",[dict(mode=mode,**row) for mode,row in result["modes"].items()])
    for name in ("cases","frontiers","bootstrap_comparisons","conditioning_comparisons"):
        write_csv(dest/f"closedloop_{name}.csv",result[name])
    print(json.dumps(dict(status=result["status"],valid_episodes=result["valid_episodes"],
        expected_episodes=40,bootstrap_passed=result["bootstrap_comparisons_passed"],
        modes={mode:dict(valid=row["valid_episodes"],success=row["success_count"],technical=row["technical_failure_count"]) for mode,row in result["modes"].items()}),indent=2))


if __name__=="__main__":
    main()
