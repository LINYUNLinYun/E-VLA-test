"""CPU-only 56-case fresh-tail quality summary; incomplete runs stay in progress."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from results.idea2.fresh_tail_20261025.common import CONFIGS,HERE,ROOT,save,write_csv
from results.idea2.progressive_kv_20261025.analyze_resource_quality import compare,helpers


def analyze(directory):
    directory=Path(directory).resolve()
    plan=json.loads((directory/"plan.json").read_text())
    ep=directory/"execution.json"
    execution=json.loads(ep.read_text()) if ep.exists() else dict(status="not_started",cases=[])
    observed={(r["case"],r["label"]):r for r in execution.get("cases",[])}
    episodes={};cases=[];configurations={**CONFIGS,"fresh18":dict(ae_sm_min=0,fresh_tail_steps=0)}
    for case in plan["cases"]:
        key=(case["case"],case["label"]);entry=observed.get(key,{})
        leaf=directory/case["relative_directory"]
        row=dict(case,configuration=case["label"],status=entry.get("status","pending"),success=None,technical_failure=False)
        try:
            if entry.get("status")=="technical_failure":raise ValueError("Executor recorded technical failure")
            receipt_path=leaf.parent/(leaf.name+".worker_receipt.json")
            if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status")=="technical_failure":
                raise ValueError("Worker recorded technical failure")
            if entry.get("status")!="valid":cases.append(row);continue
            complete=json.loads((leaf/"complete.json").read_text());receipt=json.loads((leaf/"smoke_receipt.json").read_text())
            protocol=json.loads((leaf/"native_protocol.json").read_text())
            records=[json.loads(line) for line in (leaf/"infer.jsonl").read_text().splitlines()]
            if receipt["status"]!="passed" or complete["status"]!="valid" or not complete["valid"]:raise ValueError("Missing valid episode receipts")
            if len(records)!=complete["predictions"] or not records:raise ValueError("Incomplete inference history")
            condition=configurations[case["label"]];sms=condition["ae_sm_min"];tail=condition["fresh_tail_steps"]
            for record in records:
                p=record["protocol"];resource=p.get("resource_adapter",{});requested=resource.get("requested",resource)
                if (p.get("num_steps"),p.get("H"),p.get("R"))!=(10,10,5):raise ValueError("D/H/R protocol changed")
                if record["next_encoder"] or record["lookahead"]:raise ValueError("Future-observation lookahead enabled")
                if requested.get("ae_sm_min")!=sms or requested.get("encoder_after_step")!=1 or p.get("fresh_tail_steps")!=tail:
                    raise ValueError("Condition protocol changed")
                actual=resource.get("actual_resources")
                if sms and (not actual or any(r["sm_counts"]!=[sms,128-sms] for r in actual)):raise ValueError("Wrong actual SM allocation")
                if not sms and actual!=[]:raise ValueError("Fresh reference must be unpartitioned")
                if tail and record["frontiers"][-tail:]!=[18]*tail:raise ValueError("Incomplete fresh tail")
                if record["graph_kv_buffers"]!=2 or record["pool"]["live_snapshots"]>2:raise ValueError("KV slot bound violated")
            steady=[r for r in records if not r["bootstrap"] and not r["graph_setup_ms"]]
            row.update(status="valid",success=bool(complete["success"]),task_failure=not bool(complete["success"]),
                steps=complete["steps"],requests=len(records),steady_requests=len(steady),
                steady_rpc_ms=helpers.stats([r["client_roundtrip_ms"] for r in steady]),
                first_nonbootstrap_frontiers=records[1]["frontiers"] if len(records)>1 else None,
                graph_capture_requests=sum(r["graph_setup_ms"]>0 for r in records),all_work_wall_s=complete["all_work_wall_s"],
                validation_receipt=str(leaf/"smoke_receipt.json"),original_seed_validation="Reused passed original model_seed and D10/H10/R5 validation receipt")
            episodes[key]=dict(rows=records,protocol=protocol,case=case,plan=plan,gpu_uuid=plan["gpu_uuid"])
        except (KeyError,ValueError,TypeError,FileNotFoundError) as error:
            row.update(status="technical_failure",technical_failure=True,error=repr(error))
        cases.append(row)
    summaries=[];frontiers=[];bootstrap=[];deltas=[]
    for name in configurations:
        selected=[r for r in cases if r["label"]==name];valid=[r for r in selected if r["status"]=="valid"]
        records=[r for case in valid for r in episodes[case["case"],name]["rows"]]
        steady=[r for r in records if not r["bootstrap"] and not r["graph_setup_ms"]]
        summaries.append(dict(configuration=name,**configurations[name],expected_cases=8,valid_cases=len(valid),
            successes=sum(r["success"] for r in valid),task_failures=sum(r["task_failure"] for r in valid),
            technical_failures=sum(r["technical_failure"] for r in selected),total_steps=sum(r["steps"] for r in valid),
            steps=helpers.stats([r["steps"] for r in valid]),steady_rpc_ms=helpers.stats([r["client_roundtrip_ms"] for r in steady]),
            steady_requests=len(steady),graph_capture_requests=sum(r["graph_capture_requests"] for r in valid),gpu_uuid=plan["gpu_uuid"]))
        for iteration in range(10):
            values=[r["frontiers"][iteration] for r in steady]
            frontiers.append(dict(configuration=name,euler_iteration=iteration+1,requests=len(values),
                mean_new_layers=float(np.mean(values)) if values else None,
                mean_new_layer_fraction=float(np.mean(values)/18) if values else None))
        if name!="fresh18":
            for case in range(8):
                for request,table in ((0,bootstrap),(1,deltas)):
                    compared=compare(episodes.get((case,"fresh18")),episodes.get((case,name)),request=request,configuration=name,case=case)
                    compared["reference"]="same_gpu_unpartitioned_fresh18"
                    table.append(compared)
    bootstrap_passed=len(bootstrap)==48 and all(r["status"]=="comparable" and
        all(r[key]["max_abs"]<=1e-5 and r[key]["mae"]<=1e-6 for key in ("normalized","decoded_full_H10")) for r in bootstrap)
    all_cases=len(cases)==56 and all(r["status"]=="valid" for r in cases)
    finished=execution.get("status")=="passed" and execution.get("source_unchanged") is True and all_cases
    failed=execution.get("status") in ("failed","excluded_source_drift") or any(r["technical_failure"] for r in cases)
    status="complete" if finished and bootstrap_passed else "validation_failed" if finished else "technical_failure" if failed else "in_progress"
    return dict(status=status,generated_unix=time.time(),run=str(directory),execution_status=execution.get("status"),
        source_unchanged=execution.get("source_unchanged"),configurations=summaries,cases=cases,frontiers=frontiers,
        bootstrap_comparisons=bootstrap,bootstrap_gate_passed=bootstrap_passed,approximation_delta=deltas,
        notes=["Task failures are valid quality outcomes; technical failures are separate. All 56 cases are required before complete.",
          "Bootstrap checks same-GPU fresh18 input, original seed, model protocol, normalized and decoded actions; predefined max1e-5/MAE1e-6 tolerances.",
          "Request1 approximation deltas require matching current and actual old/bootstrap inputs, seeds and model D/H/R. Frontiers may differ deliberately; no later diverged trajectory is compared.",
          "Steady RPC excludes bootstrap/capture. Grouped free trajectories do not establish a paired hardware speedup. No gripper threshold is invented."])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default=str(HERE/"closedloop_gpu2"))
    parser.add_argument("--output",default=str(ROOT/"reports/idea2_fresh_tail_20261025"))
    args=parser.parse_args();result=analyze(args.run);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    save(output/"closedloop_summary.json",result)
    for key in ("configurations","cases","frontiers","bootstrap_comparisons","approximation_delta"):
        write_csv(output/(key+".csv"),result[key])
    print(json.dumps(dict(status=result["status"],valid_cases=sum(r["status"]=="valid" for r in result["cases"]),bootstrap_gate=result["bootstrap_gate_passed"])))


if __name__=="__main__":main()
