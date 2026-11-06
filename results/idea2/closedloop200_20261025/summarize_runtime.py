"""Terminal-only CPU runtime summary for the fixed 200-case quality study.

Historical pilot observations contribute to quality and step counts only.
Runtime metrics read the 192 new episodes/arm, once after dispatch passes.
No model, simulator, Torch or GPU tool is imported or launched.
"""
import argparse
from collections import defaultdict
import datetime as dt
import json
from pathlib import Path
import time
import traceback

import numpy as np

from results.idea2.fresh_tail_20261025.common import save,sha,write_csv

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ARMS=("fresh18","sm16_tail0","sm32_tail0","sm32_tail2")
SUITES=("libero_object","libero_spatial","libero_goal","libero_10")


def stats(values):
    a=np.asarray(values,dtype=np.float64)
    if not a.size:return dict(n=0,mean=None,p50=None,p95=None,min=None,max=None,sum=0.0)
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError("Non-finite or non-scalar timing sample")
    return dict(n=int(a.size),mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),
                min=float(a.min()),max=float(a.max()),sum=float(a.sum()))


def seconds(value):
    return dt.datetime.fromisoformat(value.replace("Z","+00:00")).timestamp()


def elapsed(start,end):
    value=seconds(end)-seconds(start)
    if value<0:raise ValueError("Negative observed wall interval")
    return value


def identity(row):
    arm=row.get("configuration",row.get("label",row.get("arm")))
    if any(row.get(key,arm)!=arm for key in ("configuration","label","arm")):raise ValueError("Conflicting arm aliases")
    return arm,row["suite"],int(row["task"]),int(row["init"])


def pilot(row):
    return row["task"] in (0,5) and row["init"]==0


def compact_request(row):
    frontiers=row["frontiers"]
    if len(frontiers)!=10 or any(type(v) is not int or not 0<=v<=18 for v in frontiers):raise ValueError("Invalid frontier")
    if (row["protocol"]["num_steps"],row["protocol"]["H"],row["protocol"]["R"])!=(10,10,5):raise ValueError("D/H/R changed")
    rpc=float(row["client_roundtrip_ms"]);capture=float(row["graph_setup_ms"])
    if not np.isfinite([rpc,capture]).all() or min(rpc,capture)<0:raise ValueError("Invalid request timing")
    return dict(rpc_ms=rpc,action_ready_ms=float(row["timing"]["action_ready_host_ms"]),
        bootstrap=bool(row["bootstrap"]),capture_ms=capture,bootstrap_ms=float(row["bootstrap_ms"]),
        frontiers=frontiers,steady=not row["bootstrap"] and capture==0)


def episode_runtime(row):
    leaf=Path(row["directory"])
    complete=json.loads((leaf/"complete.json").read_text())
    receipt=json.loads((leaf/"smoke_receipt.json").read_text())
    if complete.get("status")!="valid" or not complete.get("valid") or receipt.get("status")!="passed":
        raise ValueError("Episode validation receipts missing: "+str(leaf))
    if any(complete[k]!=row[k] for k in ("success","steps")) or complete["predictions"]!=row["requests"]:
        raise ValueError("Normalized/episode result mismatch")
    samples=[]
    with (leaf/"infer.jsonl").open() as stream:
        for index,line in enumerate(stream):
            raw=json.loads(line)
            if raw["request"]!=index or bool(raw["bootstrap"])!=(index==0):raise ValueError("Request/bootstrap identity mismatch")
            samples.append(compact_request(raw))
    if len(samples)!=row["requests"]:raise ValueError("Incomplete inference history")
    steady=[r for r in samples if r["steady"]]
    native=complete["runtime_result"]
    cost=dict(configuration=identity(row)[0],case_key=row.get("case_key"),case_id=row.get("case_id"),suite=row["suite"],task=row["task"],init=row["init"],
        gpu_uuid=row["gpu_uuid"],directory=str(leaf),success=row["success"],steps=row["steps"],requests=len(samples),steady_requests=len(steady),
        episode_all_work_wall_s=float(complete["all_work_wall_s"]),
        episode_run_episode_wall_s=float(native["wall_s"]),episode_record_io_s=float(native["record_io_s"]),
        episode_drain_wall_s=float(complete["drain_wall_s"]),
        bootstrap_requests=sum(r["bootstrap"] for r in samples),capture_requests=sum(r["capture_ms"]>0 for r in samples),
        capture_ms_sum=sum(r["capture_ms"] for r in samples),bootstrap_ms_sum=sum(r["bootstrap_ms"] for r in samples),
        all_rpc_ms_sum=sum(r["rpc_ms"] for r in samples),steady_rpc_ms=stats([r["rpc_ms"] for r in steady]),
        termination=native.get("termination"))
    return dict(cost=cost,samples=samples,steady=steady)


def frontier_summary(episodes,configuration,scope="all_gpus"):
    rows=[];with_steady=[e for e in episodes if e["steady"]]
    for i in range(10):
        values=[r["frontiers"][i] for e in with_steady for r in e["steady"]]
        per_episode=[float(np.mean([r["frontiers"][i] for r in e["steady"]])) for e in with_steady]
        request_mean=float(np.mean(values)) if values else None
        episode_mean=float(np.mean(per_episode)) if per_episode else None
        rows.append(dict(configuration=configuration,scope=scope,euler_iteration=i+1,steady_requests=len(values),episodes_with_steady=len(per_episode),
            request_weighted_mean_new_layers=request_mean,request_weighted_old_layer_fraction=(18-request_mean)/18 if values else None,
            episode_balanced_mean_new_layers=episode_mean,episode_balanced_old_layer_fraction=(18-episode_mean)/18 if per_episode else None))
    return rows


def runtime_metrics(episodes,batches):
    costs=[e["cost"] for e in episodes];samples=[r for e in episodes for r in e["samples"]]
    steady=[r for r in samples if r["steady"]];boot=[r for r in samples if r["bootstrap"]]
    work=sum(c["episode_all_work_wall_s"] for c in costs)
    batch_work=sum(r["client_process_wall_s"] for r in batches)
    return dict(episodes=len(costs),requests=len(samples),steady_requests=len(steady),
        episode_all_work_wall_s=stats([c["episode_all_work_wall_s"] for c in costs]),
        successful_episode_all_work_wall_s=stats([c["episode_all_work_wall_s"] for c in costs if c["success"]]),
        original_run_episode_wall_s=stats([c["episode_run_episode_wall_s"] for c in costs]),
        episode_record_io_s=stats([c["episode_record_io_s"] for c in costs]),
        episode_drain_wall_s=stats([c["episode_drain_wall_s"] for c in costs]),
        all_rpc_ms=stats([r["rpc_ms"] for r in samples]),steady_rpc_ms=stats([r["rpc_ms"] for r in steady]),
        steady_action_ready_host_ms=stats([r["action_ready_ms"] for r in steady]),bootstrap_rpc_ms=stats([r["rpc_ms"] for r in boot]),
        capture_requests=sum(r["capture_ms"]>0 for r in samples),capture_ms=stats([r["capture_ms"] for r in samples if r["capture_ms"]>0]),
        bootstrap_requests=len(boot),bootstrap_module_ms=stats([r["bootstrap_ms"] for r in boot]),
        client_batches=len(batches),batch_attempted_episodes=sum(r["attempted_episodes"] for r in batches),
        client_batch_process_wall_s=stats([r["client_process_wall_s"] for r in batches]),
        episode_work_normalized_episodes_per_hour=len(costs)*3600/work if work else None,
        batch_process_normalized_episodes_per_hour=len(costs)*3600/batch_work if batch_work else None,
        rate_scope="Rates above divide counts by summed work/process time, not two-GPU makespan; bootstrap/capture are nested within episode/RPC/batch time, never additive overhead.")


def load_chunks(dispatch):
    chunks=[];batches=[];owners={};failed_launches=[]
    for item in dispatch["chunks"]:
        if item["status"]!="passed" or item.get("parent_exit_code")!=0:raise ValueError("Chunk parent did not exit0/passed")
        path=Path(item["output"]);execution=json.loads((path/"execution.json").read_text());plan=json.loads((path/"plan.json").read_text())
        if execution["status"]!="passed" or execution.get("server_exit_code")!=0 or execution.get("source_unchanged") is not True:
            raise ValueError("Chunk service/provenance is incomplete")
        cases={r["case_key"]:r for r in execution["cases"]}
        for row in execution["cases"]:
            if row["source_kind"]=="new":
                key=identity(row)
                if key in owners:raise ValueError("New episode has two owning chunks")
                owners[key]=dict(row,chunk=item["id"],chunk_directory=str(path))
        records=execution["batches"]
        for batch in records:
            if batch.get("client_exit_code")!=0 or "render_gpu_verified" not in batch:raise ValueError("Client exit/render ownership failed")
            labels={cases[k]["label"] for k in batch["case_keys"]}
            if len(labels)!=1:raise ValueError("Batch mixes conditions")
            batches.append(dict(chunk=item["id"],configuration=labels.pop(),gpu_uuid=plan["gpu_uuid"],tag=batch["tag"],status=batch["status"],
                attempt=batch["attempt"],attempted_episodes=len(batch["case_keys"]),client_process_wall_s=float(batch["wall_s"]),
                observed_batch_interval_s=elapsed(batch["start_utc"],batch["end_utc"]),
                parent_post_batch_drain_ms=batch.get("drain",{}).get("drain_ms")))
        startup=elapsed(execution["start_utc"],records[0]["start_utc"]) if records else None
        final=elapsed(records[-1]["end_utc"],execution["end_utc"]) if records else None
        gaps=[elapsed(left["end_utc"],right["start_utc"]) for left,right in zip(records,records[1:])]
        chunks.append(dict(chunk=item["id"],gpu_uuid=plan["gpu_uuid"],directory=str(path),task_ids=plan["selected_task_ids"],
            start_utc=execution["start_utc"],end_utc=execution["end_utc"],chunk_wall_s=elapsed(execution["start_utc"],execution["end_utc"]),
            new_episodes=sum(r["source_kind"]=="new" for r in execution["cases"]),
            startup_and_initial_configuration_wall_s_proxy=startup,
            between_batch_parent_work_wall_s=stats(gaps),final_batch_postwork_and_shutdown_wall_s_proxy=final,
            pure_model_load_s=None,pure_shutdown_s=None,
            observed_client_process_wall_s_sum=sum(float(b["wall_s"]) for b in records),server_exit_code=0,source_unchanged=True))
        for prior in item.get("previous_attempts",[]):
            leaf=Path(prior["output"])
            failed_launches.append(dict(chunk=item["id"],directory=str(leaf),status=prior["status"],parent_exit_code=prior.get("parent_exit_code"),
                observed_launch_wall_s=elapsed(prior["start_utc"],prior["end_utc"]) if prior.get("start_utc") and prior.get("end_utc") else None,
                execution_receipt_exists=(leaf/"execution.json").exists(),excluded_from_quality=True))
    return chunks,batches,owners,failed_launches


def analyze(root):
    root=Path(root).resolve();dispatch_path=root/"dispatch_execution.json";dispatch=json.loads(dispatch_path.read_text())
    if dispatch.get("status")!="passed":
        return dict(status="in_progress" if dispatch.get("status")=="running" else "blocked_on_dispatch",dispatch_status=dispatch.get("status"),
                    full_episode_scan_performed=False,reason="Wait for terminal passed dispatch; no running episode histories scanned")
    payload=json.loads((root/"normalized_cases.json").read_text());rows=payload["cases"] if isinstance(payload,dict) else payload
    expected={(a,s,t,i) for a in ARMS for s in SUITES for t in range(10) for i in range(5)}
    selected={}
    for row in rows:
        key=identity(row)
        if key not in expected or key in selected:raise ValueError("Unknown/duplicate normalized case")
        if row["status"]!="valid" or type(row["success"]) is not bool:raise ValueError("Invalid case in terminal aggregate")
        selected[key]=row
    if set(selected)!=expected:raise ValueError("Terminal normalized coverage is not 800 unique cases")
    reuse=json.loads((root/"reuse32.json").read_text())
    if reuse["status"]!="passed":raise ValueError("Historical reuse proof did not pass")
    old={(r["arm"],r["suite"],r["task"],r["init"]):r for r in reuse["selected_cases"]}
    for key,row in selected.items():
        if pilot(row):
            if key not in old or Path(row["directory"]).resolve()!=Path(old[key]["source_directory"]).resolve():raise ValueError("Pilot source differs from proven reuse")
    chunks,batches,owners,failed_launches=load_chunks(dispatch)
    episodes=[];failures=[]
    for key,row in selected.items():
        if not row["success"]:
            failures.append(dict(configuration=key[0],suite=key[1],task=key[2],init=key[3],case_id=row.get("case_id"),
                steps=row["steps"],requests=row["requests"],directory=row["directory"],source_kind=row.get("source_kind"),
                status="valid",task_failure=True,technical_failure=False,failed_task_never_reclassified=True))
        if pilot(row):continue
        if key not in owners or Path(owners[key]["directory"]).resolve()!=Path(row["directory"]).resolve():raise ValueError("New episode owner/directory mismatch")
        row=dict(row,gpu_uuid=owners[key]["gpu_uuid"])
        episode=episode_runtime(row);episode["cost"]["chunk"]=owners[key]["chunk"];episodes.append(episode)
    if len(episodes)!=768:raise ValueError("Expected 768 newly executed episodes")
    configurations=[];frontiers=[];gpu_configurations=[]
    for arm in ARMS:
        quality=[r for k,r in selected.items() if k[0]==arm];new=[r for r in quality if not pilot(r)]
        arm_episodes=[e for e in episodes if e["cost"]["configuration"]==arm]
        arm_batches=[b for b in batches if b["configuration"]==arm]
        configurations.append(dict(configuration=arm,valid_cases=len(quality),successes=sum(r["success"] for r in quality),
            task_failures=sum(not r["success"] for r in quality),success_rate=sum(r["success"] for r in quality)/len(quality),
            steps_all200=stats([r["steps"] for r in quality]),successful_only_steps_all200=stats([r["steps"] for r in quality if r["success"]]),
            new192_valid_cases=len(new),new192_successes=sum(r["success"] for r in new),new192_steps=stats([r["steps"] for r in new]),
            new192_successful_only_steps=stats([r["steps"] for r in new if r["success"]]),new192_runtime=runtime_metrics(arm_episodes,arm_batches)))
        frontiers.extend(frontier_summary(arm_episodes,arm))
        for gpu in sorted({e["cost"]["gpu_uuid"] for e in arm_episodes}):
            group=[e for e in arm_episodes if e["cost"]["gpu_uuid"]==gpu];group_batches=[b for b in arm_batches if b["gpu_uuid"]==gpu]
            gpu_configurations.append(dict(configuration=arm,gpu_uuid=gpu,new_runtime=runtime_metrics(group,group_batches)))
            frontiers.extend(frontier_summary(group,arm,gpu))
    per_gpu=[]
    for gpu in sorted({r["gpu_uuid"] for r in chunks}):
        group=[r for r in chunks if r["gpu_uuid"]==gpu]
        first=min(r["start_utc"] for r in group);last=max(r["end_utc"] for r in group)
        per_gpu.append(dict(gpu_uuid=gpu,model_service_startups=len(group),model_service_restarts_after_first=max(0,len(group)-1),
            new_episodes=sum(r["new_episodes"] for r in group),service_chunk_wall_s_sum=sum(r["chunk_wall_s"] for r in group),
            first_to_last_service_window_s=elapsed(first,last),startup_initial_configuration_proxy_s_sum=sum(r["startup_and_initial_configuration_wall_s_proxy"] or 0 for r in group)))
    first=min(r["start_utc"] for r in chunks);last=max(r["end_utc"] for r in chunks);wall=elapsed(first,last)
    return dict(status="complete",generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),full_episode_scan_performed=True,
        provenance=dict(normalized_cases_sha256=sha(root/"normalized_cases.json"),dispatch_sha256=sha(dispatch_path),reuse32_sha256=sha(root/"reuse32.json"),
            summary_script_sha256=sha(Path(__file__)),dispatch_status="passed",runtime_scope="768 new episodes only; 32 historical episodes contribute quality/steps only"),
        configurations=configurations,gpu_configurations=gpu_configurations,frontiers=frontiers,valid_failures=failures,
        episode_costs=[e["cost"] for e in episodes],batches=batches,chunks=chunks,per_gpu_service_costs=per_gpu,previous_launch_failures=failed_launches,
        campaign=dict(new_episodes=768,reused_quality_only=32,first_new_chunk_start_utc=first,last_new_chunk_end_utc=last,
            observed_new_campaign_window_s=wall,new_valid_episodes_per_observed_wall_hour=768*3600/wall,
            new_actions=sum(e["cost"]["steps"] for e in episodes),new_requests=sum(e["cost"]["requests"] for e in episodes),
            new_actions_per_observed_wall_s=sum(e["cost"]["steps"] for e in episodes)/wall,
            new_requests_per_observed_wall_s=sum(e["cost"]["requests"] for e in episodes)/wall,
            model_service_startups=len(chunks),model_service_restarts_after_first_per_gpu=sum(r["model_service_restarts_after_first"] for r in per_gpu)),
        definitions=["Overall quality and steps use all200 per arm; successful-only steps exclude valid task failures without deleting them. Full failure listing remains available.",
            "Every runtime, RPC, capture, bootstrap, frontier and throughput metric uses only the192 new cases per arm. Historical pilot timing is never read or added.",
            "Steady requests exclude any bootstrap or graph_setup_ms>0 request. Mean/P50/P95 use concatenated raw request samples, never averages of per-case percentiles.",
            "Episode all_work_wall_s is the original worker's observed health/run_episode/drain interval; native run_episode wall begins after environment construction. Final artifact validation is outside episode all_work.",
            "Record I/O, drain, RPC, capture and bootstrap are nested observed subintervals, not independent additive costs. Batch process wall includes client import/setup and episode work; do not add these columns.",
            "The capture column uses recorded graph_setup_ms, which includes graph preparation, warmup/capture and initial replay setup; it is not isolated CUDA capture API time.",
            "Pure model load and pure shutdown were not recorded. Startup proxy is chunk start through first batch start (load/readiness/configure/checks); final proxy is last batch end through chunk end (drain/checks/report/shutdown). Between-batch gaps include configuration and parent work.",
            "Service restarts count new chunk lifetimes per GPU after its first. Failed preflight launches are listed separately; campaign wall includes any gaps inside its observed start/end window.",
            "Campaign throughput counts768 new valid episodes, including task failures, over the actual two-GPU first-to-last chunk window. It reflects GPU handoff, waits and task composition, not isolated inference throughput.",
            "Frontiers use steady requests only. Request weighting gives long episodes more weight; episode-balanced means average each episode's mean equally. Neither freely diverging trajectory statistic establishes a causal speedup or a causal algorithm-quality effect."])


def self_test():
    pooled=stats([1,2,3,100]);mean_case_p95=np.mean([stats([1,2])["p95"],stats([3,100])["p95"]])
    assert abs(pooled["p95"]-85.45)<1e-10 and pooled["p95"]!=mean_case_p95
    def raw(rpc,bootstrap=False,capture=0,frontier=18):
        return dict(frontiers=[frontier]*10,protocol=dict(num_steps=10,H=10,R=5),client_roundtrip_ms=rpc,
            graph_setup_ms=capture,bootstrap=bootstrap,bootstrap_ms=2 if bootstrap else 0,timing=dict(action_ready_host_ms=rpc/2))
    samples=[compact_request(raw(1000,True,10)),compact_request(raw(1)),compact_request(raw(3)),compact_request(raw(500,False,20))]
    assert [r["rpc_ms"] for r in samples if r["steady"]]==[1,3]
    e1=dict(steady=[compact_request(raw(1,frontier=0))]);e2=dict(steady=[compact_request(raw(1)) for _ in range(3)])
    f=frontier_summary([e1,e2],"test")[0]
    assert f["request_weighted_mean_new_layers"]==13.5 and f["episode_balanced_mean_new_layers"]==9
    assert f["request_weighted_old_layer_fraction"]==.25 and f["episode_balanced_old_layer_fraction"]==.5
    assert pilot(dict(task=5,init=0)) and not pilot(dict(task=5,init=1)) and not pilot(dict(task=1,init=0))
    assert elapsed("2026-10-25T00:00:00+00:00","2026-10-25T00:00:02+00:00")==2
    assert identity(dict(configuration="fresh18",label="fresh18",arm="fresh18",suite="libero_object",task=0,init=0))[0]=="fresh18"
    return dict(status="passed",checks=["raw pooled percentile differs from mean per-case P95","bootstrap/capture exclusion","request vs episode frontier weighting","old32 coordinate exclusion","observed UTC interval","arm alias consistency"],gpu_operations=0,episode_histories_scanned=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default=str(HERE));p.add_argument("--output",default=str(ROOT/"reports/idea2_closedloop200_20261025"))
    p.add_argument("--self-test",action="store_true");p.add_argument("--wait",action="store_true");p.add_argument("--timeout",type=float,default=43200)
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.self_test:
        receipt=self_test();save(output/"runtime_summary_selftest.json",receipt);print(json.dumps(receipt));return
    started=time.monotonic()
    while args.wait:
        d=json.loads((Path(args.root)/"dispatch_execution.json").read_text())
        if d.get("status")!="running":break
        if time.monotonic()-started>=args.timeout:raise TimeoutError("Runtime summary wait exhausted; no episode scan performed")
        time.sleep(30)
    try:result=analyze(args.root)
    except BaseException:
        save(output/"runtime_summary.json",dict(status="validation_failed",traceback=traceback.format_exc(),full_episode_scan_performed=None))
        raise
    save(output/"runtime_summary.json",result)
    if result["status"]=="complete":
        for filename,key in (("runtime_summary","configurations"),("runtime_by_gpu","gpu_configurations"),("runtime_frontiers","frontiers"),
            ("valid_failures","valid_failures"),("runtime_episode_costs","episode_costs"),("runtime_batches","batches"),("runtime_chunks","chunks")):
            write_csv(output/(filename+".csv"),result[key])
    print(json.dumps(dict(status=result["status"],full_episode_scan_performed=result["full_episode_scan_performed"])))


if __name__=="__main__":main()
