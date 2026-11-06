"""Bounded task-chunk runner for 200 fixed LIBERO cases per condition.

Only this new results harness changes. The frozen original run_episode,
fresh-tail policy/runtime, environment seed, noise seed and GPU guards remain.
One client process handles each condition/task's up-to-five init episodes;
each episode still constructs/closes its own environment and resets policy KV.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from results.idea2.fresh_tail_20261025.common import save,sha,write_csv

QUALITY=ROOT/"results/idea2/fresh_tail_20261025/quality.py"
spec=importlib.util.spec_from_file_location("_closedloop200_quality",QUALITY)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
q.base.legacy.__file__=str(Path(__file__).resolve())
ARMS=("fresh18","sm16_tail0","sm32_tail0","sm32_tail2")
SUITES=tuple(q.base.legacy.SUITES)
GPU_DEFAULTS={0:dict(gpu_uuid="GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25",egl_device=1,port=19102,cpus="0-15"),
              1:dict(gpu_uuid="GPU-555cde36-5e41-bc44-0c69-ea47358fd703",egl_device=0,port=19103,cpus="96-111")}


def case_identity(suite_index,task,init):
    return suite_index*50+task*5+init


def universe():
    return [dict(case=case_identity(s,task,init),case_id=case_identity(s,task,init),task_id=s*10+task,
        suite=suite,task=task,init=init,label=arm,arm=arm,mode="fresh18" if arm=="fresh18" else "lead3",
        lead=18 if arm=="fresh18" else 3,**q.CONDITIONS[arm],
        case_key=f"{arm}/{suite}/task{task:02d}/init{init:02d}",
        relative_directory=f"episodes/{arm}/{suite}/task{task:02d}/init{init:02d}")
        for task in range(10) for s,suite in enumerate(SUITES) for arm in ARMS for init in range(5)]


def source_hashes():
    result=q.source_hashes()
    for path in (QUALITY,Path(__file__).resolve(),ROOT/"preparations/idea1_v3_3/egl_mapping.py"):
        result[str(path.relative_to(ROOT))]=sha(path)
    return result


def read_reuse(path):
    path=Path(path)
    if not path.exists():return {},dict(status="pending",path=str(path))
    value=json.loads(path.read_text())
    if value.get("status")!="passed":return {},dict(status="pending",path=str(path),observed_status=value.get("status"))
    selected={}
    for row in value["selected_cases"]:
        arm=row.get("arm",row.get("label"));key=(int(row["case_id"]),arm)
        expected=case_identity(SUITES.index(row["suite"]),row["task"],row["init"])
        if not row["compatible"] or expected!=key[0] or arm not in ARMS or key in selected:
            raise ValueError("Reuse manifest case identity/compatibility error")
        selected[key]=row
    if len(selected)!=32 or any(sum(key[1]==arm for key in selected)!=8 for arm in ARMS):
        raise ValueError("Expected exactly eight proven reusable cases per arm")
    return selected,dict(status="passed",path=str(path.resolve()),sha256=sha(path),cases=32)


def completed_cases(paths):
    result={}
    for path in paths:
        p=Path(path).resolve();execution=json.loads((p/"execution.json").read_text())
        if execution.get("status")!="passed" or not execution.get("source_unchanged") or execution.get("server_exit_code")!=0:
            raise ValueError("Completed-run deduplication accepts only passed provenance and graceful exit")
        for row in execution["cases"]:
            if row["status"]!="valid":raise ValueError("Completed run contains invalid case")
            key=(row["case_id"],row["label"])
            if key in result and result[key]["directory"]!=row["directory"]:raise ValueError("Duplicate completed case sources")
            result[key]=dict(row,completed_run=str(p))
    return result


def make_plan(args):
    selected_ids=args.task_ids
    if selected_ids is None:selected_ids=[s*10+t for t in range(10) for s in range(4) if (s*10+t)%2==args.shard]
    if len(set(selected_ids))!=len(selected_ids):raise ValueError("Duplicate task IDs")
    all_cases=universe();reuse,reuse_status=read_reuse(args.reuse_manifest);completed=completed_cases(args.completed_runs)
    cases=[];batches=[]
    for tid in selected_ids:
        offset=tid%len(ARMS);order=list(ARMS[offset:]+ARMS[:offset])
        if args.shard==1:order.reverse()
        for arm in order:
            selected=[dict(r) for r in all_cases if r["task_id"]==tid and r["label"]==arm]
            pending=[]
            for row in selected:
                key=(row["case_id"],arm)
                if key in reuse:
                    source=reuse[key];leaf=Path(source["source_directory"])
                    value=json.loads((leaf/"complete.json").read_text())
                    row.update(source_kind="reused_original8",status="valid",directory=str(leaf),result=value,
                        reuse_proof=source,gpu_uuid=source.get("gpu_uuid",source.get("source_gpu_uuid")))
                elif key in completed:
                    source=completed[key]
                    row.update(source_kind="reused_completed_chunk",status="valid",directory=source["directory"],result=source["result"],
                        completed_run=source["completed_run"],gpu_uuid=source.get("gpu_uuid"))
                else:
                    row.update(source_kind="new",status="pending",gpu_uuid=args.gpu_uuid,attempts=[])
                    pending.append(row["case_key"])
                cases.append(row)
            if pending:batches.append(dict(task_id=tid,label=arm,case_keys=pending))
    if not cases:raise ValueError("No task selected")
    q.base.legacy.episode_cases=lambda reverse=False:cases
    q.base.legacy.source_hashes=source_hashes
    result=q.base.original_plan(args)
    pending=[r for r in cases if r["source_kind"]=="new"]
    first=pending[0] if pending else cases[0]
    affinity=q.base.parse_cpus(args.cpus)
    if not affinity<=os.sched_getaffinity(0):raise ValueError("CPU affinity outside available CPUs")
    for role in ("server","client"):
        result[role]["environment"]["PROGRESSIVE_EXPECTED_CPUS"]=",".join(map(str,sorted(affinity)))
        result[role]["environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    result["server"]["argv"]=[args.server_python,"-u","-m","results.idea2.fresh_tail_20261025.adapter","--serve",
        "--ae-sm-min",str(first["ae_sm_min"]),"--fresh-tail-steps",str(first["fresh_tail_steps"]),"--encoder-after-step","1",
        "--checkpoint",result["checkpoint"],"--port",str(args.port),"--mode",first["mode"]]
    env=dict(os.environ);env.pop("CUDA_VISIBLE_DEVICES",None)
    mapping_code="import json; from preparations.idea1_v3_3.egl_mapping import mapping; print(json.dumps(mapping()))"
    egl_map=json.loads(subprocess.check_output([args.client_python,"-c",mapping_code],cwd=ROOT,env=env,text=True))
    devices=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    physical=next(int(line.split(",")[0]) for line in devices.splitlines() if line.split(",")[1].strip()==args.gpu_uuid)
    if egl_map.get(str(args.egl_device))!=physical:raise ValueError("EGL index does not map to the assigned physical GPU")
    result.update(source_sha256=source_hashes(),args=vars(args),protocol=q.base.expected_protocol(),cases=cases,batches=batches,
        selected_task_ids=selected_ids,cpu_affinity=sorted(affinity),reuse_manifest=reuse_status,egl_mapping=egl_map,
        full_matrix=dict(configurations=list(ARMS),suites=list(SUITES),tasks=list(range(10)),inits=list(range(5)),cases_per_arm=200,total_episodes=800),
        selected_episodes=len(cases),new_episodes=len(pending),reused_episodes=len(cases)-len(pending),
        schedule="Task blocks; all four conditions on the same GPU, rotated by global task ID and reversed on GPU3. Up to five original episodes per client process. At most four task blocks per model-service lifetime.",
        environment_reuse=False,client_batching="Python/import reuse only; original run_episode constructs/closes an environment and resets KV every episode",
        retry_policy=dict(max_technical_retries=args.max_technical_retries,task_failures_never_retried=True,guard_failures_abort=True),
        semantics="D10/H10/R5, original model_seed/envseed7/settling10/suite limits, no lookahead; original inference and per-episode reset protocol unchanged.",
        limitations=["200 fixed cases per condition is an exploratory expanded evaluation, not the official 50-init evaluation.",
            "Single root dispatcher owns unique task-chunk assignment; --completed-runs deduplicates passed prior chunks.",
            "Actual task failures remain valid; technical retry uses a new attempt directory, bounded to one retry."])
    return result


def run_worker_batch(path):
    payload=json.loads(Path(path).read_text());records=[];last_episode=None;last_epoch=None
    for case in payload["cases"]:
        leaf=Path(case["directory"]);request=leaf.parent/(leaf.name+".request.json")
        save(request,case);row=dict(case_key=case["case_key"],directory=str(leaf),attempt=case["attempt"])
        try:
            q.run_worker(request)
            infers=[json.loads(line) for line in (leaf/"infer.jsonl").read_text().splitlines()]
            identity=json.loads((leaf/"identity.json").read_text())
            if any(identity[k]!=case[k] for k in ("suite","task","init")) or identity["env_seed"]!=7:
                raise RuntimeError("Original episode identity changed")
            first=infers[0]
            if not first["bootstrap"] or first["request"]!=0 or first["frontiers"]!=[18]*10:
                raise RuntimeError("Episode did not start with fresh bootstrap")
            if last_episode is not None and (first["episode"]<=last_episode or first["snapshot"]["epoch"]<=last_epoch):
                raise RuntimeError("Policy episode/epoch did not advance between init episodes")
            last_episode=first["episode"];last_epoch=first["snapshot"]["epoch"]
            row.update(status="valid",bootstrap_input_hash=first["input_hash"],bootstrap_seed=first["seed"],
                episode=last_episode,bootstrap_epoch=last_epoch,result=json.loads((leaf/"complete.json").read_text()))
        except BaseException:
            row.update(status="technical_failure",traceback=traceback.format_exc())
        records.append(row);save(payload["receipt"],dict(status="running",cases=records))
    save(payload["receipt"],dict(status="passed" if all(r["status"]=="valid" for r in records) else "technical_failure",cases=records))


def normalized(cases):
    return [dict(case_id=r["case_id"],case_key=r["case_key"],task_id=r["task_id"],configuration=r["label"],label=r["label"],arm=r["label"],suite=r["suite"],task=r["task"],init=r["init"],
        status=r["status"],source_kind=r["source_kind"],gpu_uuid=r.get("gpu_uuid"),directory=r.get("directory"),
        success=r.get("result",{}).get("success"),steps=r.get("result",{}).get("steps"),requests=r.get("result",{}).get("predictions"),
        technical_attempts=len(r.get("attempts",[])),task_failure=(not r["result"]["success"]) if r["status"]=="valid" else None) for r in cases]


def execute(args,plan,dest):
    from preparations.idea2.layer_bench import require_idle_gpu,require_same_processes,gpu_snapshot
    if len(plan["selected_task_ids"])>4:raise ValueError("Each service lifetime is limited to four task blocks; split the plan into chunks")
    if plan["reuse_manifest"]["status"]!="passed":raise ValueError("Reuse proof is not passed yet")
    if (dest/"execution.json").exists():raise ValueError("Fresh output required; existing attempts must remain intact")
    checks=plan["dependency_checks"]
    if not(checks["server"]["passed"] and checks["client"]["passed"] and checks["checkpoint_exists"] and checks["weights_exists"] and checks["libero_config_exists"]):raise ValueError("Dependency preflight failed")
    if source_hashes()!=plan["source_sha256"]:raise ValueError("Frozen sources changed")
    with socket.socket() as test:test.bind(("127.0.0.1",args.port))
    os.sched_setaffinity(0,set(plan["cpu_affinity"]))
    state=dict(status="running",start_utc=q.base.legacy.now_utc(),cases=plan["cases"],batches=[],performance_result=False,
        gpu_before_init=require_idle_gpu(args.gpu_uuid),plan=str(dest/"plan.json"))
    def checkpoint():
        save(dest/"execution.json",state);rows=normalized(state["cases"]);save(dest/"normalized_cases.json",rows);write_csv(dest/"normalized_cases.csv",rows)
    bykey={r["case_key"]:r for r in state["cases"]};server=None;ready=False;own_pids=set()
    def rpc(message,timeout=600):
        response=q.base.original_rpc(args.port,message,timeout)
        if "protocol" in response:q.check_protocol(response["protocol"])
        return response
    def batch(keys,attempt):
        selected=[bykey[k] for k in keys];first=selected[0];q.current=dict(q.CONDITIONS[first["label"]])
        rpc(dict(op="configure",mode=first["mode"],interval=1,**q.current))
        serial=len(state["batches"]);tag=f"batch{serial:03d}_task{first['task_id']:02d}_{first['label']}_attempt{attempt}"
        payload=[]
        for case in selected:
            leaf=dest/case["relative_directory"]
            if attempt:leaf=leaf.with_name(leaf.name+f"_attempt{attempt}")
            payload.append(dict(case,directory=str(leaf),port=args.port,attempt=attempt))
            case["status"]="running"
        request=dest/"requests"/(tag+".json");receipt=dest/"requests"/(tag+".receipt.json")
        save(request,dict(cases=payload,receipt=str(receipt)))
        record=dict(tag=tag,case_keys=keys,attempt=attempt,status="running",start_utc=q.base.legacy.now_utc(),gpu_before=require_same_processes(args.gpu_uuid,own_pids))
        state["batches"].append(record);checkpoint()
        (dest/"logs").mkdir(exist_ok=True)
        with (dest/"logs"/(tag+".log")).open("x") as log:
            child=subprocess.Popen([args.client_python,"-u",str(Path(__file__).resolve()),"--worker-batch",str(request)],cwd=ROOT,
                env=dict(os.environ,**plan["client"]["environment"]),stdout=log,stderr=subprocess.STDOUT)
            record["client_pid"]=child.pid;checkpoint();started=time.monotonic()
            try:
                while child.poll() is None:
                    if time.monotonic()-started>args.episode_timeout*len(selected)+90:raise TimeoutError("Task batch deadline")
                    ownership=q.base.legacy.client_gpu_ownership(gpu_snapshot(args.gpu_uuid),own_pids,child.pid)
                    if ownership is not None and "render_gpu_verified" not in record:
                        record["render_gpu_verified"]=ownership;checkpoint()
                    if "render_gpu_verified" not in record and time.monotonic()-started>90:raise RuntimeError("Client render GPU unverified")
                    time.sleep(1)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
        record.update(client_exit_code=child.returncode,end_utc=q.base.legacy.now_utc(),wall_s=time.monotonic()-started)
        record["drain"]=rpc(dict(op="drain"));record["gpu_after"]=require_same_processes(args.gpu_uuid,own_pids)
        if child.returncode!=0:
            record["status"]="technical_failure";checkpoint()
            raise RuntimeError(f"Batch worker exited {child.returncode}; partial receipts cannot validate a failed process")
        if "render_gpu_verified" not in record:raise RuntimeError("Batch technically invalid: missing graphics ownership")
        observed={r["case_key"]:r for r in json.loads(receipt.read_text()).get("cases",[])} if receipt.exists() else {}
        failed=[]
        for case,request_case in zip(selected,payload):
            value=observed.get(case["case_key"],dict(status="technical_failure",error="Batch worker did not return this case"))
            case["attempts"].append({**value,"attempt":attempt,"batch":tag,"client_exit_code":child.returncode})
            if value["status"]=="valid":
                case.update(status="valid",directory=request_case["directory"],result=value["result"],
                    render_gpu_verified=record["render_gpu_verified"],bootstrap_episode=value["episode"],bootstrap_epoch=value["bootstrap_epoch"])
            else:case["status"]="technical_failure";failed.append(case["case_key"])
        record["status"]="passed" if not failed else "technical_failure";checkpoint();return failed
    try:
        checkpoint()
        first=next((r for r in state["cases"] if r["source_kind"]=="new"),state["cases"][0]);q.current=dict(q.CONDITIONS[first["label"]])
        with (dest/"server.log").open("x") as log:
            server=subprocess.Popen(plan["server"]["argv"],cwd=ROOT,env=dict(os.environ,**plan["server"]["environment"]),stdout=log,stderr=subprocess.STDOUT)
        state["server_pid"]=server.pid;checkpoint();deadline=time.monotonic()+args.server_ready_timeout
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError("Server exited during loading")
            try:state["protocol"]=rpc(dict(op="health"),timeout=2)["protocol"];ready=True;break
            except (ConnectionError,OSError,TimeoutError):time.sleep(.5)
        if not ready:raise TimeoutError("Model readiness deadline")
        occupancy=gpu_snapshot(args.gpu_uuid);own_pids={r["pid"] for r in occupancy["processes"]}
        if len(own_pids)!=1 or occupancy["graphics_processes"]:raise RuntimeError("Expected one exclusively owned model process")
        state["gpu_after_load"]=occupancy;checkpoint()
        for block in plan["batches"]:
            if source_hashes()!=plan["source_sha256"]:raise RuntimeError("Source changed during execution")
            failed=batch(block["case_keys"],0)
            if failed and args.max_technical_retries:failed=batch(failed,1)
            if failed:raise RuntimeError("Bounded technical retry exhausted; valid task failures are never retried")
        state["status"]="passed"
    except BaseException:
        state.update(status="failed",traceback=traceback.format_exc())
        for row in state["cases"]:
            if row["status"]=="running":row["status"]="technical_failure"
        raise
    finally:
        if server is not None:
            if ready and server.poll() is None:
                try:state["shutdown"]=rpc(dict(op="shutdown"),timeout=120)
                except Exception as error:state["shutdown_error"]=repr(error)
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.terminate()
                try:server.wait(timeout=15)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=15)
            state["server_exit_code"]=server.returncode
            if server.returncode!=0:state["status"]="failed"
        state["source_unchanged"]=source_hashes()==plan["source_sha256"]
        if not state["source_unchanged"]:state["status"]="excluded_source_drift"
        state["end_utc"]=q.base.legacy.now_utc();checkpoint()
    if state["status"]!="passed":raise RuntimeError("Service shutdown/provenance failed")


def parser():
    p=q.base.original_parser();p.description=__doc__
    p.add_argument("--shard",type=int,choices=(0,1),default=0)
    p.add_argument("--task-ids",type=int,nargs="+",choices=range(40))
    p.add_argument("--cpus")
    p.add_argument("--reuse-manifest",default=str(HERE/"reuse32.json"))
    p.add_argument("--completed-runs",nargs="*",default=[])
    p.add_argument("--max-technical-retries",type=int,choices=(0,1),default=1)
    p.add_argument("--worker-batch",help=argparse.SUPPRESS)
    p.set_defaults(output=None,gpu_uuid=None,egl_device=None,port=None)
    return p


def main():
    args=parser().parse_args()
    if args.worker_batch:return run_worker_batch(args.worker_batch)
    for key,value in GPU_DEFAULTS[args.shard].items():
        if getattr(args,key) is None:setattr(args,key,value)
    if args.output is None:args.output=str(HERE/f"gpu{args.shard+2}")
    dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    candidate=make_plan(args);plan_path=dest/"plan.json"
    if args.execute:
        previous=json.loads(plan_path.read_text()) if plan_path.exists() else candidate
        for key in ("server","client","cases","checkpoint","checkpoint_files","gpu_uuid","egl_device","port","source_sha256","reuse_manifest","selected_task_ids","batches"):
            if previous[key]!=candidate[key]:raise ValueError("Execution differs from saved plan: "+key)
    save(plan_path,candidate)
    if not args.execute:
        print(json.dumps(dict(status="planned_only",new=candidate["new_episodes"],reused=candidate["reused_episodes"],tasks=candidate["selected_task_ids"],reuse=candidate["reuse_manifest"]["status"],plan=str(plan_path)),indent=2));return
    execute(args,candidate,dest)


if __name__=="__main__":main()
