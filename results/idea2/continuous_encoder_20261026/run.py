"""Task-shard runner for the fixed300 continuous-observation closed-loop cells."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

R=Path(__file__).resolve().parent
ROOT=R.parents[2]
ARMS=("fresh18_no_encoder","sm16_no_encoder","sm16_encoder")
SUITES=("libero_object","libero_spatial","libero_goal","libero_10")
DEFAULTS={0:dict(gpu_uuid="GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25",egl=1,cpus="0-15"),
          1:dict(gpu_uuid="GPU-555cde36-5e41-bc44-0c69-ea47358fd703",egl=0,cpus="96-111")}


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(path)


def universe():
    return [dict(configuration=a,arm=a,suite=s,task=t,init=i,task_id=si*10+t,
                 case_id=si*50+t*5+i,case_key=f"{a}/{s}/task{t:02d}/init{i:02d}")
            for si,s in enumerate(SUITES) for t in range(10) for i in ([0,1,2] if t%2==0 else [0,1]) for a in ARMS]


def sources():
    old=json.loads((ROOT/"results/idea2/closedloop200_20261025/gpu2_chunk0/plan.json").read_text())["source_sha256"]
    paths=[ROOT/name for name in old]
    paths += [R/name for name in ("run.py","policy.py","server.py","client.py")]
    paths += [ROOT/"results/idea2/online_encoder_20261025"/name for name in ("policy.py","wire.py","client.py")]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--shard",type=int,choices=(0,1),default=0)
    p.add_argument("--port",type=int,required=True);p.add_argument("--task-ids",type=int,nargs="+")
    p.add_argument("--case-keys",nargs="+",help="suite/task/init, all three conditions; gate cells remain in final universe")
    p.add_argument("--completed-runs",nargs="*",default=[])
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((ROOT/"results/idea2/closedloop200_20261025/gpu2_chunk0/plan.json").read_text())
    options=DEFAULTS[args.shard];cpus=set(range(0,16)) if args.shard==0 else set(range(96,112));os.sched_setaffinity(0,cpus)
    env_server={**os.environ,**prior["server"]["environment"],"CUDA_VISIBLE_DEVICES":options["gpu_uuid"],"PROGRESSIVE_EXPECTED_CPUS":",".join(map(str,sorted(cpus)))}
    env_client={**os.environ,**prior["client"]["environment"],"MUJOCO_EGL_DEVICE_ID":str(options["egl"]),"PROGRESSIVE_EXPECTED_CPUS":",".join(map(str,sorted(cpus)))}
    rows=universe()
    if args.task_ids is not None:rows=[r for r in rows if r["task_id"] in args.task_ids]
    if args.case_keys:
        keys=set(args.case_keys);rows=[r for r in rows if f"{r['suite']}/{r['task']}/{r['init']}" in keys]
        if len(rows)!=len(keys)*3:raise ValueError("Gate keys are outside fixed100-case universe")
    completed={}
    for path in args.completed_runs:
        s=json.loads((Path(path)/"execution.json").read_text())
        if s["status"]!="passed" or not s["source_unchanged"] or s["server_exit_code"]!=0:raise ValueError("Only passed completed sources can be reused")
        for c in s["cases"]:
            if c["status"]!="valid":raise ValueError("Invalid source case")
            completed[c["case_key"]]=c
    for row in rows:
        if row["case_key"] in completed:
            original=completed[row["case_key"]];row.update(status="valid",directory=original["directory"],result=original["result"],source_kind="completed_gate")
        else:row.update(status="pending",source_kind="new")
    if not rows:raise ValueError("No selected cases")
    if len({r["task_id"] for r in rows if r["status"]=="pending"})>4:raise ValueError("At most4 task blocks per model lifetime")
    from preparations.idea1_v2_1.runtime_client import rpc
    from preparations.idea2.layer_bench import require_idle_gpu,require_same_processes,gpu_snapshot
    from results.idea2.d10_dynamic_sm_20261024.closedloop_smoke import client_gpu_ownership
    with socket.socket() as test:test.bind(("127.0.0.1",args.port))
    frozen=sources()
    state=dict(status="starting",start_utc=now(),pid=os.getpid(),args=vars(args),cases=rows,
        source_before=frozen,gpu_before=require_idle_gpu(options["gpu_uuid"]),gpu_uuid=options["gpu_uuid"],
        server_environment={k:env_server.get(k) for k in prior["server"]["environment"]},
        client_environment={k:env_client.get(k) for k in prior["client"]["environment"]},cpu_affinity=sorted(cpus))
    save(out/"execution.json",state)
    server=None;child=None
    try:
        command=[prior["server"]["python"],"-u","-m","results.idea2.continuous_encoder_20261026.server",
                 "--port",str(args.port),"--arm",ARMS[0],"--cleanup",str(out/"cleanup.json")]
        state["server_command"]=command
        with (out/"server.log").open("x") as log:
            server=subprocess.Popen(command,cwd=ROOT,env=env_server,stdout=log,stderr=subprocess.STDOUT)
            state["server_pid"]=server.pid;save(out/"execution.json",state);deadline=time.monotonic()+300
            while True:
                if server.poll() is not None:raise RuntimeError("Model server exited during startup")
                try:
                    with socket.create_connection(("127.0.0.1",args.port),timeout=.2):pass
                    break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError("Model startup timeout")
                    time.sleep(1)
            state["health"]=rpc(args.port,dict(op="health"))
            snapshot=gpu_snapshot(options["gpu_uuid"]);own={x["pid"] for x in snapshot["processes"]}
            if len(own)!=1 or snapshot["graphics_processes"]:raise RuntimeError("Expected one exclusively owned model C process")
            state.update(status="running",owned_host_pids=sorted(own));save(out/"execution.json",state)
            for tid in sorted({r["task_id"] for r in rows}):
                offset=tid%3;order=list(ARMS[offset:]+ARMS[:offset])
                if args.shard:order.reverse()
                for arm in order:
                    group=[r for r in rows if r["task_id"]==tid and r["arm"]==arm and r["status"]=="pending"]
                    if not group:continue
                    state.update(current_task_id=tid,current_arm=arm);rpc(args.port,dict(op="configure",arm=arm))
                    for row in group:
                        row["gpu_before"]=require_same_processes(options["gpu_uuid"],own)
                        leaf=out/"episodes"/arm/row["suite"]/f"task{row['task']:02d}"/f"init{row['init']:02d}"
                        leaf.parent.mkdir(parents=True,exist_ok=True)
                        row.update(status="running",directory=str(leaf),start_utc=now());save(out/"execution.json",state)
                        command=[prior["client"]["python"],"-u","-m","results.idea2.continuous_encoder_20261026.client",
                                 "--port",str(args.port),"--suite",row["suite"],"--task",str(row["task"]),"--init",str(row["init"]),"--arm",arm,"--output",str(leaf)]
                        with (leaf.parent/f"init{row['init']:02d}.log").open("x") as client_log:
                            child=subprocess.Popen(command,cwd=ROOT,env=env_client,stdout=client_log,stderr=subprocess.STDOUT)
                            row["client_command"]=command;row["client_pid"]=child.pid;save(out/"execution.json",state);started=time.monotonic()
                            while child.poll() is None:
                                if time.monotonic()-started>600:raise TimeoutError("Episode deadline")
                                ownership=client_gpu_ownership(gpu_snapshot(options["gpu_uuid"]),own,child.pid)
                                if ownership is not None and "render_gpu_verified" not in row:
                                    row["render_gpu_verified"]=ownership;save(out/"execution.json",state)
                                if "render_gpu_verified" not in row and time.monotonic()-started>90:raise RuntimeError("EGL ownership unverified")
                                time.sleep(1)
                        row.update(client_exit_code=child.returncode,end_utc=now(),gpu_after=require_same_processes(options["gpu_uuid"],own))
                        if child.returncode or "render_gpu_verified" not in row:raise RuntimeError("Client failed or EGL not verified")
                        value=json.loads((leaf/"complete.json").read_text())
                        if value.get("valid") is not True:raise RuntimeError("Technical invalid episode")
                        row.update(status="valid",result=value);save(out/"execution.json",state)
                        print(row["case_key"],value["success"],value["steps"],flush=True)
            rpc(args.port,dict(op="shutdown"));server.wait(timeout=60)
            state.update(server_exit_code=server.returncode,source_after=sources(),end_utc=now())
            if server.returncode or frozen!=state["source_after"]:raise RuntimeError("Server exit or source guard failed")
            cleanup=json.loads((out/"cleanup.json").read_text())
            if not cleanup["passed"]:raise RuntimeError("Green Context cleanup failed")
            state.update(status="passed",source_unchanged=True,cleanup=cleanup)
    except BaseException:
        state.update(status="technical_failure",traceback=traceback.format_exc(),end_utc=now())
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=15)
        if server is not None and server.poll() is None:
            try:rpc(args.port,dict(op="shutdown"));server.wait(timeout=30)
            except Exception:server.terminate();server.wait(timeout=15)
        if server is not None:state["server_exit_code"]=server.returncode
        state["source_after"]=sources();state["source_unchanged"]=frozen==state["source_after"]
        raise
    finally:save(out/"execution.json",state)


if __name__=="__main__":main()
