"""Bounded four-arm causal online pilot; all changes are in this new harness."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parent
ROOT = R.parents[2]
ARMS = ["sm16_no_encoder", "sm16_encoder", "fresh18_no_encoder", "fresh18_encoder"]


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def sources():
    paths = [R/name for name in ("run.py", "client.py", "policy.py", "server.py", "wire.py")]
    paths += list((ROOT/"preparations/idea2").glob("progressive_kv*.py"))
    paths += [ROOT/"results/idea2/fresh_tail_20261025/adapter.py", ROOT/"results/idea2/progressive_kv_20261025/resource_adapter.py",
              ROOT/"preparations/idea1_v2_1/runtime_client.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    p.add_argument("--case-indices", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--port", type=int, default=19220)
    args = p.parse_args()
    out = Path(args.output).resolve();out.mkdir(parents=True, exist_ok=False)
    old = json.loads((ROOT/"results/idea2/closedloop200_20261025/gpu2_chunk0/plan.json").read_text())
    server_env = dict(os.environ, **old["server"]["environment"])
    client_env = dict(os.environ, **old["client"]["environment"])
    cpus = set(old["cpu_affinity"]);os.sched_setaffinity(0, cpus)
    from preparations.idea1_v2_1.runtime_client import rpc
    from preparations.idea2.layer_bench import require_idle_gpu, gpu_snapshot, require_same_processes
    from results.idea2.d10_dynamic_sm_20261024.closedloop_smoke import client_gpu_ownership
    uuid = server_env["CUDA_VISIBLE_DEVICES"]
    all_cases = [(s,t,0) for s in ["libero_object","libero_spatial","libero_goal","libero_10"] for t in [0,5]]
    cases = [all_cases[i] for i in args.case_indices]
    if len(set(args.case_indices)) != len(args.case_indices) or len(set(args.arms)) != len(args.arms): raise ValueError("Duplicate pilot cells")
    with socket.socket() as test:test.bind(("127.0.0.1",args.port))
    before = sources()
    state = dict(status="starting", start_utc=now(), args=vars(args), cases=[],
                 gpu_before=require_idle_gpu(uuid), source_before=before,
                 semantics="One returned R5 chunk executes during next request; actual end-of-chunk observation supplied in-request. Explicit nominal250ms lag; not wall-clock paced.",
                 hypothesis="No late observation will be treated as a GPU overlap; no future replay frames in this closed loop.")
    save(out/"execution.json",state)
    server = None
    child = None
    try:
        command = [old["server"]["python"],"-u","-m","results.idea2.online_encoder_20261025.server",
                   "--port",str(args.port),"--arm",args.arms[0],"--cleanup",str(out/"cleanup.json")]
        state.update(server_command=command, server_environment=old["server"]["environment"],
                     client_environment=old["client"]["environment"])
        with (out/"server.log").open("x") as log:
            server = subprocess.Popen(command,cwd=ROOT,env=server_env,stdout=log,stderr=subprocess.STDOUT)
            state["server_pid"] = server.pid;save(out/"execution.json",state)
            deadline=time.monotonic()+300
            while True:
                if server.poll() is not None:raise RuntimeError("Online model server exited during loading")
                try:
                    with socket.create_connection(("127.0.0.1",args.port),timeout=.2):pass
                    break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError("Online server startup timeout")
                    time.sleep(1)
            # A readiness probe closes without a request; it must not kill the server.
            health=rpc(args.port,dict(op="health"));state["health"]=health
            occupancy=gpu_snapshot(uuid)
            own={r["pid"] for r in occupancy["processes"]}
            if len(own)!=1 or occupancy["graphics_processes"]:raise RuntimeError("Unexpected GPU processes after exclusive model load")
            state.update(status="running",owned_gpu_pids=sorted(own));save(out/"execution.json",state)
            for arm in args.arms:
                state["current_arm"]=arm
                rpc(args.port,dict(op="configure",arm=arm))
                for suite,task,init in cases:
                    require_same_processes(uuid,own)
                    leaf=out/"episodes"/arm/suite/f"task{task:02d}"/f"init{init:02d}"
                    leaf.parent.mkdir(parents=True,exist_ok=True)
                    row=dict(arm=arm,suite=suite,task=task,init=init,directory=str(leaf),status="running",start_utc=now())
                    state["cases"].append(row);save(out/"execution.json",state)
                    cmd=[old["client"]["python"],"-u","-m","results.idea2.online_encoder_20261025.client",
                         "--port",str(args.port),"--suite",suite,"--task",str(task),"--init",str(init),
                         "--arm",arm,"--output",str(leaf)]
                    with (leaf.parent/f"init{init:02d}.log").open("x") as client_log:
                        child=subprocess.Popen(cmd,cwd=ROOT,env=client_env,stdout=client_log,stderr=subprocess.STDOUT)
                        row["client_pid"]=child.pid;save(out/"execution.json",state)
                        started=time.monotonic()
                        while child.poll() is None:
                            if time.monotonic()-started>600:
                                child.terminate();child.wait(timeout=15);raise TimeoutError("Online episode timeout")
                            snapshot=gpu_snapshot(uuid)
                            ownership=client_gpu_ownership(snapshot,own,child.pid)
                            if ownership is not None and "render_gpu_verified" not in row:
                                row["render_gpu_verified"]=ownership;save(out/"execution.json",state)
                            if "render_gpu_verified" not in row and time.monotonic()-started>90:
                                child.terminate();child.wait(timeout=15);raise RuntimeError("Client render GPU unverified")
                            time.sleep(2)
                    row.update(client_exit_code=child.returncode,end_utc=now())
                    if child.returncode:raise RuntimeError("Online client technical failure; inspect preserved log")
                    row["gpu_after"]=require_same_processes(uuid,own)
                    if "render_gpu_verified" not in row:raise RuntimeError("Missing verified EGL process")
                    result=json.loads((leaf/"complete.json").read_text())
                    if result.get("valid") is not True:raise RuntimeError("Invalid online result")
                    row.update(status="valid",result=result)
                    save(out/"execution.json",state)
                    print(arm,suite,task,"success",result["success"],flush=True)
            rpc(args.port,dict(op="shutdown"));server.wait(timeout=60)
            state.update(server_exit_code=server.returncode,source_after=sources())
            if server.returncode or before!=state["source_after"]:raise RuntimeError("Server cleanup or frozen source guard failed")
            cleanup=json.loads((out/"cleanup.json").read_text())
            if not cleanup.get("passed"):raise RuntimeError("Green Context cleanup failed")
            state.update(status="passed",end_utc=now(),source_unchanged=True,cleanup=cleanup)
    except BaseException:
        state.update(status="technical_failure",traceback=traceback.format_exc(),end_utc=now())
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=15)
        if server is not None and server.poll() is None:
            try:rpc(args.port,dict(op="shutdown"));server.wait(timeout=30)
            except Exception:server.terminate();server.wait(timeout=15)
        raise
    finally:
        save(out/"execution.json",state)


if __name__ == "__main__":main()
