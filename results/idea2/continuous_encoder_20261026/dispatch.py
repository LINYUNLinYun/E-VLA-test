"""One dispatcher owns the remaining unique task chunks on two leased GPUs."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .run import ARMS,DEFAULTS,ROOT,R,save,universe

P=ROOT/"reports/idea2_continuous_encoder_20261026"
GATES=[R/"gate_gpu2_v3",R/"gate_gpu3_v3"]
OWNER="user-authorized idea2 continuous closedloop100"


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def aggregate(state):
    selected={};directory_gpu={}
    for path in GATES+[Path(c["output"]) for c in state["chunks"] if "output" in c]:
        f=path/"execution.json"
        if not f.exists():continue
        value=json.loads(f.read_text())
        for row in value["cases"]:
            if row["status"]!="valid":continue
            key=row["case_key"]
            if key in selected and selected[key]["directory"]!=row["directory"]:raise ValueError("Duplicate valid episode sources")
            directory=row["directory"]
            if row["source_kind"]!="completed_gate":directory_gpu[directory]=value["gpu_uuid"]
            selected[key]=row
    normalized=[]
    for row in universe():
        source=selected.get(row["case_key"])
        if source is None:normalized.append(dict(row,status="pending",success=None));continue
        result=source["result"]
        normalized.append(dict(row,status="valid",success=result["success"],steps=result["steps"],
            directory=source["directory"],gpu_uuid=directory_gpu[source["directory"]],
            predictions=result["predictions"],consumed_chunks=result["consumed_chunks"],
            actor_wall_s=result["actor_wall_s"],source_kind=source["source_kind"]))
    save(R/"normalized_cases.json",normalized)
    valid=[r for r in normalized if r["status"]=="valid"]
    case_sets=[{(r["suite"],r["task"],r["init"]) for r in valid if r["configuration"]==a} for a in ARMS]
    common=set.intersection(*case_sets)
    progress=dict(status=state["status"],updated_utc=now(),valid=len(valid),expected=300,common_paired_cases=len(common),
        by_arm={a:dict(valid=sum(r["configuration"]==a for r in valid),
                      successes=sum(r["success"] for r in valid if r["configuration"]==a),
                      paired_successes=sum(r["success"] for r in valid if r["configuration"]==a and (r["suite"],r["task"],r["init"]) in common)) for a in ARMS},
        chunks=[{k:c.get(k) for k in ["id","gpu","status"]} for c in state["chunks"]])
    save(P/"progress.json",progress)
    return len(valid)


def release(shard):
    gpu=2+shard;p=ROOT/"results/idea1_v3_3/external_gpu_leases.json"
    with p.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);value=json.loads(p.read_text());row=value[str(gpu)]
        if row.get("owner")==OWNER and row.get("active"):
            row.update(active=False,released_utc=now(),release_note="Continuous closedloop worker services exited; no remaining assigned tasks")
            save(p,value)


def main():
    for path in GATES:
        s=json.loads((path/"execution.json").read_text())
        if s["status"]!="passed" or not s["source_unchanged"] or s["server_exit_code"]!=0:raise ValueError("Both protocol gates must pass before dispatch")
    if (R/"dispatch_execution.json").exists():raise ValueError("Existing dispatcher state: inspect rather than duplicate work")
    P.mkdir(parents=True,exist_ok=True)
    save(R/"case_universe.json",universe())
    state=dict(status="running",pid=os.getpid(),start_utc=now(),chunks=[dict(id=t,task_ids=[t,10+t,20+t,30+t],status="pending") for t in range(10)])
    active={};logs={};released=set();failed=False
    def checkpoint():save(R/"dispatch_execution.json",state);return aggregate(state)
    checkpoint()
    while True:
        for shard,process in list(active.items()):
            if process.poll() is None:continue
            chunk=next(c for c in state["chunks"] if c.get("pid")==process.pid)
            logs.pop(shard).close();active.pop(shard)
            receipt=Path(chunk["output"])/"execution.json"
            value=json.loads(receipt.read_text()) if receipt.exists() else {}
            passed=process.returncode==0 and value.get("status")=="passed" and value.get("source_unchanged") and value.get("server_exit_code")==0
            chunk.update(status="passed" if passed else "failed",parent_exit_code=process.returncode,end_utc=now(),
                         server_exit_code=value.get("server_exit_code"),source_unchanged=value.get("source_unchanged"))
            if not passed:failed=True
            checkpoint()
        if not failed:
            for shard in (0,1):
                if shard in active:continue
                pending=next((c for c in state["chunks"] if c["status"]=="pending"),None)
                if pending is None:
                    if shard not in released:release(shard);released.add(shard)
                    continue
                target=R/f"gpu{2+shard}_chunk{pending['id']}"
                command=["python3","-u","-m","results.idea2.continuous_encoder_20261026.run",
                         "--output",str(target),"--shard",str(shard),"--port",str(19400+2*pending["id"]+shard),
                         "--task-ids",*[str(t) for t in pending["task_ids"]],"--completed-runs",*[str(x) for x in GATES]]
                log=target.with_suffix(".console.log").open("x")
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                active[shard]=process;logs[shard]=log
                pending.update(status="running",gpu=2+shard,gpu_uuid=DEFAULTS[shard]["gpu_uuid"],
                               output=str(target),command=command,pid=process.pid,start_utc=now())
                checkpoint()
        if not active:
            if failed:state.update(status="needs_review",end_utc=now());checkpoint();return
            count=checkpoint()
            if count!=300:raise RuntimeError(f"Completed chunks but fixed grid only{count}/300")
            state.update(status="passed",end_utc=now());checkpoint()
            for shard in (0,1):release(shard)
            return
        time.sleep(20)
        checkpoint()


if __name__=="__main__":main()
