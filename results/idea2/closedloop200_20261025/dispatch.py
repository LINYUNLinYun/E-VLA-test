"""Single task-chunk dispatcher; no inference, simulator or model changes.

GPU3 is enabled only after the explicit handoff marker is written. Each chunk
keeps all four arms on one physical device. Completed task failures are never
rescheduled. An interrupted chunk requires human-agent review, not auto retry.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parent
W=R.parents[2]
P=W/"reports/idea2_closedloop200_20261025"
LEASE=W/"results/idea1_v3_3/external_gpu_leases.json"
OWNER="user-authorized idea2 closedloop200"
DEVICES={2:(0,"GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25"),3:(1,"GPU-555cde36-5e41-bc44-0c69-ea47358fd703")}


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(path)


def identity(pid):
    try:
        stat=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        return stat[19] if stat[0]!="Z" else None
    except FileNotFoundError:return None


def release(gpu):
    with LEASE.open("r+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);data=json.load(stream)
        if data[str(gpu)].get("owner")==OWNER:
            data[str(gpu)].update(active=False,released_utc=utc(),release_note="Own chunk services exited; no further assigned work")
        stream.seek(0);json.dump(data,stream,indent=2);stream.truncate()


def claim(gpu):
    with LEASE.open("r+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);data=json.load(stream);row=data[str(gpu)]
        if row.get("active") and row.get("owner")!=OWNER:return False
        row.update(active=True,owner=OWNER,time_utc=utc(),purpose="200 paired cases x4 conditions; fixed task chunks, no GPU sharing")
        stream.seek(0);json.dump(data,stream,indent=2);stream.truncate()
    return True


def aggregate(state):
    merged={}
    reuse=json.loads((R/"reuse32.json").read_text())
    for old in reuse["selected_cases"]:
        arm=old.get("arm",old.get("label"));leaf=Path(old["source_directory"]);v=json.loads((leaf/"complete.json").read_text())
        key=(arm,old["suite"],old["task"],old["init"])
        merged[key]=dict(configuration=arm,label=arm,arm=arm,suite=key[1],task=key[2],init=key[3],case_id=old["case_id"],
                         status="valid",success=v["success"],steps=v["steps"],requests=v["predictions"],directory=str(leaf),source_kind="reused_original8",reuse_manifest=str(R/"reuse32.json"))
    for chunk in state["chunks"]:
        if "output" not in chunk:continue
        path=Path(chunk["output"])/"normalized_cases.json"
        if not path.exists():continue
        for row in json.loads(path.read_text()):
            arm=row.get("configuration",row.get("label",row.get("arm")))
            key=(arm,row["suite"],row["task"],row["init"])
            if row["status"]=="valid" or key not in merged:merged[key]=dict(row,configuration=arm,chunk=chunk["id"])
    rows=list(merged.values());save(R/"normalized_cases.json",dict(cases=rows))
    valid=[r for r in rows if r["status"]=="valid"]
    arms=["fresh18","sm16_tail0","sm32_tail0","sm32_tail2"]
    sets={arm:{(r["suite"],r["task"],r["init"]):r for r in valid if r["configuration"]==arm} for arm in arms}
    common=set.intersection(*(set(v) for v in sets.values()))
    progress=dict(status=state["status"],updated_utc=utc(),valid=len(valid),expected=800,new_valid=sum(r.get("source_kind")=="new" for r in valid),reused=32,
                  common_paired_cases=len(common),by_arm={arm:dict(valid=len(sets[arm]),successes=sum(r["success"] for r in sets[arm].values()),
                  paired_successes=sum(sets[arm][k]["success"] for k in common)) for arm in arms},
                  chunks=[dict(id=c["id"],gpu=c.get("gpu"),status=c["status"]) for c in state["chunks"]])
    save(P/"progress.json",progress)
    return progress


def main():
    P.mkdir(exist_ok=True,parents=True)
    existing=R/"dispatch_execution.json"
    if existing.exists():
        if "--resume-port-fix" not in sys.argv:raise RuntimeError("Refuse duplicate dispatcher")
        state=json.loads(existing.read_text());assert state["status"]=="needs_review"
        assert all(not identity(c["pid"]) for c in state["chunks"] if c.get("pid"))
        save(R/"dispatch_execution_before_port_fix.json",state)
        for chunk in state["chunks"]:
            if chunk["status"]!="failed":continue
            output=Path(chunk["output"])
            log=R/(output.name+".console.log")
            assert not (output/"execution.json").exists() and "[Errno 98] Address already in use" in log.read_text()
            chunk["previous_attempts"]=[dict(chunk)]
            chunk.update(status="pending",port_retry=1)
        state.update(status="running",pid=os.getpid(),resumed_utc=utc())
        first=state["chunks"][0]
        if (R/"initial_parent_exit.json").exists():first["parent_exit_code"]=json.loads((R/"initial_parent_exit.json").read_text())["exit_code"]
        running={}
    else:
        initial=json.loads((R/"initial_launch.json").read_text());assert len(initial["processes"])==1
        pid=initial["processes"][0]["pid"]
        state=dict(status="running",pid=os.getpid(),start_utc=utc(),chunks=[dict(id=t,task_ids=[t,10+t,20+t,30+t],status="pending") for t in range(10)])
        state["chunks"][0].update(status="running",gpu=2,output=str(R/"gpu2_chunk0"),pid=pid,start_tick=identity(pid),attached_tool_session=25329)
        running={2:(state["chunks"][0],None)}
    handles=[]
    env=dict(os.environ,CUDA_DEVICE_MAX_CONNECTIONS="32",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",JAX_PLATFORMS="cpu",JAX_PLATFORM_NAME="cpu",XLA_PYTHON_CLIENT_PREALLOCATE="false",PYTHONPATH=".:policies/openpi/src",OPENPI_DATA_HOME=str(W/"shared/checkpoints/openpi"))
    while True:
        for gpu,(chunk,process) in list(running.items()):
            live=process.poll() is None if process is not None else identity(chunk["pid"])==chunk["start_tick"] and chunk["start_tick"] is not None
            if live:continue
            receipt=Path(chunk["output"])/"execution.json"
            result=json.loads(receipt.read_text()) if receipt.exists() else {}
            code=process.returncode if process is not None else None
            passed=result.get("status")=="passed" and result.get("source_unchanged") and result.get("server_exit_code")==0 and (code==0 or process is None)
            chunk.update(status="passed" if passed else "failed",parent_exit_code=code,end_utc=utc(),source_unchanged=result.get("source_unchanged"),server_exit_code=result.get("server_exit_code"))
            if process is None:chunk["exit_evidence"]="Attached process exited; terminal result verified; direct tool-session exit code collected separately"
            del running[gpu];print("CHUNK",chunk["id"],chunk["status"],flush=True)
        failed=any(c["status"]=="failed" for c in state["chunks"])
        for gpu in DEVICES:
            if gpu in running or failed:continue
            if gpu==3 and not (R/"ENABLE_GPU3").exists():continue
            pending=next((c for c in state["chunks"] if c["status"]=="pending"),None)
            if pending is None:release(gpu);continue
            if not claim(gpu):continue
            shard,uuid=DEVICES[gpu];suffix="_port_retry1" if pending.get("port_retry") else ""
            output=R/f"gpu{gpu}_chunk{pending['id']}{suffix}"
            port=19120+pending["id"]*2+shard
            command=[str(W/"policies/openpi/.venv/bin/python"),"-u","-m","results.idea2.closedloop200_20261025.runner","--execute","--shard",str(shard),"--port",str(port),"--task-ids",*map(str,pending["task_ids"]),"--output",str(output)]
            log=(R/(output.name+".console.log")).open("x");handles.append(log)
            child=subprocess.Popen(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
            pending.update(status="running",gpu=gpu,gpu_uuid=uuid,output=str(output),command=command,pid=child.pid,start_utc=utc(),start_tick=identity(child.pid));running[gpu]=(pending,child)
            print("START",pending["id"],"GPU",gpu,flush=True)
        if all(c["status"]=="passed" for c in state["chunks"]):state["status"]="passed"
        elif failed and not running:state["status"]="needs_review"
        save(R/"dispatch_execution.json",state);progress=aggregate(state)
        print(utc(),"VALID",progress["valid"],"/800; paired",progress["common_paired_cases"],"live",[(g,c[0]["pid"]) for g,c in running.items()],flush=True)
        if state["status"]!="running":break
        time.sleep(20)
    for gpu in DEVICES:release(gpu)
    if state["status"]!="passed":raise RuntimeError("Chunk failure retained; remaining pending work requires review")
    command=[sys.executable,str(R/"statistics.py"),"--input",str(R/"normalized_cases.json"),"--output",str(P/"statistics.json")]
    # CLI contract is checked before this final stage; no inference is launched.
    save(R/"statistics_pending_command.json",dict(command=command))
    for h in handles:h.close()


if __name__=="__main__":main()
