"""Finish only nine unexecuted/interrupted cells after foreign GPU0 intrusion."""
import datetime as dt
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from .dispatch_gpu0 import claim, release, key
from .runner import R, ROOT, save


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    path = R/"dispatch_execution.json"
    old = json.loads(path.read_text())
    pid = old["pid"]
    deadline = time.monotonic()+900
    while Path(f"/proc/{pid}/cmdline").exists():
        try: command = Path(f"/proc/{pid}/cmdline").read_bytes()
        except FileNotFoundError: break
        if b"control_alignment100_20261026.dispatch_gpu0" not in command: break
        if time.monotonic()>deadline: raise TimeoutError("Prior GPU3 stage did not finish")
        time.sleep(2)
    old = json.loads(path.read_text())
    save(R/"dispatch_before_final_cells.json",old)
    rows = json.loads((R/"normalized_cases.json").read_text())["cases"]
    index = {key(r):r for r in rows}
    pending = ["/".join(map(str,key(r))) for r in rows if r["status"] != "valid"]
    expected = json.loads((R/"pending_final_cells.json").read_text())["cells"]
    if set(pending) != set(expected): raise ValueError("Unexpected pending cells; retain every valid case")
    cross = [["libero_spatial",9,1]]
    save(R/"resource_final_revision.json",dict(created_utc=now(),reason="foreign GPU0 process appeared during a case; guard stopped own client/model",
        failed_stage="chunk09_gpu0_resume2",preserved_valid_cells=7,remaining_cells=pending,
        destination_gpu=3,cross_gpu_pairs=cross,
        rule="Only interrupted/unstarted cells rerun; one valid GPU0 partner retained; report same-GPU99-pair sensitivity",
        model_control_statistics_functions_unchanged=True))
    save(R/"cross_gpu_exception.json",dict(pairs=cross,reason="User-authorized migration to idle GPU3 after foreign GPU0 intrusion; valid partner retained",
        require_bootstrap_tolerance=True,report_excluding_exception=True))
    while not claim(3): time.sleep(30)
    state = dict(old,status="running",continuation="finish_pending",error=None,pid=__import__("os").getpid(),cases=rows,active_gpus=[3])
    for stage in state["stages"]:
        receipt = json.loads((Path(stage["output"])/"execution.json").read_text())
        if receipt["status"] == "passed":
            stage.update(status="passed",exit_code=0,server_exit_code=0,source_unchanged=True,end_utc=receipt["end_utc"])
    with socket.socket() as s:
        s.bind(("127.0.0.1",0));port=s.getsockname()[1]
    out = R/"final9_gpu3"
    command = [sys.executable,"-u","-m","results.idea2.control_alignment100_20261026.runner_finish",
               "--gpu","3","--port",str(port),"--output",str(out),"--cell-keys",*expected]
    stage = dict(chunk_id="final9",gpu=3,output=str(out),command=command,status="running",start_utc=now(),expected_episodes=9)
    state["stages"].append(stage)
    def checkpoint():
        state["valid_cells"] = sum(r["status"] == "valid" for r in rows)
        state["successes"] = {m:sum(r.get("success",False) for r in rows if r["configuration"]==m and r["status"]=="valid") for m in ("off","control_steps")}
        state["last_update_utc"] = now()
        save(path,state);save(R/"normalized_cases.json",dict(cases=rows))
    def merge():
        p = out/"execution.json"
        if not p.exists(): return None
        receipt = json.loads(p.read_text())
        for r in receipt["cases"]:
            if r["status"] != "valid": continue
            target = index[key(r)]
            if target["status"] == "valid":
                if target["directory"]!=r["directory"]:raise ValueError("Valid data overwrite")
                continue
            target.update(status="valid",success=r["result"]["success"],steps=r["result"]["steps"],directory=r["directory"],
                result=r["result"],gpu=3,source_execution=str(p),source_kind="expanded",client_exit_code=0,render_gpu_verified=r["render_gpu_verified"])
        return receipt
    checkpoint()
    try:
        with (R/"final9_gpu3.console.log").open("x") as log:
            process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            stage["pid"] = process.pid;checkpoint()
            while process.poll() is None:
                time.sleep(10);merge();checkpoint()
        receipt = merge()
        stage.update(exit_code=process.returncode,end_utc=now())
        if process.returncode or receipt["status"] != "passed" or not receipt["source_unchanged"] or receipt["server_exit_code"] != 0:
            raise RuntimeError("Final-cell continuation failed; do not retry valid results")
        stage.update(status="passed",source_unchanged=True,server_exit_code=0)
        if len(rows)!=200 or any(r["status"]!="valid" for r in rows):raise ValueError("Incomplete final grid")
        state.update(status="passed",end_utc=now(),pending=[],active_gpus=[])
    except BaseException:
        state.update(status="technical_failure",error=__import__("traceback").format_exc())
        raise
    finally:
        state["releases"].append(dict(gpu=3,time_utc=now(),snapshot=release(3)))
        checkpoint()


if __name__ == "__main__": main()
