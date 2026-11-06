"""Continue pending pairs on GPU3 and user-authorized idle GPU0."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from .runner_alt import R, ROOT, DEVICES, save
from preparations.idea2.layer_bench import require_idle_gpu, gpu_snapshot

OWNER = "user-authorized idea2 control alignment100"
LEASES = ROOT/"results/idea1_v3_3/external_gpu_leases.json"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def claim(gpu):
    try:
        snapshot = require_idle_gpu(DEVICES[gpu][0])
    except RuntimeError:
        return False
    with LEASES.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        leases = json.load(f)
        if leases.get(str(gpu), {}).get("active"):
            return False
        leases[str(gpu)] = dict(gpu=gpu, uuid=DEVICES[gpu][0], active=True, owner=OWNER,
            time_utc=now(), owner_pid=os.getpid(), purpose="paired unaligned/offset fixed100; model and control code frozen",
            idle_snapshot=snapshot)
        f.seek(0); json.dump(leases, f, indent=2); f.write("\n"); f.truncate()
    return True


def release(gpu):
    snapshot = gpu_snapshot(DEVICES[gpu][0])
    with LEASES.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        leases = json.load(f)
        if leases.get(str(gpu), {}).get("owner") == OWNER:
            leases[str(gpu)].update(active=False, released_utc=now(),
                release_note="Expanded study assigned model/EGL services exited", release_snapshot=snapshot)
            f.seek(0); json.dump(leases, f, indent=2); f.write("\n"); f.truncate()
    return snapshot


def key(row):
    return row.get("alignment", row.get("configuration")), row["suite"], row["task"], row["init"]


def main():
    dest = R/"dispatch_execution.json"
    if not dest.exists():
        raise ValueError("Continuation requires the original receipt")
    previous = json.loads(dest.read_text())
    for stage in previous["stages"]:
        if stage["gpu"] == 3:
            receipt = json.loads((Path(stage["output"])/"execution.json").read_text())
            if receipt["status"] != "passed" or not receipt["source_unchanged"] or receipt["server_exit_code"] != 0:
                raise ValueError("Wait for the original GPU3 stage to exit normally")
    historical_path = R/"dispatch_before_gpu0.json"
    if historical_path.exists():
        raise ValueError("Continuation already started")
    save(historical_path, previous)
    manifest = json.loads((R/"manifest.json").read_text())
    rows = json.loads((R/"normalized_cases.json").read_text())["cases"]
    bykey = {key(r):r for r in rows}
    pending = []
    for original in manifest["chunks"]:
        remaining = []
        for case in original["case_keys"]:
            suite, task, init = case.split("/")
            pair = [bykey[(m,suite,int(task),int(init))] for m in ("off","control_steps")]
            if all(r["status"] == "valid" for r in pair):
                continue
            if any(r["status"] == "valid" for r in pair):
                raise ValueError("Partial valid pair needs a separate audited continuation")
            remaining.append(case)
        if remaining:
            pending.append(dict(original,case_keys=remaining,episodes=2*len(remaining)))
    state = dict(status="running", start_utc=now(), pid=os.getpid(), stages=[dict(x) for x in previous["stages"]], cases=rows,
                 gpu_candidates=[3,0], releases=[], pending_chunks=[r["chunk_id"] for r in pending])
    active = {}
    leased = set()
    last_probe = {0:0.,3:0.}
    last_save = 0.

    def checkpoint():
        state["pending_chunks"] = [r["chunk_id"] for r in pending]
        state["active_gpus"] = sorted(active)
        state["valid_cells"] = sum(r["status"] == "valid" for r in rows)
        state["successes"] = {m:sum(r.get("success", False) for r in rows if r["configuration"] == m and r["status"] == "valid") for m in ("off", "control_steps")}
        state["last_update_utc"] = now()
        save(dest, state)
        save(R/"normalized_cases.json", dict(cases=rows))

    def merge(stage):
        p = Path(stage["output"])/"execution.json"
        if not p.exists(): return None
        receipt = json.loads(p.read_text())
        for row in receipt.get("cases", []):
            if row["status"] != "valid": continue
            target = bykey[key(row)]
            if target["status"] == "valid":
                if target["directory"] != row["directory"]:
                    raise ValueError("Duplicate valid case source")
                continue
            target.update(status="valid", success=row["result"]["success"], steps=row["result"]["steps"],
                directory=row["directory"], result=row["result"], gpu=stage["gpu"], source_execution=str(p),
                source_kind="expanded", client_exit_code=row["client_exit_code"], render_gpu_verified=row["render_gpu_verified"])
        return receipt

    try:
        for stage in state["stages"]:
            receipt = json.loads((Path(stage["output"])/"execution.json").read_text())
            if receipt["status"] == "passed":
                stage.update(status="passed",exit_code=0,server_exit_code=0,source_unchanged=True,end_utc=receipt["end_utc"])
        checkpoint()
        while pending or active:
            for gpu, job in list(active.items()):
                code = job["process"].poll()
                if code is None: continue
                job["log"].close()
                stage = job["stage"]
                receipt = merge(stage)
                stage.update(exit_code=code, end_utc=now())
                if code or receipt is None or receipt["status"] != "passed" or not receipt["source_unchanged"] or receipt["server_exit_code"] != 0:
                    stage["status"] = "technical_failure"
                    raise RuntimeError("Expanded stage failed; preserve all completed cells for diagnosis")
                stage.update(status="passed", source_unchanged=True, server_exit_code=0)
                del active[gpu]
                checkpoint()
                if not pending:
                    state["releases"].append(dict(gpu=gpu, time_utc=now(), snapshot=release(gpu)))
                    leased.remove(gpu)
            for gpu in (3,0):
                if not pending or gpu in active: continue
                if gpu not in leased:
                    if time.monotonic()-last_probe[gpu]<30: continue
                    last_probe[gpu] = time.monotonic()
                    if not claim(gpu): continue
                    leased.add(gpu)
                chunk = pending.pop(0)
                with socket.socket() as s:
                    s.bind(("127.0.0.1",0));port=s.getsockname()[1]
                out = R/f"chunk{chunk['chunk_id']:02d}_gpu{gpu}_resume2"
                command = [sys.executable,"-u","-m","results.idea2.control_alignment100_20261026.runner_alt",
                           "--output",str(out),"--port",str(port),"--gpu",str(gpu),"--case-keys",*chunk["case_keys"]]
                stage = dict(chunk_id=chunk["chunk_id"], gpu=gpu, output=str(out), command=command,
                             status="running", start_utc=now(), expected_episodes=chunk["episodes"])
                log = (R/(out.name+".console.log")).open("x")
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                state["stages"].append(stage)
                active[gpu] = dict(process=child, log=log, stage=stage)
                checkpoint()
            if time.monotonic()-last_save>=30:
                for job in active.values(): merge(job["stage"])
                checkpoint()
                print(json.dumps({k:state[k] for k in ("valid_cells","successes","active_gpus","pending_chunks","last_update_utc")}),flush=True)
                last_save=time.monotonic()
            time.sleep(2)
        if any(r["status"] != "valid" for r in rows) or len(rows) != 200:
            raise RuntimeError("Expanded grid incomplete")
        state.update(status="passed", end_utc=now(), pending=[])
        checkpoint()
    except BaseException:
        state.update(status="technical_failure", error=traceback.format_exc(), end_utc=now())
        checkpoint()
        # Allow already-running sibling stages to exit gracefully and retain
        # their valid results. Do not launch further jobs after a guard failure.
        for gpu, job in active.items():
            job["process"].wait()
            job["log"].close()
            merge(job["stage"])
        raise
    finally:
        for gpu in leased:
            state["releases"].append(dict(gpu=gpu, time_utc=now(), snapshot=release(gpu)))
        checkpoint()


if __name__ == "__main__":
    main()
