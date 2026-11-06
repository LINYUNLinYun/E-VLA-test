"""Bounded read-only observer: no GPU queries, helper calls, or process signals."""
import argparse
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT/"results/ease_insights_20261103"
COORDINATOR = RESULTS/"gate_opportunity_coordinator"
OUT = Path(__file__).resolve().parent
EXPECTED_PID = 356386
EXPECTED_TICKS = "2617020452"


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def identity(pid):
    if not isinstance(pid, int):
        return None
    path = Path("/proc")/str(pid)
    try:
        first = (path/"stat").read_text().rsplit(")",1)[1].split()
        argv = [s.decode(errors="replace") for s in (path/"cmdline").read_bytes().split(b"\0") if s]
        last = (path/"stat").read_text().rsplit(")",1)[1].split()
        return dict(pid=pid,state=last[0],ppid=int(last[1]),start_ticks=last[19],argv=argv,
            stable_start_ticks=first[19]==last[19],
            cpu_ticks=dict(zip(("user","system","child_user","child_system"),map(int,last[11:15]))),
            wait_channel=(path/"wchan").read_text().strip())
    except FileNotFoundError:
        return dict(pid=pid,state="absent")


def snapshot():
    state = read(COORDINATOR/"state.json", {})
    status = read(COORDINATOR/"status.json", {})
    process = identity(EXPECTED_PID)
    events_path = COORDINATOR/"events.jsonl"
    events = [json.loads(s) for s in events_path.read_text().splitlines()] if events_path.exists() else []
    audits = sorted({e["pending_audit"] for e in events if e.get("pending_audit")})
    if state.get("pending"):
        audits = sorted(set(audits)|{state["pending"]["audit"]})
    transactions = []
    states = read(RESULTS/"scheduler/jobs_state.json", {})
    for filename in audits:
        directory = Path(filename)
        if directory.parent != RESULTS/"gate_reassignments":
            raise ValueError("Unexpected audit root")
        audit = read(directory/"audit.json")
        if audit is None:
            transactions.append(dict(path=str(directory),stage="not_yet_written")); continue
        old = audit.get("gate", {})
        new_id = audit.get("new_job", {}).get("id")
        new_state = states.get(new_id, {})
        transactions.append(dict(path=str(directory),stage=audit.get("stage"),
            updated_at=audit.get("updated_at"), committed_at=audit.get("committed_at"),
            commit_outcome=audit.get("commit_outcome"), old_job_id=audit.get("old_job_id"), new_job_id=new_id,
            old_gate_now=identity(old.get("pid")),new_scheduler_state=new_state,
            new_process_now=identity(new_state.get("pid")),error=audit.get("error"),
            audit_sha256=hashlib.sha256((directory/"audit.json").read_bytes()).hexdigest()))
    return dict(observed_unix=time.time(),coordinator=process,
        coordinator_identity_matches=process.get("start_ticks")==EXPECTED_TICKS and process.get("stable_start_ticks") is True,
        phase=state.get("phase"),pending=state.get("pending"),status=status,
        controllers={k:identity(v["pid"]) for k,v in state.get("controllers",{}).items()},
        event_count=len(events),events_tail=events[-8:],transactions=transactions)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--deadline-unix",type=float,required=True)
    args = parser.parse_args()
    start = time.time(); next_print = 0; previous = None; pending_started = None
    receipt = dict(observer_started_unix=start,deadline_unix=args.deadline_unix,
        coordinator_pid=EXPECTED_PID,coordinator_start_ticks=EXPECTED_TICKS,
        gpu_queries=False,helper_invocations=False,signals=False,queue_or_scheduler_state_writes=False)
    (OUT/"observation_start.json").write_text(json.dumps(receipt,indent=2)+"\n")
    reason = "verified_wait_deadline"
    while True:
        try:
            row = snapshot()
        except Exception as error:
            row = dict(observed_unix=time.time(),observer_error=repr(error))
            reason = "observation_error"
        with (OUT/"observations.jsonl").open("a") as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        now = time.time()
        signature = (row.get("phase"),row.get("event_count"),tuple((a.get("path"),a.get("stage"),a.get("new_scheduler_state",{}).get("status")) for a in row.get("transactions",[])))
        if signature != previous or now>=next_print or "observer_error" in row:
            print(json.dumps(row,ensure_ascii=False),flush=True)
            previous=signature;next_print=now+45
        if "observer_error" in row:
            break
        if not row["coordinator_identity_matches"]:
            reason="coordinator_identity_changed_or_exited";break
        if row["phase"]=="needs_attention":
            reason="needs_attention";break
        if row["pending"]:
            pending_started = pending_started or now
            if now-pending_started>90:
                reason="pending_exceeds_90_seconds";break
        else:
            pending_started=None
        if any(a.get("stage")=="complete" for a in row["transactions"]):
            reason="first_completed_transaction";break
        if now>=args.deadline_unix:
            break
        time.sleep(min(15,max(0,args.deadline_unix-now)))
    result=dict(receipt,observer_finished_unix=time.time(),reason=reason,last_snapshot=row)
    (OUT/"observation_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(observer_finished=True,reason=reason,output=str(OUT)),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
