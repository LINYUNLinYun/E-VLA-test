"""Independently verify the first live reassignment; no helper or GPU calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT/"results/ease_insights_20261103"
OUT = Path(__file__).resolve().parent
TRANSACTION = RESULTS/"gate_reassignments/opportunity_1793793271326258_000001"
GATE = "preparations.ease_insights_20261103.await_assigned_gpu"
RUNNER = "preparations.ease_insights_20261103.vlm_local_search_v1.runner"
SCHEDULER = "preparations.ease_insights_20261103.scheduler"
HELPER_SHA = "0238d949fd5585fc52dae19fc56c54b5b970cb4542714c4599024418c8169498"
COORDINATOR_SHA = "70ce6ef8a86153ad1f3cc6ffe0c4d05cc7e2d2dd334423ceb3af9598a7207432"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def identity(pid):
    path = Path("/proc")/str(pid)
    try:
        before = (path/"stat").read_text().rsplit(")",1)[1].split()
        argv = [x.decode(errors="replace") for x in (path/"cmdline").read_bytes().split(b"\0") if x]
        after = (path/"stat").read_text().rsplit(")",1)[1].split()
        assert before[19] == after[19]
        return dict(pid=pid,state=after[0],ppid=int(after[1]),start_ticks=after[19],argv=argv,
            children=(path/"task"/str(pid)/"children").read_text().split())
    except FileNotFoundError:
        return dict(pid=pid,state="absent")


def module(argv):
    return argv[argv.index("-m")+1] if "-m" in argv else None


def split_gate(argv):
    assert module(argv) == GATE
    separator = argv.index("--")
    receipt_index = argv.index("--status-file")+1
    return receipt_index,argv[separator+1:]


def main():
    observed = time.time()
    audit_path = TRANSACTION/"audit.json"
    audit = load(audit_path)
    before, proposed = load(TRANSACTION/"queue.before.json"),load(TRANSACTION/"queue.proposed.json")
    assert audit["stage"] == "complete" and audit["gate_exited"] is True
    assert sha(TRANSACTION/"queue.before.json") == audit["queue_before_sha256"]
    assert sha(TRANSACTION/"queue.proposed.json") == audit["queue_after_sha256"]
    old,new = audit["original_job"],audit["new_job"]
    old_slot,old_inner = split_gate(old["argv"])
    new_slot,new_inner = split_gate(new["argv"])
    assert old_inner == new_inner == old["launch_gate"]["original_argv"]
    assert module(old_inner) == RUNNER
    expected_job = copy.deepcopy(old)
    expected_job["id"] = new["id"]
    expected_job["argv"][old_slot] = new["argv"][new_slot]
    assert expected_job == new and old["complete_file"] == new["complete_file"]
    expected = copy.deepcopy(before)
    expected["jobs"] = [copy.deepcopy(new) if job["id"]==old["id"] else job for job in expected["jobs"]]
    dependency_changes = []
    for job in expected["jobs"]:
        if "depends_on" in job:
            original = list(job["depends_on"])
            job["depends_on"] = [new["id"] if dep==old["id"] else dep for dep in original]
            if original != job["depends_on"]:
                dependency_changes.append(dict(job=job["id"],before=original,after=job["depends_on"]))
    assert expected == proposed
    ids = [job["id"] for job in proposed["jobs"]]
    assert len(ids)==len(set(ids)) and old["id"] not in ids and new["id"] in ids
    assert all(set(job.get("depends_on",[]))<=set(ids) for job in proposed["jobs"])
    manifest_path = Path(audit["profile"]["manifest_path"])
    manifest = load(manifest_path); unhashed = dict(manifest); given = unhashed.pop("manifest_hash")
    assert digest(unhashed)==given==old["manifest_hash"]==new["manifest_hash"]==audit["profile"]["manifest_hash"]
    assert sha(manifest_path)==audit["profile"]["manifest_sha256"]
    source_checks = {name:sha(ROOT/name)==value for name,value in manifest["source_files"].items()}
    analysis_source_checks = {name:sha(ROOT/name)==value for name,value in manifest.get("analysis_sources",{}).items()}
    assert all(source_checks.values()) and all(analysis_source_checks.values())
    assert sha(ROOT/"preparations/ease_insights_20261103/reassign_waiting_gate_v2.py")==HELPER_SHA
    assert sha(ROOT/"preparations/ease_insights_20261103/gate_opportunity_coordinator.py")==COORDINATOR_SHA
    assert module(audit["gate"]["argv"])==GATE and not audit["gate"]["children"] and audit["gate"]["sigterm_default"]
    assert audit["stop_authorization"]["gate_argv"]==audit["gate"]["argv"]
    assert audit["gate"]["ppid"]==audit["scheduler"]["pid"]
    baseline = load(OUT/"baseline_running_jobs.json")
    previous = next(row for row in baseline["running_jobs"] if row["job_id"]==old["id"])
    assert previous["actual_process"]["pid"]==audit["gate"]["pid"]
    assert previous["actual_process"]["start_ticks"]==audit["gate"]["start_ticks"]
    assert previous["actual_process"]["argv"]==audit["gate"]["argv"]
    stages = {row["stage"]:row["at"] for row in audit["events"]}
    assert audit["scheduler_ack_job"]==new["id"] and audit["supervisor_profile"]=="local_search"
    assert stages["stop_intent"]<=stages["stopped_verified"]<audit["committed_at"]
    assert audit["committed_at"]<audit["supervisor_ack_time"]<=stages["controllers_acknowledged"]<stages["terminating_cpu_gate"]<stages["complete"]
    old_now = identity(audit["gate"]["pid"])
    assert old_now["state"] in ("absent","Z","X","x") or old_now["start_ticks"]!=audit["gate"]["start_ticks"]
    states = load(RESULTS/"scheduler/jobs_state.json")
    old_state,new_state = states[old["id"]],states[new["id"]]
    assert old_state["returncode"]==-15 and old_state["finished"]>stages["terminating_cpu_gate"]
    new_now = identity(new_state["pid"])
    assert new_state["status"]=="running" and new_now["ppid"]==audit["scheduler"]["pid"]
    assert new_state["started"]>=old_state["finished"]
    _,concrete_inner = split_gate(new_state["argv"])
    assert new_now["argv"] in (new_state["argv"],concrete_inner)
    current_queue = load(RESULTS/"queue.json")
    assert [j for j in current_queue["jobs"] if j["id"]==new["id"]]==[new]
    assert all(j["id"]!=old["id"] for j in current_queue["jobs"])
    matching = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit(): continue
        try:
            argv=[x.decode(errors="replace") for x in (directory/"cmdline").read_bytes().split(b"\0") if x]
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if module(argv)==SCHEDULER and "--queue" in argv and argv[argv.index("--queue")+1]==str(RESULTS/"queue.json"):
            matching.append(identity(int(directory.name)))
    assert len(matching)==1 and matching[0]["pid"]==audit["scheduler"]["pid"]
    controller_now = {name:identity(audit[name]["pid"]) for name in ("scheduler","supervisor")}
    for name,actual in controller_now.items():
        expected = audit[name]
        assert actual["start_ticks"]==expected["start_ticks"]
        assert actual["argv"]==expected.get("_scheduler_proof",{}).get("raw_argv",expected["argv"])
    coordinator_state = load(RESULTS/"gate_opportunity_coordinator/state.json")
    gate_status = load(RESULTS/"scheduler/launch_gates"/(new["id"]+".json"))
    report = dict(schema="ease-first-real-gate-reassignment-independent-audit-v1",observed_unix=observed,
        verdict="queue_migration_and_scheduler_continuation_verified",
        actual_model_runner_started=new_now["argv"]==concrete_inner,
        current_execution_kind="model_runner" if new_now["argv"]==concrete_inner else "cpu_assigned_gpu_wait_gate",
        scientific_job_equivalent_except_id_and_gate_receipt=True,complete_file_unchanged=True,
        queue_changes_exactly_replacement_and_dependency_renames=True,dependency_changes=dependency_changes,
        unchanged_other_job_count=len(before["jobs"])-1,queue_job_count=len(before["jobs"]),
        current_queue_equals_proposal=current_queue==proposed,
        manifest_hash=given,manifest_file_sha256=sha(manifest_path),source_file_checks=source_checks,
        analysis_source_checks=analysis_source_checks,full_condition=next(c for c in manifest["conditions"] if c["id"]=="full_d10"),
        old_cpu_gate_independently_observed_before_transaction=True,old_gate_snapshot=audit["gate"],old_process_now=old_now,
        dual_ack_record=dict(scheduler_job=audit["scheduler_ack_job"],supervisor_tick=audit["supervisor_ack_time"],
            committed_at=audit["committed_at"],acknowledged_at=stages["controllers_acknowledged"],termination_intent_at=stages["terminating_cpu_gate"]),
        transaction_events=audit["events"],original_scheduler_state=old_state,new_scheduler_state=new_state,new_process_now=new_now,
        matching_live_schedulers=matching,controller_identities_now=controller_now,
        new_gate_status=gate_status,coordinator_phase=coordinator_state["phase"],coordinator_pending=coordinator_state["pending"],
        source_gpu_uuid=audit["source_uuid"],opportunity_gpu_uuid=audit["request"]["candidate_gpu_uuid"],
        actual_new_gpu_uuid=new_state["gpu_uuid"],gpu_indices=dict(old=old_state["gpu"],new=new_state["gpu"]),
        qualification="Queue handoff is verified. A running scheduler job may still be a CPU gate; only actual argv distinguishes runner startup. No throughput or success claim follows.",
        evidence_scope="Dual acknowledgements are recorded in the pinned helper audit and corroborated by unchanged live controller identities and scheduler handoff. Historical jobs_state/status generations are not independently retained.",
        observer_helper_calls=False,observer_gpu_queries=False,observer_signals=False,observer_queue_or_runtime_writes=False,
        evidence={str(path):sha(path) for path in (audit_path,TRANSACTION/"queue.before.json",TRANSACTION/"queue.proposed.json",OUT/"baseline_running_jobs.json")},
        auditor=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)))
    destination = OUT/"first_transaction_1793793271326258.json"
    with destination.open("x") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write("\n")
    print(json.dumps(dict(report=str(destination),verdict=report["verdict"],actual_model_runner_started=report["actual_model_runner_started"],
        current_execution_kind=report["current_execution_kind"],old_pid=audit["gate"]["pid"],new_pid=new_state["pid"],
        old_gpu=old_state["gpu"],new_gpu=new_state["gpu"],source_files_verified=len(source_checks),analysis_sources_verified=len(analysis_source_checks),
        unchanged_other_jobs=report["unchanged_other_job_count"],pending=coordinator_state["pending"]),ensure_ascii=False))


if __name__=="__main__":
    main()
