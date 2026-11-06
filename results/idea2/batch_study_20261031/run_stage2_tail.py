"""One finite stage2 queue: wait for idle GPUs, retain failures, never stop outsiders."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from preparations.idea2.batch_plan import source_hashes
from .wait_then_run import HERE, ROOT, candidates, now, save, snapshot


def specifications():
    jobs = []
    def add(key, batch, quota, *, kind="performance", input_mode="online", mode="green",
            placement="caller", depends=(), same_gpu_as=None, reverse=False):
        stage = "profile" if kind == "profile" else "performance"
        template_mode = "green_sm16" if mode == "green" else "paired"
        dependencies = list(dict.fromkeys([*depends, *([same_gpu_as] if same_gpu_as else [])]))
        jobs.append(dict(key=key, base_name="stage2_tail_" + key, batch=batch, quota=quota,
                         kind=kind, input_mode=input_mode, mode=mode, placement=placement,
                         template=f"{stage}_b{batch}_{template_mode}_{input_mode}",
                         depends=dependencies, same_gpu_as=same_gpu_as, reverse=reverse,
                         status="pending", attempts=[]))
    add("b2_sm24_dedicated", 2, 24, placement="dedicated")
    add("b1_sm16_dedicated", 1, 16, placement="dedicated")
    add("b4_sm20_dedicated", 4, 20, placement="dedicated")
    add("b2_sm16_dedicated", 2, 16, placement="dedicated", depends=("b2_sm24_dedicated",))
    add("b4_sm16_dedicated", 4, 16, placement="dedicated", depends=("b4_sm20_dedicated",))
    add("b1_full_online_reverse", 1, 16, mode="paired", reverse=True,
        depends=("b1_sm16_dedicated",), same_gpu_as="b1_sm16_dedicated")
    add("b4_sm20_prepared_reverse", 4, 20, input_mode="prepared")
    add("b4_sm24_prepared_reverse", 4, 24, input_mode="prepared",
        depends=("b4_sm20_prepared_reverse",), same_gpu_as="b4_sm20_prepared_reverse")
    for batch, quota in ((1, 16), (4, 20)):
        parent = f"b{batch}_sm{quota}_dedicated"
        add(f"b{batch}_green_continuous", batch, quota, kind="continuity", placement="dedicated",
            depends=(parent,))
        add(f"b{batch}_full_continuous", batch, 16, kind="continuity", mode="paired", reverse=True,
            depends=(parent,), same_gpu_as=f"b{batch}_green_continuous")
        add(f"b{batch}_dedicated_profile", batch, quota, kind="profile", placement="dedicated",
            depends=(parent,))
    return jobs


def latency_specifications():
    previous = json.loads((HERE / "stage2_tail_execution.json").read_text())
    if previous["status"] != "passed":
        raise RuntimeError("Finish the frozen tail queue before latency-boundary follow-up")
    index = {job["key"]: job for job in previous["jobs"]}
    jobs = []
    for batch, quota in ((1, 16), (2, 24), (4, 20)):
        parent = index[f"b{batch}_sm{quota}_dedicated"]
        preceding = None
        for label, mode, placement, disabled in (("caller_matched", "green", "caller", False),
                                                  ("fresh_full", "paired", "caller", True),
                                                  ("fresh_green", "green", "dedicated", True)):
            key = f"b{batch}_{label}"
            template_mode = "green_sm16" if mode == "green" else "paired"
            jobs.append(dict(key=key, base_name="stage2_latency_"+key, batch=batch, quota=quota,
                kind="performance", input_mode="online", mode=mode, placement=placement,
                disable_lookahead=disabled, template=f"performance_b{batch}_{template_mode}_online",
                depends=[preceding] if preceding else [], same_gpu_as=None,
                required_uuid=parent["gpu_uuid"], reverse=mode == "paired", status="pending", attempts=[]))
            preceding = key
    return jobs


def command(job, device, name):
    argv = ["python3", "-u", "-m", "results.idea2.batch_study_20261031.run_one",
            "--job", job["template"], "--gpu-uuid", device["uuid"], "--gpu-index", str(device["index"]),
            "--name", name, "--ae-sm-min", str(job["quota"])]
    if job["batch"] > 1:
        argv.append("--diagnostic-batch-drift")
    if job["placement"] != "caller":
        argv += ["--prepare-stream", job["placement"]]
    if job["reverse"]:
        argv.append("--reverse")
    if job.get("disable_lookahead"):
        argv.append("--disable-lookahead")
    if job["kind"] == "continuity":
        argv += ["--pairs", "2", "--length", "100"]
    return argv


def catalog_start(job, name):
    path = HERE / "study_catalog.json"
    data = json.loads(path.read_text())
    section = {"performance": "performance", "profile": "profiles", "continuity": "continuity"}[job["kind"]]
    row = dict(name=name, phase="stage2", eligibility="strict" if job["batch"] == 1 else "same_batch_diagnostic",
               batch=job["batch"], mode=job["mode"], input_mode=job["input_mode"],
               ae_sm_min=job["quota"], prepare_stream=job["placement"], queue_key=job["key"])
    row["lookahead_enabled"] = not job.get("disable_lookahead", False)
    data.setdefault(section, []).append(row)
    save(path, data)


def catalog_failure(job, name, kind):
    path = HERE / "study_catalog.json"
    data = json.loads(path.read_text())
    for section in ("performance", "profiles", "continuity"):
        data[section] = [row for row in data.get(section, []) if row["name"] != name]
    data["failures"].append(dict(name=name, kind=kind, queue_key=job["key"]))
    save(path, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-wait-hours", type=float, default=12)
    parser.add_argument("--phase", choices=("tail", "latency"), default="tail")
    args = parser.parse_args()
    jobs = specifications() if args.phase == "tail" else latency_specifications()
    if args.dry_run:
        for job in jobs:
            command(job, dict(uuid="GPU-UNASSIGNED", index=3), job["base_name"])
        print(json.dumps(dict(status="prepared", jobs=len(jobs), gpu_queried=False, launched=0)))
        return 0
    path = HERE / ("stage2_tail_execution.json" if args.phase == "tail" else "latency_boundary_execution.json")
    if path.exists():
        raise RuntimeError("Existing queue must be inspected; never restart from a stale observation")
    for job in jobs:
        if (HERE / job["base_name"]).exists() or (HERE / (job["base_name"] + ".execution.json")).exists():
            raise RuntimeError("An output already exists: " + job["base_name"])
    state = dict(status="waiting", pid=os.getpid(), start_utc=now(), poll_count=0,
                 source_before=source_hashes(), jobs=jobs, exports=[])
    path.write_text(json.dumps(state, indent=2)+"\n")
    deadline = time.monotonic() + args.max_wait_hours * 3600
    active = {}
    while True:
        for key, (child, log, job, name) in list(active.items()):
            if child.poll() is None:
                continue
            log.close()
            receipt = json.loads((HERE / (name + ".execution.json")).read_text())
            data_path = HERE / name / "result.json"
            data = json.loads(data_path.read_text()) if data_path.exists() else {}
            error = data.get("error") or receipt.get("error") or ""
            valid = (child.returncode == 0 and receipt.get("source_unchanged")
                     and data.get("status") in ("passed", "diagnostic_passed")
                     and data.get("source_unchanged"))
            job["attempts"][-1].update(exit_code=child.returncode, end_utc=now())
            if valid:
                job.update(status="passed", result_name=name)
                print("FINISHED", key, "passed", flush=True)
                if job["kind"] == "profile":
                    export_log = (HERE / (name + ".export.console.log")).open("x")
                    process = subprocess.Popen(["taskset", "-c", "48-63", "python3", "-u", "-m",
                        "results.idea2.batch_study_20261031.export_trace", name], cwd=ROOT,
                        stdout=export_log, stderr=subprocess.STDOUT)
                    state["exports"].append(dict(name=name, pid=process.pid, status="running"))
                    # Keep real Popen handles; a timeout never becomes a restart.
                    export_handles[name] = (process, export_log)
            else:
                admission = "GPU is not exclusive/idle" in error or "already has an active lease" in error
                interference = "GPU process occupancy changed" in error
                retry = (admission or interference) and len(job["attempts"]) < 3
                catalog_failure(job, name, "admission_rejected_external_occupancy" if admission else
                                "external_interference_invalidated" if interference else "runtime_or_numerical_failure")
                job.update(status="pending" if retry else "failed", last_error=error)
                print("FINISHED", key, "retry_on_release" if retry else "failed", flush=True)
            del active[key]
        for row in state["exports"]:
            if row["status"] != "running":
                continue
            child, log = export_handles[row["name"]]
            if child.poll() is not None:
                log.close()
                row.update(status="passed" if child.returncode == 0 else "failed", exit_code=child.returncode)
                print("EXPORTED", row["name"], row["status"], flush=True)
        by_key = {job["key"]: job for job in jobs}
        for job in jobs:
            if job["status"] == "pending" and any(by_key[key]["status"] in ("failed", "skipped_failed_dependency")
                                                    for key in job["depends"]):
                job["status"] = "skipped_failed_dependency"
        pending = [job for job in jobs if job["status"] == "pending"]
        if not pending and not active and not any(row["status"] == "running" for row in state["exports"]):
            state.update(status="passed" if all(job["status"] == "passed" for job in jobs)
                         and all(row["status"] == "passed" for row in state["exports"]) else "needs_inspection",
                         end_utc=now(), source_after=source_hashes())
            save(path, state)
            print("QUEUE_COMPLETE", state["status"], flush=True)
            return 0 if state["status"] == "passed" else 1
        if pending and len(active) < 3:
            state.update(last_poll_utc=now(), poll_count=state["poll_count"]+1)
            try:
                rows = snapshot()
                state["devices"] = rows
                free = candidates(rows)
                state.pop("last_poll_error", None)
            except Exception as error:
                free = []
                state["last_poll_error"] = str(error)
            busy = {entry[2]["gpu_uuid"] for entry in active.values()}
            free = [row for row in free if row["uuid"] not in busy]
            for job in pending:
                if len(active) >= 3:
                    break
                if any(by_key[key]["status"] != "passed" for key in job["depends"]):
                    continue
                same = job["same_gpu_as"]
                if same and by_key[same]["status"] != "passed":
                    continue
                required = job.get("required_uuid") or (by_key[same]["gpu_uuid"] if same else None)
                chosen = next((row for row in free if not required or row["uuid"] == required), None)
                if chosen is None:
                    continue
                name = job["base_name"] + (f"_admission_retry{len(job['attempts'])}" if job["attempts"] else "")
                argv = command(job, chosen, name)
                log = (HERE / (name + ".launcher.console.log")).open("x")
                child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job.update(status="running", gpu_uuid=chosen["uuid"], gpu_index=chosen["index"])
                job["attempts"].append(dict(name=name, pid=child.pid, command=argv, start_utc=now()))
                active[job["key"]] = (child, log, job, name)
                catalog_start(job, name)
                free.remove(chosen)
                print("STARTED", job["key"], chosen["index"], child.pid, flush=True)
        if time.monotonic() >= deadline and not active:
            state.update(status="wait_deadline", end_utc=now())
            save(path, state)
            return 2
        state["status"] = "running" if active else "waiting"
        save(path, state)
        time.sleep(30)


export_handles = {}

if __name__ == "__main__":
    raise SystemExit(main())
