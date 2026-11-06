"""CPU continuation for one bounded local search, sharing the existing allocator.

No GPU query/allocation and no process signals occur here. Technical errors stop
this supervisor for inspection; a valid unsuccessful policy rollout is retained.
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from preparations.ease_insights_20261103.pipeline import (
    Paths as SchedulerPaths, SchedulerWatch, append_jobs, argv_value, file_lock,
    marker_status, proc_identity, read, scheduler_match, sha, with_assigned_gpu_wait, write,
)
from .common import ROOT, MODEL_PYTHON, MODULE, read_manifest
from .selection import check_scope, read_selection

TERMINAL = {"report_ready", "no_candidate", "needs_attention"}


@dataclass
class Paths:
    discovery: Path
    confirmation: Path
    queue: Path
    scheduler_output: Path
    output: Path
    @classmethod
    def defaults(cls, root, confirmation_root=None, queue=None, scheduler_output=None, output=None):
        discovery = Path(root).resolve()
        global_root = ROOT / "results/ease_insights_20261103"
        queue = Path(queue or global_root / "queue.json").resolve()
        return cls(discovery, Path(confirmation_root or discovery.parent / "confirmation").resolve(), queue,
            Path(scheduler_output or queue.parent / "scheduler").resolve(),
            Path(output or discovery.parent / "supervisor").resolve())


class SharedSchedulerWatch(SchedulerWatch):
    """Also adopt the original pipeline's freshly written shared restart receipt."""
    def observe(self, state):
        receipt = read(self.paths.scheduler_output / "controller_process.json", {})
        continuation = read(self.paths.root / "continuation.json", {})
        restart = read(self.paths.output / "scheduler_restart.json", {})
        pids = {int(p) for p in (state.get("pid"), self.initial_pid, receipt.get("pid"),
            continuation.get("scheduler_pid"), restart.get("pid")) if p}
        processes = [self.probe(pid) for pid in sorted(pids)]
        matching = [p for p in processes if p["status"] == "live" and scheduler_match(p, self.paths.queue, self.paths.scheduler_output)]
        if len(matching) > 1:
            return dict(status="needs_attention", reason="multiple_matching_live_schedulers", processes=processes)
        if matching:
            process = matching[0]
            if (int(argv_value(process["argv"], "--max-gpus", "2")) != 2 or
                    float(argv_value(process["argv"], "--idle-seconds", "25")) != 0):
                return dict(status="needs_attention", reason="live_scheduler_configuration_differs", processes=processes)
            return dict(status="live", process=process, processes=processes)
        if any(p["status"] == "unknown" for p in processes):
            return dict(status="needs_attention", reason="process_observation_uncertain_no_restart", processes=processes)
        return dict(status="exited", processes=processes)

    def _spawn(self, argv, log_path):
        # SchedulerWatch.ensure holds the SAME global pipeline restart lock here.
        # Wait until its own identity receipt exists before releasing that lock,
        # so the original pipeline can observe/adopt this restart as well.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as stream:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            receipt = read(self.paths.scheduler_output / "controller_process.json", {})
            identity = self.probe(process.pid)
            if receipt.get("pid") == process.pid and identity["status"] == "live" and scheduler_match(identity, self.paths.queue, self.paths.scheduler_output):
                return process
            if identity["status"] in ("absent", "dead"):
                raise RuntimeError("Scheduler restart exited before identity registration; inspect logs")
            time.sleep(.05)
        raise RuntimeError("Scheduler restart identity not confirmed; process preserved for inspection, no retry")


def jobs_for(root, manifest):
    stage = "discovery" if manifest["phase"] == "local_search_discovery_v1" else "confirm"
    root = Path(root).resolve()
    return [dict(id="vlm_local_v1_%s_%s_all%03d" % (stage, cid, manifest["cases_per_condition"]),
        argv=[str(MODEL_PYTHON), "-m", MODULE + ".runner", "--output", str(root), "--condition", cid,
            "--gpu-uuid", "{gpu_uuid}", "--port", "{port}", "--launch"],
        complete_file=str(root / cid / (stage + "_shard00of01_complete.json")),
        depends_on=[], ready_files=[str(root / "manifest.json")], expected_episodes=manifest["cases_per_condition"],
        scientific_phase=manifest["phase"], manifest_hash=manifest["manifest_hash"])
        for cid in manifest["planned_conditions"]]


class Supervisor:
    def __init__(self, paths, scheduler_pid=None, command_runner=None, scheduler_watch=None):
        self.paths = paths
        self.state_path = paths.output / "state.json"
        self.state = read(self.state_path, dict(revision="ease-vlm-local-search-supervisor-v1",
            phase="waiting_discovery", created_unix=time.time(), actions={}, scheduler={}))
        self.command_runner = command_runner
        # Reuse the original pipeline's restart lock, not a second lock under
        # this supervisor's output. Only scheduler.py owns GPU leases/state.
        watch_paths = SchedulerPaths.defaults(paths.queue.parent, queue=paths.queue,
            scheduler_output=paths.scheduler_output, output=paths.queue.parent / "pipeline")
        self.watch = scheduler_watch or SharedSchedulerWatch(watch_paths, scheduler_pid)

    def save(self):
        self.state.update(updated_unix=time.time(), goal_complete_claim=False)
        write(self.state_path, self.state)

    def run_action(self, label, argv):
        actions = self.state.setdefault("actions", {})
        if actions.get(label, {}).get("status") == "complete":
            return
        if label in actions:
            raise RuntimeError("Prior CPU action did not complete; inspect before retry: " + label)
        log = self.paths.output / "logs" / (label + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        actions[label] = dict(status="starting", argv=argv, started_unix=time.time(), log=str(log))
        self.save()
        if self.command_runner:
            code = self.command_runner(argv, log)
        else:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            with log.open("w") as stream:
                process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, env=env)
                actions[label].update(status="running", pid=process.pid)
                self.save()
                try:
                    code = process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("CPU analysis exceeded observation bound; process preserved, no automatic retry")
        actions[label].update(status="complete" if code == 0 else "failed", returncode=code, finished_unix=time.time())
        self.save()
        if code != 0:
            raise RuntimeError("CPU stage failed, no automatic retry: " + label)

    def completion_gate(self, root, stage, queue, job_state):
        manifest = read_manifest(Path(root) / "manifest.json", verify_sources=True)
        check_scope(root, manifest, stage)
        jobs = jobs_for(root, manifest)
        markers = [marker_status(j["complete_file"], "vlm", dict(condition=cid, manifest_hash=manifest["manifest_hash"]))
            for j, cid in zip(jobs, manifest["planned_conditions"])]
        expected_paths = {str(Path(j["complete_file"]).resolve()): m for j, m in zip(jobs, markers)}
        failures = [dict(job=j["id"], status=job_state.get(j["id"], {}).get("status")) for j in queue["jobs"]
            if str(Path(j["complete_file"]).resolve()) in expected_paths and
            expected_paths[str(Path(j["complete_file"]).resolve())]["status"] != "complete" and
            job_state.get(j["id"], {}).get("status") in ("failed", "interrupted")]
        if failures or any(m["status"] in ("invalid", "failed") for m in markers):
            raise RuntimeError("Technical stage requires inspection: " + json.dumps(dict(markers=markers, failures=failures)))
        return dict(complete=all(m["status"] == "complete" for m in markers),
            completed=sum(m["status"] == "complete" for m in markers), expected=len(jobs), markers=markers)

    def advance(self, queue, job_state):
        phase = self.state["phase"]
        if phase in TERMINAL:
            return
        if phase == "waiting_discovery":
            gate = self.completion_gate(self.paths.discovery, "discovery", queue, job_state)
            self.state["gate"] = gate
            if gate["complete"]:
                self.state["phase"] = "freeze_selection"
        elif phase == "freeze_selection":
            output = self.paths.discovery / "analysis"
            self.run_action("freeze_selection", [str(MODEL_PYTHON), "-m", MODULE + ".selection",
                "--root", str(self.paths.discovery), "--output", str(output), "--freeze-selection"])
            selection_path = output / "selection.json"
            selection = read_selection(selection_path)
            status = selection["selection"]["status"]
            self.state.update(selection=str(selection_path), selection_sha256=sha(selection_path))
            if status == "no_candidate":
                self.state.update(phase="no_candidate", reason="Frozen bounded search found no qualified complementary pair; no automatic expansion")
            elif status == "selected":
                self.state["phase"] = "prepare_confirmation"
            else:
                raise ValueError("Frozen selection has unexpected status")
        elif phase == "prepare_confirmation":
            self.verify_selection_sha()
            discovery = read_manifest(self.paths.discovery / "manifest.json", verify_sources=True)
            self.run_action("prepare_confirmation", [str(MODEL_PYTHON), "-m", MODULE + ".prepare",
                "--stage", "confirm", "--output", str(self.paths.confirmation), "--selection", self.state["selection"],
                "--discovery-root", str(self.paths.discovery), "--protocol", discovery["protocol_evidence"]["path"],
                "--design", discovery["design_evidence"]["path"]])
            manifest = read_manifest(self.paths.confirmation / "manifest.json", verify_sources=True)
            check_scope(self.paths.confirmation, manifest, "confirm")
            self.state.update(phase="enqueue_confirmation", jobs=jobs_for(self.paths.confirmation, manifest),
                confirmation_manifest_hash=manifest["manifest_hash"])
        elif phase == "enqueue_confirmation":
            self.verify_selection_sha()
            jobs = [with_assigned_gpu_wait(job, self.paths.scheduler_output / "launch_gates") for job in self.state["jobs"]]
            self.state.update(phase="waiting_confirmation", queue_append=append_jobs(self.paths.queue, jobs))
        elif phase == "waiting_confirmation":
            self.verify_selection_sha()
            gate = self.completion_gate(self.paths.confirmation, "confirm", queue, job_state)
            self.state["gate"] = gate
            if gate["complete"]:
                self.state["phase"] = "analyze_confirmation"
        elif phase == "analyze_confirmation":
            self.verify_selection_sha()
            output = self.paths.confirmation / "analysis"
            self.run_action("analyze_confirmation", [str(MODEL_PYTHON), "-m", MODULE + ".analyze",
                "--root", str(self.paths.confirmation), "--output", str(output),
                "--contention-root", str(self.paths.queue.parent / "resource_contention")])
            result = read(output / "analysis.json", {})
            if (not result.get("complete") or result.get("valid_episodes") != 600 or
                    result.get("conclusion") not in ("complementarity_reproduced", "not_reproduced")):
                raise ValueError("Confirmation analysis is incomplete or inconclusive technically")
            self.state.update(phase="report_ready", analysis=str(output / "analysis.json"), conclusion=result["conclusion"])
        else:
            raise ValueError("Unknown local-search phase: " + phase)

    def verify_selection_sha(self):
        if sha(self.state["selection"]) != self.state["selection_sha256"]:
            raise ValueError("Frozen selection changed during continuation")

    def tick(self):
        try:
            queue = read(self.paths.queue)
            state = read(self.paths.scheduler_output / "jobs_state.json", {})
            self.advance(queue, state)
            if self.state["phase"] not in TERMINAL:
                observation = self.watch.ensure(self.state["scheduler"], read(self.paths.queue), read(self.paths.scheduler_output / "jobs_state.json", {}))
                self.state["scheduler_observation"] = observation
                if observation["status"] == "needs_attention":
                    raise RuntimeError("Scheduler requires inspection: " + json.dumps(observation))
        except Exception:
            self.state.update(phase="needs_attention", error=traceback.format_exc(), attention_unix=time.time())
        self.save()
        status = dict(time=time.time(), phase=self.state["phase"], terminal=self.state["phase"] in TERMINAL,
            scheduler=self.state.get("scheduler_observation"), goal_complete_claim=False)
        write(self.paths.output / "status.json", status)
        return status


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--confirmation-root")
    parser.add_argument("--queue")
    parser.add_argument("--scheduler-output")
    parser.add_argument("--scheduler-pid", type=int)
    parser.add_argument("--output")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 3:
        parser.error("--poll-seconds must be >=3")
    paths = Paths.defaults(args.root, args.confirmation_root, args.queue, args.scheduler_output, args.output)
    supervisor = Supervisor(paths, args.scheduler_pid)
    if not args.launch:
        print(json.dumps(dict(mode="read_only", phase=supervisor.state["phase"],
            scheduler=supervisor.watch.observe(supervisor.state["scheduler"])), indent=2))
        return
    with file_lock(paths.output / "supervisor.lock", blocking=False):
        write(paths.output / "supervisor_process.json", dict(pid=os.getpid(), started_unix=time.time(),
            argv=sys.argv, source_sha256=sha(__file__), gpu_allocation=False))
        previous = None
        while True:
            status = supervisor.tick()
            signature = (status["phase"], (status.get("scheduler") or {}).get("status"))
            if signature != previous:
                print(json.dumps(status), flush=True)
                previous = signature
            if args.once or status["terminal"]:
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
