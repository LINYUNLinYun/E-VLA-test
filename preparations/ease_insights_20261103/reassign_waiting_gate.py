"""Review or supersede one verified CPU-only waiting gate in this experiment.

Default: dry-run. --apply pauses only the selected gate using a pidfd, replaces
its queue entry, waits for scheduler AND pipeline acknowledgements, then ends
that CPU wait. The original scheduler remains the only GPU allocator; the
candidate UUID is evidence of an opportunity, not a reservation or binding.
The pipeline acknowledgement can take one polling cycle (up to 45 seconds).

After queue commit there is no rollback or unconditional SIGCONT. A failed
transaction leaves a durable needs_cleanup receipt; repeat the exact command
with the same --output-audit directory to finish identity-checked cleanup.
SIGKILL or machine failure can leave a stopped gate until that recovery runs.
No command-line option accepts an arbitrary PID, experiment root, or proc root.
"""
import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import time

from preparations.ease_insights_20261103.await_assigned_gpu import assigned_uuid, scan


WORKSPACE = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = WORKSPACE / "results/ease_insights_20261103"
GATE = "preparations.ease_insights_20261103.await_assigned_gpu"
SCHEDULER = "preparations.ease_insights_20261103.scheduler"
PIPELINE = "preparations.ease_insights_20261103.pipeline"
RECEIPT_MAX_AGE = 15.0
GPU_PROOF_MAX_AGE = 10.0
ACK_TIMEOUT = 45.0


class Refused(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Refused(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def durable_file(path, data):
    path = Path(path)
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    durable_file(temporary, json_bytes(value))
    os.replace(temporary, path)
    sync_directory(path.parent)


@contextmanager
def lock(path):
    with Path(path).open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def option(argv, name, default=None):
    matches = [argv[index + 1] for index, arg in enumerate(argv[:-1]) if arg == name]
    require(len(matches) <= 1, "Ambiguous repeated option: " + name)
    return matches[0] if matches else default


def exact_module(argv, module):
    return isinstance(argv, list) and len(argv) >= 3 and argv[1:3] == ["-m", module]


def gate_parts(argv):
    require(exact_module(argv, GATE) and len(argv) >= 9
            and argv[3] == "--gpu-uuid" and argv[5] == "--status-file" and argv[7] == "--",
            "Only the exact experiment await_assigned_gpu module wrapper is supported")
    return argv[4], Path(argv[6]).resolve(), argv[8:]


def concrete(argv, state):
    values = {name: str(state[name]) for name in ("gpu", "gpu_uuid", "port") if name in state}
    result = list(argv)
    for name, value in values.items():
        result = [part.replace("{" + name + "}", value) for part in result]
    return result


class LocalRuntime:
    """The only real process/GPU operations; tests substitute a CPU fake."""
    def now(self):
        return time.time()

    def monotonic(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(seconds)

    def identity(self, pid):
        require(isinstance(pid, int) and not isinstance(pid, bool) and pid > 1, "Invalid recorded PID")
        directory = Path("/proc") / str(pid)
        try:
            before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            argv = [x.decode(errors="replace") for x in (directory / "cmdline").read_bytes().split(b"\0") if x]
            fields = dict(line.split(":", 1) for line in (directory / "status").read_text().splitlines() if ":" in line)
            children = (directory / "task" / str(pid) / "children").read_text().split()
            after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            require(before[19] == after[19], "PID changed during inspection")
            return dict(pid=pid, start_ticks=after[19], state=after[0], ppid=int(after[1]), argv=argv,
                        uid=int(fields["Uid"].split()[0]), children=children,
                        sigterm_default=all(not (int(fields[name].strip(), 16) & (1 << (signal.SIGTERM - 1)))
                                            for name in ("SigIgn", "SigCgt", "SigBlk")))
        except FileNotFoundError:
            return dict(pid=pid, state="absent")

    def matches_started(self, identity, started):
        boot = next(line.split()[1] for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
        born = int(boot) + int(identity["start_ticks"]) / os.sysconf("SC_CLK_TCK")
        return math.isfinite(float(started)) and float(started) - 5 <= born <= float(started) + 1

    def gpu(self, uuid):
        result = scan(uuid)
        return dict(result, gpu_uuid=uuid, checked_at=self.now())

    def lease_available(self, uuid):
        # The lease is immediately released: this is a probe, never allocation.
        with (Path("/tmp") / f"embodied-benchmark-gpu-{uuid}.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(stream, fcntl.LOCK_UN)
            return True

    def open_pidfd(self, pid):
        require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "pidfd support is required")
        return os.pidfd_open(pid)

    def close_pidfd(self, fd):
        os.close(fd)

    def send(self, fd, sig):
        signal.pidfd_send_signal(fd, sig)

    def exited(self, fd):
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        return bool(poller.poll(0))

    def replace_queue(self, temporary, queue):
        os.replace(temporary, queue)


def same_process(actual, expected):
    return (actual.get("pid") == expected["pid"] and actual.get("start_ticks") == expected["start_ticks"]
            and actual.get("argv") == expected["argv"] and actual.get("state") not in ("absent", "Z", "X", "x"))


def controller(runtime, receipt, module, queue, output):
    process = runtime.identity(receipt["pid"])
    require(process.get("state") not in ("absent", "Z", "X", "x", "T", "t")
            and exact_module(process.get("argv"), module), module + " is not a verified live controller")
    require(runtime.matches_started(process, receipt.get("started", receipt.get("started_unix"))),
            "Controller PID start identity mismatch")
    argv = process["argv"]
    if module == SCHEDULER:
        require(Path(option(argv, "--queue", "/invalid")).resolve() == queue
                and Path(option(argv, "--output", "/invalid")).resolve() == output,
                "Scheduler does not own the specified queue/output")
        require(option(argv, "--gpus", "all") == "all", "Restricted scheduler GPU sets require separate candidate eligibility review")
    else:
        root = Path(option(argv, "--root", str(EXPERIMENT_ROOT))).resolve()
        require(root == EXPERIMENT_ROOT.resolve()
                and Path(option(argv, "--queue", str(root / "queue.json"))).resolve() == queue
                and Path(option(argv, "--output", str(root / "pipeline"))).resolve() == output
                and "--launch" in argv, "Pipeline does not own this experiment")
    return process


def checked_paths(args):
    root = EXPERIMENT_ROOT.resolve()
    queue, states, scheduler = Path(args.queue).resolve(), Path(args.jobs_state).resolve(), Path(args.scheduler_output).resolve()
    require(queue == root / "queue.json" and scheduler == root / "scheduler"
            and states == scheduler / "jobs_state.json", "Only this experiment's standard queue and scheduler are supported")
    audit = Path(args.output_audit).resolve()
    require(audit.is_relative_to(root / "gate_reassignments") and audit != root / "gate_reassignments",
            "--output-audit must be a transaction directory under this experiment's gate_reassignments/")
    assigned_uuid(args.candidate_gpu_uuid)
    return dict(root=root, queue=queue, states=states, scheduler=scheduler, pipeline=root / "pipeline", audit=audit)


def inspect_job(args, paths, runtime, queue, states):
    jobs = queue["jobs"]
    ids = [job["id"] for job in jobs]
    require(len(ids) == len(set(ids)) and ids.count(args.job_id) == 1, "Selected job must occur exactly once in the current queue")
    job = next(job for job in jobs if job["id"] == args.job_id)
    state = states.get(args.job_id, {})
    require(state.get("status") == "running", "Selected current queue job is not running")
    uuid_template, receipt_path, original = gate_parts(job["argv"])
    require(uuid_template == "{gpu_uuid}" and receipt_path.parent == paths["scheduler"] / "launch_gates",
            "Gate must use scheduler UUID substitution and this scheduler's receipt directory")
    require(job.get("launch_gate", {}).get("original_argv") == original,
            "Frozen original_argv does not match gate command")
    require(state.get("argv") == concrete(job["argv"], state), "Recorded argv differs from the current queue job")
    source = assigned_uuid(state.get("gpu_uuid", ""))
    require(source != args.candidate_gpu_uuid, "Candidate must differ from the source GPU")
    require(not Path(job["complete_file"]).exists(), "Completion target already exists; no migration allowed")
    require(not (paths["scheduler"] / "stop_after_jobs.json").exists(), "Scheduler stop marker is present")
    sched = controller(runtime, read_json(paths["scheduler"] / "controller_process.json"), SCHEDULER,
                       paths["queue"], paths["scheduler"])
    pipe = controller(runtime, read_json(paths["pipeline"] / "supervisor_process.json"), PIPELINE,
                      paths["queue"], paths["pipeline"])
    phases = read_json(paths["pipeline"] / "state.json").get("branches", {})
    require(phases and all(record.get("phase") in ("waiting_confirmation", "needs_attention", "needs_scientific_review",
                                                    "report_ready", "analyze_confirmation") for record in phases.values()),
            "Pipeline is not in a stable waiting/report phase; enqueue replay is unsafe")
    process = runtime.identity(state["pid"])
    require(process.get("state") not in ("absent", "Z", "X", "x", "T", "t")
            and process.get("argv") == state["argv"] and process.get("ppid") == sched["pid"]
            and process.get("uid") == os.getuid(), "PID is not this scheduler's live owned CPU gate")
    require(runtime.matches_started(process, state["started"]), "Gate start identity does not match scheduler state")
    if state.get("start_ticks", state.get("proc_start_ticks")) is not None:
        require(str(state.get("start_ticks", state.get("proc_start_ticks"))) == process["start_ticks"], "Gate start ticks mismatch")
    gate_parts(process["argv"])
    receipt = check_busy_receipt(runtime, receipt_path, process["pid"], source)
    return dict(job=job, state=state, gate=process, scheduler=sched, pipeline=pipe,
                receipt_path=str(receipt_path), receipt=receipt, source_uuid=source)


def check_busy_receipt(runtime, path, pid, uuid):
    path = Path(path)
    before = path.stat()
    value = read_json(path)
    after = path.stat()
    require((before.st_ino, before.st_mtime_ns, before.st_size) == (after.st_ino, after.st_mtime_ns, after.st_size),
            "Gate receipt changed during read; retry later")
    require(0 <= runtime.now() - after.st_mtime <= RECEIPT_MAX_AGE and value.get("status") == "busy"
            and value.get("idle_seconds") == 0 and value.get("pid") == pid and value.get("gpu_uuid") == uuid,
            "Gate must have a fresh busy receipt with idle_seconds=0")
    return value


def make_proposal(queue, states, inspected, candidate):
    old = inspected["job"]
    identity = f"{old['id']}:{inspected['gate']['pid']}:{inspected['gate']['start_ticks']}:{candidate}"
    new_id = old["id"] + "__wait_reassign_" + digest(identity.encode())[:12]
    require(new_id not in states and all(job["id"] != new_id for job in queue["jobs"]), "Replacement ID already exists")
    new_job = copy.deepcopy(old)
    new_job["id"] = new_id
    new_job["argv"][6] = str(Path(inspected["receipt_path"]).with_name(new_id + ".json"))
    require(not Path(new_job["argv"][6]).exists(), "Replacement gate receipt already exists")
    proposed = copy.deepcopy(queue)
    proposed["jobs"] = [copy.deepcopy(new_job) if job["id"] == old["id"] else job for job in proposed["jobs"]]
    for job in proposed["jobs"]:
        if "depends_on" in job:
            job["depends_on"] = [new_id if dep == old["id"] else dep for dep in job["depends_on"]]
    return proposed, new_job


def retained_commit(queue, audit):
    jobs = queue["jobs"]
    return (all(job["id"] != audit["old_job_id"] for job in jobs)
            and sum(job == audit["new_job"] for job in jobs) == 1
            and all(audit["old_job_id"] not in job.get("depends_on", []) for job in jobs))


def commit_outcome(queue, audit):
    if retained_commit(queue, audit):
        return "committed"
    jobs = queue["jobs"]
    if (sum(job == audit["original_job"] for job in jobs) == 1
            and all(job["id"] != audit["new_job"]["id"] for job in jobs)):
        return "not_committed"
    return "unknown"


def save_audit(paths, audit, stage, runtime, **details):
    audit.update(details, stage=stage, updated_at=runtime.now())
    audit.setdefault("events", []).append(dict(stage=stage, at=runtime.now()))
    durable_json(paths["audit"] / "audit.json", audit)


def verify_controllers(runtime, audit):
    for name in ("scheduler", "pipeline"):
        actual = runtime.identity(audit[name]["pid"])
        require(same_process(actual, audit[name]) and actual["state"] not in ("T", "t"), name + " identity/liveness changed")


def verified_stopped(runtime, audit):
    process = runtime.identity(audit["gate"]["pid"])
    require(same_process(process, audit["gate"]) and process["state"] == "T"
            and process.get("ppid") == audit["scheduler"]["pid"] and process.get("uid") == os.getuid()
            and process.get("sigterm_default") and not process.get("children"),
            "Process is no longer the same stopped CPU-only gate with default SIGTERM and no children")
    gate_parts(process["argv"])
    return process


def finish_committed(args, paths, runtime, audit, pidfd):
    """Irreversible side: never resume a live gate without first queuing TERM."""
    deadline = runtime.monotonic() + ACK_TIMEOUT
    while True:
        verify_controllers(runtime, audit)
        require(retained_commit(read_json(paths["queue"]), audit), "Committed queue replacement changed; manual review required")
        exited = pidfd is None or runtime.exited(pidfd)
        if not exited:
            verified_stopped(runtime, audit)
        states = read_json(paths["states"])
        pipeline_status = read_json(paths["pipeline"] / "status.json")
        acknowledged = (audit["new_job"]["id"] in states
                        and isinstance(pipeline_status.get("time"), (int, float))
                        and math.isfinite(pipeline_status["time"])
                        and pipeline_status["time"] > audit["committed_at"])
        if acknowledged:
            break
        require(runtime.monotonic() < deadline, "Timed out awaiting scheduler and pipeline post-commit acknowledgements")
        runtime.sleep(0.2)
    if exited:
        save_audit(paths, audit, "complete", runtime, gate_exited=True)
        return audit
    with lock(paths["queue"].with_name(paths["queue"].name + ".lock")):
        verify_controllers(runtime, audit)
        require(retained_commit(read_json(paths["queue"]), audit), "Committed queue changed before gate cleanup")
        verified_stopped(runtime, audit)
        save_audit(paths, audit, "controllers_acknowledged", runtime,
                   scheduler_ack_job=audit["new_job"]["id"], pipeline_ack_time=pipeline_status["time"])
        # Durably record intent before either signal. The gate is stopped, has
        # default unblocked SIGTERM, and cannot execute its model between these.
        save_audit(paths, audit, "terminating_cpu_gate", runtime)
        # Audit fsync/rename can block on NFS: recheck after ALL durable writes,
        # immediately before sending a terminating signal to the pidfd target.
        verify_controllers(runtime, audit)
        verified_stopped(runtime, audit)
        runtime.send(pidfd, signal.SIGTERM)
        runtime.send(pidfd, signal.SIGCONT)
    deadline = runtime.monotonic() + 3.0
    while not runtime.exited(pidfd):
        require(runtime.monotonic() < deadline, "CPU gate exit was not observed; cleanup requires review")
        runtime.sleep(0.05)
    save_audit(paths, audit, "complete", runtime, gate_exited=True)
    return audit


def operate(args, runtime=None):
    runtime = runtime or LocalRuntime()
    paths = checked_paths(args)
    paths["audit"].mkdir(parents=True, exist_ok=True)
    request = dict(queue=str(paths["queue"]), jobs_state=str(paths["states"]), scheduler_output=str(paths["scheduler"]),
                   job_id=args.job_id, candidate_gpu_uuid=args.candidate_gpu_uuid)
    with lock(paths["audit"] / "transaction.lock"):
        audit_path = paths["audit"] / "audit.json"
        if audit_path.exists():
            prior = read_json(audit_path)
            require(prior.get("request") == request, "Audit directory belongs to a different request")
            if prior.get("stage") == "complete":
                return prior
            if prior.get("stage") == "aborted_restored":
                return prior
            outcome = commit_outcome(read_json(paths["queue"]), prior) if prior.get("new_job") else "not_committed"
            if prior.get("committed_at") is not None or outcome == "committed":
                require(args.apply, "A committed transaction requires --apply to finish CPU-gate cleanup")
                if prior.get("committed_at") is None:
                    prior["committed_at"] = runtime.now()
                current = runtime.identity(prior["gate"]["pid"])
                exited = current.get("state") in ("absent", "Z", "X", "x")
                require(exited or same_process(current, prior["gate"]), "Recovery PID identity changed; no signals sent")
                fd = None if exited else runtime.open_pidfd(current["pid"])
                try:
                    return finish_committed(args, paths, runtime, prior, fd)
                except BaseException as error:
                    save_audit(paths, prior, "needs_cleanup", runtime, error=str(error))
                    raise
                finally:
                    if fd is not None:
                        runtime.close_pidfd(fd)
            if prior.get("stage") in ("stop_intent", "stopped_verified", "commit_intent", "needs_cleanup"):
                require(args.apply, "Interrupted pre-commit transaction requires --apply to recover its CPU wait")
                require(outcome == "not_committed", "Commit outcome is ambiguous; leave gate stopped for review")
                current = runtime.identity(prior["gate"]["pid"])
                require(same_process(current, prior["gate"]), "Pre-commit recovery identity changed; no signals sent")
                if current["state"] == "T":
                    fd = runtime.open_pidfd(current["pid"])
                    try:
                        require(same_process(runtime.identity(current["pid"]), prior["gate"]), "Recovery identity changed")
                        runtime.send(fd, signal.SIGCONT)
                    finally:
                        runtime.close_pidfd(fd)
                save_audit(paths, prior, "aborted_restored", runtime, error="Recovered an interrupted pre-commit transaction; inspect before retry")
                return prior

        committed, stopped, fd, audit = False, False, None, None
        temporary = paths["queue"].with_name(paths["queue"].name + f".reassign.{os.getpid()}.tmp")
        try:
            with lock(paths["queue"].with_name(paths["queue"].name + ".lock")):
                original_bytes = paths["queue"].read_bytes()
                queue, states = json.loads(original_bytes), read_json(paths["states"])
                inspected = inspect_job(args, paths, runtime, queue, states)
                source = runtime.gpu(inspected["source_uuid"])
                candidate = runtime.gpu(args.candidate_gpu_uuid)
                require(not source["idle"], "Source GPU is already idle; continue its existing gate")
                require(candidate["idle"], "Candidate GPU is busy")
                require(runtime.lease_available(args.candidate_gpu_uuid), "Candidate project GPU lease is busy")
                proposal, new_job = make_proposal(queue, states, inspected, args.candidate_gpu_uuid)
                audit = dict(request=request, old_job_id=args.job_id, new_job=new_job, gate=inspected["gate"],
                             original_job=inspected["job"],
                             scheduler=inspected["scheduler"], pipeline=inspected["pipeline"],
                             source_uuid=inspected["source_uuid"], receipt_path=inspected["receipt_path"],
                             queue_before_sha256=digest(original_bytes), queue_after_sha256=digest(json_bytes(proposal)),
                             source_observation=source, candidate_observation=candidate,
                             candidate_is_only_an_opportunity=True, allocator="existing_scheduler_only")
                durable_file(paths["audit"] / "queue.before.json", original_bytes)
                durable_file(paths["audit"] / "queue.proposed.json", json_bytes(proposal))
                save_audit(paths, audit, "prepared" if args.apply else "dry_run", runtime)
                if not args.apply:
                    return audit
                durable_file(temporary, json_bytes(proposal))
                fd = runtime.open_pidfd(inspected["gate"]["pid"])
                save_audit(paths, audit, "stop_intent", runtime)
                # All slow preparation is before this final freshness/identity
                # check. A busy receipt means a new 60-second idle window is
                # required before this gate can exec anything.
                verify_controllers(runtime, audit)
                current = runtime.identity(audit["gate"]["pid"])
                require(same_process(current, audit["gate"]) and current["state"] not in ("T", "t"), "Gate identity changed before stop")
                check_busy_receipt(runtime, audit["receipt_path"], audit["gate"]["pid"], audit["source_uuid"])
                require(all(0 <= runtime.now() - proof["checked_at"] <= GPU_PROOF_MAX_AGE for proof in (source, candidate)),
                        "GPU observations expired during preparation; retry later")
                stopped = True  # SIGSTOP may have taken effect even if a wrapper raises.
                runtime.send(fd, signal.SIGSTOP)
                deadline = runtime.monotonic() + 2.0
                while runtime.identity(audit["gate"]["pid"]).get("state") != "T":
                    require(not runtime.exited(fd) and runtime.monotonic() < deadline, "Gate did not stop promptly")
                    runtime.sleep(0.01)
                if not same_process(runtime.identity(audit["gate"]["pid"]), audit["gate"]):
                    # It exec'd between the last check and STOP: immediately
                    # undo only our STOP. Never send TERM to the changed image.
                    runtime.send(fd, signal.SIGCONT)
                    stopped = False
                    raise Refused("Gate exec identity changed during stop; resumed without queue changes")
                verified_stopped(runtime, audit)
                save_audit(paths, audit, "stopped_verified", runtime)
                verify_controllers(runtime, audit)
                require(paths["queue"].read_bytes() == original_bytes, "Queue changed before commit")
                require(read_json(paths["states"]).get(args.job_id) == inspected["state"], "Scheduler job state changed before commit")
                require(not Path(inspected["job"]["complete_file"]).exists(), "Completion target appeared before commit")
                save_audit(paths, audit, "commit_intent", runtime)
                runtime.replace_queue(temporary, paths["queue"])
                committed = True
                audit["committed_at"] = runtime.now()
                audit["commit_outcome"] = "committed"
                sync_directory(paths["queue"].parent)
                save_audit(paths, audit, "committed_waiting_ack", runtime)
            return finish_committed(args, paths, runtime, audit, fd)
        except BaseException as error:
            # An injected/delayed replace may have succeeded before raising.
            if audit and not committed:
                try:
                    outcome = commit_outcome(read_json(paths["queue"]), audit)
                    committed = outcome != "not_committed"
                except Exception:
                    # Uncertain commit outcome must never resume an old gate.
                    committed = stopped
                    outcome = "unknown"
                audit["commit_outcome"] = outcome
            if audit:
                if committed:
                    if audit.get("commit_outcome") == "committed":
                        audit.setdefault("committed_at", runtime.now())
                    save_audit(paths, audit, "needs_cleanup", runtime, error=str(error))
                else:
                    if stopped and fd is not None and not runtime.exited(fd):
                        # pidfd binds the exact target even if it exec'd before
                        # STOP. Only CONT is allowed on this pre-commit path.
                        runtime.send(fd, signal.SIGCONT)
                    save_audit(paths, audit, "aborted_restored", runtime, error=str(error))
            raise
        finally:
            if fd is not None:
                runtime.close_pidfd(fd)
            if temporary.exists():
                temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--jobs-state", required=True)
    parser.add_argument("--scheduler-output", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--candidate-gpu-uuid", required=True)
    parser.add_argument("--output-audit", required=True, help="New directory under results/ease_insights_20261103/gate_reassignments/")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise InterruptedError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    result = operate(args)
    print(json.dumps(dict(stage=result["stage"], old_job_id=result["old_job_id"], new_job_id=result["new_job"]["id"],
                          output_audit=args.output_audit), ensure_ascii=False))


if __name__ == "__main__":
    main()
