"""Review or supersede one verified CPU-only waiting gate in this experiment.

Default: dry-run. --apply pauses only the selected gate using a pidfd, replaces
its queue entry, waits for scheduler AND bound supervisor acknowledgements, then ends
that CPU wait. The original scheduler remains the only GPU allocator; the
candidate UUID is evidence of an opportunity, not a reservation or binding.
The supervisor acknowledgement can take one polling cycle (up to 45 seconds).

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
import re
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
LOCAL_SUPERVISOR = "preparations.ease_insights_20261103.vlm_local_search_v1.supervisor"
LOCAL_RUNNER = "preparations.ease_insights_20261103.vlm_local_search_v1.runner"
LEGACY_CONFIRM_RUNNER = "preparations.ease_insights_20261103.vlm.confirm_runner"
LEGACY_DISCOVERY_RUNNER = "preparations.ease_insights_20261103.vlm.runner_v2"
SCHEMA = "ease-waiting-gate-reassignment-v2"
DEAD_STATES = ("absent", "Z", "X", "x")
WAITING_PHASES = ("waiting_discovery", "waiting_confirmation")
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


def module_invocation(argv):
    """Recognize only Python [-u/-B] -m MODULE, never -c/scripts/-X/env."""
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        return None
    executable = Path(argv[0])
    if not executable.is_absolute() or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable.name):
        return None
    index, seen = 1, set()
    while index < len(argv) and argv[index] != "-m":
        arg = argv[index]
        if len(arg) < 2 or not arg.startswith("-") or any(c not in "uB" for c in arg[1:]):
            return None
        if any(c in seen for c in arg[1:]) or len(set(arg[1:])) != len(arg[1:]):
            return None
        seen.update(arg[1:]); index += 1
    if index + 1 >= len(argv) or argv[index] != "-m" or not re.fullmatch(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*", argv[index+1]):
        return None
    return dict(module=argv[index+1], args=argv[index+2:], args_index=index+2)


def exact_module(argv, module):
    parsed = module_invocation(argv)
    return parsed is not None and parsed["module"] == module


def parse_options(argv, module, values, flags=()):
    parsed = module_invocation(argv)
    require(parsed is not None and parsed["module"] == module, "Unexpected Python module invocation")
    args, result, index = parsed["args"], {}, 0
    while index < len(args):
        name = args[index]
        require(name not in result and name in (*values, *flags), "Unknown or repeated module option: " + name)
        if name in flags:
            result[name] = True; index += 1
        else:
            require(index+1 < len(args) and not args[index+1].startswith("--"), "Missing option value: " + name)
            result[name] = args[index+1]; index += 2
    return result


def gate_parts(argv):
    parsed = module_invocation(argv)
    require(parsed is not None and parsed["module"] == GATE,
            "Only the exact experiment await_assigned_gpu module wrapper is supported")
    args = parsed["args"]
    require(len(args) >= 6 and args[0] == "--gpu-uuid" and args[2] == "--status-file" and args[4] == "--",
            "Unexpected gate wrapper options")
    require(module_invocation(args[5:]) is not None, "Gate inner command must be an explicitly supported Python module")
    return args[1], Path(args[3]).resolve(), args[5:]


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


def bound_profile(job, paths):
    """Select a fixed supervisor from the scientific command, never caller PID."""
    _, _, inner = gate_parts(job["argv"])
    invocation = module_invocation(inner); module = invocation["module"]
    require(module in (LOCAL_RUNNER, LEGACY_CONFIRM_RUNNER, LEGACY_DISCOVERY_RUNNER),
            "Scientific runner is outside the reviewed local/legacy VLM profiles")
    options = parse_options(inner, module,
        ("--output", "--condition", "--gpu-uuid", "--port", "--stage", "--shard-index", "--shards"),
        ("--launch", "--preflight-only"))
    require(options.get("--launch") is True and not options.get("--preflight-only") and
            options.get("--gpu-uuid") == "{gpu_uuid}" and options.get("--port") == "{port}",
            "Scientific command must launch only the scheduler-assigned GPU/port")
    root = paths["root"]
    output = Path(options.get("--output", "/invalid")).resolve()
    local = module == LOCAL_RUNNER
    if local:
        base = root / "vlm_local_search_v1"
        require(output in (base/"discovery", base/"confirmation"), "Local runner output is not a fixed local-search stage")
        stage = "discovery" if output == base/"discovery" else "confirm"
        profile = dict(kind="local_search", module=LOCAL_SUPERVISOR, output=str(base/"supervisor"),
            root=str(base/"discovery"), confirmation_root=str(base/"confirmation"))
    else:
        stage = "confirm" if module == LEGACY_CONFIRM_RUNNER else "discovery"
        require(output == root/("vlm_confirmation" if stage == "confirm" else "vlm"),
                "Legacy VLM runner output is not its fixed stage directory")
        profile = dict(kind="legacy_vlm", module=PIPELINE, output=str(root/"pipeline"), root=str(root))
    require(options.get("--stage", stage) == stage, "Runner stage differs from fixed output root")
    try:
        shard, shards = int(options.get("--shard-index", "0")), int(options.get("--shards", "1"))
    except ValueError:
        raise Refused("Invalid shard arguments") from None
    require(0 <= shard < shards, "Invalid task shard")
    manifest_path = output/"manifest.json"
    manifest = read_json(manifest_path)
    unhashed = dict(manifest); supplied = unhashed.pop("manifest_hash", None)
    require(supplied == digest(json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()),
            "Scientific manifest hash differs")
    condition = options.get("--condition")
    require(condition in manifest.get("planned_conditions", [c["id"] for c in manifest.get("conditions", [])]),
            "Condition is not planned in the scientific manifest")
    if local:
        phase = "local_search_discovery_v1" if stage == "discovery" else "local_search_confirmation_v1"
        require(manifest.get("phase") == phase and Path(manifest.get("output_root", "/invalid")).resolve() == output
                and job.get("manifest_hash") == supplied and job.get("scientific_phase") == phase,
                "Local job/manifest phase and identities differ")
        require(manifest.get("discovery_inits" if stage == "discovery" else "confirmation_inits") ==
                ([36,37,38] if stage == "discovery" else list(range(39,44))), "Local case reservation changed")
    elif stage == "confirm":
        require(manifest.get("phase") == "independent_confirmation" and manifest.get("confirmation_inits") == list(range(31,36)),
                "Legacy confirmation manifest differs")
    else:
        require(manifest.get("discovery_inits") == [30], "Legacy discovery manifest differs")
    marker = output/condition/f"{stage}_shard{shard:02d}of{shards:02d}_complete.json"
    require(Path(job.get("complete_file", "/invalid")).resolve() == marker,
            "Completion marker is not bound to this scientific stage/condition/shard")
    require(str(manifest_path) in job.get("ready_files", []), "Scientific manifest is not a ready-file dependency")
    profile.update(stage=stage, expected_phase="waiting_discovery" if stage == "discovery" else "waiting_confirmation",
        runner_module=module, scientific_output=str(output), manifest_path=str(manifest_path),
        manifest_sha256=digest(manifest_path.read_bytes()), manifest_hash=supplied)
    return profile


def controller(runtime, receipt, module, queue, output, profile=None):
    process = runtime.identity(receipt["pid"])
    require(process.get("state") not in (*DEAD_STATES, "T", "t")
            and exact_module(process.get("argv"), module) and process.get("uid") == os.getuid(),
            module + " is not a verified live owned controller")
    require(runtime.matches_started(process, receipt.get("started", receipt.get("started_unix"))),
            "Controller PID start identity mismatch")
    if receipt.get("start_ticks") is not None:
        require(str(receipt["start_ticks"]) == process["start_ticks"], "Controller receipt start ticks differ")
    if module == SCHEDULER:
        opts = parse_options(process["argv"], module,
            ("--queue", "--output", "--gpus", "--max-gpus", "--scan-seconds", "--idle-seconds"))
        require(Path(opts.get("--queue", "/invalid")).resolve() == queue and
                Path(opts.get("--output", "/invalid")).resolve() == output,
                "Scheduler does not own the specified queue/output")
        require(opts.get("--gpus", "all") == "all" and int(opts.get("--max-gpus", "2")) == 2,
                "Scheduler GPU eligibility/concurrency differs from reviewed configuration")
    else:
        require(profile is not None and module == profile["module"], "Wrong supervisor profile")
        values = ("--root", "--queue", "--output", "--scheduler-output", "--scheduler-pid", "--poll-seconds")
        values += ("--confirmation-root",) if module == LOCAL_SUPERVISOR else ("--draft-plan", "--reports")
        opts = parse_options(process["argv"], module, values, ("--launch", "--once"))
        require(opts.get("--launch") is True and not opts.get("--once"), "Supervisor is not a continuing launch process")
        require(Path(opts.get("--root", str(EXPERIMENT_ROOT) if module == PIPELINE else "/invalid")).resolve() == Path(profile["root"])
                and Path(opts.get("--queue", str(EXPERIMENT_ROOT/"queue.json"))).resolve() == queue
                and Path(opts.get("--scheduler-output", str(EXPERIMENT_ROOT/"scheduler"))).resolve() == queue.parent/"scheduler"
                and Path(opts.get("--output", profile["output"])).resolve() == output,
                "Supervisor root/queue/scheduler/output binding differs")
        if module == LOCAL_SUPERVISOR:
            require(Path(opts.get("--confirmation-root", profile["confirmation_root"])).resolve() == Path(profile["confirmation_root"]),
                    "Local confirmation root differs")
    return process


def stable_supervisor_state(profile):
    state = read_json(Path(profile["output"])/"state.json")
    if profile["kind"] == "local_search":
        require(state.get("phase") == profile["expected_phase"] and state.get("phase") in WAITING_PHASES,
                "Local supervisor must be in the matching waiting phase")
    else:
        branches = state.get("branches", {})
        require(branches.get("vlm", {}).get("phase") == profile["expected_phase"],
                "Legacy VLM branch must be in the matching waiting phase")
        require(all(b.get("phase") in (*WAITING_PHASES, "report_ready", "needs_attention", "needs_scientific_review") for b in branches.values()),
                "Legacy pipeline has an unstable selection/preparation/enqueue phase")
    return state


def checked_paths(args):
    root = EXPERIMENT_ROOT.resolve()
    queue, states, scheduler = Path(args.queue).resolve(), Path(args.jobs_state).resolve(), Path(args.scheduler_output).resolve()
    require(queue == root / "queue.json" and scheduler == root / "scheduler"
            and states == scheduler / "jobs_state.json", "Only this experiment's standard queue and scheduler are supported")
    audit = Path(args.output_audit).resolve()
    require(audit.is_relative_to(root / "gate_reassignments") and audit != root / "gate_reassignments",
            "--output-audit must be a transaction directory under this experiment's gate_reassignments/")
    assigned_uuid(args.candidate_gpu_uuid)
    return dict(root=root, queue=queue, states=states, scheduler=scheduler, audit=audit)


def inspect_job(args, paths, runtime, queue, states):
    jobs = queue["jobs"]
    ids = [job["id"] for job in jobs]
    require(len(ids) == len(set(ids)) and ids.count(args.job_id) == 1, "Selected job must occur exactly once in the current queue")
    job = next(job for job in jobs if job["id"] == args.job_id)
    state = states.get(args.job_id, {})
    require(state.get("status") == "running", "Selected current queue job is not running")
    uuid_template, receipt_path, original = gate_parts(job["argv"])
    require(uuid_template == "{gpu_uuid}" and receipt_path == paths["scheduler"] / "launch_gates" / (args.job_id + ".json"),
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
    profile = bound_profile(job, paths)
    supervisor_path = Path(profile["output"])
    supervisor = controller(runtime, read_json(supervisor_path / "supervisor_process.json"), profile["module"],
                            paths["queue"], supervisor_path, profile)
    stable_supervisor_state(profile)
    process = runtime.identity(state["pid"])
    require(process.get("state") not in ("absent", "Z", "X", "x", "T", "t")
            and process.get("argv") == state["argv"] and process.get("ppid") == sched["pid"]
            and process.get("uid") == os.getuid(), "PID is not this scheduler's live owned CPU gate")
    require(runtime.matches_started(process, state["started"]), "Gate start identity does not match scheduler state")
    if state.get("start_ticks", state.get("proc_start_ticks")) is not None:
        require(str(state.get("start_ticks", state.get("proc_start_ticks"))) == process["start_ticks"], "Gate start ticks mismatch")
    gate_parts(process["argv"])
    receipt = check_busy_receipt(runtime, receipt_path, process["pid"], source)
    return dict(job=job, state=state, gate=process, scheduler=sched, supervisor=supervisor, profile=profile,
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
    receipt_index = module_invocation(new_job["argv"])["args_index"] + 3
    new_job["argv"][receipt_index] = str(Path(inspected["receipt_path"]).with_name(new_id + ".json"))
    require(not Path(new_job["argv"][receipt_index]).exists(), "Replacement gate receipt already exists")
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
    for name in ("scheduler", "supervisor"):
        actual = runtime.identity(audit[name]["pid"])
        require(same_process(actual, audit[name]) and actual["state"] not in ("T", "t") and actual.get("uid") == audit[name].get("uid"), name + " identity/liveness changed")
    stable_supervisor_state(audit["profile"])
    require(digest(Path(audit["profile"]["manifest_path"]).read_bytes()) == audit["profile"]["manifest_sha256"],
            "Bound scientific manifest changed during transaction")


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
        pipeline_status = read_json(Path(audit["profile"]["output"]) / "status.json")
        if audit["profile"]["kind"] == "local_search":
            require(pipeline_status.get("terminal") is not True and
                    pipeline_status.get("phase") == audit["profile"]["expected_phase"],
                    "Local supervisor status is no longer in its matching nonterminal waiting phase")
        else:
            require(pipeline_status.get("terminal") is not True and
                    pipeline_status.get("branches", {}).get("vlm") == audit["profile"]["expected_phase"],
                    "Legacy supervisor status is no longer in its matching nonterminal waiting phase")
        acknowledged = (audit["new_job"]["id"] in states
                        and isinstance(pipeline_status.get("time"), (int, float))
                        and math.isfinite(pipeline_status["time"])
                        and audit["committed_at"] < pipeline_status["time"] <= runtime.now()+1)
        if acknowledged:
            break
        require(runtime.monotonic() < deadline, "Timed out awaiting scheduler and supervisor post-commit acknowledgements")
        runtime.sleep(0.2)
    if exited:
        save_audit(paths, audit, "complete", runtime, gate_exited=True)
        return audit
    with lock(paths["queue"].with_name(paths["queue"].name + ".lock")):
        verify_controllers(runtime, audit)
        require(retained_commit(read_json(paths["queue"]), audit), "Committed queue changed before gate cleanup")
        verified_stopped(runtime, audit)
        save_audit(paths, audit, "controllers_acknowledged", runtime,
                   scheduler_ack_job=audit["new_job"]["id"], supervisor_ack_time=pipeline_status["time"], supervisor_profile=audit["profile"]["kind"])
        # Durably record intent before either signal. The gate is stopped, has
        # default unblocked SIGTERM, and cannot execute its model between these.
        save_audit(paths, audit, "terminating_cpu_gate", runtime)
        # Audit fsync/rename can block on NFS: recheck after ALL durable writes,
        # immediately before sending a terminating signal to the pidfd target.
        verify_controllers(runtime, audit)
        verified_stopped(runtime, audit)
        runtime.send(pidfd, signal.SIGTERM)
        # Revalidate before the second signal too. A queued default TERM cannot
        # run Python while the same gate remains stopped.
        if not runtime.exited(pidfd):
            verify_controllers(runtime, audit)
            verified_stopped(runtime, audit)
            require(retained_commit(read_json(paths["queue"]), audit), "Committed queue changed before CONT cleanup")
            runtime.send(pidfd, signal.SIGCONT)
    deadline = runtime.monotonic() + 3.0
    while not runtime.exited(pidfd):
        require(runtime.monotonic() < deadline, "CPU gate exit was not observed; cleanup requires review")
        runtime.sleep(0.05)
    save_audit(paths, audit, "complete", runtime, gate_exited=True)
    return audit


def validate_transaction(audit, paths, request):
    require(audit.get("schema") == SCHEMA and audit.get("request") == request,
            "Audit directory is not this exact v2 request; v1 transactions require separate review")
    require(audit.get("profile") == bound_profile(audit["original_job"], paths), "Recorded supervisor/scientific profile changed")
    require(audit["original_job"]["id"] == request["job_id"] and
            audit["gate"]["argv"] == concrete(audit["original_job"]["argv"], audit["job_state"]),
            "Audit gate differs from its original concrete job")


def stop_authorization(audit):
    gate = audit["gate"]
    require(gate.get("state") not in (*DEAD_STATES, "T", "t"), "Original gate precheck was stopped/dead")
    return dict(pid=gate["pid"], start_ticks=gate["start_ticks"], uid=gate["uid"], ppid=gate["ppid"],
        pre_stop_state=gate["state"], gate_argv=gate["argv"], inner_argv=gate_parts(gate["argv"])[2], signal="SIGSTOP")


def verified_restore_target(runtime, audit):
    authorization = audit.get("stop_authorization")
    require(authorization == stop_authorization(audit) and any(e.get("stage") == "stop_intent" for e in audit.get("events", [])),
            "No durable non-stopped precheck and own STOP intent; recovery must not signal")
    current = runtime.identity(audit["gate"]["pid"])
    require(current.get("pid") == authorization["pid"] and current.get("start_ticks") == authorization["start_ticks"]
            and current.get("uid") == authorization["uid"] == os.getuid()
            and current.get("ppid") == authorization["ppid"] == audit["scheduler"]["pid"]
            and current.get("state") not in (*DEAD_STATES, "t")
            and current.get("argv") in (authorization["gate_argv"], authorization["inner_argv"]),
            "Pre-commit recovery is not the exact owned gate or its recorded inner argv; no signals sent")
    # An already exec'd runner may have children. CONT only undoes our own STOP;
    # it never terminates that runner, its children, or any other process.
    return current


def restore_precommit(paths, runtime, audit, pidfd=None):
    """Only undo our STOP with exact proof that the queue was not committed."""
    owned_fd = False
    try:
        with lock(paths["queue"].with_name(paths["queue"].name + ".lock")):
            require(audit.get("committed_at") is None and commit_outcome(read_json(paths["queue"]), audit) == "not_committed",
                    "Commit outcome is ambiguous/committed; no pre-commit CONT allowed")
            require(audit["new_job"]["id"] not in read_json(paths["states"]),
                    "Scheduler already persisted replacement ID; pre-commit absence is not proven")
            current = verified_restore_target(runtime, audit)
            if pidfd is None:
                pidfd = runtime.open_pidfd(current["pid"]); owned_fd = True
            # Opening a pidfd can race process exit/reuse; repeat full identity.
            current = verified_restore_target(runtime, audit)
            require(not runtime.exited(pidfd), "Original pidfd target exited; no signal")
            if current["state"] == "T":
                save_audit(paths, audit, "precommit_restore_intent", runtime, commit_outcome="not_committed",
                           restore_target="cpu_gate" if current["argv"] == audit["gate"]["argv"] else "exact_inner_runner",
                           restore_from_state="T")
                # Audit fsync may block: queue outcome and every identity field
                # are checked again immediately before the sole permitted CONT.
                require(audit.get("committed_at") is None and commit_outcome(read_json(paths["queue"]), audit) == "not_committed",
                        "Queue commit proof changed before pre-commit CONT")
                require(audit["new_job"]["id"] not in read_json(paths["states"]),
                        "Replacement ID appeared before pre-commit CONT")
                current = verified_restore_target(runtime, audit)
                require(current["state"] == "T" and not runtime.exited(pidfd), "Restore target no longer stopped")
                runtime.send(pidfd, signal.SIGCONT)
            else:
                require(audit.get("stage") == "precommit_restore_intent" and audit.get("restore_from_state") == "T"
                        and current["state"] in ("S", "R"),
                        "STOP has not been observed or recovery state is uncertain; leave pending cleanup")
            save_audit(paths, audit, "aborted_restored", runtime, commit_outcome="not_committed",
                       error="Pre-commit STOP undone or already resumed; no scientific rerun/termination or queue change")
        return audit
    finally:
        if owned_fd:
            runtime.close_pidfd(pidfd)


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
            validate_transaction(prior, paths, request)
            if prior.get("stage") in ("complete", "aborted_restored"):
                return prior
            try:
                outcome = commit_outcome(read_json(paths["queue"]), prior)
            except (OSError, ValueError, KeyError, TypeError):
                outcome = "unknown"
            if prior.get("committed_at") is not None or outcome == "committed":
                require(args.apply, "A committed transaction requires --apply to finish CPU-gate cleanup")
                if prior.get("committed_at") is None:
                    prior["committed_at"] = runtime.now()
                current = runtime.identity(prior["gate"]["pid"])
                exited = current.get("state") in DEAD_STATES
                require(exited or same_process(current, prior["gate"]), "Recovery PID identity changed; no signals sent")
                fd = None if exited else runtime.open_pidfd(current["pid"])
                try:
                    if fd is not None:
                        verified_stopped(runtime, prior)
                    return finish_committed(args, paths, runtime, prior, fd)
                except BaseException as error:
                    save_audit(paths, prior, "needs_cleanup", runtime, error=str(error))
                    raise
                finally:
                    if fd is not None:
                        runtime.close_pidfd(fd)
            if prior.get("stage") in ("stop_intent", "stopped_verified", "commit_intent", "needs_cleanup", "precommit_restore_intent"):
                require(args.apply, "Interrupted pre-commit transaction requires --apply for identity-checked recovery")
                require(outcome == "not_committed", "Commit outcome is ambiguous; leave process stopped for review")
                return restore_precommit(paths, runtime, prior)

        committed, stopped, fd, audit = False, False, None, None
        temporary = paths["queue"].with_name(paths["queue"].name + f".reassign-v2.{os.getpid()}.tmp")
        try:
            with lock(paths["queue"].with_name(paths["queue"].name + ".lock")):
                original_bytes = paths["queue"].read_bytes()
                queue, states = json.loads(original_bytes), read_json(paths["states"])
                inspected = inspect_job(args, paths, runtime, queue, states)
                source = runtime.gpu(inspected["source_uuid"])
                candidate = runtime.gpu(args.candidate_gpu_uuid)
                require(source.get("gpu_uuid") == inspected["source_uuid"] and not source["idle"] and
                        (source.get("compute_process_count", 0) > 0 or source.get("memory_mib", 0) >= 500),
                        "Source GPU is already idle or lacks positive busy evidence; continue its existing gate")
                require(candidate.get("gpu_uuid") == args.candidate_gpu_uuid and candidate["idle"] and
                        candidate.get("compute_process_count") == 0 and 0 <= candidate.get("memory_mib", -1) < 500,
                        "Candidate GPU is busy or lacks complete idle evidence")
                require(runtime.lease_available(args.candidate_gpu_uuid), "Candidate project GPU lease is busy")
                proposal, new_job = make_proposal(queue, states, inspected, args.candidate_gpu_uuid)
                audit = dict(schema=SCHEMA, request=request, old_job_id=args.job_id, new_job=new_job, gate=inspected["gate"],
                    original_job=inspected["job"], job_state=inspected["state"], scheduler=inspected["scheduler"],
                    supervisor=inspected["supervisor"], profile=inspected["profile"],
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
                save_audit(paths, audit, "stop_intent", runtime, stop_authorization=stop_authorization(audit))
                verify_controllers(runtime, audit)
                current = runtime.identity(audit["gate"]["pid"])
                require(same_process(current, audit["gate"]) and current["state"] not in ("T", "t")
                        and current.get("uid") == os.getuid() and current.get("ppid") == audit["scheduler"]["pid"],
                        "Gate identity changed before stop")
                check_busy_receipt(runtime, audit["receipt_path"], audit["gate"]["pid"], audit["source_uuid"])
                require(all(0 <= runtime.now() - proof["checked_at"] <= GPU_PROOF_MAX_AGE for proof in (source, candidate)),
                        "GPU observations expired during preparation; retry later")
                require(not runtime.exited(fd), "Pidfd target exited before STOP")
                stopped = True  # STOP may take effect even when a wrapper raises.
                runtime.send(fd, signal.SIGSTOP)
                deadline = runtime.monotonic() + 2.0
                while runtime.identity(audit["gate"]["pid"]).get("state") != "T":
                    require(not runtime.exited(fd) and runtime.monotonic() < deadline, "Gate did not stop promptly")
                    runtime.sleep(0.01)
                require(same_process(runtime.identity(audit["gate"]["pid"]), audit["gate"]),
                        "Gate exec identity changed during stop; exact pre-commit restoration required")
                verified_stopped(runtime, audit)
                save_audit(paths, audit, "stopped_verified", runtime)
                save_audit(paths, audit, "commit_intent", runtime)
                # Every slow write precedes this final queue/process/controller CAS.
                verify_controllers(runtime, audit)
                verified_stopped(runtime, audit)
                require(paths["queue"].read_bytes() == original_bytes, "Queue changed before commit")
                require(read_json(paths["states"]).get(args.job_id) == inspected["state"], "Scheduler job state changed before commit")
                require(not Path(inspected["job"]["complete_file"]).exists(), "Completion target appeared before commit")
                runtime.replace_queue(temporary, paths["queue"])
                committed = True
                audit["committed_at"] = runtime.now(); audit["commit_outcome"] = "committed"
                sync_directory(paths["queue"].parent)
                save_audit(paths, audit, "committed_waiting_ack", runtime)
            return finish_committed(args, paths, runtime, audit, fd)
        except BaseException as error:
            if audit and not committed:
                try:
                    outcome = commit_outcome(read_json(paths["queue"]), audit)
                    committed = outcome != "not_committed"
                except Exception:
                    committed, outcome = stopped, "unknown"
                audit["commit_outcome"] = outcome
            if audit:
                if committed:
                    if audit.get("commit_outcome") == "committed":
                        audit.setdefault("committed_at", runtime.now())
                    save_audit(paths, audit, "needs_cleanup", runtime, error=str(error))
                elif stopped and fd is not None and not runtime.exited(fd):
                    try:
                        restore_precommit(paths, runtime, audit, fd)
                    except BaseException as recovery_error:
                        save_audit(paths, audit, "needs_cleanup", runtime, error=str(error), recovery_error=str(recovery_error))
                else:
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
