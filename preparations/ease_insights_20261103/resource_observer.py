"""Read-only, assigned-GPU resource snapshots for currently executing queue jobs.

Run with --queue PATH --jobs-state PATH --output DIRECTORY [--once]. No GPU is
queried while jobs are absent, unverified, dead, or still in await_assigned_gpu.
The observer never allocates GPUs, launches models, or changes scientific data.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


GATE_MODULE = "preparations.ease_insights_20261103.await_assigned_gpu"
UUID_PATTERN = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
POLL_SECONDS = 15.0
QUERY_TIMEOUT_SECONDS = 5.0


def atomic_status(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def proc_identity(pid, proc_root=Path("/proc")):
    """Read local process identity twice to detect PID reuse during inspection."""
    directory = Path(proc_root) / str(pid)
    try:
        before = (directory / "stat").read_text()
        argv = [part.decode(errors="replace")
                for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
        after = (directory / "stat").read_text()
        first = before.rsplit(")", 1)[1].split()
        last = after.rsplit(")", 1)[1].split()
        if (int(before.split(" ", 1)[0]) != pid or int(after.split(" ", 1)[0]) != pid
                or first[19] != last[19]):
            return dict(status="unknown", reason="process_identity_changed")
        return dict(status="dead" if last[0] in ("Z", "X", "x") else ("live" if argv else "unknown"),
                    proc_start_ticks=last[19], argv=argv, process_state=last[0])
    except FileNotFoundError:
        return dict(status="absent" if not directory.exists() else "unknown")
    except (OSError, ValueError, IndexError) as error:
        return dict(status="unknown", error=str(error))


def gate_target(argv):
    if not argv:
        return None
    if Path(argv[0]).name == "await_assigned_gpu.py":
        return 0
    # Current jobs use a Python module or script. Stop at its first target so a
    # downstream literal argument mentioning the gate cannot hide an experiment.
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "-m":
            return index + 1 if index + 1 < len(argv) and argv[index + 1] == GATE_MODULE else None
        if argument == "-c":
            return None
        if argument == "--":
            index += 1
            return index if index < len(argv) and Path(argv[index]).name == "await_assigned_gpu.py" else None
        if argument in ("-W", "-X", "--check-hash-based-pycs"):
            index += 2
            continue
        if not argument.startswith("-"):
            return index if Path(argument).name == "await_assigned_gpu.py" else None
        index += 1
    return None


def is_gate(argv):
    return gate_target(argv) is not None


def normalized_argv(argv, state):
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        return None
    values = {name: str(state[name]) for name in ("gpu", "gpu_uuid", "port") if name in state}
    argv = list(argv)
    for name, value in values.items():
        argv = [argument.replace("{" + name + "}", value) for argument in argv]
    target = gate_target(argv)
    if target is not None:
        try:
            return argv[argv.index("--", target + 1) + 1:] or None
        except ValueError:
            return None
    return argv


def expected_argv(job, state):
    queued = normalized_argv(job.get("argv"), state)
    recorded = normalized_argv(state.get("argv", job.get("argv")), state)
    return queued if queued is not None and queued == recorded else None


def running_jobs(queue, states, proc_root=Path("/proc")):
    active, skipped = [], []
    for job in queue["jobs"]:
        job_id = job["id"]
        state = states.get(job_id, {})
        if state.get("status") != "running":
            continue
        record = dict(job_id=job_id, local_pid=state.get("pid"),
                      assigned_gpu_uuid=state.get("gpu_uuid"))
        pid = state.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            skipped.append(dict(record, reason="unknown_pid"))
            continue
        process = proc_identity(pid, proc_root)
        record["proc_start_ticks"] = process.get("proc_start_ticks")
        if process["status"] != "live":
            skipped.append(dict(record, reason="process_" + process["status"],
                                error=process.get("error")))
            continue
        known_ticks = state.get("proc_start_ticks", state.get("start_ticks"))
        if known_ticks is not None and str(known_ticks) != process["proc_start_ticks"]:
            skipped.append(dict(record, reason="process_start_ticks_mismatch"))
            continue
        if known_ticks is None:
            # The existing scheduler records an epoch start time, not start
            # ticks. Bound that fallback to account for /proc btime's one-second
            # resolution and a short Popen delay; uncertainty means no query.
            try:
                recorded_start = float(state["started"])
                boot = next(line.split()[1] for line in (Path(proc_root) / "stat").read_text().splitlines()
                            if line.startswith("btime "))
                born = int(boot) + int(process["proc_start_ticks"]) / os.sysconf("SC_CLK_TCK")
                if not math.isfinite(recorded_start) or recorded_start <= 0:
                    raise ValueError("Invalid scheduler process start time")
                record["proc_started_at"] = born
                if not recorded_start - 5 <= born <= recorded_start + 1:
                    skipped.append(dict(record, reason="process_start_time_mismatch"))
                    continue
            except (OSError, KeyError, ValueError, TypeError, StopIteration, ZeroDivisionError) as error:
                skipped.append(dict(record, reason="unknown_recorded_process_start", error=str(error)))
                continue
        if is_gate(process["argv"]):
            skipped.append(dict(record, reason="waiting_launch_gate"))
            continue
        expected = expected_argv(job, state)
        if expected is None or process["argv"] != expected:
            skipped.append(dict(record, reason="unknown_command_identity" if expected is None
                                else "process_argv_mismatch"))
            continue
        if not isinstance(record["assigned_gpu_uuid"], str) or not UUID_PATTERN.fullmatch(record["assigned_gpu_uuid"]):
            skipped.append(dict(record, reason="unknown_assigned_gpu_uuid"))
            continue
        active.append(record)
    return active, skipped


def numeric_or_unknown(value, integer=False):
    try:
        result = int(value) if integer else float(value)
        return result if math.isfinite(result) and result >= (1 if integer else 0) else None
    except (TypeError, ValueError, OverflowError):
        return None


def query_gpu(gpu_uuid):
    """Two bounded reads explicitly restricted to one scheduler-assigned UUID."""
    if not isinstance(gpu_uuid, str) or not UUID_PATTERN.fullmatch(gpu_uuid):
        raise ValueError("One full assigned GPU UUID is required")
    result = dict(gpu_uuid=gpu_uuid, query_started_at=time.time(), memory_used_mib=None,
                  utilization_gpu_percent=None, compute_processes=None, compute_process_count=None,
                  contention_indication="unknown", query_errors=[], raw={}, query_intervals={})
    specs = (("gpu", "uuid,memory.used,utilization.gpu"),
             ("compute_apps", "gpu_uuid,pid,used_memory"))
    for name, fields in specs:
        started = time.time()
        try:
            flag = "--query-gpu=" if name == "gpu" else "--query-compute-apps="
            raw = subprocess.check_output(
                ["nvidia-smi", "--id=" + gpu_uuid, flag + fields, "--format=csv,noheader,nounits"],
                text=True, timeout=QUERY_TIMEOUT_SECONDS, stderr=subprocess.PIPE,
            )
            result["raw"][name] = raw
            rows = [[part.strip() for part in line.split(",")] for line in raw.splitlines() if line.strip()]
            if name == "gpu":
                if len(rows) != 1 or len(rows[0]) != 3 or rows[0][0] != gpu_uuid:
                    raise ValueError("GPU query returned an unexpected UUID or row shape")
                result["memory_used_mib"] = numeric_or_unknown(rows[0][1])
                result["utilization_gpu_percent"] = numeric_or_unknown(rows[0][2])
            else:
                processes = []
                for row in rows:
                    if len(row) != 3 or row[0] != gpu_uuid:
                        raise ValueError("Compute query returned an unexpected UUID or row shape")
                    processes.append(dict(gpu_uuid=row[0], host_pid=numeric_or_unknown(row[1], integer=True),
                                          used_memory_mib=numeric_or_unknown(row[2]), raw_fields=row))
                result["compute_processes"] = processes
                result["compute_process_count"] = len(processes)
                result["contention_indication"] = ("multiple_compute_processes" if len(processes) >= 2
                                                    else "not_observed_in_snapshot")
        except Exception as error:
            detail = dict(query=name, error_type=type(error).__name__, error=str(error))
            for field in ("output", "stderr"):
                value = getattr(error, field, None)
                if value is not None:
                    detail[field] = value.decode(errors="replace") if isinstance(value, bytes) else str(value)
            result["query_errors"].append(detail)
        finally:
            result["query_intervals"][name] = dict(started_at=started, finished_at=time.time())
    result["query_finished_at"] = time.time()
    return result


def sample(queue, states, proc_root=Path("/proc")):
    observed_at = time.time()
    jobs, skipped = running_jobs(queue, states, proc_root)
    if not jobs:
        return None, dict(status="waiting_experiment", updated_at=observed_at, skipped_jobs=skipped)
    started = time.time()
    gpus = [query_gpu(uuid) for uuid in sorted({job["assigned_gpu_uuid"] for job in jobs})]
    errors = [dict(gpu_uuid=gpu["gpu_uuid"], **error) for gpu in gpus for error in gpu["query_errors"]]
    host_gpu_sets = {}
    for gpu in gpus:
        for process in gpu["compute_processes"] or []:
            if process["host_pid"] is not None:
                host_gpu_sets.setdefault(str(process["host_pid"]), set()).add(process["gpu_uuid"])
    record = dict(schema_version=1, observed_at=observed_at, query_started_at=started,
                  query_finished_at=time.time(), jobs=jobs, skipped_jobs=skipped, gpus=gpus, query_errors=errors,
                  multi_gpu_processes={pid: sorted(uuids) for pid, uuids in host_gpu_sets.items() if len(uuids) >= 2},
                  pid_namespace="local_pid is from this /proc namespace; host_pid is from nvidia-smi and is not mapped to local_pid",
                  timing_limit="Snapshots indicate possible contention; they do not establish exclusive GPU access or complete overlap intervals")
    return record, dict(status="query_error" if errors else "observing", updated_at=time.time(),
                        active_jobs=jobs, skipped_jobs=skipped, queried_gpu_uuids=[gpu["gpu_uuid"] for gpu in gpus],
                        query_error_count=len(errors))


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "observer.lock").open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A resource observer already holds this output directory", flush=True)
            return 1
        samples = 0
        last_status = None
        while True:
            started = time.monotonic()
            try:
                queue = json.loads(Path(args.queue).read_text())
                states = json.loads(Path(args.jobs_state).read_text())
                record, status = sample(queue, states)
                if record is not None:
                    with (output / "samples.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    samples += 1
            except Exception as error:
                status = dict(status="observation_error", updated_at=time.time(), error=str(error),
                              error_type=type(error).__name__)
            status.update(observer_pid=os.getpid(), samples_this_run=samples, queue=str(Path(args.queue).resolve()),
                          jobs_state=str(Path(args.jobs_state).resolve()), poll_seconds=POLL_SECONDS)
            atomic_status(output / "status.json", status)
            if status["status"] != last_status:
                print(json.dumps(dict(status=status["status"], observer_pid=os.getpid()), ensure_ascii=False), flush=True)
                last_status = status["status"]
            if args.once:
                return 1 if status["status"] == "observation_error" else 0
            time.sleep(max(0.0, POLL_SECONDS - (time.monotonic() - started)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--jobs-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--once", action="store_true")
    raise SystemExit(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
