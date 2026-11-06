"""Wait for the scheduler-assigned GPU to stay idle, then replace this process.

Usage: python -m preparations.ease_insights_20261103.await_assigned_gpu \
    --gpu-uuid GPU-... --status-file /path/to/status.json -- /absolute/python ...

This gate does not allocate GPUs. The scheduler's inherited GPU lease remains
open across exec, and a busy GPU or failed query causes indefinite waiting.
EASE_GPU_GATE_IDLE_SECONDS (1..3600) and EASE_GPU_GATE_POLL_SECONDS (0.1..60)
override the resource gate's default 60-second idle window and 10-second poll.
These two settings are removed from the environment before the original command.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


IDLE_SECONDS = 60.0
POLL_SECONDS = 10.0
IDLE_ENV = "EASE_GPU_GATE_IDLE_SECONDS"
POLL_ENV = "EASE_GPU_GATE_POLL_SECONDS"
RESOURCE_REVISION = "assigned-gpu-idle-v2"
MEMORY_LIMIT_MIB = 500.0
UUID_PATTERN = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def assigned_uuid(value):
    """Require one full GPU UUID; never accept indexes or multi-device selectors."""
    if not UUID_PATTERN.fullmatch(value):
        raise ValueError("--gpu-uuid must be one full GPU UUID (GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)")
    return value


def scan(gpu_uuid):
    """Query only the assigned GPU, rejecting incomplete or unexpected output."""
    assigned_uuid(gpu_uuid)
    selector = "--id=" + gpu_uuid
    apps = subprocess.check_output(
        ["nvidia-smi", selector, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    )
    pids = []
    for line in apps.splitlines():
        if not line.strip():
            continue
        identity, pid = [part.strip() for part in line.split(",")]
        if identity != gpu_uuid or int(pid) <= 0:
            raise ValueError("Unexpected compute process in assigned GPU query")
        pids.append(int(pid))
    raw = subprocess.check_output(
        ["nvidia-smi", selector, "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    )
    rows = [line for line in raw.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("Assigned GPU query must return exactly one GPU")
    identity, memory = [part.strip() for part in rows[0].split(",")]
    memory_mib = float(memory)
    if identity != gpu_uuid or not math.isfinite(memory_mib) or memory_mib < 0:
        raise ValueError("Unexpected UUID or memory in assigned GPU query")
    return dict(memory_mib=memory_mib, compute_process_count=len(pids),
                idle=not pids and memory_mib < MEMORY_LIMIT_MIB)


def write_status(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def resource_policy():
    """Validate explicit resource-only overrides; never silently use defaults."""
    values = []
    for name, default, lower, upper in (
        (IDLE_ENV, IDLE_SECONDS, 1.0, 3600.0),
        (POLL_ENV, POLL_SECONDS, 0.1, 60.0),
    ):
        raw = os.environ.get(name)
        try:
            value = default if raw is None else float(raw)
        except ValueError:
            raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]") from None
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
        values.append(value)
    return tuple(values)


def wait_and_exec(gpu_uuid, status_file, command):
    assigned_uuid(gpu_uuid)
    if not command:
        raise ValueError("An original command is required")
    idle_threshold, poll_seconds = resource_policy()
    started = time.monotonic()
    idle_since = None
    previous_status = None
    final_idle_check = None
    while True:
        details = {}
        checked_before_exec = False
        try:
            observation = scan(gpu_uuid)
        except Exception as error:
            idle_since = None
            status = "check_error"
            details["error"] = str(error)
        else:
            now = time.monotonic()
            details.update(memory_mib=observation["memory_mib"],
                           compute_process_count=observation["compute_process_count"])
            if observation["idle"]:
                if idle_since is None:
                    idle_since = now
                status = "launching" if now - idle_since >= idle_threshold else "waiting"
            else:
                idle_since = None
                status = "busy"
        if status == "launching":
            # Close the polling race as far as a resource-only gate can: require
            # another fresh query of the same assigned UUID before exec.
            final_idle_check = dict(gpu_uuid=gpu_uuid)
            checked_before_exec = True
            try:
                confirmation = scan(gpu_uuid)
            except Exception as error:
                idle_since = None
                status = "check_error"
                details["error"] = str(error)
                final_idle_check.update(status="check_error", error=str(error))
            else:
                details.update(memory_mib=confirmation["memory_mib"],
                               compute_process_count=confirmation["compute_process_count"])
                final_idle_check.update(status="idle" if confirmation["idle"] else "busy",
                                        **confirmation)
                if not confirmation["idle"]:
                    idle_since = None
                    status = "busy"
        now = time.monotonic()
        if checked_before_exec:
            final_idle_check["elapsed_seconds"] = round(now - started, 3)
        record = dict(status=status, pid=os.getpid(), gpu_uuid=gpu_uuid,
                      elapsed_seconds=round(now - started, 3),
                      idle_seconds=round(now - idle_since, 3) if idle_since is not None else 0.0,
                      idle_threshold_seconds=idle_threshold, poll_seconds=poll_seconds,
                      resource_revision=RESOURCE_REVISION,
                      **details)
        if final_idle_check is not None:
            record["final_idle_check"] = final_idle_check
        write_status(status_file, record)
        if status != previous_status:
            print(json.dumps(record, ensure_ascii=False), flush=True)
            previous_status = status
        if status == "launching":
            for name in (IDLE_ENV, POLL_ENV):
                os.environ.pop(name, None)
            # execvp preserves the PID and inheritable lease FD
            # supplied by scheduler.Popen(pass_fds=...). Do not close that FD.
            os.execvp(command[0], command)
            raise RuntimeError("execvp unexpectedly returned")
        time.sleep(poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        assigned_uuid(args.gpu_uuid)
    except ValueError as error:
        parser.error(str(error))
    if len(args.command) < 2 or args.command[0] != "--":
        parser.error("pass the original command after --")
    try:
        wait_and_exec(args.gpu_uuid, args.status_file, args.command[1:])
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
