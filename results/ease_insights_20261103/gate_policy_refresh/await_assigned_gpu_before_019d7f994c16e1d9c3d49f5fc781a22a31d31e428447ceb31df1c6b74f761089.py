"""Wait for the scheduler-assigned GPU to stay idle, then replace this process.

Usage: python -m preparations.ease_insights_20261103.await_assigned_gpu \
    --gpu-uuid GPU-... --status-file /path/to/status.json -- /absolute/python ...

This gate does not allocate GPUs. The scheduler's inherited GPU lease remains
open across exec, and a busy GPU or failed query causes indefinite waiting.
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


def wait_and_exec(gpu_uuid, status_file, command):
    assigned_uuid(gpu_uuid)
    if not command:
        raise ValueError("An original command is required")
    started = time.monotonic()
    idle_since = None
    previous_status = None
    while True:
        details = {}
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
                status = "launching" if now - idle_since >= IDLE_SECONDS else "waiting"
            else:
                idle_since = None
                status = "busy"
        now = time.monotonic()
        record = dict(status=status, pid=os.getpid(), gpu_uuid=gpu_uuid,
                      elapsed_seconds=round(now - started, 3),
                      idle_seconds=round(now - idle_since, 3) if idle_since is not None else 0.0,
                      **details)
        write_status(status_file, record)
        if status != previous_status:
            print(json.dumps(record, ensure_ascii=False), flush=True)
            previous_status = status
        if status == "launching":
            # execvp preserves the PID, environment, and inheritable lease FD
            # supplied by scheduler.Popen(pass_fds=...). Do not close that FD.
            os.execvp(command[0], command)
            raise RuntimeError("execvp unexpectedly returned")
        time.sleep(POLL_SECONDS)


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
    wait_and_exec(args.gpu_uuid, args.status_file, args.command[1:])


if __name__ == "__main__":
    main()
