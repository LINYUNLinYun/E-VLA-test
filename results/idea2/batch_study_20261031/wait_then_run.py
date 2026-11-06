"""Wait for a genuinely free GPU, then run one bounded first-stage job.

This is an execution receipt/launcher, not a model change. The benchmark owns
the atomic shared lease and rechecks occupancy before CUDA initialization.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LEASES = ROOT / "results/idea1_v3_3/external_gpu_leases.json"
AFFINITIES = {0: range(32, 48), 1: range(64, 80), 2: range(0, 16), 3: range(96, 112)}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def snapshot():
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    devices = subprocess.check_output(command, text=True, timeout=20)
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True, timeout=20)
    occupied = {}
    for line in processes.splitlines():
        pid, uuid, memory = (part.strip() for part in line.split(","))
        occupied.setdefault(uuid, []).append(dict(pid=int(pid), memory_mib=memory))
    leases = json.loads(LEASES.read_text())
    rows = []
    for line in devices.splitlines():
        index, uuid, memory, utilization = (part.strip() for part in line.split(","))
        rows.append(dict(index=int(index), uuid=uuid, memory_mib=float(memory),
                         utilization=float(utilization), processes=occupied.get(uuid, []),
                         active_lease=leases.get(index, {}).get("active", False)))
    return rows


def candidates(rows):
    return sorted((row for row in rows if not row["processes"] and not row["active_lease"]
                   and row["memory_mib"] <= 16 and row["utilization"] <= 1),
                  key=lambda row: [3, 0, 2, 1].index(row["index"]) if row["index"] in (0, 1, 2, 3) else 99)


def command_for(device, output):
    preparation = ROOT / "results/idea2/batch_preparation_20261031/experiment_plan/plan.json"
    plan = json.loads(preparation.read_text())
    job = next(row for row in plan["jobs"] if row["id"] == "smoke_b1_paired_prepared")
    argv = list(job["argv"])
    argv[argv.index("--gpu-uuid") + 1] = device["uuid"]
    argv[argv.index("--output") + 1] = str(output)
    environment = dict(os.environ, **job["environment"])
    environment["CUDA_VISIBLE_DEVICES"] = device["uuid"]
    return argv, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=18)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    receipt = HERE / "initial_execution.json"
    output = HERE / "stage1_smoke_b1_paired"
    if args.dry_run:
        command, _ = command_for(dict(uuid="GPU-UNASSIGNED"), output)
        print(json.dumps(dict(status="dry_run", command=command, gpu_queried=False)))
        return
    if receipt.exists() or output.exists():
        raise RuntimeError("An existing initial execution must be inspected, never restarted blindly")
    state = dict(status="waiting", pid=os.getpid(), start_utc=now(), poll_count=0,
                 objective="First-stage batch results -> analysis and plan -> second-stage improvement and validation")
    save(receipt, state)
    deadline = time.monotonic() + args.max_wait_hours * 3600
    child = None
    try:
        while True:
            state["poll_count"] += 1
            state["last_poll_utc"] = now()
            try:
                rows = snapshot()
                state["devices"] = rows
                state.pop("last_poll_error", None)
                free = candidates(rows)
            except Exception as error:
                state["last_poll_error"] = f"{type(error).__name__}: {error}"
                free = []
            save(receipt, state)
            if free:
                # Graphics processes and an external startup race are checked
                # again by exclusive_gpu before the child imports CUDA.
                device = free[0]
                command, environment = command_for(device, output)
                usable = set(AFFINITIES.get(device["index"], os.sched_getaffinity(0))) & os.sched_getaffinity(0)
                if usable:
                    os.sched_setaffinity(0, usable)
                state.update(status="running", selected_gpu=device, command=command,
                             cpu_affinity=sorted(os.sched_getaffinity(0)), output=str(output))
                with (HERE / "stage1_smoke_b1_paired.console.log").open("x") as console:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                             stdout=console, stderr=subprocess.STDOUT)
                    state["child_pid"] = child.pid
                    save(receipt, state)
                    print("STARTED", device["uuid"], child.pid, flush=True)
                    child.wait()
                state.update(status="passed" if child.returncode == 0 else "failed",
                             exit_code=child.returncode, end_utc=now())
                result = output / "result.json"
                if result.is_file():
                    value = json.loads(result.read_text())
                    state["benchmark_status"] = value.get("status")
                    state["benchmark_error"] = value.get("error")
                save(receipt, state)
                print(json.dumps({key: state[key] for key in ("status", "exit_code", "output")}), flush=True)
                return child.returncode
            if time.monotonic() >= deadline:
                state.update(status="wait_deadline", end_utc=now())
                save(receipt, state)
                return 2
            if state["poll_count"] == 1 or state["poll_count"] % 20 == 0:
                print("WAITING", state["last_poll_utc"], "poll", state["poll_count"], flush=True)
            time.sleep(args.poll_seconds)
    except BaseException as error:
        state.update(status="launcher_failed", error=f"{type(error).__name__}: {error}", end_utc=now())
        save(receipt, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
