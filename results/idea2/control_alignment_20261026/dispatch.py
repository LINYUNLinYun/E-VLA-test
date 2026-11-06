"""One GPU owner, bounded continuation after the synchronous pilot finishes."""
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .run_async import ROOT, R, save


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    state = dict(status="waiting_for_sync", start_utc=now(), stages=[])
    output = R/"dispatch_execution.json"
    if output.exists():
        raise ValueError("Dispatcher already has a receipt; do not overwrite")
    save(output, state)
    try:
        deadline = time.monotonic()+1800
        while not (R/"sync_launcher.json").exists():
            if time.monotonic()>deadline:
                raise TimeoutError("Synchronous pilot did not finish")
            time.sleep(3)
        sync = json.loads((R/"sync_launcher.json").read_text())
        if sync["exit_code"] != 0:
            raise RuntimeError("Synchronous pilot failed technically")
        completed = json.loads((R/"sync_gate/execution.json").read_text())
        if completed["status"] != "passed" or not completed["source_unchanged"]:
            raise RuntimeError("Synchronous source/exit audit failed")
        for i, suites in enumerate((("libero_object", "libero_spatial"), ("libero_goal", "libero_10"))):
            command = [sys.executable, "-u", "-m", "results.idea2.control_alignment_20261026.run_async",
                       "--output", str(R/f"async_gate{i}"), "--port", str(19631+i), "--case-keys",
                       *(f"{s}/{t}/0" for s in suites for t in (0, 5))]
            row = dict(stage=i, command=command, start_utc=now(), status="running")
            state["stages"].append(row)
            state.update(status="running", current_stage=i)
            save(output, state)
            with (R/f"async_gate{i}.console.log").open("x") as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row.update(exit_code=completed.returncode, end_utc=now())
            if completed.returncode:
                raise RuntimeError("Async pilot process failed technically")
            receipt = json.loads((R/f"async_gate{i}/execution.json").read_text())
            if receipt["status"] != "passed" or not receipt["source_unchanged"]:
                raise RuntimeError("Async pilot source/exit audit failed")
            row.update(status="passed", successes=sum(c["result"]["success"] for c in receipt["cases"]))
            save(output, state)
        state.update(status="passed", pending=[])
    except BaseException:
        state.update(status="technical_failure", error=traceback.format_exc())
        raise
    finally:
        from preparations.idea2.layer_bench import gpu_snapshot
        snapshot = gpu_snapshot("GPU-555cde36-5e41-bc44-0c69-ea47358fd703")
        state.update(end_utc=now(), gpu_after=snapshot)
        # A failed child must have exited its model and EGL processes before
        # this owner releases its lease. Never touch another session's jobs.
        if not snapshot["processes"] and not snapshot["graphics_processes"]:
            path = ROOT/"results/idea1_v3_3/external_gpu_leases.json"
            with path.open("r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                leases = json.load(f)
                if leases["3"].get("owner") == "user-authorized idea2 control alignment pilot":
                    leases["3"].update(active=False, released_utc=now(), release_note="Bounded control-alignment pilot services exited")
                    f.seek(0)
                    json.dump(leases, f, indent=2)
                    f.write("\n")
                    f.truncate()
                    state["gpu_lease_released"] = True
        save(output, state)


if __name__ == "__main__":
    main()
