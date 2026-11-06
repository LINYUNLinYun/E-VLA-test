"""One resource waiter; do not interfere with existing GPU jobs."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

from preparations.idea2.layer_bench import require_idle_gpu, gpu_snapshot
from .run import R, ROOT, GPU, LEASES


def main():
    out = R / "waiting.json"
    while True:
        leases = json.loads(LEASES.read_text())
        try:
            snapshot = require_idle_gpu(GPU)
            free = not leases.get("3", {}).get("active")
        except RuntimeError:
            snapshot = gpu_snapshot(GPU)
            free = False
        state = dict(status="ready" if free else "waiting_for_idle_gpu", gpu=3,
            time_utc=dt.datetime.now(dt.timezone.utc).isoformat(), snapshot=snapshot,
            poll_seconds=30, other_processes_untouched=True)
        out.write_text(json.dumps(state, indent=2)+"\n")
        print(state["status"], state["time_utc"], flush=True)
        if free:
            command = [sys.executable, "-u", "-m", "results.idea2.simple_labels_20261026.run",
                       "--receipt-name", "capture_execution.json", "--pipeline-alignment", "off"]
            state.update(status="running", command=command)
            out.write_text(json.dumps(state, indent=2)+"\n")
            with (R/"capture.console.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state.update(status="passed" if result.returncode==0 else "failed",exit_code=result.returncode)
            out.write_text(json.dumps(state, indent=2)+"\n")
            raise SystemExit(result.returncode)
        time.sleep(30)


if __name__ == "__main__":
    main()
