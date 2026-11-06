"""Finish the diagnostic only after both unchanged Torch paths exit normally."""
import json
from pathlib import Path
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
ROOT = R.parents[2]


def main():
    while True:
        state = json.loads((R/"torch_capture_execution.json").read_text())
        if state["status"] == "failed":
            raise RuntimeError("Torch capture failed; inspect retained artifacts")
        if state["status"] == "passed" and state.get("lease_released"):
            break
        time.sleep(5)
    command = [sys.executable, "-u", "-m", "results.idea2.simple_labels_20261026.run_native_dynamic",
        "--only", "native_jax", "--stage-suffix", "_dynamic_dt",
        "--receipt-name", "native_dynamic_execution.json", "--pipeline-alignment", "off"]
    with (R/"native_dynamic.console.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
