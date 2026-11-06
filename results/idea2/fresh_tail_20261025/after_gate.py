"""Start an explicitly specified isolated matrix only after the gate passes."""
import json
from pathlib import Path
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
while True:
    gate = json.loads((R / "gate_matrix.execution.json").read_text())
    if gate["status"] == "passed":
        break
    if gate["status"] != "running":
        raise RuntimeError("GPU gate failed; formal matrix not launched")
    print("Waiting for the finite gate matrix", flush=True)
    time.sleep(20)
raise SystemExit(subprocess.call([sys.executable, str(R / "run_matrix.py"), *sys.argv[1:]]))
