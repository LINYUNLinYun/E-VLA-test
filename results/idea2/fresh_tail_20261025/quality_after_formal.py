"""One finite quality launch after GPU2 formal and GPU3 matched checks pass."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parent
W=R.parents[2]
for i in range(150):
    paths=[R/"formal_sm16_matrix.execution.json",R/"MATCHED_READY.json"]
    states=[json.loads(p.read_text()) if p.exists() else {} for p in paths]
    if all(s.get("status")=="passed" for s in states):break
    if any(s.get("status") in ("failed","stopped_at_boundary") for s in states):
        raise RuntimeError("Required experiment failed; quality not launched")
    if i%3==0:print("Quality waiting for formal SM16 and matched-input checks",flush=True)
    time.sleep(20)
else:raise RuntimeError("Finite quality prerequisite wait exhausted")
env=dict(os.environ,CUDA_DEVICE_MAX_CONNECTIONS="32",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONPATH=".:policies/openpi/src",OPENPI_DATA_HOME=str(W/"shared/checkpoints/openpi"),JAX_PLATFORMS="cpu",JAX_PLATFORM_NAME="cpu")
command=[str(W/"policies/openpi/.venv/bin/python"),"-u","-m","results.idea2.fresh_tail_20261025.quality","--execute","--output",str(R/"closedloop_gpu2"),"--gpu-uuid","GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25","--egl-device","0","--port","19089","--cpus","0-15"]
with (R/"closedloop_gpu2.console.log").open("x") as log:
    code=subprocess.call(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
(R/"quality_launch_receipt.json").write_text(json.dumps(dict(status="passed" if code==0 else "failed",command=command,exit_code=code),indent=2)+"\n")
raise SystemExit(code)
