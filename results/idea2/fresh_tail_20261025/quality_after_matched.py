"""One finite quality launch after matched checks pass and GPU0 owns the SM32 queue."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parent
W=R.parents[2]
for i in range(150):
    paths=[R/"MATCHED_READY.json"]
    states=[json.loads(p.read_text()) if p.exists() else {} for p in paths]
    if all(s.get("status")=="passed" for s in states):break
    if any(s.get("status") in ("failed","stopped_at_boundary") for s in states):
        raise RuntimeError("Required experiment failed; quality not launched")
    if i%3==0:print("Quality waiting for matched-input checks",flush=True)
    time.sleep(20)
else:raise RuntimeError("Finite quality prerequisite wait exhausted")
env=dict(os.environ,CUDA_DEVICE_MAX_CONNECTIONS="32",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONPATH=".:policies/openpi/src",OPENPI_DATA_HOME=str(W/"shared/checkpoints/openpi"),JAX_PLATFORMS="cpu",JAX_PLATFORM_NAME="cpu")
command=[str(W/"policies/openpi/.venv/bin/python"),"-u","-m","results.idea2.fresh_tail_20261025.quality","--execute","--output",str(R/"closedloop_gpu3_grouped"),"--gpu-uuid","GPU-555cde36-5e41-bc44-0c69-ea47358fd703","--egl-device","0","--port","19088","--cpus","96-111"]
with (R/"closedloop_gpu3_grouped.console.log").open("x") as log:
    code=subprocess.call(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
(R/"quality_launch_receipt.json").write_text(json.dumps(dict(status="passed" if code==0 else "failed",command=command,exit_code=code),indent=2)+"\n")
raise SystemExit(code)
