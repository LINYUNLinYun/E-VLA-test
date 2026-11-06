"""Poll the authorized cards every 10 seconds and start immediately on release."""
import os,subprocess,time
from pathlib import Path
from experiments.pi05_hidden_probe.common import write_json

base=Path(__file__).resolve().parent;root=base.parents[2]
while True:
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    idle=[int(parts[0]) for line in snapshot.splitlines() for parts in [[v.strip() for v in line.split(',')]]
          if int(parts[0]) in [2,3] and int(parts[1])<1000 and int(parts[2])<2]
    write_json(base/'cached_validation_job.json',{'state':'waiting_for_gpu','checked_unix':time.time(),'poll_seconds':10,'eligible_gpus':[2,3],'gpu_snapshot':snapshot})
    if idle:break
    time.sleep(10)
gpu=idle[0]
env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(root),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
with (base/'cached_validation.log').open('a') as log:
    process=subprocess.Popen([str(root/'policies/openpi/.venv/bin/python'),'-u',str(base/'validate_cached_target.py')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    write_json(base/'cached_validation_job.json',{'state':'running','gpu':gpu,'pid':process.pid,'started_unix':time.time()})
    print('validation started immediately on GPU',gpu,'pid',process.pid,flush=True)
    code=process.wait()
write_json(base/'cached_validation_job.json',{'state':'finished','gpu':gpu,'pid':process.pid,'exit_code':code,'finished_unix':time.time()})
print('validation finished',code,flush=True)
