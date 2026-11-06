"""Reserve the released GPU0 and bound each no-graph optimization probe."""
import json,os,signal,subprocess,time
from pathlib import Path
import torch
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3.io import atomic
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/idea1_v3_4_selection_cost/batch200';OUT.mkdir(parents=True,exist_ok=True)
UUID='GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1'
with lock(Path('/tmp')/f'embodied-benchmark-gpu-{UUID}.lock'):
 torch.cuda.init();keepalive=torch.zeros(1,device='cuda')
 atomic(OUT/'owner.json',dict(pid=os.getpid(),gpu=0,gpu_uuid=UUID,time=time.time()))
 script=ROOT/'reports/idea1_v3_4_batch200_probe.py'
 while not script.exists():time.sleep(.2)
 results=[]
 for mode in ('profile','compile'):
  env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4')
  start=time.time()
  with (OUT/f'{mode}.log').open('w') as stream:
   process=subprocess.Popen([str(ROOT/'policies/realtime-vla-flash/.venv/bin/python'),'-u',str(script),'--mode',mode],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  atomic(OUT/'current.json',dict(mode=mode,pid=process.pid,time=time.time()))
  limit=120 if mode=='profile' else 300
  try:code=process.wait(timeout=limit);row=dict(mode=mode,exit_code=code,wall_s=time.time()-start,timeout=False)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   row=dict(mode=mode,exit_code=process.returncode,wall_s=time.time()-start,timeout=True,limit_s=limit)
  results.append(row);atomic(OUT/'runs.json',dict(results=results,time=time.time()))
 atomic(OUT/'probe_complete.json',dict(results=results,time=time.time()))
 # The current closed-loop queue excludes GPU0. Keep its lease until that
 # queue finishes, so its scheduler does not reload an ineligible worker.
 deadline=time.time()+1800
 while not (ROOT/'results/idea1_v3_4_selection_cost/complete.json').exists() and time.time()<deadline:time.sleep(5)
 atomic(OUT/'released.json',dict(time=time.time()))
