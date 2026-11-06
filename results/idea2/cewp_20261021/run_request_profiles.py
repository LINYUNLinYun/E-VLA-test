import json,os,subprocess,time
from pathlib import Path
root=Path('results/idea2/cewp_20261021')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENPI_DATA_HOME='/root/nfs/embodied/benchmark/shared/checkpoints/openpi',PYTHONPATH='.:policies/openpi/src')
py='policies/openpi/.venv/bin/python'
plans=[(10,'full',None,['eager_serial','eager_dual_prefill_first','graph_serial','graph_ae_high','graph_pre_high']),
 (10,'vision','vision_only',['graph_serial_high','graph_ae_high']),
 (10,'all','all',['graph_serial_high','graph_ae_high']),
 (10,'vlm','vlm_only',['graph_serial_high','graph_pre_high']),
 (1,'full',None,['graph_serial','graph_ae_high']),
 (1,'vision','vision_only',['graph_serial_high','graph_ae_high']),
 (5,'full',None,['graph_serial','graph_ae_high']),
 (5,'vision','vision_only',['graph_serial_high','graph_ae_high'])]
manifest=[]
for steps,kind,selective,modes in plans:
 name=f'profile_d{steps}_{kind}';prefix=f'nsys/idea2_experiment/cewp_d{steps}_{kind}'
 command=['nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',f'--capture-range-end=repeat:{len(modes)}','--force-overwrite=false','-o',prefix,py,'-m','preparations.idea2.cewp_bench','--checkpoint','shared/checkpoints/openpi/pi05_libero_pytorch_idea2','--output',str(root/name),'--steps',str(steps),'--pairs','--warmup','5','--continuous-count','5','--reset-every','0','--profile','--continuous',*modes]
 if selective:command+=['--selective',selective]
 row=dict(name=name,command=command,start=time.time(),status='running');manifest.append(row);(root/'request_profile_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('START',name,flush=True)
 with (root/(name+'.log')).open('w') as log:
  process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
  row['pid']=process.pid;(root/'request_profile_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  code=process.wait()
 row['profile_exit_code']=code
 if code:row['status']='failed';(root/'request_profile_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');raise SystemExit(code)
 row['traces']=[]
 for number,mode in enumerate(modes,1):
  trace=f'{prefix}.{number}.nsys-rep';sql=f'{prefix}.{number}.sqlite';report=str(root/name/f'trace_{mode}.json')
  with (root/name/f'export_{number}.log').open('w') as log:
   subprocess.run(['nsys','export','--type','sqlite','--output',sql,trace],stdout=log,stderr=subprocess.STDOUT,check=True)
  with (root/name/f'analysis_{number}.log').open('w') as log:
   subprocess.run([py,'-m','preparations.idea2.cewp_trace',sql,'--output',report],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  row['traces'].append(dict(mode=mode,nsys=trace,sqlite=sql,analysis=report))
 row.update(status='done',end=time.time());(root/'request_profile_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('DONE',name,flush=True)
