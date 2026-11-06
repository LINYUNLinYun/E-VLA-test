"""Use an idle authorized GPU immediately after weights pass checksum validation."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from experiments.pi05_hidden_probe.common import write_json

base=Path(__file__).resolve().parent
root=base.parents[2]
python=root/'policies/openpi/.venv/bin/python'
checkpoint=root/'shared/checkpoints/pi05_hidden_probe/paligemma-3b-mix-224'
while True:
    status=json.loads((base/'target_download_status.json').read_text())
    if status['state']=='complete':break
    if status['state']=='model_access_still_denied':raise RuntimeError('External Target access denied')
    time.sleep(2)
os.environ['HF_HOME']='/root/.cache/huggingface'
from huggingface_hub import HfApi
info=HfApi().model_info(status['repository'],revision=status['revision'],files_metadata=True,token=True)
verified=[]
for sibling in info.siblings:
    if not sibling.rfilename.endswith('.safetensors'):continue
    lfs=sibling.lfs
    expected=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']
    path=checkpoint/sibling.rfilename
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=expected:raise RuntimeError('Model shard checksum mismatch: '+sibling.rfilename)
    verified.append({'file':sibling.rfilename,'sha256':actual,'bytes':path.stat().st_size})
    print('verified',sibling.rfilename,flush=True)
write_json(base/'target_weight_verification.json',{'verified':True,'repository':info.id,'revision':info.sha,'shards':verified})
jobs=[('external_readiness','external_readiness.py'),('target_profile','profile_target.py')]
running={};completed=[]
while jobs or running:
    for name,(process,gpu,log) in list(running.items()):
        code=process.poll()
        if code is not None:
            log.close();completed.append({'job':name,'gpu':gpu,'pid':process.pid,'exit_code':code})
            del running[name]
            print('finished',name,code,flush=True)
    if jobs:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        occupied={gpu for _,gpu,_ in running.values()}
        idle=[int(a[0]) for line in output.splitlines() for a in [[x.strip() for x in line.split(',')]]
              if int(a[0]) in [2,3] and int(a[0]) not in occupied and int(a[1])<1000 and int(a[2])<2]
        for gpu in idle:
            if not jobs:break
            name,script=jobs.pop(0)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(root),
                 'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
            log=(base/(name+'.log')).open('a')
            process=subprocess.Popen([str(python),'-u',str(base/script)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            running[name]=(process,gpu,log)
            print('launched',name,'gpu',gpu,'pid',process.pid,flush=True)
    write_json(base/'target_jobs.json',{'pending':[j[0] for j in jobs],
        'running':[{'job':n,'gpu':g,'pid':p.pid} for n,(p,g,_) in running.items()],'completed':completed})
    if jobs or running:time.sleep(10)
print('Target readiness and profiling jobs ended',flush=True)
