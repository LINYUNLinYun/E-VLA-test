"""One resource watcher; launches only on an otherwise unused candidate GPU."""
import json,os,subprocess,time
from pathlib import Path
from .common import ROOT,OUT,PLAN,dump,sha

PY=ROOT/'policies/realtime-vla-flash/.venv/bin/python'

def available():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={l.split(',')[0].strip() for l in apps.splitlines() if l.strip()}
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    out=[]
    for line in rows.splitlines():
        index,uuid,mem=[s.strip() for s in line.split(',')]
        if int(index) in (0,1,2) and uuid not in busy and int(mem)<=1500:out.append((int(index),uuid))
    return out

def execute(stage,module,args=()):
    attempts={0:0,1:0,2:1 if stage=='export' else 0}
    while True:
        cards=[c for c in available() if attempts[c[0]]<2]
        if not cards:
            dump(OUT/'status.json',dict(status='waiting_gpu',stage=stage,time=time.time()))
            print(json.dumps(dict(waiting_gpu=True,stage=stage,time=time.time())),flush=True);time.sleep(30);continue
        gpu,uuid=cards[0];env=dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',OPENPI_DISABLE_TORCH_COMPILE='1',
            PYTHONPATH=':'.join(map(str,[ROOT,ROOT/'policies/realtime-vla-flash/src',ROOT/'policies/realtime-vla-flash/packages/openpi-client/src'])))
        log=OUT/f'{stage}_gpu{gpu}_attempt{attempts[gpu]+1}.log';start=time.time()
        dump(OUT/'status.json',dict(status='running',stage=stage,gpu=gpu,log=str(log),started=start))
        with log.open('w') as f:
            p=subprocess.Popen([str(PY),'-u','-m',f'preparations.idea1_v3_4_E2_E4_single.{module}',*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            dump(OUT/f'{stage}_process.json',dict(pid=p.pid,gpu=gpu,uuid=uuid,started=start,log=str(log)))
            code=p.wait()
        dump(OUT/f'{stage}_exit.json',dict(code=code,gpu=gpu,wall_s=time.time()-start,time=time.time()))
        if code==0:return
        attempts[gpu]+=1
        text=log.read_text()
        if 'out of memory' in text.lower():
            print(json.dumps(dict(stage=stage,technical_oom=True,gpu=gpu)),flush=True);time.sleep(5);continue
        raise RuntimeError(f'{stage} failed: {log}')

def freeze():
    m=dict(selection_sha256=sha(OUT/'selection.json'),plan_sha256=sha(PLAN),model_identity=json.loads((OUT/'model_identity.json').read_text()),
        sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        probe=json.loads((OUT/'engineering_probe.json').read_text()),E2_requests=40,E4_requests=160,layers=list(range(1,18)),
        gpus=[0,1,2],excluded_gpus=[3],time=time.time())
    p=OUT/'manifest.json'
    if p.exists():
        old=json.loads(p.read_text());assert old['sources']==m['sources'] and old['plan_sha256']==m['plan_sha256']
    else:dump(p,m)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'export_complete.json').exists():execute('export','export')
    if not (OUT/'engineering_probe.json').exists():execute('probe','probe')
    freeze()
    execute('E2','e2')
    if not (OUT/'fit_complete.json').exists():execute('fit','fit')
    execute('E4','e4')
    subprocess.run([str(PY),'-m','preparations.idea1_v3_4_E2_E4_single.finish'],cwd=ROOT,check=True)

if __name__=='__main__':main()
