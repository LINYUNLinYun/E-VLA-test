import json,os,subprocess,time
from pathlib import Path
from .common import ROOT,OUT,OLD,PLAN,dump,sha
PY=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
def available():
    a=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={s.split(',')[0].strip() for s in a.splitlines() if s.strip()}
    d=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    return [(int(g),u) for g,u,m in ([x.strip() for x in s.split(',')] for s in d.splitlines()) if u not in busy and float(m)<=1500]
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'manifest.json').exists():
        dump(OUT/'manifest.json',dict(plan_sha256=sha(PLAN),sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},reference_selection_sha256=sha(OLD/'selection.json'),time=time.time()))
    m=json.loads((OUT/'manifest.json').read_text());assert all(sha(ROOT/p)==h for p,h in m['sources'].items())
    from .quality import main as quality
    quality();pending=['fp32','bf16'];running={};failures={k:0 for k in pending};last_scan=-1e9
    while pending or running:
        for dtype,w in list(running.items()):
            code=w['process'].poll()
            if code is None:continue
            w['log'].close();dump(OUT/f'{dtype}_exit.json',dict(code=code,gpu=w['gpu'],time=time.time()));del running[dtype]
            if code:
                failures[dtype]+=1
                if failures[dtype]>1:raise RuntimeError(f'{dtype} failed twice')
                pending.append(dtype)
        if pending and time.monotonic()-last_scan>=30:
            last_scan=time.monotonic()
            for gpu,uuid in available():
                if not pending:break
                dtype=pending.pop(0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                log=(OUT/f'{dtype}_attempt{failures[dtype]+1}.log').open('w')
                process=subprocess.Popen([str(PY),'-u','-m','preparations.idea1_v3_4_E2_single_action.worker','--dtype',dtype],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                running[dtype]=dict(process=process,gpu=gpu,log=log);dump(OUT/f'{dtype}_process.json',dict(pid=process.pid,gpu=gpu,uuid=uuid,time=time.time()))
        dump(OUT/'status.json',dict(status='running' if running else 'waiting_gpu',pending=pending,running={k:v['gpu'] for k,v in running.items()},completed_inputs={k:len(list((OUT/k).glob('sample*/complete.json'))) for k in ('fp32','bf16')},time=time.time()))
        time.sleep(5)
    from .finish import main as finish
    finish()
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        dump(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
