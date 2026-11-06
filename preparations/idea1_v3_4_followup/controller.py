import concurrent.futures
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3.cli import verify_gpu
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3_4.dispatch import render_environment
from .common import ROOT,OUT,OLD
MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'


def results(name,old=False,screen=False):
    rows=[]
    for p in ((OLD if old else OUT)/name/'episodes').glob('*/task*/init*/result.json'):
        if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists(): continue
        r=json.loads(p.read_text())
        if r['valid'] and (not screen or r['case']['init']==14): rows.append((p.parent,r))
    return rows


def historical_case_gate(name):
    checks=[]
    for d,r in results(name):
        c=r['case']; old=OLD/'full_d10/episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
        current_rows=[json.loads(x) for x in (d/'infer.jsonl').read_text().splitlines()]
        old_rows=[json.loads(x) for x in (old/'infer.jsonl').read_text().splitlines()]
        with np.load(d/'arrays.npz') as a,np.load(old/'arrays.npz') as b:
            rawdiff=float(np.abs(a['raw_0']-b['raw_0']).max()); actdiff=float(np.abs(a['actions_0']-b['actions_0']).max())
            passed=current_rows[0]['input_hash']==old_rows[0]['input_hash'] and np.allclose(a['raw_0'],b['raw_0'],atol=1e-5,rtol=1e-5) and np.allclose(a['actions_0'],b['actions_0'],atol=1e-5,rtol=1e-5)
        checks.append(dict(case=c,passed=bool(passed),raw_max_abs=rawdiff,decoded_max_abs=actdiff))
    atomic(OUT/f'{name}_historical_gate.json',dict(passed=all(x['passed'] for x in checks),checks=checks))
    if not all(x['passed'] for x in checks): raise RuntimeError('Historical case equivalence failed')


def main():
    with lock(OUT/'controller.lock',blocking=False):
        m=json.loads((OUT/'manifest.json').read_text())
        for path,expected in m['source_files'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected: raise ValueError('Frozen source changed '+path)
        services=[]; logs=[]; render={}; ports={}
        atomic(OUT/'process.json',dict(pid=os.getpid(),started=time.time()))
        try:
            for shard,gpu in enumerate((0,3)):
                identity=verify_gpu(gpu); wd=OUT/'workers'/f'gpu{gpu}'; wd.mkdir(parents=True,exist_ok=True)
                env=environment(gpu); env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
                render[gpu],devices=render_environment(env,gpu); ports[gpu]=19540+gpu
                atomic(wd/'identity.json',dict(identity,gpu=gpu,shard=shard,egl_mapping=devices))
                log=(wd/'service.log').open('a'); logs.append(log)
                p=subprocess.Popen([str(MODEL),'-m','preparations.idea1_v3_4_followup.server','--port',str(ports[gpu]),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                services.append((gpu,p)); atomic(wd/'process.json',dict(pid=p.pid,port=ports[gpu]))
            for gpu,p in services:
                tick=time.monotonic()
                while True:
                    if p.poll() is not None: raise RuntimeError('Service failed gpu'+str(gpu))
                    try: health=rpc(ports[gpu],dict(op='health')); break
                    except (OSError,ConnectionError):
                        if time.monotonic()-tick>1200: raise TimeoutError('Readiness')
                        time.sleep(1)
                wd=OUT/'workers'/f'gpu{gpu}'; atomic(wd/'protocol.json',health)
                with np.load(wd/'gate_actions.npz') as got,np.load(OLD/'cross_service_reference.npz') as ref:
                    checks={k:float(np.abs(got[k]-ref[k]).max()) for k in ref.files}
                    passed=all(np.allclose(got[k],ref[k],atol=1e-5,rtol=1e-5) for k in ref.files)
                atomic(wd/'historical_service_gate.json',dict(passed=passed,max_abs=checks))
                if not passed: raise RuntimeError('Historical service gate failed')
            def evaluate(name,stage):
                atomic(OUT/'status.json',dict(status='running',stage=stage,condition=name,time=time.time()))
                running=[]; handles=[]
                for shard,(gpu,service) in enumerate(services):
                    log=(OUT/'workers'/f'gpu{gpu}'/f'{stage}_{name}.log').open('a'); handles.append(log)
                    cmd=[str(EVAL),'-m','preparations.idea1_v3_4_followup.rollout','--manifest',str(OUT/'manifest.json'),'--condition',name,'--output',str(OUT),'--port',str(ports[gpu]),'--shard',str(shard),'--shards','2','--stage',stage]
                    running.append(subprocess.Popen(cmd,cwd=ROOT,env=render[gpu],stdout=log,stderr=subprocess.STDOUT))
                while any(p.poll() is None for p in running):
                    if any(p.poll() is not None and p.returncode!=0 for p in running):
                        for p in running:
                            if p.poll() is None: p.terminate()
                        raise RuntimeError('Rollout failed '+name)
                    atomic(OUT/'progress.json',{c['id']:len(results(c['id'])) for c in m['conditions']}); time.sleep(5)
                for h in handles: h.close()
                if any(p.returncode for p in running): raise RuntimeError('Rollout failed '+name)
                historical_case_gate(name)
                print(json.dumps(dict(stage=stage,condition=name,n=len(results(name)),success=sum(r['success'] for d,r in results(name)))),flush=True)
            for name in ('A17_all','B16_first'): evaluate(name,'screen')
            with concurrent.futures.ThreadPoolExecutor(2) as ex:
                futures=[]
                for shard,(gpu,service) in enumerate(services):
                    pairs=[p for p in m['timing_pairs'] if p['case']['task']%2==shard]
                    futures.append(ex.submit(rpc,ports[gpu],dict(op='bench',pairs=pairs,output=str(OUT/'workers'/f'gpu{gpu}'/'timing.json'))))
                for f in futures: f.result()
            timing=[]
            for gpu,_ in services: timing+=json.loads((OUT/'workers'/f'gpu{gpu}'/'timing.json').read_text())['rows']
            means={path:float(np.mean([r['wall_ms'] for r in timing if r['path']==path])) for path in ('full10','v14','old16','old17')}
            cycle=dict(baseline=(means['full10']+2*means['v14'])/3,
                A17_all=(means['full10']+2*means['old17'])/3,
                B16_first=(means['full10']+means['old16']+means['v14'])/3,
                C17_first=(means['full10']+means['old17']+means['v14'])/3)
            old_success=sum(r['success'] for d,r in results('periodic_old16',old=True,screen=True))
            success={name:sum(r['success'] for d,r in results(name,screen=True)) for name in ('A17_all','B16_first')}
            effects={name:success[name]>=old_success+2 and cycle[name]<cycle['baseline'] for name in success}
            decision=dict(screen_success=success,old16_success=old_success,effects=effects,combo=all(effects.values()),request_ms=means,cycle_ms=cycle)
            atomic(OUT/'decision.json',decision)
            if decision['combo']:
                evaluate('C17_first','screen'); success['C17_first']=sum(r['success'] for d,r in results('C17_first',screen=True))
            expanded=[name for name in success if success[name]>=35 and cycle[name]<=cycle['baseline']]
            decision.update(screen_success=success,expanded=expanded); atomic(OUT/'decision.json',decision)
            for name in expanded: evaluate(name,'expand')
            atomic(OUT/'status.json',dict(status='rollouts_complete',time=time.time()))
        finally:
            for gpu,p in services:
                if p.poll() is None:
                    p.terminate()
                    try: p.wait(timeout=20)
                    except subprocess.TimeoutExpired: p.kill(); p.wait()
            for h in logs: h.close()
            atomic(OUT/'service_exits.json',{str(gpu):dict(pid=p.pid,exit=p.poll()) for gpu,p in services})
        from .finish import main as finish
        finish()

if __name__=='__main__':
    try: main()
    except Exception as exc:
        atomic(OUT/'failed.json',dict(time=time.time(),error=repr(exc)))
        raise
