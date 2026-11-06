import argparse,concurrent.futures,hashlib,json,os,random,subprocess,time,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,OUT
from preparations.idea1_v3.io import atomic

CASES=[dict(suite=s,task=t,init=19) for s,t in [('libero_spatial',4),('libero_10',3),('libero_10',8),('libero_10',9)]]
REV=OUT/'initialization_revision'

def probe():
    from preparations.idea1_v3_1.rollout import Environments
    from preparations.idea1_v2_1.runtime_client import official_init_states,input_digest
    checks=[]
    for case in CASES:
        hashes=[]
        for repeat in range(2):
            random.seed(7);np.random.seed(7);pool=Environments(reuse=False)
            try:
                env,definition,_,selected,_=pool.acquire(case['suite'],case['task'])
                env.seed(7);env.reset();obs=env.set_init_state(np.asarray(official_init_states(selected,case['task'])[19]).copy())
                for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                model_obs={k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
                hashes.append(input_digest(model_obs,definition.language))
            finally:pool.close()
        check=dict(case=case,hashes=hashes,passed=hashes[0]==hashes[1]);checks.append(check);print(json.dumps(check),flush=True)
    result=dict(passed=all(r['passed'] for r in checks),checks=checks,time=time.time())
    atomic(REV/'probe.json',result)
    if not result['passed']:raise RuntimeError('Fresh environment initialization probe failed')

def evaluate(port,worker,workers):
    from preparations.idea1_v3_4.rollout import episode
    from preparations.idea1_v3_1.rollout import Environments
    m=json.loads((OUT/'manifest.json').read_text())
    for index,case in enumerate(CASES):
        if index%workers!=worker:continue
        cfgs=m['conditions'];offset=index%len(cfgs)
        for cfg in cfgs[offset:]+cfgs[:offset]:
            d=OUT/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/'init19'
            random.seed(7);np.random.seed(7);pool=Environments(reuse=False)
            try:
                r=episode(d,port,case,cfg,pool,m['manifest_hash'])
                atomic(d/'initialization_revision.json',dict(revision='fresh-environment-seed-before-construction',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=time.time()))
                print(json.dumps(dict(case=case,condition=cfg['id'],success=r['success'])),flush=True)
            finally:pool.close()

def controller():
    from .controller import MODEL,EVAL,free,cross_gate
    from preparations.idea1_v3_1.common import environment
    from preparations.idea1_v3_4.dispatch import render_environment
    from preparations.idea1_v2_1.runtime_client import rpc
    assert json.loads((REV/'probe.json').read_text())['passed']
    m=json.loads((OUT/'manifest.json').read_text());gpus=[g for g in (0,2,3) if free(g)]
    if not gpus:raise RuntimeError('No free authorized GPU for initialization revision')
    for case in CASES:
        for cfg in m['conditions']:
            d=OUT/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/'init19'
            target=REV/'original'/cfg['id']/case['suite']/f"task{case['task']:02d}"/'init19'
            target.parent.mkdir(parents=True,exist_ok=True);d.rename(target)
    atomic(REV/'manifest.json',dict(cases=CASES,gpus=gpus,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=time.time()))
    atomic(OUT/'status.json',dict(status='repairing_initialization',cases=CASES,time=time.time()))
    def worker(gpu,index):
        wd=REV/f'gpu{gpu}';wd.mkdir(exist_ok=True);port=21010+gpu
        env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        render,_=render_environment(env,gpu);model=None
        try:
            with (wd/'service.log').open('a') as log:
                model=subprocess.Popen([str(MODEL),'-u','-m','preparations.idea1_v3_4_v12_compensation.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                begin=time.monotonic()
                while True:
                    if model.poll() is not None:raise RuntimeError('Repair model exited')
                    if time.monotonic()-begin>900:raise TimeoutError('Repair readiness')
                    try:rpc(port,dict(op='health'));break
                    except (OSError,ConnectionError):time.sleep(1)
                cross_gate(wd)
                with (wd/'evaluate.log').open('a') as f:
                    r=subprocess.run([str(EVAL),'-u','-m','preparations.idea1_v3_4_v12_compensation.repair_initialization','--evaluate','--port',str(port),'--worker',str(index),'--workers',str(len(gpus))],cwd=ROOT,env=render,stdout=f,stderr=subprocess.STDOUT)
                if r.returncode:raise RuntimeError('Repair evaluation failed')
        finally:
            if model is not None and model.poll() is None:
                model.terminate()
                try:model.wait(timeout=20)
                except subprocess.TimeoutExpired:model.kill();model.wait()
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as ex:
        fs=[ex.submit(worker,g,i) for i,g in enumerate(gpus)]
        for f in fs:f.result()
    checks=[]
    for case in CASES:
        hashes=[]
        for cfg in m['conditions']:
            d=OUT/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/'init19'
            hashes.append(json.loads((d/'infer.jsonl').open().readline())['input_hash'])
        checks.append(dict(case=case,passed=len(set(hashes))==1,hashes=hashes))
    assert all(r['passed'] for r in checks),'Repaired cases initial observations differ'
    atomic(REV/'audit.json',dict(passed=True,checks=checks,replaced_episodes=28,time=time.time()))
    from .finish import main
    main()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--port',type=int);p.add_argument('--worker',type=int);p.add_argument('--workers',type=int);a=p.parse_args()
    REV.mkdir(parents=True,exist_ok=True)
    try:
        if a.probe:probe()
        elif a.evaluate:evaluate(a.port,a.worker,a.workers)
        else:controller()
    except Exception:
        atomic(REV/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
