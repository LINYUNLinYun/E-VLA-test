import hashlib,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_online_time_weights'
OLD=ROOT/'results/idea1_v3_4_E2_E4_single'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_online_time_weights_plan.md'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
CONDITIONS=[dict(id=name,model='pi0',mode='online',D=2,H=50,R=5,alpha=alpha,direction=direction) for name,alpha,direction in [
    ('uniform_random',[.2]*5,'random'),('linear_random',[x/15 for x in range(1,6)],'random'),
    ('uniform_fixed',[.2]*5,'fixed'),('exponential_fixed',[x/31 for x in [1,2,4,8,16]],'fixed')]]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def coefficients(config):
    v=np.random.default_rng(340710).choice([-1.,1.],(4,7))[0] if config['direction']=='random' else np.ones(7)
    return (np.asarray(config['alpha'])[:,None]*v[None,:]).astype(np.float32)
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=json.loads((ROOT/'results/idea1_v3_4_E2_closedloop/manifest.json').read_text())['cases']
    sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for name in ['preparations/idea1_v3_4/rollout.py','preparations/idea1_v3/policy.py','preparations/idea1_v3_4_E2_E4_single/torch_model.py','preparations/idea1_v3_4_E2_single_action/common.py']:
        sources[name]=sha(ROOT/name)
    m=dict(conditions=CONDITIONS,cases=cases,source_files=sources,plan_sha256=sha(PLAN),checkpoint=json.loads((OLD/'model_identity.json').read_text())['protocol']['checkpoint_sha256'],constraint=dict(skip_count=4,max_consecutive=2),selection='online every replan gate=1 single VJP',time=time.time())
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    atomic(OUT/'manifest.json',m);atomic(OUT/'queue.json',dict(cases=[dict(case=c,state='pending') for c in cases]))
def claim(worker):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text())
        for r in q['cases']:
            if r['state']=='pending':r.update(state='running',worker=worker,time=time.time());atomic(OUT/'queue.json',q);return r['case']
def finish_case(case):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text());next(r for r in q['cases'] if r['case']==case).update(state='complete',time=time.time());atomic(OUT/'queue.json',q)
def release(worker):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text())
        for r in q['cases']:
            if r['state']=='running' and r.get('worker')==worker:r['state']='pending'
        atomic(OUT/'queue.json',q)
def pending():return sum(r['state']=='pending' for r in json.loads((OUT/'queue.json').read_text())['cases'])
if __name__=='__main__':freeze()
