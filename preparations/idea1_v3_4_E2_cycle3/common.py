import hashlib,itertools,json,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_cycle3'
REFERENCE=ROOT/'results/idea1_v3_4_E2_online_time_weights'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle3_plan.md'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
CONDITIONS=[dict(id=f'v{budget}_{method}',budget=budget,skip_count=18-budget,method=method,model='pi0',mode='periodic',period=3,full_steps=10,fast_steps=2,H=50,R=5,random_seed=341101) for budget in [14,12] for method in ['cosine','random_online','jacobian','random_fixed']]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@lru_cache(None)
def combinations(k):
    return np.array([c for c in itertools.combinations(range(17),k) if not any({i,i+1,i+2}<=set(c) for i in range(15))])
def choose(score,k):
    a=combinations(k);return (a[np.argmin(np.asarray(score)[a].sum(1))]+1).tolist()
def legal(skip,k):return len(skip)==k and len(set(skip))==k and min(skip)>=1 and max(skip)<=17 and not any({i,i+1,i+2}<=set(skip) for i in range(1,16))
def random_mask(root,request_seed,k):
    seed=int.from_bytes(hashlib.sha256(f'E2/cycle3/random/{k}/{root}/{request_seed}'.encode()).digest()[:8],'little');a=combinations(k)
    return (a[np.random.default_rng(seed).integers(len(a))]+1).tolist()
def coefficients():return (np.array([1.,2.,4.,8.,16.])[:,None]/31*np.ones((1,7))).astype(np.float32)
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);reference=json.loads((REFERENCE/'manifest.json').read_text())
    sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for name in ['preparations/idea1_v3/policy.py','preparations/idea1_v3_4/rollout.py','preparations/idea1_v3_4_E2_online_time_weights/selector.py','preparations/idea1_v3_4_E2_online_time_weights/server.py','preparations/idea1_v3_4_E2_E4_single/torch_model.py','preparations/idea1_v3_4_E2_max2_repair/evaluate.py']:
        sources[name]=sha(ROOT/name)
    m=dict(conditions=CONDITIONS,cases=reference['cases'],source_files=sources,plan_sha256=sha(PLAN),legal_combinations={k:len(combinations(k)) for k in [4,6]},schedule=['full10','fast2','fast2'],time=time.time())
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    atomic(OUT/'manifest.json',m);atomic(OUT/'queue.json',dict(cases=[dict(case=c,state='pending') for c in m['cases']]))
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
