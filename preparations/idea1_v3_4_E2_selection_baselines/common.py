import hashlib,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_single_action.common import COMBOS
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_selection_baselines'
REFERENCE=ROOT/'results/idea1_v3_4_E2_online_time_weights'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_selection_baselines_plan.md'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
CONDITIONS=[dict(id='cosine_online',model='pi0',mode='baseline',D=2,H=50,R=5,method='cosine')]+[dict(id=f'random_online_s{i+1}',model='pi0',mode='baseline',D=2,H=50,R=5,method='random_online',random_seed=s) for i,s in enumerate([341101,341102,341103])]+[dict(id='random_episode_s1',model='pi0',mode='baseline',D=2,H=50,R=5,method='random_episode',random_seed=341101)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def random_mask(root,request_seed):
    seed=int.from_bytes(hashlib.sha256(f'E2/legal-random-mask/{root}/{request_seed}'.encode()).digest()[:8],'little')
    return (COMBOS[np.random.default_rng(seed).integers(len(COMBOS))]+1).tolist()
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);reference=json.loads((REFERENCE/'manifest.json').read_text());assert json.loads((REFERENCE/'audit.json').read_text())['passed']
    sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for name in ['preparations/idea1_v3_4/rollout.py','preparations/idea1_v3/policy.py','preparations/idea1_v3_4_E2_E4_single/torch_model.py','preparations/idea1_v3_4_E2_online_time_weights/selector.py','preparations/idea1_v3_4_E2_online_time_weights/server.py','preparations/idea1_v3_4_E2_single_action/common.py']:
        sources[name]=sha(ROOT/name)
    m=dict(conditions=CONDITIONS,cases=reference['cases'],source_files=sources,plan_sha256=sha(PLAN),reference_manifest_sha256=sha(REFERENCE/'manifest.json'),constraint=reference['constraint'],reference_condition='exponential_fixed',time=time.time())
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
