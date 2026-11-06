import hashlib,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_cycle3.common import ROOT,SUITES,choose,legal,coefficients
OUT=ROOT/'results/idea1_v3_4_E2_cycle3_extrapolate'
COS_REFERENCE=ROOT/'results/idea1_v3_4_E2_cycle3'
JAC_REFERENCE=ROOT/'results/idea1_v3_4_E2_cycle3_full10_jacobian'
REFERENCE=COS_REFERENCE
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle3_extrapolate_plan.md'
CONDITIONS=[dict(id=f'v{budget}_{method}_extrapolate',base_id=f'v{budget}_{method}',budget=budget,skip_count=18-budget,method=method,model='pi0',mode='periodic',period=3,full_steps=10,fast_steps=2,H=50,R=5,compensation='extrapolate') for budget in [14,12] for method in ['cosine','jacobian']]
def reference_root(cfg):return COS_REFERENCE if cfg['method']=='cosine' else JAC_REFERENCE
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);old=json.loads((COS_REFERENCE/'manifest.json').read_text());refs={}
    for root in [COS_REFERENCE,JAC_REFERENCE]:
        assert json.loads((root/'audit.json').read_text())['passed'];assert json.loads((root/'manifest.json').read_text())['cases']==old['cases']
        refs[str(root.relative_to(ROOT))]=dict(manifest_sha256=sha(root/'manifest.json'),summary_sha256=sha(root/'summary.csv'))
    sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for name in ['preparations/idea1_v3/policy.py','preparations/idea1_v3_4/rollout.py','preparations/idea1_v3_4_E2_cycle3/selection.py','preparations/idea1_v3_4_E2_cycle3_full10_jacobian/model.py','preparations/idea1_v3_4_E2_cycle3_full10_jacobian/server.py','preparations/idea1_v3_4_v12_compensation/backend.py']:
        sources[name]=sha(ROOT/name)
    m=dict(conditions=CONDITIONS,cases=old['cases'],source_files=sources,plan_sha256=sha(PLAN),references=refs,schedule=['full10','fast2','fast2'],captured_layers=list(range(1,18)),history_count=2,time=time.time());m['manifest_hash']=digest(m)
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
