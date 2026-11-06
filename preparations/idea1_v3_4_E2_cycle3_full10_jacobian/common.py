import hashlib,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_cycle3.common import ROOT,SUITES,choose,legal,coefficients
OUT=ROOT/'results/idea1_v3_4_E2_cycle3_full10_jacobian'
REFERENCE=ROOT/'results/idea1_v3_4_E2_cycle3'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle3_full10_jacobian_plan.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);ref=json.loads((REFERENCE/'manifest.json').read_text());assert json.loads((REFERENCE/'audit.json').read_text())['passed']
    conditions=[dict(c,jacobian_model_steps=10) for c in ref['conditions'] if c['method']=='jacobian']
    sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for name in ['preparations/idea1_v3/policy.py','preparations/idea1_v3_4/rollout.py','preparations/idea1_v3_4_E2_cycle3/policy.py','preparations/idea1_v3_4_E2_cycle3/selection.py','preparations/idea1_v3_4_E2_cycle3/common.py','preparations/idea1_v3_4_E2_online_time_weights/selector.py','preparations/idea1_v3_4_E2_E4_single/torch_model.py']:
        sources[name]=sha(ROOT/name)
    m=dict(conditions=conditions,cases=ref['cases'],source_files=sources,plan_sha256=sha(PLAN),reference_manifest_sha256=sha(REFERENCE/'manifest.json'),reference_summary_sha256=sha(REFERENCE/'summary.csv'),schedule=['full10','fast2','fast2'],jacobian_model_steps=10,output_steps=[1,2,3,4,5],projection='five temporal exponential weights times seven +1 weights; one VJP',time=time.time());m['manifest_hash']=digest(m)
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
