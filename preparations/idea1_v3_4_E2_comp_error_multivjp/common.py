import hashlib,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_cycle3.common import ROOT,SUITES,choose,legal,coefficients

OUT=ROOT/'results/idea1_v3_4_E2_comp_error_multivjp'
REFERENCE=ROOT/'results/idea1_v3_4_E2_comp_error_closedloop'
STATIC_REFERENCE=ROOT/'results/idea1_v3_4_E2_cycle3_extrapolate'
COST=ROOT/'results/idea1_v3_4_E2_comp_error_cost'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_comp_error_multivjp_plan.md'
CONDITIONS=[dict(id=f'v12_comp_error_q{q}_extrapolate',base_id='v12_comp_error_extrapolate',budget=12,skip_count=6,projections=q,method='comp_error',model='pi0',mode='periodic',period=3,full_steps=10,fast_steps=2,H=50,R=5,compensation='extrapolate',warm_full_count=2,jacobian_model_steps=10) for q in [2,3,4]]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    assert json.loads((REFERENCE/'audit.json').read_text())['passed']
    assert json.loads((OUT/'cpu_verification.json').read_text())['passed']
    cost=json.loads((COST/'summary.json').read_text());assert cost['passed'] and cost['actual_gpu_measurement']
    old=json.loads((REFERENCE/'manifest.json').read_text())
    assert json.loads((STATIC_REFERENCE/'static_mask_reference.json').read_text())['passed']
    files=list(Path(__file__).parent.glob('*.py'))
    files += [ROOT/p for p in ['preparations/idea1_v3/policy.py','preparations/idea1_v3_4/rollout.py','preparations/idea1_v3_4_E2_max2_repair/evaluate.py','preparations/idea1_v3_4_E2_cycle3/common.py','preparations/idea1_v3_4_E2_cycle3/selection.py','preparations/idea1_v3_4_E2_cycle3/policy.py','preparations/idea1_v3_4_E2_cycle3_full10_jacobian/model.py','preparations/idea1_v3_4_E2_online_time_weights/selector.py','preparations/idea1_v3_4_E2_E4_single/torch_model.py','preparations/idea1_v3_4_E2_cycle3_extrapolate/backend.py','preparations/idea1_v3_4_v12_compensation/backend.py','preparations/idea1_v3_4_E2_comp_error_cost/model.py','policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py']]
    refs={str(p.relative_to(ROOT)):sha(p) for p in [REFERENCE/'manifest.json',REFERENCE/'summary.csv',REFERENCE/'audit.json',STATIC_REFERENCE/'static_mask_reference.json',COST/'manifest.json',COST/'summary.json',REFERENCE/'prefix_audit.json']}
    m=dict(conditions=CONDITIONS,cases=old['cases'],source_files={str(p.relative_to(ROOT)):sha(p) for p in files},plan_sha256=sha(PLAN),references=refs,projection_seed=340610,projection_weights=__import__('preparations.idea1_v3_4_E2_comp_error_multivjp.model',fromlist=['projection_weights']).projection_weights(4).tolist(),warm_full_count=2,steady_cycle_start_replan=9,schedule=['full10','fast2','fast2'],time=time.time());m['manifest_hash']=digest(m)
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
