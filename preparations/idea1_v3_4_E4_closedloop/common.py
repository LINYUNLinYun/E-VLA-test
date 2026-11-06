import hashlib,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E4_closedloop'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E4_closedloop_plan.md'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
MASK=(3,4,9,11)

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    configs=[dict(id=name,model='pi0',mode=mode,compensation=patch,period=2,H=50,R=5,D=2,refresh_D=10,skip=list(MASK))
             for name,mode,patch in [('pure_v14','pure',None),('periodic_v14','periodic',None),
                 ('periodic_reuse','periodic','reuse'),('periodic_extrapolate','periodic','extrapolate'),
                 ('full10_reference','full',None)]]
    cases=[dict(suite=s,task=t,init=19) for t in range(10) for s in SUITES]
    sources=list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in (
        'preparations/idea1_v3_4/rollout.py','preparations/idea1_v3/policy.py','preparations/idea1_v3/safe_qk.py',
        'preparations/idea1_v2_2/pi0_adapter.py','policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py')]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    m=dict(conditions=configs,cases=cases,source_files={str(p.relative_to(ROOT)):sha(p) for p in sources},
           plan_sha256=sha(PLAN),gpus=[0,2,3],time=time.time(),mask=list(MASK),batch=1)
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    atomic(OUT/'manifest.json',m)

if __name__=='__main__':freeze()
