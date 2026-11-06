import hashlib
import json
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def conditions():
    return [dict(id='full_d10',model='pi0',experiment='E1',mode='full',full_steps=10,n=200),
        dict(id='v14_d2',model='pi0',experiment='E1',mode='pure',n=200),
        dict(id='periodic_v14',model='pi0',experiment='E1',mode='periodic',n=200),
        dict(id='periodic_old16',model='pi0',experiment='E1',mode='periodic',cutoff=16,n=200),
        dict(id='full_d2',model='pi0',experiment='E1',mode='full',full_steps=2,n=40),
        dict(id='periodic_old13',model='pi0',experiment='E1',mode='periodic',cutoff=13,n=40),
        dict(id='periodic_old12',model='pi0',experiment='E1',mode='periodic',cutoff=12,n=40),
        dict(id='E3_parent',model='pi0',experiment='E3',mode='verified',tau=.1,n=40)]


def cases(n):
    return [dict(suite=s,task=t,init=i) for i in range(14,14+n//40) for t in range(10) for s in SUITES]


def main():
    out=OUT/'manifest.json'
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    for name in ('preparations/idea1_v3/policy.py','preparations/idea1_v2_2/pi0_adapter.py','preparations/idea1_v3_3/policy.py','preparations/idea1_v3_3/rules.py','preparations/idea1_v3/safe_qk.py','policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py'):
        sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    plan=ROOT/'docs/experiments/idea1/idea1_v3_4_experiment_plan.md'
    m=dict(schema=1,conditions=conditions(),cases={c['id']:cases(c['n']) for c in conditions()},source_files=sources,
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),idea1_sha256=hashlib.sha256((ROOT/'idea1.md').read_bytes()).hexdigest(),
        H=50,R=5,batch=1,period=2,mask=[3,4,9,11],gpus=[0,2,3],seed='idea1-v22/1701 suite/task/init/request',video=False)
    m['manifest_hash']=digest(m)
    if out.exists(): raise FileExistsError(out)
    atomic(out,m); print(out)

if __name__=='__main__': main()
