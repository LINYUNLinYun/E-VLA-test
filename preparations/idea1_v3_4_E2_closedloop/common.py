import csv,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_closedloop'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_closedloop_plan.md'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')

def freeze():
    import hashlib
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'results/idea1_v3_4_E2_E4_single/E2_requests.csv'
    with path.open() as f:rows=[r for r in csv.DictReader(f) if float(r['gate'])==0 and int(r['task'])%2==0]
    scores={k:{l:[] for l in range(1,18)} for k in ('cosine4','jacobian4','action_mse4')}
    ids=sorted({r['id'] for r in rows},key=int);assert len(ids)==20
    for case in ids:
        a=[r for r in rows if r['id']==case and r['group']=='translation']
        for rank,r in enumerate(sorted(a,key=lambda r:(-float(r['layer_cosine']),int(r['layer'])))):
            scores['cosine4'][int(r['layer'])].append(rank)
        for name,field in [('jacobian4','predicted_rms'),('action_mse4','actual_rms')]:
            for group in ('translation','rotation','gripper'):
                a=[r for r in rows if r['id']==case and r['group']==group]
                for rank,r in enumerate(sorted(a,key=lambda r:(float(r[field]),int(r['layer'])))):
                    scores[name][int(r['layer'])].append(rank)
    ranks={name:{l:float(np.mean(v)) for l,v in layers.items()} for name,layers in scores.items()}
    masks={name:sorted(sorted(ranks[name],key=lambda l:(ranks[name][l],l))[:4]) for name in ranks}
    masks.update(v14=[3,4,9,11],full_d2=[])
    configs=[];aliases={};seen={}
    for name,mask in masks.items():
        k=tuple(mask)
        if k in seen:aliases[name]=seen[k];continue
        seen[k]=name;configs.append(dict(id=name,model='pi0',mode='pure',skip=mask,D=2,H=50,R=5))
    cases=[dict(suite=s,task=t,init=i) for t in (1,3,5,7,9) for s in SUITES for i in (19,20)]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    sources.update({p:sha(ROOT/p) for p in ('preparations/idea1_v3_4/rollout.py','preparations/idea1_v3/policy.py','preparations/idea1_v2_2/pi0_adapter.py','policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py')})
    m=dict(conditions=configs,cases=cases,masks=masks,aliases=aliases,ranks=ranks,calibration_ids=ids,
        calibration_csv_sha256=sha(path),source_files=sources,plan_sha256=sha(PLAN),time=time.time(),gpus=[0,1,2,3])
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists():raise FileExistsError('Manifest already frozen')
    atomic(OUT/'manifest.json',m);print(json.dumps(masks),flush=True)

if __name__=='__main__':freeze()
