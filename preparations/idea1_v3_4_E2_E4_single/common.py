import hashlib,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_E4_single'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_E4_single_plan.md'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
LAYERS=tuple(range(1,18))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)

def select():
    pairs=[]
    for p in sorted((ROOT/'results/idea1_v3_4/workers').glob('gpu*/mechanism_selection.json')):
        pairs+=json.loads(p.read_text())
    pairs.sort(key=lambda p:(SUITES.index(p['case']['suite']),p['case']['task'],p['current']['request']))
    assert len(pairs)==160
    counts={}
    for i,p in enumerate(pairs):
        task=p['case']['task'];k=(p['case']['suite'],task);ordinal=counts.get(k,0);counts[k]=ordinal+1
        p.update(id=i,e2=ordinal==1,split='calibration' if task%2==0 else ('validation' if task in (1,5) else 'heldout'))
        ep=Path(p['current']['path']).parent
        rows=[json.loads(x) for x in (ep/'infer.jsonl').read_text().splitlines()]
        earlier=[r for r in rows if r['step']<=p['previous']['step']-15]
        p['previous2']=None
        if earlier:
            r=earlier[-1]
            p['previous2']=dict(path=r['input_path'],sha256=r['input_file_sha256'],seed=r['seed'],request=r['replan'],step=r['step'],input_hash=r['input_hash'])
    assert len(counts)==40 and set(counts.values())=={4}
    return pairs

def metrics(ref,actions,rawref,raw,h=5):
    a,b=np.asarray(ref)[:h],np.asarray(actions)[:h]
    return dict(translation_mm=float(np.linalg.norm(np.clip(a[:,:3],-1,1)-np.clip(b[:,:3],-1,1),axis=-1).mean()*50),
        rotation_l2=float(np.linalg.norm(a[:,3:6]-b[:,3:6],axis=-1).mean()),gripper_sign=float(np.mean((a[:,6]<0)!=(b[:,6]<0))),
        raw_rms6=float(np.sqrt(np.mean((np.asarray(rawref)[:h,:6]-np.asarray(raw)[:h,:6])**2))))
