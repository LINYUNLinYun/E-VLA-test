import hashlib,json
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4.manifest import ROOT,SUITES

OUT=ROOT/'results/idea1_v3_4_E3_refine_sweep'
SOURCE=ROOT/'results/idea1_v3_4/E3_noise_control'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E3_refine_sweep_plan.md'
GRID=[.1,.3,.5,.7,.9,1.]
METRICS=['translation_mm','rotation_l2','gripper_sign','translation_l2','raw_rms6']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tag(q):return f'q{int(round(q*100)):03d}'


def phase2_qs(q):return sorted({round(max(.05,q-.1),2),round(q,2),round(min(.95,q+.1),2),1.})


def freeze():
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    original=json.loads((SOURCE/'manifest.json').read_text());selected=[]
    for r in original['selected']:
        if r['group']!='zero':continue
        r=dict(r);ref=SOURCE/'samples'/f"sample{r['id']:03d}"/'actions.npz'
        r.update(reference_arrays=str(ref),reference_sha256=sha(ref),split='calibration' if r['case']['task']%2==0 else 'heldout')
        selected.append(r)
    assert len(selected)==127 and sum(r['split']=='calibration' for r in selected)==65
    sources=dict(json.loads((ROOT/'results/idea1_v3_4_l12_14_first/manifest.json').read_text())['source_files'])
    for path,h in original['sources'].items():sources[str(Path(path).relative_to(ROOT))]=h
    for p in Path(__file__).parent.glob('*.py'):sources[str(p.relative_to(ROOT))]=sha(p)
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    m=dict(selected=selected,grid=GRID,steps=2,atol=1e-5,rtol=1e-5,sources=sources,plan_sha256=sha(PLAN),source_manifest_sha256=sha(SOURCE/'manifest.json'),
        gpus=[0,1,2,3],max_services=2,split='task even calibration / odd heldout',selection='gripper <= fresh; then min translation; fallback min gripper then translation; q<1 only')
    m['manifest_hash']=digest(m);atomic(OUT/'manifest.json',m)


def metrics(a,raw,ref,rawref,h):
    a=a[:h].astype(np.float64);ref=ref[:h].astype(np.float64);delta=a-ref
    return dict(translation_mm=float(50*np.linalg.norm(np.clip(a[:,:3],-1,1)-np.clip(ref[:,:3],-1,1),axis=1).mean()),
        rotation_l2=float(np.linalg.norm(delta[:,3:6],axis=1).mean()),gripper_sign=float(((a[:,6]>=0)!=(ref[:,6]>=0)).mean()),
        translation_l2=float(np.linalg.norm(delta[:,:3],axis=1).mean()),raw_rms6=float(np.sqrt(np.mean((raw[:h,:6].astype(np.float64)-rawref[:h,:6])**2))))


def balanced(rows,field):
    per_suite=[]
    for suite in SUITES:
        selected=[r for r in rows if r['suite']==suite];tasks=sorted({r['task'] for r in selected})
        assert tasks,suite
        per_suite.append(np.mean([np.mean([r[field] for r in selected if r['task']==t]) for t in tasks]))
    return float(np.mean(per_suite))


def select_q(rows,grid,fresh):
    # Caller supplies only the calibration split, matching-reference H5 rows.
    assert rows and all(r['split']=='calibration' and r['horizon']==5 and r['noise']==r['reference'] for r in rows)
    base=[r for r in rows if r['candidate']==fresh];bg=balanced(base,'gripper_sign')
    values=[]
    for q in grid:
        if q>=1:continue
        rs=[r for r in rows if r['candidate']==tag(q)]
        values.append(dict(q=q,gripper=balanced(rs,'gripper_sign'),translation_mm=balanced(rs,'translation_mm')))
    feasible=[r for r in values if r['gripper']<=bg+1e-12]
    best=min(feasible,key=lambda r:(r['translation_mm'],r['q'])) if feasible else min(values,key=lambda r:(r['gripper'],r['translation_mm'],r['q']))
    return dict(q=best['q'],has_gripper_not_higher_candidate=bool(feasible),fresh_gripper=bg,candidates=values,selected=best)


if __name__=='__main__':freeze()
