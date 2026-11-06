import hashlib,json
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4.manifest import ROOT,cases

OUT=ROOT/'results/idea1_v3_4_l12_14_first'
OLD=ROOT/'results/idea1_v3_4'
FOLLOWUP=ROOT/'results/idea1_v3_4_followup'
L15=ROOT/'results/idea1_v3_4_l15_first'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_l12_14_first_plan.md'
CONFIGS=[dict(id=f'L{layer}_first',model='pi0',experiment='E1',mode='periodic',cutoff=layer,first_only=True,n=200) for layer in (12,13,14)]


def results(root=OUT,name=None,scope='all'):
    rows=[]
    for condition in ([name] if name is not None else [c['id'] for c in CONFIGS]):
        for p in (root/condition/'episodes').glob('*/task*/init*/result.json'):
            if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists():continue
            r=json.loads(p.read_text())
            if r['valid'] and (scope=='all' or (r['case']['init']==14)==(scope=='first40')):rows.append((p.parent,r))
    return rows


def progress():
    return {c['id']:dict(completed=len(rs),success=sum(r['success'] for d,r in rs)) for c in CONFIGS for rs in [results(name=c['id'])]}


def freeze():
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    previous=json.loads((L15/'manifest.json').read_text());sources=dict(previous['source_files'])
    for p in Path(__file__).parent.glob('*.py'):sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for path,h in sources.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    m=dict(conditions=CONFIGS,cases={c['id']:cases(200) for c in CONFIGS},timing_pairs=previous['timing_pairs'],source_files=sources,
        plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        historical_manifest_sha256=hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest(),
        followup_manifest_sha256=hashlib.sha256((FOLLOWUP/'manifest.json').read_bytes()).hexdigest(),
        l15_manifest_sha256=hashlib.sha256((L15/'manifest.json').read_bytes()).hexdigest(),
        gpus=[0,1,2,3],H=50,R=5,period=2,mask=[3,4,9,11],atol=1e-5,rtol=1e-5,
        expansion=dict(min_screen_success=35,max_cycle_ratio_to_v14=1.0))
    m['manifest_hash']=digest(m);atomic(OUT/'manifest.json',m)


if __name__=='__main__':freeze()
