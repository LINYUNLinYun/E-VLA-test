import hashlib,json
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4.manifest import ROOT,cases

OUT=ROOT/'results/idea1_v3_4_l15_first'
OLD=ROOT/'results/idea1_v3_4'
FOLLOWUP=ROOT/'results/idea1_v3_4_followup'
NAME='D15_first'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_l15_first_plan.md'
CONFIG=dict(id=NAME,model='pi0',experiment='E1',mode='periodic',cutoff=15,first_only=True,n=200)


def results(root=OUT,name=NAME,scope='all'):
    rows=[]
    for p in (root/name/'episodes').glob('*/task*/init*/result.json'):
        if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists():continue
        r=json.loads(p.read_text())
        if r['valid'] and (scope=='all' or (r['case']['init']==14)==(scope=='first40')):rows.append((p.parent,r))
    return rows


def freeze():
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    old=json.loads((OLD/'manifest.json').read_text())
    previous=json.loads((FOLLOWUP/'manifest.json').read_text())
    sources=dict(old['source_files'])
    sources.update(previous['source_files'])
    for p in Path(__file__).parent.glob('*.py'):
        sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    # Reject changed inherited scientific implementation before freezing the new run.
    for path,expected in sources.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    m=dict(conditions=[CONFIG],cases={NAME:cases(200)},timing_pairs=previous['timing_pairs'],source_files=sources,
        plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        historical_manifest_sha256=hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest(),
        followup_manifest_sha256=hashlib.sha256((FOLLOWUP/'manifest.json').read_bytes()).hexdigest(),
        gpus=[0,1,2,3],H=50,R=5,period=2,mask=[3,4,9,11],atol=1e-5,rtol=1e-5,
        expansion=dict(min_screen_success=35,max_cycle_ratio_to_v14=1.0))
    m['manifest_hash']=digest(m);atomic(OUT/'manifest.json',m)


if __name__=='__main__':freeze()
