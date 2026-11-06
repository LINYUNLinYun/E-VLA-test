import hashlib,json
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4.manifest import ROOT,cases
OUT=ROOT/'results/idea1_v3_4_followup'
OLD=ROOT/'results/idea1_v3_4'


def conditions():
    return [dict(id='A17_all',model='pi0',experiment='E1',mode='periodic',cutoff=17,n=200),
            dict(id='B16_first',model='pi0',experiment='E1',mode='periodic',cutoff=16,first_only=True,n=200),
            dict(id='C17_first',model='pi0',experiment='E1',mode='periodic',cutoff=17,first_only=True,n=200)]


def freeze():
    pairs=[]
    for p in (OLD/'workers').glob('gpu*/mechanism_selection.json'): pairs+=json.loads(p.read_text())
    episodes=sorted({p['episode'] for p in pairs}); selected=[]
    for episode in episodes:
        group=sorted([p for p in pairs if p['episode']==episode],key=lambda p:p['current']['request'])
        selected += [group[0],group[-1]]
    assert len(selected)==80
    paths=list(Path(__file__).parent.glob('*.py'))
    original=json.loads((OLD/'manifest.json').read_text())
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in original['source_files']}
    sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    plan=ROOT/'docs/experiments/idea1/idea1_v3_4_followup_plan.md'
    m=dict(conditions=conditions(),cases={c['id']:cases(200) for c in conditions()},timing_pairs=selected,source_files=sources,
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),historical_manifest_sha256=hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest(),
        gpus=[0,3],shards=2,H=50,R=5,period=2,atol=1e-5,rtol=1e-5)
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists(): raise FileExistsError('Already frozen')
    atomic(OUT/'manifest.json',m)

if __name__=='__main__': freeze()
