import hashlib,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/idea1_v3_4_v12_compensation'
OUT=ROOT/'results/idea1_v3_4_v12_compensation_200'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_v12_compensation_200_plan.md'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
MASK=(2,3,5,14,16,17)
MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'

def casekey(c):return f"{c['suite']}_task{c['task']:02d}_init{c['init']:02d}"
def directory(name,c):return OUT/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((OLD/'manifest.json').read_text());assert json.loads((OLD/'final_complete.json').read_text())['passed']
    assert all(sha(ROOT/p)==h for p,h in old['source_files'].items())
    cfgs=old['conditions']+[dict(id='periodic_'+k,model='pi0',mode='periodic',compensation=k,period=2,H=50,R=5,D=2,refresh_D=10,skip=list(MASK)) for k in ('channel','hadamard','linear')]
    cases=[dict(suite=s,task=t,init=i) for t in range(10) for s in SUITES for i in (20,21,22,23,19)]
    reuse={};old_hashes={}
    for c in cases:
        if c['init']!=19:continue
        key=casekey(c);reuse[key]={};initial=[]
        for cfg in old['conditions']:
            d=OLD/cfg['id']/'episodes'/c['suite']/f"task{c['task']:02d}"/'init19'
            r=json.loads((d/'result.json').read_text());assert r['valid'] and r['config']==cfg
            h=json.loads((d/'infer.jsonl').open().readline())['input_hash'];initial.append(h)
            reuse[key][cfg['id']]=dict(directory=str(d),manifest_hash=r['manifest_hash'],
                hashes={n:sha(d/n) for n in ('result.json','identity.json','infer.jsonl','complete.json')},initial_input_hash=h)
        assert len(set(initial))==1
        old_hashes[key]=initial[0]
    sources=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'preparations/idea1_v3_4_v12_compensation').glob('*.py'))
    sources += [ROOT/p for p in ('preparations/idea1_v3_4/rollout.py','preparations/idea1_v3/policy.py','preparations/idea1_v3/safe_qk.py','preparations/idea1_v2_2/pi0_adapter.py','policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py')]
    m=dict(conditions=cfgs,cases=cases,reuse=reuse,old_initial_hashes=old_hashes,source_files={str(p.relative_to(ROOT)):sha(p) for p in sources},
        plan_sha256=sha(PLAN),adapter_sha256=sha(OLD/'adapters.pt'),gpus=[0,2,3],environment_workers_per_gpu=2,
        timing_files={str(p):sha(p) for p in (OLD/'workers').glob('gpu*/benchmark.json')},time=time.time())
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    atomic(OUT/'manifest.json',m)

if __name__=='__main__':freeze()
