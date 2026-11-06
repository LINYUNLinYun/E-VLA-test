import hashlib,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_residual_replay_quick'
OLD=ROOT/'results/idea1_v3_4_v12_compensation'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_residual_replay_quick_plan.md'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'
MASKS=dict(all_reuse=list(range(1,18)),prefix3_reuse=list(range(4,18)),v12_reuse_reference=[2,3,5,14,16,17])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(c):return f"{c['suite']}_task{c['task']:02d}_init{c['init']:02d}"
def directory(name,c):return OUT/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    configs={name:dict(id=name,model='pi0',skip=skip,period=2,H=50,R=5,D=2,refresh_D=10) for name,skip in MASKS.items()}
    cases=[dict(suite=s,task=t,init=19) for t in range(10) for s in SUITES]
    refs={}
    for c in cases:
        d=OLD/'periodic_reuse'/'episodes'/c['suite']/f"task{c['task']:02d}"/'init19'
        r=json.loads((d/'result.json').read_text());assert r['valid']
        refs[key(c)]=dict(directory=str(d),initial_input_hash=json.loads((d/'infer.jsonl').open().readline())['input_hash'],
            hashes={n:sha(d/n) for n in ('result.json','identity.json','infer.jsonl','complete.json')})
    sources=list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in ('preparations/idea1_v3_4_v12_compensation/backend.py','preparations/idea1_v3_4/rollout.py',
        'preparations/idea1_v3/policy.py','preparations/idea1_v3/safe_qk.py','preparations/idea1_v2_2/pi0_adapter.py','policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py')]
    m=dict(conditions=[configs[k] for k in ('all_reuse','prefix3_reuse')],reference_config=configs['v12_reuse_reference'],cases=cases,references=refs,
        source_files={str(p.relative_to(ROOT)):sha(p) for p in sources},plan_sha256=sha(PLAN),gpus=[0,2,3],time=time.time())
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    atomic(OUT/'manifest.json',m)
if __name__=='__main__':freeze()
