import json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4_residual_replay_quick.common import ROOT,MODEL,EVAL,SUITES,sha,key
OUT=ROOT/'results/idea1_v3_4_v12_refresh_sweep'
OLD=ROOT/'results/idea1_v3_4_residual_replay_quick'
PREFIX=ROOT/'results/idea1_v3_4_prefix3_refresh_sweep'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_v12_refresh_sweep_plan.md'
MASK=[2,3,5,14,16,17]
def directory(name,c):return OUT/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
def record(d):
    r=json.loads((d/'result.json').read_text());assert r['valid'] and (d/'complete.json').exists()
    return dict(directory=str(d),initial_input_hash=json.loads((d/'infer.jsonl').open().readline())['input_hash'],
        hashes={n:sha(d/n) for n in ('result.json','identity.json','infer.jsonl','complete.json','arrays.npz')})
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    for previous in (OLD,PREFIX):
        assert json.loads((previous/'final_complete.json').read_text())['passed']
        manifest=json.loads((previous/'manifest.json').read_text())
        assert all(sha(ROOT/p)==h for p,h in manifest['source_files'].items())
    old=json.loads((OLD/'manifest.json').read_text());cases=old['cases'];refs={};prefix={}
    configs={p:dict(id=f'v12_p{p}',model='pi0',mode='periodic',compensation='reuse',skip=MASK,period=p,H=50,R=5,D=2,refresh_D=10) for p in (2,4,6)}
    for c in cases:
        k=key(c);d=Path(old['references'][k]['directory']);refs[k]=record(d)
        assert all(sha(d/n)==h for n,h in old['references'][k]['hashes'].items())
        r=json.loads((d/'result.json').read_text());assert r['config']['skip']==MASK and r['config']['period']==2 and r['config']['compensation']=='reuse'
        prefix[k]={}
        for p in (2,4,6):
            d=(OLD/'prefix3_reuse' if p==2 else PREFIX/f'prefix3_p{p}')/'episodes'/c['suite']/f"task{c['task']:02d}"/'init19'
            prefix[k][f'prefix3_p{p}']=record(d)
            r=json.loads((d/'result.json').read_text());assert r['config']['skip']==list(range(4,18)) and r['config']['period']==p
            assert prefix[k][f'prefix3_p{p}']['initial_input_hash']==refs[k]['initial_input_hash']
    sources=list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in (
        'preparations/idea1_v3_4_prefix3_refresh_sweep/policy.py','preparations/idea1_v3_4_residual_replay_quick/backend.py',
        'preparations/idea1_v3_4_v12_compensation/backend.py','preparations/idea1_v3_4/rollout.py','preparations/idea1_v3/policy.py',
        'preparations/idea1_v3/safe_qk.py','preparations/idea1_v2_2/pi0_adapter.py','policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py')]
    m=dict(conditions=[configs[p] for p in (4,6)],reference_config=configs[2],cases=cases,references=refs,prefix_references=prefix,gpus=[0,2,3],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in sources},plan_sha256=sha(PLAN),
        timing_files={str(p):sha(p) for p in (OLD/'workers').glob('gpu*/benchmark.json')},time=time.time())
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    atomic(OUT/'manifest.json',m)
if __name__=='__main__':freeze()
