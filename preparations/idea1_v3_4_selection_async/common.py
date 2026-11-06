import json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4_selection_cost.common import ROOT,SUITES,sha,write_csv,ck
from .runtime import OUT
MODULE='preparations.idea1_v3_4_selection_async'


def conditions():
    old=json.loads((ROOT/'results/idea1_v3_4_verified_paths/manifest.json').read_text())
    template=next(c for c in old['conditions'] if c['mode']=='verified' and c['budget']==12 and c['period']==3)
    base=dict(template,budget=12,selection_stride=1,projections=4,batched=False,
        aggregation='layer_rms_sum',asynchronous=True,attention='original',cuda_graph=True,n=40)
    result=[dict(base,id='full_d10',mode='full',period=1,compensation='none',selection='none')]
    result.append(dict(base,id='v12_p3_q4_serial_sync',asynchronous=False,n=8))
    for period in (3,5):
        result.append(dict(base,id=f'v12_p{period}_q4_serial_async',period=period))
        if period==3:
            result.append(dict(base,id='v12_p3_q4_batch_async',batched=True))
        for q in (1,2):
            for aggregation,label in (('layer_rms_sum','layer'),('signed_sum_rms','combo')):
                result.append(dict(base,id=f'v12_p{period}_q{q}_{label}',period=period,projections=q,
                    batched=q>1,aggregation=aggregation))
    for stride in (2,3):
        result.append(dict(base,id=f'v12_p3_q4_stride{stride}',selection_stride=stride))
    return result


def directory(condition,case):
    return OUT/'closed_loop'/condition/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def cases():return [dict(suite=s,task=t,init=19) for t in range(10) for s in SUITES]


def eligible(cfg,case):return cfg['n']==40 or case['task'] in (0,2)


def freeze():
    path=OUT/'closed_loop_manifest.json'
    if path.exists():return json.loads(path.read_text())
    source={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    original=ROOT/'results/idea1_v3_4_verified_paths/manifest.json'
    for filename in json.loads(original.read_text())['source_files']:source[filename]=sha(ROOT/filename)
    m=dict(cases=cases(),conditions=conditions(),source_files=source,
        protocol_reference=str(original),protocol_reference_sha256=sha(original),
        compression_sha256=sha(OUT/'compression/complete.json'),overlap_sha256=sha(OUT/'overlap/complete.json'),
        env_seed=7,settling=10,H=50,R=5,full_d=10,fast_d=2,budget=12,
        stride_counts_actual_full=True,residual_refresh_every_full=True,
        sparse_refresh_authorization='New active goal: after compression and demonstrated overlap',
        actions_paced=False,closed_loop_scope='40-case exploratory evaluation, not official 50-init benchmark',
        action_atol=1e-5,action_rtol=1e-5,gradient_atol=1e-4,gradient_rtol=1e-3,
        time=time.time())
    m['manifest_hash']=digest(m);atomic(path,m);return m


def verify(m):
    for filename,expected in m['source_files'].items():assert sha(ROOT/filename)==expected,filename


def progress():
    m=json.loads((OUT/'closed_loop_manifest.json').read_text());rows={}
    for cfg in m['conditions']:
        valid=[]
        for case in m['cases']:
            d=directory(cfg['id'],case)
            if (d/'complete.json').exists():valid.append(json.loads((d/'result.json').read_text()))
        rows[cfg['id']]=dict(completed=len(valid),target=cfg['n'],success=sum(r['success'] for r in valid))
    return rows
