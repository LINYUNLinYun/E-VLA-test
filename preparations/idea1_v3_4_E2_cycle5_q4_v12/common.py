import hashlib,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_cycle3.common import SUITES
from preparations.idea1_v3_4_E2_adjacency_200.common import combinations,choose,legal,longest_run
from preparations.idea1_v3_4_E2_cycle5_spaced.common import config as old_config

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_cycle5_q4_v12'
BASE=ROOT/'results/idea1_v3_4_E2_cycle5_spaced'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle5_q4_v12_plan.md'
PHASES=('main200',)
CONDITIONS=[dict(old_config(12,'jacobian',comp,200,'main200'),id=f'v12_jacobian_q4_{comp}',base_id=f'v12_jacobian_{comp}',projections=4) for comp in ('none','extrapolate')]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(c):return f"{c['suite']}_task{c['task']:02d}_init{c['init']:02d}"
def directory(name,c):return OUT/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
def source(m,cfg,c):
    d=directory(cfg['id'],c)
    return (d,False) if (d/'complete.json').exists() else (None,False)

def freeze():
    from .model import projection_weights
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    assert json.loads((OUT/'cpu_gate.json').read_text())['passed']
    old=json.loads((BASE/'manifest.json').read_text());assert json.loads((BASE/'audit.json').read_text())['passed']
    assert all(sha(ROOT/p)==h for p,h in old['source_files'].items())
    baseline={}
    for cfg in CONDITIONS:
        name=cfg['base_id'];baseline[name]={}
        for c in old['cases']:
            d=BASE/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
            r=json.loads((d/'result.json').read_text());assert r['valid'] and r['case']==c and (d/'complete.json').exists()
            baseline[name][ck(c)]=dict(directory=str(d),hashes={f:sha(d/f) for f in ('result.json','identity.json','infer.jsonl','complete.json','arrays.npz')})
    files=dict(old['source_files']);files.update({str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    for p in ('preparations/idea1_v3_4_E2_comp_error_multivjp/model.py','preparations/idea1_v3_4_E2_comp_error_cost/model.py','preparations/idea1_v3_4_E2_comp_error_cost/verify.py'):
        files[p]=sha(ROOT/p)
    refs={str((BASE/f).relative_to(ROOT)):sha(BASE/f) for f in ('manifest.json','audit.json','summary.csv','costs.csv','complete.json')}
    bank=ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json';refs[str(bank.relative_to(ROOT))]=sha(bank)
    m=dict(conditions=CONDITIONS,cases=old['cases'],small_cases=old['small_cases'],baseline=baseline,source_files=files,references=refs,plan_sha256=sha(PLAN),
           projection_seed=340610,projection_weights=projection_weights(4).tolist(),max_gpus=3,max_actors=2,new_episodes=400,target_records=400,reused_episodes=0,phases=PHASES,
           legal_combinations=924,schedule=['Full-D10']+['Fast-D2']*4,time=time.time())
    m['manifest_hash']=digest(m);atomic(OUT/'manifest.json',m)
    groups=[dict(key=f'q4_cycle5_{suite}_task{task:02d}',phase='main200',suite=suite,task=task,state='pending',cases=[dict(case=c,state='pending') for c in m['cases'] if c['suite']==suite and c['task']==task]) for task in range(10) for suite in SUITES]
    assert len(groups)==40 and all(len(g['cases'])==5 for g in groups)
    atomic(OUT/'queue.json',dict(groups=groups));atomic(OUT/'progress.json',dict(conditions={c['id']:dict(target=200,completed=0,success=0,reused=0,new=0,seen=[]) for c in CONDITIONS}))

def register(cfg,c,d,r):
    with lock(OUT/'progress.lock'):
        p=json.loads((OUT/'progress.json').read_text());row=p['conditions'][cfg['id']];key=ck(c)
        if key in row['seen']:return
        atomic(OUT/'completions'/cfg['id']/f'{key}.json',dict(case=c,directory=str(d),success=r['success'],actions=r['actions'],wall_s=r['wall_s'],time=time.time()))
        row['seen'].append(key);row['completed']+=1;row['success']+=int(r['success']);row['new']+=1
        atomic(OUT/'progress.json',p)

def progress():
    return {k:{f:v for f,v in row.items() if f!='seen'} for k,row in json.loads((OUT/'progress.json').read_text())['conditions'].items()}

def claim_group(gpu):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text())
        for g in q['groups']:
            if g['state']=='pending':
                g.update(state='running',gpu=gpu,started=time.time());atomic(OUT/'queue.json',q)
                return {k:g[k] for k in ('key','phase','suite','task')}

def claim_case(group,client):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text());g=next(g for g in q['groups'] if g['key']==group)
        for r in g['cases']:
            if r['state']=='pending':
                r.update(state='running',client=client,started=time.time());atomic(OUT/'queue.json',q);return r['case']

def finish_case(group,c):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text());g=next(g for g in q['groups'] if g['key']==group)
        next(r for r in g['cases'] if r['case']==c).update(state='complete',finished=time.time());atomic(OUT/'queue.json',q)

def finish_group(group):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text());g=next(g for g in q['groups'] if g['key']==group)
        assert all(r['state']=='complete' for r in g['cases']);g.update(state='complete',finished=time.time());atomic(OUT/'queue.json',q)

def release(gpu):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text())
        for g in q['groups']:
            if g.get('gpu')==gpu and g['state']=='running':
                g['state']='pending'
                for r in g['cases']:
                    if r['state']=='running':r['state']='pending'
        atomic(OUT/'queue.json',q)

def pending():return any(g['state']!='complete' for g in json.loads((OUT/'queue.json').read_text())['groups'])
def phase_done(phase):return not pending()

if __name__=='__main__':freeze()
