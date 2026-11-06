import hashlib,itertools,json,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_cycle3.common import SUITES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_adjacency_200'
BASE=ROOT/'results/idea1_v3_4_E2_selection_200'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_adjacency_200_plan.md'
PHASES=('adjacency200',)
CONDITIONS=[dict(id=f'v{b}_jacobian_{rule}_none',budget=b,skip_count=18-b,method='jacobian',compensation='none',adjacency_rule=rule,model='pi0',mode='periodic',period=3,full_steps=10,fast_steps=2,H=50,R=5,n=200,phase='adjacency200') for b in (14,12) for rule in ('unrestricted','spaced')]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(c):return f"{c['suite']}_task{c['task']:02d}_init{c['init']:02d}"
def directory(name,c):return OUT/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
def source(m,cfg,c):
    d=directory(cfg['id'],c)
    return (d,False) if (d/'complete.json').exists() else (None,False)

def longest_run(skip):
    run=best=0;previous=None
    for layer in sorted(skip):
        run=run+1 if previous is not None and layer==previous+1 else 1
        best=max(best,run);previous=layer
    return best

def legal(skip,k,rule):
    if len(skip)!=k or len(set(skip))!=k or any(i<1 or i>17 for i in skip):return False
    if rule=='unrestricted':return True
    if rule=='spaced':return longest_run(skip)<=1
    if rule=='max2':return longest_run(skip)<=2
    raise ValueError(rule)

@lru_cache(None)
def combinations(k,rule):
    a=np.array([c for c in itertools.combinations(range(17),k) if legal([i+1 for i in c],k,rule)],dtype=np.int64)
    assert len(a)>0;a.setflags(write=False);return a

def choose(score,k,rule):
    a=combinations(k,rule)
    return (a[np.argmin(np.asarray(score)[a].sum(1))]+1).tolist()

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    old=json.loads((BASE/'manifest.json').read_text())
    assert json.loads((BASE/'audit.json').read_text())['passed']
    assert all(sha(ROOT/p)==h for p,h in old['source_files'].items())
    baseline={}
    for b in (14,12):
        for method in ('jacobian','cosine'):
            name=f'v{b}_{method}_none';baseline[name]={}
            for c in old['cases']:
                ref=old['reuse'][name].get(ck(c));d=Path(ref['directory']) if ref else BASE/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
                r=json.loads((d/'result.json').read_text())
                assert r['valid'] and r['case']==c and (d/'complete.json').exists()
                baseline[name][ck(c)]=dict(directory=str(d),hashes={f:sha(d/f) for f in ('result.json','identity.json','infer.jsonl','complete.json','arrays.npz')})
    files={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    files.update(old['source_files'])
    for name in ('preparations/kv_reuse_probe/probe.py','preparations/idea1_v2_1/runtime_client.py','preparations/idea1_v2_2/common.py','preparations/idea1_v3_4_E2_selection_200/preflight.py'):
        files[name]=sha(ROOT/name)
    refs={str((BASE/f).relative_to(ROOT)):sha(BASE/f) for f in ('manifest.json','audit.json','summary.csv','complete.json','aggregation_recovery.json')}
    bank=ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json';refs[str(bank.relative_to(ROOT))]=sha(bank)
    m=dict(conditions=CONDITIONS,cases=old['cases'],baseline=baseline,references=refs,source_files=files,plan_sha256=sha(PLAN),max_gpus=3,max_actors=2,new_episodes=800,target_records=800,phases=PHASES,legal_combinations={f'{k}_{rule}':len(combinations(k,rule)) for k in (4,6) for rule in ('unrestricted','spaced','max2')},time=time.time())
    m['manifest_hash']=digest(m);atomic(OUT/'manifest.json',m)
    groups=[dict(key=f'adjacency200_{s}_task{t:02d}',phase='adjacency200',suite=s,task=t,state='pending',cases=[dict(case=c,state='pending') for c in m['cases'] if c['suite']==s and c['task']==t]) for t in range(10) for s in SUITES]
    assert len(groups)==40 and all(len(g['cases'])==5 for g in groups)
    atomic(OUT/'queue.json',dict(groups=groups))
    atomic(OUT/'progress.json',dict(conditions={c['id']:dict(target=200,completed=0,success=0,reused=0,new=0,seen=[]) for c in CONDITIONS}))

def register(cfg,c,d,r):
    with lock(OUT/'progress.lock'):
        p=json.loads((OUT/'progress.json').read_text());row=p['conditions'][cfg['id']];key=ck(c)
        if key in row['seen']:return
        atomic(OUT/'completions'/cfg['id']/f'{key}.json',dict(case=c,directory=str(d),success=r['success'],actions=r['actions'],wall_s=r['wall_s'],time=time.time()))
        row['seen'].append(key);row['completed']+=1;row['success']+=int(r['success']);row['new']+=1
        atomic(OUT/'progress.json',p)

def progress():
    p=json.loads((OUT/'progress.json').read_text())
    return {k:{f:v for f,v in r.items() if f!='seen'} for k,r in p['conditions'].items()}

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
            if r['state']=='pending':r.update(state='running',client=client,started=time.time());atomic(OUT/'queue.json',q);return r['case']

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
