"""Frozen cycle-five comparison, all skipped layers non-adjacent."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_cycle3.common import SUITES
from preparations.idea1_v3_4_E2_adjacency_200.common import combinations,choose,legal,longest_run

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_cycle5_spaced'
BASE=ROOT/'results/idea1_v3_4_E2_selection_200'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle5_spaced_plan.md'
PHASES=('main200','random40')
RANDOM_SEED=341101

def random_mask(root,request_seed,k):
    # Cycle-independent seed mapping; future spaced comparisons can reuse it.
    seed=int.from_bytes(hashlib.sha256(f'E2/spaced/random/{k}/{root}/{request_seed}'.encode()).digest()[:8],'little')
    a=combinations(k,'spaced')
    return (a[np.random.default_rng(seed).integers(len(a))]+1).tolist()

def global_mask(k):
    return random_mask(RANDOM_SEED,'global_fixed',k)

def config(b,method,comp,n,phase):
    cfg=dict(id=f'v{b}_{method}_{comp}',budget=b,skip_count=18-b,method=method,compensation=comp,
             adjacency_rule='spaced',random_seed=RANDOM_SEED,model='pi0',mode='periodic',period=5,
             full_steps=10,fast_steps=2,H=50,R=5,n=n,phase=phase)
    if method=='global_fixed':cfg['fixed_skip']=global_mask(18-b)
    return cfg

CONDITIONS=([config(b,method,comp,200,'main200') for b in (14,12) for method in ('cosine','jacobian') for comp in ('none','extrapolate')]+
            [config(12,'global_fixed',comp,200,'main200') for comp in ('none','extrapolate')]+
            [config(b,method,comp,40,'random40') for b in (14,12) for method in ('random_online','random_fixed') for comp in ('none','extrapolate')])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(c):return f"{c['suite']}_task{c['task']:02d}_init{c['init']:02d}"
def directory(name,c):return OUT/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
def source(m,cfg,c):
    d=directory(cfg['id'],c)
    return (d,False) if (d/'complete.json').exists() else (None,False)

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    old=json.loads((BASE/'manifest.json').read_text())
    prior_root=ROOT/'results/idea1_v3_4_E2_adjacency_extrapolate_200'
    prior=json.loads((prior_root/'manifest.json').read_text())
    assert json.loads((prior_root/'audit.json').read_text())['passed']
    assert all(sha(ROOT/p)==h for p,h in prior['source_files'].items())
    baseline={}
    # Historical identity and cycle-three spaced Jacobian comparisons only.
    for b in (14,12):
        for comp in ('none','extrapolate'):
            name=f'v{b}_jacobian_spaced_{comp}'
            root=ROOT/f"results/idea1_v3_4_E2_adjacency{'_extrapolate' if comp=='extrapolate' else ''}_200"
            baseline[name]={}
            for c in old['cases']:
                d=root/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
                r=json.loads((d/'result.json').read_text())
                assert r['valid'] and r['case']==c and (d/'complete.json').exists()
                baseline[name][ck(c)]=dict(directory=str(d),hashes={f:sha(d/f) for f in ('result.json','identity.json','infer.jsonl','complete.json','arrays.npz')})
    files=dict(prior['source_files'])
    files.update({str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    references={str((prior_root/f).relative_to(ROOT)):sha(prior_root/f) for f in ('manifest.json','audit.json','summary.csv','complete.json')}
    references[str((BASE/'manifest.json').relative_to(ROOT))]=sha(BASE/'manifest.json')
    m=dict(conditions=CONDITIONS,cases=old['cases'],small_cases=old['small_cases'],baseline=baseline,
           source_files=files,references=references,plan_sha256=sha(PLAN),max_gpus=3,max_actors=2,
           new_episodes=sum(c['n'] for c in CONDITIONS),target_records=sum(c['n'] for c in CONDITIONS),
           reused_episodes=0,phases=PHASES,legal_combinations={k:len(combinations(k,'spaced')) for k in (4,6)},
           global_fixed=global_mask(6),schedule=['Full-D10']+['Fast-D2']*4,time=time.time())
    assert m['new_episodes']==2320 and len(m['cases'])==200 and len(m['small_cases'])==40
    m['manifest_hash']=digest(m);atomic(OUT/'manifest.json',m)
    groups=[]
    # One task group per GPU; small cases run alongside the main cases so the
    # task/prompt runtime and two independent environments are reused safely.
    for task in range(10):
        for suite in SUITES:
            cases=[dict(case=c,state='pending') for c in m['cases'] if c['suite']==suite and c['task']==task]
            groups.append(dict(key=f'cycle5_{suite}_task{task:02d}',phase='all',suite=suite,task=task,state='pending',cases=cases))
    atomic(OUT/'queue.json',dict(groups=groups))
    atomic(OUT/'progress.json',dict(conditions={c['id']:dict(target=c['n'],completed=0,success=0,reused=0,new=0,seen=[]) for c in CONDITIONS}))

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
