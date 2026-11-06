import hashlib,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_cycle3.common import SUITES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_verified_paths'
BASE=ROOT/'results/idea1_v3_4_E2_cycle5_spaced'
PRIOR=ROOT/'results/idea1_v3_4_E2_cycle5_q4_v12'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_verified_paths_plan.md'
SETTINGS=Path(__file__).with_name('selection.json')
PHASES=('main200',)

def conditions():
    s=json.loads(SETTINGS.read_text());method=s['selection'];q=s['projections']
    masks={12:[2,3,5,14,16,17],14:[3,4,9,11]}
    cfgs=[dict(id=f'v{b}_extrapolate_d2_verify_p{p}',model='pi0',mode='verified',budget=b,skip_count=18-b,adjacency_rule='spaced' if method=='jacobian' else 'max2',period=p,selection=method,projections=q,
               skip=masks[b] if method=='fixed' else [],compensation='extrapolate',tau=.10,protect_gripper=True,kalman=False,H=50,R=5,full_steps=10,fast_steps=2,n=200,phase='main200') for b in (12,14) for p in (3,5)]
    return cfgs+[dict(id='full_d10',model='pi0',mode='full',budget=18,period=1,selection='none',projections=0,skip=[],compensation='none',tau=.10,protect_gripper=True,kalman=False,H=50,R=5,full_steps=10,fast_steps=None,n=200,phase='main200')]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(c):return f"{c['suite']}_task{c['task']:02d}_init{c['init']:02d}"
def directory(name,c):return OUT/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
def source(m,cfg,c):
    d=directory(cfg['id'],c)
    return (d,False) if (d/'complete.json').exists() else (None,False)

def freeze():
    assert json.loads((OUT/'cpu_gate.json').read_text())['passed']
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    old=json.loads((BASE/'manifest.json').read_text());prior=json.loads((PRIOR/'manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in prior['source_files'].items())
    controls={};baseline={};legacy=ROOT/'results/idea1_v3_4_E4_closedloop'
    assert json.loads((legacy/'audit.json').read_text())['passed']
    for c in old['cases']:
        d=BASE/'v14_jacobian_none'/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
        controls[ck(c)]=dict(directory=str(d),identity_sha256=sha(d/'identity.json'))
        if c['init']==19:
            d=legacy/'full10_reference'/'episodes'/c['suite']/f"task{c['task']:02d}"/'init19'
            assert json.loads((d/'result.json').read_text())['valid'] and (d/'complete.json').exists()
            baseline[ck(c)]=dict(directory=str(d),hashes={f:sha(d/f) for f in ('result.json','identity.json','infer.jsonl','arrays.npz','complete.json')})
    assert len(baseline)==40
    files=dict(prior['source_files']);files.update({str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    for p in ('preparations/idea1_v3_3/rules.py','preparations/idea1_v3_3/policy.py','policies/realtime-vla-flash/scripts/spec/triton/pi0_spec_infer.py'):
        files[p]=sha(ROOT/p)
    refs={str(SETTINGS.relative_to(ROOT)):sha(SETTINGS),str((BASE/'manifest.json').relative_to(ROOT)):sha(BASE/'manifest.json'),str((PRIOR/'manifest.json').relative_to(ROOT)):sha(PRIOR/'manifest.json'),str((legacy/'audit.json').relative_to(ROOT)):sha(legacy/'audit.json')}
    cfgs=conditions();m=dict(conditions=cfgs,cases=old['cases'],controls=controls,baseline_candidates=baseline,source_files=files,references=refs,plan_sha256=sha(PLAN),
                          dependency=str(PRIOR/'complete.json'),dependency_audit=str(PRIOR/'audit.json'),max_gpus=3,max_actors=2,target_records=1000,
                          planned_new_min=960,planned_new_max=1000,potential_baseline_reuse=40,phases=PHASES,time=time.time(),selection=json.loads(SETTINGS.read_text()))
    m['manifest_hash']=digest(m);atomic(OUT/'manifest.json',m)
    groups=[dict(key=f'verified_{s}_task{t:02d}',phase='main200',suite=s,task=t,state='pending',cases=[dict(case=c,state='pending') for c in m['cases'] if c['suite']==s and c['task']==t]) for t in range(10) for s in SUITES]
    atomic(OUT/'queue.json',dict(groups=groups));atomic(OUT/'progress.json',dict(conditions={c['id']:dict(target=200,completed=0,success=0,reused=0,new=0,seen=[]) for c in cfgs}))

def register(cfg,c,d,r,reused=False):
    with lock(OUT/'progress.lock'):
        p=json.loads((OUT/'progress.json').read_text());row=p['conditions'][cfg['id']];key=ck(c)
        if key in row['seen']:return
        atomic(OUT/'completions'/cfg['id']/f'{key}.json',dict(case=c,directory=str(d),success=r['success'],actions=r['actions'],wall_s=r['wall_s'],reused=reused,time=time.time()))
        row['seen'].append(key);row['completed']+=1;row['success']+=int(r['success']);row['reused' if reused else 'new']+=1
        atomic(OUT/'progress.json',p)

def progress():return {k:{f:v for f,v in r.items() if f!='seen'} for k,r in json.loads((OUT/'progress.json').read_text())['conditions'].items()}

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
