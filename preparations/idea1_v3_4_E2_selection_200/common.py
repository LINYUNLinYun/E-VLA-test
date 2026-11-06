import hashlib,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_cycle3.common import ROOT,SUITES,legal,random_mask

OUT=ROOT/'results/idea1_v3_4_E2_selection_200'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_selection_200_plan.md'
BASE=ROOT/'results/idea1_v3_4_E2_cycle3'
JAC=ROOT/'results/idea1_v3_4_E2_cycle3_full10_jacobian'
EXTRAP=ROOT/'results/idea1_v3_4_E2_cycle3_extrapolate'
FIXED=ROOT/'results/idea1_v3_4_v12_compensation_200'
PHASES=('random40','expand200')

def config(budget,method,comp,n,phase):
    return dict(id=f'v{budget}_{method}_{comp}',budget=budget,skip_count=18-budget,method=method,compensation=comp,random_seed=341101,model='pi0',mode='periodic',period=3,full_steps=10,fast_steps=2,H=50,R=5,n=n,phase=phase)

CONDITIONS=([config(b,m,'extrapolate',40,'random40') for b in (14,12) for m in ('random_online','random_fixed')]+
            [config(b,m,c,200,'expand200') for b in (14,12) for m in ('cosine','jacobian') for c in ('none','extrapolate')])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(c):return f"{c['suite']}_task{c['task']:02d}_init{c['init']:02d}"
def gk(phase,suite,task):return f'{phase}_{suite}_task{task:02d}'
def directory(name,c):return OUT/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
def old_directory(root,name,c):return root/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"

def source(m,cfg,c):
    reuse=m['reuse'].get(cfg['id'],{}).get(ck(c))
    if reuse:return Path(reuse['directory']),True
    d=directory(cfg['id'],c)
    if (d/'complete.json').exists():return d,False
    return None,False

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    old=json.loads((BASE/'manifest.json').read_text());small=old['cases']
    cases=json.loads((FIXED/'manifest.json').read_text())['cases']
    assert len(cases)==200 and len(small)==40 and {ck(c) for c in small}<={ck(c) for c in cases}
    refs={};reuse={};fixed={};fixed_none={}
    for root in (BASE,JAC,EXTRAP):
        assert json.loads((root/'audit.json').read_text())['passed']
        for file in ('manifest.json','summary.csv','audit.json'):refs[str((root/file).relative_to(ROOT))]=sha(root/file)
    for c in cases:
        p=FIXED/'case_done'/f'{ck(c)}.json';record=json.loads(p.read_text());r=next(r for r in record['records'] if r['condition']=='periodic_extrapolate')
        d=Path(r['directory']);assert r['valid'] and (d/'complete.json').exists()
        fixed[ck(c)]=dict(directory=str(d),result_sha256=sha(d/'result.json'),identity_sha256=sha(d/'identity.json'),case_done_sha256=sha(p))
        nr=next(r for r in record['records'] if r['condition']=='periodic_v12');nd=Path(nr['directory'])
        fixed_none[ck(c)]=dict(directory=str(nd),result_sha256=sha(nd/'result.json'),identity_sha256=sha(nd/'identity.json'))
    for cfg in CONDITIONS:
        reuse[cfg['id']]={}
        if cfg['phase']!='expand200':continue
        if cfg['compensation']=='extrapolate':root=EXTRAP;name=f"v{cfg['budget']}_{cfg['method']}_extrapolate"
        else:root=JAC if cfg['method']=='jacobian' else BASE;name=f"v{cfg['budget']}_{cfg['method']}"
        for c in small:
            d=old_directory(root,name,c);r=json.loads((d/'result.json').read_text())
            assert r['valid'] and r['case']==c and (d/'complete.json').exists()
            reuse[cfg['id']][ck(c)]=dict(directory=str(d),manifest_hash=r['manifest_hash'],hashes={file:sha(d/file) for file in ('result.json','identity.json','infer.jsonl','complete.json')})
    files=list(Path(__file__).parent.glob('*.py'))
    files += [ROOT/p for p in ['preparations/idea1_v3/policy.py','preparations/idea1_v3/safe_qk.py','preparations/idea1_v2_2/pi0_adapter.py','preparations/idea1_v3_4/rollout.py','preparations/idea1_v3_4/analyze.py','preparations/idea1_v3_4_E2_max2_repair/evaluate.py','preparations/idea1_v3_4_E2_cycle3/common.py','preparations/idea1_v3_4_E2_cycle3/policy.py','preparations/idea1_v3_4_E2_cycle3/selection.py','preparations/idea1_v3_4_E2_cycle3_full10_jacobian/model.py','preparations/idea1_v3_4_E2_cycle3_full10_jacobian/policy.py','preparations/idea1_v3_4_E2_online_time_weights/selector.py','preparations/idea1_v3_4_E2_E4_single/torch_model.py','preparations/idea1_v3_4_E2_cycle3_extrapolate/backend.py','preparations/idea1_v3_4_E2_cycle3_extrapolate/policy.py','preparations/idea1_v3_4_v12_compensation/backend.py','policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py']]
    m=dict(conditions=CONDITIONS,cases=cases,small_cases=small,reuse=reuse,fixed=fixed,fixed_none=fixed_none,references=refs,source_files={str(p.relative_to(ROOT)):sha(p) for p in files},plan_sha256=sha(PLAN),max_gpus=3,max_actors=2,new_episodes=1440,reused_episodes=320,target_records=1760,phases=PHASES,time=time.time());m['manifest_hash']=digest(m)
    atomic(OUT/'manifest.json',m)
    groups=[]
    for phase in PHASES:
        for task in range(10):
            for suite in SUITES:
                owned=[c for c in (small if phase=='random40' else cases) if c['task']==task and c['suite']==suite]
                if not owned:continue
                rows=[dict(case=c,state='complete' if phase=='expand200' and ck(c) in {ck(s) for s in small} else 'pending') for c in owned]
                groups.append(dict(key=gk(phase,suite,task),phase=phase,suite=suite,task=task,state='pending',cases=rows))
    atomic(OUT/'queue.json',dict(groups=groups))
    progress=dict(conditions={cfg['id']:dict(target=cfg['n'],completed=0,success=0,reused=0,new=0,seen=[]) for cfg in CONDITIONS})
    for cfg in CONDITIONS:
        for key,ref in reuse[cfg['id']].items():
            r=json.loads((Path(ref['directory'])/'result.json').read_text());row=progress['conditions'][cfg['id']]
            row['completed']+=1;row['success']+=int(r['success']);row['reused']+=1;row['seen'].append(key)
    atomic(OUT/'progress.json',progress)

def register(cfg,c,d,r):
    with lock(OUT/'progress.lock'):
        p=json.loads((OUT/'progress.json').read_text());row=p['conditions'][cfg['id']];key=ck(c)
        if key in row['seen']:return
        atomic(OUT/'completions'/cfg['id']/f'{key}.json',dict(case=c,directory=str(d),success=r['success'],actions=r['actions'],wall_s=r['wall_s'],time=time.time()))
        row['seen'].append(key);row['completed']+=1;row['success']+=int(r['success']);row['new']+=1
        atomic(OUT/'progress.json',p)

def progress():
    p=json.loads((OUT/'progress.json').read_text())
    return {k:{field:value for field,value in r.items() if field!='seen'} for k,r in p['conditions'].items()}

def claim_group(gpu):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text())
        for phase in PHASES:
            pending=[g for g in q['groups'] if g['phase']==phase and g['state']!='complete']
            if not pending:continue
            for g in pending:
                if g['state']=='pending':g.update(state='running',gpu=gpu,started=time.time());atomic(OUT/'queue.json',q);return {k:g[k] for k in ('key','phase','suite','task')}
            return None
        return None

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
        q=json.loads((OUT/'queue.json').read_text());g=next(g for g in q['groups'] if g['key']==group);assert all(r['state']=='complete' for r in g['cases'])
        g.update(state='complete',finished=time.time());atomic(OUT/'queue.json',q)

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
def phase_done(phase):return all(g['state']=='complete' for g in json.loads((OUT/'queue.json').read_text())['groups'] if g['phase']==phase)

if __name__=='__main__':freeze()
