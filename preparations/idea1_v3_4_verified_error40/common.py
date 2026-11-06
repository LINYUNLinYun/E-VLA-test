import hashlib,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_cycle3.common import SUITES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_verified_error40'
BASE=ROOT/'results/idea1_v3_4_verified_paths'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_verified_error40_plan.md'
PHASES=('main40',)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(c):return f"{c['suite']}_task{c['task']:02d}_init{c['init']:02d}"
def directory(name,c):return OUT/name/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
def source(m,cfg,c):
    d=directory(cfg['id'],c)
    return (d,False) if (d/'complete.json').exists() else (None,False)
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    assert json.loads((OUT/'cpu_gate.json').read_text())['passed']
    old=json.loads((BASE/'manifest.json').read_text());assert json.loads((BASE/'audit.json').read_text())['passed']
    assert all(sha(ROOT/p)==h for p,h in old['source_files'].items())
    cases=[c for c in old['cases'] if c['init']==19];assert len(cases)==40
    configs=[dict(c,id=c['id']+'_error_score',base_id=c['id'],scoring='compensation_error',n=40,phase='main40') for c in old['conditions'] if c['mode']=='verified']
    controls={};baseline={};full={}
    for cfg in configs:
        bank={}
        for c in cases:
            d=BASE/cfg['base_id']/'episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
            assert json.loads((d/'result.json').read_text())['valid'] and (d/'complete.json').exists()
            bank[ck(c)]=dict(directory=str(d),hashes={f:sha(d/f) for f in ('result.json','identity.json','infer.jsonl','arrays.npz','complete.json')})
            if cfg==configs[0]:controls[ck(c)]=dict(directory=str(d),identity_sha256=sha(d/'identity.json'))
        baseline[cfg['base_id']]=bank
    for c in cases:
        rec=json.loads((BASE/'completions/full_d10'/f'{ck(c)}.json').read_text());d=Path(rec['directory'])
        full[ck(c)]=dict(directory=str(d),result_sha256=sha(d/'result.json'))
    files=dict(old['source_files']);files.update({str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    for p in ('preparations/idea1_v3_4_E2_comp_error_cost/model.py','preparations/idea1_v3_4_E2_comp_error_cost/verify.py','preparations/idea1_v3_4_E2_comp_error_multivjp/verify.py'):
        files[p]=sha(ROOT/p)
    refs={str((BASE/f).relative_to(ROOT)):sha(BASE/f) for f in ('manifest.json','audit.json','complete.json','summary.csv')}
    m=dict(cases=cases,conditions=configs,controls=controls,baseline=baseline,full_reference=full,source_files=files,references=refs,plan_sha256=sha(PLAN),target_records=160,new_episodes=160,max_gpus=3,max_actors=1,projection_seed=340610,phases=PHASES,time=time.time())
    from preparations.idea1_v3_4_E2_comp_error_multivjp.model import projection_weights
    m['projection_weights']=projection_weights(4).tolist()
    m['manifest_hash']=digest(m);atomic(OUT/'manifest.json',m)
    groups=[dict(key=f"error40_{c['suite']}_task{c['task']:02d}",phase='main40',suite=c['suite'],task=c['task'],state='pending',cases=[dict(case=c,state='pending')]) for c in cases]
    atomic(OUT/'queue.json',dict(groups=groups));atomic(OUT/'progress.json',dict(conditions={c['id']:dict(target=40,completed=0,success=0,reused=0,new=0,seen=[]) for c in configs}))

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
