import hashlib,itertools,json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import lock

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_closedloop_max2'
OLD=ROOT/'results/idea1_v3_4_E2_closedloop'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_closedloop_max2_plan.md'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')

def legal(mask):return len(mask)==4 and len(set(mask))==4 and all(not {i,i+1,i+2}<=set(mask) for i in range(1,16))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);old=json.loads((OLD/'manifest.json').read_text());masks={};conditions=[]
    feasible=[c for c in itertools.combinations(range(1,18),4) if legal(c)]
    for name,ranks in old['ranks'].items():
        mask=list(min(feasible,key=lambda c:(sum(ranks[str(l)] for l in c),c)))
        masks[name]=mask;conditions.append(dict(id=name,model='pi0',mode='pure',skip=mask,D=2,H=50,R=5))
    sources=dict(old['source_files'])
    for p in Path(__file__).parent.glob('*.py'):sources[str(p.relative_to(ROOT))]=sha(p)
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    m=dict(conditions=conditions,masks=masks,cases=old['cases'],source_files=sources,ranks=old['ranks'],
        original_manifest_sha256=sha(OLD/'manifest.json'),plan_sha256=sha(PLAN),constraint=dict(skip_count=4,max_run=2),gpus=[0,1,2,3],time=time.time())
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    atomic(OUT/'manifest.json',m);atomic(OUT/'queue.json',dict(cases=[dict(case=c,state='pending') for c in m['cases']]))
    print(json.dumps(masks),flush=True)

def claim(worker):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text())
        for row in q['cases']:
            if row['state']=='pending':
                row.update(state='running',worker=worker,claimed=time.time());atomic(OUT/'queue.json',q);return row['case']
    return None

def finish_case(case):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text())
        row=next(r for r in q['cases'] if r['case']==case);row.update(state='complete',finished=time.time());atomic(OUT/'queue.json',q)

def release(worker):
    with lock(OUT/'queue.lock'):
        q=json.loads((OUT/'queue.json').read_text())
        for r in q['cases']:
            if r['state']=='running' and r.get('worker')==worker:r['state']='pending'
        atomic(OUT/'queue.json',q)

def pending():return sum(r['state']=='pending' for r in json.loads((OUT/'queue.json').read_text())['cases'])

if __name__=='__main__':freeze()
