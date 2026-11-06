import json,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4_E2_closedloop_max2.common import sha
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_E2_max2_repair'
OLD=ROOT/'results/idea1_v3_4_E2_closedloop'
NEW=ROOT/'results/idea1_v3_4_E2_closedloop_max2'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_max2_reset_repair_plan.md'
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);old=json.loads((OLD/'manifest.json').read_text());new=json.loads((NEW/'manifest.json').read_text());configs=[];mapping={}
    for c in old['conditions']:
        name=c['id'] if c['id'] in ('v14','full_d2') else 'old_'+c['id']
        configs.append(dict(c,id=name));mapping[name]=dict(root=str(OLD),id=c['id'])
    for c in new['conditions']:
        name='new_'+c['id'];configs.append(dict(c,id=name));mapping[name]=dict(root=str(NEW),id=c['id'])
    m=dict(conditions=configs,masks={c['id']:c['skip'] for c in configs},mapping=mapping,cases=[dict(suite='libero_10',task=t,init=i) for t in (3,9) for i in (19,20)],
        original_cases=old['cases'],plan_sha256=sha(PLAN),sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},time=time.time())
    m['manifest_hash']=digest(m)
    if (OUT/'manifest.json').exists():raise FileExistsError('Frozen')
    atomic(OUT/'manifest.json',m)
if __name__=='__main__':freeze()
