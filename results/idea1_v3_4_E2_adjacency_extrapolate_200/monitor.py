import json,os
from pathlib import Path
p=Path(__file__).parent
s=json.loads((p/'status.json').read_text())
pr=json.loads((p/'process.json').read_text())
try:os.kill(pr['pid'],0);alive=True
except ProcessLookupError:alive=False
q=json.loads((p/'queue.json').read_text())
print(json.dumps(dict(status=s['status'],controller_alive=alive,completed=s.get('new_completed'),groups_done=sum(g['state']=='complete' for g in q['groups']),progress={k.replace('_jacobian','').replace('_extrapolate',''):[v['completed'],v['success']] for k,v in s.get('progress',{}).items()})))
for wd in sorted((p/'workers').glob('gpu*')):
    fields={}
    for name in ('state.json','compatibility_gate.json','isolation_gate.json'):
        f=wd/name
        if f.exists():
            v=json.loads(f.read_text());fields[name.split('.')[0]]=v.get('status',v.get('passed'))
    print(wd.name,json.dumps(fields))
for f in [p/'failed.json',*(p/'workers').glob('gpu*/error_attempt*.json')]:
    if f.exists():print(str(f),f.read_text())
