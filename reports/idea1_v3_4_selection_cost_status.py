"""Compact, read-only status for the single-card selection experiment."""
import csv,json,time
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'results/idea1_v3_4_selection_cost'
s=json.loads((p/'status.json').read_text())
wait=p/'coordinator_waiting_for_children.json'
if wait.exists() and s.get('status')!='complete':
 manifest=json.loads((p/'manifest.json').read_text());live={}
 for cfg in manifest['conditions']:
  if cfg['variant'] not in ('baseline','combination'):continue
  rr=[]
  for f in (p/cfg['id']).glob('episodes/*/task*/init*/result.json'):
   if (f.parent/'complete.json').exists():rr.append(json.loads(f.read_text()))
  live[cfg['id']]=dict(completed=len(rr),target=8,success=sum(r['success'] for r in rr))
 s=dict(s,progress=live,completed=sum(x['completed'] for x in live.values()))
r={k:v for k,v in s.items() if k!='progress'}
if 'progress' in s:
 r['success_so_far']={k.removeprefix('v').replace('_extrapolate_d2_verify_','/'):f"{v['success']}/{v['completed']}" for k,v in s['progress'].items() if v['completed']}
f=p/'benchmark/checks.json'
if f.exists():
 c=json.loads(f.read_text());r['benchmark_samples']=c['finished_samples']
 r['mask_differences']={str(b):sum(not x.get('masks_equal',False) for x in c['checks'] if x.get('batch_size')==b) for b in (2,4)}
f=p/'benchmark/timings.csv'
if f.exists():
 with f.open() as stream:rows=list(csv.DictReader(stream))
 r['batch_ms']={str(b):round(sum(float(x['measured_wall_ms']) for x in rows if int(x['batch_size'])==b)/sum(int(x['batch_size'])==b for x in rows),2) for b in (1,2,4) if any(int(x['batch_size'])==b for x in rows)}
f=p/'current.json'
if f.exists():
 c=json.loads(f.read_text());r['current']=c;r['current_age_s']=round(time.time()-c['time'])
f=p/'failed.json'
if f.exists():r['failure']=json.loads(f.read_text())
f=p/'worker/service.log'
if f.exists() and 'failure' in r:r['service_tail']=f.read_text(errors='replace').splitlines()[-12:]
f=p/'worker/evaluate.log'
if f.exists() and 'failure' in r:r['evaluate_tail']=f.read_text(errors='replace').splitlines()[-16:]
f=p/'continuation_failure.json'
if f.exists():r['continuation_failure']=json.loads(f.read_text())
if (p/'continuation_process.json').exists():
 r.pop('current',None);r.pop('current_age_s',None)
 r['workers']={}
 for wd in sorted((p/'continuation_workers').glob('gpu*')):
  row={}
  for name in ('identity.json','current.json','numerical_gate.json','failure.json'):
   f=wd/name
   if f.exists():
    x=json.loads(f.read_text())
    if name=='numerical_gate.json':x={k:v for k,v in x.items() if k!='checks'}
    if name=='identity.json':x={k:v for k,v in x.items() if k in ('gpu','pid','attached')}
    row[name]=x
  r['workers'][wd.name]=row
print(json.dumps(r))
