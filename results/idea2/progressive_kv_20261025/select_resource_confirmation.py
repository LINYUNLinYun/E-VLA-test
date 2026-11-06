from pathlib import Path
import json,time,subprocess,sys
r=Path(__file__).resolve().parent
while True:
 matrix=r/'gpu0_resource_clean_matrix.execution.json'
 q16=r/'closedloop_sm16_gpu3_clean/execution.json'
 select32=r/'sm32_quality_selection.json'
 if all(p.exists() for p in [matrix,q16,select32]):
  m,a,s=map(lambda p:json.loads(p.read_text()),[matrix,q16,select32])
  if m['status']=='failed' or a['status']=='failed':raise RuntimeError('Technical upstream failure')
  if m['status']=='passed' and a['status']=='passed':
   if s['decision'].startswith('skip'):b=None;break
   q32=r/'closedloop_sm32_gpu3_clean_port19087/execution.json'
   if q32.exists():
    b=json.loads(q32.read_text())
    if b['status']=='failed':raise RuntimeError('SM32 technical quality failure')
    if b['status']=='passed':break
 time.sleep(5)
ok16=all(c['result']['success'] for c in a['cases'])
ok32=b is not None and all(c['result']['success'] for c in b['cases'])
def latency(sms):return json.loads((r/f'gpu0_resource_sm{sms}_lead3_screen_clean/result.json').read_text())['summary']['lead3/encoder']['trajectory_ms']['mean']
selected=16 if ok16 else 32 if ok32 else None
if ok16 and ok32 and latency(32)<.98*latency(16):selected=32
selection=dict(decision='confirm' if selected else 'skip_quality_gate_not_met',selected_ae_sms=selected,sm16_quality_pass=ok16,sm32_quality_pass=ok32,rule='Require 8/8 fixed quality. Prefer SM16 unless SM32 is >2% faster; source freshness breaks small performance ties. No population-equivalence claim.')
(r/'resource_confirmation_selection.json').write_text(json.dumps(selection,indent=2))
if selected is None:raise SystemExit(0)
m=json.loads((r/'gpu0_sm16_confirmation_matrix.json').read_text())
if selected==32:
 for case in m:
  case['name']=case['name'].replace('sm16','sm32')
  if '--ae-sm-min' in case['args']:case['args'][case['args'].index('--ae-sm-min')+1]='32'
  if case.get('nsys_output'):case['nsys_output']=case['nsys_output'].replace('sm16','sm32')
p=r/'gpu0_selected_resource_confirmation_matrix.json';p.write_text(json.dumps(m,indent=2))
raise SystemExit(subprocess.call([sys.executable,str(r/'run_isolated_matrix.py'),str(p),'GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1',','.join(map(str,range(32,48)))]))
