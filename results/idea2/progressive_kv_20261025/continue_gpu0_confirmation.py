from pathlib import Path
import json,time,subprocess,sys
r=Path(__file__).resolve().parent
while True:
 a=r/'gpu0_resource_clean_matrix.execution.json';b=r/'closedloop_sm16_gpu3_clean/execution.json'
 if a.exists() and b.exists():
  x,y=json.loads(a.read_text()),json.loads(b.read_text())
  if x['status']=='failed' or y['status']=='failed':raise RuntimeError('Upstream technical failure; inspect before confirmation')
  if x['status']=='passed' and y['status']=='passed':break
 time.sleep(5)
if not all(c['result']['success'] for c in y['cases']):
 (r/'sm16_confirmation_selection.json').write_text(json.dumps({'decision':'skip_formal_quality_gate_not_met','valid_episodes':len(y['cases']),'successes':sum(c['result']['success'] for c in y['cases'])},indent=2))
 raise SystemExit(0)
(r/'sm16_confirmation_selection.json').write_text(json.dumps({'decision':'confirm','reason':'8/8 fixed quality cases match fresh success count; small sample only, not proof of population equivalence'},indent=2))
raise SystemExit(subprocess.call([sys.executable,str(r/'run_isolated_matrix.py'),str(r/'gpu0_sm16_confirmation_matrix.json'),'GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1',','.join(map(str,range(32,48)))]))
