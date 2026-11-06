from pathlib import Path
import json,time,subprocess,sys
r=Path(__file__).resolve().parent
while True:
 a=r/'closedloop_sm16_gpu3_clean/execution.json';b=r/'gpu0_resource_sm32_lead3_screen_clean/result.json'
 if a.exists() and b.exists():
  x,y=json.loads(a.read_text()),json.loads(b.read_text())
  if x['status']=='failed' or y['status']=='failed':raise RuntimeError('Upstream technical failure')
  if x['status']=='passed' and y['status']=='passed':break
 time.sleep(5)
base=json.loads((r/'gpu0_full_control_screen/result.json').read_text())['summary']['lead3/encoder']['trajectory_ms']['mean']
candidate=y['summary']['lead3/encoder']['trajectory_ms']['mean']
if candidate>=.98*base:
 (r/'sm32_quality_selection.json').write_text(json.dumps({'decision':'skip_no_material_screen_gain','baseline_ms':base,'candidate_ms':candidate},indent=2))
 raise SystemExit(0)
(r/'sm32_quality_selection.json').write_text(json.dumps({'decision':'run_eight_cases','baseline_ms':base,'candidate_ms':candidate},indent=2))
cmd=[str(r.parents[2]/'policies/openpi/.venv/bin/python'),'-u',str(r/'resource32_closedloop.py'),'--execute','--output',str(r/'closedloop_sm32_gpu3_clean')]
raise SystemExit(subprocess.call(cmd))
