from pathlib import Path
import json,time,subprocess,sys
r=Path(__file__).resolve().parent
while True:
 p=r/'closedloop_gpu0/execution.json'
 if p.exists():
  d=json.loads(p.read_text())
  if d['status']=='passed':break
  if d['status']=='failed':raise RuntimeError('Quality execution failed; do not reuse GPU blindly')
 time.sleep(5)
raise SystemExit(subprocess.call([sys.executable,str(r/'run_gpu0_matrix.py'),str(r/'gpu0_resource_matrix.json')]))
