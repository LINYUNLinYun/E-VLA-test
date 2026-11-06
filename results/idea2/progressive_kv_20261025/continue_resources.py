from pathlib import Path
import json,time,subprocess,sys
r=Path(__file__).resolve().parent
while True:
 p=r/'decoded_confirmation_matrix.execution.json'
 if p.exists():
  d=json.loads(p.read_text())
  if d['status']=='passed':break
  if d['status']=='failed':raise RuntimeError('upstream failed')
 time.sleep(5)
raise SystemExit(subprocess.call([sys.executable,str(r/'run_matrix.py'),str(r/'bounded_resource_matrix.json')]))
