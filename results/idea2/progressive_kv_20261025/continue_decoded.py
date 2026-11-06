from pathlib import Path
import json,time,subprocess,sys
r=Path(__file__).resolve().parent
while True:
 p=r/'same_task_matrix.execution.json'
 if p.exists():
  d=json.loads(p.read_text())
  if d['status']=='stopped_at_case_boundary':break
  if d['status']=='failed':raise RuntimeError('upstream failed')
 time.sleep(3)
(r/'STOP_AFTER_CURRENT').unlink(missing_ok=True)
raise SystemExit(subprocess.call([sys.executable,str(r/'run_matrix.py'),str(r/'decoded_confirmation_matrix.json')]))
