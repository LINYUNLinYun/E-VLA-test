from pathlib import Path
import json,time,subprocess,sys
r=Path(__file__).resolve().parent
p=r/'gate_same_task.result.json'
while True:
 if p.exists():
  d=json.loads(p.read_text())
  if d['status']=='passed':break
  if d['status']=='failed':raise RuntimeError('same-task gate failed')
 time.sleep(3)
(r/'STOP_AFTER_CURRENT').unlink(missing_ok=True)
raise SystemExit(subprocess.call([sys.executable,str(r/'run_matrix.py'),str(r/'same_task_matrix.json')]))
