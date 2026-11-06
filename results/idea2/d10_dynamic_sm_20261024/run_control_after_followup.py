"""Bounded fairness control, serialized after SLO and deployment smoke."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
state_path=root/'async_control_followup.execution.json'
if state_path.exists():
    raise RuntimeError('Refusing to restart an existing control coordinator')
state=dict(status='waiting_for_followup',upstream_pid=int(sys.argv[1]))
def save():
    temp=state_path.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(state_path)
try:
    save()
    while True:
        receipt=json.loads((root/'followup.execution.json').read_text())
        if receipt['status']=='passed':
            break
        if receipt['status']=='failed':
            raise RuntimeError('Followup failed; inspect actual blocker before continuation')
        p=Path(f"/proc/{state['upstream_pid']}/cmdline")
        if not p.exists() or b'run_followup.py' not in p.read_bytes():
            raise RuntimeError('Upstream process missing without completion receipt')
        if (root/'STOP_AFTER_CURRENT').exists():
            raise RuntimeError('Explicit stop marker present')
        time.sleep(30)
    state['status']='running'
    command=[sys.executable,str(root/'run_matrix.py'),str(root/'async_serial_control_matrix.json')]
    with (root/'async_serial_control_queue.log').open('x') as log:
        p=subprocess.Popen(command,cwd=root.parents[2],stdout=log,stderr=subprocess.STDOUT)
        state.update(pid=p.pid,command=command)
        save()
        state['exit_code']=p.wait()
    state['status']='passed' if state['exit_code']==0 else 'failed'
except BaseException as e:
    state.update(status='failed',error=repr(e))
    raise
finally:
    save()
