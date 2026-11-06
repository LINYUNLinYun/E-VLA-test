"""Resume proven-unstarted work after the idle-telemetry startup rejection."""
from pathlib import Path
import json
import subprocess
import sys
import time
from idle_settle import wait_for_idle

root=Path(__file__).resolve().parent
workspace=root.parents[2]
path=root/'remaining.execution.json'
if path.exists():
    raise RuntimeError('Existing completion queue; do not restart blindly')
state=dict(status='running',commands=[])
def save():
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(path)
try:
    save()
    commands=[('slo_remaining',[sys.executable,str(root/'run_matrix.py'),str(root/'slo_remaining_matrix.json')]),
              ('closedloop_remaining',[str(workspace/'policies/openpi/.venv/bin/python'),str(root/'closedloop_smoke.py'),'--execute']),
              ('async_control_remaining',[sys.executable,str(root/'run_matrix.py'),str(root/'async_serial_control_matrix.json')])]
    for name,command in commands:
        if (root/'STOP_AFTER_CURRENT').exists():
            raise RuntimeError('Explicit stop marker present')
        row=dict(name=name,command=command,start=time.time(),status='preparing',idle_settle_observations=wait_for_idle())
        state['commands'].append(row)
        save()
        with (root/(name+'.queue.log')).open('x') as log:
            p=subprocess.Popen(command,cwd=workspace,stdout=log,stderr=subprocess.STDOUT)
            row.update(pid=p.pid,status='running')
            save()
            print('START',name,flush=True)
            row['exit_code']=p.wait()
        row['elapsed_seconds']=time.time()-row['start']
        row['status']='passed' if row['exit_code']==0 else 'failed'
        save()
        print(row['status'].upper(),name,flush=True)
        if row['status']!='passed':
            raise RuntimeError(name+' failed; inspect before recovery')
    state['status']='passed'
except BaseException as e:
    state.update(status='failed',error=repr(e))
    raise
finally:
    save()
