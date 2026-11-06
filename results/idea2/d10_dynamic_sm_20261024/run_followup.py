"""Complete SLO and closed-loop work after the already-running queue succeeds.

No restart on missing heartbeat: check authoritative state AND the known process.
Each downstream command is gated by the prior command's actual completion.
"""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]
STATUS=ROOT/'followup.execution.json'
if STATUS.exists():
    raise RuntimeError('Refusing to restart an existing followup execution')
state=dict(status='waiting_for_confirmation',upstream_pid=int(sys.argv[1]),commands=[])

def save():
    state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp=STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(STATUS)

def run(name,command):
    state['status']='running_'+name
    row=dict(name=name,command=command,start=time.time())
    state['commands'].append(row)
    with (ROOT/(name+'.queue.log')).open('x') as log:
        p=subprocess.Popen(command,cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT)
        row['pid']=p.pid
        save()
        print('START',name,flush=True)
        row['exit_code']=p.wait()
    row['elapsed_seconds']=time.time()-row['start']
    save()
    if row['exit_code']:
        raise RuntimeError(name+' failed; retain evidence and inspect before any retry')
    print('PASSED',name,flush=True)

try:
    save()
    while True:
        receipt=json.loads((ROOT/'resume_confirmation_matrix.execution.json').read_text())
        if receipt['status']=='passed':
            break
        if receipt['status']!='running':
            raise RuntimeError('Upstream terminal status: '+receipt['status'])
        pid=state['upstream_pid']
        path=Path(f'/proc/{pid}/cmdline')
        if not path.exists() or b'run_matrix.py' not in path.read_bytes():
            raise RuntimeError('Upstream process missing, no completed receipt; no automatic restart')
        if (ROOT/'STOP_AFTER_CURRENT').exists():
            raise RuntimeError('Explicit stop marker present; do not launch downstream work')
        time.sleep(30)
    run('prepare_slo',[sys.executable,str(ROOT/'prepare_slo_matrix.py')])
    run('slo_confirmation',[sys.executable,str(ROOT/'run_matrix.py'),str(ROOT/'slo_confirmation_matrix.json')])
    if (ROOT/'STOP_AFTER_CURRENT').exists():
        raise RuntimeError('Explicit stop marker present before smoke')
    run('closedloop_smoke',[str(WORKSPACE/'policies/openpi/.venv/bin/python'),str(ROOT/'closedloop_smoke.py'),'--execute'])
    state['status']='passed'
except BaseException as error:
    state.update(status='failed',error=repr(error))
    raise
finally:
    save()
