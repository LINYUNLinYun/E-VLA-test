"""Hand GPU1 to the user-authorized idea2 lease after a durable episode."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from preparations.idea1_v3.io import atomic,append
from .dispatch import OUT,PY,ROOT


def main():
    client,server,owner=39054,38716,38509
    base=OUT/'pi0_pure_d2/episodes'
    before=set(base.glob('*/task*/init*/complete.json')); start=time.monotonic()
    while True:
        now=set(base.glob('*/task*/init*/complete.json'))
        if now-before:
            os.kill(client,signal.SIGSTOP); break
        if time.monotonic()-start>120: raise TimeoutError('No durable episode boundary within120s')
        time.sleep(.005)
    for pid in (client,server,owner):
        if b'preparations.idea1_v3_3.' not in Path(f'/proc/{pid}/cmdline').read_bytes(): raise ValueError('PID identity changed')
    os.kill(client,signal.SIGTERM); os.kill(client,signal.SIGCONT)
    os.kill(server,signal.SIGTERM); os.kill(owner,signal.SIGKILL)
    lease=dict(gpu=1,uuid='GPU-599477fb-0507-40dc-14df-d491f7a7b202',active=True,owner='user-authorized idea2',time=time.time(),
        released_at_complete=[str(p) for p in sorted(now-before)],preserved_valid=len(now),condition='pi0_pure_d2',old_pids=[owner,server,client])
    atomic(OUT/'external_gpu_leases.json',{'1':lease})
    state=json.loads((OUT/'campaign_state.json').read_text()); state['workers'].pop('1',None); state['external_leases']={'1':lease}; atomic(OUT/'campaign_state.json',state)
    append(OUT/'campaign_events.jsonl',dict(event='external_gpu_lease',**lease))
    time.sleep(2)
    apps=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    lease['compute_pids_after_release']=apps; atomic(OUT/'gpu1_release_receipt.json',lease)
    with (OUT/'campaign.log').open('a') as log:
        supervisor=subprocess.Popen([str(PY['pi0']),'-m','preparations.idea1_v3_3.campaign'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(lease=lease,supervisor_pid=supervisor.pid)),flush=True)


if __name__=='__main__': main()
