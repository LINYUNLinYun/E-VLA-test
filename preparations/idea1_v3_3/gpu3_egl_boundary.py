"""Drain GPU3 tau.1 then retain its model and correct environment EGL binding."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from preparations.idea1_v3.io import atomic,append
from .metadata_boundary import children_active
from .dispatch import ROOT,OUT,PY


def main():
    owner,server,cid=32618,32888,'E4_tau0.1';start=time.monotonic()
    while True:
        n=len(list((OUT/cid/'episodes').glob('*/task*/init*/complete.json')))
        atomic(OUT/'gpu3_egl_boundary_state.json',dict(status='draining_condition80',condition=cid,valid=n,time=time.time()))
        if n==80 and not children_active(owner):break
        if time.monotonic()-start>3600:raise TimeoutError('EGL boundary')
        time.sleep(5)
    # New dispatcher waits on old owner's flock. Publish the new live PID
    # before dropping the old owner, so campaign never mistakes the handoff.
    with (OUT/'resume_gpu3_egl.log').open('a') as log:
        resumed=subprocess.Popen([str(PY['pi0']),'-m','preparations.idea1_v3_3.resume_egl','--gpu','3','--port','9343','--server-pid',str(server)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state=json.loads((OUT/'campaign_state.json').read_text());state['workers']['3']=dict(pid=resumed.pid,model='xiaomi');atomic(OUT/'campaign_state.json',state)
    atomic(OUT/'worker_handoffs/gpu3_egl.json',dict(gpu=3,old_pid=owner,new_worker=state['workers']['3'],expires=time.time()+120))
    running=json.loads((OUT/'conditions'/cid/'running.json').read_text());atomic(OUT/'conditions'/cid/'done.json',dict(config=running['config'],gpu=3,wall_s=time.time()-running['started'],finished=time.time(),completion='80 valid; model retained for environment-only EGL correction'))
    if b'preparations.idea1_v3_3.' not in Path(f'/proc/{owner}/cmdline').read_bytes():raise ValueError('Owner PID changed')
    os.kill(owner,signal.SIGKILL)
    append(OUT/'campaign_events.jsonl',dict(event='GPU3_egl_boundary',old_dispatch=owner,retained_server=server,new_dispatch=resumed.pid,valid=80,time=time.time()))
    atomic(OUT/'gpu3_egl_boundary_state.json',dict(status='corrected_env_driver_launched',pid=resumed.pid,time=time.time()))


if __name__=='__main__':main()
