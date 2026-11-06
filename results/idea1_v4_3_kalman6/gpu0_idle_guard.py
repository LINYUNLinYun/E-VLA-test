"""Protect other users' short job-transition gaps using the existing advisory lock.

No CUDA calls and no additional nvidia-smi poller: consume the controller's
30-second snapshots. Release after 60 seconds of observed continuous idle.
"""
import fcntl
import json
import os
from pathlib import Path
import time

OUT=Path(__file__).resolve().parent
UUID='GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1'


def record(state, **extra):
    value=dict(state=state,pid=os.getpid(),gpu=0,uuid=UUID,time=time.time(),**extra)
    temporary=OUT/'gpu0_idle_guard.tmp'
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(OUT/'gpu0_idle_guard.json')


def main():
    owner=json.loads((OUT/'controller_process.json').read_text())['pid']
    with Path('/tmp',f'embodied-benchmark-gpu-{UUID}.lock').open('a') as lease:
        while True:
            try:
                os.kill(owner,0)
            except ProcessLookupError:
                return record('controller_exited')
            status=json.loads((OUT/'status.json').read_text())
            if status.get('status')=='complete' or (OUT/'failure.json').exists():
                return record('experiment_ended')
            if any(w['gpu']==0 and w['phase'] in ('replay','tune','confirm','idle') for w in status.get('workers',[])):
                return record('owned_worker_already_running')
            try:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(2)
        idle_since=None
        record('waiting_stable_idle')
        while True:
            try:
                os.kill(owner,0)
            except ProcessLookupError:
                return record('controller_exited')
            if (OUT/'complete.json').exists() or (OUT/'failure.json').exists():
                return record('experiment_ended')
            scan=json.loads((OUT/'resource_scan.json').read_text())
            row=next(r for r in scan['gpus'] if r['uuid']==UUID)
            if row['occupied'] or row['used_mib']>1500:
                idle_since=None
            elif idle_since is None:
                idle_since=scan['time']
            elif scan['time']-idle_since>=60:
                return record('released_after_stable_idle',idle_since=idle_since)
            time.sleep(2)


if __name__=='__main__':
    main()
