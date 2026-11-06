"""Locked durable fallback: observe the existing monitor, then adopt only on exit."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from preparations.ease_insights_20261103.vlm.common import atomic_json

AUDIT=Path(__file__).resolve().parent
WORKSPACE=Path('/root/nfs/embodied/benchmark')
ORIGINAL_PID=336799
ORIGINAL_START_TICKS=2615997095
ORIGINAL_ARG='reports/ease_insights_20261103/vlm_confirmed_audit/continue_formal.py'


def original_live():
    try:
        proc=Path('/proc',str(ORIGINAL_PID))
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        return int(fields[19])==ORIGINAL_START_TICKS and fields[0]!='Z' and ORIGINAL_ARG in args
    except (OSError,ValueError):return False


def formal_complete():
    try:
        audit=json.loads((AUDIT/'audit.json').read_text())
        monitor=json.loads((AUDIT/'monitor_status.json').read_text())
        return audit['status']=='complete' and audit['valid_episodes']==600 and monitor['status']=='complete'
    except (OSError,ValueError,KeyError):return False


def status(phase,**extra):
    record=dict(status=phase,observed_unix=time.time(),pid=os.getpid(),original_pid=ORIGINAL_PID,
        original_start_ticks=ORIGINAL_START_TICKS,exclusive_lock=str(AUDIT/'durable_continuation.lock'),**extra)
    atomic_json(AUDIT/'durable_status.json',record)
    print(json.dumps(record),flush=True)


def main():
    with (AUDIT/'durable_continuation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing durable continuation owns the exclusive lock; exiting without writes.',flush=True);return
        while original_live() and not formal_complete():
            status('observing_original_monitor; no_analysis_writes')
            time.sleep(30)
        if not formal_complete():
            status('original_monitor_gone; taking_over_after_identity_check')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            completed=subprocess.run([sys.executable,'-u',str(AUDIT/'continue_formal.py')],cwd=WORKSPACE,
                env=env,stdin=subprocess.DEVNULL,check=False)
            if completed.returncode!=0 or not formal_complete():
                status('needs_attention',returncode=completed.returncode);return
        status('formal_complete; updating_authorized_summary')
        completed=subprocess.run([sys.executable,str(AUDIT/'update_summary.py')],cwd=WORKSPACE,
            stdin=subprocess.DEVNULL,check=False)
        status('complete' if completed.returncode==0 else 'needs_attention',summary_returncode=completed.returncode,
            whole_goal_complete_claim=False)


if __name__=='__main__':main()
