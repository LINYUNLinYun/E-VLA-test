"""CPU-only bounded receipt watcher; scan the completed 300 cells exactly once."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parent
ROOT=R.parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours',type=float,default=4.)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    receipt=R/'final_audit_watcher.json'
    if receipt.exists():raise FileExistsError('Existing watcher receipt: do not start a duplicate watcher')
    state=dict(status='waiting_for_terminal_dispatch',gpu_launched=False,pid=os.getpid(),
        start_utc=datetime.now(timezone.utc).isoformat(),deadline_hours=args.hours,
        watcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(receipt)
    save();deadline=time.monotonic()+args.hours*3600
    while time.monotonic()<deadline:
        path=R/'dispatch_execution.json'
        try:dispatch=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(30);continue
        if dispatch['status']=='passed':
            state.update(status='auditing_completed_cells_once',dispatch_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                audit_source_sha256=hashlib.sha256((R/'audit.py').read_bytes()).hexdigest())
            save()
            command=[sys.executable,'-m','results.idea2.continuous_encoder_20261026.audit','--mode','full']
            with (R/'final_audit.log').open('x') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False,env=os.environ)
            state.update(status='passed' if result.returncode==0 else 'failed',exit_code=result.returncode,command=command,
                finish_utc=datetime.now(timezone.utc).isoformat())
            output=R/'full_audit/audit.json'
            if output.exists():state['audit_output_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
            save();print(json.dumps(state,indent=2));return result.returncode
        if dispatch['status'] not in ('running','pending'):
            state.update(status='pending_parent_review',dispatch_status=dispatch['status'],finish_utc=datetime.now(timezone.utc).isoformat());save();return 2
        time.sleep(30)
    state.update(status='pending_timeout',finish_utc=datetime.now(timezone.utc).isoformat());save();return 3

if __name__=='__main__':sys.exit(main())
