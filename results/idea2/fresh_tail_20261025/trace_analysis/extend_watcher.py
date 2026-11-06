"""Bounded CPU-only continuation; preserve the original wait receipt."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.sched_setaffinity(0, set(range(64, 96)))
receipt = HERE / 'execution_receipt.json'
deadline = time.monotonic() + 12 * 60
while time.monotonic() < deadline:
    try:
        state = json.loads(receipt.read_text())
    except (OSError, json.JSONDecodeError):
        time.sleep(10)
        continue
    if state['status'] == 'both_analyzed':
        print('Original watcher completed both traces; no extension needed.', flush=True)
        raise SystemExit(0)
    if state['status'] == 'timed_out_incomplete':
        shutil.copy2(receipt, HERE / 'execution_receipt_initial_30min.json')
        shutil.copy2(HERE / 'watcher.log', HERE / 'watcher_initial_30min.log')
        command = ['taskset', '-c', '64-95', sys.executable, str(HERE.parent / 'analyze_traces.py'),
                   '--watch', '--watch-minutes', '20']
        print(json.dumps(dict(status='extending', command=command)), flush=True)
        with (HERE / 'watcher_extension_20min.log').open('w') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        raise SystemExit(run.returncode)
    if state['status'] not in ('waiting',):
        print(json.dumps(dict(status='unexpected_prior_status', receipt=state)), flush=True)
        raise SystemExit(2)
    time.sleep(10)
raise SystemExit('Initial watcher did not reach its deadline within bounded extension handoff wait')
