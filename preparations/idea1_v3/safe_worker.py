"""Continue existing E1/E3 static shard with safe fresh child servers only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from preparations.idea1_v3.cli import ROOT,EVAL_PYTHON
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3.safety_runtime import require_recovery_pass,kernel_digest


def run(args,manifest):
    gate=require_recovery_pass(args.safety_gate)
    if not 0<=args.shard<manifest['shards']: raise ValueError('Static shard outside manifest')
    wd=args.attempt/'workers'/f'worker{args.shard}'; wd.mkdir(parents=True,exist_ok=True)
    stamp=time.time_ns(); reference=wd/f'qk_reference_safe_worker_{stamp}.json'
    # Never automatically retry a second invalid attempt or an unrelated invalid
    # key. The explicitly diagnosed key must already be valid before this entry.
    events=wd/'events.jsonl'
    if events.exists():
        invalid={json.loads(line)['key'] for line in events.read_text().splitlines() if json.loads(line).get('kind')=='invalid'}
        for row in manifest['tasks']:
            if row['owner']!=args.shard or row['key'] not in invalid: continue
            complete=args.attempt/'episodes'/row['experiment']/row['condition']/row['execution']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/'complete.json'
            if not complete.exists(): raise RuntimeError('Unresolved invalid key: use its diagnosed recovery, not continuation')
    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('safe_worker.py','safe_worker_server.py','safety_runtime.py','safe_qk.py')}
    atomic(wd/f'safe_worker_revision_{stamp}.json',dict(revision='qk-row-bounds-v1',source_sha256=sources,
        safety_gate=gate,reference_receipt=str(reference),manifest_unchanged=True,scientific_keys_unchanged=True,
        status='launcher_preflight_before_exclusive_worker_lock'))
    original=subprocess.Popen
    def launch(command,*positional,**kwargs):
        command=list(command)
        accepted=('preparations.idea1_v3.server','preparations.idea1_v3.branch_server')
        selected=next((name for name in accepted if name in command),None)
        if selected:
            command[command.index(selected)]='preparations.idea1_v3.safe_worker_server'
            env=dict(kwargs.get('env',os.environ),V3_QK_WORKER_RECEIPT=str(reference),V3_SAFE_BASE_SERVER=selected)
            kwargs['env']=env
        return original(command,*positional,**kwargs)
    subprocess.Popen=launch
    try:
        if manifest['stage']=='E1':
            from preparations.idea1_v3.cli import worker
        else:
            from preparations.idea1_v3.branches import worker
        worker(args)
    finally: subprocess.Popen=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=Path,required=True); parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True); parser.add_argument('--safety-gate',type=Path,required=True)
    parser.add_argument('--launch',action='store_true'); args=parser.parse_args(); args.attempt=args.attempt.resolve()
    args.safety_gate=args.safety_gate.resolve(); manifest=json.loads((args.attempt/'manifest.json').read_text())
    if manifest['stage'] not in ('E1','E3'): raise ValueError('Safe continuation is only for existing E1/E3 scientific shards')
    if not args.launch:
        print(json.dumps(dict(dry_run=True,stage=manifest['stage'],shard=args.shard,revision='qk-row-bounds-v1',
            kernel_sha256=kernel_digest(),requires_completed_single_key_gate=str(args.safety_gate),
            valid_episodes_skipped=True,unresolved_invalid_keys_blocked=True),indent=2)); return
    require_recovery_pass(args.safety_gate)
    if Path(sys.executable).resolve()!=EVAL_PYTHON.resolve() or os.environ.get('V3_SAFE_WORKER_CHILD')!='1':
        from preparations.idea1_v2_2.common import evaluation_env
        env=evaluation_env(args.gpu); env.update(V3_SAFE_WORKER_CHILD='1',PYTHONPATH=str(ROOT)+':'+env['PYTHONPATH'])
        subprocess.run([str(EVAL_PYTHON),'-m','preparations.idea1_v3.safe_worker',*sys.argv[1:]],cwd=ROOT,env=env,check=True)
    else: run(args,manifest)

if __name__=='__main__': main()
