"""At most one diagnosed invalid-key retry in a fresh bounded-QK service.

No queue expansion, no valid episode re-run, no mutation of the frozen sources.
An existing incomplete episode directory is preserved by rename before retry.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from preparations.idea1_v3.cli import ROOT,EVAL_PYTHON,MODEL_PYTHON,verify_gpu
from preparations.idea1_v3.io import atomic,append


def recover(args,row):
    from preparations.idea1_v2_1.runtime_client import rpc
    from preparations.idea1_v3.episode import run
    wd=args.attempt/'workers'/f"worker{row['owner']}"; wd.mkdir(parents=True,exist_ok=True)
    guard=wd/f"diagnosed_retry_{row['key']}.json"
    directory=args.attempt/'episodes'/row['experiment']/row['condition']/row['execution']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"
    with (wd/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if guard.exists(): raise RuntimeError('This key already used its one diagnosed retry; no automatic further retry')
        if (directory/'complete.json').exists(): raise RuntimeError('Valid episode exists; recovery forbidden')
        events=[json.loads(line) for line in (wd/'events.jsonl').read_text().splitlines()]
        if not any(event.get('kind')=='invalid' and event.get('key')==row['key'] for event in events):
            raise ValueError('No preserved invalid event for requested key')
        gpu=verify_gpu(args.gpu)
        with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        receipt=wd/f"qk_reference_{row['key']}.json"; env=dict(os.environ,V3_QK_SAFETY_RECEIPT=str(receipt))
        stamp=time.time_ns(); logfile=wd/f'safe_recovery_{stamp}.log'
        with logfile.open('x') as log:
            server=subprocess.Popen([str(MODEL_PYTHON),'-m','preparations.idea1_v3.safe_server','--port',str(port),
                '--config',json.dumps(row['config']),'--launch'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+1200
                while True:
                    if server.poll() is not None: raise RuntimeError(f'Recovery server exited; inspect {logfile}')
                    try: rpc(port,dict(op='health')); break
                    except (OSError,RuntimeError):
                        if time.monotonic()>deadline: raise TimeoutError('Recovery startup')
                        time.sleep(1)
                preserved=None
                if directory.exists():
                    preserved=directory.with_name(directory.name+f'.invalid_before_qk_bounds_{stamp}')
                    directory.rename(preserved)
                sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('recover_one.py','safe_server.py','safe_qk.py','policy.py','episode.py')}
                atomic(guard,dict(key=row['key'],status='reserved',revision='qk-row-bounds-v1',source_sha256=sources,
                    physical_gpu=args.gpu,gpu_identity=gpu,preserved_directory=str(preserved) if preserved else None,
                    reference_receipt=str(receipt),log=str(logfile),scientific_config_unchanged=True))
                append(wd/'events.jsonl',dict(kind='diagnosed_retry_start',key=row['key'],revision='qk-row-bounds-v1'))
                try:
                    result=run(directory,port,row,row['config'],args.attempt/'profiling'/f"profile_worker{row['owner']}.jsonl")
                    data=json.loads(guard.read_text()); data.update(status='valid',success=result['success']); atomic(guard,data)
                    append(wd/'events.jsonl',dict(kind='episode_complete',key=row['key'],success=result['success'],diagnosed_retry=True))
                except Exception:
                    data=json.loads(guard.read_text()); data.update(status='invalid',error=traceback.format_exc()); atomic(guard,data)
                    append(wd/'events.jsonl',dict(kind='invalid',key=row['key'],diagnosed_retry=True,error=traceback.format_exc()))
                    raise
            finally:
                server.terminate()
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired: server.kill(); server.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=Path,required=True); parser.add_argument('--key',required=True)
    parser.add_argument('--gpu',type=int,required=True); parser.add_argument('--launch',action='store_true')
    args=parser.parse_args(); args.attempt=args.attempt.resolve()
    data=json.loads((args.attempt/'manifest.json').read_text()); row=next(r for r in data['tasks'] if r['key']==args.key)
    if row['experiment']!='E1': raise ValueError('This diagnosed recovery entrypoint is limited to E1')
    if not args.launch:
        print(json.dumps(dict(dry_run=True,row=row,revision='qk-row-bounds-v1',one_retry_only=True),indent=2)); return
    if Path(sys.executable).resolve()!=EVAL_PYTHON.resolve() or os.environ.get('V3_RECOVERY_CHILD')!='1':
        from preparations.idea1_v2_2.common import evaluation_env
        env=evaluation_env(args.gpu); env.update(V3_RECOVERY_CHILD='1',PYTHONPATH=str(ROOT)+':'+env['PYTHONPATH'])
        subprocess.run([str(EVAL_PYTHON),'-m','preparations.idea1_v3.recover_one',*sys.argv[1:]],cwd=ROOT,env=env,check=True)
    else: recover(args,row)

if __name__=='__main__': main()
