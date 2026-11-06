"""Separate E2/E3 manifest and workers; running E1/P0 files are not modified."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

from preparations.idea1_v3.cli import ROOT,SUITES,EVAL_PYTHON,MODEL_PYTHON,verify_gpu
from preparations.idea1_v3.io import atomic,append,digest


def manifest(args):
    from preparations.idea1_v3.pi05 import MASKS,CHECKPOINT
    if args.experiment=='E2':
        conditions=[dict(experiment='E2',mask=name) for name in MASKS]
        checkpoint=dict(path=str(CHECKPOINT),config='pi05_libero',native_H=10,D=10,
            norm_stats_sha256=hashlib.sha256((CHECKPOINT/'assets/physical-intelligence/libero/norm_stats.json').read_bytes()).hexdigest())
    else:
        if not args.parent: raise ValueError('E3 requires explicit frozen E1-D2 parent JSON config')
        parent=json.loads(args.parent)
        if parent.get('experiment')!='E1' or parent.get('draft_steps')!=2: raise ValueError('E3 parent must be frozen E1-D2')
        conditions=[dict(parent,experiment='E3',history_anchor='previous-executed-round',parent=parent)]
        provenance=ROOT/'policies/realtime-vla-flash/results/experiment11_layer_skip_prefix_20261010/provenance/preflight.json'
        checkpoint=json.loads(provenance.read_text())['checkpoint']
    rows=[]
    for config in conditions:
        if args.only_mask and config.get('mask') not in args.only_mask.split(','): continue
        cid=f"E2_{config['mask']}" if args.experiment=='E2' else f"E3_D2_tau{config['tau']}_P{config['period']}_previous_anchor"
        # Expansion is a separate immutable attempt containing only new inits.
        for init in ([6] if not args.expand else range(7,11)):
            for suite in SUITES:
                for task in range(10):
                    identity=dict(experiment=args.experiment,condition=cid,config=config,checkpoint=checkpoint,
                        suite=suite,task=task,init=init,seed_rule='idea1-v22/1701 suite/task/init/replan')
                    rows.append(dict(identity,key=digest(identity),owner=len(rows)%args.shards,execution='A'))
    unique={x['condition']:dict(id=x['condition'],config=x['config']) for x in rows}
    return dict(schema=1,stage=args.experiment,shards=args.shards,tasks=rows,
        conditions=list(unique.values()),
        no_pi0_baseline=True,created=time.time())


def worker(args):
    from preparations.idea1_v2_1.runtime_client import rpc
    from preparations.idea1_v3.branch_episode import run
    data=json.loads((args.attempt/'manifest.json').read_text()); wd=args.attempt/'workers'/f'worker{args.shard}'
    wd.mkdir(parents=True,exist_ok=True)
    with (wd/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu=verify_gpu(args.gpu); atomic(wd/'identity.json',dict(gpu,physical_gpu=args.gpu,pid=os.getpid()))
        # Record only the small new source revision; never rehash frozen outputs.
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        atomic(wd/'source_revision.json',sources)
        server=None; log=None; config=None
        try:
            for row in data['tasks']:
                if row['owner']!=args.shard: continue
                if (args.attempt/'STOP').exists(): break
                if shutil.disk_usage(args.attempt).free<2*1024**3: raise RuntimeError('disk free below 2GiB')
                directory=args.attempt/'episodes'/row['experiment']/row['condition']/'A'/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"
                if (directory/'complete.json').exists():
                    if json.loads((directory/'complete.json').read_text())['key']!=row['key']: raise ValueError('completion key mismatch')
                    continue
                if directory.exists():
                    suffix=1
                    while directory.with_name(directory.name+f'.interrupted{suffix}').exists(): suffix+=1
                    directory.rename(directory.with_name(directory.name+f'.interrupted{suffix}'))
                if config!=row['config']:
                    if server: server.terminate(); server.wait(timeout=30); log.close()
                    with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
                    python=ROOT/'policies/openpi/.venv/bin/python' if row['experiment']=='E2' else MODEL_PYTHON
                    log=(wd/f'server_{time.time_ns()}.log').open('a'); tick=time.perf_counter()
                    server=subprocess.Popen([str(python),'-m','preparations.idea1_v3.branch_server','--port',str(port),
                        '--config',json.dumps(row['config']),'--launch'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                    deadline=time.monotonic()+1800
                    while True:
                        if server.poll() is not None: raise RuntimeError('branch server exited; inspect server log')
                        try: rpc(port,dict(op='health')); break
                        except (OSError,RuntimeError):
                            if time.monotonic()>deadline: raise TimeoutError('branch model startup')
                            time.sleep(1)
                    config=row['config']; append(wd/'events.jsonl',dict(kind='startup',config=config,seconds=time.perf_counter()-tick))
                episode_run=run
                if row['experiment']=='E3':
                    from preparations.idea1_v3.history_client import run as episode_run
                try: episode_run(directory,port,row,args.attempt/'profiling'/f'profile_worker{args.shard}.jsonl')
                except Exception:
                    append(wd/'events.jsonl',dict(kind='invalid',key=row['key'],error=traceback.format_exc())); raise
        finally:
            if server:
                server.terminate()
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired: server.kill(); server.wait()
            if log: log.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('prepare','worker','stop','resume'))
    parser.add_argument('--attempt',type=Path,required=True); parser.add_argument('--experiment',choices=('E2','E3'),default='E2')
    parser.add_argument('--parent'); parser.add_argument('--only-mask'); parser.add_argument('--expand',action='store_true')
    parser.add_argument('--shards',type=int,default=1); parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--gpu',type=int,default=0); parser.add_argument('--launch',action='store_true')
    args=parser.parse_args(); args.attempt=args.attempt.resolve()
    if not args.launch:
        print(json.dumps(manifest(args) if args.command=='prepare' else dict(command=args.command,dry_run=True),indent=2)); return
    if args.command=='prepare':
        if (args.attempt/'manifest.json').exists(): raise FileExistsError('immutable existing manifest')
        atomic(args.attempt/'manifest.json',manifest(args))
    elif args.command=='stop': atomic(args.attempt/'STOP',dict(time=time.time()))
    elif args.command=='resume':
        if (args.attempt/'STOP').exists(): (args.attempt/'STOP').unlink()
    else:
        from preparations.idea1_v2_2.common import evaluation_env
        env=evaluation_env(args.gpu); env['PYTHONPATH']=str(ROOT)+':'+env['PYTHONPATH']; env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
        if Path(sys.executable).resolve()!=EVAL_PYTHON.resolve() or os.environ.get('V3_BRANCH_CHILD')!='1':
            env['V3_BRANCH_CHILD']='1'
            subprocess.run([str(EVAL_PYTHON),'-m','preparations.idea1_v3.branches',*sys.argv[1:]],cwd=ROOT,env=env,check=True)
        else: worker(args)

if __name__=='__main__': main()
