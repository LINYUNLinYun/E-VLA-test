"""Independent E2 supplement manifest/worker. Default is dry run.

--masks JSON is an explicit object mapping new condition names to skip lists.
Only init6 is prepared (40 episodes/condition); no old Full quality rerun.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from preparations.idea1_v3.cli import ROOT,SUITES,EVAL_PYTHON
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_supplement.pi05 import validate_skip,REVISION


def manifest(args):
    from preparations.idea1_v3.pi05 import CHECKPOINT
    masks=json.loads(args.masks)
    if not isinstance(masks,dict) or not masks: raise ValueError('Explicit nonempty name->skip JSON required')
    conditions=[]; rows=[]
    norm=CHECKPOINT/'assets/physical-intelligence/libero/norm_stats.json'
    checkpoint=dict(path=str(CHECKPOINT),config='pi05_libero',native_H=10,D=10,norm_stats_sha256=hashlib.sha256(norm.read_bytes()).hexdigest())
    for name,skip in masks.items():
        if not name.replace('_','').isalnum() or name=='Full': raise ValueError('New simple mask name required; no Full rerun')
        config=dict(experiment='E2',mask=name,skip=list(validate_skip(skip)),revision=REVISION)
        cid='E2_'+name; conditions.append(dict(id=cid,config=config))
        for suite in SUITES:
            for task in range(10):
                identity=dict(experiment='E2',condition=cid,config=config,checkpoint=checkpoint,suite=suite,task=task,init=6,seed_rule='idea1-v22/1701 suite/task/init/replan')
                rows.append(dict(identity,key=digest(identity),owner=len(rows)%args.shards,execution='A'))
    return dict(schema=1,stage='E2',revision=REVISION,shards=args.shards,conditions=conditions,tasks=rows,no_pi0_baseline=True)


def worker(args):
    from preparations.idea1_v3 import branches
    original=subprocess.Popen
    def popen(command,*a,**kw):
        command=list(command)
        if 'preparations.idea1_v3.branch_server' in command:
            command[command.index('preparations.idea1_v3.branch_server')]='preparations.idea1_v3_supplement.server'
        return original(command,*a,**kw)
    revision={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    atomic(args.attempt/'workers'/f'worker{args.shard}'/'supplement_source_revision.json',revision)
    subprocess.Popen=popen
    try: branches.worker(args)
    finally: subprocess.Popen=original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','worker','stop','resume'))
    p.add_argument('--attempt',type=Path,required=True); p.add_argument('--masks')
    p.add_argument('--shards',type=int,default=1); p.add_argument('--shard',type=int,default=0)
    p.add_argument('--gpu',type=int,default=0); p.add_argument('--launch',action='store_true')
    args=p.parse_args(); args.attempt=args.attempt.resolve()
    if args.shards<1: p.error('positive shards required')
    if args.command=='prepare':
        if not args.masks: p.error('--masks required')
        data=manifest(args)
        if not args.launch: print(json.dumps(data,indent=2)); return
        if (args.attempt/'manifest.json').exists(): raise FileExistsError('immutable manifest')
        atomic(args.attempt/'manifest.json',data); return
    if not args.launch: print(json.dumps(dict(command=args.command,dry_run=True))); return
    if args.command=='stop': atomic(args.attempt/'STOP',dict(stop=True)); return
    if args.command=='resume':
        (args.attempt/'STOP').unlink(missing_ok=True); return
    if Path(sys.executable).resolve()!=EVAL_PYTHON.resolve() or os.environ.get('V3_SUPPLEMENT_CHILD')!='1':
        from preparations.idea1_v2_2.common import evaluation_env
        env=evaluation_env(args.gpu); env.update(V3_SUPPLEMENT_CHILD='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        env['PYTHONPATH']=str(ROOT)+':'+env['PYTHONPATH']
        subprocess.run([str(EVAL_PYTHON),'-m','preparations.idea1_v3_supplement.cli',*sys.argv[1:]],cwd=ROOT,env=env,check=True)
    else: worker(args)


if __name__=='__main__': main()
