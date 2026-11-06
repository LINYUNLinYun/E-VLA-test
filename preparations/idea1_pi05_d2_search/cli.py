"""Explicit skip-n/D2 screen manifests and static workers; dry-run default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from preparations.idea1_v3.cli import ROOT,SUITES,EVAL_PYTHON
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3.pi05 import CHECKPOINT
from preparations.idea1_pi05_d2_search.policy import validate_skip,REVISION


def manifest(args):
    masks=json.loads(args.masks)
    if not isinstance(masks,dict) or not 1<=len(masks)<=20: raise ValueError('Explicit1..20 mask-name->skip lists required')
    norm=CHECKPOINT/'assets/physical-intelligence/libero/norm_stats.json'
    checkpoint=dict(path=str(CHECKPOINT),config='pi05_libero',native_H=10,D=2,norm_stats_sha256=hashlib.sha256(norm.read_bytes()).hexdigest())
    rows=[]; conditions=[]; unique=set()
    for name,skip in masks.items():
        if not isinstance(name,str) or not name.replace('_','').isalnum(): raise ValueError('Simple explicit mask names required')
        skip=validate_skip(skip)
        if skip in unique: raise ValueError('Duplicate semantic skip list')
        unique.add(skip)
        config=dict(experiment='E2',model='pi05',mask=name,skip=list(skip),denoise_steps=2,denoise_nodes=[1.,.5,0.],H=10,R=5,revision=REVISION)
        cid='pi05D2_'+name; conditions.append(dict(id=cid,config=config))
        for suite in SUITES:
            for task in range(10):
                identity=dict(experiment='E2',condition=cid,config=config,checkpoint=checkpoint,suite=suite,task=task,init=6,seed_rule='idea1-v22/1701 suite/task/init/replan')
                rows.append(dict(identity,key=digest(identity),owner=len(rows)%args.shards,execution='A'))
    return dict(schema=1,stage='E2',search='pi05_skip_n_D2',revision=REVISION,shards=args.shards,conditions=conditions,tasks=rows,
        quality_threshold=dict(overall=.65,per_suite_gate=False),created_reference='FullD2 new quality; old FullD10 quality only reused',screen_limit_conditions=20)


def worker(args):
    from preparations.idea1_v3 import branches
    data=json.loads((args.attempt/'manifest.json').read_text())
    if data.get('revision')!=REVISION: raise ValueError('Explicit D2 search manifest required')
    if not 0<=args.shard<data['shards']: raise ValueError('Static shard outside manifest')
    wd=args.attempt/'workers'/f'worker{args.shard}'; wd.mkdir(parents=True,exist_ok=True)
    eventpath=wd/'events.jsonl'
    if eventpath.exists():
        invalid={e['key'] for e in map(json.loads,eventpath.read_text().splitlines()) if e.get('kind')=='invalid'}
        for row in data['tasks']:
            completed=args.attempt/'episodes'/'E2'/row['condition']/'A'/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/'complete.json'
            if row['owner']==args.shard and row['key'] in invalid and not completed.exists(): raise RuntimeError('Unresolved technical invalid requires diagnosis; no automatic retry')
    sources={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text()) for p in Path(__file__).parent.glob('*.py')}
    atomic(wd/'d2_search_source_revision.json',sources)
    original=subprocess.Popen
    def popen(command,*a,**kw):
        command=list(command)
        if 'preparations.idea1_v3.branch_server' in command: command[command.index('preparations.idea1_v3.branch_server')]='preparations.idea1_pi05_d2_search.server'
        return original(command,*a,**kw)
    subprocess.Popen=popen
    try: branches.worker(args)
    finally: subprocess.Popen=original


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','worker','stop','resume'))
    p.add_argument('--attempt',type=Path,required=True); p.add_argument('--masks'); p.add_argument('--shards',type=int,default=4); p.add_argument('--shard',type=int,default=0); p.add_argument('--gpu',type=int,default=0); p.add_argument('--launch',action='store_true')
    args=p.parse_args(); args.attempt=args.attempt.resolve()
    if args.shards<1: p.error('positive shards required')
    if args.command=='prepare':
        if not args.masks: p.error('--masks JSON required')
        data=manifest(args)
        if not args.launch: print(json.dumps(dict(dry_run=True,conditions=data['conditions'],tasks=len(data['tasks']),shards=data['shards']),indent=2)); return
        if (args.attempt/'manifest.json').exists(): raise FileExistsError('immutable manifest')
        atomic(args.attempt/'manifest.json',data); return
    if not args.launch: print(json.dumps(dict(command=args.command,dry_run=True))); return
    if args.command=='stop': atomic(args.attempt/'STOP',dict(stop=True)); return
    if args.command=='resume': (args.attempt/'STOP').unlink(missing_ok=True); return
    if Path(sys.executable).resolve()!=EVAL_PYTHON.resolve() or os.environ.get('PI05_D2_SEARCH_CHILD')!='1':
        from preparations.idea1_v2_2.common import evaluation_env
        env=evaluation_env(args.gpu); env.update(PI05_D2_SEARCH_CHILD='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONPATH=str(ROOT)+':'+env['PYTHONPATH'])
        subprocess.run([str(EVAL_PYTHON),'-m','preparations.idea1_pi05_d2_search.cli',*sys.argv[1:]],cwd=ROOT,env=env,check=True)
    else: worker(args)


if __name__=='__main__': main()
