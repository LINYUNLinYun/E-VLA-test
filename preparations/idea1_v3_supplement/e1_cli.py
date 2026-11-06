"""Explicit frozen E1 supplement configurations and safe static workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from preparations.idea1_v3.cli import ROOT,SUITES,EVAL_PYTHON
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3.safety_runtime import require_recovery_pass,kernel_digest
from preparations.idea1_v3_supplement.policy import validate,REVISION


def phase_a():
    return [dict(experiment='E1',draft_steps=2,tau=tau,period=period,gripper_rule='strict',reject_steps=10)
        for tau,period in [(t,p) for t in (.10,.25,.40) for p in (4,8)]+[(.25,None),(.40,None)]]


def manifest(args):
    configs=phase_a() if args.phase_a else json.loads(args.configs)
    if not isinstance(configs,list) or not configs or len(configs)>8: raise ValueError('Explicit 1..8 frozen configs required')
    provenance=ROOT/'policies/realtime-vla-flash/results/experiment11_layer_skip_prefix_20261010/provenance/preflight.json'
    checkpoint=json.loads(provenance.read_text())['checkpoint']; rows=[]; conditions=[]
    for config in configs:
        validate(config)
        config=dict(config,draft_steps=2,gripper_rule=config.get('gripper_rule','strict'),reject_steps=config.get('reject_steps',10),revision=REVISION,safety_revision='qk-row-bounds-v1',safety_sha256=kernel_digest())
        period='none' if config['period'] is None else config['period']
        cid=f"supp_E1_D2_tau{config['tau']}_P{period}_G{config['gripper_rule']}_rejectD{config['reject_steps']}"
        if cid in {c['id'] for c in conditions}: raise ValueError('Duplicate semantic config')
        conditions.append(dict(id=cid,config=config))
        for suite in SUITES:
            for task in range(10):
                identity=dict(experiment='E1',condition=cid,config=config,checkpoint=checkpoint,suite=suite,task=task,init=6,seed_rule='idea1-v22/1701 suite/task/init/replan')
                rows.append(dict(identity,key=digest(identity),owner=len(rows)%args.shards,execution='A'))
    return dict(schema=1,stage='E1',revision=REVISION,shards=args.shards,conditions=conditions,tasks=rows,no_pi0_baseline=True)


def worker(args,data):
    gate=require_recovery_pass(args.safety_gate)
    if not 0<=args.shard<data['shards']: raise ValueError('Static shard outside manifest')
    wd=args.attempt/'workers'/f'worker{args.shard}'; wd.mkdir(parents=True,exist_ok=True)
    events=wd/'events.jsonl'
    if events.exists():
        invalid={json.loads(line)['key'] for line in events.read_text().splitlines() if json.loads(line).get('kind')=='invalid'}
        for row in data['tasks']:
            complete=args.attempt/'episodes'/row['experiment']/row['condition']/'A'/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/'complete.json'
            if row['owner']==args.shard and row['key'] in invalid and not complete.exists(): raise RuntimeError('Unresolved technical invalid; diagnose, no automatic retry')
    stamp=time.time_ns(); reference=wd/f'qk_reference_{stamp}.json'
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    atomic(wd/f'supplement_revision_{stamp}.json',dict(sources=sources,safety_gate=gate,reference_receipt=str(reference)))
    original=subprocess.Popen
    def popen(command,*a,**kw):
        command=list(command)
        if 'preparations.idea1_v3.server' in command:
            command[command.index('preparations.idea1_v3.server')]='preparations.idea1_v3_supplement.e1_server'
            kw['env']=dict(kw.get('env',os.environ),V3_QK_WORKER_RECEIPT=str(reference))
        return original(command,*a,**kw)
    subprocess.Popen=popen
    try:
        from preparations.idea1_v3.cli import worker as base_worker
        base_worker(args)
    finally: subprocess.Popen=original


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','worker','stop','resume'))
    p.add_argument('--attempt',type=Path,required=True); p.add_argument('--phase-a',action='store_true'); p.add_argument('--configs')
    p.add_argument('--shards',type=int,default=2); p.add_argument('--shard',type=int,default=0); p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--safety-gate',type=Path); p.add_argument('--launch',action='store_true')
    args=p.parse_args(); args.attempt=args.attempt.resolve()
    if args.shards<1: p.error('positive shards required')
    if args.command=='prepare':
        if args.phase_a==bool(args.configs): p.error('choose --phase-a or --configs JSON')
        data=manifest(args)
        if not args.launch: print(json.dumps(dict(dry_run=True,conditions=data['conditions'],tasks=len(data['tasks']),shards=data['shards']),indent=2)); return
        if (args.attempt/'manifest.json').exists(): raise FileExistsError('immutable existing manifest')
        atomic(args.attempt/'manifest.json',data); return
    if not args.launch: print(json.dumps(dict(command=args.command,dry_run=True))); return
    if args.command=='stop': atomic(args.attempt/'STOP',dict(stop=True)); return
    if args.command=='resume': (args.attempt/'STOP').unlink(missing_ok=True); return
    if not args.safety_gate: p.error('completed --safety-gate required')
    args.safety_gate=args.safety_gate.resolve(); require_recovery_pass(args.safety_gate)
    if Path(sys.executable).resolve()!=EVAL_PYTHON.resolve() or os.environ.get('V3_SUPPLEMENT_E1_CHILD')!='1':
        from preparations.idea1_v2_2.common import evaluation_env
        env=evaluation_env(args.gpu); env.update(V3_SUPPLEMENT_E1_CHILD='1',PYTHONPATH=str(ROOT)+':'+env['PYTHONPATH'])
        subprocess.run([str(EVAL_PYTHON),'-m','preparations.idea1_v3_supplement.e1_cli',*sys.argv[1:]],cwd=ROOT,env=env,check=True)
    else: worker(args,json.loads((args.attempt/'manifest.json').read_text()))


if __name__=='__main__': main()
