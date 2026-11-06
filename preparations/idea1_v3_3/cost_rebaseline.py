"""Drain one scientific condition, borrow its resident service for cost80.

Only the dispatcher is stopped; current scientific rollout finishes normally.
Engineering data is nested outside every scientific completion glob.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import numpy as np
from preparations.idea1_v3.io import atomic,digest,append
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v2_1.runtime_client import rpc
from .dispatch import ROOT,OUT,PY
from .egl_mapping import worker_environment
from .manifest import cases


def active_rollouts(owner):
    rows=subprocess.check_output(['ps','-eo','pid,ppid,stat,args'],text=True).splitlines();result=[]
    for line in rows:
        p=line.split(None,3)
        if len(p)==4 and p[1]==str(owner) and p[2][0]!='Z' and 'preparations.idea1_v3_3.rollout' in p[3]:
            args=shlex.split(p[3]);result.append(dict(pid=int(p[0]),condition=args[args.index('--condition')+1],command=args))
    return result


def compare(base,new):
    checks=[]
    for case in cases('pi0'):
        tail=Path('episodes')/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
        old,newdir=base/tail,new/tail
        a,b=np.load(old/'arrays.npz',allow_pickle=False),np.load(newdir/'arrays.npz',allow_pickle=False)
        fields=sorted((k for k in a.files if k.startswith('actions_')),key=lambda k:int(k.split('_')[1]));other=sorted((k for k in b.files if k.startswith('actions_')),key=lambda k:int(k.split('_')[1]))
        raw=[f'raw_{i}' for i in range(len(fields))]
        oi=[json.loads(l) for l in (old/'infer.jsonl').read_text().splitlines()];ni=[json.loads(l) for l in (newdir/'infer.jsonl').read_text().splitlines()]
        ores=json.loads((old/'result.json').read_text());nres=json.loads((newdir/'result.json').read_text())
        checks.append(dict(case=case,decoded_exact=fields==other and all(np.array_equal(a[k],b[k]) for k in fields),raw_exact=fields==other and all(np.array_equal(a[k],b[k]) for k in raw),
            execution_lengths_equal=[r['executed_length'] for r in oi]==[r['executed_length'] for r in ni],routes_equal=[r['route'] for r in oi]==[r['route'] for r in ni],success_equal=ores['success']==nres['success'],
            old_result_sha256=hashlib.sha256((old/'result.json').read_bytes()).hexdigest(),new_result_sha256=hashlib.sha256((newdir/'result.json').read_bytes()).hexdigest()))
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True,type=int);parser.add_argument('--owner',required=True,type=int);args=parser.parse_args()
    gpu,owner=args.gpu,args.owner;port=9340+gpu;root=OUT/'engineering_cost_rebaseline';root.mkdir(exist_ok=True);status=root/f'gpu{gpu}_status.json'
    if b'preparations.idea1_v3_3.' not in Path(f'/proc/{owner}/cmdline').read_bytes():raise ValueError('Owner PID identity changed')
    before=active_rollouts(owner);start=time.monotonic();child=None
    try:
        while active_rollouts(owner):
            if time.monotonic()-start>7200:raise TimeoutError('Scientific condition drain')
            atomic(status,dict(state='waiting_current_scientific_condition',gpu=gpu,owner=owner,science_children=active_rollouts(owner),time=time.time()))
            time.sleep(5)
        completed=before[0]['condition'] if before else None
        if completed:
            n=len(list((OUT/completed/'episodes').glob('*/task*/init*/complete.json')))
            if n!=80:raise ValueError('Scientific rollout exited before its complete80; do not borrow RPC')
        parent=json.loads((OUT/'manifest.json').read_text());cfg=dict(next(c for c in parent['conditions'] if c['id']=='pi0_parent'),id=f'pi0_parent_cost_gpu{gpu}')
        health=rpc(port,dict(op='health'));modelpid=health['protocol']['server_pid']
        m=dict(stage='engineering_cost_rebaseline',engineering_only=True,budget_episodes=80,scientific_denominator_increment=0,conditions=[cfg],cases=parent['cases'],scientific_reference='pi0_parent',parent_manifest_hash=parent['manifest_hash'],
            actual_model_gpu=gpu,actual_render_gpu=gpu,correct_EGL_index=3-gpu,retained_model_pid=modelpid,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reason='Match corrected EGL/current three-pi0 workload; raw full-policy wall retained; no science sample overwrite')
        m['manifest_hash']=digest(m);manifest=root/f'gpu{gpu}_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=m:raise ValueError('Engineering manifest already frozen differently')
        atomic(manifest,m)
        receipt=root/f'gpu{gpu}_boundary_receipt.json'
        atomic(receipt,dict(time=time.time(),gpu=gpu,owner_paused=owner,science_rollout_before=before,science_rollout_after=active_rollouts(owner),drained_complete_condition=completed,
            retained_model_pid=modelpid,model_health=health,no_concurrent_scientific_RPC=True,model_reload=False,engineering_manifest_hash=m['manifest_hash'],budget=80,output=str(root/cfg['id'])))
        env=worker_environment(environment(gpu),gpu)
        command=[str(PY['env']),'-m','preparations.idea1_v3_3.rollout_bound','--manifest',str(manifest),'--condition',cfg['id'],'--output',str(root),'--port',str(port)]
        with (root/f'gpu{gpu}_rollout.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                n=len(list((root/cfg['id']/'episodes').glob('*/task*/init*/complete.json')))
                atomic(status,dict(state='running_engineering80',gpu=gpu,owner_paused=owner,render_pid=child.pid,valid=n,time=time.time()))
                if time.monotonic()-start>10800:raise TimeoutError('Engineering bounded total wall')
                time.sleep(10)
            if child.returncode:raise RuntimeError('Engineering rollout failed; inspect isolated log')
        new=root/cfg['id'];checks=compare(OUT/'pi0_parent',new)
        rows=[json.loads(p.read_text()) for p in new.glob('episodes/*/task*/init*/result.json') if '.invalid.' not in str(p)]
        atomic(root/f'gpu{gpu}_comparison.json',dict(time=time.time(),gpu=gpu,cases=len(rows),successes=sum(r['success'] for r in rows),policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in rows)/sum(r['actions'] for r in rows),
            decoded_all_exact=all(c['decoded_exact'] for c in checks),raw_all_exact=all(c['raw_exact'] for c in checks),routes_all_equal=all(c['routes_equal'] for c in checks),success_all_equal=all(c['success_equal'] for c in checks),checks=checks,
            cost_scope='Complete policy wall / all actually executed actions, original request timings preserved. Engineering80 only; original science results unchanged.'))
        atomic(status,dict(state='engineering80_complete',gpu=gpu,valid=80,time=time.time()))
    except Exception as exc:
        atomic(root/f'gpu{gpu}_error.json',dict(error=repr(exc),time=time.time()))
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
        raise
    finally:
        # Restore scientific queue even if engineering evidence cannot be used.
        os.kill(owner,signal.SIGCONT)
        append(OUT/'campaign_events.jsonl',dict(event='cost_rebaseline_parent_resumed',gpu=gpu,owner_pid=owner,time=time.time()))


if __name__=='__main__':main()
