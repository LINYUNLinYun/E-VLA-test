"""Durable two-GPU rollout/extraction dispatch with existing pilot reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from preparations.idea1_v3_1.server import rpc
from .common import ROOT,SUITES,PYTHONS,atomic_json,episode_path,worker_env
from .selection import select_episode


def run(root,stage):
    processes=[]
    cpus=sorted(os.sched_getaffinity(0))
    for shard,gpu,port in ((0,1,48161),(1,2,48162)):
        if stage=='rollout':
            cases=[f'{s}/{t}/{i}' for si,s in enumerate(SUITES) for t in range(10)
                   for i in range(3) if (si*10+t)%2==shard]
            command=[str(PYTHONS['environment']),'-u','-m','preparations.idea1_visualization.rollout',
                     '--root',str(root),'--port',str(port),'--cases',*cases]
            env=worker_env(gpu,render=True)
        else:
            command=[sys.executable,'-u','-m','preparations.idea1_visualization.extract',
                     '--root',str(root),'--port',str(port),'--shard',str(shard),'--shards','2']
            env=worker_env(gpu)
        command=['taskset','-c',','.join(map(str,cpus[16+shard*8:24+shard*8])),*command]
        log=(root/f'{stage}_shard{shard}.log').open('a')
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close()
        processes.append(process)
        atomic_json(root/f'{stage}_shard{shard}_process.json',dict(pid=process.pid,command=command))
    while any(p.poll() is None for p in processes):
        failures=[p.returncode for p in processes if p.poll() not in (None,0)]
        if failures:
            # Let other active scientific work finish its episode; retain output.
            raise RuntimeError(f'{stage} shard failed: {failures}')
        if stage=='rollout':
            completed=sum((episode_path(root,s,t,i)/'complete.json').exists()
                          for s in SUITES for t in range(10) for i in range(3))
        else:
            completed=sum(json.loads(p.read_text())['completed'] for p in root.glob('extract_progress_*.json'))
        atomic_json(root/'status.json',dict(stage=stage,status='running',completed=completed,
                    updated_at=time.time(),controller_pid=os.getpid(),worker_pids=[p.pid for p in processes]))
        time.sleep(10)
    if any(p.returncode!=0 for p in processes):
        raise RuntimeError(f'{stage} failed: {[p.returncode for p in processes]}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    args=p.parse_args();root=Path(args.root)
    stage='pilot_gate'
    try:
        records=[]
        for suite in SUITES:
            directory=episode_path(root,suite,0,0)
            if not (directory/'complete.json').exists():
                raise ValueError(f'Pilot incomplete: {directory}')
            samples=select_episode(directory)
            # First and last selected inputs span different task progress.
            for sample in (samples[0],samples[-1]):
                output=root/'features'/suite/'task00/init00'/f"request_{sample['request']:04d}"
                records.append(rpc(48161,dict(op='extract',request_path=sample['path'],output=str(output))))
        atomic_json(root/'pilot_gate.json',dict(passed=all(r.get('passed') for r in records),
                    samples=records,episodes=4,selected_requests=8))
        # Freeze the experiment-specific implementation used for formal work.
        source={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob('*.py')}
        atomic_json(root/'source_identity.json',source)
        stage='rollout';run(root,stage)
        total=sum((episode_path(root,s,t,i)/'complete.json').exists()
                  for s in SUITES for t in range(10) for i in range(3))
        if total!=120:
            raise ValueError(f'Expected 120 episodes, got {total}')
        expected=0
        for suite in SUITES:
            for task in range(10):
                for init in range(3):
                    expected+=len(select_episode(episode_path(root,suite,task,init)))
        atomic_json(root/'collection_complete.json',dict(episodes=120,selected_requests=expected))
        stage='extract';run(root,stage)
        actual=sum(json.loads(p.read_text())['completed'] for p in root.glob('extract_done_*.json'))
        if actual!=expected:
            raise ValueError(f'Feature count {actual} != selected count {expected}')
        atomic_json(root/'features_complete.json',dict(episodes=120,selected_requests=expected,passed=True))
        stage='analysis'
        atomic_json(root/'status.json',dict(stage=stage,status='running',updated_at=time.time()))
        with (root/'analysis.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','preparations.idea1_visualization.analyze','--root',str(root)],
                           cwd=ROOT,env=worker_env(1),stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic_json(root/'status.json',dict(stage='complete',status='complete',episodes=120,
                    selected_requests=expected,updated_at=time.time()))
        # Release only the two model services started for this experiment.
        for port in (48161,48162):
            rpc(port,dict(op='stop'))
    except Exception as exc:
        atomic_json(root/'status.json',dict(stage=stage,status='technical_failure',error=repr(exc),
                    updated_at=time.time(),controller_pid=os.getpid()))
        raise


if __name__=='__main__':
    main()
