import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from .manifest import ROOT,OUT


def snapshot():
    m=json.loads((OUT/'manifest.json').read_text()); rows=[]
    for cfg in m['conditions']:
        found=[json.loads(p.read_text()) for p in (OUT/cfg['id']/'episodes').glob('*/task*/init*/result.json')]
        valid=[r for r in found if r.get('valid')]
        rows.append(dict(condition=cfg['id'],completed=len(valid),target=cfg['n'],success=sum(r['success'] for r in valid),
            policy_s=sum(r['policy_wall_s'] for r in valid),wall_s=sum(r['wall_s'] for r in valid),actions=sum(r['actions'] for r in valid)))
    atomic(OUT/'progress.json',dict(time=time.time(),conditions=rows,total_completed=sum(r['completed'] for r in rows),total_target=sum(r['target'] for r in rows)))
    return rows


def main():
    with lock(OUT/'controller.lock',blocking=False):
        m=json.loads((OUT/'manifest.json').read_text())
        for name,expected in m['source_files'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected: raise ValueError('Frozen source changed '+name)
        processes={}; logs=[]
        for shard,gpu in enumerate((0,2,3)):
            wd=OUT/'workers'/f'gpu{gpu}'; wd.mkdir(parents=True,exist_ok=True)
            log=(wd/'dispatch.log').open('a'); logs.append(log)
            process=subprocess.Popen([sys.executable,'-m','preparations.idea1_v3_4.dispatch','--gpu',str(gpu),'--shard',str(shard)],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
            processes[gpu]=process
        atomic(OUT/'controller_process.json',dict(pid=os.getpid(),workers={str(g):p.pid for g,p in processes.items()},started=time.time()))
        while any(p.poll() is None for p in processes.values()):
            snapshot()
            atomic(OUT/'status.json',dict(status='running',time=time.time(),workers={str(g):dict(pid=p.pid,exit=p.poll()) for g,p in processes.items()}))
            time.sleep(30)
        snapshot(); failed=any(p.returncode for p in processes.values())
        atomic(OUT/'status.json',dict(status='failed' if failed else 'rollouts_complete',time=time.time(),workers={str(g):dict(pid=p.pid,exit=p.returncode) for g,p in processes.items()}))
        for log in logs: log.close()
        if failed: raise RuntimeError('Worker failed; inspect evidence before recovery')

if __name__=='__main__': main()
