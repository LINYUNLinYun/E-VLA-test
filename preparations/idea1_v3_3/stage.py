"""Run a subsequent frozen stage using the same persistent B1 service code.

Stage manifests contain only selected conditions and exact additional case sets.
Existing complete scientific cases are skipped by the unchanged rollout runner.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v3.cli import verify_gpu
from preparations.idea1_v2_1.runtime_client import rpc
from .dispatch import ROOT,OUT,PY
from .egl_mapping import worker_environment


def run(manifest,gpu,model,shard,shards):
    m=json.loads(Path(manifest).read_text()); name=m['stage']; wd=OUT/'stages'/name/f'gpu{gpu}'; wd.mkdir(parents=True,exist_ok=True)
    with (OUT/'workers'/f'gpu{gpu}'/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=verify_gpu(gpu); atomic(wd/'identity.json',dict(identity,gpu=gpu,pid=os.getpid(),manifest=str(manifest)))
        env=environment(gpu)
        if model=='pi0': env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        port=9350+gpu
        with (wd/'service.log').open('a') as log:
            module=('preparations.idea1_v3_3.server_c1' if (OUT/'E2_manifest.json').exists() else 'preparations.idea1_v3_3.server_r2') if model=='pi0' else 'preparations.idea1_v3_3.server'
            server=subprocess.Popen([str(PY[model]),'-m',module,'--model',model,'--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                start=time.perf_counter()
                while True:
                    if server.poll() is not None: raise RuntimeError('Service failed')
                    try: health=rpc(port,dict(op='health')); break
                    except (ConnectionError,OSError):
                        if time.perf_counter()-start>1200: raise TimeoutError('Readiness')
                        time.sleep(1)
                atomic(wd/'loaded_protocol.json',dict(health,load_s=time.perf_counter()-start))
                configs=[c for c in m['conditions'] if c['model']==model]
                if not 1<=shards<=10 or not 0<=shard<shards:raise ValueError('Task-case shard outside1..10')
                probed=False
                for index,c in enumerate(configs):
                    state=OUT/'stages'/name/'conditions'/c['id']; state.mkdir(parents=True,exist_ok=True)
                    shard_receipt=state/f'gpu{gpu}_shard{shard}of{shards}.done.json'
                    if shard_receipt.exists(): continue
                    owned=[case for case in m['cases'][model] if case['task']%shards==shard]
                    paths=[OUT/c['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"/'complete.json' for case in owned]
                    incomplete=any(not p.exists() for p in paths)
                    command=[str(PY['env']),'-m','preparations.idea1_v3_3.rollout_bound','--manifest',str(manifest),'--condition',c['id'],'--output',str(OUT),'--port',str(port),'--shard',str(shard),'--shards',str(shards)]
                    if not probed:command+=['--probe']
                    tick=time.perf_counter()
                    with (state/f'rollout_gpu{gpu}_shard{shard}of{shards}.log').open('a') as clientlog:
                        subprocess.run(command,cwd=ROOT,env=worker_environment(env,gpu),stdout=clientlog,stderr=subprocess.STDOUT,check=True)
                    if incomplete:probed=True
                    expected=[]
                    for case in owned:
                        path=OUT/c['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"/'complete.json'
                        if not path.exists():raise ValueError('Worker returned without every owned case complete')
                        expected.append(dict(case=case,complete=str(path)))
                    atomic(shard_receipt,dict(config=c,gpu=gpu,shard=shard,shards=shards,owned_cases=expected,wall_s=time.perf_counter()-tick,finished=time.time(),
                        scientific_manifest_hash=m['manifest_hash'],completion='Owned fixed task%shards case set, not whole condition; supervisor uses actual global case coverage'))
                if model=='xiaomi' and name=='bounded_followups':
                    # At this boundary every owned scientific rollout has
                    # returned. Keep the service resident during isolated80.
                    if shards!=1:raise ValueError('Xiaomi engineering hook requires one exclusive model-stage worker')
                    from .xiaomi_cost import run as engineering_cost80
                    engineering_cost80(gpu,port,manifest,os.getpid())
                atomic(wd/'finished.json',dict(time=time.time()))
            finally:
                server.terminate()
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired: server.kill(); server.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--gpu',required=True,type=int); p.add_argument('--model',required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=1); a=p.parse_args()
    run(a.manifest,a.gpu,a.model,a.shard,a.shards)
