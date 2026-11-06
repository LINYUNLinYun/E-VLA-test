"""Persistent per-GPU service, same task/service across scientific conditions."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3.cli import verify_gpu
from preparations.idea1_v2_1.runtime_client import rpc
from .manifest import ROOT,OUT
PYMODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
PYENV=ROOT/'policies/libero-eval-v21/.venv/bin/python'


def render_environment(env,gpu):
    # Enumerate EGL without CUDA_VISIBLE_DEVICES before model CUDA is imported.
    clean=dict(env); clean.pop('CUDA_VISIBLE_DEVICES',None)
    code='import json; from preparations.idea1_v3_3.egl_mapping import mapping; print(json.dumps(mapping()))'
    devices=json.loads(subprocess.check_output([str(PYENV),'-c',code],cwd=ROOT,env=clean,text=True).strip())
    selected=next(int(index) for index,physical in devices.items() if physical==gpu)
    result=dict(env)
    # Existing MuJoCo validates EGL index against visible CUDA device count.
    result.update(CUDA_VISIBLE_DEVICES=','.join(map(str,dict.fromkeys((gpu,selected)))),MUJOCO_EGL_DEVICE_ID=str(selected))
    return result,devices


def cross_service_gate(wd):
    # Every service has completed the same four-input initialization history.
    with lock(OUT/'cross_service.lock'):
        reference=OUT/'cross_service_reference.npz'
        if not reference.exists():
            reference.write_bytes((wd/'gate_actions.npz').read_bytes())
            atomic(OUT/'cross_service_reference.json',dict(worker=str(wd),sha256=hashlib.sha256(reference.read_bytes()).hexdigest()))
        checks=[]
        with np.load(reference) as ref,np.load(wd/'gate_actions.npz') as got:
            for key in ref.files:
                checks.append(dict(key=key,max_abs=float(np.abs(ref[key]-got[key]).max()),passed=bool(np.allclose(ref[key],got[key],atol=1e-5,rtol=1e-5))))
        passed=all(c['passed'] for c in checks)
        atomic(wd/'cross_service.json',dict(passed=passed,checks=checks,atol=1e-5,rtol=1e-5))
        if not passed: raise RuntimeError('Cross-service gate failed; do not pool services')


def select_mechanism(shard):
    pairs=[]
    for case in json.loads((OUT/'manifest.json').read_text())['cases']['full_d10']:
        if case['init']!=14 or case['task']%3!=shard: continue
        d=OUT/'full_d10'/'episodes'/case['suite']/f"task{case['task']:02d}"/'init14'
        result=json.loads((d/'result.json').read_text())
        if not result['valid']: raise ValueError('Invalid source Full episode')
        rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
        available=[(old,cur) for old,cur in zip(rows,rows[1:]) if cur['replan']==old['replan']+1 and cur['step']-old['step']==5 and old['executed_length']==5]
        if not available: raise ValueError('No sequential R5 input pair')
        indices=np.unique(np.rint(np.linspace(0,len(available)-1,min(4,len(available)))).astype(int))
        for index in indices:
            old,cur=available[index]; pair=dict(episode=str(d.relative_to(OUT/'full_d10'/'episodes')),case=case)
            for name,row in (('previous',old),('current',cur)):
                pair[name]=dict(path=row['input_path'],sha256=row['input_file_sha256'],seed=row['seed'],request=row['replan'],step=row['step'],input_hash=row['input_hash'])
            pairs.append(pair)
    return pairs


def worker(gpu,shard):
    wd=OUT/'workers'/f'gpu{gpu}'; wd.mkdir(parents=True,exist_ok=True)
    with lock(wd/'owner.lock',blocking=False):
        identity=verify_gpu(gpu)
        env=environment(gpu); env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        render_env,devices=render_environment(env,gpu)
        atomic(wd/'identity.json',dict(identity,gpu=gpu,shard=shard,pid=os.getpid(),egl_mapping=devices,render_cuda_mask=render_env['CUDA_VISIBLE_DEVICES'],egl_index=render_env['MUJOCO_EGL_DEVICE_ID']))
        port=19440+gpu
        with (wd/'service.log').open('a') as log:
            server=subprocess.Popen([str(PYMODEL),'-m','preparations.idea1_v3_4.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic(wd/'service_process.json',dict(pid=server.pid,port=port,started=time.time()))
            try:
                start=time.perf_counter(); health=None
                while time.perf_counter()-start<1200:
                    if server.poll() is not None: raise RuntimeError('Service exited '+str(server.returncode))
                    try: health=rpc(port,dict(op='health')); break
                    except (ConnectionError,OSError): time.sleep(1)
                if health is None: raise TimeoutError('Model readiness/gate')
                atomic(wd/'loaded_protocol.json',dict(health,load_and_gate_s=time.perf_counter()-start))
                cross_service_gate(wd)
                m=json.loads((OUT/'manifest.json').read_text())
                for stage in ('screen','mechanism','expand'):
                    if stage=='mechanism':
                        pairs=select_mechanism(shard)
                        atomic(wd/'mechanism_selection.json',pairs)
                        result=rpc(port,dict(op='mechanism',pairs=pairs,output=str(OUT/'mechanism'/f'shard{shard}')))
                        atomic(wd/'mechanism_done.json',result)
                        continue
                    for cfg in m['conditions']:
                        if stage=='expand' and cfg['n']==40: continue
                        atomic(wd/'state.json',dict(stage=stage,condition=cfg['id'],started=time.time(),service_pid=server.pid))
                        command=[str(PYENV),'-m','preparations.idea1_v3_4.rollout','--manifest',str(OUT/'manifest.json'),'--condition',cfg['id'],'--output',str(OUT),'--port',str(port),'--shard',str(shard),'--stage',stage]
                        tick=time.perf_counter()
                        with (wd/f'{stage}_{cfg["id"]}.log').open('a') as rollog:
                            completed=subprocess.run(command,cwd=ROOT,env=render_env,stdout=rollog,stderr=subprocess.STDOUT)
                        if completed.returncode: raise RuntimeError('Rollout failed '+cfg['id'])
                        atomic(wd/f'{stage}_{cfg["id"]}_done.json',dict(finished=time.time(),wall_s=time.perf_counter()-tick))
                atomic(wd/'finished.json',dict(time=time.time()))
            finally:
                server.terminate()
                try: server.wait(timeout=20)
                except subprocess.TimeoutExpired: server.kill(); server.wait()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,required=True); p.add_argument('--shard',type=int,required=True); a=p.parse_args()
    try: worker(a.gpu,a.shard)
    except Exception as exc:
        atomic(OUT/'workers'/f'gpu{a.gpu}'/'failed.json',dict(error=repr(exc),time=time.time()))
        raise

if __name__=='__main__': main()
