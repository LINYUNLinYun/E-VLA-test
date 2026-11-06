"""Default dry run; static task ownership, model resident worker and bounded P0."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

from preparations.idea1_v3.io import atomic, append, digest

ROOT=Path(__file__).resolve().parents[2]
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
MODEL_PYTHON=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL_PYTHON=ROOT/'policies/libero-eval-v21/.venv/bin/python'


def configs(stage):
    if stage=='calibration':
        return [dict(experiment='E1',draft_steps=d,tau=t,period=p) for d in (2,10)
                for t in (.10,.15,.25) for p in (2,4)]
    if stage=='E4': return [dict(experiment='E4')]
    return [dict(experiment='E1',draft_steps=2,tau=.15,period=2)]


def config_id(config):
    if config['experiment']=='E4': return 'E4_pure_learned'
    return f"{config['experiment']}_D{config['draft_steps']}_tau{config['tau']}_P{config['period']}"


def make_manifest(a):
    config_list=configs(a.stage)
    if a.config: config_list=[json.loads(a.config)]
    for c in config_list:
        if c['experiment'] not in ('E1','E4'): raise ValueError('This initial CLI implements only E1/E4')
    provenance=ROOT/'policies/realtime-vla-flash/results/experiment11_layer_skip_prefix_20261010/provenance/preflight.json'
    checkpoint=json.loads(provenance.read_text())['checkpoint']
    draft_names=dict(zip(SUITES,('draft_spatial','draft_object','draft','draft_10')))
    checkpoints={suite:dict(base_path=checkpoint['path'],base_sha256=checkpoint['sha256'],
        draft_path=str(ROOT/'shared/checkpoints/realtime-vla-flash/converted'/name/'draft_triton.pkl'),
        draft_bytes=(ROOT/'shared/checkpoints/realtime-vla-flash/converted'/name/'draft_triton.pkl').stat().st_size,
        draft_manifest_sha256=digest(json.loads((ROOT/'shared/checkpoints/realtime-vla-flash/converted'/name/'manifest.json').read_text())))
        for suite,name in draft_names.items()}
    conditions=[dict(id=config_id(c),config=c,scientific_hash=digest(dict(config=c,checkpoints=checkpoints))) for c in config_list]
    tasks=[]
    for condition in conditions:
        variants=('A','B') if a.stage=='P0' else (a.execution,)
        for execution in variants:
            for init in ([0] if a.stage in ('P0','calibration') else ([6] if not a.expand else range(6,11))):
                for suite in SUITES:
                    for task in ([0] if a.stage=='P0' else ([0,5] if a.stage=='calibration' else range(10))):
                        identity=dict(experiment=condition['config']['experiment'],scientific_hash=condition['scientific_hash'],
                            checkpoint=checkpoints[suite],suite=suite,task=task,init=init,
                            seed_rule='idea1-v22/1701 suite/task/init/replan',stage=a.stage)
                        if a.stage=='P0': identity['engineering_execution']=execution
                        key=digest(identity)
                        tasks.append(dict(identity,key=key,condition=condition['id'],execution=execution,
                                          config=condition['config'],owner=len(tasks)%a.shards))
    return dict(schema=1,stage=a.stage,created=time.time(),shards=a.shards,tasks=tasks,conditions=conditions,
        p0_budget_seconds=3600 if a.stage=='P0' else None,
        engineering_candidates=dict(A='consolidated compressed episode NPZ',B='consolidated uncompressed episode NPZ'),
        scientific_protocol=dict(H=50,Rmax=5,Full_D=10,V14_skip=[3,4,9,11],K=2,t=[.1,.05]),
        limitations=['E2 and E3 are separate pending branches; no pi0 baseline tasks scheduled'])


def verify_gpu(gpu):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    devices={int(parts[0]):dict(uuid=parts[1].strip(),memory_mb=float(parts[2]))
             for parts in (line.split(',') for line in output.splitlines())}
    if gpu not in devices: raise ValueError('physical GPU index absent')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(line.split(',')[0].strip()==devices[gpu]['uuid'] for line in apps.splitlines()):
        raise RuntimeError('GPU has another compute owner; no process will be stopped')
    if devices[gpu]['memory_mb']>1500: raise RuntimeError('GPU memory indicates another owner')
    return devices[gpu]


def worker(a):
    from preparations.idea1_v2_1.runtime_client import rpc
    from preparations.idea1_v3.episode import run
    manifest=json.loads((a.attempt/'manifest.json').read_text())
    if not 0<=a.shard<manifest['shards']: raise ValueError('shard outside static ownership')
    wd=a.attempt/'workers'/f'worker{a.shard}'; wd.mkdir(parents=True,exist_ok=True)
    with (wd/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=verify_gpu(a.gpu)
        atomic(wd/'identity.json',dict(pid=os.getpid(),physical_gpu=a.gpu,**identity,
            inference_binding=os.environ.get('CUDA_VISIBLE_DEVICES'),egl_binding=os.environ.get('MUJOCO_EGL_DEVICE_ID')))
        todo=[row for row in manifest['tasks'] if row['owner']==a.shard]
        # A and B each complete the same four suite keys before the next config.
        config=None; server=None; log=None; port=None
        try:
            for row in todo:
                if (a.attempt/'STOP').exists(): break
                if shutil.disk_usage(a.attempt).free<2*1024**3: raise RuntimeError('Less than 2 GiB free; stop dispatch')
                if manifest['stage']=='P0':
                    budget=a.attempt/'profiling'/'budget.json'
                    if budget.exists() and time.monotonic()-json.loads(budget.read_text())['monotonic_start']>=3600: break
                directory=a.attempt/'episodes'/row['experiment']/row['condition']/row['execution']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"
                if (directory/'complete.json').exists():
                    complete=json.loads((directory/'complete.json').read_text())
                    if complete['key']!=row['key']: raise ValueError('completed key mismatch')
                    continue
                if directory.exists():
                    # Interrupted attempt preserved; never overwrite a valid failure.
                    suffix=1
                    while directory.with_name(directory.name+f'.interrupted{suffix}').exists(): suffix+=1
                    directory.rename(directory.with_name(directory.name+f'.interrupted{suffix}'))
                if config!=row['config']:
                    if server:
                        server.terminate(); server.wait(timeout=30); log.close()
                    with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
                    log=(wd/f'server_{time.time_ns()}.log').open('a')
                    tick=time.perf_counter()
                    server=subprocess.Popen([str(MODEL_PYTHON),'-m','preparations.idea1_v3.server','--port',str(port),
                        '--config',json.dumps(row['config']),'--launch'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                    deadline=time.monotonic()+1200
                    while True:
                        if server.poll() is not None: raise RuntimeError('model server exited; inspect worker server log')
                        try: rpc(port,dict(op='health')); break
                        except (OSError,RuntimeError):
                            if time.monotonic()>deadline: raise TimeoutError('model startup')
                            time.sleep(1)
                    append(wd/'events.jsonl',dict(kind='startup',seconds=time.perf_counter()-tick,config=row['config']))
                    config=row['config']
                    if manifest['stage']=='P0' and not (a.attempt/'profiling'/'budget.json').exists():
                        atomic(a.attempt/'profiling'/'budget.json',dict(monotonic_start=time.monotonic(),wall_start=time.time()))
                profile=a.attempt/'profiling'/f'profile_worker{a.shard}.jsonl'
                append(wd/'events.jsonl',dict(kind='episode_start',key=row['key']))
                try:
                    result=run(directory,port,row,row['config'],profile,collect_inputs=manifest['stage']=='P0')
                    append(wd/'events.jsonl',dict(kind='episode_complete',key=row['key'],success=result['success']))
                except Exception:
                    append(wd/'events.jsonl',dict(kind='invalid',key=row['key'],error=traceback.format_exc()))
                    raise
        finally:
            if server:
                server.terminate()
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired: server.kill(); server.wait()
            if log: log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','worker','stop','resume','summary','micro'))
    p.add_argument('--attempt',type=Path,required=True); p.add_argument('--launch',action='store_true')
    p.add_argument('--stage',choices=('P0','calibration','E1','E4'),default='P0')
    p.add_argument('--config'); p.add_argument('--expand',action='store_true'); p.add_argument('--execution',choices=('A','B'),default='A')
    p.add_argument('--shards',type=int,default=1); p.add_argument('--shard',type=int,default=0); p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args(); a.attempt=a.attempt.resolve()
    if not a.launch:
        result=make_manifest(a) if a.command=='prepare' else dict(command=a.command,attempt=str(a.attempt),dry_run=True)
        print(json.dumps(result,indent=2)); return
    if a.command=='prepare':
        if (a.attempt/'manifest.json').exists(): raise FileExistsError('manifest exists; use worker to resume')
        if a.stage=='P0' and a.shards!=1: raise ValueError('P0 A/B uses one serial worker')
        a.attempt.mkdir(parents=True,exist_ok=True); atomic(a.attempt/'manifest.json',make_manifest(a))
    elif a.command=='stop': atomic(a.attempt/'STOP',dict(reason='user stop dispatch',time=time.time()))
    elif a.command=='resume':
        if (a.attempt/'STOP').exists(): (a.attempt/'STOP').unlink()
    elif a.command=='summary':
        from preparations.idea1_v3.profile import summarize
        summarize(a.attempt)
    elif a.command=='micro':
        from preparations.idea1_v3.profile import micro
        micro(a.attempt)
    else:
        from preparations.idea1_v2_2.common import evaluation_env
        env=evaluation_env(a.gpu); env['PYTHONPATH']=str(ROOT)+':'+env['PYTHONPATH']
        if Path(sys.executable).resolve()!=EVAL_PYTHON.resolve() or os.environ.get('V3_WORKER_CHILD')!='1':
            env['V3_WORKER_CHILD']='1'
            subprocess.run([str(EVAL_PYTHON),'-m','preparations.idea1_v3.cli',*sys.argv[1:]],cwd=ROOT,env=env,check=True)
        else: worker(a)

if __name__=='__main__': main()
