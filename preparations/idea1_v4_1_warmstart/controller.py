"""One resource poller; lease only unused cards, then run bounded tuning stages."""
import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock,environment
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v4_0 import controller as native_controller
from .common import ROOT,OUT,MODULE,freeze,key,stage_progress
from .resource_watch import scan
from .select import replay_selection,champions

native_controller.OUT = OUT
native_controller.MODULE = MODULE
native_controller.scan = scan
MODEL,EVAL = native_controller.MODEL,native_controller.EVAL
LEASE_UUIDS={}


def pinned_environment(gpu):
    result=environment(gpu)
    result.update(CUDA_VISIBLE_DEVICES=LEASE_UUIDS[gpu],CUDA_DEVICE_ORDER='PCI_BUS_ID')
    return result


native_controller.environment=pinned_environment


def release_resource_failure(worker,stage):
    """Retire only this owned worker; CUDA OOM is not a policy failure."""
    native_controller.finish(worker)
    stamp=time.time_ns()
    archive=OUT/'attempts'/f'resource_conflict_{stamp}'/f"gpu{worker['gpu']}"
    archive.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(worker['wd']),str(archive))
    atomic(archive/'resource_failure.json',dict(gpu=worker['gpu'],uuid=worker['uuid'],stage=stage,time=time.time()))


def dispatch(worker,stage):
    module = '.replay' if stage=='replay' else '.evaluate'
    executable = MODEL if stage=='replay' else EVAL
    command = [str(executable),'-u','-m',MODULE+module,'--port',str(worker['port']),'--worker',f"gpu{worker['gpu']}"]
    if stage!='replay':command.extend(['--stage',stage])
    with (worker['wd']/f'{stage}.log').open('a') as stream:
        worker['evaluation'] = subprocess.Popen(command,cwd=ROOT,
            env=worker['env'] if stage=='replay' else worker['render'],stdout=stream,stderr=subprocess.STDOUT)
    worker['phase'] = stage
    worker['last_stage'] = stage
    atomic(worker['wd']/f'{stage}_process.json',dict(pid=worker['evaluation'].pid,time=time.time()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-gpus',type=int,default=4)
    args=parser.parse_args()
    manifest=freeze()
    workers,last_scan=[],0.
    cooldown={}
    stage='confirm' if (OUT/'champions.json').exists() else ('tune' if (OUT/'replay_selection.json').exists() else 'replay')
    with lock(OUT/'controller.lock',blocking=False):
        atomic(OUT/'controller_process.json',dict(pid=os.getpid(),started=time.time()))
        # Implementation-time watcher exits; this controller becomes sole poller.
        atomic(OUT/'stop_resource_watch.json',dict(time=time.time()))
        try:
            while True:
                if (OUT/'stop_controller.json').exists():return
                if time.time()-last_scan>=30 and len(workers)<args.max_gpus:
                    resource=scan()
                    for gpu,uuid in resource['free']:
                        if any(w['gpu']==gpu for w in workers):continue
                        if time.time()<cooldown.get(gpu,0.):continue
                        LEASE_UUIDS[gpu]=uuid
                        worker=native_controller.start(gpu,uuid)
                        if worker is not None:
                            worker['last_stage']=None
                            workers.append(worker)
                            print(f"Leased GPU {gpu}",flush=True)
                        if len(workers)>=args.max_gpus:break
                    last_scan=time.time()
                for worker in list(workers):
                    phase=worker['phase']
                    if worker['service'].poll() is not None:
                        raise RuntimeError(f"Service exited on GPU {worker['gpu']}")
                    if phase=='loading':
                        try:rpc(worker['port'],dict(op='health'))
                        except (OSError,ConnectionError):
                            if time.time()-worker['phase_start']>600:raise TimeoutError('Model load')
                            continue
                        with (worker['wd']/'preflight.log').open('a') as stream:
                            worker['gate']=subprocess.Popen([str(MODEL),'-u','-m',MODULE+'.preflight',
                                '--port',str(worker['port'])],cwd=ROOT,env=worker['env'],stdout=stream,stderr=subprocess.STDOUT)
                        worker['phase']='preflight'
                        worker['phase_start']=time.time()
                    elif phase=='preflight':
                        if worker['gate'].poll() is None:
                            if time.time()-worker['phase_start']>600:raise TimeoutError('Preflight')
                            continue
                        if worker['gate'].returncode:
                            failure=(worker['wd']/'preflight.log').read_text()
                            if 'CUDA out of memory' in failure:
                                print(f"GPU {worker['gpu']} preflight hit a memory conflict; retaining other workers",flush=True)
                                release_resource_failure(worker,'preflight')
                                cooldown[worker['gpu']]=time.time()+120
                                workers.remove(worker)
                                continue
                            raise RuntimeError(f"Preflight failed: {worker['wd']}")
                        worker['phase']='idle'
                    elif phase in ('replay','tune','confirm') and worker['evaluation'].poll() is not None:
                        if worker['evaluation'].returncode:
                            raise RuntimeError(f"{phase} failed: {worker['wd']}")
                        worker['phase']='idle'
                    if worker['phase']=='idle' and worker['last_stage']!=stage:
                        dispatch(worker,stage)
                if stage=='replay':
                    completed=sum((OUT/'replay'/key(b['case'])/'complete.json').exists() for b in manifest['replay_bank'])
                    status=dict(completed=completed,target=len(manifest['replay_bank']))
                else:status=stage_progress(stage,manifest)
                atomic(OUT/'status.json',dict(status='running' if workers else 'waiting_gpu',stage=stage,
                    progress=status,workers=[dict(gpu=w['gpu'],phase=w['phase']) for w in workers],time=time.time()))
                if status['completed']==status['target'] and all(w['phase'] not in ('replay','tune','confirm') for w in workers):
                    print(json.dumps(dict(stage=stage,complete=True,time=time.time())),flush=True)
                    if stage=='replay':
                        print(json.dumps(replay_selection()['candidates']),flush=True)
                        stage='tune'
                    elif stage=='tune':
                        print(json.dumps(champions()['candidates']),flush=True)
                        stage='confirm'
                    else:
                        atomic(OUT/'execution_complete.json',dict(valid=True,time=time.time()))
                        break
                time.sleep(3)
        finally:
            for worker in workers:native_controller.finish(worker)
    from .finish import finish
    finish()


if __name__=='__main__':
    try:main()
    except Exception:
        atomic(OUT/'failure.json',dict(error=traceback.format_exc(),time=time.time()))
        raise
