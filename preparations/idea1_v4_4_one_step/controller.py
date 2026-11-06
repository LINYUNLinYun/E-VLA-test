"""One poller, stable idle cards only, paired conditions within each GPU service."""
import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment, lock
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v4_0 import controller as native
from .common import ROOT, OUT, MODULE, freeze, progress, sha

UUIDS = {}


def scan():
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    occupied = {line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    free = []
    for line in raw.splitlines():
        gpu, identity, used = [s.strip() for s in line.split(',')]
        if identity not in occupied and float(used) <= 1500:
            free.append((int(gpu), identity))
    atomic(OUT/'resource_scan.json',dict(free=free,apps=apps,time=time.time()))
    return dict(free=free)


def pinned_environment(gpu):
    env = environment(gpu)
    env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu], CUDA_DEVICE_ORDER='PCI_BUS_ID')
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-gpus',type=int,default=4)
    args = parser.parse_args()
    manifest = freeze()
    assert all(sha(ROOT/name)==expected for name,expected in manifest['source_files'].items())
    native.OUT, native.MODULE = OUT, MODULE
    native.scan, native.environment = scan, pinned_environment
    workers, last_scan, idle_since, retries, cooldown = [], 0., {}, {}, {}
    with lock(OUT/'controller.lock',blocking=False):
        atomic(OUT/'controller_process.json',dict(pid=os.getpid(),started=time.time()))
        try:
            while True:
                counts = progress(manifest)
                completed = sum(c['completed'] for c in counts.values())
                if time.time()-last_scan >= 30 and len(workers)<args.max_gpus and completed<manifest['target_episodes']:
                    now = time.time()
                    free = scan()['free']
                    identities = {identity for gpu,identity in free}
                    idle_since = {k:v for k,v in idle_since.items() if k in identities}
                    for gpu, identity in free:
                        if any(w['gpu']==gpu for w in workers) or retries.get(gpu,0)>=2:
                            continue
                        idle_since.setdefault(identity, now)
                        if now-idle_since[identity]<60 or now<cooldown.get(gpu,0):
                            continue
                        UUIDS[gpu] = identity
                        worker = native.start(gpu,identity)
                        if worker is not None:
                            workers.append(worker)
                            print(f'Leased GPU {gpu} ({identity})',flush=True)
                        if len(workers)==args.max_gpus:
                            break
                    last_scan = now
                for worker in list(workers):
                    if worker['phase']=='done':
                        continue
                    if worker['service'].poll() is not None:
                        failure = (worker['wd']/'service.log').read_text()
                        if 'out of memory' in failure.lower():
                            gpu = worker['gpu']
                            native.finish(worker)
                            archive = OUT/'attempts'/f'resource_conflict_{time.time_ns()}'/worker['wd'].name
                            archive.parent.mkdir(parents=True,exist_ok=True)
                            shutil.move(str(worker['wd']),str(archive))
                            retries[gpu] = retries.get(gpu,0)+1
                            cooldown[gpu] = time.time()+120
                            idle_since.pop(worker['uuid'],None)
                            workers.remove(worker)
                            print(f'GPU {gpu} resource conflict; retained healthy workers',flush=True)
                            continue
                        raise RuntimeError(f"Service failed: {worker['wd']}")
                    if worker['phase']=='loading':
                        try:
                            rpc(worker['port'],dict(op='health'))
                        except (OSError,ConnectionError):
                            if time.time()-worker['phase_start']>600:
                                raise TimeoutError('Load and preflight')
                            continue
                        with lock(OUT/'cross_service.lock'):
                            reference = OUT/'cross_service_reference.npz'
                            if not reference.exists():
                                reference.write_bytes((worker['wd']/'gate_actions.npz').read_bytes())
                            with np.load(reference) as a, np.load(worker['wd']/'gate_actions.npz') as b:
                                checks = [dict(key=k,passed=bool(np.allclose(a[k],b[k],atol=1e-5,rtol=1e-5)),max_abs=float(np.abs(a[k]-b[k]).max())) for k in a.files]
                            atomic(worker['wd']/'cross_service.json',dict(passed=all(c['passed'] for c in checks),checks=checks))
                            assert all(c['passed'] for c in checks), 'Cross-service numerical mismatch'
                        with (worker['wd']/'evaluate.log').open('a') as stream:
                            worker['evaluation'] = subprocess.Popen([str(native.EVAL),'-u','-m',MODULE+'.evaluate','--port',str(worker['port']),'--worker',worker['wd'].name],cwd=ROOT,env=worker['render'],stdout=stream,stderr=subprocess.STDOUT)
                        atomic(worker['wd']/'evaluation_process.json',dict(pid=worker['evaluation'].pid,time=time.time()))
                        worker['phase']='evaluating'
                    elif worker['phase']=='evaluating' and worker['evaluation'].poll() is not None:
                        if worker['evaluation'].returncode:
                            raise RuntimeError(f"Evaluation failed: {worker['wd']}")
                        native.finish(worker)
                        worker['phase']='done'
                counts = progress(manifest)
                completed = sum(c['completed'] for c in counts.values())
                atomic(OUT/'status.json',dict(status='running' if workers else 'waiting_gpu',completed=completed,target=manifest['target_episodes'],progress=counts,workers=[dict(gpu=w['gpu'],phase=w['phase']) for w in workers],time=time.time()))
                if completed==manifest['target_episodes'] and all(w['phase']=='done' for w in workers):
                    atomic(OUT/'execution_complete.json',dict(time=time.time(),progress=counts))
                    break
                if workers and all(w['phase']=='done' for w in workers):
                    raise RuntimeError('Evaluators ended before all cases completed')
                time.sleep(3)
        finally:
            for worker in workers:
                if worker['phase']!='done':
                    native.finish(worker)
    from .finish import finish
    finish()


if __name__=='__main__':
    try:
        main()
    except Exception:
        atomic(OUT/'failure.json',dict(error=traceback.format_exc(),time=time.time()))
        raise
