"""Four authorized devices, persistent model services and atomic condition claims."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from preparations.idea1_v3.io import atomic,append
from preparations.idea1_v3.cli import verify_gpu
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v2_1.runtime_client import rpc

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_3'
PY=dict(pi0=ROOT/'policies/realtime-vla-flash/.venv/bin/python',xiaomi=ROOT/'policies/xiaomi-robotic0/.venv/bin/python',env=ROOT/'policies/libero-eval-v21/.venv/bin/python')


def worker(gpu,model):
    wd=OUT/'workers'/f'gpu{gpu}'; wd.mkdir(parents=True,exist_ok=True)
    with (wd/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=verify_gpu(gpu); atomic(wd/'identity.json',dict(identity,gpu=gpu,pid=os.getpid()))
        port=9340+gpu; env=environment(gpu)
        if model=='pi0': env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        # Each model sees its assigned device as local CUDA0. EGL uses the
        # physical device; simulation runs in a separate no-CUDA context.
        serverlog=(wd/'service.log').open('a')
        server=subprocess.Popen([str(PY[model]),'-m','preparations.idea1_v3_3.server','--model',model,'--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=serverlog,stderr=subprocess.STDOUT)
        try:
            start=time.perf_counter(); health=None
            while time.perf_counter()-start<1200:
                if server.poll() is not None: raise RuntimeError('Service exited; inspect '+str(wd/'service.log'))
                try: health=rpc(port,dict(op='health')); break
                except (ConnectionError,OSError): time.sleep(1)
            if health is None: raise TimeoutError('Readiness')
            atomic(wd/'loaded_protocol.json',dict(health,load_wall_s=time.perf_counter()-start))
            probed=False
            while True:
                m=json.loads((OUT/'manifest.json').read_text())
                rows=[c for c in m['conditions'] if c['model']==model]
                claimed=None; handle=None
                for cfg in rows:
                    state=OUT/'conditions'/cfg['id']; state.mkdir(parents=True,exist_ok=True)
                    if (state/'done.json').exists() or (state/'failed.json').exists(): continue
                    candidate=(state/'owner.lock').open('a')
                    try: fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError: candidate.close(); continue
                    if (state/'done.json').exists(): candidate.close(); continue
                    claimed=cfg; handle=candidate; break
                if claimed is None: break
                state=OUT/'conditions'/claimed['id']
                atomic(state/'running.json',dict(gpu=gpu,pid=os.getpid(),config=claimed,started=time.time()))
                try:
                    command=[str(PY['env']),'-m','preparations.idea1_v3_3.rollout','--manifest',str(OUT/'manifest.json'),'--condition',claimed['id'],'--output',str(OUT),'--port',str(port)]
                    if not probed: command+=['--probe']
                    tick=time.perf_counter()
                    with (state/'rollout.log').open('a') as log:
                        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    if completed.returncode: raise RuntimeError('Rollout exit '+str(completed.returncode))
                    probed=True
                    atomic(state/'done.json',dict(gpu=gpu,wall_s=time.perf_counter()-tick,finished=time.time(),config=claimed))
                except Exception as exc:
                    atomic(state/'failed.json',dict(gpu=gpu,error=repr(exc),time=time.time()))
                    # A native/kernel/protocol failure stops this worker; it
                    # must not silently poison every remaining condition.
                    raise
                finally: handle.close()
            atomic(wd/'finished.json',dict(time=time.time()))
        finally:
            server.terminate()
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired: server.kill(); server.wait()
            serverlog.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,required=True); p.add_argument('--model',choices=['pi0','xiaomi'],required=True); a=p.parse_args(); worker(a.gpu,a.model)


if __name__=='__main__': main()
