"""Wait for and lease exactly one unoccupied GPU; release it when finished."""
import argparse
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path
from .common import ROOT,OUT,MODULE,freeze,verify_frozen,progress
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc

MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'


def available():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'],text=True)
    occupied={row.split(',')[0].strip() for row in apps.splitlines() if row.strip()}
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'],text=True)
    result=[]
    for row in rows.splitlines():
        gpu,uuid,used=[v.strip() for v in row.split(',')]
        if uuid not in occupied and float(used)<=1500:result.append((int(gpu),uuid))
    atomic(OUT/'resource_scan.json',dict(time=time.time(),gpus=rows,apps=apps,free=result))
    return result


def main():
    with lock(OUT/'controller.lock',blocking=False):
        manifest=freeze();verify_frozen(manifest)
        atomic(OUT/'process.json',dict(pid=os.getpid(),started=time.time()))
        chosen=None;lease=None
        while chosen is None:
            for gpu,uuid in available():
                candidate=lock(Path('/tmp')/f'embodied-benchmark-gpu-{uuid}.lock',blocking=False)
                try:candidate.__enter__()
                except BlockingIOError:continue
                if (gpu,uuid) not in available():candidate.__exit__(None,None,None);continue
                lease=candidate;chosen=(gpu,uuid);break
            if chosen is None:
                atomic(OUT/'status.json',dict(status='waiting_gpu',completed=0,target=128,time=time.time()))
                time.sleep(30)
        gpu,uuid=chosen;worker=OUT/'worker';worker.mkdir(parents=True,exist_ok=True)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        env=environment(gpu);env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4')
        env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        render,mapping=render_environment(env,gpu)
        atomic(worker/'identity.json',dict(gpu=gpu,gpu_uuid=uuid,egl_mapping=mapping,
            port=port,lease_owner=os.getpid(),time=time.time()))
        service=evaluation=None;start=time.time()
        try:
            with (worker/'service.log').open('a') as stream:
                service=subprocess.Popen([str(MODEL),'-u','-m',MODULE+'.server','--port',str(port),
                    '--output',str(worker)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            atomic(worker/'process.json',dict(pid=service.pid,port=port,time=time.time()))
            atomic(OUT/'status.json',dict(status='loading_and_benchmarking',gpu=gpu,completed=0,target=128,time=time.time()))
            while True:
                if service.poll() is not None:raise RuntimeError(f'Service exited {service.returncode}; see {worker}/service.log')
                if time.time()-start>2400:raise TimeoutError('Bounded 40-minute service/benchmark startup')
                try:rpc(port,dict(op='health'));break
                except (OSError,ConnectionError):time.sleep(2)
            eval_start=time.time()
            with (worker/'evaluate.log').open('a') as stream:
                evaluation=subprocess.Popen([str(EVAL),'-u','-m',MODULE+'.evaluate','--port',str(port)],
                    cwd=ROOT,env=render,stdout=stream,stderr=subprocess.STDOUT)
            atomic(worker/'evaluation_process.json',dict(pid=evaluation.pid,time=time.time()))
            while evaluation.poll() is None:time.sleep(10)
            if evaluation.returncode:raise RuntimeError(f'Evaluator exited {evaluation.returncode}; see {worker}/evaluate.log')
            atomic(OUT/'execution.json',dict(gpu=gpu,gpu_uuid=uuid,service_and_benchmark_start=start,
                evaluation_start=eval_start,evaluation_finish=time.time(),time=time.time()))
            rpc(port,dict(op='stop'));service.wait(timeout=30)
        finally:
            for process in (evaluation,service):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            lease.__exit__(None,None,None)
            atomic(worker/'exits.json',dict(service=None if service is None else service.poll(),
                evaluation=None if evaluation is None else evaluation.poll(),released=time.time()))
        from .finish import main as finish
        finish()


if __name__=='__main__':
    try:main()
    except Exception:
        error=dict(error=traceback.format_exc(),time=time.time())
        atomic(OUT/'failed.json',error);print(error['error'],flush=True);raise
