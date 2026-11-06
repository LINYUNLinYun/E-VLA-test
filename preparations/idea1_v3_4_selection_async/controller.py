"""Single scheduler; claim only idle GPUs and preserve every valid episode."""
import json,os,socket,subprocess,time,traceback
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc
from .common import ROOT,OUT,MODULE,freeze,progress

MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'


def available():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    occupied={line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    free=[]
    for line in rows.splitlines():
        gpu,uuid,used=[x.strip() for x in line.split(',')]
        if uuid not in occupied and float(used)<=1500:free.append((int(gpu),uuid))
    atomic(OUT/'closed_loop_resource_scan.json',dict(time=time.time(),gpus=rows,apps=apps,free=free));return free


def start(gpu,uuid):
    lease=lock(Path('/tmp')/f'embodied-benchmark-gpu-{uuid}.lock',blocking=False)
    try:lease.__enter__()
    except BlockingIOError:return None
    if (gpu,uuid) not in available():lease.__exit__(None,None,None);return None
    wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True)
    env=environment(gpu);env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4')
    env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
    render,mapping=render_environment(env,gpu)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    atomic(wd/'identity.json',dict(gpu=gpu,gpu_uuid=uuid,port=port,egl_mapping=mapping,
        controller_pid=os.getpid(),time=time.time()))
    with (wd/'service.log').open('a') as stream:
        service=subprocess.Popen([str(MODEL),'-u','-m',MODULE+'.server','--port',str(port),'--output',str(wd)],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    atomic(wd/'service_process.json',dict(pid=service.pid,port=port,time=time.time()))
    return dict(gpu=gpu,uuid=uuid,port=port,lease=lease,wd=wd,env=env,render=render,
        service=service,phase='loading',phase_start=time.time(),gate=None,evaluation=None)


def cross_gate(worker):
    wd=worker['wd'];path=wd/'gate/gradients.npz';reference=OUT/'cross_service_reference.npz'
    with lock(OUT/'cross_service_reference.lock'):
        if not reference.exists():
            reference.write_bytes(path.read_bytes());atomic(OUT/'cross_service_reference.json',dict(worker=str(wd)))
    checks=[]
    with np.load(path) as got,np.load(reference) as ref:
        assert got.files==ref.files
        for key in ref.files:checks.append(dict(key=key,max_abs=float(np.max(np.abs(got[key]-ref[key]))),
            passed=bool(np.allclose(got[key],ref[key],atol=1e-4,rtol=1e-3))))
    atomic(wd/'cross_service_gate.json',dict(passed=all(c['passed'] for c in checks),checks=checks))
    assert all(c['passed'] for c in checks),'Cross-GPU selector gate failed'


def finish(worker):
    try:
        if worker['service'].poll() is None:
            try:rpc(worker['port'],dict(op='stop'))
            except (OSError,RuntimeError):worker['service'].terminate()
            worker['service'].wait(timeout=30)
    finally:
        for name in ('evaluation','gate','service'):
            process=worker[name]
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        worker['lease'].__exit__(None,None,None)
        atomic(worker['wd']/'exits.json',dict(time=time.time(),released=True))


def main():
    with lock(OUT/'closed_loop_controller.lock',blocking=False):
        atomic(OUT/'closed_loop_process.json',dict(pid=os.getpid(),started=time.time()))
        workers=[];last_scan=0.;frozen=False;completed=False
        try:
            while not completed:
                # A single scheduler checks newly released cards at ~30s cadence.
                if time.time()-last_scan>=30 and len(workers)<3:
                    for gpu,uuid in available():
                        if any(w['gpu']==gpu for w in workers):continue
                        worker=start(gpu,uuid)
                        if worker is not None:workers.append(worker)
                        if len(workers)>=3:break
                    last_scan=time.time()
                probes_ready=all((OUT/phase/'complete.json').exists() for phase in ('compression','overlap'))
                if probes_ready and not frozen:freeze();frozen=True
                for w in workers:
                    if w['service'].poll() is not None and w['phase']!='done':raise RuntimeError(f"Service {w['gpu']} exited")
                    if w['phase']=='loading':
                        try:rpc(w['port'],dict(op='health'))
                        except (OSError,ConnectionError):
                            if time.time()-w['phase_start']>600:raise TimeoutError('Service load')
                            continue
                        with (w['wd']/'gate.log').open('a') as stream:
                            w['gate']=subprocess.Popen([str(MODEL),'-u','-m',MODULE+'.gate','--port',str(w['port']),
                                '--output',str(w['wd']/'gate')],cwd=ROOT,env=w['env'],stdout=stream,stderr=subprocess.STDOUT)
                        w['phase']='gate';w['phase_start']=time.time()
                    elif w['phase']=='gate':
                        if w['gate'].poll() is None:
                            if time.time()-w['phase_start']>600:raise TimeoutError('Policy gate')
                            continue
                        if w['gate'].returncode:raise RuntimeError(f"Gate {w['gpu']} failed")
                        cross_gate(w);w['phase']='ready'
                    elif w['phase']=='ready' and frozen:
                        with (w['wd']/'evaluate.log').open('a') as stream:
                            w['evaluation']=subprocess.Popen([str(EVAL),'-u','-m',MODULE+'.evaluate','--port',str(w['port']),
                                '--worker',f"gpu{w['gpu']}"],cwd=ROOT,env=w['render'],stdout=stream,stderr=subprocess.STDOUT)
                        atomic(w['wd']/'evaluation_process.json',dict(pid=w['evaluation'].pid,time=time.time()))
                        w['phase']='evaluating'
                    elif w['phase']=='evaluating' and w['evaluation'].poll() is not None:
                        if w['evaluation'].returncode:raise RuntimeError(f"Evaluator {w['gpu']} failed")
                        w['phase']='done';finish(w)
                if frozen:
                    p=progress();completed=all(r['completed']==r['target'] for r in p.values())
                    atomic(OUT/'closed_loop_status.json',dict(progress=p,completed=sum(r['completed'] for r in p.values()),
                        target=sum(r['target'] for r in p.values()),workers=[dict(gpu=w['gpu'],phase=w['phase']) for w in workers],time=time.time()))
                    if workers and all(w['phase']=='done' for w in workers) and not completed:raise RuntimeError('Work queue ended before completion')
                time.sleep(3)
            atomic(OUT/'closed_loop_execution_complete.json',dict(time=time.time(),progress=progress()))
        finally:
            for w in workers:
                if w['phase']!='done':finish(w)


if __name__=='__main__':
    try:main()
    except Exception:atomic(OUT/'closed_loop_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
