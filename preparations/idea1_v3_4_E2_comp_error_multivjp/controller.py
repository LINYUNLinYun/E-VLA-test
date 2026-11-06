import concurrent.futures,json,os,subprocess,time,traceback
from .common import ROOT,OUT,pending,release
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc
from .finish import progress

MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'
def available():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={r.split(',')[0].strip() for r in apps.splitlines() if r.strip()}
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    return [int(g) for g,u,m in ([x.strip() for x in r.split(',')] for r in rows.splitlines()) if u not in busy and float(m)<=1500]
def worker(gpu):
    wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True);port=25410+gpu
    for attempt in range(2):
        env=environment(gpu);env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4');env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        render,mapping=render_environment(env,gpu);atomic(wd/'identity.json',dict(gpu=gpu,egl_mapping=mapping));service=None;run=None
        try:
            with (wd/f'service_attempt{attempt+1}.log').open('w') as log:
                tick=time.monotonic();service=subprocess.Popen([str(MODEL),'-u','-m','preparations.idea1_v3_4_E2_comp_error_multivjp.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic(wd/'process.json',dict(pid=service.pid,gpu=gpu,port=port,time=time.time()));atomic(wd/'state.json',dict(status='loading',time=time.time()))
                while True:
                    if service.poll() is not None:raise RuntimeError('Service startup failed')
                    if time.monotonic()-tick>600:raise TimeoutError('Service readiness')
                    try:rpc(port,dict(op='health'));break
                    except (OSError,ConnectionError):time.sleep(1)
                atomic(wd/'state.json',dict(status='evaluating',startup_s=time.monotonic()-tick,time=time.time()))
                with (wd/f'evaluate_attempt{attempt+1}.log').open('w') as elog:
                    run=subprocess.Popen([str(EVAL),'-u','-m','preparations.idea1_v3_4_E2_comp_error_multivjp.evaluate','--port',str(port),'--gpu',str(gpu)],cwd=ROOT,env=render,stdout=elog,stderr=subprocess.STDOUT)
                    if run.wait():raise RuntimeError('Evaluator failed')
            atomic(wd/'state.json',dict(status='complete',time=time.time()));return
        except Exception:
            atomic(wd/f'error_attempt{attempt+1}.json',dict(error=traceback.format_exc(),time=time.time()))
            if attempt:raise
        finally:
            for process in [run,service]:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            release(gpu);atomic(wd/'exits.json',dict(service=None if service is None else service.poll(),evaluation=None if run is None else run.poll(),time=time.time()))
        while pending() and gpu not in available():time.sleep(30)
        if not pending():return
def main():
    with lock(OUT/'controller.lock',blocking=False):
        atomic(OUT/'process.json',dict(pid=os.getpid(),time=time.time()));last_scan=-1e9;active={}
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            while pending() or active:
                for gpu,future in list(active.items()):
                    if future.done():future.result();del active[gpu]
                if pending() and len(active)<4 and time.monotonic()-last_scan>=30:
                    last_scan=time.monotonic()
                    for gpu in available():
                        if gpu not in active:active[gpu]=pool.submit(worker,gpu)
                atomic(OUT/'status.json',dict(status='running' if active else 'waiting_gpu',progress=progress(),pending_cases=pending(),active_gpus=list(active),time=time.time()));time.sleep(5)
        from .finish import main as finish
        finish()
if __name__=='__main__':
    try:main()
    except Exception:
        atomic(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
