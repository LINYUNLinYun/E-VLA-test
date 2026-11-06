import concurrent.futures,json,os,subprocess,time,traceback
from .common import ROOT,OUT
from .finish import progress
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc

MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'

def free(gpu):
    apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    mem=float(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    return not apps and mem<=1500

def worker(gpu):
    wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True);port=20760+gpu
    for attempt in range(2):
        while not free(gpu):
            atomic(wd/'state.json',dict(status='waiting_gpu',time=time.time()));time.sleep(30)
        env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        render,mapping=render_environment(env,gpu)
        identity=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip()
        atomic(wd/'identity.json',dict(identity=identity,egl_mapping=mapping))
        model=None;run=None
        try:
            with (wd/f'service_attempt{attempt+1}.log').open('w') as f:
                model=subprocess.Popen([str(MODEL),'-u','-m','preparations.idea1_v3_4_E2_closedloop.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                atomic(wd/'process.json',dict(pid=model.pid,port=port,time=time.time()))
                atomic(wd/'state.json',dict(status='loading',attempt=attempt+1,time=time.time()))
                begin=time.monotonic()
                while True:
                    if model.poll() is not None:raise RuntimeError('Model exited during startup')
                    if time.monotonic()-begin>600:raise TimeoutError('Model readiness')
                    try:rpc(port,dict(op='health'));break
                    except (OSError,ConnectionError):time.sleep(1)
                atomic(wd/'state.json',dict(status='evaluating',load_and_gate_s=time.monotonic()-begin,time=time.time()))
                with (wd/f'evaluate_attempt{attempt+1}.log').open('w') as log:
                    run=subprocess.Popen([str(EVAL),'-u','-m','preparations.idea1_v3_4_E2_closedloop.evaluate','--port',str(port),'--shard',str(gpu)],cwd=ROOT,env=render,stdout=log,stderr=subprocess.STDOUT)
                    code=run.wait()
                if code:raise RuntimeError(f'Evaluator exited {code}')
                atomic(wd/'state.json',dict(status='complete',time=time.time()));return
        except Exception:
            atomic(wd/f'error_attempt{attempt+1}.json',dict(error=traceback.format_exc(),time=time.time()))
            if attempt:raise
        finally:
            for p in (run,model):
                if p is not None and p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
            atomic(wd/'exits.json',dict(model=None if model is None else model.poll(),evaluation=None if run is None else run.poll(),time=time.time()))

def main():
    with lock(OUT/'controller.lock',blocking=False):
        atomic(OUT/'process.json',dict(pid=os.getpid(),time=time.time()))
        with concurrent.futures.ThreadPoolExecutor(4) as ex:
            futures=[ex.submit(worker,g) for g in range(4)]
            while not all(f.done() for f in futures):
                atomic(OUT/'status.json',dict(status='running',progress=progress(),workers={str(g):json.loads((OUT/'workers'/f'gpu{g}'/'state.json').read_text()) if (OUT/'workers'/f'gpu{g}'/'state.json').exists() else {} for g in range(4)},time=time.time()))
                time.sleep(5)
            for f in futures:f.result()
        from .finish import main as finish
        finish()

if __name__=='__main__':
    try:main()
    except Exception:
        atomic(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()))
        atomic(OUT/'status.json',dict(status='failed',progress=progress(),time=time.time()))
        raise
