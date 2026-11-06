import concurrent.futures,json,os,subprocess,time,traceback
import numpy as np
from .common import ROOT,OUT,MODEL,EVAL,sha
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3_4_v12_compensation_200.controller import free

def worker(gpu):
    wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True);port=21410+gpu
    env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH'];render,mapping=render_environment(env,gpu)
    atomic(wd/'identity.json',dict(gpu=gpu,egl_mapping=mapping,identity=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip()))
    model=None;run=None
    try:
        with (wd/'service.log').open('a') as f:
            model=subprocess.Popen([str(MODEL),'-u','-m','preparations.idea1_v3_4_v12_refresh_sweep.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            atomic(wd/'process.json',dict(pid=model.pid,port=port,time=time.time()));atomic(wd/'state.json',dict(status='loading',time=time.time()));start=time.monotonic()
            while True:
                if model.poll() is not None:raise RuntimeError('Service exited')
                if time.monotonic()-start>600:raise TimeoutError('Readiness')
                try:rpc(port,dict(op='health'));break
                except (OSError,ConnectionError):time.sleep(1)
            with lock(OUT/'cross.lock'):
                ref=OUT/'gate_reference.npz'
                if not ref.exists():ref.write_bytes((wd/'gate_actions.npz').read_bytes())
                with np.load(ref) as a,np.load(wd/'gate_actions.npz') as b:passed=all(np.allclose(a[k],b[k],atol=1e-5,rtol=1e-5) for k in a.files)
                atomic(wd/'cross_service.json',dict(passed=passed));assert passed
            atomic(wd/'state.json',dict(status='closedloop',time=time.time()))
            with (wd/'evaluate.log').open('a') as log:
                run=subprocess.Popen([str(EVAL),'-u','-m','preparations.idea1_v3_4_v12_refresh_sweep.evaluate','--port',str(port)],cwd=ROOT,env=render,stdout=log,stderr=subprocess.STDOUT)
                code=run.wait()
            if code:raise RuntimeError('Evaluator exited '+str(code))
            atomic(wd/'state.json',dict(status='complete',time=time.time()))
    except Exception:
        atomic(wd/'error.json',dict(error=traceback.format_exc(),time=time.time()));raise
    finally:
        for p in (run,model):
            if p is not None and p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()

def main():
    from .finish import progress,main as finish
    with lock(OUT/'controller.lock',blocking=False):
        m=json.loads((OUT/'manifest.json').read_text());assert all(sha(ROOT/p)==h for p,h in m['source_files'].items())
        atomic(OUT/'process.json',dict(pid=os.getpid(),time=time.time()));pending=list(m['gpus']);futures={}
        with concurrent.futures.ThreadPoolExecutor(3) as ex:
            while pending or any(not f.done() for f in futures.values()):
                if len(list((OUT/'case_done').glob('*.json')))==40:pending=[]
                for gpu in list(pending):
                    if free(gpu):futures[gpu]=ex.submit(worker,gpu);pending.remove(gpu)
                atomic(OUT/'status.json',dict(status='running',progress=progress(),workers={str(g):json.loads((OUT/'workers'/f'gpu{g}'/'state.json').read_text()) if (OUT/'workers'/f'gpu{g}'/'state.json').exists() else {} for g in futures},time=time.time()))
                for f in futures.values():
                    if f.done():f.result()
                time.sleep(15 if not pending else 30)
        finish()
if __name__=='__main__':
    try:main()
    except Exception:
        atomic(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
