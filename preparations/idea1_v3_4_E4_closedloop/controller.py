import concurrent.futures,hashlib,json,os,subprocess,time,traceback
import numpy as np
from .common import ROOT,OUT
from .finish import progress
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc

MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'

def free(gpu):
    assert gpu in (0,2,3)
    apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    mem=float(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    return not apps and mem<=1500

def cross_gate(wd):
    with lock(OUT/'cross_service.lock'):
        reference=OUT/'cross_service_reference.npz'
        if not reference.exists():reference.write_bytes((wd/'gate_actions.npz').read_bytes())
        checks=[]
        with np.load(reference) as ref,np.load(wd/'gate_actions.npz') as got:
            for k in ref.files:checks.append(dict(key=k,max_abs=float(np.abs(ref[k]-got[k]).max()),passed=bool(np.allclose(ref[k],got[k],atol=1e-5,rtol=1e-5))))
        atomic(wd/'cross_service.json',dict(passed=all(r['passed'] for r in checks),checks=checks))
        if not all(r['passed'] for r in checks):raise RuntimeError('Cross-service comparison failed')

def worker(gpu):
    wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True);port=20890+gpu
    env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
    render,mapping=render_environment(env,gpu)
    identity=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip()
    atomic(wd/'identity.json',dict(identity=identity,egl_mapping=mapping,model_cuda_mask=env['CUDA_VISIBLE_DEVICES'],render_cuda_mask=render['CUDA_VISIBLE_DEVICES']))
    model=None;run=None
    try:
        with (wd/'service.log').open('a') as f:
            model=subprocess.Popen([str(MODEL),'-u','-m','preparations.idea1_v3_4_E4_closedloop.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            atomic(wd/'process.json',dict(pid=model.pid,port=port,time=time.time()))
            atomic(wd/'state.json',dict(status='loading',time=time.time()))
            begin=time.monotonic()
            while True:
                if model.poll() is not None:raise RuntimeError('Model exited during startup')
                if time.monotonic()-begin>900:raise TimeoutError('Model readiness')
                try:rpc(port,dict(op='health'));break
                except (OSError,ConnectionError):time.sleep(1)
            cross_gate(wd)
            atomic(wd/'state.json',dict(status='fixed_input_timing',load_and_gate_s=time.monotonic()-begin,time=time.time()))
            rpc(port,dict(op='benchmark'))
            atomic(wd/'state.json',dict(status='closedloop',time=time.time()))
            with (wd/'evaluate.log').open('a') as log:
                run=subprocess.Popen([str(EVAL),'-u','-m','preparations.idea1_v3_4_E4_closedloop.evaluate','--port',str(port)],cwd=ROOT,env=render,stdout=log,stderr=subprocess.STDOUT)
                code=run.wait()
            if code:raise RuntimeError(f'Evaluator exited {code}')
            atomic(wd/'state.json',dict(status='complete',time=time.time()))
    except Exception:
        atomic(wd/'error.json',dict(error=traceback.format_exc(),time=time.time()));raise
    finally:
        for p in (run,model):
            if p is not None and p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        atomic(wd/'exits.json',dict(model=None if model is None else model.poll(),evaluation=None if run is None else run.poll(),time=time.time()))

def main():
    with lock(OUT/'controller.lock',blocking=False):
        m=json.loads((OUT/'manifest.json').read_text())
        assert m['gpus']==[0,2,3]
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in m['source_files'].items())
        atomic(OUT/'process.json',dict(pid=os.getpid(),time=time.time()))
        pending=list(m['gpus']);futures={}
        with concurrent.futures.ThreadPoolExecutor(3) as ex:
            while pending or any(not f.done() for f in futures.values()):
                for gpu in list(pending):
                    if free(gpu):futures[gpu]=ex.submit(worker,gpu);pending.remove(gpu)
                atomic(OUT/'status.json',dict(status='running',progress=progress(),waiting_gpus=pending,
                    workers={str(g):json.loads((OUT/'workers'/f'gpu{g}'/'state.json').read_text()) if (OUT/'workers'/f'gpu{g}'/'state.json').exists() else {} for g in futures},time=time.time()))
                if any(f.done() and f.exception() for f in futures.values()):
                    for f in futures.values():
                        if f.done():f.result()
                time.sleep(15 if not pending else 30)
            for f in futures.values():f.result()
        from .finish import main as finish
        finish()

if __name__=='__main__':
    try:main()
    except Exception:
        atomic(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()))
        atomic(OUT/'status.json',dict(status='failed',progress=progress(),time=time.time()));raise
