import concurrent.futures,json,os,subprocess,time,traceback
from .common import ROOT,OLD,OUT,MODEL,EVAL,SUITES,sha
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc

def free(gpu):
    assert gpu in (0,2,3)
    apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    mem=float(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    return not apps and mem<=1500

def worker(gpu):
    wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True);port=21120+gpu
    env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
    render,mapping=render_environment(env,gpu);model=None;runs=[]
    identity=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip()
    atomic(wd/'identity.json',dict(identity=identity,egl_mapping=mapping,model_cuda_mask=env['CUDA_VISIBLE_DEVICES'],render_cuda_mask=render['CUDA_VISIBLE_DEVICES']))
    try:
        with (wd/'service.log').open('a') as log:
            model=subprocess.Popen([str(MODEL),'-u','-m','preparations.idea1_v3_4_v12_compensation_200.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic(wd/'process.json',dict(pid=model.pid,port=port,time=time.time()))
            atomic(wd/'state.json',dict(status='loading',time=time.time()));start=time.monotonic()
            while True:
                if model.poll() is not None:raise RuntimeError('Service exited during startup')
                if time.monotonic()-start>900:raise TimeoutError('Service readiness')
                try:health=rpc(port,dict(op='health'));break
                except (OSError,ConnectionError):time.sleep(1)
            import numpy as np
            with np.load(OLD/'workers/gpu0/gate_actions.npz') as ref,np.load(wd/'gate_actions.npz') as got:
                checks=[dict(key=k,passed=bool(np.allclose(ref[k],got[k],atol=1e-5,rtol=1e-5))) for k in ref.files]
            atomic(wd/'cross_service.json',dict(passed=all(r['passed'] for r in checks),checks=checks))
            assert all(r['passed'] for r in checks)
            actors=health['protocol']['environment_workers']
            atomic(wd/'state.json',dict(status='evaluating',actors=actors,time=time.time()))
            for task in range(10):
                for suite in SUITES:
                    key=f'{suite}_task{task:02d}';done=OUT/'group_done'/f'{key}.json'
                    if done.exists():continue
                    try:
                        with lock(OUT/'group_claims'/f'{key}.lock',blocking=False):
                            if done.exists():continue
                            tick=time.time();atomic(wd/'state.json',dict(status='evaluating',suite=suite,task=task,actors=actors,time=tick))
                            files=[];runs=[]
                            try:
                                for actor in range(actors):
                                    f=(wd/f'{key}_actor{actor}.log').open('a');files.append(f)
                                    runs.append(subprocess.Popen([str(EVAL),'-u','-m','preparations.idea1_v3_4_v12_compensation_200.evaluate','--port',str(port),'--gpu',str(gpu),'--actor',str(actor),'--suite',suite,'--task',str(task)],cwd=ROOT,env=render,stdout=f,stderr=subprocess.STDOUT))
                                codes=[p.wait() for p in runs]
                                if any(codes):raise RuntimeError(f'Evaluation failed {key}: {codes}')
                            finally:
                                for f in files:f.close()
                            from .common import casekey
                            m=json.loads((OUT/'manifest.json').read_text());owned=[c for c in m['cases'] if c['suite']==suite and c['task']==task]
                            assert all((OUT/'case_done'/f'{casekey(c)}.json').exists() for c in owned)
                            atomic(done,dict(suite=suite,task=task,gpu=gpu,started=tick,finished=time.time(),wall_s=time.time()-tick))
                    except BlockingIOError:continue
            atomic(wd/'state.json',dict(status='complete',time=time.time()))
    except Exception:
        atomic(wd/'error.json',dict(error=traceback.format_exc(),time=time.time()));raise
    finally:
        for p in runs+[model]:
            if p is not None and p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        atomic(wd/'exits.json',dict(model=None if model is None else model.poll(),evaluations=[p.poll() for p in runs],time=time.time()))

def main():
    from .finish import progress,main as finish
    with lock(OUT/'controller.lock',blocking=False):
        m=json.loads((OUT/'manifest.json').read_text());assert all(sha(ROOT/p)==h for p,h in m['source_files'].items())
        assert sha(OLD/'adapters.pt')==m['adapter_sha256']
        atomic(OUT/'process.json',dict(pid=os.getpid(),time=time.time()))
        pending=list(m['gpus']);futures={}
        with concurrent.futures.ThreadPoolExecutor(3) as ex:
            while pending or any(not f.done() for f in futures.values()):
                for gpu in list(pending):
                    if free(gpu):futures[gpu]=ex.submit(worker,gpu);pending.remove(gpu)
                atomic(OUT/'status.json',dict(status='running',progress=progress(),waiting_gpus=pending,
                    workers={str(g):json.loads((OUT/'workers'/f'gpu{g}'/'state.json').read_text()) if (OUT/'workers'/f'gpu{g}'/'state.json').exists() else {} for g in futures},time=time.time()))
                for f in futures.values():
                    if f.done() and f.exception():f.result()
                time.sleep(30)
            for f in futures.values():f.result()
        finish()

if __name__=='__main__':
    try:main()
    except Exception:
        atomic(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
