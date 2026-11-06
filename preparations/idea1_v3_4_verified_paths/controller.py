import concurrent.futures,json,os,subprocess,time,traceback
from .common import PRIOR,sha
from .common import ROOT,OUT,pending,release,claim_group,finish_group,phase_done,progress
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc

MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'

def available():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={r.split(',')[0].strip() for r in apps.splitlines() if r.strip()}
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    free=[int(g) for g,u,m in ([x.strip() for x in r.split(',')] for r in rows.splitlines()) if u not in busy and float(m)<=1500]
    atomic(OUT/'resource_scan.json',dict(time=time.time(),free=free,gpus=rows,compute_apps=apps))
    return free

def worker(gpu):
    wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True);port=26010+gpu
    for attempt in range(2):
        env=environment(gpu);env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4');env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        render,mapping=render_environment(env,gpu);gpu_uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();atomic(wd/'identity.json',dict(gpu=gpu,gpu_uuid=gpu_uuid,egl_mapping=mapping));service=None;runs=[]
        try:
            with (wd/f'service_attempt{attempt+1}.log').open('w') as log:
                tick=time.monotonic();service=subprocess.Popen([str(MODEL),'-u','-m','preparations.idea1_v3_4_verified_paths.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic(wd/'process.json',dict(pid=service.pid,gpu=gpu,port=port,time=time.time()));atomic(wd/'state.json',dict(status='loading',time=time.time()))
                while True:
                    if service.poll() is not None:raise RuntimeError('Service startup failed')
                    if time.monotonic()-tick>900:raise TimeoutError('Service readiness')
                    try:health=rpc(port,dict(op='health'));break
                    except (OSError,ConnectionError):time.sleep(1)
                actors=health['protocol']['environment_workers'];startup=dict(status='evaluating',actors=actors,startup_s=time.monotonic()-tick,time=time.time());atomic(wd/'startup.json',startup);atomic(wd/f'startup_attempt{attempt+1}.json',startup)
                while pending():
                    group=claim_group(gpu)
                    if group is None:atomic(wd/'state.json',dict(status='waiting_phase',time=time.time()));time.sleep(5);continue
                    rpc(port,dict(op='reset_all'));atomic(wd/'state.json',dict(status='evaluating',group=group,actors=actors,time=time.time()));runs=[];files=[]
                    try:
                        for actor in range(actors):
                            f=(wd/f"{group['key']}_actor{actor}_attempt{attempt+1}.log").open('a');files.append(f)
                            runs.append(subprocess.Popen([str(EVAL),'-u','-m','preparations.idea1_v3_4_verified_paths.evaluate','--port',str(port),'--gpu',str(gpu),'--actor',str(actor),'--group',group['key'],'--phase',group['phase']],cwd=ROOT,env=render,stdout=f,stderr=subprocess.STDOUT))
                        codes=[p.wait() for p in runs]
                        if any(codes):raise RuntimeError(f"Evaluators failed {group['key']}: {codes}")
                        if group['task'] in (0,2):
                            atomic(wd/'state.json',dict(status='profiling',group=group,time=time.time()))
                            with (wd/f"{group['key']}_profile.log").open('a') as profile_log:
                                subprocess.run([str(EVAL),'-u','-m','preparations.idea1_v3_4_verified_paths.profile','--port',str(port),'--suite',group['suite'],'--task',str(group['task']),'--gpu',str(gpu)],cwd=ROOT,env=render,stdout=profile_log,stderr=subprocess.STDOUT,check=True)
                        finish_group(group['key'])
                    finally:
                        for f in files:f.close()
            atomic(wd/'state.json',dict(status='complete',time=time.time()));return
        except Exception:
            atomic(wd/f'error_attempt{attempt+1}.json',dict(error=traceback.format_exc(),time=time.time()))
            if attempt:raise
        finally:
            for process in runs+[service]:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            release(gpu);atomic(wd/'exits.json',dict(service=None if service is None else service.poll(),evaluations=[p.poll() for p in runs],time=time.time()))
        while pending() and gpu not in available():time.sleep(30)
        if not pending():return

def main():
    with lock(OUT/'controller.lock',blocking=False):
        from .common import freeze
        if not (OUT/'manifest.json').exists():freeze()
        from .finish import main as finish
        atomic(OUT/'process.json',dict(pid=os.getpid(),time=time.time()));last_scan=-1e9;active={}
        while not ((PRIOR/'complete.json').exists() and (PRIOR/'audit.json').exists() and json.loads((PRIOR/'audit.json').read_text())['passed']):
            old=json.loads((PRIOR/'status.json').read_text());old.pop('progress',None)
            atomic(OUT/'status.json',dict(status='waiting_prior_experiment',prior=old,time=time.time()));time.sleep(30)
        atomic(OUT/'dependency_receipt.json',dict(complete_sha256=sha(PRIOR/'complete.json'),audit_sha256=sha(PRIOR/'audit.json'),time=time.time()))
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            while pending() or active:
                for gpu,future in list(active.items()):
                    if future.done():future.result();del active[gpu]
                if pending() and len(active)<3 and time.monotonic()-last_scan>=30:
                    last_scan=time.monotonic()
                    for gpu in available():
                        if gpu not in active and len(active)<3:active[gpu]=pool.submit(worker,gpu)
                p=progress();atomic(OUT/'status.json',dict(status='running' if active else 'waiting_gpu',progress=p,completed=sum(r['completed'] for r in p.values()),target=1000,new_completed=sum(r['new'] for r in p.values()),reused=sum(r['reused'] for r in p.values()),active_gpus=list(active),phase='verified_paths',time=time.time()));time.sleep(10)
        finish()

if __name__=='__main__':
    try:main()
    except Exception:atomic(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
