"""Wait for an unowned authorized GPU, then schedule task shards dynamically."""
import concurrent.futures
import hashlib,json,os,subprocess,time
import numpy as np
from preparations.idea1_v3.io import atomic,append,digest
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3_4.dispatch import render_environment
from .common import ROOT,OUT,OLD,PLAN,CONFIGS,results,progress

MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'


class StartupOOM(RuntimeError):
    pass


def available_gpus():
    # Only query the authorized candidate set; utilization alone is not ownership.
    output=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    devices={int(x[0]):dict(uuid=x[1].strip(),memory_mb=float(x[2])) for x in (line.split(',') for line in output.splitlines())}
    apps=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    owners={line.split(',')[0].strip() for line in apps.splitlines()}
    return devices,[gpu for gpu,info in devices.items() if info['uuid'] not in owners and info['memory_mb']<=1500]


def historical_case_gate(name):
    checks=[]
    for d,r in results(name=name):
        c=r['case']; old=OLD/'full_d10/episodes'/c['suite']/f"task{c['task']:02d}"/f"init{c['init']:02d}"
        first=json.loads((d/'infer.jsonl').read_text().splitlines()[0])
        reference=json.loads((old/'infer.jsonl').read_text().splitlines()[0])
        with np.load(d/'arrays.npz') as a,np.load(old/'arrays.npz') as b:
            passed=first['input_hash']==reference['input_hash'] and all(np.allclose(a[k],b[k],atol=1e-5,rtol=1e-5) for k in ('raw_0','actions_0'))
            checks.append(dict(case=c,passed=bool(passed),raw_max_abs=float(np.abs(a['raw_0']-b['raw_0']).max())))
    atomic(OUT/f'{name}_historical_case_gate.json',dict(passed=bool(checks) and all(x['passed'] for x in checks),checks=checks))
    if not checks or not all(x['passed'] for x in checks):raise RuntimeError('Historical initial-input/Full gate failed')


class Scheduler:
    def __init__(self):
        self.workers=[];self.last_scan=-1e10
        self.idle_since={}
        self.startup_failures={gpu:len(list((OUT/'startup_failures').glob(f'gpu{gpu}_attempt*/startup_failure.json'))) for gpu in range(4)}

    def discover(self):
        if time.monotonic()-self.last_scan<30:return
        self.last_scan=time.monotonic();devices,free=available_gpus()
        now=time.monotonic()
        for gpu in list(self.idle_since):
            if gpu not in free:self.idle_since.pop(gpu)
        for gpu in free:self.idle_since.setdefault(gpu,now)
        eligible=[gpu for gpu in free if now-self.idle_since[gpu]>=60 and self.startup_failures[gpu]<2]
        atomic(OUT/'gpu_scan.json',dict(time=time.time(),devices=devices,available=free,stable_available=eligible,idle_seconds={str(g):now-t for g,t in self.idle_since.items()},startup_failures=self.startup_failures))
        for gpu in eligible:
            if any(w['gpu']==gpu for w in self.workers):continue
            # Recheck immediately before launching, including process ownership.
            again,eligible=available_gpus()
            if gpu not in eligible:continue
            wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True)
            env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
            rendering,mapping=render_environment(env,gpu);port=19740+gpu
            log=(wd/'service.log').open('a')
            process=subprocess.Popen([str(MODEL),'-m','preparations.idea1_v3_4_l12_14_first.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            w=dict(gpu=gpu,wd=wd,render=rendering,port=port,service=process,service_log=log,started=time.monotonic(),ready=False,rollout=None)
            self.workers.append(w)
            atomic(wd/'identity.json',dict(gpu=gpu,**again[gpu],egl_mapping=mapping))
            atomic(wd/'process.json',dict(pid=process.pid,port=port,time=time.time()))
            append(OUT/'events.jsonl',dict(event='service_started',gpu=gpu,pid=process.pid))

    def health(self,w):
        if w['service'].poll() is not None:
            log=(w['wd']/'service.log').read_text()
            if not w['ready'] and 'CUDA out of memory' in log:raise StartupOOM(str(w['gpu']))
            raise RuntimeError('Model service exited: GPU'+str(w['gpu']))
        if w['ready']:return True
        if time.monotonic()-w['started']>1200:raise TimeoutError('Model readiness/gate')
        try:health=rpc(w['port'],dict(op='health'))
        except (OSError,ConnectionError):return False
        with np.load(w['wd']/'gate_actions.npz') as a,np.load(OLD/'cross_service_reference.npz') as b:
            checks={k:float(np.abs(a[k]-b[k]).max()) for k in b.files}
            passed=all(np.allclose(a[k],b[k],atol=1e-5,rtol=1e-5) for k in b.files)
        atomic(w['wd']/'historical_service_gate.json',dict(passed=passed,max_abs=checks))
        if not passed:raise RuntimeError('Historical service equivalence failed')
        atomic(w['wd']/'protocol.json',health);w['ready']=True
        append(OUT/'events.jsonl',dict(event='service_ready',gpu=w['gpu']))
        return True

    def discard_startup_oom(self,w):
        gpu=w['gpu'];self.startup_failures[gpu]+=1;attempt=self.startup_failures[gpu]
        w['service_log'].close()
        target=OUT/'startup_failures'/f'gpu{gpu}_attempt{attempt}';target.parent.mkdir(parents=True,exist_ok=True)
        w['wd'].rename(target)
        atomic(target/'startup_failure.json',dict(gpu=gpu,attempt=attempt,cause='CUDA out of memory during startup',exit=w['service'].poll(),time=time.time(),episodes=0))
        self.workers.remove(w);self.idle_since.pop(gpu,None)
        append(OUT/'events.jsonl',dict(event='startup_oom_retry_wait',gpu=gpu,attempt=attempt,max_attempts_per_gpu=2))

    def evaluate(self,stage,names):
        pending=[(name,task) for task in range(10) for name in names];finished=[]
        while pending or any(w['rollout'] is not None for w in self.workers):
            if pending:self.discover()
            if pending and not self.workers and all(n>=2 for n in self.startup_failures.values()):
                raise RuntimeError('All GPU startup OOM retry budgets exhausted')
            for w in list(self.workers):
                try:
                    if not self.health(w):continue
                except StartupOOM:
                    self.discard_startup_oom(w);continue
                running=w['rollout']
                if running is not None:
                    code=running.poll()
                    if code is None:continue
                    w['rollout_log'].close();w['rollout']=None
                    if code:raise RuntimeError(f"Rollout {stage} {w['condition']} task{w['task']} failed with {code}")
                    finished.append((w['condition'],w['task']))
                if pending:
                    name,task=pending.pop(0);w['task']=task;w['condition']=name
                    log=(w['wd']/f'{stage}_{name}_task{task:02d}.log').open('a');w['rollout_log']=log
                    cmd=[str(EVAL),'-m','preparations.idea1_v3_4.rollout','--manifest',str(OUT/'manifest.json'),'--condition',name,'--output',str(OUT),'--port',str(w['port']),'--shard',str(task),'--shards','10','--stage',stage]
                    w['rollout']=subprocess.Popen(cmd,cwd=ROOT,env=w['render'],stdout=log,stderr=subprocess.STDOUT)
                    append(OUT/'events.jsonl',dict(event='task_started',stage=stage,condition=name,task=task,gpu=w['gpu'],pid=w['rollout'].pid))
            atomic(OUT/'status.json',dict(status='running' if self.workers else 'waiting_for_gpu',stage=stage,conditions=progress(),pending_tasks=pending,finished_tasks=finished,gpus=[w['gpu'] for w in self.workers],time=time.time()))
            time.sleep(5)
        for name in names:historical_case_gate(name)
        append(OUT/'events.jsonl',dict(event='stage_complete',stage=stage,conditions=progress()))

    def benchmark(self,pairs):
        # Services already gate-checked and idle; use only those actually acquired.
        for w in list(self.workers):
            try:
                while not self.health(w):time.sleep(1)
            except StartupOOM:self.discard_startup_oom(w)
        with concurrent.futures.ThreadPoolExecutor(len(self.workers)) as ex:
            futures=[ex.submit(rpc,w['port'],dict(op='bench',pairs=pairs[i::len(self.workers)],output=str(w['wd']/'timing.json'))) for i,w in enumerate(self.workers)]
            for future in futures:future.result()
        rows=[]
        for w in self.workers:rows+=json.loads((w['wd']/'timing.json').read_text())['rows']
        means={name:float(np.mean([r['wall_ms'] for r in rows if r['path']==name])) for name in ('full10','v14','old12','old13','old14','old15')}
        cycle=dict(baseline=(means['full10']+2*means['v14'])/3)
        cycle.update({f'L{layer}_first':(means['full10']+means[f'old{layer}']+means['v14'])/3 for layer in (12,13,14,15)})
        return means,cycle

    def close(self):
        exits={}
        for w in self.workers:
            for key in ('rollout','service'):
                p=w[key]
                if p is not None and p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
            w['service_log'].close()
            if 'rollout_log' in w and not w['rollout_log'].closed:w['rollout_log'].close()
            exits[str(w['gpu'])]=dict(pid=w['service'].pid,exit=w['service'].poll())
        atomic(OUT/'service_exits.json',exits)


def main():
    with lock(OUT/'controller.lock',blocking=False):
        m=json.loads((OUT/'manifest.json').read_text());body=dict(m);body.pop('manifest_hash');assert digest(body)==m['manifest_hash']
        for path,h in m['source_files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
        assert hashlib.sha256(PLAN.read_bytes()).hexdigest()==m['plan_sha256']
        atomic(OUT/'process.json',dict(pid=os.getpid(),started=time.time()))
        scheduler=Scheduler()
        try:
            names=[c['id'] for c in CONFIGS]
            scheduler.evaluate('screen',names)
            atomic(OUT/'status.json',dict(status='benchmarking',conditions=progress(),time=time.time()))
            means,cycle=scheduler.benchmark(m['timing_pairs'])
            success={name:sum(r['success'] for d,r in results(name=name,scope='first40')) for name in names}
            expanded=[name for name in names if success[name]>=m['expansion']['min_screen_success'] and cycle[name]<=cycle['baseline']]
            atomic(OUT/'decision.json',dict(screen_success=success,expanded=expanded,request_ms=means,cycle_ms=cycle,time=time.time()))
            if expanded:scheduler.evaluate('expand',expanded)
        finally:scheduler.close()
        from .finish import main as finish
        finish()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic(OUT/'failed.json',dict(error=repr(exc),time=time.time()))
        atomic(OUT/'status.json',dict(status='failed',error=repr(exc),time=time.time()))
        raise
