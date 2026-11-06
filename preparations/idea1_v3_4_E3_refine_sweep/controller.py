import concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
from preparations.idea1_v3.io import atomic,append,digest
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v2_1.runtime_client import rpc
from .common import ROOT,OUT,PLAN,sha,phase2_qs,select_q
MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
class StartupOOM(RuntimeError):pass

def available_gpus():
    # Only query the authorized candidate set; utilization alone is not ownership.
    output=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    devices={int(x[0]):dict(uuid=x[1].strip(),memory_mb=float(x[2])) for x in (line.split(',') for line in output.splitlines())}
    apps=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    owners={line.split(',')[0].strip() for line in apps.splitlines()}
    return devices,[gpu for gpu,info in devices.items() if info['uuid'] not in owners and info['memory_mb']<=1500]


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
            if len(self.workers)>=2:break
            if any(w['gpu']==gpu for w in self.workers):continue
            # Recheck immediately before launching, including process ownership.
            again,eligible=available_gpus()
            if gpu not in eligible:continue
            wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True)
            env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
            port=19840+gpu
            log=(wd/'service.log').open('a')
            process=subprocess.Popen([str(MODEL),'-m','preparations.idea1_v3_4_E3_refine_sweep.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            w=dict(gpu=gpu,wd=wd,port=port,service=process,service_log=log,started=time.monotonic(),ready=False,rollout=None)
            self.workers.append(w)
            atomic(wd/'identity.json',dict(gpu=gpu,**again[gpu]))
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

    def run_stage(self,stage,selected):
        tasks=sorted({(r['case']['suite'],r['case']['task']) for r in selected})
        pending=[[r['id'] for r in selected if (r['case']['suite'],r['case']['task'])==key] for key in tasks]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            while pending or any(w.get('future') is not None for w in self.workers):
                if pending and len(self.workers)<2:self.discover()
                if pending and not self.workers and all(n>=2 for n in self.startup_failures.values()):raise RuntimeError('Startup OOM retry budgets exhausted')
                for w in list(self.workers):
                    try:
                        if not self.health(w):continue
                    except StartupOOM:self.discard_startup_oom(w);continue
                    if w.get('future') is not None:
                        if not w['future'].done():continue
                        result=w['future'].result();assert set(result['completed'])==set(w['ids']);w['future']=None
                    if pending:
                        ids=pending.pop(0);w['ids']=ids
                        w['future']=pool.submit(rpc,w['port'],dict(op='run',stage=stage,ids=ids))
                        append(OUT/'events.jsonl',dict(event='batch_started',stage=stage,gpu=w['gpu'],ids=ids))
                count=len(list((OUT/f'stage{stage}').glob('sample*/complete.json')))
                atomic(OUT/'status.json',dict(status='running' if self.workers else 'waiting_for_gpu',stage=stage,completed_requests=count,total_requests=127,gpus=[w['gpu'] for w in self.workers],time=time.time()))
                time.sleep(2)
        append(OUT/'events.jsonl',dict(event='stage_complete',stage=stage,requests=127))

    def close(self):
        exits={}
        for w in self.workers:
            p=w['service']
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            w['service_log'].close();exits[str(w['gpu'])]=dict(pid=p.pid,exit=p.poll())
        atomic(OUT/'service_exits.json',exits)


def main():
    with lock(OUT/'controller.lock',blocking=False):
        m=json.loads((OUT/'manifest.json').read_text());body=dict(m);body.pop('manifest_hash');assert digest(body)==m['manifest_hash']
        assert all(sha(ROOT/p)==h for p,h in m['sources'].items()) and sha(PLAN)==m['plan_sha256']
        atomic(OUT/'process.json',dict(pid=os.getpid(),started=time.time()))
        scheduler=Scheduler()
        try:
            scheduler.run_stage(1,m['selected'])
            from .analyze import read_stage,summaries,write_csv
            rows=read_stage(1);write_csv(OUT/'stage1_summary.csv',summaries(rows))
            calibration=[r for r in rows if r['split']=='calibration' and r['horizon']==5 and r['noise']==r['reference']]
            selection=select_q(calibration,m['grid'],'fresh2')
            if (OUT/'stage1_decision.json').exists():assert json.loads((OUT/'stage1_decision.json').read_text())['selection']==selection
            else:atomic(OUT/'stage1_decision.json',dict(selection=selection,selection_split='calibration',time=time.time()))
            spec=dict(grid=phase2_qs(selection['q']),steps=1,parent_manifest_hash=m['manifest_hash'],stage1_decision_sha256=sha(OUT/'stage1_decision.json'),stage1_audit_sha256=sha(OUT/'stage1_audit.json'),time=time.time())
            spec['spec_hash']=digest(spec)
            if (OUT/'stage2_manifest.json').exists():
                old=json.loads((OUT/'stage2_manifest.json').read_text());assert old['grid']==spec['grid'] and old['parent_manifest_hash']==m['manifest_hash'] and old['stage1_decision_sha256']==spec['stage1_decision_sha256']
            else:atomic(OUT/'stage2_manifest.json',spec)
            scheduler.run_stage(2,m['selected'])
        finally:scheduler.close()
        from .analyze import finish
        finish()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic(OUT/'failed.json',dict(error=repr(exc),time=time.time()))
        atomic(OUT/'status.json',dict(status='failed',error=repr(exc),time=time.time()))
        raise

