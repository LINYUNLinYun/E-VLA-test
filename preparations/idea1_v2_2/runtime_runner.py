"""Default-dry-run staged experiment entrypoint and four-GPU event scheduler."""
import argparse, json, os, signal, socket, subprocess, sys, time, traceback
from pathlib import Path
from common import ROOT, MODELS, SUITES, Ledger, atomic_json, conditions, planned, sha, lock, evaluation_env, bootstrap_environment,gpu_lease,exclusive,digest,prior_diagnostic_budget

HERE=Path(__file__).resolve().parent
CLIENT=ROOT/'policies/libero-eval-v21/.venv/bin/python'
def model_python(model):
    project={'pi0':'realtime-vla-flash','xiaomi-robotic0':'xiaomi-robotic0','db-cogact':'db-cogact'}[model]
    return ROOT/'policies'/project/'.venv/bin/python'
def free_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    busy={line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}; uuids=dict(x.strip().split(', ') for x in ids.splitlines())
    return [int(i) for i,mem,util in (map(str.strip,line.split(',')) for line in output.splitlines()) if float(mem)<1500 and float(util)<10 and uuids[i] not in busy]
def available_port():
    with socket.socket() as s: s.bind(('127.0.0.1',0)); return s.getsockname()[1]
def worker(root,model,gpu,shard=0,shards=2,initial=False,smoke=False):
    with exclusive(Path(root)/'leases'/f'{model}.shard{shard}.lock'),gpu_lease(gpu) as fd:
        if gpu not in free_gpus(): raise RuntimeError('GPU became busy before native server launch')
        os.environ['V22_GPU_LEASE_FD']=str(fd)
        _worker(root,model,gpu,shard,shards,initial,fd,smoke)
def _worker(root,model,gpu,shard,shards,initial,lease_fd,smoke):
    from runtime_client import rpc,run_episode
    root=Path(root); manifest=json.loads((root/'condition_manifest.json').read_text()); ledger=Ledger(root)
    if not json.loads((root/'parity'/f'{model}.json').read_text()).get('passed'): raise RuntimeError('Model parity has not passed')
    conds={c['id']:c for c in manifest['conditions']}; port=available_port(); env=evaluation_env(gpu)
    service_dir=root/'services'/f'{model}_gpu{gpu}'; service_dir.mkdir(parents=True,exist_ok=True)
    with (service_dir/'server.log').open('a') as log:
        server=subprocess.Popen([str(model_python(model)),str(HERE/'runtime_server.py'),'--model',model,'--port',str(port),'--output-dir',str(service_dir),'--launch'],env=env,pass_fds=(lease_fd,),stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+900
            while True:
                if server.poll() is not None: raise RuntimeError('Native server exited; inspect server.log')
                try: health=rpc(port,dict(op='health')); break
                except (OSError,RuntimeError):
                    if time.monotonic()>deadline: raise TimeoutError('Native server startup timed out')
                    time.sleep(2)
            runtime=health['protocol']; expected=json.loads((root/'protocols'/f'{model}.json').read_text())
            from runtime_data import verify_parity,verify_acceptance
            parity=verify_parity(root,runtime,model)
            if smoke:
                for condition in [c for c in conds.values() if c['model']==model and c['schedule']['num_steps']==2 and (c['schedule']['kind']=='native' if model=='db-cogact' else c['schedule'].get('m')==.5)]:
                    directory=root/'readiness'/model/'smoke'/condition['id']
                    if (directory/'validated.json').exists(): continue
                    budget_path=root/'readiness'/model/'diagnostic_budget.json'
                    with lock(budget_path.with_suffix('.lock')):
                        budget=json.loads(budget_path.read_text()) if budget_path.exists() else dict(smoke_reserved=0,smoke_actual=0,entries=[])
                        prior=prior_diagnostic_budget(manifest,model)
                        if parity['reserved_calls']+budget['smoke_reserved']+prior+2>256: raise RuntimeError('Combined model diagnostic budget exceeds 256')
                        entry=dict(condition=condition['id'],reserved_predictions=2,directory=str(directory),status='reserved')
                        budget['smoke_reserved']+=2; budget['entries'].append(entry); atomic_json(budget_path,budget)
                    row=dict(model=model,condition=condition['id'],suite='libero_object',task=0,init=6,key=digest(['diagnostic_smoke',model,condition['id'],manifest['manifest_hash']]))
                    result=run_episode(directory,port,condition,row,diagnostic_max_predictions=2)
                    with lock(budget_path.with_suffix('.lock')):
                        budget=json.loads(budget_path.read_text()); budget['smoke_actual']+=result['predictions']; entry=next(x for x in reversed(budget['entries']) if x['directory']==str(directory)); entry.update(status='valid_diagnostic',actual_predictions=result['predictions'],validated_sha256=sha(directory/'validated.json'),not_formal=True); atomic_json(budget_path,budget)
                return
            verify_acceptance(root,model,parity)
            for row in planned(manifest,initial):
                if row['model']!=model or int(row['key'][:8],16)%shards!=shard: continue
                if (root/'PAUSE').exists(): break
                token=ledger.reserve(row,require_artifacts=True)
                if token is None: continue
                directory=root/'episodes'/row['condition']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/token
                ledger.update(token,directory=str(directory))
                try:
                    result=run_episode(directory,port,conds[row['condition']],row)
                    ledger.update(token,status='valid',success=result['success'],validated_sha256=sha(directory/'validated.json'))
                except Exception:
                    ledger.update(token,status='invalid',error=traceback.format_exc())
                    # A model/numerical/media error requires investigation, not automatic retry.
                    raise
        finally:
            server.terminate()
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired: server.kill(); server.wait()

def schedule(root,stage,models,initial=False):
    with exclusive(Path(root)/'leases'/'scheduler.lock'):
        _schedule(root,stage,models,initial)
def _schedule(root,stage,models,initial=False):
    root=Path(root); tasks=[(m,s) for m in models for s in (range(2) if stage=='run' else [0])]; active={}; failures=[]; stop=False; next_scan=0
    def pause(*_):
        nonlocal stop
        stop=True; atomic_json(root/'PAUSE',dict(reason='scheduler signal',time=time.time()))
    signal.signal(signal.SIGINT,pause); signal.signal(signal.SIGTERM,pause)
    while tasks or active:
        for gpu,(proc,task,stream,lease) in list(active.items()):
            code=proc.poll()
            if code is not None:
                stream.close(); lease.__exit__(None,None,None); del active[gpu]
                next_scan=0
                if code: failures.append(dict(task=task,returncode=code))
        if stop or (root/'PAUSE').exists(): tasks=[]
        if tasks and len(active)<4 and (not active or next_scan==0) and time.monotonic()>=next_scan:
            free=[g for g in free_gpus() if g not in active]
            next_scan=time.monotonic()+30
            for gpu in free[:4-len(active)]:
                if not tasks: break
                lease=gpu_lease(gpu,allow_inherited=False)
                try: fd=lease.__enter__()
                except RuntimeError: continue
                if gpu not in free_gpus(): lease.__exit__(None,None,None); continue
                model,shard=tasks.pop(0); log=root/'logs'/f'{stage}_{model}_shard{shard}.log'; log.parent.mkdir(parents=True,exist_ok=True); stream=log.open('a')
                cmd=([str(CLIENT),str(HERE/'runtime_runner.py'),'worker','--shard',str(shard),'--gpu',str(gpu)] if stage in ('run','smoke') else [str(model_python(model)),str(HERE/'runtime_inference.py'),stage])
                if stage=='smoke': cmd+=['--smoke']
                cmd += ['--attempt',str(root),'--model',model,'--launch']
                if initial and stage=='run': cmd+=['--initial']
                env=evaluation_env(gpu); env['V22_GPU_LEASE_FD']=str(fd)
                proc=subprocess.Popen(cmd,env=env,pass_fds=(fd,),stdout=stream,stderr=subprocess.STDOUT); active[gpu]=(proc,(model,shard),stream,lease)
        if active:
            # Check completion events without rescanning nvidia-smi while all slots are owned.
            time.sleep(2)
        elif tasks: time.sleep(30)
    atomic_json(root/'logs'/f'{stage}_scheduler.json',dict(failures=failures,paused=stop))
    if failures: raise RuntimeError(f'Workers failed: {failures}')

def main():
    bootstrap_environment()
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['freeze','parity','smoke','offline','run','benchmark','analyze','audit','worker','retry']); p.add_argument('--attempt',type=Path,required=True); p.add_argument('--config',type=Path); p.add_argument('--extension',action='store_true'); p.add_argument('--model',choices=MODELS); p.add_argument('--initial',action='store_true'); p.add_argument('--launch',action='store_true'); p.add_argument('--gpu',type=int); p.add_argument('--shard',type=int,default=0); p.add_argument('--key'); p.add_argument('--diagnosis'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if not a.launch:
        manifest=json.loads((a.attempt/'condition_manifest.json').read_text()) if (a.attempt/'condition_manifest.json').exists() else dict(conditions=conditions(a.extension))
        print(json.dumps(dict(dry_run=True,stage=a.stage,conditions=len(manifest['conditions']),episodes=len(manifest['conditions'])*(40 if a.initial else 200),benchmark_predictions=len(manifest['conditions'])*32*25,attempt=str(a.attempt)),indent=2)); return
    if a.stage=='retry': Ledger(a.attempt).authorize_retry(a.key,a.diagnosis or '')
    elif a.stage=='freeze':
        from runtime_data import freeze
        freeze(a.attempt,json.loads(a.config.read_text()),a.extension)
    elif a.stage=='worker': worker(a.attempt,a.model,a.gpu,a.shard,initial=a.initial,smoke=a.smoke)
    elif a.stage in ('parity','smoke','offline','run','benchmark'): schedule(a.attempt,a.stage,[a.model] if a.model else list(MODELS),a.initial)
    else:
        from runtime_analysis import analyze,audit
        (analyze if a.stage=='analyze' else audit)(a.attempt)
if __name__=='__main__': main()
