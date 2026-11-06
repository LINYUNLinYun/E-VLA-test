"""Bounded parity, shared-conditioning offline analysis and independent benchmarks."""
import argparse, json, os, subprocess, time, pickle, socket, struct, threading
from pathlib import Path
import numpy as np
from common import atomic_json, digest, sha, lock,gpu_lease,prior_diagnostic_budget
from runtime_data import load_observation,verify_frozen,parity_identity,verify_parity,verify_acceptance

def benchmark_transport(policy):
    """Real framed TCP transport, co-resident client/server; no amortized features."""
    from runtime_client import legacy,rpc
    server=socket.socket(); server.bind(('127.0.0.1',0)); server.listen(1); port=server.getsockname()[1]
    def serve():
        while True:
            conn,_=server.accept()
            with conn:
                try:
                    size=struct.unpack('!I',legacy.recv_exact(conn,4))[0]
                    req=pickle.loads(legacy.recv_exact(conn,size)); req.pop('op',None)
                    result=policy.predict(**req); result.setdefault('ok',True)
                except Exception as exc: result=dict(ok=False,error=repr(exc))
                payload=pickle.dumps(result,protocol=5); conn.sendall(struct.pack('!I',len(payload))+payload)
    threading.Thread(target=serve,daemon=True).start()
    return lambda **req:rpc(port,dict(op='predict',**req))

def serial(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    return value
def save_prediction(path,result,**identity):
    actions=np.asarray(result['actions'])
    if actions.ndim!=2 or actions.shape[1]!=7 or not np.isfinite(actions).all(): raise ValueError('Nonfinite/invalid action output')
    row=serial({k:v for k,v in result.items() if k!='actions'}); row.update(identity,actions=actions.tolist())
    if not row.get('noise_hash') or not row.get('input_hash') or not row.get('audit'): raise ValueError('Missing native inference provenance')
    for source,key in [('noise_path','noise_file_sha256'),('prediction_path','raw_prediction_file_sha256')]:
        if not row.get(source): raise ValueError('Native raw/noise artifact was not persisted')
        row[key]=sha(row[source])
    row['server_prediction_path']=row['prediction_path']
    atomic_json(path,row); return row
def assert_exclusive_gpu(snapshot=False):
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not gpu or ',' in gpu: raise ValueError('Select exactly one physical GPU')
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    pids=[int(line.split(',')[1]) for line in output.splitlines() if line.split(',')[0].strip()==uuid]
    if set(pids)-{os.getpid()}: raise RuntimeError(f'GPU is not exclusive: {pids}')
    row=dict(gpu=gpu,uuid=uuid,pids=pids,timestamp=time.time(),cpu_isolation='not guaranteed; initial/final process snapshots retained')
    if snapshot: row['processes']=subprocess.check_output(['ps','-eo','pid,pcpu,comm'],text=True)
    return row

def execute(root,model,stage):
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu is None or ',' in gpu: raise ValueError('One physical GPU is required')
    with gpu_lease(gpu) as fd:
        from runtime_runner import free_gpus
        if int(gpu) not in free_gpus(): raise RuntimeError('GPU became busy before native model load')
        os.environ['V22_GPU_LEASE_FD']=str(fd)
        _execute(root,model,stage)
def _execute(root,model,stage):
    from runtime_models import NativePolicy
    root=Path(root); manifest=json.loads((root/'condition_manifest.json').read_text()); points=manifest['observations'][model]
    conds=[c for c in manifest['conditions'] if c['model']==model]
    with lock(root/'leases'/f'{model}.inference.lock'):
        policy=NativePolicy(model,root/'native'/model)
        verify_frozen(root,policy.protocol,model)
        atomic_json(root/'protocols'/f'{model}.runtime.json',serial(policy.protocol))
        if stage=='parity':
            path=root/'parity'/f'{model}.json'
            identity=parity_identity(manifest,serial(policy.protocol),model)
            previous=json.loads(path.read_text()) if path.exists() else {'reserved_calls':0}
            if previous.get('passed') and previous.get('identity')==identity: return
            if previous.get('identity') and previous['identity']!=identity: raise ValueError('Parity identity changed; create a new attempt')
            calls=previous['reserved_calls']; checks=[]
            # Exactly two real points and both masks; native switching included by policy.parity.
            for point in points[:2]:
                obs,instruction,seed=load_observation(point)
                for skip in ([],next(c['skip'] for c in conds if c['mask']!='full')):
                    schedules=[]
                    for c in conds:
                        if c['skip']==skip and c['schedule'] not in schedules: schedules.append(c['schedule'])
                    budget=4*len(schedules)+8
                    if calls+budget>256: raise RuntimeError('Diagnostic logical prediction budget exhausted')
                    # Persist reservation before a diagnostic: a crash cannot reset the budget.
                    budget_path=root/'readiness'/model/'diagnostic_budget.json'
                    with lock(budget_path.with_suffix('.lock')):
                        previous=json.loads(path.read_text()) if path.exists() else {'reserved_calls':0}
                        smoke=json.loads(budget_path.read_text()) if budget_path.exists() else {'smoke_reserved':0}
                        prior=prior_diagnostic_budget(manifest,model)
                        if previous['reserved_calls']+budget+smoke['smoke_reserved']+prior>256: raise RuntimeError('Persistent combined diagnostic budget exhausted')
                        atomic_json(path,dict(passed=False,reserved_calls=previous['reserved_calls']+budget,checks=checks,identity=identity,prior_predictions=prior))
                    result=policy.parity(obs,instruction,seed=seed,skip=skip,schedules=schedules)
                    actual=result.get('actual_predictions')
                    if not isinstance(actual,int) or actual>budget: raise RuntimeError('Parity did not provide bounded actual prediction count')
                    checks.append(serial(result)); calls+=budget
                    if result.get('passed',result.get('ok',False)) is not True: raise RuntimeError(f'Native/cache parity failed: {result}')
            atomic_json(path,dict(passed=True,reserved_calls=calls,actual_predictions=sum(x['actual_predictions'] for x in checks),checks=checks,rtol=1e-4,atol=1e-5,identity=identity,manifest_hash=manifest['manifest_hash'],runtime_protocol_hash=digest(serial(policy.protocol))))
            return
        parity=verify_parity(root,serial(policy.protocol),model)
        verify_acceptance(root,model,parity)
        if stage=='offline':
            for point in points:
                if (root/'PAUSE').exists(): return
                obs,instruction,seed=load_observation(point)
                for mask in ('full',next(c['mask'] for c in conds if c['mask']!='full')):
                    selected=[c for c in conds if c['mask']==mask]
                    handle=policy.prepare_condition(obs,instruction,skip=selected[0]['skip'])
                    noise=None
                    for c in selected:
                        path=root/'offline'/model/f"point{point['index']:02d}"/(c['id']+'.json')
                        if path.exists(): result=json.loads(path.read_text())
                        else:
                            result=policy.sample_condition(handle,seed,c['schedule'],category='offline')
                            result=save_prediction(path,result,condition=c['id'],point=point,seed=seed,schedule=c['schedule'],cache_shared=True)
                        if noise and result['noise_hash']!=noise: raise ValueError('Initial noise differs across schedules')
                        noise=result['noise_hash']
            return
        if stage!='benchmark': raise ValueError(stage)
        evidence=assert_exclusive_gpu(snapshot=True); evidence['client_server_placement']='co-resident process, real localhost TCP framing; no simulation/video background launched by scheduler'; evidence['timing_boundaries']=dict(policy.protocol['timing_boundaries'],client_roundtrip_ms='real localhost request/response serialization and TCP plus full predict including native NPZ/noise and ledger persistence; excludes caller sample JSON, exclusivity scans and simulator'); atomic_json(root/'benchmark'/model/'exclusivity.json',evidence)
        transport=benchmark_transport(policy)
        for point in points:
            if (root/'PAUSE').exists(): return
            assert_exclusive_gpu()
            obs,instruction,seed=load_observation(point); expected_noise=None
            for repeat in range(25):
                rotation=(point['index']+repeat)%len(conds); ordered=conds[rotation:]+conds[:rotation]
                for c in ordered:
                    if (root/'PAUSE').exists(): return
                    path=root/'benchmark'/model/'samples'/f"p{point['index']:02d}_r{repeat:02d}_{c['id']}.json"
                    if path.exists(): row=json.loads(path.read_text())
                    else:
                        before=policy.calls.copy() if isinstance(policy.calls,dict) else policy.calls
                        start=time.perf_counter(); result=transport(obs=obs,instruction=instruction,seed=seed,skip=c['skip'],schedule=c['schedule'],category='warmup' if repeat<5 else 'measurement'); elapsed=(time.perf_counter()-start)*1000
                        result.setdefault('timing',{})['client_roundtrip_ms']=elapsed
                        row=save_prediction(path,result,condition=c['id'],point=point['index'],seed=seed,phase='warmup' if repeat<5 else 'measurement',repeat=repeat,independent_forward=True,calls_before=before,calls_after=policy.calls)
                    if expected_noise and row['noise_hash']!=expected_noise: raise ValueError('Benchmark noise not paired')
                    expected_noise=row['noise_hash']
        atomic_json(root/'benchmark'/model/'final_exclusivity.json',assert_exclusive_gpu(snapshot=True))
        atomic_json(root/'benchmark'/model/'complete.json',dict(conditions=len(conds),points=32,warmup=5,measurement=20,exclusive=True))
def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['parity','offline','benchmark']); p.add_argument('--attempt',required=True); p.add_argument('--model',required=True); p.add_argument('--launch',action='store_true'); a=p.parse_args()
    if not a.launch: print(json.dumps(dict(stage=a.stage,attempt=a.attempt,model=a.model,dry_run=True))); return
    execute(a.attempt,a.model,a.stage)
if __name__=='__main__': main()
