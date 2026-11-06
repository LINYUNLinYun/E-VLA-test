"""User-authorized continuation: pause stride trials, use available GPUs.

The frozen model, projections, policy and per-episode audit are unchanged.
Each case is atomically claimed by one GPU; its remaining conditions stay there.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from .common import ROOT,OUT,MODULE,SUITES,directory,ck,sha,verify_frozen,progress,write_csv
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc

MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'


def active_conditions(m):return [c for c in m['conditions'] if c['variant'] in ('baseline','combination')]
def state():
    ids={c['id'] for c in active_conditions(json.loads((OUT/'manifest.json').read_text()))}
    return {k:v for k,v in progress().items() if k in ids}


def claim(gpu):
    with lock(OUT/'continuation_queue.lock'):
        q=json.loads((OUT/'continuation_queue.json').read_text())
        candidates=[r for r in q['cases'] if r['state']=='pending' and r.get('preferred_gpu',gpu)==gpu]
        candidates.sort(key=lambda r:r.get('preferred_gpu')!=gpu)
        if not candidates:return None
        row=candidates[0];row.update(state='running',gpu=gpu,started=time.time())
        atomic(OUT/'continuation_queue.json',q);return row['case']


def finish_case(case):
    with lock(OUT/'continuation_queue.lock'):
        q=json.loads((OUT/'continuation_queue.json').read_text())
        next(r for r in q['cases'] if r['case']==case).update(state='complete',finished=time.time())
        atomic(OUT/'continuation_queue.json',q)


def pending():return any(r['state']!='complete' for r in json.loads((OUT/'continuation_queue.json').read_text())['cases'])


def evaluate(args):
    from .finish import audit_episode
    from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
    from preparations.idea1_v3_4_verified_paths.rollout import episode
    m=json.loads((OUT/'manifest.json').read_text());verify_frozen(m);pool=Fresh();wd=Path(args.output)
    try:
        while True:
            case=claim(args.gpu)
            if case is None:break
            offset=m['cases'].index(case)%len(m['conditions'])
            order=m['conditions'][offset:]+m['conditions'][:offset]
            for cfg in [c for c in order if c['variant'] in ('baseline','combination')]:
                d=directory(cfg['id'],case)
                if (d/'complete.json').exists():r=json.loads((d/'result.json').read_text())
                else:
                    assert not d.exists(),f'Unarchived interrupted output: {d}'
                    atomic(wd/'current.json',dict(case=case,condition=cfg['id'],time=time.time(),gpu=args.gpu))
                    r=episode(d,args.port,case,cfg,pool,m['manifest_hash'])
                a=audit_episode(cfg,d,r)
                atomic(OUT/'episode_audits'/cfg['id']/f'{ck(case)}.json',a)
                print(json.dumps(dict(condition=cfg['id'],case=case,success=r['success'],actions=r['actions'])),flush=True)
            finish_case(case);rpc(args.port,dict(op='reset_all'))
    finally:pool.close()


def quick_gate(backend,unused_output):
    import numpy as np
    import torch
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_cycle3.common import coefficients
    from .model import Selector
    from .scoring import score_and_choose
    samples=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples']
    checks=[];start=time.perf_counter();wd=Path(os.environ['SELECTION_CONTINUATION_WORKER'])
    with np.load(OUT/'benchmark/gradients.npz') as reference:
        for i,sample in enumerate(samples):
            obs,prompt=load_observation(sample['current']);seed=sample['current']['seed']
            context,_=backend.prepare(obs,prompt,());_,raw,_=backend.sample(context,seed,10)
            backend.verify(context,context,raw,seed,sample['suite'])
            selector=Selector(backend);_,g,_,_=selector.run(context,seed,coefficients(),6)
            expected=reference[f'serial_gradient_{i}'];expected_raw=reference[f'serial_raw_{i}']
            row=dict(sample=i,gradient_max_abs=float(np.max(np.abs(g-expected))),
                gradient_passed=bool(np.allclose(g,expected,atol=1e-4,rtol=1e-3)),
                raw_passed=bool(np.allclose(selector.raw.float().cpu().numpy(),expected_raw,atol=1e-5,rtol=1e-5)),
                masks_passed=all(score_and_choose(g,k)[0]==score_and_choose(expected,k)[0] for k in (4,6)))
            row['passed']=all(row[k] for k in ('gradient_passed','raw_passed','masks_passed'))
            checks.append(row);selector.clear()
            atomic(wd/'numerical_gate.json',dict(passed=all(r['passed'] for r in checks),checks=checks,finished_samples=i+1,wall_s=time.perf_counter()-start))
            assert row['passed'],row
    return dict(complete=True,equivalence={'1':True},reused_single_gpu_benchmark=True,wall_s=time.perf_counter()-start)


def server(args):
    from . import benchmark
    from .server import main as original_main
    os.environ['SELECTION_CONTINUATION_WORKER']=args.output
    benchmark.run=quick_gate
    sys.argv=[sys.argv[0],'--port',str(args.port),'--output',args.output]
    original_main()


def free_gpus(record=True):
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={r.split(',')[0].strip() for r in apps.splitlines() if r.strip()}
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    free=[]
    for r in rows.splitlines():
        gpu,uuid,used=[v.strip() for v in r.split(',')]
        if uuid not in busy and float(used)<=1500:free.append((int(gpu),uuid))
    if record:atomic(OUT/'continuation_resource_scan.json',dict(time=time.time(),rows=rows,apps=apps,free=free))
    return free


def worker(gpu,uuid,attach=None):
    wd=OUT/'continuation_workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True)
    service=evaluation=None;owned=False
    with lock(Path('/tmp')/f'embodied-benchmark-gpu-{uuid}.lock',blocking=False):
        env=environment(gpu);env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4')
        env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        render,mapping=render_environment(env,gpu)
        try:
            if attach:
                port=attach['port'];pid=attach['pid']
                command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                assert MODULE+'.server' in command and str(port) in command
                assert rpc(port,dict(op='health'))['protocol']['cuda_graph'] is False
            else:
                if (gpu,uuid) not in free_gpus(record=False):return False
                with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
                with (wd/'service.log').open('a') as stream:
                    service=subprocess.Popen([str(MODEL),'-u','-m',MODULE+'.continuation','--role','server',
                        '--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                pid=service.pid;owned=True;begin=time.time()
                while True:
                    if service.poll() is not None:raise RuntimeError('Cross-GPU service/gate failed')
                    if time.time()-begin>600:raise TimeoutError('Cross-GPU model/gate startup')
                    try:rpc(port,dict(op='health'));break
                    except (OSError,ConnectionError):time.sleep(1)
            atomic(wd/'identity.json',dict(gpu=gpu,gpu_uuid=uuid,port=port,pid=pid,
                egl_mapping=mapping,attached=bool(attach),time=time.time()))
            with (wd/'evaluate.log').open('a') as stream:
                evaluation=subprocess.Popen([str(EVAL),'-u','-m',MODULE+'.continuation','--role','evaluate',
                    '--gpu',str(gpu),'--port',str(port),'--output',str(wd)],cwd=ROOT,env=render,stdout=stream,stderr=subprocess.STDOUT)
            atomic(wd/'evaluation_process.json',dict(pid=evaluation.pid,time=time.time()))
            code=evaluation.wait()
            if code:raise RuntimeError(f'Continuation evaluator exited {code}')
            return True
        except Exception:
            atomic(wd/'failure.json',dict(error=traceback.format_exc(),time=time.time()));return False
        finally:
            if evaluation is not None and evaluation.poll() is None:evaluation.terminate();evaluation.wait(timeout=30)
            if 'port' in locals():
                try:rpc(port,dict(op='stop'))
                except Exception:pass
            if service is not None:
                try:service.wait(timeout=30)
                except subprocess.TimeoutExpired:service.terminate();service.wait(timeout=30)
            atomic(wd/'exits.json',dict(service=None if service is None else service.poll(),evaluation=None if evaluation is None else evaluation.poll(),time=time.time()))


def controller(args):
    import concurrent.futures
    m=json.loads((OUT/'manifest.json').read_text());verify_frozen(m)
    amendment=json.loads((OUT/'execution_amendment.json').read_text())
    assert amendment['supplemental_source_sha256']==sha(Path(__file__))
    with lock(OUT/'continuation_controller.lock',blocking=False):
        atomic(OUT/'continuation_process.json',dict(pid=os.getpid(),time=time.time()))
        original=json.loads((OUT/'worker/process.json').read_text())
        identity=json.loads((OUT/'worker/identity.json').read_text());active={};failed=set();last_scan=0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            active[identity['gpu']]=pool.submit(worker,identity['gpu'],identity['gpu_uuid'],original)
            while pending() or active:
                for gpu,future in list(active.items()):
                    if future.done():
                        if not future.result():failed.add(gpu)
                        del active[gpu]
                if pending() and time.time()-last_scan>=30:
                    last_scan=time.time()
                    for gpu,uuid in free_gpus():
                        if gpu not in active and gpu not in failed:active[gpu]=pool.submit(worker,gpu,uuid)
                s=state();atomic(OUT/'status.json',dict(status='evaluating_active_conditions',completed=sum(r['completed'] for r in s.values()),
                    target=64,progress=s,active_gpus=list(active),sparse_refresh='paused_by_user',time=time.time()))
                if pending() and not active and all(g in failed for g in range(4)):raise RuntimeError('All available workers failed; see per-GPU evidence')
                time.sleep(5)
        atomic(OUT/'continuation_execution.json',dict(finished=time.time(),failed_gpus=list(failed),time=time.time()))
    finish()


def finish():
    import numpy as np
    from .finish import audit_episode,prefix
    m=json.loads((OUT/'manifest.json').read_text());verify_frozen(m);summary=[];episodes=[];requests=[];pairs=[];prefix_checks=[]
    for cfg in active_conditions(m):
        ee=[];qq=[]
        for case in m['cases']:
            d=directory(cfg['id'],case);r=json.loads((d/'result.json').read_text())
            assert r['config']==cfg and r['manifest_hash']==m['manifest_hash'] and (d/'complete.json').exists()
            audit_episode(cfg,d,r);ee.append(r)
            episodes.append(dict(condition=cfg['id'],**case,success=r['success'],actions=r['actions'],policy_s=r['policy_wall_s'],wall_s=r['wall_s']))
            for line in (d/'infer.jsonl').read_text().splitlines():
                q=json.loads(line);t=q['timing'];rt=q['route']
                qq.append(dict(condition=cfg['id'],variant=cfg['variant'],**case,replan=q['replan'],executed=q['executed_length'],full=rt['path']=='full',
                    refresh=rt['mask_refresh'],policy_ms=t['policy_wall_ms'],selector_ms=t['selector_total_ms'],scoring_ms=t.get('selector_scoring_ms',0.),
                    full_ms=t.get('full_path_wall_ms',0.),rpc_ms=q['rpc_ms']))
        requests+=qq;hot=[q for q in qq if q['replan']>0];triggers=[q for q in hot if q['refresh']]
        summary.append(dict(condition=cfg['id'],variant=cfg['variant'],base_id=cfg['base_id'],budget=cfg['budget'],period=cfg['period'],n=8,
            success=sum(r['success'] for r in ee),**{s:sum(r['success'] for r in ee if r['case']['suite']==s) for s in SUITES},
            requests=len(qq),fulls=sum(q['full'] for q in qq),selectors=sum(q['refresh'] for q in qq),
            hot_policy_ms_per_request=float(np.mean([q['policy_ms'] for q in hot])),selector_ms_per_request=float(np.mean([q['selector_ms'] for q in hot])),
            trigger_full_mean_ms=float(np.mean([q['full_ms'] for q in triggers])),trigger_request_p95_ms=float(np.percentile([q['policy_ms'] for q in triggers],95)),
            scoring_mean_ms=float(np.mean([q['scoring_ms'] for q in triggers])),
            policy_ms_per_action=sum(q['policy_ms'] for q in qq)/sum(q['executed'] for q in qq),total_policy_s=sum(q['policy_ms'] for q in qq)/1000))
    for base_cfg in [c for c in active_conditions(m) if c['variant']=='baseline']:
        cfg=next(c for c in active_conditions(m) if c['base_id']==base_cfg['base_id'] and c['variant']=='combination');values=[]
        for case in m['cases']:
            a=directory(base_cfg['id'],case);b=directory(cfg['id'],case);check=prefix(a,b)
            prefix_checks.append(dict(base=base_cfg['id'],candidate=cfg['id'],case=case,**check))
            values.append(dict(passed=check['passed'],baseline=json.loads((a/'result.json').read_text())['success'],candidate=json.loads((b/'result.json').read_text())['success']))
        for scope,rr in [('all8',values),('strict',[r for r in values if r['passed']])]:
            pairs.append(dict(base=base_cfg['id'],candidate=cfg['id'],scope=scope,n=len(rr),baseline_success=sum(r['baseline'] for r in rr),candidate_success=sum(r['candidate'] for r in rr),recovered=sum(not r['baseline'] and r['candidate'] for r in rr),new_failures=sum(r['baseline'] and not r['candidate'] for r in rr)))
    write_csv(OUT/'summary.csv',summary);write_csv(OUT/'requests.csv',requests);write_csv(OUT/'episodes.csv',episodes);write_csv(OUT/'paired.csv',pairs)
    atomic(OUT/'prefix_checks.json',dict(checks=prefix_checks,anomalies=sum(not r['passed'] for r in prefix_checks)))
    workers=[json.loads(p.read_text()) for p in (OUT/'continuation_workers').glob('gpu*/identity.json')]
    atomic(OUT/'audit.json',dict(passed=True,active_episodes=64,sparse_refresh='paused_by_user',
        source_hashes_checked=True,graph_capture_count=0,prefix_anomalies=sum(not r['passed'] for r in prefix_checks),workers=workers))
    atomic(OUT/'complete.json',dict(passed=True,active_episodes=64,sparse_refresh='paused_by_user',time=time.time()))
    atomic(OUT/'status.json',dict(status='complete',completed=64,target=64,sparse_refresh='paused_by_user',time=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['controller','server','evaluate','finish'],required=True)
    p.add_argument('--port',type=int);p.add_argument('--gpu',type=int);p.add_argument('--output');args=p.parse_args()
    {'controller':controller,'server':server,'evaluate':evaluate,'finish':lambda args:finish()}[args.role](args)


if __name__=='__main__':
    try:main()
    except Exception:
        if '--role' in sys.argv and sys.argv[sys.argv.index('--role')+1]=='controller':
            atomic(OUT/'continuation_failure.json',dict(error=traceback.format_exc(),time=time.time()))
        raise
