"""At most five GPU minutes for the physical SDPA reverse-batch experiment."""
import argparse,gc,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4_selection_async.runtime import OUT,ROOT,REFERENCE,setup,samples,prepare,write_csv

PROBE=OUT/'sdpa_batch_probe'


def gpu_probe():
    import torch
    from preparations.idea1_v3_4_selection_async import core
    from reports import idea1_v3_4_selection_async_batched_attention as direct
    PROBE.mkdir(parents=True,exist_ok=True);backend=setup(PROBE);start=time.perf_counter()
    original_attention=core.attention;rows=[];checks=[];preparation=[];saved={};disabled=set()
    with np.load(REFERENCE) as reference:
        for index,sample in enumerate(samples()):
            c,seed=prepare(backend,sample);baseline=None
            for name in ('baseline','direct_masked','direct_split'):
                if name in disabled:continue
                graph=None
                try:
                    core.attention=original_attention
                    if name!='baseline':direct.install(split=name=='direct_split')
                    direct.Efficient.backward_calls=direct.Flash.backward_calls=0
                    atomic(PROBE/'current.json',dict(sample=index,method=name,time=time.time()))
                    torch.cuda.reset_peak_memory_stats();graph=core.Graph(backend,c,seed,4,'original',True)
                    for _ in range(5):graph.run(c,seed)
                    for repeat in range(5):
                        r=graph.run(c,seed);rows.append(dict(sample=index,method=name,repeat=repeat,
                            total_ms=r['wall_ms'],gpu_ms=r['gpu_ms']))
                    raw=graph.raw.float().cpu().numpy().copy();gradient=r['gradient'].copy()
                    saved[f'{name}_raw_{index}']=raw;saved[f'{name}_gradient_{index}']=gradient
                    check=core.compare(raw,gradient,reference[f'serial_raw_{index}'],reference[f'serial_gradient_{index}'])
                    if name=='baseline':baseline=(raw,gradient)
                    checks.append(dict(sample=index,method=name,against='serial',**check))
                    if name!='baseline' and baseline is not None:
                        checks.append(dict(sample=index,method=name,against='baseline_batch',
                            **core.compare(raw,gradient,*baseline)))
                    preparation.append(dict(sample=index,method=name,capture_s=graph.capture_s,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        efficient_backward_calls_per_selector=direct.Efficient.backward_calls/4 if name!='baseline' else None,
                        flash_backward_calls_per_selector=direct.Flash.backward_calls/4 if name!='baseline' else None))
                except Exception:
                    atomic(PROBE/f'failure_{index}_{name}.json',dict(error=traceback.format_exc(),time=time.time()))
                    disabled.add(name)
                finally:
                    if graph is not None:graph.close()
                    core.attention=original_attention;original_attention('original');gc.collect();torch.cuda.empty_cache()
                write_csv(PROBE/'timings.csv',rows);write_csv(PROBE/'preparation.csv',preparation)
                atomic(PROBE/'checks.json',dict(checks=checks));np.savez(PROBE/'gradients.npz',**saved)
                print(json.dumps(dict(sample=index,method=name,disabled=sorted(disabled))),flush=True)
    summary=[]
    for name in ('baseline','direct_masked','direct_split'):
        rr=[r for r in rows if r['method']==name]
        if not rr:continue
        summary.append(dict(method=name,n=len(rr),mean_ms=float(np.mean([r['total_ms'] for r in rr])),
            p95_ms=float(np.percentile([r['total_ms'] for r in rr],95)),
            serial_passes=sum(c['passed'] for c in checks if c['method']==name and c['against']=='serial'),
            batch_passes=sum(c['passed'] for c in checks if c['method']==name and c['against']=='baseline_batch')))
    atomic(PROBE/'complete.json',dict(summary=summary,disabled=sorted(disabled),wall_s=time.perf_counter()-start,time=time.time()))


def supervise():
    from preparations.idea1_v3_4_selection_async import controller
    controller.OUT=PROBE;available=controller.available
    PROBE.mkdir(parents=True,exist_ok=True);atomic(PROBE/'supervisor.json',dict(pid=os.getpid(),time=time.time()))
    chosen=None
    while chosen is None:
        if (OUT/'closed_loop_failure.json').exists():raise RuntimeError('Free-loop evaluation failed')
        state=OUT/'closed_loop_status.json'
        # The main scheduler stops discovery once its three slots are filled.
        # Only then can this probe discover a fourth or newly released card.
        # Followup waits for the probe, so resource discovery has one owner.
        main_done=(OUT/'closed_loop_execution_complete.json').exists()
        main_full=state.exists() and len(json.loads(state.read_text())['workers'])>=3
        if not main_done and not main_full:time.sleep(10);continue
        for gpu,uuid in available():
            lease=lock(Path('/tmp')/f'embodied-benchmark-gpu-{uuid}.lock',blocking=False)
            try:lease.__enter__()
            except BlockingIOError:continue
            # Confirm after the shared lease; another scheduler may have won.
            if (gpu,uuid) not in available():lease.__exit__(None,None,None);continue
            chosen=(gpu,uuid,lease);break
        if chosen is None:time.sleep(30)
    gpu,uuid,lease=chosen;child=None
    try:
        env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        with (PROBE/'run.log').open('a') as stream:
            child=subprocess.Popen([str(ROOT/'policies/realtime-vla-flash/.venv/bin/python'),'-u','-m',
                'reports.idea1_v3_4_selection_async_sdpa_probe','--gpu-probe'],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT)
        files=[ROOT/'reports'/f'idea1_v3_4_selection_async_{name}.py' for name in ('sdpa_probe','batched_attention')]
        atomic(PROBE/'launch.json',dict(gpu=gpu,uuid=uuid,pid=child.pid,time=time.time(),timeout_s=300,
            source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
        try:child.wait(timeout=300)
        except subprocess.TimeoutExpired:
            child.terminate();child.wait(timeout=30)
            atomic(PROBE/'bounded_stop.json',dict(reason='Five-minute bound reached',time=time.time()))
        if child.returncode and not (PROBE/'bounded_stop.json').exists():
            atomic(PROBE/'failure.json',dict(returncode=child.returncode,time=time.time()))
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
        lease.__exit__(None,None,None);atomic(PROBE/'released.json',dict(gpu=gpu,uuid=uuid,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-probe',action='store_true');a=p.parse_args()
    try:gpu_probe() if a.gpu_probe else supervise()
    except Exception:
        atomic(PROBE/('failure.json' if a.gpu_probe else 'supervisor_failure.json'),dict(error=traceback.format_exc(),time=time.time()));raise
