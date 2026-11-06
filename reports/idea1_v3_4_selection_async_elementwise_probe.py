"""Final bounded operator ablation: rotary and GLU reverse fusion."""
import argparse,gc,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4_selection_async.runtime import OUT,ROOT,REFERENCE,setup,samples,prepare,write_csv
PROBE=OUT/'elementwise_batch_probe'


def gpu_probe():
    import torch
    from preparations.idea1_v3_4_selection_async import core
    from reports import idea1_v3_4_selection_async_batched_attention as direct
    from reports import idea1_v3_4_selection_async_rms_backward as rms
    from reports import idea1_v3_4_selection_async_elementwise as element
    backend=setup(PROBE);rows=[];checks=[];preparation=[];saved={};tick=time.perf_counter()
    methods=('rms','rms_rope','rms_rope_glu')
    with np.load(REFERENCE) as reference:
        for index,sample in enumerate(samples()):
            c,seed=prepare(backend,sample);baseline=None
            for name in methods:
                graph=None
                try:
                    element.restore();direct.install(False);rms.install()
                    if name!='rms':element.install(glu=name=='rms_rope_glu')
                    atomic(PROBE/'current.json',dict(sample=index,method=name,time=time.time()))
                    torch.cuda.reset_peak_memory_stats();graph=core.Graph(backend,c,seed,4,'original',True)
                    for _ in range(5):graph.run(c,seed)
                    for repeat in range(5):
                        r=graph.run(c,seed);rows.append(dict(sample=index,method=name,repeat=repeat,total_ms=r['wall_ms'],gpu_ms=r['gpu_ms']))
                    raw=graph.raw.float().cpu().numpy().copy();gradient=r['gradient'].copy()
                    saved[f'{name}_raw_{index}']=raw;saved[f'{name}_gradient_{index}']=gradient
                    checks.append(dict(sample=index,method=name,against='serial',
                        **core.compare(raw,gradient,reference[f'serial_raw_{index}'],reference[f'serial_gradient_{index}'])))
                    if name=='rms':baseline=(raw,gradient)
                    else:
                        check=core.compare(raw,gradient,*baseline)
                        got=core.choose(gradient,6)[1];expected=core.choose(baseline[1],6)[1]
                        check.update(layer_score_max_abs=float(np.max(np.abs(got-expected))),
                            layer_score_passed=bool(np.allclose(got,expected,atol=1e-4,rtol=1e-3)))
                        check['passed']=check['passed'] and check['layer_score_passed']
                        checks.append(dict(sample=index,method=name,against='rms',**check))
                    preparation.append(dict(sample=index,method=name,capture_s=graph.capture_s,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
                except Exception:
                    atomic(PROBE/f'failure_{index}_{name}.json',dict(error=traceback.format_exc(),time=time.time()));raise
                finally:
                    if graph is not None:graph.close()
                    element.restore();gc.collect();torch.cuda.empty_cache()
                write_csv(PROBE/'timings.csv',rows);write_csv(PROBE/'preparation.csv',preparation)
                atomic(PROBE/'checks.json',dict(checks=checks));np.savez(PROBE/'gradients.npz',**saved)
                print(json.dumps(dict(sample=index,method=name)),flush=True)
    summary=[]
    for name in methods:
        rr=[r for r in rows if r['method']==name];cc=[c for c in checks if c['method']==name and c['against']=='serial']
        summary.append(dict(method=name,n=len(rr),mean_ms=float(np.mean([r['total_ms'] for r in rr])),
            p95_ms=float(np.percentile([r['total_ms'] for r in rr],95)),max_ms=max(r['total_ms'] for r in rr),
            raw_passes=sum(c['raw_passed'] for c in cc),serial_passes=sum(c['passed'] for c in cc),
            v12_mask_changes=sum(c['masks']['6']!=c['reference_masks']['6'] for c in cc)))
    atomic(PROBE/'complete.json',dict(summary=summary,wall_s=time.perf_counter()-tick,time=time.time()))


def supervise():
    from preparations.idea1_v3_4_selection_async import controller
    PROBE.mkdir(parents=True,exist_ok=True);controller.OUT=PROBE
    atomic(PROBE/'supervisor.json',dict(pid=os.getpid(),time=time.time()))
    while not (OUT/'rms_batch_probe/released.json').exists():time.sleep(3)
    chosen=None
    while chosen is None:
        for gpu,uuid in controller.available():
            lease=lock(Path('/tmp')/f'embodied-benchmark-gpu-{uuid}.lock',blocking=False)
            try:lease.__enter__()
            except BlockingIOError:continue
            if (gpu,uuid) not in controller.available():lease.__exit__(None,None,None);continue
            chosen=(gpu,uuid,lease);break
        if chosen is None:time.sleep(30)
    gpu,uuid,lease=chosen;process=None
    try:
        env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        with (PROBE/'run.log').open('a') as stream:
            process=subprocess.Popen([str(ROOT/'policies/realtime-vla-flash/.venv/bin/python'),'-u','-m',
                'reports.idea1_v3_4_selection_async_elementwise_probe','--gpu-probe'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        sources=[ROOT/'reports'/f'idea1_v3_4_selection_async_{name}.py' for name in ('elementwise_probe','elementwise','rms_backward','batched_attention')]
        atomic(PROBE/'launch.json',dict(gpu=gpu,uuid=uuid,pid=process.pid,time=time.time(),timeout_s=300,
            source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        try:process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            process.terminate();process.wait(timeout=30);atomic(PROBE/'bounded_stop.json',dict(time=time.time()))
        if process.returncode and not (PROBE/'bounded_stop.json').exists():atomic(PROBE/'failure.json',dict(returncode=process.returncode,time=time.time()))
    finally:
        if process is not None and process.poll() is None:process.terminate();process.wait(timeout=30)
        lease.__exit__(None,None,None);atomic(PROBE/'released.json',dict(gpu=gpu,uuid=uuid,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-probe',action='store_true');a=p.parse_args()
    try:gpu_probe() if a.gpu_probe else supervise()
    except Exception:
        atomic(PROBE/('failure.json' if a.gpu_probe else 'supervisor_failure.json'),dict(error=traceback.format_exc(),time=time.time()));raise
