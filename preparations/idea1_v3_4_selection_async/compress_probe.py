"""Bounded real-input operator/profile and q1/q2/q4 graph comparisons."""
import argparse,gc,json,os,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from .core import Graph,attention,compare,calculate
from .runtime import OUT,REFERENCE,setup,samples,prepare,write_csv


def profile(graph,output):
    for _ in range(2):graph.launch(graph.item);graph.collect()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        with torch.profiler.record_function('frozen_full10_four_direction_backward'):
            calculate(graph.model,graph.item,graph.W,graph.g,True)
        torch.cuda.synchronize()
    p.export_chrome_trace(str(output/'trace.json'))
    (output/'operators.txt').write_text(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    operators=[dict(name=e.key,self_device_us=e.self_device_time_total,device_us=e.device_time_total,calls=e.count)
        for e in p.key_averages()]
    kernels=[]
    for e in p.events():
        if e.device_type==torch.autograd.DeviceType.CUDA:
            kernels.append(dict(name=e.name,start_us=e.time_range.start,end_us=e.time_range.end,duration_us=e.time_range.elapsed_us(),stream=e.device_resource_id))
    atomic(output/'profile.json',dict(operators=operators,kernels=kernels,
        cumulative_kernel_ms=sum(k['duration_us'] for k in kernels)/1000))


def main(args):
    output=OUT/'compression';output.mkdir(parents=True,exist_ok=True)
    with lock(Path('/tmp')/f'embodied-benchmark-gpu-{args.uuid}.lock',blocking=False):
        backend=setup(output);rows=[];checks=[];prep=[]
        methods=[('q4_serial',4,'original',False),('q4_batched',4,'original',True),
            ('q4_split_nomask',4,'split_flash',True),('q2_batched',2,'original',True),('q1',1,'original',False)]
        with np.load(REFERENCE) as ref:
            for index,sample in enumerate(samples()):
                c,seed=prepare(backend,sample)
                # Rotate method order by sample; all share the exact input/noise.
                order=methods[index%len(methods):]+methods[:index%len(methods)]
                for name,q,mode,batch in order:
                    atomic(output/'current.json',dict(sample=index,method=name,time=time.time()))
                    graph=None
                    try:
                        graph=Graph(backend,c,seed,q,mode,batch)
                        for _ in range(5):graph.run(c,seed)
                        for repeat in range(10):
                            r=graph.run(c,seed);rows.append(dict(sample=index,method=name,q=q,repeat=repeat,
                                total_ms=r['wall_ms'],gpu_ms=r['gpu_ms'],score_ms=r['score_ms']))
                        raw=graph.raw.float().cpu().numpy()
                        check=compare(raw,r['gradient'],ref[f'serial_raw_{index}'],ref[f'serial_gradient_{index}'])
                        checks.append(dict(sample=index,method=name,**check))
                        prep.append(dict(sample=index,method=name,capture_s=graph.capture_s,
                            allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20))
                        if index==0 and name in ('q4_batched','q4_split_nomask'):
                            trace=output/name;trace.mkdir(exist_ok=True);profile(graph,trace)
                    except Exception as exc:
                        atomic(output/f'failure_{index}_{name}.json',dict(error=traceback.format_exc(),time=time.time()))
                        if isinstance(exc,torch.cuda.OutOfMemoryError):raise
                    finally:
                        if graph is not None:graph.close()
                        attention('original');gc.collect();torch.cuda.empty_cache()
                    write_csv(output/'timings.csv',rows);write_csv(output/'preparation.csv',prep)
                    atomic(output/'checks.json',dict(checks=checks,finished_sample=index,finished_method=name))
                    print(json.dumps(dict(sample=index,method=name,n=sum(r['method']==name for r in rows))),flush=True)
        summary={}
        for name,_,_,_ in methods:
            rr=[r for r in rows if r['method']==name]
            if rr:
                summary[name]=dict(n=len(rr),mean_ms=float(np.mean([r['total_ms'] for r in rr])),
                    p95_ms=float(np.percentile([r['total_ms'] for r in rr],95)),
                    numerical_passes=sum(c['passed'] for c in checks if c['method']==name),
                    mask_changes=sum(not c['masks_passed'] for c in checks if c['method']==name))
        atomic(output/'complete.json',dict(summary=summary,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--uuid',required=True);args=p.parse_args()
    try:main(args)
    except Exception:atomic(OUT/'compression/failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
