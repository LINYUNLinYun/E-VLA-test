"""Measure action readiness and the remaining selector stall, not summed work."""
import argparse,gc,json,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from .core import Graph,attention
from .runtime import OUT,setup,samples,prepare,write_csv


def merge(intervals):
    result=[]
    for a,b in sorted(intervals):
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result


def intersection(a,b):
    i=j=0;total=0.
    while i<len(a) and j<len(b):
        total+=max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total


def trace_call(fn,output):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        fn();torch.cuda.synchronize()
    p.export_chrome_trace(str(output/'overlap_trace.json'))
    streams=defaultdict(list);kernels=[]
    for e in p.events():
        if e.device_type==torch.autograd.DeviceType.CUDA and not any(x in e.name.lower() for x in ('memcpy','memset')):
            streams[str(e.device_resource_id)].append((e.time_range.start,e.time_range.end))
            kernels.append(dict(name=e.name,stream=e.device_resource_id,start_us=e.time_range.start,end_us=e.time_range.end))
    unions={s:merge(a) for s,a in streams.items()}
    ordered=sorted(unions,key=lambda s:sum(b-a for a,b in unions[s]),reverse=True)
    overlap=intersection(unions[ordered[0]],unions[ordered[1]]) if len(ordered)>1 else 0.
    result=dict(kernel_overlap_ms=overlap/1000,streams=ordered,
        kernel_work_ms={s:sum(b-a for a,b in intervals)/1000 for s,intervals in unions.items()},kernels=kernels)
    atomic(output/'kernel_overlap.json',result)
    return result


def main(args):
    output=OUT/'overlap';output.mkdir(parents=True,exist_ok=True)
    with lock(Path('/tmp')/f'embodied-benchmark-gpu-{args.uuid}.lock',blocking=False):
        backend=setup(output);rows=[];checks=[];preparation=[]
        low=torch.cuda.Stream(priority=0);high=torch.cuda.Stream(priority=-1)
        for index,sample in enumerate(samples()):
            c,seed=prepare(backend,sample)
            native=[];reference_raw=reference_actions=None
            for _ in range(3):
                torch.cuda.synchronize();tick=time.perf_counter();a,r,_=backend.sample(c,seed,10)
                native.append((time.perf_counter()-tick)*1000)
                reference_raw=r.float().cpu().numpy().copy();reference_actions=a.copy()
            native_ms=float(np.mean(native))
            for q in (4,2,1):
                atomic(output/'current.json',dict(sample=index,q=q,time=time.time()))
                graph=Graph(backend,c,seed,q,'original',q>1)
                preparation.append(dict(sample=index,q=q,capture_s=graph.capture_s))
                solo=[graph.run(c,seed) for _ in range(3)]
                reference_gradient=solo[-1]['gradient'].copy();reference_mask=solo[-1]['mask']
                selector_ms=float(np.mean([r['wall_ms'] for r in solo]))

                def call(priority,execution_ms,record=True):
                    torch.cuda.synchronize();tick=time.perf_counter()
                    prefix=torch.cuda.Event(enable_timing=True);prefix.record()
                    item=graph.stage(c,seed);graph.launch(item,low)
                    infer=torch.cuda.current_stream() if priority=='default' else high
                    if priority!='default':infer.wait_stream(torch.cuda.current_stream())
                    ae_begin=torch.cuda.Event(enable_timing=True);ae_end=torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(infer):
                        ae_begin.record();actions,raw,_=backend.sample(c,seed,10);ae_end.record()
                    action_ms=(time.perf_counter()-tick)*1000
                    begin_execution=time.perf_counter()
                    if execution_ms:time.sleep(execution_ms/1000)
                    actual_execution_ms=(time.perf_counter()-begin_execution)*1000
                    ready_before_wait=graph.end.query();result=graph.collect();ae_end.synchronize()
                    total_ms=(time.perf_counter()-tick)*1000
                    g0=prefix.elapsed_time(graph.begin);g1=prefix.elapsed_time(graph.end)
                    a0=prefix.elapsed_time(ae_begin);a1=prefix.elapsed_time(ae_end)
                    raw_array=raw.float().cpu().numpy()
                    check=dict(sample=index,q=q,priority=priority,execution_ms=execution_ms,
                        native_raw_max_abs=float(np.max(np.abs(raw_array-reference_raw))),
                        native_actions_max_abs=float(np.max(np.abs(actions-reference_actions))),
                        gradient_max_abs=float(np.max(np.abs(result['gradient']-reference_gradient))),
                        native_passed=bool(np.allclose(raw_array,reference_raw,atol=1e-5,rtol=1e-5) and np.allclose(actions,reference_actions,atol=1e-5,rtol=1e-5)),
                        gradient_passed=bool(np.allclose(result['gradient'],reference_gradient,atol=1e-4,rtol=1e-3)),
                        mask_passed=result['mask']==reference_mask)
                    check['passed']=all(check[k] for k in ('native_passed','gradient_passed','mask_passed'))
                    row=dict(sample=index,q=q,priority=priority,execution_ms=execution_ms,
                        actual_execution_ms=actual_execution_ms,action_ready_ms=action_ms,
                        next_fast_ready_ms=total_ms,critical_compute_ms=total_ms-actual_execution_ms,
                        selector_wait_ms=result['wait_ms'],selector_ready_before_wait=ready_before_wait,
                        selector_standalone_ms=selector_ms,native_ae_standalone_ms=native_ms,
                        serial_compute_ms=selector_ms+native_ms,
                        graph_gpu_ms=result['gpu_ms'],ae_event_ms=a1-a0,
                        graph_finished_ms=g1,native_finished_ms=a1,
                        interval_overlap_ms=max(0.,min(g1,a1)-max(g0,a0)))
                    if record:rows.append(row);checks.append(check)
                    return row

                # First call is warm-up for the stream priority, never a hot row.
                for priority in ('default','high'):
                    call(priority,0,False)
                    for execution_ms in (0,100,250,500):
                        for repeat in range(2):
                            call(priority,execution_ms)
                if index==0 and q==4:trace_call(lambda:call('high',0,False),output)
                write_csv(output/'timings.csv',rows);write_csv(output/'preparation.csv',preparation)
                atomic(output/'checks.json',dict(checks=checks,passed=all(c['passed'] for c in checks)))
                print(json.dumps(dict(sample=index,q=q,rows=len(rows),passed=all(c['passed'] for c in checks))),flush=True)
                graph.close();attention('original');gc.collect();torch.cuda.empty_cache()
        summary=[]
        for q in (4,2,1):
            for priority in ('default','high'):
                for execution_ms in (0,100,250,500):
                    rr=[r for r in rows if (r['q'],r['priority'],r['execution_ms'])==(q,priority,execution_ms)]
                    summary.append(dict(q=q,priority=priority,execution_ms=execution_ms,n=len(rr),
                        **{key:float(np.mean([r[key] for r in rr])) for key in ('action_ready_ms','critical_compute_ms','selector_wait_ms','serial_compute_ms','graph_gpu_ms','interval_overlap_ms')},
                        fully_hidden=sum(r['selector_ready_before_wait'] for r in rr)))
        write_csv(output/'summary.csv',summary)
        atomic(output/'complete.json',dict(passed=all(c['passed'] for c in checks),rows=len(rows),time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--uuid',required=True);args=p.parse_args()
    try:main(args)
    except Exception:atomic(OUT/'overlap/failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
