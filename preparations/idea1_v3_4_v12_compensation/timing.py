import subprocess,time
import numpy as np
from .common import MASK,STATIC
from preparations.idea1_v3.io import atomic

def processes(gpu):
    rows=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return rows

def benchmark(b,out):
    from .server import selection
    from preparations.kv_reuse_probe.probe import load_observation
    gpu=int(out.name.removeprefix('gpu'));rows=[];samples=[];warm_s=0.;begin=time.perf_counter()
    names=('full10','full10_capture','v12_d2','channel','hadamard','linear','reuse','extrapolate5','extrapolate10')
    for index,p in enumerate(selection()):
        samples.append(dict(time=time.time(),id=p['id'],processes=processes(gpu)))
        obs,prompt=load_observation(p['current']);seed=p['current']['seed'];hist=[]
        for which in ('previous2','previous'):
            oo,pp=load_observation(p[which]);c,_=b.prepare_residual(oo,pp,capture=True);hist.append(c['residuals'])
        ratio=(p['current']['step']-p['previous']['step'])/(p['previous']['step']-p['previous2']['step'])
        def forward(name):
            tick=time.perf_counter();full=name.startswith('full10');is_history=name in ('reuse','extrapolate5','extrapolate10')
            c,t=b.prepare_residual(obs,prompt,() if full else MASK,capture=name=='full10_capture',
                kind=name if name in STATIC else None,history=hist[1] if is_history else None,
                older=hist[0] if name.startswith('extrapolate') else None,ratio=ratio*(2 if name=='extrapolate10' else 1))
            _,_,cost=b.sample(c,seed,10 if full else 2);b.torch.cuda.synchronize()
            return dict(policy_wall_ms=(time.perf_counter()-tick)*1000,**t,**cost)
        tick=time.perf_counter()
        for _ in range(5):
            for name in names:forward(name)
        warm_s+=time.perf_counter()-tick
        for repeat in range(2):
            offset=index%len(names);order=names[offset:]+names[:offset]
            if repeat:order=tuple(reversed(order))
            for name in order:rows.append(dict(id=p['id'],suite=p['case']['suite'],task=p['case']['task'],path=name,repeat=repeat,**forward(name)))
        samples.append(dict(time=time.time(),id=p['id'],processes=processes(gpu)))
        atomic(out/'timing_progress.json',dict(completed_inputs=index+1,target=40,time=time.time()))
    exclusive=all(len(r['processes'])==1 for r in samples)
    atomic(out/'benchmark.json',dict(rows=rows,inputs=40,repetitions=2,warmup_calls=1800,warmup_s=warm_s,
        wall_s=time.perf_counter()-begin,selection=selection(),resource_samples=samples,exclusive=exclusive,
        history_source='fixed Full bank; timing only'))
    return dict(measurements=len(rows),exclusive=exclusive)
