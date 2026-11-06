"""Technical timing supplement; original scientific implementation stays frozen."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,MASK
from preparations.idea1_v3.io import atomic

def run(gpu):
    from preparations.idea1_v3.safe_qk import install
    from .server import gate,selection
    from .backend import Backend
    from .controller import cross_gate
    from preparations.kv_reuse_probe.probe import load_observation
    out=OUT/'workers'/f'gpu{gpu}'/'warm_timing';out.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter();install(str(out/'qk_receipt.json'));b=Backend();gate(b,out);cross_gate(out)
    load=time.perf_counter()-begin;begin=time.perf_counter();rows=[];warm_s=0.
    names=('full10','full10_capture','v14_d2','reuse','extrapolate5','extrapolate10')
    for index,p in enumerate(selection()):
        obs,prompt=load_observation(p['current']);seed=p['current']['seed'];hist=[]
        for which in ('previous2','previous'):
            oo,pp=load_observation(p[which]);c,_=b.prepare_residual(oo,pp,capture=True);hist.append(c['residuals'])
        ratio=(p['current']['step']-p['previous']['step'])/(p['previous']['step']-p['previous2']['step'])
        def forward(name):
            tick=time.perf_counter();full=name.startswith('full10')
            c,t=b.prepare_residual(obs,prompt,() if full else MASK,capture=name=='full10_capture',
                history=None if full or name=='v14_d2' else hist[1],
                older=hist[0] if name.startswith('extrapolate') else None,
                ratio=ratio*(2 if name=='extrapolate10' else 1))
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
        print(json.dumps(dict(gpu=gpu,completed_inputs=index+1)),flush=True)
    atomic(out/'benchmark.json',dict(rows=rows,inputs=40,repetitions=2,warmup_calls=1200,warmup_s=warm_s,
        wall_s=time.perf_counter()-begin,service_load_and_gate_s=load,selection=selection(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),history_source='fixed Full bank; timing only'))

def orchestrate():
    from .controller import free,MODEL
    from preparations.idea1_v3_1.common import environment,lock
    from .finish import main as finish,progress
    with lock(OUT/'warm_timing.lock',blocking=False):
        atomic(OUT/'warm_timing_process.json',dict(pid=os.getpid(),time=time.time()))
        while not (OUT/'complete.json').exists():
            if (OUT/'failed.json').exists():raise RuntimeError('Closed-loop controller failed')
            time.sleep(15)
        original_report=ROOT/'docs/experiments/idea1/idea1_v3_4_E4_closedloop_report.md'
        (OUT/'report_before_warm_timing.md').write_text(original_report.read_text())
        atomic(OUT/'status.json',dict(status='warm_timing_supplement',progress=progress(),time=time.time()))
        def worker(gpu):
            while not free(gpu):time.sleep(30)
            wd=OUT/'workers'/f'gpu{gpu}'
            env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
            with (wd/'warm_timing.log').open('a') as log:
                result=subprocess.run([str(MODEL),'-u','-m','preparations.idea1_v3_4_E4_closedloop.warm_timing','--gpu',str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Warm timing failed GPU{gpu}')
            supplement=json.loads((wd/'warm_timing/benchmark.json').read_text())
            if not (wd/'startup_mixed_benchmark.json').exists():(wd/'startup_mixed_benchmark.json').write_bytes((wd/'benchmark.json').read_bytes())
            atomic(wd/'benchmark.json',supplement)
        with concurrent.futures.ThreadPoolExecutor(3) as ex:
            futures=[ex.submit(worker,g) for g in (0,2,3)]
            for f in futures:f.result()
        finish()
        extra=['','热态计时技术补充：每个固定输入的每条路径先预热5次，再进行AB/BA两遍测量；首轮混有编译的记录保留在各worker的startup_mixed_benchmark.json。补充不改变闭环实现或科学条件。']
        for gpu in (0,2,3):
            b=json.loads((OUT/'workers'/f'gpu{gpu}'/'benchmark.json').read_text())
            extra.append(f"- GPU{gpu}补充准备：加载及gate {b['service_load_and_gate_s']:.2f}s，逐输入预热 {b['warmup_s']:.2f}s；预热与测量总计 {b['wall_s']:.2f}s。")
        extra+=['','[计时补充方案](idea1_v3_4_E4_closedloop_timing_supplement.md)']
        original_report.write_text(original_report.read_text()+'\n'.join(extra)+'\n')
        atomic(OUT/'warm_timing_complete.json',dict(passed=True,gpus=[0,2,3],time=time.time(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int);a=p.parse_args()
    try:run(a.gpu) if a.gpu is not None else orchestrate()
    except Exception:
        atomic(OUT/('warm_timing_failed.json' if a.gpu is None else f'warm_timing_gpu{a.gpu}_failed.json'),dict(error=traceback.format_exc(),time=time.time()));raise
