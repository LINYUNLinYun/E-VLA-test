"""Bounded, isolated optimization probe; original mechanism implementation frozen."""
import argparse,gc,json,time,hashlib,os
from pathlib import Path
import numpy as np
import torch
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm
from preparations.idea1_v3_4_E2_E4_single.probe import setup
from preparations.idea1_v3_4_E2_E4_single.export import key
from preparations.idea1_v3_4_E2_single_action.common import OLD,ROOT,weights,mask,write_csv,dump

OUT=ROOT/'results/idea1_v3_4_E2_backward_opt'

def initialize():
    setup();model=tm.Model(OLD/'model.pt')
    for k,v in model.w.items():model.w[k]=v.to(torch.bfloat16)
    for k in ('biases','outw','outb'):setattr(model,k,[v.to(torch.bfloat16) for v in getattr(model,k)])
    torch.cuda.empty_cache()
    tm.rms=lambda x:(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
    return model

def pairs():
    return [p for p in json.loads((OLD/'selection.json').read_text()) if p['e2'] and p['case']['task'] in (0,2)]

def load_item(model,p):
    item=model.item(OLD/'inputs'/f"{key(p['current'])}.pt")
    for k in ('x','state','encoder_rope','decoder_rope','noise'):item[k]=item[k].to(torch.bfloat16)
    return item

def calculate(model,item,R,gates=None):
    if gates is None:gates=torch.ones(17,device='cuda',requires_grad=True)
    raw=model.forward(item,gates=gates)
    loss=(R*raw[0,:7].float()).sum()
    grad=torch.autograd.grad(loss,gates)[0]
    return raw.detach(),grad.detach()

def finish(raw,grad):
    # Same host outputs and constrained selector as prior latency experiment.
    grad_cpu=grad.float().cpu().numpy()
    selected=mask(np.abs(grad_cpu))
    raw_cpu=raw.float().cpu().numpy()
    return raw_cpu,grad_cpu,selected

def timed(fn):
    torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize()
    return out,(time.perf_counter()-t)*1000

def check(ref,got,item):
    a,b=ref,got;d=a[0]-b[0];g=a[1]-b[1]
    raw_ok=bool(np.allclose(a[0],b[0],atol=1e-5,rtol=1e-5))
    grad_ok=bool(np.allclose(a[1],b[1],atol=1e-4,rtol=1e-3))
    mask_ok=bool(np.array_equal(a[2],b[2]))
    physical=d[:5,:7]@item['decode_matrix'].float().cpu().numpy()
    return dict(raw_ok=raw_ok,gradient_ok=grad_ok,mask_ok=mask_ok,passed=raw_ok and grad_ok and mask_ok,
                raw_max_abs=float(abs(d).max()),prefix7_rms=float(np.sqrt(np.mean(d[:5,:7]**2))),
                gradient_max_abs=float(abs(g).max()),gradient_relative_error=float(np.linalg.norm(g)/max(np.linalg.norm(a[1]),1e-12)),
                translation_max_mm=float(np.linalg.norm(physical[:,:3],axis=1).max()*1000),
                mask_reference=(a[2]+1).tolist(),mask_got=(b[2]+1).tolist())

class Graphed:
    def __init__(self,model,item,R):
        t=time.perf_counter()
        self.item={k:v.clone() if torch.is_tensor(v) else v for k,v in item.items()}
        self.gates=torch.ones(17,device='cuda',requires_grad=True)
        self.stream=torch.cuda.Stream();self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(3):calculate(model,self.item,R,self.gates)
        torch.cuda.current_stream().wait_stream(self.stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=self.stream):
            self.raw,self.grad=calculate(model,self.item,R,self.gates)
        torch.cuda.synchronize();self.prepare_ms=(time.perf_counter()-t)*1000
    def __call__(self,item):
        for k in ('x','state','encoder_rope','decoder_rope','noise'):self.item[k].copy_(item[k])
        self.graph.replay()
        return finish(self.raw,self.grad)

def profiling(model,item,R):
    for _ in range(3):finish(*calculate(model,item,R))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function('forward_plus_one_vjp'):calculate(model,item,R)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(OUT/'eager_trace.json'))
    (OUT/'profiler.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
    events=[]
    for e in prof.events():
        if e.device_type==torch.autograd.DeviceType.CUDA:
            events.append(dict(name=e.name,start_us=e.time_range.start,end_us=e.time_range.end,duration_us=e.time_range.elapsed_us()))
    dump(OUT/'cuda_events.json',events)
    print((OUT/'profiler.txt').read_text(),flush=True)

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=8);ap.add_argument('--reps',type=int,default=10);ap.add_argument('--profile',action='store_true');ap.add_argument('--deterministic',action='store_true');ap.add_argument('--out',default=str(OUT));args=ap.parse_args()
    OUT=Path(args.out)
    if args.deterministic:torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True)
    model=initialize();R=torch.as_tensor(weights()[0],device='cuda');rows=[];checks=[];prepares=[]
    dump(OUT/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),visible=os.environ.get('CUDA_VISIBLE_DEVICES'),deterministic=args.deterministic,time=time.time()))
    for index,p in enumerate(pairs()[:args.limit]):
        item=load_item(model,p)
        for _ in range(5):finish(*calculate(model,item,R))
        ref=finish(*calculate(model,item,R))
        if args.profile and index==0:profiling(model,item,R)
        graph=Graphed(model,item,R)
        prepares.append(dict(id=p['id'],shape=list(item['x'].shape),prepare_ms=graph.prepare_ms,memory_mb=torch.cuda.memory_allocated()/2**20))
        print(json.dumps(prepares[-1]),flush=True)
        # Fresh noise of the same actual shape detects stale input/output buffers.
        changed=dict(item);changed['noise']=item['noise'].roll(1,0)
        changed_ref=finish(*calculate(model,changed,R));got=graph(changed)
        checks.append(dict(id=p['id'],test='changed_noise',**check(changed_ref,got,changed)))
        checks.append(dict(id=p['id'],test='restore_input',**check(ref,graph(item),item)))
        for rep in range(args.reps):
            methods=['eager','graph'] if rep%2==0 else ['graph','eager']
            for method in methods:
                got,ms=timed(lambda:finish(*calculate(model,item,R)) if method=='eager' else graph(item))
                rows.append(dict(id=p['id'],**p['case'],method=method,rep=rep,total_ms=ms))
                if rep==0:checks.append(dict(id=p['id'],test=method,**check(ref,got,item)))
        write_csv(OUT/'latency.csv',rows);dump(OUT/'checks.json',checks);dump(OUT/'preparation.json',prepares)
        print(json.dumps(dict(completed=p['id'],checks_passed=all(c['passed'] for c in checks),means={m:float(np.mean([x['total_ms'] for x in rows if x['id']==p['id'] and x['method']==m])) for m in ['eager','graph']})),flush=True)
        del graph,item,ref,changed,changed_ref,got;gc.collect();torch.cuda.empty_cache()
    summary={m:dict(n=len(v),mean_ms=float(np.mean(v)),p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95))) for m in ['eager','graph'] if (v:=[r['total_ms'] for r in rows if r['method']==m])}
    dump(OUT/'summary.json',dict(summary=summary,passed=all(c['passed'] for c in checks),time=time.time()))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
