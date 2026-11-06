"""Batch four reverse seeds without changing their weights or target action."""
import gc,json,time,traceback
import numpy as np
import torch
from . import probe as p
from .advanced import finish,multigrad

OUT=p.OUT/'batched'

def calc(model,item,R,batched=True,gates=None):
    if gates is None:gates=torch.ones(17,device='cuda',requires_grad=True)
    raw=model.forward(item,gates=gates)
    if batched:
        G=torch.autograd.grad(raw[0,:7].float(),gates,grad_outputs=R,is_grads_batched=True)[0]
    else:G=multigrad(raw,gates,R)
    return raw.detach(),G.detach()

class Graph:
    def __init__(self,model,item,R):
        self.g=torch.ones(17,device='cuda',requires_grad=True)
        s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):calc(model,item,R,gates=self.g)
        torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=s):self.raw,self.G=calc(model,item,R,gates=self.g)
        torch.cuda.synchronize()
    def __call__(self):self.graph.replay();return finish(self.raw,self.G)

def main():
    torch.use_deterministic_algorithms(True);OUT.mkdir(parents=True,exist_ok=True)
    model=p.initialize();R=torch.as_tensor(p.weights(),device='cuda');rows=[];checks=[];prep=[]
    torch._C._debug_only_display_vmap_fallback_warnings(True)
    for pair in p.pairs():
        item=p.load_item(model,pair);ref=finish(*calc(model,item,R,False));t=time.perf_counter()
        graph=Graph(model,item,R);prep.append(dict(id=pair['id'],prepare_ms=(time.perf_counter()-t)*1000,memory_mb=torch.cuda.memory_allocated()/2**20))
        got=graph();checks.append(dict(id=pair['id'],**p.check(ref,got,item)))
        # Update a captured input in place and compare to a fresh serial reference.
        item['noise'].copy_(item['noise'].roll(1,0));changed=finish(*calc(model,item,R,False))
        checks.append(dict(id=pair['id'],test='changed_noise',**p.check(changed,graph(),item)))
        item['noise'].copy_(item['noise'].roll(-1,0))
        for _ in range(5):graph()
        for rep in range(10):
            got,ms=p.timed(graph);rows.append(dict(id=pair['id'],rep=rep,total_ms=ms))
        p.dump(OUT/'checks.json',checks);p.dump(OUT/'preparation.json',prep);p.write_csv(OUT/'latency.csv',rows)
        print(json.dumps(dict(**prep[-1],total_ms=float(np.mean([v['total_ms'] for v in rows if v['id']==pair['id']])),passed=all(c['passed'] for c in checks))),flush=True)
        del graph,item,ref,got,changed;gc.collect();torch.cuda.empty_cache()
    v=[r['total_ms'] for r in rows]
    p.dump(OUT/'summary.json',dict(n=len(v),mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),passed=all(c['passed'] for c in checks)))

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);p.dump(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
