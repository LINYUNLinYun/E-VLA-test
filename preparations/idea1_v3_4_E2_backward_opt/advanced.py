"""Split graphs, additional projections, and an explicit GPU contention probe."""
import gc,json,time
import numpy as np
import torch
from . import probe as p

OUT=p.OUT/'advanced'

def multigrad(raw,gates,R):
    values=R@raw[0,:7].float()
    return torch.stack([torch.autograd.grad(values[i],gates,retain_graph=i+1<len(R))[0] for i in range(len(R))])

def finish(raw,G):
    a=G.float().cpu().numpy();chosen=p.mask(np.linalg.norm(a,axis=0))
    return raw.detach().float().cpu().numpy(),a,chosen

def eager(model,item,R):
    gates=torch.ones(17,device='cuda',requires_grad=True)
    raw=model.forward(item,gates=gates)
    return finish(raw,multigrad(raw,gates,R))

class SplitGraph:
    def __init__(self,model,item,R):
        t=time.perf_counter()
        self.item={k:v.clone() if torch.is_tensor(v) else v for k,v in item.items()}
        self.gates=torch.ones(17,device='cuda',requires_grad=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):eager(model,self.item,R)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.fwd=torch.cuda.CUDAGraph();self.bwd=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.fwd,stream=stream):
            self.raw=model.forward(self.item,gates=self.gates)
        self.fwd.replay();torch.cuda.synchronize()
        with torch.cuda.graph(self.bwd,stream=stream):
            self.G=multigrad(self.raw,self.gates,R)
        torch.cuda.synchronize();self.prepare_ms=(time.perf_counter()-t)*1000
    def update(self,item):
        for k in ('x','state','encoder_rope','decoder_rope','noise'):self.item[k].copy_(item[k])
    def __call__(self,item):
        self.update(item);self.fwd.replay();self.bwd.replay()
        return finish(self.raw,self.G)

class InferenceGraph:
    def __init__(self,model,item):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):model.forward(item)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream),torch.no_grad():self.raw=model.forward(item)
        torch.cuda.synchronize()
    def __call__(self):self.graph.replay();return self.raw

def events(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();a.record();fn();b.record();b.synchronize();return a.elapsed_time(b)

def contention(model,item,graph):
    inference=InferenceGraph(model,item)
    stream=torch.cuda.Stream(priority=0);front=torch.cuda.Stream(priority=-1)
    rows=[]
    # Existing split graph has a separate activation pool from inference.
    # Previous request's backward may overlap the NEXT request's forward.
    for rep in range(20):
        for mode in (['serial','overlap'] if rep%2==0 else ['overlap','serial']):
            graph.update(item);graph.fwd.replay();torch.cuda.synchronize()
            s,b,f,done=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
            t=time.perf_counter()
            if mode=='serial':
                s.record();graph.bwd.replay();b.record();inference();f.record();done.record()
            else:
                with torch.cuda.stream(stream):
                    s.record();graph.bwd.replay();b.record()
                with torch.cuda.stream(front):
                    front.wait_event(s);inference();f.record()
                    front.wait_event(b);done.record()
            done.synchronize()
            rows.append(dict(rep=rep,mode=mode,backward_complete_ms=s.elapsed_time(b),next_forward_complete_ms=s.elapsed_time(f),pair_complete_ms=s.elapsed_time(done),wall_ms=(time.perf_counter()-t)*1000))
    standalone=dict(inference_ms=events(inference),backward_ms=events(graph.bwd.replay))
    p.write_csv(OUT/'contention.csv',rows);p.dump(OUT/'standalone.json',standalone)
    print(json.dumps(dict(standalone=standalone,contention={m:float(np.mean([r['pair_complete_ms'] for r in rows if r['mode']==m])) for m in ['serial','overlap']})),flush=True)
    del inference;gc.collect();torch.cuda.empty_cache()

def main():
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True);model=p.initialize()
    R=torch.as_tensor(p.weights(),device='cuda');rows=[];checks=[];prep=[]
    for index,pair in enumerate(p.pairs()):
        item=p.load_item(model,pair)
        for n in (1,4):
            ref=eager(model,item,R[:n]);graph=SplitGraph(model,item,R[:n]);got=graph(item)
            checks.append(dict(id=pair['id'],projections=n,**p.check(ref,got,item)))
            changed=dict(item);changed['noise']=item['noise'].roll(1,0)
            checks.append(dict(id=pair['id'],projections=n,test='changed_noise',**p.check(eager(model,changed,R[:n]),graph(changed),changed)))
            graph.update(item)
            prep.append(dict(id=pair['id'],projections=n,prepare_ms=graph.prepare_ms))
            for _ in range(5):graph(item)
            for rep in range(10):
                got,ms=p.timed(lambda:graph(item))
                rows.append(dict(id=pair['id'],**pair['case'],projections=n,method='split_graph',rep=rep,total_ms=ms))
            decomp=[]
            for _ in range(5):
                a,b,c=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
                torch.cuda.synchronize();a.record();graph.fwd.replay();b.record();graph.bwd.replay();c.record();c.synchronize()
                decomp.append(dict(forward_ms=a.elapsed_time(b),backward_ms=b.elapsed_time(c)))
            prep[-1]['forward_ms']=float(np.mean([v['forward_ms'] for v in decomp]));prep[-1]['backward_ms']=float(np.mean([v['backward_ms'] for v in decomp]))
            print(json.dumps(dict(**prep[-1],total_ms=float(np.mean([r['total_ms'] for r in rows if r['id']==pair['id'] and r['projections']==n])),passed=all(c['passed'] for c in checks))),flush=True)
            if index==0 and n==1:contention(model,item,graph)
            p.write_csv(OUT/'latency.csv',rows);p.dump(OUT/'checks.json',checks);p.dump(OUT/'preparation.json',prep)
            del graph,got,ref,changed;gc.collect();torch.cuda.empty_cache()
        del item;gc.collect();torch.cuda.empty_cache()
    summary={n:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),n=len(v)) for n in (1,4) if (v:=[r['total_ms'] for r in rows if r['projections']==n])}
    p.dump(OUT/'summary.json',dict(summary=summary,passed=all(c['passed'] for c in checks)))

if __name__=='__main__':main()
