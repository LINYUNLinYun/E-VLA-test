"""Bounded regional pointwise compilation, reported separately from exact graphs."""
import gc,json,time,traceback,os
import numpy as np
import torch
from . import probe as p
from .advanced import SplitGraph,eager

PRESERVE=os.environ.get('TORCHINDUCTOR_EMULATE_PRECISION_CASTS')=='1'
OUT=p.OUT/('fusion_preserve' if PRESERVE else 'fusion')

def main():
    torch.use_deterministic_algorithms(True);OUT.mkdir(parents=True,exist_ok=True)
    model=p.initialize();R=torch.as_tensor(p.weights()[:1],device='cuda')
    cases=p.pairs();references={}
    # Fixed eager BF16 references are computed before any compilation patch.
    for pair in cases:
        item=p.load_item(model,pair);references[pair['id']]=eager(model,item,R)
        del item
    old_rms,old_rope=p.tm.rms,p.tm.rope
    options={'emulate_precision_casts':True} if PRESERVE else None
    p.tm.rms=torch.compile(old_rms,fullgraph=True,dynamic=True,options=options)
    p.tm.rope=torch.compile(old_rope,fullgraph=True,dynamic=True,options=options)
    rows=[];checks=[];prep=[]
    for pair in cases:
        item=p.load_item(model,pair);t=time.perf_counter()
        got=eager(model,item,R);compile_ms=(time.perf_counter()-t)*1000
        checks.append(dict(id=pair['id'],**p.check(references[pair['id']],got,item)))
        graph=SplitGraph(model,item,R)
        for _ in range(5):graph(item)
        for rep in range(10):
            got,ms=p.timed(lambda:graph(item));rows.append(dict(id=pair['id'],rep=rep,total_ms=ms))
        checks.append(dict(id=pair['id'],test='graph',**p.check(references[pair['id']],got,item)))
        prep.append(dict(id=pair['id'],compile_first_call_ms=compile_ms,graph_prepare_ms=graph.prepare_ms))
        p.dump(OUT/'checks.json',checks);p.dump(OUT/'preparation.json',prep);p.write_csv(OUT/'latency.csv',rows)
        print(json.dumps(dict(**prep[-1],passed=all(c['passed'] for c in checks),total_ms=float(np.mean([r['total_ms'] for r in rows if r['id']==pair['id']])))),flush=True)
        del graph,item,got;gc.collect();torch.cuda.empty_cache()
    v=[r['total_ms'] for r in rows]
    p.dump(OUT/'summary.json',dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),n=len(v),passed=all(c['passed'] for c in checks)))

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);p.dump(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
