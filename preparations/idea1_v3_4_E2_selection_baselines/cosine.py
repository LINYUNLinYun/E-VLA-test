import gc,time
import numpy as np
import torch
from preparations.idea1_v3_4_E2_online_time_weights.selector import borrow_model,current_item
from preparations.idea1_v3_4_E2_single_action.common import mask

def calculate(model,item):
    with torch.no_grad():
        full=model.encode(item,trace=True)
        return torch.stack([1-torch.nn.functional.cosine_similarity(x.float(),y.float(),dim=-1).mean() for x,y in zip(full['inputs'][:17],full['outputs'][:17])])
class Cosine:
    def __init__(self,b):self.backend=b;self.clear()
    def clear(self):
        self.graph=None;self.model=None;self.item=None;self.score=None;self.runtime_id=None;gc.collect();torch.cuda.empty_cache()
    def capture(self,c,item):
        tick=time.perf_counter();self.model=borrow_model(self.backend,c);self.runtime_id=id(c['runtime']);self.item={k:v.clone() for k,v in item.items()}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):calculate(self.model,self.item)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.score=calculate(self.model,self.item)
        torch.cuda.synchronize();return (time.perf_counter()-tick)*1000
    def run(self,c,seed):
        tick=time.perf_counter();item=current_item(c,seed);cold=0.
        if self.graph is None or self.runtime_id!=id(c['runtime']):
            if self.graph is not None:self.clear()
            cold=self.capture(c,item)
        for k,v in item.items():self.item[k].copy_(v)
        self.graph.replay();score=self.score.cpu().numpy().copy();chosen=(mask(score)+1).tolist();torch.cuda.synchronize();total=(time.perf_counter()-tick)*1000
        return chosen,score,dict(selector_total_ms=total,selector_hot_ms=total-cold,selector_capture_ms=cold)
