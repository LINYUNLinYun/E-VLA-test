import time
import numpy as np
import torch
from .common import choose
from preparations.idea1_v3_4_E2_online_time_weights.selector import Selector as Parent,current_item

class Selector(Parent):
    def run(self,c,seed,coeff,k=4):
        tick=time.perf_counter();item=current_item(c,seed);W=torch.as_tensor(coeff,device='cuda');cold=0.
        if self.graph is None or self.runtime_id!=id(c['runtime']):
            if self.graph is not None:self.clear()
            cold=self.capture(c,item,W)
        for name,value in item.items():self.item[name].copy_(value)
        self.W.copy_(W);self.graph.replay();grad=self.grad.float().cpu().numpy().copy();score=abs(grad);skip=choose(score,k);torch.cuda.synchronize();total=(time.perf_counter()-tick)*1000
        return skip,grad,score,dict(selector_total_ms=total,selector_hot_ms=total-cold,selector_capture_ms=cold)

def prepare_full(backend,obs,instruction,trace=False):
    if not trace:return backend.prepare(obs,instruction,())
    from scripts.spec.triton import pi0_infer as kernels
    original=kernels.transformer_encoder;hidden=[]
    def wrapped(*args,**kwargs):
        kwargs['hidden_trace']=hidden;return original(*args,**kwargs)
    kernels.transformer_encoder=wrapped
    try:c,t=backend.prepare(obs,instruction,())
    finally:kernels.transformer_encoder=original
    assert len(hidden)==19
    c['hidden_trace']=hidden;return c,t

def cosine_scores(hidden):
    with torch.no_grad():return torch.stack([1-torch.nn.functional.cosine_similarity(x.float(),y.float(),dim=-1).mean() for x,y in zip(hidden[:17],hidden[1:18])])
