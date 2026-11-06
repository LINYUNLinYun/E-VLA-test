"""Original residual gates, one Full-D10 forward and four reverse passes."""
import time
import numpy as np
import torch
from preparations.idea1_v3_4_E2_online_time_weights.selector import Selector as Base
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import borrow,current_item
from preparations.idea1_v3_4_E2_comp_error_multivjp.model import calculate,projection_weights,PROJECTION_SEED
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3_4_E2_adjacency_200.common import choose

class Selector(Base):
    q=4
    rule='spaced'
    def capture(self,c,item,W):
        tick=time.perf_counter();self.model=borrow(self.backend,c);self.runtime_id=id(c['runtime'])
        self.item={k:v.clone() for k,v in item.items()};self.W=W.clone();self.g=torch.ones(17,device='cuda',requires_grad=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):calculate(self.model,self.item,self.W,self.g)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.raw,self.grad=calculate(self.model,self.item,self.W,self.g)
        torch.cuda.synchronize();return (time.perf_counter()-tick)*1000
    def run(self,c,seed,coeff,k=6):
        assert np.array_equal(coeff,coefficients()) and self.q in (1,4)
        tick=time.perf_counter();item=current_item(c,seed);W=torch.as_tensor(projection_weights(self.q),device='cuda');cold=0.
        if self.graph is None or self.runtime_id!=id(c['runtime']) or self.W.shape!=W.shape:
            if self.graph is not None:self.clear()
            cold=self.capture(c,item,W)
        for name,value in item.items():self.item[name].copy_(value)
        self.W.copy_(W);self.graph.replay();grads=self.grad.float().cpu().numpy().copy();assert np.isfinite(grads).all()
        score=abs(grads[0]) if self.q==1 else np.sqrt(np.mean(np.square(grads),axis=0))
        skip=choose(score,k,'spaced');torch.cuda.synchronize();total=(time.perf_counter()-tick)*1000
        return skip,grads,score,dict(selector_total_ms=total,selector_hot_ms=total-cold,selector_capture_ms=cold)
