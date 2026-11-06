"""Four-VJP forecast-error scores; keep the original first-Full score."""
import time
import numpy as np
import torch
from preparations.idea1_v3_4_E2_cycle5_q4_v12.model import Selector as OriginalSelector
from preparations.idea1_v3_4_E2_comp_error_cost.model import ErrorModel,forecast_error
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import borrow,current_item
from preparations.idea1_v3_4_E2_comp_error_multivjp.model import calculate,projection_weights
from preparations.idea1_v3_4_E2_adjacency_200.common import choose

class ErrorSelector(OriginalSelector):
    def capture(self,c,item,W):
        tick=time.perf_counter();self.model=ErrorModel(borrow(self.backend,c));self.runtime_id=id(c['runtime'])
        self.item={k:v.clone() for k,v in item.items()};self.W=W.clone();self.g=torch.ones(17,device='cuda',requires_grad=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):calculate(self.model,self.item,self.W,self.g)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.raw,self.grad=calculate(self.model,self.item,self.W,self.g)
        torch.cuda.synchronize();return (time.perf_counter()-tick)*1000
    def run(self,c,seed,coeff,error,k):
        tick=time.perf_counter();item=current_item(c,seed);item['gate_error']=error.detach();W=torch.as_tensor(projection_weights(4),device='cuda');cold=0.
        if self.graph is None or self.runtime_id!=id(c['runtime']):
            if self.graph is not None:self.clear()
            cold=self.capture(c,item,W)
        for name,value in item.items():self.item[name].copy_(value)
        self.W.copy_(W);self.graph.replay();g=self.grad.float().cpu().numpy().copy();assert np.isfinite(g).all()
        score=np.sqrt(np.mean(g*g,axis=0));skip=choose(score,k,'spaced');torch.cuda.synchronize();total=(time.perf_counter()-tick)*1000
        return skip,g,score,dict(selector_total_ms=total,selector_capture_ms=cold,selector_hot_ms=total-cold)

class CompensatedSelector:
    q=4;rule='spaced'
    def __init__(self,b,owner):
        self.backend=b;self.owner=owner;self.warm=OriginalSelector(b);self.error=ErrorSelector(b);self.clear()
    def clear(self):
        self.warm.clear();self.error.clear();self.history=[];self.error_buffer=None;self.record=None;self.error_rms=None
    def run(self,c,seed,coeff,k):
        assert self.q==4 and self.owner.history[-1]['step']==self.owner.step
        start=time.perf_counter();current=self.owner.history[-1];step=current['step'];history=self.history
        self.error_rms=None;build=transition=0.
        if not history:
            result=self.warm.run(c,seed,coeff,k);mode='warm_original';source=[];ratio=None
        else:
            if self.warm.graph is not None:
                tick=time.perf_counter();self.warm.clear();transition=(time.perf_counter()-tick)*1000
            last=history[-1];older=history[-2] if len(history)>1 else last
            ratio=(step-last['step'])/(last['step']-older['step']) if len(history)>1 else 0.
            mode='comp_error_extrapolate' if len(history)>1 else 'comp_error_hold'
            source=([older['step']] if len(history)>1 else [])+[last['step'],step]
            self.error_buffer,build=self.backend.adapter.measure(lambda:forecast_error(current['residuals'],last['residuals'],older['residuals'],ratio,out=self.error_buffer))
            result=self.error.run(c,seed,coeff,self.error_buffer,k)
            self.error_rms=self.error_buffer.square().mean((1,2)).sqrt().cpu().numpy()
        skip,g,score,t=result;total=(time.perf_counter()-start)*1000;capture=t['selector_capture_ms']+transition
        self.record=dict(scoring_mode=mode,score_source_full_steps=source,score_error_ratio=ratio,projection_seed=340610,score_aggregation='projection_rms',configured_projections=4)
        self.history=(history+[current])[-2:]
        return skip,g,score,dict(t,selector_total_ms=total,selector_capture_ms=capture,selector_hot_ms=total-capture,error_build_ms=build,selector_transition_ms=transition)
