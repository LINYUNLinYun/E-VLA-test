import gc,time
import numpy as np
import torch
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm
from preparations.idea1_v3_4_E2_single_action.common import mask
from preparations.idea1_v2_2.schedules import flow_schedule

tm.rms=lambda x:(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)

def calculate(model,item,W,g):
    with torch.enable_grad():
        raw=model.forward(item,gates=g);loss=(raw[:5,:7].float()*W).sum()
        grad=torch.autograd.grad(loss,g)[0]
    return raw.detach(),grad.detach()

def borrow_model(backend,c):
    model=tm.Model.__new__(tm.Model);model.device='cuda'
    model.w={name:v for name,v in c['runtime'].weights.items() if name.startswith(('encoder_attn','encoder_ffn','decoder_'))}
    model.biases,model.outw,model.outb,_=backend.adapter.schedule(c['runtime'],flow_schedule('pi0',None,2))
    return model

def current_item(c,seed):
    from scripts.spec.triton import pi0_infer as k
    r=c['runtime'];w,b=r.weights,r.buffers
    k.layer_norm_matmul_n256_1152_2048_bias(b['vision_x'],w['vision_final_norm_w'],w['vision_final_norm_b'],w['encoder_multi_modal_projector_w'],w['encoder_multi_modal_projector_b'],b['encoder_x'],b['vision_x_norm'])
    return dict(x=b['encoder_x'].detach().clone(),state=c['state'].detach().to(torch.bfloat16),encoder_rope=b['encoder_rope_weights'],decoder_rope=b['decoder_rope_weights'],noise=torch.as_tensor(np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32),device='cuda',dtype=torch.bfloat16))

class Selector:
    def __init__(self,backend):self.backend=backend;self.clear()
    def clear(self):
        self.graph=None;self.model=None;self.item=None;self.raw=None;self.grad=None;self.W=None;self.g=None;self.runtime_id=None
        gc.collect();torch.cuda.empty_cache()
    def capture(self,c,item,W):
        tick=time.perf_counter();self.model=borrow_model(self.backend,c);self.runtime_id=id(c['runtime'])
        self.item={k:v.clone() for k,v in item.items()};self.W=W.clone();self.g=torch.ones(17,device='cuda',requires_grad=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):calculate(self.model,self.item,self.W,self.g)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.raw,self.grad=calculate(self.model,self.item,self.W,self.g)
        torch.cuda.synchronize();return (time.perf_counter()-tick)*1000
    def run(self,c,seed,coeff):
        tick=time.perf_counter();item=current_item(c,seed);W=torch.as_tensor(coeff,device='cuda');cold=0.
        if self.graph is None or self.runtime_id!=id(c['runtime']):
            if self.graph is not None:self.clear()
            cold=self.capture(c,item,W)
        for k,v in item.items():self.item[k].copy_(v)
        self.W.copy_(W);self.graph.replay()
        grad=self.grad.float().cpu().numpy().copy();score=abs(grad);chosen=mask(score)+1
        torch.cuda.synchronize();total=(time.perf_counter()-tick)*1000
        return chosen.tolist(),grad,score,dict(selector_total_ms=total,selector_capture_ms=cold,selector_hot_ms=total-cold)
