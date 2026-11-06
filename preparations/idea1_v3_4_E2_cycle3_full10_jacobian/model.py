"""Full-depth D10 differentiable replay. Frozen old D2 implementation untouched."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm
from preparations.idea1_v3_4_E2_online_time_weights.selector import Selector as Base,current_item as old_item,calculate
from preparations.idea1_v2_2.schedules import flow_schedule
from preparations.idea1_v3_4_E2_cycle3.common import choose

def current_item(c,seed):
    # Native Full updated the entire encoder_x, including language rows. The
    # projector only overwrites vision rows, so restore the original language
    # embedding before rebuilding exactly the input used to start Full.
    runtime=c['runtime']
    runtime.buffers['encoder_x'][512:].copy_(runtime.weights['language_embeds'])
    return old_item(c,seed)

class Full10Model(tm.Model):
    def sample(self,item,c):
        assert len(self.biases)==10 and len(self.outw)==10 and len(self.outb)==10
        w=self.w;state=item['state']@w['decoder_state_in_proj_w']+w['decoder_state_in_proj_b'];z=item['noise']
        seq=len(item['x']);r=item['decoder_rope'];mask=torch.ones(51,seq+51,dtype=torch.bool,device=z.device);mask[0,seq+1:]=False
        for t in range(10):
            a=F.silu(z@w['decoder_action_fused_in_proj_w']+self.biases[t])
            x=torch.cat((state.reshape(1,-1),a@w['decoder_action_mlp_w']+w['decoder_action_mlp_b']),0)
            for i in range(18):
                proj=tm.rms(x)@w['decoder_attn_qkv_w'][i];q,k,v=proj.split((2048,256,256),-1)
                q=tm.rope(q,r);k=tm.rope(k,r)
                x=x+tm.attend(q,torch.cat((c['K'][i],k)),torch.cat((c['V'][i],v)),mask)@w['decoder_attn_o_w'][i]
                n=tm.rms(x);h=F.gelu(n@w['decoder_ffn_gate_w'][i],approximate='tanh')*(n@w['decoder_ffn_up_w'][i])
                x=x+h@w['decoder_ffn_down_w'][i]
            z=z+tm.rms(x[1:])@self.outw[t]+self.outb[t]
        return z

def borrow(backend,c):
    model=Full10Model.__new__(Full10Model);model.device='cuda'
    model.w={k:v for k,v in c['runtime'].weights.items() if k.startswith(('encoder_attn','encoder_ffn','decoder_'))}
    model.biases,model.outw,model.outb,_=backend.adapter.schedule(c['runtime'],flow_schedule('pi0',None,10))
    return model

class Selector(Base):
    def capture(self,c,item,W):
        tick=time.perf_counter();self.model=borrow(self.backend,c);self.runtime_id=id(c['runtime'])
        self.item={k:v.clone() for k,v in item.items()};self.W=W.clone();self.g=torch.ones(17,device='cuda',requires_grad=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):calculate(self.model,self.item,self.W,self.g)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.raw,self.grad=calculate(self.model,self.item,self.W,self.g)
        torch.cuda.synchronize();return (time.perf_counter()-tick)*1000
    def run(self,c,seed,coeff,k=4):
        tick=time.perf_counter();item=current_item(c,seed);W=torch.as_tensor(coeff,device='cuda');cold=0.
        if self.graph is None or self.runtime_id!=id(c['runtime']):
            if self.graph is not None:self.clear()
            cold=self.capture(c,item,W)
        for name,value in item.items():self.item[name].copy_(value)
        self.W.copy_(W);self.graph.replay();grad=self.grad.float().cpu().numpy().copy();assert np.isfinite(grad).all()
        score=abs(grad);skip=choose(score,k);torch.cuda.synchronize();total=(time.perf_counter()-tick)*1000
        return skip,grad,score,dict(selector_total_ms=total,selector_hot_ms=total-cold,selector_capture_ms=cold)
