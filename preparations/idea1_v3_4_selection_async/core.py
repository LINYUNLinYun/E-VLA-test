import time
import numpy as np
import torch
import torch.nn.functional as F
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import borrow,current_item
from preparations.idea1_v3_4_E2_comp_error_multivjp.model import projection_weights
from preparations.idea1_v3_4_E2_adjacency_200.common import combinations

ORIGINAL_ATTEND=tm.attend


def split_attend(q,k,v,mask=None):
    if mask is None:return ORIGINAL_ATTEND(q,k,v,None)
    # Full10Model uses precisely this fixed mask: state sees prefix + state,
    # action rows see all prefix/state/action keys. Split it into two calls.
    if len(q)!=51 or mask.shape!=(51,len(k)):
        raise ValueError('Split attention is scoped to the frozen Full-D10 mask')
    prefix_and_state=len(k)-51+1
    qq=q.reshape(len(q),8,256).permute(1,0,2)[None]
    kk=k[None,None].expand(1,8,-1,-1);vv=v[None,None].expand(1,8,-1,-1)
    state=F.scaled_dot_product_attention(qq[:,:,:1],kk[:,:,:prefix_and_state],
        vv[:,:,:prefix_and_state],dropout_p=0.,scale=256**-.5)
    actions=F.scaled_dot_product_attention(qq[:,:,1:],kk,vv,dropout_p=0.,scale=256**-.5)
    return torch.cat((state,actions),2)[0].permute(1,0,2).reshape(len(q),2048)


def attention(mode):
    if mode not in ('original','split_flash'):raise ValueError(mode)
    tm.attend=ORIGINAL_ATTEND if mode=='original' else split_attend


def calculate(model,item,W,g,batch=True):
    with torch.enable_grad():
        raw=model.forward(item,gates=g);values=(raw[:5,:7].float()[None]*W).sum((1,2));q=len(W)
        if q==1 or not batch:
            grad=torch.stack([torch.autograd.grad(values[i],g,retain_graph=i+1<q)[0] for i in range(q)])
        else:
            grad=torch.autograd.grad(values,g,grad_outputs=torch.eye(q,device=g.device),is_grads_batched=True)[0]
    return raw.detach(),grad.detach()


def choose(gradient,k,aggregation='layer_rms_sum'):
    g=np.asarray(gradient);masks=combinations(k,'spaced')
    assert g.ndim==2 and g.shape[1]==17 and np.isfinite(g).all()
    layers=np.sqrt(np.mean(g*g,axis=0))
    if aggregation=='layer_rms_sum':scores=layers[masks].sum(1)
    elif aggregation=='signed_sum_rms':scores=np.sqrt(np.mean(g[:,masks].sum(2)**2,axis=0))
    else:raise ValueError(aggregation)
    return (masks[np.argmin(scores)]+1).tolist(),layers,scores


class Graph:
    def __init__(self,backend,context,seed,q=4,mode='original',batch=True):
        self.backend=backend;self.q=q;self.mode=mode;self.batch=batch
        attention(mode);tick=time.perf_counter()
        self.model=borrow(backend,context)
        self.item={k:v.clone() for k,v in current_item(context,seed).items()}
        self.W=torch.as_tensor(projection_weights(q),device='cuda')
        self.g=torch.ones(17,device='cuda',requires_grad=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):calculate(self.model,self.item,self.W,self.g,batch)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.raw,self.gradient=calculate(self.model,self.item,self.W,self.g,batch)
        torch.cuda.synchronize();self.capture_s=time.perf_counter()-tick
        self.begin=torch.cuda.Event(enable_timing=True);self.end=torch.cuda.Event(enable_timing=True)
        self.pending=False

    def stage(self,context,seed):return current_item(context,seed)

    def launch(self,item,stream=None):
        if self.pending and not self.end.query():raise RuntimeError('Graph inputs still in use')
        source=torch.cuda.current_stream()
        target=source if stream is None else stream
        if target!=source:target.wait_stream(source)
        with torch.cuda.stream(target):
            for k,v in item.items():self.item[k].copy_(v)
            self.begin.record();self.graph.replay();self.end.record()
        self.pending=True

    def collect(self,k=6,aggregation='layer_rms_sum'):
        assert self.pending
        start=time.perf_counter();self.end.synchronize();wait_ms=(time.perf_counter()-start)*1000
        g=self.gradient.float().cpu().numpy().copy()
        tick=time.perf_counter();mask,layers,scores=choose(g,k,aggregation)
        self.pending=False
        return dict(gradient=g,mask=mask,layer_score=layers,combination_scores=scores,
            wait_ms=wait_ms,score_ms=(time.perf_counter()-tick)*1000,
            gpu_ms=float(self.begin.elapsed_time(self.end)))

    def run(self,context,seed,k=6,aggregation='layer_rms_sum'):
        torch.cuda.synchronize();tick=time.perf_counter()
        self.launch(self.stage(context,seed));result=self.collect(k,aggregation)
        result['wall_ms']=(time.perf_counter()-tick)*1000
        return result

    def close(self):
        if self.pending:self.end.synchronize()
        self.graph.reset();self.model=self.item=self.raw=self.gradient=self.graph=None


def compare(raw,gradient,reference_raw,reference_gradient):
    q=len(gradient);reference_gradient=reference_gradient[:q]
    raw=np.asarray(raw);g=np.asarray(gradient)
    masks={str(k):choose(g,k)[0] for k in (4,6)}
    expected={str(k):choose(reference_gradient,k)[0] for k in (4,6)}
    r=dict(raw_max_abs=float(np.max(np.abs(raw-reference_raw))),
        gradient_max_abs=float(np.max(np.abs(g-reference_gradient))),
        gradient_relative_l2=float(np.linalg.norm(g-reference_gradient)/max(np.linalg.norm(reference_gradient),1e-12)),
        raw_passed=bool(np.allclose(raw,reference_raw,atol=1e-5,rtol=1e-5)),
        gradient_passed=bool(np.allclose(g,reference_gradient,atol=1e-4,rtol=1e-3)),
        masks=masks,reference_masks=expected,masks_passed=masks==expected)
    r['passed']=r['raw_passed'] and r['gradient_passed'] and r['masks_passed']
    return r
