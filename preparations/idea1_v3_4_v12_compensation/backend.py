import torch
import triton
import triton.language as tl
from preparations.idea1_v3.policy import TritonBackend
from .common import MASK,OUT,STATIC
from preparations.idea1_v3_4_E2_E4_single.torch_model import hadamard

def static_patch(x,kind,a,rank=32):
    z=x.float()
    if kind=='channel':return z*a['scale']+a['bias']
    if kind=='hadamard':return hadamard(hadamard(z)*a['scale']+a['bias'])
    if kind=='linear':return z+((z-a['input_mean'])@a['left'][:,:rank])@a['right'][:rank]+a['residual_mean']
    raise ValueError(kind)

@triton.jit
def difference(X,B,R,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<N,other=0).to(tl.float32)
    b=tl.load(B+i,i<N,other=0).to(tl.float32)
    tl.store(R+i,x-b,i<N)

@triton.jit
def patch_hidden(X,R,S,ratio,N:tl.constexpr,EXTRAP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<N,other=0).to(tl.float32)
    r=tl.load(R+i,i<N,other=0).to(tl.float32)
    if EXTRAP:
        s=tl.load(S+i,i<N,other=0).to(tl.float32)
        r=r+ratio*(r-s)
    tl.store(X+i,x+r,i<N)

def encoder(k,w,b,seq,skip=(),capture=False,history=None,older=None,ratio=0.,reference=False,static=None,rank=32,traced=()):
    residuals={}
    traces={}
    k.layer_norm_matmul_n256_1152_2048_bias(b['vision_x'],w['vision_final_norm_w'],w['vision_final_norm_b'],
        w['encoder_multi_modal_projector_w'],w['encoder_multi_modal_projector_b'],b['encoder_x'],b['vision_x_norm'])
    for i in range(18):
        layer=i+1;x=b['encoder_x']
        before=x.clone() if capture and layer in MASK else None
        if layer in traced:traces[layer]=dict(input=x.clone())
        if layer not in skip:
            k.rms_matmul_n_2048_2560_qkv_rope(x,w['encoder_attn_qkv_w'][i],b['encoder_rope_weights'],
                b['encoder_Q'],b['encoder_K'][i,:seq],b['encoder_V'][i,:seq],b['encoder_x_norm'])
        else:
            k.rms_matmul_n_2048_2560_kv_rope(x,w['encoder_attn_qkv_w'][i],b['encoder_rope_weights'],
                b['encoder_K'][i,:seq],b['encoder_V'][i,:seq],b['encoder_x_norm'])
        if i<17 and layer not in skip:
            attn=k.AttnSingleKey(b['encoder_Q'],b['encoder_K'][i,:seq],b['encoder_V'][i,:seq],256**-.5)
            k.matmul_n_2048_2048_res(attn,w['encoder_attn_o_w'][i],x)
            k.rms_matmul_n_2048_16384_gate(x,w['encoder_ffn_gate_w'][i],w['encoder_ffn_up_w'][i],b['encoder_hidden'],b['encoder_x_norm'])
            k.matmul_n_16384_2048_res(b['encoder_hidden'],w['encoder_ffn_down_w'][i],x)
        if before is not None:
            r=torch.empty_like(x,dtype=torch.float32)
            difference[(triton.cdiv(x.numel(),1024),)](x,before,r,x.numel(),1024)
            residuals[layer]=r
        if layer in skip and history is not None:
            r=history[layer];s=r if older is None else older[layer]
            if r.shape!=x.shape or s.shape!=x.shape:raise ValueError('Residual shape/prompt mismatch')
            if reference:
                estimate=r if older is None else r+ratio*(r-s)
                x.copy_((x.float()+estimate).to(x.dtype))
            else:
                patch_hidden[(triton.cdiv(x.numel(),1024),)](x,r,s,ratio,x.numel(),older is not None,1024,enable_fp_fusion=False)
        if layer in skip and static is not None and layer in static[1]:
            x.copy_(static_patch(x,static[0],static[1][layer],rank).to(x.dtype))
        if layer in traced:traces[layer]['output']=x.clone()
    return residuals,traces

class Backend(TritonBackend):
    def load_adapters(self):
        data=torch.load(OUT/'adapters.pt',map_location='cuda',weights_only=False)
        if tuple(data['mask'])!=MASK:raise ValueError('Adapter mask mismatch')
        self.adapters=data['methods'];self.rank=data['selected_rank']

    def prepare_residual(self,obs,instruction,skip=(),capture=False,history=None,older=None,ratio=0.,reference=False,kind=None,rank=None,static=None,traced=()):
        from scripts.spec.triton import pi0_infer as k
        original=k.transformer_encoder;captured={};traces={}
        if kind in STATIC:static=(kind,self.adapters[kind])
        if rank is None:rank=getattr(self,'rank',32)
        def wrapped(w,b,seq,**kwargs):
            residuals,tr=encoder(k,w,b,seq,skip,capture,history,older,ratio,reference,static,rank,traced)
            captured.update(residuals);traces.update(tr)
        k.transformer_encoder=wrapped
        try:c,t=super().prepare(obs,instruction,skip)
        finally:k.transformer_encoder=original
        if capture:c['residuals']=captured
        if traced:c['traces']=traces
        return c,t
