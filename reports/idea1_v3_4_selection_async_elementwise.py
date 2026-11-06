"""Preserve forward expressions; fuse GLU and rotary reverse kernels."""
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from preparations.idea1_v3_4_selection_async import core
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import Full10Model

ORIGINAL_ROPE=tm.rope
ORIGINAL_BLOCK=tm.Model.block
ORIGINAL_SAMPLE=Full10Model.sample
Q=4


def unbatch(g):
    legacy=torch._C._functorch.is_legacy_batchedtensor(g)
    return (torch._remove_batch_dim(g,1,Q,0) if legacy else g[None]),legacy


def wrap(g,legacy):return torch._add_batch_dim(g,0,1) if legacy else g[0]


@triton.jit
def glu_backward(X,U,A,G,DX,DU,rows:tl.constexpr,width:tl.constexpr,total:tl.constexpr,
                 gs0:tl.constexpr,gs1:tl.constexpr,gs2:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<total
    col=i%width;row=i//width%rows;direction=i//(rows*width);offset=row*width+col
    x=tl.load(X+offset,valid,0).to(tl.float32)
    u=tl.load(U+offset,valid,0).to(tl.float32)
    a=tl.load(A+offset,valid,0).to(tl.float32)
    g=tl.load(G+direction*gs0+row*gs1+col*gs2,valid,0).to(tl.float32)
    upstream=(g*u).to(X.dtype.element_ty).to(tl.float32)
    square=x*x;cube=square*x;inner=0.7978845608028654*(x+0.044715*cube)
    t=libdevice.tanh(inner)
    derivative=0.5*(1+t)+(0.5*x)*(1-t*t)*(0.7978845608028654*(1+3*0.044715*square))
    tl.store(DX+i,upstream*derivative,valid);tl.store(DU+i,g*a,valid)


class GLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,gate,up):
        activation=F.gelu(gate,approximate='tanh');ctx.save_for_backward(gate,up,activation)
        return activation*up

    @staticmethod
    def backward(ctx,gradient):
        gate,up,activation=ctx.saved_tensors
        if not gate.is_cuda:
            return torch.ops.aten.gelu_backward.default(gradient*up,gate,approximate='tanh'),gradient*activation
        assert gate.ndim==2 and gate.is_contiguous() and up.is_contiguous() and activation.is_contiguous()
        g,legacy=unbatch(gradient);directions=Q if legacy else 1;rows,width=gate.shape
        assert g.shape==(directions,rows,width)
        dx=torch.empty((directions,rows,width),device=gate.device,dtype=gate.dtype);du=torch.empty_like(dx)
        total=directions*rows*width
        glu_backward[(triton.cdiv(total,1024),)](gate,up,activation,g,dx,du,rows,width,total,*g.stride(),B=1024,
            num_warps=4,enable_fp_fusion=False)
        return wrap(dx,legacy),wrap(du,legacy)


@triton.jit
def rope_backward(R,G,O,rows:tl.constexpr,width:tl.constexpr,total_pairs:tl.constexpr,
                  gs0:tl.constexpr,gs1:tl.constexpr,gs2:tl.constexpr,B:tl.constexpr):
    p=tl.program_id(0)*B+tl.arange(0,B);valid=p<total_pairs
    col=p%(width//2)*2;row=p//(width//2)%rows;direction=p//(rows*(width//2))
    ro=row*256+(col%256)
    c=tl.load(R+ro,valid,0).to(tl.float32);s=tl.load(R+ro+1,valid,0).to(tl.float32)
    base=direction*gs0+row*gs1+col*gs2
    a=tl.load(G+base,valid,0).to(tl.float32);b=tl.load(G+base+gs2,valid,0).to(tl.float32)
    ac=(a*c).to(O.dtype.element_ty).to(tl.float32);bs=(b*s).to(O.dtype.element_ty).to(tl.float32)
    bc=(b*c).to(O.dtype.element_ty).to(tl.float32);as_=(a*s).to(O.dtype.element_ty).to(tl.float32)
    tl.store(O+2*p,ac+bs,valid);tl.store(O+2*p+1,bc-as_,valid)


class Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,r):
        assert x.ndim==2 and r.dtype==x.dtype and r.numel()==len(x)*256 and r.is_contiguous() and not r.requires_grad
        ctx.save_for_backward(r);ctx.shape=x.shape;return ORIGINAL_ROPE(x,r)

    @staticmethod
    def backward(ctx,gradient):
        r,=ctx.saved_tensors;rows,width=ctx.shape
        if not gradient.is_cuda:
            g=gradient.reshape(rows,-1,128,2);rr=r.reshape(rows,1,128,2)
            return torch.stack((g[...,0]*rr[...,0]+g[...,1]*rr[...,1],
                g[...,1]*rr[...,0]-g[...,0]*rr[...,1]),-1).reshape(ctx.shape),None
        g,legacy=unbatch(gradient);directions=Q if legacy else 1
        assert g.shape==(directions,rows,width)
        out=torch.empty((directions,rows,width),device=gradient.device,dtype=gradient.dtype)
        total=directions*rows*width//2
        rope_backward[(triton.cdiv(total,512),)](r,g,out,rows,width,total,*g.stride(),B=512,
            num_warps=4,enable_fp_fusion=False)
        return wrap(out,legacy),None


def block(self,x,i,r,update=True):
    w=self.w;z=tm.rms(x)@w['encoder_attn_qkv_w'][i]
    q,k,v=z.split((2048,256,256),-1);k=tm.rope(k,r)
    if not update:return x,k,v
    q=tm.rope(q,r);a=x+tm.attend(q,k,v)@w['encoder_attn_o_w'][i]
    n=tm.rms(a);f=GLU.apply(n@w['encoder_ffn_gate_w'][i],n@w['encoder_ffn_up_w'][i])
    return a+f@w['encoder_ffn_down_w'][i],k,v


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
            n=tm.rms(x);h=GLU.apply(n@w['decoder_ffn_gate_w'][i],n@w['decoder_ffn_up_w'][i])
            x=x+h@w['decoder_ffn_down_w'][i]
        z=z+tm.rms(x[1:])@self.outw[t]+self.outb[t]
    return z


def install(glu=True):
    tm.rope=Rotary.apply
    if glu:tm.Model.block=block;Full10Model.sample=sample


def restore():tm.rope=ORIGINAL_ROPE;tm.Model.block=ORIGINAL_BLOCK;Full10Model.sample=ORIGINAL_SAMPLE


def test_wiring():
    torch.manual_seed(340610)
    for dtype in (torch.float32,torch.bfloat16):
        x=torch.randn(3,256,dtype=dtype,requires_grad=True);r=torch.randn(3,256,dtype=dtype)
        g=torch.randn(4,3,256,dtype=dtype)
        a=torch.autograd.grad(ORIGINAL_ROPE(x,r),x,grad_outputs=g,is_grads_batched=True)[0]
        b=torch.autograd.grad(Rotary.apply(x,r),x,grad_outputs=g,is_grads_batched=True)[0]
        torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-3)
        up=torch.randn_like(x,requires_grad=True)
        a=torch.autograd.grad(F.gelu(x,approximate='tanh')*up,(x,up),grad_outputs=g,is_grads_batched=True)
        b=torch.autograd.grad(GLU.apply(x,up),(x,up),grad_outputs=g,is_grads_batched=True)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,atol=1e-4,rtol=1e-3)
    return True


if __name__=='__main__':print(test_wiring())
