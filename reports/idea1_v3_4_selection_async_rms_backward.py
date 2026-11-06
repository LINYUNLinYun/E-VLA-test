"""Keep the original FP32 RMS forward; fuse its reverse formula only."""
import torch
import triton
import triton.language as tl
from preparations.idea1_v3_4_selection_async import core  # installs frozen FP32 RMS
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm

ORIGINAL_RMS=tm.rms
Q=4


@triton.jit
def backward_kernel(X,R,G,O,rows:tl.constexpr,width:tl.constexpr,
                    xs0:tl.constexpr,xs1:tl.constexpr,rs0:tl.constexpr,
                    gs0:tl.constexpr,gs1:tl.constexpr,gs2:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0);row=pid%rows;direction=pid//rows
    col=tl.arange(0,BLOCK);valid=col<width
    x=tl.load(X+row*xs0+col*xs1,valid,0).to(tl.float32)
    g=tl.load(G+direction*gs0+row*gs1+col*gs2,valid,0).to(tl.float32)
    inv=tl.load(R+row*rs0).to(tl.float32)
    dot=tl.sum(x*g,0)
    # The frozen expression calls x.float() twice. Autograd casts its two
    # branches back separately, then adds them in x's dtype. Preserve that
    # rounding boundary instead of adding both branches in FP32 first.
    direct=(g*inv).to(X.dtype.element_ty)
    indirect=(((-0.5*dot)*(inv*inv*inv))/width*(2*x)).to(X.dtype.element_ty)
    dx=direct.to(tl.float32)+indirect.to(tl.float32)
    tl.store(O+pid*width+col,dx,valid)


class RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        xf=x.float();inv=torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-6)
        ctx.save_for_backward(x,inv)
        return (xf*inv).to(x.dtype)

    @staticmethod
    def backward(ctx,gradient):
        x,inv=ctx.saved_tensors
        if not x.is_cuda:
            g=gradient.float();xf=x.float()
            direct=(g*inv).to(x.dtype)
            indirect=((-.5*(g*xf).sum(-1,keepdim=True))*inv.pow(3)/x.shape[-1]*(2*xf)).to(x.dtype)
            return direct+indirect
        assert x.ndim==2
        legacy=torch._C._functorch.is_legacy_batchedtensor(gradient)
        g=torch._remove_batch_dim(gradient,1,Q,0) if legacy else gradient[None]
        directions=Q if legacy else 1;rows,width=x.shape
        assert g.shape==(directions,rows,width)
        out=torch.empty((directions,rows,width),device=x.device,dtype=x.dtype)
        backward_kernel[(directions*rows,)](x,inv,g,out,rows,width,*x.stride(),inv.stride(0),*g.stride(),
            BLOCK=triton.next_power_of_2(width),num_warps=4 if width<=2048 else 8,enable_fp_fusion=False)
        return torch._add_batch_dim(out,0,1) if legacy else out[0]


def install():tm.rms=RMS.apply


def test_formula():
    torch.manual_seed(340610)
    for dtype in (torch.float32,torch.bfloat16):
        x=torch.randn(7,32,dtype=dtype,requires_grad=True)
        weight=torch.randn(4,7,32,dtype=dtype)
        expected=torch.autograd.grad(ORIGINAL_RMS(x),x,grad_outputs=weight,is_grads_batched=True)[0]
        got=torch.autograd.grad(RMS.apply(x),x,grad_outputs=weight,is_grads_batched=True)[0]
        torch.testing.assert_close(got,expected,atol=1e-4,rtol=1e-3)
        torch.testing.assert_close(RMS.apply(x),ORIGINAL_RMS(x),atol=0,rtol=0)
    return True


if __name__=='__main__':print(test_formula())
