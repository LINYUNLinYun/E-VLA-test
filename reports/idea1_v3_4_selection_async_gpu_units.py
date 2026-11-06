"""Independent operator checks before the final paced model service starts."""
import time
import torch
import torch.nn.functional as F
from preparations.idea1_v3.io import atomic
from reports import idea1_v3_4_selection_async_rms_backward as rms
from reports import idea1_v3_4_selection_async_elementwise as element
from reports import idea1_v3_4_selection_async_batched_attention as attention


def run(path):
    checks=[];tick=time.perf_counter()
    def compare(name,expected,got,inputs,dtype):
        # A noncontiguous upstream gradient also tests the physical-batch strides.
        weight=torch.randn((4,*expected.shape[:-2],expected.shape[-1],expected.shape[-2]),device='cuda',dtype=dtype).transpose(-1,-2)
        a=torch.autograd.grad(expected,inputs,grad_outputs=weight,is_grads_batched=True)
        b=torch.autograd.grad(got,inputs,grad_outputs=weight,is_grads_batched=True)
        row=dict(name=name,dtype=str(dtype),forward_max_abs=float((expected-got).abs().max()),
            forward_passed=bool(torch.allclose(expected,got,atol=1e-5,rtol=1e-5)),
            gradient_max_abs=max(float((x-y).abs().max()) for x,y in zip(a,b)),
            gradient_relative_l2=max(float((x-y).float().norm()/x.float().norm().clamp_min(1e-12)) for x,y in zip(a,b)),
            gradient_passed=all(bool(torch.allclose(x,y,atol=1e-4,rtol=1e-3)) for x,y in zip(a,b)))
        checks.append(row)
        # FP32 tests independently validate the derivative formulas. BF16 uses
        # the original strict comparison too; any rounding failure is recorded,
        # not hidden by a relaxed tolerance or called an equivalent replacement.
        if dtype==torch.float32:assert row['forward_passed'] and row['gradient_passed'],row
    with torch.enable_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(340610)
        for dtype in (torch.float32,torch.bfloat16):
            for width in (1024,2048):
                x=torch.randn(9,width,device='cuda',dtype=dtype,requires_grad=True)
                compare(f'rms_{width}',rms.ORIGINAL_RMS(x),rms.RMS.apply(x),(x,),dtype)
            x=torch.randn(9,2048,device='cuda',dtype=dtype,requires_grad=True)
            r=torch.randn(9,256,device='cuda',dtype=dtype)
            compare('rotary',element.ORIGINAL_ROPE(x,r),element.Rotary.apply(x,r),(x,),dtype)
            for width in (4096,16384):
                x=torch.randn(9,width,device='cuda',dtype=dtype,requires_grad=True)
                u=torch.randn_like(x,requires_grad=True)
                compare(f'glu_{width}',F.gelu(x,approximate='tanh')*u,element.GLU.apply(x,u),(x,u),dtype)
        q=torch.randn(1,8,7,256,device='cuda',requires_grad=True)
        k=torch.randn(1,8,13,256,device='cuda',requires_grad=True)
        v=torch.randn_like(k,requires_grad=True)
        bias=torch.zeros(1,8,7,16,device='cuda');bias[:,:,:1,-6:]=float('-inf');bias=bias[...,:13]
        expected=((q@k.transpose(-1,-2))*(256**-.5)+bias).softmax(-1)@v
        compare('physical_efficient_sdpa',expected,attention.Efficient.apply(q,k,v,bias),(q,k,v),torch.float32)
        q=q.detach().bfloat16().requires_grad_();k=k.detach().bfloat16().requires_grad_();v=v.detach().bfloat16().requires_grad_()
        compare('physical_flash_sdpa',F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=256**-.5),
            attention.Flash.apply(q,k,v),(q,k,v),torch.bfloat16)
    torch.cuda.synchronize()
    result=dict(fp32_formulas_passed=all(c['forward_passed'] and c['gradient_passed'] for c in checks if c['dtype']=='torch.float32'),
        checks=checks,wall_s=time.perf_counter()-tick,time=time.time())
    atomic(path,result);return result
