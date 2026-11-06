"""Scoped physical batching of four SDPA reverse directions.

The legacy autograd vmap normally falls back to four attention-backward calls.
Here only that operator's batch axis is made physical, then wrapped back at the
same legacy level. This is private-API experimental code, not a default backend.
"""
import torch
import torch.nn.functional as F
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm

Q=4
SCALE=256**-.5


def physical(g):
    if not torch._C._functorch.is_legacy_batchedtensor(g):return g,None
    value=torch._remove_batch_dim(g,1,Q,0)
    if value.shape!=(Q,*g.shape):raise RuntimeError('This probe requires one legacy vmap level of four directions')
    return value.flatten(0,1),1


def repeated(value,level):return value if level is None else value.expand(Q,*value.shape[1:])


def logical(value,level):
    if value is None:return None
    return value if level is None else torch._add_batch_dim(value.reshape(Q,1,*value.shape[1:]),0,level)


class Efficient(torch.autograd.Function):
    backward_calls=0
    @staticmethod
    def forward(ctx,q,k,v,bias):
        out,lse,seed,offset=torch.ops.aten._scaled_dot_product_efficient_attention.default(q,k,v,bias,True,0.,False,scale=SCALE)
        ctx.save_for_backward(q,k,v,bias,out,lse,seed,offset);return out

    @staticmethod
    def backward(ctx,grad):
        Efficient.backward_calls+=1
        q,k,v,bias,out,lse,seed,offset=ctx.saved_tensors;go,level=physical(grad)
        flags=[*ctx.needs_input_grad[:3],False]
        result=torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(go,
            repeated(q,level),repeated(k,level),repeated(v,level),repeated(bias,level),
            repeated(out,level),repeated(lse,level).contiguous(),seed,offset,0.,flags,False,scale=SCALE)
        return tuple(logical(result[i],level) if flags[i] else None for i in range(3))+(None,)


class Flash(torch.autograd.Function):
    backward_calls=0
    @staticmethod
    def forward(ctx,q,k,v):
        out,lse,cq,ck,mq,mk,seed,offset,debug=torch.ops.aten._scaled_dot_product_flash_attention.default(q,k,v,0.,False,False,scale=SCALE)
        ctx.save_for_backward(q,k,v,out,lse,cq,ck,seed,offset);ctx.mq=mq;ctx.mk=mk;return out

    @staticmethod
    def backward(ctx,grad):
        Flash.backward_calls+=1
        q,k,v,out,lse,cq,ck,seed,offset=ctx.saved_tensors;go,level=physical(grad)
        result=torch.ops.aten._scaled_dot_product_flash_attention_backward.default(go,
            repeated(q,level),repeated(k,level),repeated(v,level),repeated(out,level),repeated(lse,level).contiguous(),
            cq,ck,ctx.mq,ctx.mk,0.,False,seed,offset,scale=SCALE)
        return tuple(logical(result[i],level) if ctx.needs_input_grad[i] else None for i in range(3))


def attend(q,k,v,mask=None,split=False):
    query=q.reshape(len(q),8,256).permute(1,0,2)[None]
    key=k[None,None].expand(1,8,-1,-1);value=v[None,None].expand(1,8,-1,-1)
    if mask is None:out=Flash.apply(query,key,value)
    elif split:
        assert len(q)==51 and mask.shape==(51,len(k));prefix=len(k)-50
        out=torch.cat((Flash.apply(query[:,:,:1],key[:,:,:prefix],value[:,:,:prefix]),
            Flash.apply(query[:,:,1:],key,value)),2)
    else:
        bias=torch.zeros_like(mask,dtype=query.dtype).masked_fill_(~mask,float('-inf'))
        padding=(-bias.shape[-1])%8
        if padding:bias=F.pad(bias,(0,padding))[...,:mask.shape[-1]]
        bias=bias[None,None].expand(1,8,-1,-1)
        out=Efficient.apply(query,key,value,bias)
    return out[0].permute(1,0,2).reshape(len(q),2048)


def install(split=False):
    from preparations.idea1_v3_4_selection_async import core
    def change(mode):
        assert mode=='original';tm.attend=lambda q,k,v,mask=None:attend(q,k,v,mask,split)
    core.attention=change;change('original')


def test_legacy_axis():
    class Small(torch.autograd.Function):
        @staticmethod
        def forward(ctx,q,k,v):
            p=(q@k.transpose(-1,-2)).softmax(-1);ctx.save_for_backward(q,k,v,p);return p@v
        @staticmethod
        def backward(ctx,g):
            q,k,v,p=ctx.saved_tensors;go,level=physical(g)
            dp=go@repeated(v,level).transpose(-1,-2);pp=repeated(p,level)
            ds=pp*(dp-(dp*pp).sum(-1,keepdim=True))
            return tuple(logical(x,level) for x in (ds@repeated(k,level),
                ds.transpose(-1,-2)@repeated(q,level),pp.transpose(-1,-2)@go))
    torch.manual_seed(340610)
    q=torch.randn(1,2,3,5,dtype=torch.float64,requires_grad=True)
    k=torch.randn(1,2,7,5,dtype=torch.float64,requires_grad=True)
    v=torch.randn(1,2,7,5,dtype=torch.float64,requires_grad=True)
    weights=torch.randn(4,1,2,3,5,dtype=torch.float64)
    ref=(q@k.transpose(-1,-2)).softmax(-1)@v
    expected=torch.autograd.grad(ref,(q,k,v),grad_outputs=weights,is_grads_batched=True)
    got=torch.autograd.grad(Small.apply(q,k,v),(q,k,v),grad_outputs=weights,is_grads_batched=True)
    for a,b in zip(expected,got):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    return True


if __name__=='__main__':print(test_legacy_axis())
