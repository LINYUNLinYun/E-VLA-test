"""V3-only decoder QK row-bound fix; frozen upstream kernels stay untouched.

Revision qk-row-bounds-v1. The only arithmetic change is zero-filled loads for
invalid M/N rows. Valid output (i,j) still reduces the same Q[i,k]*K[j,k], in
the same 64-wide tiles, scale and output dtype, with the original launch grid.
"""
import hashlib
from pathlib import Path

import triton
import triton.language as tl


@triton.jit
def bounded_matmul_abT_scale(q_ptr,k_ptr,out_ptr,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
    scale_factor:tl.constexpr,STORE_BF16:tl.constexpr=True,
    BLOCK_SIZE_M:tl.constexpr=32,BLOCK_SIZE_N:tl.constexpr=32,BLOCK_SIZE_K:tl.constexpr=64):
    pid=tl.program_id(0); psize=tl.num_programs(0)
    grid_m=triton.cdiv(M,BLOCK_SIZE_M); grid_n=triton.cdiv(N,BLOCK_SIZE_N)
    while pid<grid_m*grid_n:
        offs_i=(pid//grid_n)*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M)
        offs_j=(pid%grid_n)*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N)
        acc=tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),tl.float32)
        for k in range(0,K,BLOCK_SIZE_K):
            offs_k=k+tl.arange(0,BLOCK_SIZE_K)
            q=tl.load(q_ptr+offs_i[:,None]*K+offs_k[None,:],
                mask=(offs_i[:,None]<M)&(offs_k[None,:]<K),other=0)
            key=tl.load(k_ptr+offs_j[:,None]*K+offs_k[None,:],
                mask=(offs_j[:,None]<N)&(offs_k[None,:]<K),other=0)
            acc=tl.dot(q,tl.trans(key),acc)
        acc=acc*scale_factor
        if STORE_BF16: acc=acc.to(tl.bfloat16)
        tl.store(out_ptr+offs_i[:,None]*N+offs_j[None,:],acc,
            mask=(offs_i[:,None]<M)&(offs_j[None,:]<N))
        pid+=psize


def install(receipt_path):
    """Install only in a fresh isolated process; check its first real QK call."""
    import sys
    import torch
    from preparations.idea1_v3.io import atomic
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'policies/realtime-vla-flash'))
    from scripts.spec.triton import pi0_infer as kernels
    original=kernels.matmul_k8_256_n_softmax_mask0
    kernels.matmul_abT_scale=bounded_matmul_abT_scale
    checked=False
    revision=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def first_checked(Q,K,out,encoder_seq_len,logits_out=None):
        nonlocal checked
        if checked: return original(Q,K,out,encoder_seq_len,logits_out)
        if not Q.is_contiguous() or not K.is_contiguous() or not out.is_contiguous():
            raise ValueError('QK fix requires the native contiguous row-major interface')
        m,n,dim=Q.shape[0],K.shape[0],Q.shape[1]
        if dim!=256 or K.shape[1]!=dim or out.shape!=(m,n): raise ValueError('Unexpected native QK shape')
        logits=out if logits_out is None else logits_out
        bounded_matmul_abT_scale[(((m+31)//32)*((n+31)//32),)](Q,K,logits,m,n,dim,dim**-.5,
            STORE_BF16=logits.dtype==torch.bfloat16,BLOCK_SIZE_M=32,BLOCK_SIZE_N=32,BLOCK_SIZE_K=64)
        # One check within the already authorized failed episode's first request;
        # no extra policy rollout or unsafe legacy-kernel launch is performed.
        torch.cuda.synchronize()
        previous_tf32=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            reference=((Q.float()@K.float().T)*(dim**-.5)).to(logits.dtype)
        finally: torch.backends.cuda.matmul.allow_tf32=previous_tf32
        diff=(logits.float()-reference.float()).abs()
        # Fixed before GPU execution: one BF16 epsilon relative, 1/1024 absolute
        # near zero. Report actual difference; never relax on observed failure.
        rtol=.0078125; atol=.0009765625
        passed=bool(torch.allclose(logits,reference,rtol=rtol,atol=atol))
        atomic(receipt_path,dict(revision='qk-row-bounds-v1',source_sha256=revision,
            Q_shape=list(Q.shape),K_shape=list(K.shape),output_shape=list(out.shape),
            encoder_seq_len=encoder_seq_len,max_abs=float(diff.max().item()),
            reference='torch FP32 matmul with TF32 disabled, scale then native output dtype',
            rtol=rtol,atol=atol,passed=passed,unsafe_legacy_kernel_executed=False,
            diagnostic_scope='one QK call inside first request of one failed-key recovery'))
        if not passed: raise RuntimeError('Bounded QK failed predefined BF16 reference tolerance; recovery stops')
        checked=True
        kernels.softmax_kernel_mask0[((m+3)//4,)](logits,m,n,8,encoder_seq_len,out,
            BLOCK_SIZE_M=4,BLOCK_SIZE=1024)
    kernels.matmul_k8_256_n_softmax_mask0=first_checked

