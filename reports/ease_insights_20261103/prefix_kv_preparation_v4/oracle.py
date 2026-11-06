"""Unlaunched full-storage masked-attention oracle for packed-prefix validation.

The reference computes every original QK column and native V product. Dropped
image logits are set to minus infinity before native softmax. This is reference
work,not a production fast path. No CUDA/model is created on import.
"""
from contextlib import contextmanager
import threading
from reports.ease_insights_20261103.prefix_kv_preparation_v1.layout_contract import plan


def require(value,message):
    if not value:raise ValueError(message)


class MaskedPrefixOracle:
    def __init__(self,native,bounded_qk):
        self.native=native;self.bounded_qk=bounded_qk
        self.original=native.matmul_k8_256_n_softmax_mask0
        self.lock=threading.Lock()

    @contextmanager
    def use(self,original_prefix,pattern):
        layout=plan(pattern,original_prefix)
        dropped=tuple(i for i in range(original_prefix) if i not in set(layout.keep_indices))
        require(self.lock.acquire(blocking=False),'Masked reference is serial and non-reentrant')
        audit=dict(reference_only=True,pattern=pattern,original_prefix=original_prefix,
            original_total_key_tokens=original_prefix+51,kept_prefix_indices=list(layout.keep_indices),
            dropped_prefix_indices=list(dropped),calls=[],original_helper_restored=False,
            scope='Full native key storage and QK/V work with dropped image logits masked;not a fast path or GPU equivalence claim')
        index_cache={};installed=False
        def wrapped(Q,K,out,encoder_seq_len,logits_out=None):
            import torch
            require(self.native.matmul_abT_scale is self.bounded_qk,'Bounded native QK identity changed')
            m,n,dim=Q.shape[0],K.shape[0],Q.shape[1]
            require(encoder_seq_len==original_prefix and dim==256 and K.shape==(original_prefix+51,256)
                and out.shape==(m,n) and m==408,'Masked reference requires full native prefix and408 queries')
            logits=out if logits_out is None else logits_out
            require(logits.shape==out.shape and all(t.is_contiguous() for t in (Q,K,out,logits)),'Masked reference requires contiguous native buffers')
            require(logits.dtype in (torch.bfloat16,torch.float32),'Unsupported native logits dtype')
            self.bounded_qk[(((m+31)//32)*((n+31)//32),)](Q,K,logits,m,n,dim,dim**-.5,
                STORE_BF16=logits.dtype==torch.bfloat16,BLOCK_SIZE_M=32,BLOCK_SIZE_N=32,BLOCK_SIZE_K=64)
            if dropped:
                device=logits.device
                if device not in index_cache:index_cache[device]=torch.tensor(dropped,dtype=torch.long,device=device)
                logits.index_fill_(1,index_cache[device],float('-inf'))
            self.native.softmax_kernel_mask0[((m+3)//4,)](logits,m,n,8,encoder_seq_len,out,BLOCK_SIZE_M=4,BLOCK_SIZE=1024)
            audit['calls'].append(dict(queries=m,actual_QK_key_tokens=n,masked_original_prefix_tokens=len(dropped),
                retained_original_prefix_tokens=layout.packed_prefix,logits_dtype=str(logits.dtype),native_decoder_state_mask_preserved=True))
        try:
            require(self.native.matmul_k8_256_n_softmax_mask0 is self.original,'Attention helper already changed')
            require(self.native.matmul_abT_scale is self.bounded_qk,'Unsafe or unbound QK kernel')
            self.native.matmul_k8_256_n_softmax_mask0=wrapped;installed=True
            yield audit
        finally:
            try:
                if installed:
                    lost=self.native.matmul_k8_256_n_softmax_mask0 is not wrapped
                    self.native.matmul_k8_256_n_softmax_mask0=self.original;audit['original_helper_restored']=True
                    if lost:raise RuntimeError('Masked-reference helper ownership changed;original restored')
            finally:self.lock.release()
