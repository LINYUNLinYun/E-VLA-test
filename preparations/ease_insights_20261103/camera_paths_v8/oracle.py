"""Reference only: explicit original-index logit masking plus native softmax."""
from contextlib import contextmanager
import threading

def require(v,msg):
    if not v:raise ValueError(msg)

class RoleReference:
    def __init__(self,native,bounded):
        self.native=native;self.bounded=bounded;self.original=native.matmul_k8_256_n_softmax_mask0;self.lock=threading.Lock()
    @contextmanager
    def use(self,prefix,kept):
        kept=list(kept)
        require(kept==sorted(set(kept)) and len(kept)==256 and sum(i<256 for i in kept)==128 and all(type(i) is int and 0<=i<512 for i in kept),'Reference needs explicit original128+128 visual indices')
        dropped=[i for i in range(512) if i not in set(kept)]
        require(self.lock.acquire(blocking=False),'Reference is serial')
        audit=dict(reference_only=True,kept_visual_indices=kept,original_prefix=prefix,calls=[],original_helper_restored=False)
        indices={};installed=False
        def wrapped(Q,K,out,encoder_seq_len,logits_out=None):
            import torch
            m,n,d=Q.shape[0],K.shape[0],Q.shape[1]
            require(encoder_seq_len==prefix and (m,n,d)==(408,prefix+51,256) and K.shape==(prefix+51,256) and out.shape==(m,n),'Reference requires full native storage')
            require(self.native.matmul_abT_scale is self.bounded,'Bounded QK identity changed')
            logits=out if logits_out is None else logits_out
            require(logits.shape==out.shape and all(t.is_contiguous() for t in (Q,K,out,logits)),'Reference requires contiguous native buffers')
            self.bounded[(((m+31)//32)*((n+31)//32),)](Q,K,logits,m,n,d,d**-.5,STORE_BF16=logits.dtype==torch.bfloat16,BLOCK_SIZE_M=32,BLOCK_SIZE_N=32,BLOCK_SIZE_K=64)
            if logits.device not in indices:indices[logits.device]=torch.tensor(dropped,dtype=torch.long,device=logits.device)
            logits.index_fill_(1,indices[logits.device],float('-inf'))
            self.native.softmax_kernel_mask0[((m+3)//4,)](logits,m,n,8,prefix,out,BLOCK_SIZE_M=4,BLOCK_SIZE=1024)
            audit['calls'].append(dict(queries=m,actual_QK_key_tokens=n,masked_original_visual_keys=256,all_text_retained=True))
        try:
            require(self.native.matmul_k8_256_n_softmax_mask0 is self.original,'Reference helper already changed')
            self.native.matmul_k8_256_n_softmax_mask0=wrapped;installed=True;yield audit
        finally:
            try:
                if installed:
                    lost=self.native.matmul_k8_256_n_softmax_mask0 is not wrapped
                    self.native.matmul_k8_256_n_softmax_mask0=self.original;audit['original_helper_restored']=True
                    if lost:raise RuntimeError('Reference helper changed; original restored')
            finally:self.lock.release()
