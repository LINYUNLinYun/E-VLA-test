"""Full native QK/V storage with selected image edges masked inside softmax."""
from contextlib import contextmanager
import threading
import triton
import triton.language as tl
from reports.ease_insights_20261103.prefix_kv_preparation_v1.layout_contract import plan


@triton.jit
def masked_softmax(inp,queries:tl.constexpr,keys:tl.constexpr,prefix:tl.constexpr,out,
                   pattern:tl.constexpr,BLOCK_SIZE_M:tl.constexpr=4,BLOCK_SIZE:tl.constexpr=1024):
    pid=tl.program_id(0);psize=tl.num_programs(0)
    for i in range(pid*BLOCK_SIZE_M,queries,psize*BLOCK_SIZE_M):
        rows=i+tl.arange(0,BLOCK_SIZE_M)[:,None];cols=tl.arange(0,BLOCK_SIZE)[None,:]
        column=cols%16;row=(cols//16)%16
        if pattern==0:visual=(column>=4)&(column<12)
        else:visual=(row+column)%2==0
        keep=(cols>=512)|visual
        mask=(rows<queries)&(cols<keys)&((rows>=8)|(cols<=prefix))&keep
        values=tl.load(inp+rows*keys+cols,mask=mask,other=-float('inf'))
        values=tl.exp(values-tl.max(values,axis=1,keep_dims=True))
        values=values/tl.sum(values,axis=1,keep_dims=True,dtype=tl.float32)
        tl.store(out+rows*keys+cols,values.to(tl.bfloat16),mask=(rows<queries)&(cols<keys))


def require(value,message):
    if not value:raise ValueError(message)


class MaskedAttention:
    def __init__(self,native,bounded):
        self.native=native;self.bounded=bounded;self.original=native.matmul_k8_256_n_softmax_mask0
        self.kernel=masked_softmax;self.lock=threading.Lock()

    @contextmanager
    def use(self,context,pattern):
        require(pattern in ('all','center_columns','checkerboard'),'Unknown fixed attention mask')
        prefix=context['seq'];layout=plan('identity' if pattern=='all' else pattern,prefix)
        runtime=context['runtime'];buffers=runtime.buffers;original=dict(buffers);calls=[]
        require(self.lock.acquire(blocking=False),'Attention mask is serial and non-reentrant');installed=False
        audit=dict(pattern=pattern,original_prefix=prefix,storage_prefix=prefix,actual_total_key_tokens=prefix+51,
            effective_visible_prefix_keys=layout.packed_prefix,kept_prefix_indices=list(layout.keep_indices),
            visual_keys_per_camera=[256,256] if pattern=='all' else [128,128],language_tokens_retained=prefix-512,
            decoder_absolute_positions=list(layout.decoder_absolute_positions),actual_QK_calls=0,actual_QK_shapes=[],
            original_buffer_references_preserved=False,original_rope_table_reused=True,original_helper_restored=False,
            no_compaction=True,prefix_compute_reduction_claimed=False,numerical_reference_only=False,
            scope='Actual full-storage QK/V attention;image edges masked in softmax only.No prefix compute saving claimed.')
        def wrapped(Q,K,out,encoder_seq_len,logits_out=None):
            import torch
            require(runtime.buffers is buffers and buffers['decoder_rope_weights'] is original['decoder_rope_weights'],'Original absolute RoPE/storage identity changed')
            m,n,d=Q.shape[0],K.shape[0],Q.shape[1]
            require(encoder_seq_len==prefix and (m,n,d)==(408,prefix+51,256) and K.shape==(prefix+51,256)
                and out.shape==(m,n) and all(t.is_contiguous() for t in (Q,K,out)),'Native full attention storage changed')
            if pattern=='all':self.original(Q,K,out,encoder_seq_len,logits_out)
            else:
                require(self.native.matmul_abT_scale is self.bounded,'Unbound QK kernel')
                logits=out if logits_out is None else logits_out
                require(logits.shape==out.shape and logits.is_contiguous() and logits.dtype in (torch.bfloat16,torch.float32),'Invalid native logit buffer')
                self.bounded[(((m+31)//32)*((n+31)//32),)](Q,K,logits,m,n,d,d**-.5,
                    STORE_BF16=logits.dtype==torch.bfloat16,BLOCK_SIZE_M=32,BLOCK_SIZE_N=32,BLOCK_SIZE_K=64)
                self.kernel[((m+3)//4,)](logits,m,n,prefix,out,0 if pattern=='center_columns' else 1,BLOCK_SIZE_M=4,BLOCK_SIZE=1024)
            calls.append([m,n,d])
        try:
            require(self.native.matmul_k8_256_n_softmax_mask0 is self.original,'Attention helper already changed')
            self.native.matmul_k8_256_n_softmax_mask0=wrapped;installed=True;yield audit
        finally:
            try:
                if installed:
                    lost=self.native.matmul_k8_256_n_softmax_mask0 is not wrapped
                    self.native.matmul_k8_256_n_softmax_mask0=self.original;audit['original_helper_restored']=True
                    audit['actual_QK_calls']=len(calls);audit['actual_QK_shapes']=[list(x) for x in sorted({tuple(x) for x in calls})]
                    audit['original_buffer_references_preserved']=runtime.buffers is buffers and set(buffers)==set(original) and all(buffers[k] is v for k,v in original.items())
                    if lost:raise RuntimeError('Attention helper ownership changed;original restored')
                    require(audit['original_buffer_references_preserved'],'Native attention buffer references changed')
            finally:self.lock.release()
