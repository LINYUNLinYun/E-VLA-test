"""Unlaunched prefix-buffer transaction; CPU-testable without any model or CUDA context.

The original adapter may sample the yielded context while the transaction is
active. Dense VLM prefill is complete first. Decoder RoPE remains the original
absolute-position table. This module is not integrated into a policy or queue.
"""
from contextlib import contextmanager
import hashlib
import json
import torch
from reports.ease_insights_20261103.prefix_kv_preparation_v1.layout_contract import plan,validate_buffers

KEYS=('encoder_K','encoder_V','decoder_attn_buf','decoder_attn_logits_buf')


def require(value,message):
    if not value:raise ValueError(message)


class PrefixPacker:
    def __init__(self):
        self.cache={}
        self.active=False

    def clear(self):
        require(not self.active,'Cannot clear scratch during sampling')
        self.cache.clear()

    def _scratch(self,runtime,layout,buffers):
        present='decoder_attn_logits_buf' in buffers
        key=(id(runtime),layout.pattern,layout.original_prefix,present)
        if key not in self.cache:
            entry=dict(runtime=runtime,
                encoder_K=torch.empty(layout.kv_shape,dtype=buffers['encoder_K'].dtype,device=buffers['encoder_K'].device),
                encoder_V=torch.empty(layout.kv_shape,dtype=buffers['encoder_V'].dtype,device=buffers['encoder_V'].device),
                decoder_attn_buf=torch.empty(layout.attention_shape,dtype=buffers['decoder_attn_buf'].dtype,device=buffers['decoder_attn_buf'].device))
            if present:
                tensor=buffers['decoder_attn_logits_buf']
                entry['decoder_attn_logits_buf']=torch.empty(layout.attention_shape,dtype=tensor.dtype,device=tensor.device)
            self.cache[key]=entry
        result=self.cache[key]
        require(result['runtime'] is runtime,'Runtime identity reused')
        require(all(result[k].dtype==buffers[k].dtype and result[k].device==buffers[k].device
            for k in KEYS if k in result),'Cached scratch device/dtype differs')
        return {k:result[k] for k in KEYS if k in result}

    @contextmanager
    def packed(self,context,pattern):
        require(not self.active,'Prefix packing is serial; nested/concurrent use is forbidden')
        runtime=context['runtime'];buffers=runtime.buffers;prefix=context['seq'];layout=plan(pattern,prefix)
        for name in ('K','V'):
            value=context[name]
            require(tuple(value.shape)==(18,prefix,256) and value.is_contiguous(),'Need immutable full-prefix K/V snapshots')
        require(tuple(buffers['decoder_rope_weights'].shape)==(51,256),'Need the original51 decoder absolute-position embeddings')
        original={k:buffers[k] for k in KEYS if k in buffers};rope=buffers['decoder_rope_weights']
        require(tuple(original['encoder_K'].shape)==(18,prefix+51,256)
            and tuple(original['encoder_V'].shape)==(18,prefix+51,256)
            and tuple(original['decoder_attn_buf'].shape)==(408,prefix+51),'Dense runtime buffers are incompatible')
        for key in ('encoder_K','encoder_V'):
            require(original[key].is_contiguous(),'Dense K/V storage must be contiguous')
        require(context['K'].device==original['encoder_K'].device and context['V'].device==original['encoder_V'].device
            and context['K'].dtype==original['encoder_K'].dtype and context['V'].dtype==original['encoder_V'].dtype,'Context device/dtype differs')
        # All allocation/gather errors occur before replacing a runtime-owned buffer.
        scratch=self._scratch(runtime,layout,buffers)
        index=torch.tensor(layout.keep_indices,dtype=torch.long,device=context['K'].device)
        reduced=dict(context,K=torch.index_select(context['K'],1,index).contiguous(),
            V=torch.index_select(context['V'],1,index).contiguous(),seq=layout.packed_prefix)
        validate_buffers(layout,scratch['encoder_K'].shape,scratch['encoder_V'].shape,scratch['decoder_attn_buf'].shape,
            all(t.is_contiguous() for t in scratch.values()),layout.decoder_absolute_positions,
            logits_shape=scratch['decoder_attn_logits_buf'].shape if 'decoder_attn_logits_buf' in scratch else None)
        expected_kept=tuple(index.cpu().tolist()) if index.device.type=='cpu' else layout.keep_indices
        require(expected_kept==layout.keep_indices,'Gather index content differs')
        audit=dict(pattern=pattern,original_prefix=prefix,packed_prefix=layout.packed_prefix,
            kept_prefix_indices=list(layout.keep_indices),
            kept_index_sha256=hashlib.sha256(json.dumps(list(layout.keep_indices),separators=(',',':')).encode()).hexdigest(),
            visual_tokens_per_camera=[sum(256*c<=i<256*(c+1) for i in layout.keep_indices) for c in (0,1)],
            language_tokens_retained=prefix-512,decoder_absolute_positions=list(layout.decoder_absolute_positions),
            decoder_storage_positions=list(layout.decoder_storage_positions),kv_shape=list(layout.kv_shape),
            attention_shape=list(layout.attention_shape),original_rope_table_reused=True,
            dense_VLM_prefill_already_completed=True,packing_latency_measured=False,
            scope='Attention-prefix packing only; no VLM-prefill reduction,speed,quality or CUDA equivalence claim')
        self.active=True
        try:
            for name,value in scratch.items():buffers[name]=value
            yield reduced,audit
        finally:
            lost=(runtime.buffers is not buffers or any(buffers.get(k) is not v for k,v in scratch.items())
                or buffers.get('decoder_rope_weights') is not rope)
            # Restore owned references even when the consumer fails.
            runtime.buffers=buffers
            for name in KEYS:
                if name in original:buffers[name]=original[name]
                else:buffers.pop(name,None)
            buffers['decoder_rope_weights']=rope
            self.active=False
            if lost:raise RuntimeError('Sampling changed borrowed buffer ownership; original references restored')
