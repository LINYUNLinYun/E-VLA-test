"""Unlaunched native-sampler orchestration with observed attention-prefix lengths.

No model is constructed and no CUDA context is created on import. Native GPU
parity and actual packing+policy costs remain mandatory before scientific use.
"""
from contextlib import contextmanager,nullcontext
import threading
import time
from reports.ease_insights_20261103.prefix_kv_preparation_v1.layout_contract import plan


def require(value,message):
    if not value:raise ValueError(message)


class PrefixVisits:
    def __init__(self,namespace):
        self.namespace=namespace;self.original=namespace._decoder_layer;self.lock=threading.Lock()

    @contextmanager
    def use(self,context,layout):
        buffers=context['runtime'].buffers;rope=buffers['decoder_rope_weights'];attempted=[];executed=[]
        require(self.lock.acquire(blocking=False),'Prefix attention observation is serial and non-reentrant')
        observed=dict(attempted=attempted,executed=executed,original_helper_restored=False)
        def wrapped(weights,actual,seq,index):
            require(actual is buffers and type(index) is int and 0<=index<18,'Unexpected native helper buffer/layer')
            require(seq==layout.packed_prefix and tuple(actual['encoder_K'].shape)==layout.kv_shape
                and tuple(actual['encoder_V'].shape)==layout.kv_shape
                and tuple(actual['decoder_attn_buf'].shape)==layout.attention_shape,'Native helper consumed wrong prefix/token storage')
            require(actual['decoder_rope_weights'] is rope,'Native helper lost original absolute-position RoPE')
            attempted.append(index+1);result=self.original(weights,actual,seq,index);executed.append(index+1);return result
        installed=False
        try:
            require(self.namespace._decoder_layer is self.original,'Native helper was already patched')
            self.namespace._decoder_layer=wrapped;installed=True
            yield observed
        finally:
            try:
                if installed:
                    violated=self.namespace._decoder_layer is not wrapped
                    self.namespace._decoder_layer=self.original;observed['original_helper_restored']=True
                    if violated:raise RuntimeError('Native helper ownership changed; original restored')
            finally:self.lock.release()


def _sample(adapter,observer,context,layout,seed,steps):
    with observer.use(context,layout) as visits:
        result=adapter.sample(context,seed,dict(kind='native',m=None,num_steps=steps),False)
    require(len(result)==6 and result[3].get('actual_NFE')==steps,'Native sampler did not return the requested actualNFE')
    expected=list(range(1,19))*steps
    require(visits['attempted']==expected and visits['executed']==expected and visits['original_helper_restored'],
        'Native sampler did not execute all18 AE layers at each denoising step')
    ae=dict(actual_NFE=steps,attempted_layers_per_step=list(range(1,19)),executed_layers_per_step=list(range(1,19)),
        bypassed_layers_per_step=[],actual_layer_calls=18*steps,attempted_layer_calls=18*steps,bypassed_layer_calls=0,
        original_helper_restored=True,scope='Observed eager helper dispatch, not FLOPs, latency or quality')
    return result,ae


def sample_with_prefix(adapter,observer,packer,context,seed,steps,pattern='all'):
    require(type(steps) is int and steps in (2,10),'Only original D2/D10 schedules are supported')
    require(pattern in ('all','identity','center_columns','checkerboard'),'Unknown prefix selection')
    require(steps==2 or pattern in ('all','identity'),'FullD10 must keep all prefix tokens')
    layout=plan('identity' if pattern=='all' else pattern,context['seq'])
    original_buffers=context['runtime'].buffers;original=dict(original_buffers)
    manager=nullcontext((context,None)) if pattern=='all' else packer.packed(context,pattern)
    events=None
    if pattern!='all' and context['K'].device.type=='cuda':
        events=(adapter.torch.cuda.Event(enable_timing=True),adapter.torch.cuda.Event(enable_timing=True));events[0].record()
    started=time.perf_counter();pack_ms=0.;pack_cuda_ms=None
    with manager as (active,audit):
        if pattern!='all':
            if events is not None:events[1].record();events[1].synchronize();pack_cuda_ms=float(events[0].elapsed_time(events[1]))
            pack_ms=(time.perf_counter()-started)*1000
        result,ae=_sample(adapter,observer,active,layout,seed,steps)
        restore_started=time.perf_counter()
    restore_ms=(time.perf_counter()-restore_started)*1000 if pattern!='all' else 0.
    restored=context['runtime'].buffers is original_buffers and set(original_buffers)==set(original)
    restored=restored and all(original_buffers[k] is value for k,value in original.items())
    require(restored,'Prefix transaction did not restore all original runtime buffer references')
    if audit is None:
        audit=dict(pattern='all',original_prefix=layout.original_prefix,packed_prefix=layout.packed_prefix,
            kept_prefix_indices=list(layout.keep_indices),visual_tokens_per_camera=[256,256],language_tokens_retained=layout.original_prefix-512,
            decoder_absolute_positions=list(layout.decoder_absolute_positions),decoder_storage_positions=list(layout.decoder_storage_positions),
            kv_shape=list(layout.kv_shape),attention_shape=list(layout.attention_shape),original_rope_table_reused=True,
            dense_VLM_prefill_already_completed=True)
    audit=dict(audit,pattern=pattern,actual_helper_calls=18*steps,actual_prefix_length=layout.packed_prefix,
        actual_total_key_tokens=layout.packed_prefix+51,original_buffers_restored=True,
        original_helper_restored=True,packed=pattern!='all',packing_wall_ms=pack_ms,packing_cuda_ms=pack_cuda_ms,
        restore_wall_ms=restore_ms,packing_latency_measured=pattern!='all',
        scope='Actual full AE18 helper visits and packed key shape with unchanged RoPE;no quality or speed claim')
    return result,ae,audit
