"""Reference only: compute every VLM module, restore and Torch-patch selected slots.

Production bypasses selected operators; this oracle runs them all and discards
their updates. It also computes Q in attention-bypassed layers, independently
checking the production KV-only projection against the native QKV route.
"""
from preparations.idea1_v3_4_E2_module_skip import backend as modules
ORIGINAL_ENCODER=modules.encoder


def encoder(native, weights, buffers, seq, skip_attn=(), skip_ffn=(), capture=False,
            history=None, older=None, ratio=0., reference=False, traced=(),
            profile=False, identity=None):
    if capture or traced or profile:
        raise ValueError('Full-compute oracle accepts no capture/trace/profiling')
    skips={'attn':modules._layers(skip_attn,'skip_attn'),
           'ffn':modules._layers(skip_ffn,'skip_ffn')}
    latest=modules._history_payload(history,identity)
    previous=modules._history_payload(older,identity)
    x=buffers['encoder_x'];calls={'qkv':[], 'attn':[], 'ffn':[]}
    for kind,layers in skips.items():
        for layer in layers:
            modules._validated_residual(latest,kind,layer,x)
            if previous is not None:modules._validated_residual(previous,kind,layer,x)
    class AllModules:
        layer=0
        def __getattr__(self,name):return getattr(native,name)
        def rms_matmul_n_2048_2560_qkv_rope(self,*args):
            self.layer+=1;calls['qkv'].append(self.layer)
            return native.rms_matmul_n_2048_2560_qkv_rope(*args)
        def update(self,kind,method,args):
            layer=self.layer;before=x.clone() if layer in skips[kind] else None
            result=getattr(native,method)(*args);calls[kind].append(layer)
            if before is not None:
                x.copy_(before)
                modules._patch(x,latest[kind][layer],None if previous is None else previous[kind][layer],ratio,reference=True)
            return result
        def matmul_n_2048_2048_res(self,*args):
            return self.update('attn','matmul_n_2048_2048_res',args)
        def matmul_n_16384_2048_res(self,*args):
            return self.update('ffn','matmul_n_16384_2048_res',args)
    _,_,meta=ORIGINAL_ENCODER(AllModules(),weights,buffers,seq,identity=identity)
    if calls!={'qkv':list(range(1,19)),'attn':list(range(1,18)),'ffn':list(range(1,18))}:
        raise RuntimeError('Oracle did not execute every native VLM component')
    meta.update(reference_only=True,full_compute_calls=calls,
                restored_and_patched={k:sorted(v) for k,v in skips.items()})
    return {},{},meta
