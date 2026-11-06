"""Read original JAX blocks through scan; collect native residual branch values.

No upstream modules or weights are changed. The original Gemma Block.apply is
used with Linen intermediate capture; both the layer and denoising axes remain
scans. Full tensors are stored alongside small float32 pooled analysis arrays.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from types import MethodType

import numpy as np

from .common import atomic_json


def traced_llm(model,embedded,positions,mask,conds,cache=None):
    import jax
    import jax.numpy as jnp
    from flax.nnx.bridge import variables as bv
    from openpi.models import gemma
    bridge=model.PaliGemma.llm
    variables=bv.nnx_attrs_to_linen_vars({name:getattr(bridge,name) for name in bridge.linen_attributes})
    params=variables['params']
    config=bridge.module
    stream=0 if embedded[0] is not None else 1
    embedded=jax.tree.map(lambda x:x.astype(config.embed_dtype),embedded)
    initial=embedded[stream]
    block=gemma.Block(configs=tuple(config.configs),dropout=config.dropout,
                      dropout_bdims=config.dropout_bdims,parent=None)

    def step(xs,inputs):
        weights,layer_cache=inputs
        (out,kv),captured=block.apply({'params':weights},xs,layer_cache,positions,
            jnp.asarray(mask)[:,None,:,:],conds,True,
            capture_intermediates=lambda mod,method:isinstance(mod,(gemma.Attention,gemma.RMSNorm)) or (mod.name or '').startswith('mlp'),
            mutable=['intermediates'])
        captured=captured['intermediates']
        branches=captured['attn']['__call__'][0][0]
        norm_name=gemma._name('pre_attention_norm',stream)
        gate=captured[norm_name]['__call__'][0][1]
        attention_update=branches[stream] if gate is None else branches[stream]*gate
        after_attention=gemma._gated_residual(xs[stream],branches[stream],gate)
        ffn=captured[gemma._name('mlp',stream)]['__call__'][0]
        ffn_gate=captured[gemma._name('pre_ffw_norm',stream)]['__call__'][0][1]
        ffn_update=ffn if ffn_gate is None else ffn*ffn_gate
        return out,(kv,dict(post_attention=after_attention,attn_update=attention_update,
                           ffn_update=ffn_update,hidden=out[stream]))

    embedded,(kv,features)=jax.lax.scan(step,embedded,(params['layers'],cache))
    features['hidden']=jnp.concatenate([initial[None],features['hidden']],axis=0)
    outputs=[]
    for index,x in enumerate(embedded):
        outputs.append(None if x is None else gemma.RMSNorm(parent=None).apply(
            {'params':params[gemma._name('final_norm',index)]},x,conds[index])[0])
    return outputs,kv,features


def extract_model(model,observation,noise,num_steps):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    observation=model_api.preprocess_observation(None,observation,train=False)
    images={}; tokens=[]; masks=[]; camera_sizes=[]
    for name,image in observation.images.items():
        projected,info=model.PaliGemma.img(image,train=False)
        depth=len([k for k in info['encoder'] if k.startswith('block')])
        blocks=[info['encoder'][f'block{i:02d}'] for i in range(depth)]
        images[name]=dict(hidden=jnp.stack([info['with_posemb'].astype(projected.dtype),
                                          *[b['+mlp'] for b in blocks]]),
                          post_attention=jnp.stack([b['+sa'] for b in blocks]),
                          attn_update=jnp.stack([b['sa'] for b in blocks]),
                          ffn_update=jnp.stack([b['mlp'] for b in blocks]),projected=projected)
        tokens.append(projected)
        masks.append(jnp.repeat(observation.image_masks[name][:,None],projected.shape[1],axis=1))
        camera_sizes.append(projected.shape[1])
    language=model.PaliGemma.llm(observation.tokenized_prompt,method='embed')
    tokens.append(language); masks.append(observation.tokenized_prompt_mask)
    prefix=jnp.concatenate(tokens,axis=1); prefix_mask=jnp.concatenate(masks,axis=1)
    positions=jnp.cumsum(prefix_mask,axis=1)-1
    _,cache,vlm=traced_llm(model,[prefix,None],positions,
                         make_attn_mask(prefix_mask,jnp.zeros(prefix.shape[1],bool)),[None,None])
    # Preserve the native sampler's traced scalar argument and promotion rules.
    dt=-1.0/num_steps

    def denoise(carry,_):
        x,t=carry
        suffix,suffix_mask,ar,cond=model.embed_suffix(observation,x,jnp.broadcast_to(t,x.shape[0]))
        suffix_attention=make_attn_mask(suffix_mask,ar)
        prefix_attention=jnp.repeat(prefix_mask[:,None,:],suffix.shape[1],axis=1)
        attention=jnp.concatenate([prefix_attention,suffix_attention],axis=-1)
        pos=jnp.sum(prefix_mask,axis=-1)[:,None]+jnp.cumsum(suffix_mask,axis=-1)-1
        (_,out),_,ae=traced_llm(model,[None,suffix],pos,attention,[None,cond],cache)
        velocity=model.action_out_proj(out[:,-model.action_horizon:])
        return (x+dt*velocity,t+dt),dict(ae=ae,x_t=x,velocity=velocity,time=t)

    (actions,_),flow=jax.lax.scan(denoise,(noise,1.0),None,length=10)
    return dict(vision=images,vlm=vlm,action_expert=flow.pop('ae'),flow=flow,actions=actions,
                prefix_mask=prefix_mask,prompt_tokens=observation.tokenized_prompt,
                prompt_mask=observation.tokenized_prompt_mask,image_masks=observation.image_masks)


def flatten(tree,prefix=''):
    for key,value in tree.items():
        name=f'{prefix}/{key}' if prefix else key
        if isinstance(value,dict):
            yield from flatten(value,name)
        else:
            yield name,value


def compare(a,b):
    delta=np.asarray(a,np.float64)-np.asarray(b,np.float64)
    return dict(max_abs=float(np.abs(delta).max()),rmse=float(np.sqrt(np.mean(delta**2))),
                equal=bool(np.array_equal(a,b)))


def audit_model(model,observation,noise):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    observation=model_api.preprocess_observation(None,observation,train=False)
    prefix,pm,ar=model.embed_prefix(observation)
    positions=jnp.cumsum(pm,axis=1)-1;mask=make_attn_mask(pm,ar)
    normal,ncache=model.PaliGemma.llm([prefix,None],positions,mask)
    traced,tcache,_=traced_llm(model,[prefix,None],positions,mask,[None,None])
    suffix,sm,sa,cond=model.embed_suffix(observation,noise,jnp.ones((noise.shape[0],)))
    mask=jnp.concatenate([jnp.repeat(pm[:,None,:],suffix.shape[1],axis=1),make_attn_mask(sm,sa)],axis=-1)
    pos=jnp.sum(pm,axis=-1)[:,None]+jnp.cumsum(sm,axis=-1)-1
    ns,_=model.PaliGemma.llm([None,suffix],pos,mask,adarms_cond=[None,cond],kv_cache=ncache)
    ts,_,_=traced_llm(model,[None,suffix],pos,mask,[None,cond],ncache)
    diff=lambda a,b:jnp.max(jnp.abs(a.astype(jnp.float32)-b.astype(jnp.float32)))
    return dict(prefix_output_error=diff(normal[0],traced[0]),
                cache_error=jax.tree.map(diff,ncache,tcache),
                suffix_output_error=diff(ns[1],ts[1]),
                velocity_error=diff(model.action_out_proj(ns[1]),model.action_out_proj(ts[1])))


class FeatureExtractor:
    def __init__(self,policy):
        from openpi.shared.nnx_utils import module_jit
        self.policy=policy
        self.fn=module_jit(MethodType(extract_model,policy.model))
        self.audit_fn=module_jit(MethodType(audit_model,policy.model))

    def extract(self,request_path,output):
        total_begin=time.perf_counter()
        output=Path(output)
        if (output/'complete.json').exists():
            return json.loads((output/'complete.json').read_text())
        output.mkdir(parents=True,exist_ok=True)
        with np.load(request_path,allow_pickle=False) as saved:
            obs={k[5:]:saved[k] for k in saved.files if k.startswith('obs__')}
            instruction=str(saved['instruction'].item())
            noise=saved['noise'].copy(); reference_raw=saved['raw_actions'].copy()
            reference_actions=saved['actions'].copy()
        prepared=self.policy.prepare(obs,instruction)
        tick=time.perf_counter()
        result=self.fn(prepared,self.policy.jax.numpy.asarray(noise)[None],10)
        result=self.policy.jax.device_get(result)
        raw=np.asarray(result.pop('actions'))
        actions=self.policy.decode(prepared,raw)
        parity=dict(raw=compare(raw[0],reference_raw),decoded=compare(actions,reference_actions))
        # Fixed before pilot: 1e-5 absolute + relative, for instrumented scan vs
        # canonical native sampler. Never increase this threshold after failure.
        passed=(np.allclose(raw[0],reference_raw,atol=1e-5,rtol=1e-5) and
                np.allclose(actions,reference_actions,atol=1e-5,rtol=1e-5))
        atomic_json(output/'parity.json',dict(passed=bool(passed),atol=1e-5,rtol=1e-5,**parity))
        if not passed:
            audit=self.policy.jax.device_get(self.audit_fn(prepared,self.policy.jax.numpy.asarray(noise)[None]))
            audit=self.policy.jax.tree.map(lambda x:float(x),audit)
            atomic_json(output/'stage_audit.json',audit)
            raise ValueError(f'Instrumented scan action mismatch: {parity}; stage audit={audit}')
        timings=dict(extract_s=time.perf_counter()-tick)
        tick=time.perf_counter()
        pooled,statistics=self.pool(result)
        timings['pool_s']=time.perf_counter()-tick
        tick=time.perf_counter()
        prefix_mask=np.asarray(result['prefix_mask'][0],bool)
        storage={}; dtypes={}
        for key,value in flatten(result):
            value=np.asarray(value)
            if key.startswith('vlm/'):
                value=value[:,:,prefix_mask,:]
            if key.startswith('vision/'):
                name=key.split('/')[1]
                if not bool(np.asarray(result['image_masks'][name]).item()):
                    continue
            dtypes[key]=str(value.dtype)
            # NumPy's NPY header does not round-trip bfloat16. Store native bits
            # explicitly and retain the semantic dtype in metadata.
            storage[key]=value.view(np.uint16) if str(value.dtype)=='bfloat16' else value
        storage['action/final_raw']=raw[0]
        storage['action/final_decoded']=actions
        timings['pack_s']=time.perf_counter()-tick
        tick=time.perf_counter()
        np.savez(output/'activations.npz',**storage)
        np.savez(output/'pooled.npz',**pooled)
        np.savez(output/'residual_stats.npz',**statistics)
        timings['save_s']=time.perf_counter()-tick
        bytes_written=sum(p.stat().st_size for p in output.glob('*.npz'))
        timings['total_s']=time.perf_counter()-total_begin
        metadata=dict(source=str(request_path),dtypes=dtypes,prefix_valid_tokens=int(prefix_mask.sum()),
                      service=dict(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
                      language_group='valid instruction prompt tokens; selected pi05_libero disables discrete state input',
                      discrete_state_input=False,
                      hidden_index='H0=input, H1..H18=post-FFN block outputs',
                      bf16_encoding='uint16 native bits, semantic dtype in dtypes',
                      pooling_impl='cpu_cached_v2',
                      flow_times=np.asarray(result['flow']['time']).astype(float).tolist(),
                      bytes=bytes_written,timing=timings,parity=parity)
        atomic_json(output/'metadata.json',metadata)
        receipt=dict(passed=True,source=str(request_path),output=str(output),bytes=bytes_written,
                     timing=timings,parity=parity)
        atomic_json(output/'complete.json',receipt)
        return receipt

    @staticmethod
    def pool(result):
        from .pooling import pool
        return pool(result)
