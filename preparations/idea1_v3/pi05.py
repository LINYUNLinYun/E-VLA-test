"""Isolated native JAX pi0.5 VLM hidden-update skips with original KV shells.

No upstream source is edited. Only prefix stream 0 bypasses selected blocks;
action-expert suffix calls always execute every original layer with its own KV.
"""
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'shared/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero'
MASKS={'Full':(), 'transfer16':(3,9), 'transfer15':(3,9,11),
       'V14_A03':(3,4,9,11), 'V13_Custom':(3,4,7,12,15)}


def install_vlm_mask(skip):
    """Install once per isolated model service, before model load/JIT tracing."""
    import jax
    import jax.numpy as jnp
    from openpi.models import gemma
    original=getattr(gemma.Module,'_v3_original_call',gemma.Module.__call__)
    gemma.Module._v3_original_call=original
    skip=tuple(skip)
    if len(set(skip))!=len(skip) or any(not isinstance(i,int) or i<1 or i>18 for i in skip):
        raise ValueError('Explicit original 1-based mask 1..18 required')
    def call(self,embedded,positions,mask,adarms_cond=None,*,kv_cache=None,deterministic=True):
        if self.is_initializing() or not skip:
            return original(self,embedded,positions,mask,adarms_cond,kv_cache=kv_cache,deterministic=deterministic)
        depth=self.configs[0].depth
        if depth!=18: raise ValueError('Refuse unexplained skip mapping onto non-18-layer architecture')
        xs=jax.tree.map(lambda e:e.astype(self.embed_dtype),embedded)
        cond=adarms_cond if adarms_cond is not None else [None]*len(self.configs)
        attention_mask=jnp.asarray(mask)[:,None,:,:]
        params=self.variables['params']['layers']
        caches=[]
        for index in range(depth):
            layer_params=jax.tree.map(lambda value:value[index],params)
            cache=None if kv_cache is None else jax.tree.map(lambda value:value[index],kv_cache)
            prefix_only=xs[0] is not None and all(x is None for x in xs[1:])
            if prefix_only and index+1 in skip:
                # Original-index K/V shell remains conditioned on the unchanged
                # hidden stream. Unused attention output is eliminated by XLA.
                norm,_=gemma.RMSNorm().apply({'params':layer_params['pre_attention_norm']},xs[0],cond[0])
                _,cache=gemma.Attention(configs=self.configs).apply({'params':layer_params['attn']},
                    [norm]+[None]*(len(xs)-1),positions,attention_mask,cache)
            else:
                xs,cache=gemma.Block(configs=self.configs,dropout=self.dropout,dropout_bdims=self.dropout_bdims).apply(
                    {'params':layer_params},xs,cache,positions,attention_mask,cond,deterministic)
            caches.append(cache)
        kv=jax.tree.map(lambda *values:jnp.stack(values,axis=0),*caches)
        return [norm(x,c)[0] if x is not None else None for norm,x,c in zip(self.final_norms,xs,cond,strict=True)],kv
    gemma.Module.__call__=call
    return original


class Pi05Policy:
    def __init__(self,config):
        import jax
        from openpi.training import config as training
        from openpi.policies import policy_config
        self.config=config; name=config['mask']; skip=MASKS[name]
        install_vlm_mask(skip)
        train=training.get_config('pi05_libero')
        if not train.model.pi05: raise ValueError('Loaded model is not pi0.5')
        self.H=int(train.model.action_horizon)
        self.policy=policy_config.create_trained_policy(train,CHECKPOINT,sample_kwargs=dict(num_steps=10))
        self.protocol=dict(model='pi05_libero',H=self.H,R=5,D=10,checkpoint=str(CHECKPOINT),
            backend='native OpenPI JAX',skip=list(skip),mask=name,depth=18,
            mask_semantics='prefix-only hidden-update bypass, original-index KV shell; action expert complete',
            native_H_preserved=True,normalization_asset=str(CHECKPOINT/'assets/physical-intelligence/libero/norm_stats.json'))
        self.jax=jax; self.reset()

    def reset(self):
        self.pending=None; self.step=self.round=0
        return {'ok':True}

    def acknowledge(self,executed):
        executed=np.asarray(executed,np.float32)
        if self.pending is None or not 0<len(executed)<=self.pending['execute_length']:
            raise ValueError('invalid execution ack')
        if not np.array_equal(executed,self.pending['actions'][:len(executed)]): raise ValueError('command mismatch')
        self.step+=len(executed); self.pending=None
        return {'ok':True}

    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None: raise RuntimeError('execution acknowledgement missing')
        from preparations.idea1_v2_1.runtime_models import native_observation
        images,state=native_observation(obs)
        data={'observation/image':np.asarray(images[0]),'observation/wrist_image':np.asarray(images[1]),
              'observation/state':state,'prompt':instruction}
        noise=np.random.default_rng(seed).standard_normal((self.H,32),dtype=np.float32)
        tick=time.perf_counter(); response=self.policy.infer(data,noise=noise)
        actions=np.asarray(response['actions'],np.float32)
        if actions.shape!=(self.H,7) or not np.isfinite(actions).all(): raise ValueError('invalid native pi05 actions')
        result=dict(actions=actions,execute_length=min(5,self.H),raw=np.empty((0,32),np.float32),arrays={},
            route=dict(path='pi05_native',mask=self.config['mask']),counts=dict(full=self.round+1,verify=0,draft=0),
            seed=seed,round=self.round,step=self.step,timing=dict(policy_wall_ms=(time.perf_counter()-tick)*1000,
                **response.get('policy_timing',{})))
        self.pending=result; self.round+=1
        return result
