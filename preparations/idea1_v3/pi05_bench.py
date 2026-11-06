"""Diagnostic JAX stage boundaries for pi0.5, never used by scientific workers.

These are the native equations split at image-embedding, VLM-prefill and complete
AE-loop boundaries. Extra materialization/fences can alter compiler fusion and
latency, so stage sums are diagnostic and do not replace native total latency.
"""
import time
import types
import numpy as np


def image_stage(self,observation):
    return self.embed_prefix(observation)


def vlm_stage(self,prefix_tokens,prefix_mask,prefix_ar_mask):
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    positions=jnp.cumsum(prefix_mask,axis=1)-1
    _,cache=self.PaliGemma.llm([prefix_tokens,None],mask=make_attn_mask(prefix_mask,prefix_ar_mask),positions=positions)
    return cache


def _ae_stage(self,observation,prefix_mask,cache,noise,*,native_scanned_suffix=False):
    import jax
    import jax.numpy as jnp
    import einops
    from openpi.models.pi0 import make_attn_mask
    batch_size=observation.state.shape[0]; dt=-.1
    def step(carry):
        x_t,t=carry
        tokens,mask,ar_mask,condition=self.embed_suffix(observation,x_t,jnp.broadcast_to(t,batch_size))
        prefix_attention=einops.repeat(prefix_mask,'b p -> b s p',s=tokens.shape[1])
        full_mask=jnp.concatenate([prefix_attention,make_attn_mask(mask,ar_mask)],axis=-1)
        positions=jnp.sum(prefix_mask,axis=-1)[:,None]+jnp.cumsum(mask,axis=-1)-1
        method={}
        if native_scanned_suffix:
            from openpi.models import gemma
            # Explicit Flax method dispatch bypasses only the patched suffix;
            # it does not mutate the global Module or native reference sampler.
            method['method']=gemma.Module._v3_original_call
        (_,suffix),_=self.PaliGemma.llm([None,tokens],mask=full_mask,positions=positions,
            kv_cache=cache,adarms_cond=[None,condition],**method)
        velocity=self.action_out_proj(suffix[:,-self.action_horizon:])
        return x_t+dt*velocity,t+dt
    return jax.lax.while_loop(lambda carry:carry[1]>=-dt/2,step,(noise,1.))[0]


def ae_stage(self,observation,prefix_mask,cache,noise):
    return _ae_stage(self,observation,prefix_mask,cache,noise)


class StageProfiler:
    def __init__(self,native_policy,revision='v1-unrolled-ae'):
        from openpi.shared.nnx_utils import module_jit
        self.native=native_policy
        self.revision=revision
        model=native_policy.policy._model
        self.image=module_jit(types.MethodType(image_stage,model))
        self.vlm=module_jit(types.MethodType(vlm_stage,model))
        selected=ae_stage
        if revision=='v2-native-scanned-suffix':
            from preparations.idea1_v3.pi05_bench_v2 import ae_stage_scanned_suffix
            selected=ae_stage_scanned_suffix
        elif revision!='v1-unrolled-ae': raise ValueError('Unknown diagnostic stage revision')
        self.ae=module_jit(types.MethodType(selected,model))

    def run(self,case):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_module
        from preparations.idea1_v2_1.runtime_models import native_observation
        images,state=native_observation(case['obs']); policy=self.native.policy
        tick=time.perf_counter()
        transformed=policy._input_transform({'observation/image':np.asarray(images[0]),
            'observation/wrist_image':np.asarray(images[1]),'observation/state':state,'prompt':case['instruction']})
        batched=jax.tree.map(lambda value:jnp.asarray(value)[None,...],transformed)
        observation=model_module.preprocess_observation(None,model_module.Observation.from_dict(batched),train=False)
        noise=jnp.asarray(np.random.default_rng(case['seed']).standard_normal((1,self.native.H,32),dtype=np.float32))
        jax.block_until_ready((observation,noise)); input_ms=(time.perf_counter()-tick)*1000
        tick=time.perf_counter(); prefix=self.image(observation); jax.block_until_ready(prefix)
        image_ms=(time.perf_counter()-tick)*1000
        tick=time.perf_counter(); cache=self.vlm(*prefix); jax.block_until_ready(cache)
        vlm_ms=(time.perf_counter()-tick)*1000
        tick=time.perf_counter(); raw=self.ae(observation,prefix[1],cache,noise); jax.block_until_ready(raw)
        ae_ms=(time.perf_counter()-tick)*1000
        tick=time.perf_counter(); decoded=policy._output_transform(dict(state=np.asarray(batched['state'][0]),actions=np.asarray(raw[0])))['actions']
        decode_ms=(time.perf_counter()-tick)*1000
        return np.asarray(decoded,np.float32),dict(input_ms=input_ms,image_embedding_ms=image_ms,
            vlm_prefill_ms=vlm_ms,ae_d10_ms=ae_ms,decode_ms=decode_ms,
            stage_revision=self.revision,
            ae_compilation_structure='native nn.scan' if self.native.config['mask']=='Full' or self.revision=='v2-native-scanned-suffix' else '18-layer Python-unrolled JAX trace',
            native_reference_ae_structure='native nn.scan' if self.native.config['mask']=='Full' else '18-layer Python-unrolled JAX trace',
            attribution='Diagnostic revision does not alter old native total; separate JIT/fence costs remain, and old total is not isolated VLM pruning gain',
            boundary='diagnostic split JAX methods, each block_until_ready; extra boundaries may alter fusion')
