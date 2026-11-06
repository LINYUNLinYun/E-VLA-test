"""Request-local AE residual reuse at the pre-final-RMSNorm boundary.

The cache is FP32 (scanned-layer-output - dtype-cast suffix embedding).
Only the 18 scanned Transformer blocks are skipped. Every Euler step embeds
the current noisy action, state and time, recomputes final RMSNorm with the
current adaRMS condition, then recomputes the action projection. No velocity
is cached. Fresh prefix image/VLM KV is constructed for every request.
"""
from functools import partial
from types import MethodType


def require_idle_gpu():
    """Fail before model construction if assigned physical GPU has any compute PID."""
    import os
    import subprocess
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not assigned or "," in assigned:
        raise RuntimeError("Assign exactly one physical GPU explicitly")
    base = ["nvidia-smi","-i",assigned]
    processes = subprocess.check_output(base+[
        "--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True).strip()
    device = subprocess.check_output(base+[
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip()
    receipt = dict(assigned=assigned,device=device,preexisting_compute_processes=processes)
    if processes:
        raise RuntimeError("GPU_NOT_IDLE: "+str(receipt))
    return receipt


def pre_norm(module, embedded, positions, mask, adarms_cond, *, kv_cache):
    import jax
    import jax.numpy as jnp
    embedded = jax.tree.map(lambda e: e.astype(module.embed_dtype), embedded)
    embedded, _ = module.layers(embedded, kv_cache, positions,
                                jnp.asarray(mask)[:, None, :, :], adarms_cond, True)
    return embedded[1]


def final_norm(module, hidden, condition):
    return module.final_norms[1](hidden, condition)[0]


def prefix(model, observation):
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    observation = model_api.preprocess_observation(None, observation, train=False)
    tokens, mask, ar = model.embed_prefix(observation)
    _, kv = model.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar),
                              positions=jnp.cumsum(mask, axis=1)-1)
    return observation, mask, kv


def integrate(model, prepared, noise, *, refresh=(0, 3), steps=10, audit=False):
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    observation, prefix_mask, kv_cache = prepared
    batch = observation.state.shape[0]
    dt = -1.0 / steps
    initial, _, _, _ = model.embed_suffix(observation, noise, jnp.ones((batch,)))
    # Cache shape includes the pi0 state token, even though only action rows
    # reach the output projection. It lives entirely in this while-loop carry.
    residual = jnp.zeros_like(initial, dtype=jnp.float32)
    def step(carry):
        x, time, index, cached, refresh_count = carry
        tokens, mask, ar, condition = model.embed_suffix(
            observation, x, jnp.broadcast_to(time, batch))
        dtype = model.PaliGemma.llm.module.embed_dtype
        cast_tokens = tokens.astype(dtype)
        full_mask = jnp.concatenate([
            jnp.broadcast_to(prefix_mask[:, None, :],
                             (batch, tokens.shape[1], prefix_mask.shape[1])),
            make_attn_mask(mask, ar)], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None]+jnp.cumsum(mask, axis=-1)-1
        def run_full(_):
            hidden = model.PaliGemma.llm([None, tokens], positions=positions,
                mask=full_mask, adarms_cond=[None, condition], kv_cache=kv_cache,
                method=pre_norm)
            return hidden, hidden.astype(jnp.float32)-cast_tokens.astype(jnp.float32)
        def reuse(_):
            return (cast_tokens.astype(jnp.float32)+cached).astype(dtype), cached
        full = jnp.any(index == jnp.asarray(refresh, jnp.int32))
        hidden, cached = jax.lax.cond(full, run_full, reuse, operand=None)
        normalized = model.PaliGemma.llm(hidden, condition, method=final_norm)
        velocity = model.action_out_proj(normalized[:, -model.action_horizon:])
        return x+dt*velocity, time+dt, index+1, cached, refresh_count+full.astype(jnp.int32)
    result = jax.lax.while_loop(lambda c: c[1] >= -dt/2, step,
                                (noise, 1.0, 0, residual, 0))
    return (result[0],dict(euler_updates=result[2],transformer_refreshes=result[4],final_time=result[1])) if audit else result[0]


def sample(model, observation, noise, *, refresh=(0, 3), steps=10):
    return integrate(model, prefix(model, observation), noise, refresh=refresh, steps=steps)


def compiled(model, *, refresh=(0, 3), steps=10):
    from openpi.shared import nnx_utils
    function = nnx_utils.module_jit(MethodType(partial(sample, refresh=refresh), model))
    # Keep steps a dynamic JAX argument, exactly as canonical Policy.infer.
    return partial(function,steps=steps)


class ResidualPolicy:
    """Compatible with the v3.1 localhost rollout protocol; no source patches."""
    def __init__(self, mode="reuse03"):
        import jax
        import jax.numpy as jnp
        from openpi.training import config
        from openpi.policies import policy_config
        from openpi.models import model as model_api
        from openpi.shared import nnx_utils
        from preparations.idea1_v3_1.common import ROOT
        self.jax, self.jnp, self.model_api = jax, jnp, model_api
        self.policy = policy_config.create_trained_policy(config.get_config("pi0_libero"),
            ROOT/"shared/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero",
            sample_kwargs={"num_steps":10})
        self.model = self.policy._model
        self.mode = mode
        def native(obs, noise):
            return self.policy._sample_actions(jax.random.key(0),obs,num_steps=10,noise=noise)
        def native_d2(obs, noise):
            return self.policy._sample_actions(jax.random.key(0),obs,num_steps=2,noise=noise)
        self.native = native
        self.functions = {"native":self.native,
            "refresh_all":compiled(self.model, refresh=tuple(range(10))),
            "reuse03":compiled(self.model),
            "d2":native_d2,
            "d2_refresh":compiled(self.model, refresh=(0,1), steps=2)}
        self.prefill = nnx_utils.module_jit(MethodType(prefix, self.model))
        self.ae = {key:partial(nnx_utils.module_jit(MethodType(partial(integrate,
            refresh=(0,3) if key=="reuse03" else tuple(range(2 if key=="d2" else 10))), self.model)),
            steps=2 if key=="d2" else 10)
            for key in ("refresh_all","reuse03","d2")}

    def prepare(self, obs, instruction):
        from preparations.idea1_v3_1.models import raw_observation
        data = self.policy._input_transform(raw_observation(obs, instruction))
        return self.model_api.Observation.from_dict(self.jax.tree.map(
            lambda x:self.jnp.asarray(x)[None], data))

    def decode(self, raw, observation):
        import numpy as np
        return self.policy._output_transform(dict(state=np.asarray(observation.state)[0],
                                                  actions=np.asarray(raw)[0]))["actions"]

    def reset(self):
        return dict(ok=True, cache="request-local only")

    def predict(self, obs, instruction, seed, **unused):
        import time
        import numpy as np
        from preparations.idea1_v2_1.runtime_client import input_digest, value_digest
        begin = time.perf_counter()
        observation = self.prepare(obs, instruction)
        noise = np.random.default_rng(seed).standard_normal(
            (1,self.model.action_horizon,self.model.action_dim),dtype=np.float32)
        raw = self.functions[self.mode](observation, self.jnp.asarray(noise))
        actions = np.asarray(self.decode(raw, observation),np.float32)
        return dict(actions=actions,input_hash=input_digest(obs,instruction),noise_hash=value_digest(noise[0]),
            seed=seed,condition=self.mode,execute_length=5,
            timing=dict(request_wall_ms=(time.perf_counter()-begin)*1000),
            audit=dict(NFE=2 if self.mode=="d2" else 10,
                full_transformer_updates=[0,3] if self.mode=="reuse03" else list(range(2 if self.mode=="d2" else 10)),
                residual_boundary="pre-final-RMSNorm",cached_velocity=False,fresh_prefix=True))
