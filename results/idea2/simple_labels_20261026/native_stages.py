"""Original JAX equations with the original dynamic num_steps argument.

Separate JIT boundaries are diagnostic. In particular dt must not be replaced
by the Python literal -.1: the original whole sampler receives num_steps as a
traced runtime argument. All numerical gates remain unchanged.
"""


def image(self, observation):
    from openpi.models import model as model_module
    observation = model_module.preprocess_observation(None, observation, train=False)
    return self.embed_prefix(observation)


def ae(self, observation, prefix_mask, kv_cache, noise, num_steps):
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=positions,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
