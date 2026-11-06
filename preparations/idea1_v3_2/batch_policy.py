"""Native Full pi0 batch adapter: independent inputs/noises, unchanged sampler."""
import time
import numpy as np


def noises(records, horizon=50, action_dim=32):
    return np.stack([np.random.default_rng(int(r['seed'])).standard_normal(
        (horizon, action_dim), dtype=np.float32) for r in records])


def compare(reference, actual, atol=2e-3, rtol=2e-3):
    a, b = np.asarray(reference), np.asarray(actual)
    return dict(passed=bool(a.shape == b.shape and np.isfinite(b).all() and
                           np.allclose(a, b, atol=atol, rtol=rtol)),
                strict_1e5_passed=bool(a.shape == b.shape and np.isfinite(b).all() and
                                      np.allclose(a, b, atol=1e-5, rtol=1e-5)),
                max_abs=float(np.max(np.abs(a-b))), mean_abs=float(np.mean(np.abs(a-b))),
                atol=atol, rtol=rtol)


class BatchPolicy:
    def __init__(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi.training import config
        from openpi.policies import policy_config
        from openpi.models import model as model_api
        from preparations.idea1_v3_1.common import ROOT
        start = time.perf_counter()
        self.policy = policy_config.create_trained_policy(config.get_config('pi0_libero'),
            ROOT/'shared/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero',
            sample_kwargs={'num_steps': 10})
        self.jax, self.jnp, self.model_api = jax, jnp, model_api
        self.model = self.policy._model
        assert self.model.action_horizon == 50 and self.model.action_dim == 32
        graph, self.state = nnx.split(self.model)
        def sample(state, rng, observation, noise, num_steps):
            # Match native Policy's traced num_steps argument. Closing over 10
            # changes XLA constant folding and introduces avoidable numeric drift.
            return nnx.merge(graph, state).sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        self.sample = jax.jit(sample)
        self.executables = {}
        self.compiles = {}
        jax.block_until_ready(self.state)
        self.load_s = time.perf_counter()-start

    def prepare(self, records):
        from preparations.idea1_v3_1.models import raw_observation
        transformed = [self.policy._input_transform(raw_observation(r['obs'], r['instruction'])) for r in records]
        inputs = self.jax.tree.map(lambda *values: self.jnp.asarray(np.stack(values)), *transformed)
        observation = self.model_api.Observation.from_dict(inputs)
        noise = self.jnp.asarray(noises(records))
        self.jax.block_until_ready((observation, noise))
        return inputs, observation, noise

    def predict(self, records):
        started = time.perf_counter()
        inputs, observation, noise = self.prepare(records)
        prepared = time.perf_counter()
        size = len(records)
        arguments = (self.state, self.jax.random.key(0), observation, noise, 10)
        if size not in self.executables:
            from openpi.shared import array_typing
            tick = time.perf_counter()
            # AOT lowering reconstructs Observation with JAX ArgInfo metadata;
            # only that introspection phase needs the provided typing context.
            with array_typing.disable_typechecking():
                lowered = self.sample.lower(*arguments)
            lowered_s = time.perf_counter()-tick
            tick = time.perf_counter()
            self.executables[size] = lowered.compile()
            self.compiles[size] = dict(lower_s=lowered_s, compile_s=time.perf_counter()-tick)
        tick = time.perf_counter()
        raw = np.asarray(self.executables[size](*arguments))
        model_ms = (time.perf_counter()-tick)*1000
        states = np.asarray(inputs['state'])
        decoded = np.stack([self.policy._output_transform({'state': states[i], 'actions': raw[i]})['actions']
                            for i in range(size)])
        assert decoded.shape == (size, 50, 7) and np.isfinite(decoded).all()
        return decoded, raw, dict(prepare_ms=(prepared-started)*1000, model_completed_ms=model_ms,
            request_completed_ms=(time.perf_counter()-started)*1000, batch_size=size)

    def responses(self, records):
        from preparations.idea1_v2_1.runtime_client import input_digest, value_digest
        actions, _, timing = self.predict(records)
        ns = noises(records)
        return [dict(actions=actions[i], seed=int(record['seed']), execute_length=5,
                     condition='pi0_full', input_hash=input_digest(record['obs'], record['instruction']),
                     noise_hash=value_digest(ns[i]), timing=timing, audit={'NFE': 10, 'actual_batch_size': len(records)})
                for i, record in enumerate(records)]
