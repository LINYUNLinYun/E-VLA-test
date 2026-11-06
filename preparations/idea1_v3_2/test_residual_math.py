"""Behavioral scalar Transformer tests; run with JAX_PLATFORMS=cpu."""
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from .residual import integrate


class ToyLLM:
    embed_dtype = "float32"

    def __init__(self):
        self.module = self
        self.final_norms = [None,lambda hidden,condition:(hidden*.5,None)]

    def layers(self, embedded, kv, positions, mask, conditions, deterministic):
        return [None,3*embedded[1]],None

    def __call__(self,*args,method,**kwargs):
        return method(self,*args,**kwargs)


class ToyModel:
    action_horizon = 1

    def __init__(self):
        self.PaliGemma = SimpleNamespace(llm=ToyLLM())

    def embed_suffix(self,observation,x,time):
        tokens = x+observation.state[:,None,:]+time[:,None,None]
        return tokens,jnp.ones(tokens.shape[:2],bool),jnp.ones((1,),bool),None

    def action_out_proj(self,hidden):
        return 2*hidden


def reference(noise,state,refresh):
    x,t,cached = np.float32(noise),np.float32(1),np.float32(0)
    for i in range(10):
        tokens = np.float32(x+np.float32(state)+t)
        if i in refresh:
            output = np.float32(3*tokens)
            cached = np.float32(output-tokens)
        else:
            output = np.float32(tokens+cached)
        # Toy norm *0.5 and head *2 cancel exactly.
        x = np.float32(x+np.float32(-.1)*output)
        t = np.float32(t+np.float32(-.1))
    return x


class ResidualBehavior(unittest.TestCase):
    def run_toy(self,noise,state,refresh):
        obs = SimpleNamespace(state=jnp.array([[state]],jnp.float32))
        prepared = (obs,jnp.ones((1,1),bool),None)
        # Observation is closed over because SimpleNamespace is not a JAX pytree.
        function = jax.jit(lambda x,n:integrate(ToyModel(),prepared,x,refresh=refresh,audit=True,steps=n))
        return function(jnp.array([[[noise]]],jnp.float32),10)

    def test_reuse_residual_with_current_state_and_time(self):
        output,audit = self.run_toy(2.,1.,(0,3))
        np.testing.assert_allclose(output,reference(2.,1.,(0,3)),atol=1e-6)
        self.assertEqual(int(audit["euler_updates"]),10)
        self.assertEqual(int(audit["transformer_refreshes"]),2)
        self.assertAlmostEqual(float(audit["final_time"]),0.,places=6)

    def test_all_refresh_and_request_isolation(self):
        output,audit = self.run_toy(2.,1.,tuple(range(10)))
        np.testing.assert_allclose(output,reference(2.,1.,tuple(range(10))),atol=1e-6)
        self.assertEqual(int(audit["transformer_refreshes"]),10)
        initial,_ = self.run_toy(2.,1.,(0,3))
        self.run_toy(7.,9.,(0,3))
        repeated,_ = self.run_toy(2.,1.,(0,3))
        np.testing.assert_array_equal(initial,repeated)


if __name__ == "__main__":
    unittest.main()
