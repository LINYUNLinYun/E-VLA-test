"""Exercise original Gemma prefix/cache/suffix semantics with small real blocks."""
import unittest
from types import SimpleNamespace

import numpy as np


class FeatureTests(unittest.TestCase):
    def test_original_prefix_suffix_and_gated_updates(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from flax.nnx import bridge as nnx_bridge
        from openpi.models import gemma
        from .features import traced_llm
        config=gemma.Config(width=32,depth=2,mlp_dim=64,num_heads=2,num_kv_heads=1,head_dim=16)
        llm=nnx_bridge.ToNNX(gemma.Module(configs=(config,config),embed_dtype='float32',adarms=True))
        llm.lazy_init(rngs=nnx.Rngs(3),method='init',use_adarms=[False,True])
        model=SimpleNamespace(PaliGemma=SimpleNamespace(llm=llm))
        x=jax.random.normal(jax.random.key(1),(1,4,32))
        positions=jnp.arange(4)[None]
        mask=jnp.ones((1,4,4),bool)
        reference,reference_cache=llm([x,None],positions,mask)
        out,cache,features=traced_llm(model,[x,None],positions,mask,[None,None])
        np.testing.assert_allclose(out[0],reference[0],atol=1e-6,rtol=1e-6)
        for actual,expected in zip(cache,reference_cache):
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(features['hidden'][:-1]+features['attn_update'],
                                   features['post_attention'],atol=1e-6,rtol=1e-6)
        y=jax.random.normal(jax.random.key(2),(1,2,32));cond=jnp.ones((1,32))
        positions=jnp.arange(4,6)[None];mask=jnp.ones((1,2,6),bool)
        reference,_=llm([None,y],positions,mask,adarms_cond=[None,cond],kv_cache=reference_cache)
        out,_,features=traced_llm(model,[None,y],positions,mask,[None,cond],cache)
        np.testing.assert_allclose(out[1],reference[1],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(features['post_attention']+features['ffn_update'],
                                   features['hidden'][1:],atol=1e-6,rtol=1e-6)


if __name__=='__main__':
    unittest.main()
