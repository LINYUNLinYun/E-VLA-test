"""Checkpoint-free matched Full parity. Execute with JAX_PLATFORMS=cpu."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest


class VisionTests(unittest.TestCase):
    def test_native_scanned_and_manual_full_keep_identical_tokens_and_values(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi.models import siglip
        from .vision import install
        module = siglip.Encoder(depth=27,mlp_dim=16,num_heads=2,scan=True,dtype_mm="float32")
        x = jax.random.normal(jax.random.key(13),(1,256,8))
        params = module.init(jax.random.key(7),x,deterministic=True)
        original = siglip.Encoder.__call__
        reference,_ = module.apply(params,x,deterministic=True)
        try:
            install([],manual_full=True)
            matched,_ = module.apply(params,x,deterministic=True)
            self.assertEqual(matched.shape,(1,256,8))
            np.testing.assert_allclose(reference,matched,atol=1e-5,rtol=1e-5)
        finally:
            siglip.Encoder.__call__ = original
            if hasattr(siglip.Encoder,"_v31_original_call"):
                delattr(siglip.Encoder,"_v31_original_call")


if __name__ == "__main__":
    unittest.main()
