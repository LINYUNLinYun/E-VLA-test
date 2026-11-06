"""EPR: Early-exact Prefix + pooled-tail + Residual reinjection.

Training-free feature upsampling for SigLIP So400m/14.

Pipeline (no learned parameters, checkpoint untouched):
  1. patch-embed at full 224 resolution (stem)                  [exact]
  2. run `prefix_blocks` transformer blocks at 256 tokens        [exact]
  3. average-pool the hidden state to g x g tokens               [cheap]
  4. run the remaining 27-prefix_blocks blocks on g*g tokens     [cheap]
  5. half-pixel bilinear upsample back to 16x16                  [cheap]
  6. add alpha * (H_p - bilinear(pool(H_p)))                     [cheap]
     where H_p is the high-res prefix hidden state, i.e. the high-frequency
     detail that the pooled tail never sees.

The returned token count/shape/dtype is identical to the 224 teacher.
Falls back to the exact 224 forward when prefix_blocks == 27.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import common

STEM_NAME = "embedding"
MLP_DIM = 4304
NUM_HEADS = 16


# --------------------------------------------------------------------------- ops
def resize_bilinear(x: jnp.ndarray, out_h: int, out_w: int) -> jnp.ndarray:
    """Torch-compatible bilinear resize (align_corners=False), x [B,H,W,C]."""
    b, h, w, c = x.shape
    yi = (jnp.arange(out_h, dtype=jnp.float32) + 0.5) * (h / out_h) - 0.5
    xi = (jnp.arange(out_w, dtype=jnp.float32) + 0.5) * (w / out_w) - 0.5
    y0 = jnp.floor(yi).astype(jnp.int32)
    x0 = jnp.floor(xi).astype(jnp.int32)
    wy = (yi - y0).astype(jnp.float32)
    wx = (xi - x0).astype(jnp.float32)
    y1 = jnp.clip(y0 + 1, 0, h - 1)
    x1 = jnp.clip(x0 + 1, 0, w - 1)
    y0 = jnp.clip(y0, 0, h - 1)
    x0 = jnp.clip(x0, 0, w - 1)

    def take(a, iy, ix):
        return a[:, iy][:, :, ix]

    v00 = take(x, y0, x0)
    v01 = take(x, y0, x1)
    v10 = take(x, y1, x0)
    v11 = take(x, y1, x1)
    wy = wy[None, :, None, None]
    wx = wx[None, None, :, None]
    return (v00 * (1 - wy) * (1 - wx) + v01 * (1 - wy) * wx + v10 * wy * (1 - wx) + v11 * wy * wx)


def avgpool2(x: jnp.ndarray) -> jnp.ndarray:
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).mean(axis=(2, 4))


def pool_matrix(n_in: int, n_out: int) -> jnp.ndarray:
    """PyTorch adaptive_avg_pool1d weights [n_out, n_in]."""
    m = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        lo = (i * n_in) // n_out
        hi = -(-(i + 1) * n_in // n_out)
        m[i, lo:hi] = 1.0 / (hi - lo)
    return jnp.asarray(m)


def adaptive_avg_pool2d(x: jnp.ndarray, out_h: int, out_w: int) -> jnp.ndarray:
    """x [B,H,W,C] -> [B,out_h,out_w,C], matches torch.nn.functional.adaptive_avg_pool2d.

    HIGHEST precision: XLA-GPU otherwise lowers these float32 matmuls to TF32
    (~1e-3 relative error on the pooled features).
    """
    from jax.lax import Precision

    mh = pool_matrix(x.shape[1], out_h)
    mw = pool_matrix(x.shape[2], out_w)
    return jnp.einsum("ih,bhwc,jw->bijc", mh, x, mw, precision=Precision.HIGHEST)


# --------------------------------------------------------------------------- model
def _stem_module():
    import flax.linen as nn

    class Stem(nn.Module):
        width: int = common.WIDTH
        patch: int = 14

        @nn.compact
        def __call__(self, image):
            image = jnp.asarray(image, jnp.float32)
            return nn.Conv(
                self.width,
                (self.patch, self.patch),
                strides=(self.patch, self.patch),
                padding="VALID",
                name=STEM_NAME,
                dtype=jnp.float32,
            )(image)

    return Stem()


@dataclasses.dataclass
class EPRRunner:
    """Compiled EPR pipeline for a fixed configuration."""

    params: dict
    prefix_blocks: int = 2
    grid: int = 8
    alpha: float = 0.05
    dtype: str = "float32"
    pos_method: str = "bicubic"

    def __post_init__(self):
        from openpi.models import siglip as _siglip

        p = self.prefix_blocks
        depth = common.DEPTH
        assert 0 <= p <= depth
        tp = self.params["Transformer"]
        blocks = tp["encoderblock"]

        self.stem = _stem_module()
        self.stem_params = {"params": {STEM_NAME: self.params[STEM_NAME]}}

        self.prefix = _siglip.Encoder(depth=p, mlp_dim=MLP_DIM, num_heads=NUM_HEADS, scan=True,
                                      dtype_mm=self.dtype)
        self.prefix_params = {
            "params": {
                "encoderblock": jax.tree.map(lambda v: v[:p], blocks),
                "encoder_norm": tp["encoder_norm"],
            }
        }
        self.tail = _siglip.Encoder(depth=depth - p, mlp_dim=MLP_DIM, num_heads=NUM_HEADS, scan=True,
                                    dtype_mm=self.dtype)
        self.tail_params = {
            "params": {
                "encoderblock": jax.tree.map(lambda v: v[p:], blocks),
                "encoder_norm": tp["encoder_norm"],
            }
        }
        self.head_kernel = jnp.asarray(self.params["head"]["kernel"])
        self.head_bias = jnp.asarray(self.params["head"]["bias"])

        self.pos16 = jnp.asarray(self.params["pos_embedding"]).reshape(1, 256, common.WIDTH).astype(jnp.float32)

        self._jitted = jax.jit(self._forward)

    # -- forward -----------------------------------------------------------
    def _forward(self, image: jnp.ndarray, alpha: float):
        b = image.shape[0]
        c = common.WIDTH
        hw = image.shape[1] // 14
        assert hw == 16, f"EPR expects 224x224 input, got {image.shape[1:3]}"

        stem = self.stem.apply(self.stem_params, image)  # [B,16,16,C] float32
        z = (stem.reshape(b, hw * hw, c) + self.pos16).astype(self.dtype)

        if self.prefix_blocks == 0:
            hp = z  # dtype
            prefix_norm = None
        else:
            prefix_norm, pre_out = self.prefix.apply(self.prefix_params, z, deterministic=True)
            hp = pre_out[f"block{self.prefix_blocks - 1:02d}"]["+mlp"]
        hp = hp.astype(jnp.float32)
        hg = hp.reshape(b, hw, hw, c)

        if self.prefix_blocks == common.DEPTH:
            # exact 224 path (reference / sanity check)
            tokens = prefix_norm.astype(jnp.float32)
            logits = tokens.astype(self.dtype) @ self.head_kernel + self.head_bias
            return tokens, logits, hp

        g = self.grid
        if hw % g == 0:
            lg = hg.reshape(b, g, hw // g, g, hw // g, c).mean(axis=(2, 4))
        else:
            lg = adaptive_avg_pool2d(hg, g, g)
        l = lg.reshape(b, g * g, c).astype(self.dtype)

        s, _ = self.tail.apply(self.tail_params, l, deterministic=True)
        sg = s.astype(jnp.float32).reshape(b, g, g, c)

        up = resize_bilinear(sg, hw, hw)
        detail = hg - resize_bilinear(lg, hw, hw)
        pred = up + alpha * detail
        tokens = pred.reshape(b, hw * hw, c)
        logits = tokens.astype(self.dtype) @ self.head_kernel + self.head_bias
        return tokens, logits, hp

    def __call__(self, image: np.ndarray | jnp.ndarray, alpha: float | None = None):
        """image [B,224,224,3] float in [-1,1] -> tokens [B,256,1152], logits [B,256,2048]."""
        return self._jitted(jnp.asarray(image), self.alpha if alpha is None else float(alpha))

    # -- stages for latency breakdown --------------------------------------
    def stages(self):
        """Return dict name -> jitted callable taking [B,224,224,3]."""
        b = 1
        c = common.WIDTH
        hw = 16
        g = self.grid

        @jax.jit
        def stem_only(image):
            return self.stem.apply(self.stem_params, image)

        @jax.jit
        def prefix_only(image):
            stem = self.stem.apply(self.stem_params, image)
            z = (stem.reshape(b, hw * hw, c) + self.pos16).astype(self.dtype)
            _, out = self.prefix.apply(self.prefix_params, z, deterministic=True)
            return out[f"block{self.prefix_blocks - 1:02d}"]["+mlp"]

        @jax.jit
        def tail_only(l):
            s, _ = self.tail.apply(self.tail_params, l, deterministic=True)
            return s

        @jax.jit
        def upsample_only(sg):
            up = resize_bilinear(sg, hw, hw)
            return up

        @jax.jit
        def head_only(tokens):
            return tokens.astype(self.dtype) @ self.head_kernel + self.head_bias

        return {
            "stem": stem_only,
            "prefix": prefix_only,
            "tail": tail_only,
            "upsample": upsample_only,
            "head": head_only,
        }
