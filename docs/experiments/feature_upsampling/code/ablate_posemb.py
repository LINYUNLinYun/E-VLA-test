#!/usr/bin/env python3
"""Ablate the interpolation method for the learned 16x16 position embedding -> 8x8.

Also reports the scale distribution of the interpolated embedding (the doc asks whether
the learned positional signal drifts when resampled).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import common


def main(tag="libero", limit=32):
    import jax
    import jax.numpy as jnp
    from openpi.models import siglip as _siglip

    d = np.load(common.EXP_ROOT / "data" / f"features_{tag}.npz")
    n = min(limit, d["teacher_encoded"].shape[0])
    model = common.SigLip.load(dtype="float32")
    P = jax.tree.map(jnp.asarray, model.params)
    static = {k: v for k, v in P.items() if k != "pos_embedding"}
    mod = _siglip.Module(num_classes=2048, variant="So400m/14", pool_type="none", scan=True, dtype_mm="float32")

    @jax.jit
    def f(x, pos):
        return mod.apply({"params": {**static, "pos_embedding": pos}}, x, train=False)

    x112 = jnp.asarray(d["rgb112_u8"][:n].astype(np.float32) / 255.0 * 2.0 - 1.0)
    teacher = d["teacher_encoded"][:n]

    p16 = np.asarray(P["pos_embedding"]).reshape(16, 16, 1152)
    print("pos16 norm: mean=%.3f std=%.3f per-dim-mean=%.4f" % (
        np.linalg.norm(p16, axis=-1).mean(), np.linalg.norm(p16, axis=-1).std(), p16.mean()))
    candidates = {}
    for method in ("nearest", "linear", "bicubic", "lanczos3"):
        candidates[method] = model.interp_pos((8, 8), method)
    p16j = np.asarray(P["pos_embedding"]).reshape(1, 16, 16, 1152)
    candidates["avg2"] = p16j.reshape(1, 8, 2, 8, 2, 1152).mean(axis=(2, 4)).reshape(1, 64, 1152)
    near3 = candidates["nearest"]
    n16 = np.linalg.norm(p16j, axis=-1).mean()
    n8 = np.linalg.norm(near3, axis=-1).mean()
    candidates["nearest_scaled"] = near3 * (n16 / n8)

    for method, pos8 in candidates.items():
        pg = np.asarray(pos8).reshape(8, 8, 1152)
        logits, out = f(x112, jnp.asarray(pos8))
        enc = np.asarray(jax.block_until_ready(out["encoded"]))
        cs, rls = [], []
        for i in range(n):
            g = common.tokens_to_grid(enc[i], (8, 8))
            up = F.interpolate(torch.as_tensor(g)[None], size=(16, 16), mode="bilinear", align_corners=False)
            m = common.feature_metrics(common.grid_to_tokens(up.numpy())[0], teacher[i])
            cs.append(m["cos_mean"]); rls.append(m["rel_l2"])
        print(f"{method:9s} cos={np.mean(cs):.4f} relL2={np.mean(rls):.4f} | "
              f"pos8 norm mean={np.linalg.norm(pg, axis=-1).mean():.3f} "
              f"std={np.linalg.norm(pg, axis=-1).std():.3f} mean-abs={np.abs(pg).mean():.4f}")


if __name__ == "__main__":
    main()
