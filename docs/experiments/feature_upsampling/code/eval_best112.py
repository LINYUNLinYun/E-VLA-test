#!/usr/bin/env python3
"""Upsampler sweep on top of the best 112 route (nearest-interpolated pos-emb)."""
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
import upsamplers as up


def to_t(x, dev="cuda:0"):
    return torch.as_tensor(np.asarray(x), device=dev, dtype=torch.float32)


def main(tag="libero"):
    import jax
    import jax.numpy as jnp
    from openpi.models import siglip as _siglip

    d = np.load(common.EXP_ROOT / "data" / f"features_{tag}.npz")
    n = d["teacher_encoded"].shape[0]
    model = common.SigLip.load(dtype="float32")
    P = jax.tree.map(jnp.asarray, model.params)
    static = {k: v for k, v in P.items() if k != "pos_embedding"}
    mod = _siglip.Module(num_classes=2048, variant="So400m/14", pool_type="none", scan=True, dtype_mm="float32")

    @jax.jit
    def f(x, pos):
        return mod.apply({"params": {**static, "pos_embedding": pos}}, x, train=False)[1]["encoded"]

    x112 = jnp.asarray(d["rgb112_u8"].astype(np.float32) / 255.0 * 2.0 - 1.0)
    enc = np.asarray(jax.block_until_ready(f(x112, jnp.asarray(model.interp_pos((8, 8), "nearest")))))
    print("student encoded", enc.shape, flush=True)

    pos16 = to_t(d["pos16"].transpose(2, 0, 1))[None]
    pos8 = to_t(model.interp_pos((8, 8), "nearest").reshape(8, 8, 1152).transpose(2, 0, 1))[None]
    W = to_t(d["head_kernel"]); bias = to_t(d["head_bias"])
    res = {}
    for i in range(n):
        T = to_t(common.tokens_to_grid(d["teacher_encoded"][i]))[None]
        L = to_t(common.tokens_to_grid(enc[i], (8, 8)))[None]
        stem = to_t(d["stem"][i].transpose(2, 0, 1))[None]
        rgb = to_t(d["rgb224_u8"][i].astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        z_hi = stem + pos16
        z_co = F.avg_pool2d(stem, 2) + pos8
        g_hi = F.adaptive_avg_pool2d(rgb, (16, 16))
        g_lo = F.adaptive_avg_pool2d(rgb, (8, 8))
        preds = {
            "nearest": up.interp(L, mode="nearest"),
            "bilinear": up.interp(L, mode="bilinear"),
            "bicubic": up.interp(L, mode="bicubic"),
            "lanczos": up.lanczos(L),
            "jbu3": up.jbu(L, g_hi, g_lo, radius=1, sigma_s=1.0, sigma_r=0.15),
            "jbu5": up.jbu(L, g_hi, g_lo, radius=2, sigma_s=1.5, sigma_r=0.15),
            "guided_filter": up.guided_filter(L, g_hi, g_lo, radius=2, eps=1e-3),
            "patch_guided": up.patch_guided(L, z_hi, z_co, radius=1, tau=0.2, lam=2.0),
        }
        for k, Q in preds.items():
            m = common.feature_metrics(common.grid_to_tokens(Q.cpu().numpy())[0],
                                       common.grid_to_tokens(T.cpu().numpy())[0])
            res.setdefault(k, []).append(m)
        if (i + 1) % 50 == 0:
            print(i + 1, flush=True)
    for k, rs in res.items():
        print(f"real112-nearest {k:14s} cos={np.mean([r['cos_mean'] for r in rs]):.4f} "
              f"p10={np.mean([r['cos_p10'] for r in rs]):.4f} "
              f"relL2={np.mean([r['rel_l2'] for r in rs]):.4f}", flush=True)


if __name__ == "__main__":
    main()
