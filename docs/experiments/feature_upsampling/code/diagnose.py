#!/usr/bin/env python3
"""Diagnostics: (1) residual-injection tests, (2) resolution sweep, (3) guide quality."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import upsamplers as up


def to_t(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def head_apply(x, W, bias):
    return torch.einsum("bchw,cd->bdhw", x, W) + bias.view(1, -1, 1, 1)


def metrics(P, T):
    m = common.feature_metrics(common.grid_to_tokens(P.cpu().numpy())[0],
                               common.grid_to_tokens(T.cpu().numpy())[0])
    return m


def residual_tests(tag="smoke", limit=8):
    d = np.load(common.EXP_ROOT / "data" / f"features_{tag}.npz")
    n = min(limit, d["teacher_encoded"].shape[0])
    dev = "cuda:0"
    results = {}
    for i in range(n):
        T = to_t(common.tokens_to_grid(d["teacher_encoded"][i]), dev)[None]
        stem = to_t(d["stem"][i].transpose(2, 0, 1), dev)[None]
        L_real = to_t(common.tokens_to_grid(d["student_encoded"][i], (8, 8)), dev)[None]
        L_or = F.avg_pool2d(T, 2)
        pos16 = to_t(d["pos16"].transpose(2, 0, 1), dev)[None]
        pos8 = to_t(d["pos8"].transpose(2, 0, 1), dev)[None]
        z_hi = stem + pos16
        for src, name in ((L_or, "oracle"), (L_real, "real112")):
            base = up.interp(src, mode="bilinear")
            zc = F.avg_pool2d(z_hi, 2)
            base_z = up.interp(zc, mode="bilinear")
            dZ = z_hi - base_z
            # align residual RMS to base feature RMS
            scale = base.std() / (dZ.std() + 1e-8)
            cand = {"base": base, "z_hi_direct": z_hi, "dZ_only": dZ}
            for a in (0.05, 0.1, 0.25, 0.5, 1.0):
                cand[f"base+{a}zres"] = base + a * dZ
                cand[f"base+{a}zres_rms"] = base + a * dZ * scale
            for k, P in cand.items():
                m = metrics(P, T)
                results.setdefault((name, k), []).append(m)
        # guide correlations
        zc_hi = F.avg_pool2d(z_hi, 2)
        for label, A, B in (("T_vs_zhi", T, z_hi), ("T_vs_bilinear_L", T, up.interp(F.avg_pool2d(T, 2), mode="bilinear"))):
            a = common.grid_to_tokens(A.cpu().numpy())[0]
            b = common.grid_to_tokens(B.cpu().numpy())[0]
            results.setdefault((label, "align"), []).append(common.feature_metrics(b, a))
    for (src, k), rs in sorted(results.items()):
        if not rs:
            continue
        cos = np.mean([r["cos_mean"] for r in rs])
        rl2 = np.mean([r["rel_l2"] for r in rs])
        print(f"{src:18s} {k:22s} cos={cos:.4f} relL2={rl2:.4f}")


def resolution_sweep(tag="smoke", limit=8):
    import jax.numpy as jnp

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    model = common.SigLip.load(dtype="float32")
    d = np.load(common.EXP_ROOT / "data" / f"features_{tag}.npz")
    n = min(limit, d["teacher_encoded"].shape[0])
    dev = "cuda:0"
    for res in (112, 126, 140, 154, 168, 182, 196, 210):
        outs = []
        for i in range(n):
            rgb112 = d["rgb112_u8"][i]  # placeholder, redo resize from 224 below
            rgb224 = d["rgb224_u8"][i]
            import jax
            x = jnp.asarray(rgb224)
            from openpi.shared.image_tools import resize_with_pad
            xr = np.asarray(resize_with_pad(x, res, res)).astype(np.float32) / 255 * 2 - 1
            enc = model.encode(xr[None], res)["encoded"][0]  # [n_tok,1152]
            grid = common.tokens_to_grid(enc, (res // 14, res // 14))
            outs.append(grid)
        cos_l, rl2_l = [], []
        for i in range(n):
            T = to_t(common.tokens_to_grid(d["teacher_encoded"][i]), dev)[None]
            L = to_t(outs[i], dev)[None]
            P = up.interp(L, size=(16, 16), mode="bilinear")
            m = metrics(P, T)
            cos_l.append(m["cos_mean"])
            rl2_l.append(m["rel_l2"])
        print(f"res={res:4d} tok={(res // 14) ** 2:4d} bilinear->256 cos={np.mean(cos_l):.4f} relL2={np.mean(rl2_l):.4f}")


if __name__ == "__main__":
    print("=== residual injection ===")
    residual_tests()
    print("=== resolution sweep ===")
    resolution_sweep()
