#!/usr/bin/env python3
"""Data-dependent optimal linear upsampling.

Estimate the spatial mixing matrix from the high-res early-layer feature Hk (which we have)
and the pooled low-res tokens, then apply it to the final-layer low-res tokens.
A is computed per image by ridge least squares -> no training, no learned parameters.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def to_t(x, device="cuda:0"):
    return torch.as_tensor(np.array(x), device=device, dtype=torch.float32)


def ridge_map(X, Y, rel_lambda=1e-3):
    """Solve min_A ||Y - A X||^2 + lam||A||^2 ; X [n,d], Y [m,d] -> A [m,n]."""
    n, d = X.shape
    lam = rel_lambda * (X * X).sum() / n
    G = X @ X.T + lam * torch.eye(n, device=X.device, dtype=X.dtype)  # [n,n]
    A = Y @ X.T @ torch.linalg.inv(G)  # [m,n]
    return A


def main(tag="smoke", limit=8):
    import jax
    import jax.numpy as jnp
    from openpi.models import siglip as _siglip
    from openpi.shared.image_tools import resize_with_pad

    model = common.SigLip.load(dtype="float32")
    P = jax.tree.map(jnp.asarray, model.params)
    mod = _siglip.Module(num_classes=common.NUM_CLASSES, variant="So400m/14", pool_type="none", scan=True,
                         dtype_mm="float32")

    @jax.jit
    def apply(p, x):
        return mod.apply({"params": p}, x, train=False)

    d = np.load(common.EXP_ROOT / "data" / f"features_{tag}.npz")
    n = min(limit, d["teacher_encoded"].shape[0])
    Ts, Hs, Stems = [], [], []
    for i in range(n):
        x = jnp.asarray(d["rgb224_u8"][i][None]).astype(jnp.float32) / 255 * 2 - 1
        logits, out = apply(P, x)
        Ts.append(to_t(common.tokens_to_grid(np.asarray(out["encoded"][0])))[None])
        Stems.append(to_t(np.asarray(out["stem"][0]).transpose(2, 0, 1))[None])
        Hs.append([to_t(common.tokens_to_grid(np.asarray(out["encoder"][f"block{k:02d}"]["+mlp"])[0]))[None]
                   for k in range(27)])

    _cache = {}

    def get_runner(depth):
        if depth not in _cache:
            enc = _siglip.Encoder(depth=depth, mlp_dim=4304, num_heads=16, scan=True, dtype_mm="float32")

            @jax.jit
            def f(prm_, x_):
                y, _ = enc.apply({"params": prm_}, x_, deterministic=True)
                return y

            _cache[depth] = f
        return _cache[depth]

    def tokens(x):
        return x.permute(0, 2, 3, 1).reshape(1, -1, 1152)

    print("k g lam  cos_mean  rel_l2   (A from L->Hk applied to S; and oracle ceiling)")
    for k in (0, 1, 4, 8):
        tail_params = {"encoderblock": jax.tree.map(lambda v: v[k:], P["Transformer"]["encoderblock"]),
                       "encoder_norm": P["Transformer"]["encoder_norm"]}
        for g in (8,):
            for lam in (1e-4, 1e-3, 1e-2, 1e-1):
                cs, rls, ceil = [], [], []
                for i in range(n):
                    Hk = Hs[i][k] if k > 0 else Stems[i]
                    L = F.adaptive_avg_pool2d(Hk, (g, g))
                    y = np.asarray(get_runner(27 - k)(tail_params, jnp.asarray(tokens(L).cpu().numpy())))
                    S = torch.as_tensor(np.ascontiguousarray(y[0].T.reshape(1152, g, g)), device="cuda:0")[None]
                    Hx = tokens(Hk).to("cuda:0")[0]  # [256,1152]
                    Lx = tokens(L).to("cuda:0")[0]   # [g*g,1152]
                    Sx = tokens(S).to("cuda:0")[0]
                    A = ridge_map(Lx, Hx, lam)        # [256,64]
                    pred = (A @ Sx).T.reshape(1, 1152, 16, 16)
                    m = common.feature_metrics(common.grid_to_tokens(pred.cpu().numpy())[0],
                                               common.grid_to_tokens(Ts[i].cpu().numpy())[0])
                    cs.append(m["cos_mean"]); rls.append(m["rel_l2"])
                    ceil.append(float(torch.linalg.matrix_rank(Sx.float())))
                print(f"{k:2d} {g:2d} {lam:5.0e}  {np.mean(cs):.4f}  {np.mean(rls):.4f}", flush=True)


if __name__ == "__main__":
    main()
