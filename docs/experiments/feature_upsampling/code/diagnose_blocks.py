#!/usr/bin/env python3
"""Block-wise diagnostics.

1. Linear ceiling (per-image optimal linear upsampler) vs number of tokens.
2. Correlation of intermediate block features / high-frequency residuals with the final feature.
3. Hybrid early-exact / late-cheap reconstruction with high-frequency residual reinjection.
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
import upsamplers as up


def to_t(x, device="cuda:0"):
    return torch.as_tensor(np.array(x), device=device, dtype=torch.float32)


def highfreq(x):
    return x - F.interpolate(F.avg_pool2d(x, 2), size=x.shape[-2:], mode="bilinear", align_corners=False)


def cos_flat(a, b):
    a = a.flatten(); b = b.flatten()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


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
    pos16 = to_t(d["pos16"].transpose(2, 0, 1))[None]
    pos8 = to_t(d["pos8"].transpose(2, 0, 1))[None]

    # ---------- intermediate features for each image
    Hs = []  # list over images of list of [1,C,16,16] blocks
    Ts = []
    for i in range(n):
        x = jnp.asarray(d["rgb224_u8"][i][None]).astype(jnp.float32) / 255 * 2 - 1
        _, out = apply(P, x)
        Ts.append(to_t(common.tokens_to_grid(np.asarray(out["encoded"][0])))[None])
        # block output hidden state = out["+mlp"]; scan stacks all layers on axis 0
        Hs.append([to_t(common.tokens_to_grid(np.asarray(out["encoder"][f"block{k:02d}"]["+mlp"])[0]))[None]
                   for k in range(27)])
        print(f"enc {i + 1}/{n}", flush=True)

    print("=== block corr with final ===")
    print(" block  cos(feat)  cos(highfreq)")
    for k in range(27):
        cf = np.mean([cos_flat(Ts[i], Hs[i][k]) for i in range(n)])
        ch = np.mean([cos_flat(highfreq(Ts[i]), highfreq(Hs[i][k])) for i in range(n)])
        print(f" {k:5d}  {cf:9.4f}  {ch:12.4f}", flush=True)

    # ---------- hybrid: blocks 0..k-1 at 256, pool, tail at 64, residual reinjection
    print("=== hybrid early-exact / late-cheap ===", flush=True)
    for k in (2, 4, 6, 8, 12, 16, 20):
        depth = 27 - k
        enc = _siglip.Encoder(depth=depth, mlp_dim=4304, num_heads=16, scan=True, dtype_mm="float32")
        tail = jax.tree.map(lambda v: v[k:], P["Transformer"]["encoderblock"])
        tail_params = {"encoderblock": tail, "encoder_norm": P["Transformer"]["encoder_norm"]}
        # jitted tail
        @jax.jit
        def tail_apply(prm, x):
            y, _ = enc.apply({"params": prm}, x, deterministic=True)
            return y

        best = {}
        for alpha in (0.0, 0.025, 0.05, 0.1, 0.15, 0.25, 0.4):
            cos_l, rl_l = [], []
            for i in range(n):
                Hk = Hs[i][k]
                low = F.avg_pool2d(Hk, 2)
                # block-k hidden states already carry positional information; do not add pos8 again
                x64 = jnp.asarray(low.permute(0, 2, 3, 1).reshape(1, 64, 1152).cpu().numpy())
                y = np.asarray(tail_apply(tail_params, x64))  # [1,64,1152]
                S = torch.as_tensor(y[0].T.reshape(1152, 8, 8), device="cuda:0")[None]
                base = F.interpolate(S, size=(16, 16), mode="bilinear", align_corners=False)
                pred = base + alpha * highfreq(Hk)
                m = common.feature_metrics(common.grid_to_tokens(pred.cpu().numpy())[0],
                                           common.grid_to_tokens(Ts[i].cpu().numpy())[0])
                cos_l.append(m["cos_mean"]); rl_l.append(m["rel_l2"])
            best[alpha] = (np.mean(cos_l), np.mean(rl_l))
        print(f"k={k:2d} " + "  ".join(f"a={a}:{c:.4f}/{r:.4f}" for a, (c, r) in best.items()), flush=True)


if __name__ == "__main__":
    main()
