#!/usr/bin/env python3
"""Frontier sweep: exact prefix (k blocks @256) -> pooled middle (g x g) -> exact suffix (s blocks @256)."""
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
    Ts, Hs = [], []
    for i in range(n):
        x = jnp.asarray(d["rgb224_u8"][i][None]).astype(jnp.float32) / 255 * 2 - 1
        _, out = apply(P, x)
        Ts.append(to_t(common.tokens_to_grid(np.asarray(out["encoded"][0])))[None])
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

    def run_blocks(prm, depth, x):
        return np.asarray(get_runner(depth)(prm, jnp.asarray(x)))

    print("k g s alpha  cos_mean  rel_l2")
    for k in (1, 4):
        for s in (0, 1, 2, 3):
            mid = P["Transformer"]["encoderblock"]
            for g in (8, 12):
                # middle = blocks k .. 26-s ; suffix = blocks 27-s .. 26
                mid_params = {"encoderblock": jax.tree.map(lambda v: v[k:27 - s], mid),
                              "encoder_norm": P["Transformer"]["encoder_norm"]}
                suf_params = {"encoderblock": jax.tree.map(lambda v: v[27 - s:], mid),
                              "encoder_norm": P["Transformer"]["encoder_norm"]}
                Ss, hfs, midstates = [], [], []
                for i in range(n):
                    Hk = Hs[i][k]
                    low = F.adaptive_avg_pool2d(Hk, (g, g))
                    x = jnp.asarray(low.permute(0, 2, 3, 1).reshape(1, g * g, 1152).cpu().numpy())
                    y = run_blocks(mid_params, 27 - s - k, x)
                    S = torch.as_tensor(np.ascontiguousarray(y[0].T.reshape(1152, g, g)), device="cuda:0")[None]
                    base = F.interpolate(S, size=(16, 16), mode="bilinear", align_corners=False)
                    hf = Hk - F.interpolate(low, size=(16, 16), mode="bilinear", align_corners=False)
                    Ss.append(base); hfs.append(hf)
                for alpha in (0.0, 0.05, 0.1, 0.2):
                    cs, rls = [], []
                    for i in range(n):
                        x256 = Ss[i] + alpha * hfs[i]
                        if s:
                            y = run_blocks(suf_params, s, x256.permute(0, 2, 3, 1).reshape(1, 256, 1152).cpu().numpy())
                            pred = torch.as_tensor(np.ascontiguousarray(y[0].T.reshape(1152, 16, 16)),
                                                   device="cuda:0")[None]
                        else:
                            pred = x256
                        m = common.feature_metrics(common.grid_to_tokens(pred.cpu().numpy())[0],
                                                   common.grid_to_tokens(Ts[i].cpu().numpy())[0])
                        cs.append(m["cos_mean"]); rls.append(m["rel_l2"])
                    print(f"{k:2d} {g:2d} {s:2d} {alpha:4.2f}  {np.mean(cs):.4f}  {np.mean(rls):.4f}", flush=True)


if __name__ == "__main__":
    main()
