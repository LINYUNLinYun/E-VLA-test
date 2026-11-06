#!/usr/bin/env python3
"""Full-set (288) real112 baseline for several pos-embedding interpolation methods."""
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


def main(tag="libero"):
    import jax
    import jax.numpy as jnp
    from openpi.models import siglip as _siglip

    d = np.load(common.EXP_ROOT / "data" / f"features_{tag}.npz")
    n = d["teacher_encoded"].shape[0]
    teacher = d["teacher_encoded"]
    model = common.SigLip.load(dtype="float32")
    P = jax.tree.map(jnp.asarray, model.params)
    static = {k: v for k, v in P.items() if k != "pos_embedding"}
    mod = _siglip.Module(num_classes=2048, variant="So400m/14", pool_type="none", scan=True, dtype_mm="float32")

    @jax.jit
    def f(x, pos):
        return mod.apply({"params": {**static, "pos_embedding": pos}}, x, train=False)[1]["encoded"]

    x = jnp.asarray(d["rgb112_u8"].astype(np.float32) / 255.0 * 2.0 - 1.0)
    print("pos16 mean-norm", float(np.linalg.norm(np.asarray(P["pos_embedding"]), axis=-1).mean()))
    for method in ("nearest", "linear", "bicubic", "lanczos3"):
        pos8 = jnp.asarray(model.interp_pos((8, 8), method))
        enc = np.asarray(jax.block_until_ready(f(x, pos8)))
        cs, rls, p10 = [], [], []
        for i in range(n):
            g = common.tokens_to_grid(enc[i], (8, 8))
            up = F.interpolate(torch.as_tensor(g)[None], size=(16, 16), mode="bilinear", align_corners=False)
            m = common.feature_metrics(common.grid_to_tokens(up.numpy())[0], teacher[i])
            cs.append(m["cos_mean"]); rls.append(m["rel_l2"]); p10.append(m["cos_p10"])
        print(f"real112 {method:9s} cos={np.mean(cs):.4f} p10={np.mean(p10):.4f} "
              f"relL2={np.mean(rls):.4f} pos8_norm={float(jnp.linalg.norm(pos8, axis=-1).mean()):.2f}", flush=True)


if __name__ == "__main__":
    main()
