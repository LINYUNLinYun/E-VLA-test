#!/usr/bin/env python3
"""Phase 1: offline feature reconstruction error for all training-free upsamplers."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

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
    """1x1 linear projector on [B,C,H,W]."""
    return torch.einsum("bchw,cd->bdhw", x, W) + bias.view(1, -1, 1, 1)


def build_guides(stem, pos16, pos8, rgb224, use_pos=True):
    """Patch-embedding guides and RGB guides at 16x16 / 8x8."""
    z_hi = stem
    z_co = F.avg_pool2d(stem, 2)
    if use_pos:
        z_hi = z_hi + pos16
        z_co = z_co + pos8
    g_hi = F.adaptive_avg_pool2d(rgb224, (16, 16))
    g_lo = F.adaptive_avg_pool2d(rgb224, (8, 8))
    return z_hi, z_co, g_hi, g_lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="libero")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="feature_metrics")
    args = ap.parse_args()

    d = np.load(common.EXP_ROOT / "data" / f"features_{args.tag}.npz")
    meta = json.loads((common.EXP_ROOT / "data" / f"meta_{args.tag}.json").read_text())
    n = d["teacher_encoded"].shape[0] if args.limit <= 0 else min(args.limit, d["teacher_encoded"].shape[0])
    dev = "cuda:0"

    W = to_t(d["head_kernel"], dev)
    bias = to_t(d["head_bias"], dev)
    pos16 = to_t(d["pos16"].transpose(2, 0, 1), dev)[None]
    pos8 = to_t(d["pos8"].transpose(2, 0, 1), dev)[None]

    rows = []
    t0 = time.time()
    for i in range(n):
        T = to_t(common.tokens_to_grid(d["teacher_encoded"][i]), dev)[None]  # [1,C,16,16]
        stem = to_t(d["stem"][i].transpose(2, 0, 1), dev)[None]
        rgb = to_t(d["rgb224_u8"][i].astype(np.float32) / 255.0, dev).permute(2, 0, 1)[None]
        z_hi, z_co, g_hi, g_lo = build_guides(stem, pos16, pos8, rgb)

        L_real = to_t(common.tokens_to_grid(d["student_encoded"][i], (8, 8)), dev)[None]
        L_oracle = F.avg_pool2d(T, 2)
        L_ps = to_t(common.tokens_to_grid(d["pooled_stem_encoded"][i], (8, 8)), dev)[None]

        preds = {}
        for src, L in (("real112", L_real), ("oracle_pool", L_oracle), ("pooled_stem", L_ps)):
            preds[(src, "nearest")] = up.interp(L, mode="nearest")
            preds[(src, "bilinear")] = up.interp(L, mode="bilinear")
            preds[(src, "bicubic")] = up.interp(L, mode="bicubic")
            preds[(src, "jbu3")] = up.jbu(L, g_hi, g_lo, radius=1, sigma_s=1.0, sigma_r=0.15)
            preds[(src, "jbu5")] = up.jbu(L, g_hi, g_lo, radius=2, sigma_s=1.5, sigma_r=0.15)
            preds[(src, "jbu3_spatial")] = up.jbu(L, g_hi, g_lo, radius=1, sigma_s=1.0, spatial_only=True)
            preds[(src, "patch_guided")] = up.patch_guided(L, z_hi, z_co, radius=1, tau=0.2, lam=2.0)
            preds[(src, "guided_filter")] = up.guided_filter(L, g_hi, g_lo, radius=2, eps=1e-3)
        preds[("real112", "lanczos")] = up.lanczos(L_real)

        for (src, method), P in preds.items():
            for space, A, B in (("pre", P, T), ("post", head_apply(P, W, bias), head_apply(T, W, bias))):
                m = common.feature_metrics(common.grid_to_tokens(A.cpu().numpy())[0],
                                           common.grid_to_tokens(B.cpu().numpy())[0])
                sd_pred = common.spatial_delta(common.grid_to_tokens(A.cpu().numpy())[0].reshape(16, 16, -1))
                sd_teach = common.spatial_delta(common.grid_to_tokens(B.cpu().numpy())[0].reshape(16, 16, -1))
                rows.append(dict(
                    image_id=i, suite=meta["records"][i]["suite"], camera=meta["records"][i]["camera"],
                    method=method, source=src, space=space, pos_emb_method="bicubic",
                    cos_mean=m["cos_mean"], cos_median=m["cos_median"], cos_p10=m["cos_p10"],
                    rel_l2=m["rel_l2"], norm_ratio=m["norm_ratio"],
                    spatial_delta=sd_pred / (sd_teach + 1e-12), valid=True,
                ))
        # teacher identity row
        m = common.feature_metrics(common.grid_to_tokens(T.cpu().numpy())[0], common.grid_to_tokens(T.cpu().numpy())[0])
        rows.append(dict(image_id=i, suite=meta["records"][i]["suite"], camera=meta["records"][i]["camera"],
                         method="none", source="teacher", space="pre", pos_emb_method="bicubic",
                         cos_mean=m["cos_mean"], cos_median=m["cos_median"], cos_p10=m["cos_p10"],
                         rel_l2=m["rel_l2"], norm_ratio=m["norm_ratio"], spatial_delta=1.0, valid=True))
        # L0 diagnostic: student 64 tokens vs pooled teacher tokens
        m = common.feature_metrics(d["student_encoded"][i], common.grid_to_tokens(F.avg_pool2d(T, 2).cpu().numpy())[0])
        rows.append(dict(image_id=i, suite=meta["records"][i]["suite"], camera=meta["records"][i]["camera"],
                         method="none_no_upsample", source="real112", space="pre_pooled_teacher",
                         pos_emb_method="bicubic", cos_mean=m["cos_mean"], cos_median=m["cos_median"],
                         cos_p10=m["cos_p10"], rel_l2=m["rel_l2"], norm_ratio=m["norm_ratio"],
                         spatial_delta=float("nan"), valid=True))
        # diagnostic: hard linear ceiling (rank-64 row space)
        for src, L in (("real112", L_real), ("pooled_stem", L_ps)):
            ceiling = float(up.oracle_linear_ceiling(L, T)[0])
            rows.append(dict(image_id=i, suite=meta["records"][i]["suite"], camera=meta["records"][i]["camera"],
                             method="linear_ceiling_rank64", source=src, space="pre",
                             pos_emb_method="bicubic", cos_mean=float("nan"), cos_median=float("nan"),
                             cos_p10=float("nan"), rel_l2=ceiling, norm_ratio=float("nan"),
                             spatial_delta=float("nan"), valid=True))

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{n} ({time.time() - t0:.1f}s)", flush=True)

    out = common.EXP_ROOT / "results" / f"{args.out}.csv"
    fields = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows), "rows")

    # aggregate
    agg = {}
    for r in rows:
        key = (r["source"], r["method"], r["space"])
        agg.setdefault(key, []).append(r)
    srows = []
    for (src, method, space), rs in agg.items():
        def mean(k):
            vals = [r[k] for r in rs if isinstance(r[k], float) and not np.isnan(r[k])]
            return float(np.mean(vals)) if vals else float("nan")
        srows.append(dict(source=src, method=method, space=space, n=len(rs),
                          cos_mean=mean("cos_mean"), cos_p10=mean("cos_p10"),
                          cos_median=mean("cos_median"), rel_l2=mean("rel_l2"),
                          norm_ratio=mean("norm_ratio"), spatial_delta=mean("spatial_delta")))
    srows.sort(key=lambda r: ((r["source"], r["space"], -r["cos_mean"] if not np.isnan(r["cos_mean"]) else 0)))
    sout = common.EXP_ROOT / "results" / f"{args.out}_summary.csv"
    with sout.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    print("wrote", sout)
    for r in srows:
        if r["space"] == "pre":
            print(f"{r['source']:12s} {r['method']:20s} cos={r['cos_mean']:.4f} p10={r['cos_p10']:.4f} "
                  f"relL2={r['rel_l2']:.4f} norm={r['norm_ratio']:.3f} sd={r['spatial_delta']:.3f}")


if __name__ == "__main__":
    main()
