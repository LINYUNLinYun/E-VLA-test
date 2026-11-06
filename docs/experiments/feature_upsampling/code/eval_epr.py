#!/usr/bin/env python3
"""Evaluate the EPR training-free upsampler on an extracted feature set.

Also verifies that EPR with prefix_blocks=DEPTH reproduces the 224 teacher exactly.
Writes results/epr_metrics.csv and results/epr_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from epr import EPRRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="libero")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="epr")
    ap.add_argument("--sanity-full", action="store_true", default=False)
    ap.add_argument("--prefixes", nargs="+", type=int, default=[2, 5, 9, 17])
    ap.add_argument("--grids", nargs="+", type=int, default=[8, 12])
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    d = np.load(common.EXP_ROOT / "data" / f"features_{args.tag}.npz")
    n = d["teacher_encoded"].shape[0]
    if args.limit:
        n = min(n, args.limit)
    u8 = d["rgb224_u8"][:n].astype(np.float32) / 255.0 * 2.0 - 1.0
    teacher = d["teacher_encoded"][:n]
    print(f"{n} images, teacher {teacher.shape}", flush=True)

    t0 = time.time()
    model = common.SigLip.load(dtype="float32")
    params = jax.tree.map(jnp.asarray, model.params)
    print(f"checkpoint loaded in {time.time() - t0:.1f}s", flush=True)

    rows = []

    def run(prefix, grid, alphas, tag):
        runner = EPRRunner(params=params, prefix_blocks=prefix, grid=grid, dtype="float32")
        for alpha in alphas:
            tokens = []
            for s in range(0, n, args.batch):
                e = min(n, s + args.batch)
                tok, logits, _ = runner(u8[s:e], alpha)
                tokens.append(np.asarray(jax.block_until_ready(tok)))
            pred = np.concatenate(tokens, 0)
            for i in range(n):
                m = common.feature_metrics(pred[i], teacher[i])
                m["image_id"] = i
                m["method"] = f"epr_p{prefix}_g{grid}"
                m["source"] = tag
                m["alpha"] = alpha
                m["pos_emb_method"] = "bicubic"
                m["spatial_delta"] = common.spatial_delta(
                    common.tokens_to_grid(pred[i]).transpose(1, 2, 0)
                )
                rows.append(m)
            cos = np.mean([r["cos_mean"] for r in rows[-n:]])
            rl2 = np.mean([r["rel_l2"] for r in rows[-n:]])
            print(f"{tag:10s} p={prefix:2d} g={grid:2d} a={alpha:4.2f}  cos={cos:.4f} relL2={rl2:.4f}",
                  flush=True)

    if args.sanity_full:
        r = EPRRunner(params=params, prefix_blocks=common.DEPTH, grid=8, dtype="float32")
        tok = []
        for s in range(0, min(n, 32), 8):
            e = min(min(n, 32), s + 8)
            t, _, _ = r(u8[s:e])
            tok.append(np.asarray(jax.block_until_ready(t)))
        t = np.concatenate(tok, 0)
        diff = np.abs(t - teacher[: len(t)])
        print(f"SANITY full path: max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e} "
              f"rel={np.linalg.norm(diff) / np.linalg.norm(teacher[: len(t)]):.3e}", flush=True)

    for g in args.grids:
        for p in args.prefixes:
            run(p, g, (0.0, 0.05, 0.1), "epr")
            jax.clear_caches()
    if args.sanity_full:
        run(9, 16, (0.0,), "sanity")

    out_csv = common.EXP_ROOT / "results" / f"{args.out}_metrics.csv"
    fields = ["image_id", "method", "source", "alpha", "pos_emb_method", "cos_mean", "cos_median",
              "cos_p10", "rel_l2", "norm_ratio", "spatial_delta"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_csv, flush=True)


if __name__ == "__main__":
    main()
