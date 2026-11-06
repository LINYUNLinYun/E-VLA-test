#!/usr/bin/env python3
"""Visualize best/median/worst real112 bilinear cases with a shared per-row PCA basis.

Panels per row: input image | teacher feature PCA | upsampled student PCA | error heatmap.
"""
from __future__ import annotations

import csv
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import common

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pca_rgb(a, basis=None):
    x = a.reshape(-1, a.shape[-1]).astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    if basis is None:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        basis = vt[:3].T
    y = x @ basis
    y = (y - y.min(0)) / (y.max(0) - y.min(0) + 1e-8)
    return y.reshape(a.shape[0], a.shape[1], 3), basis


def main():
    tag = "libero"
    d = np.load(common.EXP_ROOT / "data" / f"features_{tag}.npz")
    rows = list(csv.DictReader(open(common.EXP_ROOT / "results" / "feature_metrics.csv")))
    real = [r for r in rows if r["source"] == "real112" and r["method"] == "bilinear" and r["space"] == "pre"]
    scores = {int(r["image_id"]): float(r["cos_mean"]) for r in real}
    ids = sorted(scores)
    vals = np.array([scores[i] for i in ids])
    order = np.argsort(vals)
    picks = [("worst", ids[int(order[0])]), ("median", ids[int(order[len(ids) // 2])]),
             ("best", ids[int(order[-1])])]

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for row, (label, i) in enumerate(picks):
        T = common.tokens_to_grid(d["teacher_encoded"][i], (16, 16))  # [C,16,16]
        S = common.tokens_to_grid(d["student_encoded"][i], (8, 8))
        up = F.interpolate(torch.as_tensor(S)[None], size=(16, 16), mode="bilinear",
                           align_corners=False)[0].numpy()
        Tn = np.transpose(T, (1, 2, 0))
        a, basis = pca_rgb(Tn)
        b, _ = pca_rgb(np.transpose(up, (1, 2, 0)), basis)
        err = np.linalg.norm(Tn - np.transpose(up, (1, 2, 0)), axis=-1)
        axes[row, 0].imshow(d["rgb224_u8"][i]); axes[row, 0].set_title(f"{label} #{i} input")
        axes[row, 1].imshow(a); axes[row, 1].set_title("teacher 224 feature PCA")
        axes[row, 2].imshow(b); axes[row, 2].set_title(f"112 bilinear PCA (cos={scores[i]:.3f})")
        im = axes[row, 3].imshow(err, cmap="magma"); axes[row, 3].set_title("L2 error")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("SigLIP So400m/14 feature reconstruction, LIBERO (worst / median / best)")
    fig.tight_layout()
    out = common.EXP_ROOT / "plots" / "cases_real112_bilinear.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
