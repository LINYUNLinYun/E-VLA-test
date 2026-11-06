#!/usr/bin/env python3
"""Summarize paired decoder attention-logit precision diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("attention_logits_precision_*_seed12345/attention_logits_precision.json"))
    if len(paths) != 5:
        raise ValueError(f"Expected exactly 5 diagnostics, found {len(paths)}")
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        if not metrics["default_triton_reproduces_saved_reference"]:
            raise ValueError(f"Default path did not reproduce saved baseline: {path}")
        label = path.parent.name.removeprefix("attention_logits_precision_").removesuffix("_seed12345")
        row = {"label": label, "path": str(path.resolve())}
        for space in ("raw32d", "actions7d"):
            default = float(metrics[space]["default_triton_vs_jax"]["rmse"])
            fp32 = float(metrics[space]["fp32_logits_triton_vs_jax"]["rmse"])
            effect = float(metrics[space]["fp32_logits_effect_vs_default_triton"]["rmse"])
            row[space] = {
                "default_vs_jax_rmse": default,
                "fp32_logits_vs_jax_rmse": fp32,
                "fp32_logits_effect_rmse": effect,
                "relative_change": fp32 / default - 1.0,
                "moves_toward_jax": fp32 < default,
            }
        rows.append(row)
    summary = {
        "schema_version": 1,
        "observation_count": len(rows),
        "all_default_paths_bitwise_reproduce_saved_baseline": True,
        "raw32d_moves_toward_jax_count": sum(row["raw32d"]["moves_toward_jax"] for row in rows),
        "actions7d_moves_toward_jax_count": sum(row["actions7d"]["moves_toward_jax"] for row in rows),
        "actions7d_median_relative_change": float(np.median([row["actions7d"]["relative_change"] for row in rows])),
        "actions7d_mean_relative_change": float(np.mean([row["actions7d"]["relative_change"] for row in rows])),
        "rows": rows,
        "conclusion": "FP32 pre-softmax logit storage changes actions but does not consistently move Triton toward JAX.",
        "not_claimed": "Five fixed observations are not a closed-loop or population-level causal estimate.",
    }
    labels = [row["label"].replace("_", " ") for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    for ax, space, title in zip(axes, ("raw32d", "actions7d"), ("Raw 32D model actions", "Output-transformed 7D commands"), strict=True):
        default = [row[space]["default_vs_jax_rmse"] for row in rows]
        fp32 = [row[space]["fp32_logits_vs_jax_rmse"] for row in rows]
        ax.bar(x - width / 2, default, width, label="Default BF16 logits", color="#4e79a7")
        ax.bar(x + width / 2, fp32, width, label="FP32 logits", color="#f28e2b")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("RMSE to official JAX")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Attention-logit precision intervention: paired fixed observations", fontsize=14)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
