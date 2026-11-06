#!/usr/bin/env python3
"""Plot furthest semantic stages for a strictly validated clean control run."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = (
    "initial",
    "approach",
    "contact",
    "grasp_or_manipulate",
    "transport",
    "goal_alignment",
    "release_or_final_action",
)
LABELS = (
    "Initial",
    "Approach",
    "Contact",
    "Grasp / manipulate",
    "Transport",
    "Goal alignment",
    "Final action",
)
COLORS = ("#dce9f5", "#a6c8e2", "#76b7b2", "#59a14f", "#edc948", "#f28e2b", "#e15759")


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize(rows: list[dict[str, str]], *, expected_count: int) -> dict[str, object]:
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} replay rows, found {len(rows)}")
    if not all(_truth(row.get("success_match", "")) for row in rows):
        bad = [row.get("sample_id", "") for row in rows if not _truth(row.get("success_match", ""))]
        raise ValueError(f"Replay success mismatch for: {bad}")
    if not all(_truth(row.get("action_count_match", "")) for row in rows):
        bad = [row.get("sample_id", "") for row in rows if not _truth(row.get("action_count_match", ""))]
        raise ValueError(f"Replay action-count mismatch for: {bad}")

    failures = [row for row in rows if not _truth(row.get("expected_success", ""))]
    unknown = sorted({row.get("max_phase", "") for row in failures} - set(STAGES))
    if unknown:
        raise ValueError(f"Unknown failure stages: {unknown}")
    groups = ("All failures", "LIBERO-90", "LIBERO-10")
    counters = {
        "All failures": Counter(row["max_phase"] for row in failures),
        "LIBERO-90": Counter(row["max_phase"] for row in failures if row["suite_name"] == "libero_90"),
        "LIBERO-10": Counter(row["max_phase"] for row in failures if row["suite_name"] == "libero_10"),
    }
    return {
        "schema_version": 1,
        "evidence": "deterministic simulator replay of saved executed actions",
        "episode_count": len(rows),
        "success_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "all_success_labels_match": True,
        "all_action_counts_match": True,
        "groups": {
            group: {
                "failure_count": sum(counters[group].values()),
                "counts": {stage: counters[group][stage] for stage in STAGES},
            }
            for group in groups
        },
        "warning": "Furthest reached stage is progress evidence, not the causal onset of the model error.",
    }


def plot(summary: dict[str, object], output_prefix: Path) -> None:
    groups_payload = summary["groups"]
    assert isinstance(groups_payload, dict)
    groups = list(groups_payload)
    fig, ax = plt.subplots(figsize=(11.5, 4.9), layout="constrained")
    left = np.zeros(len(groups), dtype=float)
    for stage, label, color in zip(STAGES, LABELS, COLORS, strict=True):
        values = np.asarray([groups_payload[group]["counts"][stage] for group in groups], dtype=float)
        if not values.any():
            continue
        ax.barh(groups, values, left=left, color=color, label=label, height=0.62, edgecolor="white")
        for index, value in enumerate(values):
            if value > 0:
                ax.text(left[index] + value / 2, index, str(int(value)), ha="center", va="center", fontsize=10)
        left += values
    for index, total in enumerate(left):
        ax.text(total + 0.4, index, f"n={int(total)}", va="center", color="#4d4d4d")
    ax.invert_yaxis()
    ax.set_xlim(0, max(left.max() + 6, 10))
    ax.set_xlabel("Failed episodes")
    ax.set_title("Clean Control D3 failures: furthest semantic stage reached", loc="left", fontsize=14, pad=14)
    ax.text(
        0,
        1.02,
        "Saved actions replayed in the simulator; stage is progress reached, not causal error onset",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#555555",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-validation", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()
    with args.replay_validation.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = summarize(rows, expected_count=args.expected_count)
    plot(summary, args.output_prefix)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
