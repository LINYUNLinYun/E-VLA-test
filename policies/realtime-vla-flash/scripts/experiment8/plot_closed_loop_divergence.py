#!/usr/bin/env python3
"""Plot paired D10 pose/gripper divergence phases and per-task onset timelines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = ROOT / "results/pi0_attention_ffn_diagnosis/closed_loop"
DEFAULT_OUTPUT_DIR = ROOT / "results/pi0_attention_ffn_diagnosis/closed_loop_stage_plots"
MODULES = (
    ("Attention 16", "attn16_pose_gripper_audit.json", "#2878B5", "o"),
    ("FFN 2", "ffn2_pose_gripper_audit.json", "#E1812C", "s"),
)
PHASE_ORDER = (
    ("initial", "Initial"),
    ("approach", "Approach"),
    ("contact", "Contact"),
    ("grasp_or_manipulate", "Grasp / manipulate"),
    ("transport", "Transport"),
    ("goal_alignment", "Goal alignment"),
    ("no_sustained_onset", "No sustained XYZ onset"),
    ("other", "Other / unknown phase"),
)
PHASE_ALIASES = {
    "grasp": "grasp_or_manipulate",
    "grasp_or_manipulation": "grasp_or_manipulate",
    "goal": "goal_alignment",
}
ONSET_SIGNALS = (
    ("xyz_sustained_divergence_state_index", "XYZ > 2 cm", "#1F77B4", "o", 0.16),
    ("rotation_sustained_divergence_state_index", "Rotation > 5°", "#9467BD", "^", 0.0),
    ("gripper_sustained_divergence_state_index", "Gripper proxy > 0.01", "#2CA02C", "s", -0.16),
)
EXPECTED_PAIR_COUNT = 12
EXPECTED_THRESHOLDS = {
    "threshold_m": 0.02,
    "rotation_threshold_degrees": 5.0,
    "gripper_width_proxy_threshold": 0.01,
    "consecutive_states": 5,
}


def _read_audit(path: Path) -> dict[str, Any]:
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read audit JSON {path}: {exc}") from exc
    if not isinstance(audit, dict) or not isinstance(audit.get("pairs"), list):
        raise ValueError(f"{path}: expected audit object with a pairs list")
    if audit.get("errors_use_only_common_valid_states") is not True:
        raise ValueError(f"{path}: audit must explicitly use only common valid states")
    if audit.get("pair_count") != EXPECTED_PAIR_COUNT or len(audit["pairs"]) != EXPECTED_PAIR_COUNT:
        raise ValueError(f"{path}: expected exactly N={EXPECTED_PAIR_COUNT} paired tasks")
    definitions = audit.get("divergence_definition")
    if not isinstance(definitions, dict):
        raise ValueError(f"{path}: missing divergence_definition")
    for field, expected in EXPECTED_THRESHOLDS.items():
        actual = definitions.get(field)
        if isinstance(expected, float):
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not np.isclose(actual, expected):
                raise ValueError(f"{path}: expected {field}={expected}, got {actual!r}")
        elif actual != expected:
            raise ValueError(f"{path}: expected {field}={expected}, got {actual!r}")
    seen: set[str] = set()
    for pair in audit["pairs"]:
        if not isinstance(pair, dict):
            raise ValueError(f"{path}: each pair must be a JSON object")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise ValueError(f"{path}: missing or duplicate pair_id {pair_id!r}")
        seen.add(pair_id)
        if not isinstance(pair.get("shared_state_count"), int) or pair["shared_state_count"] < 1:
            raise ValueError(f"{path}: pair {pair_id} has invalid shared_state_count")
        for field in (
            "xyz_sustained_divergence_state_index",
            "rotation_sustained_divergence_state_index",
            "gripper_sustained_divergence_state_index",
        ):
            value = pair.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= pair["shared_state_count"]
            ):
                raise ValueError(
                    f"{path}: pair {pair_id} {field}={value!r} is outside the common valid segment"
                )
        if pair.get("baseline_phase_at_divergence") is not None and not isinstance(
            pair["baseline_phase_at_divergence"], str
        ):
            raise ValueError(f"{path}: pair {pair_id} has invalid baseline phase")
        if not isinstance(pair.get("baseline_success"), bool) or not isinstance(pair.get("variant_success"), bool):
            raise ValueError(f"{path}: pair {pair_id} is missing success booleans")
    return audit


def _phase_counts(audit: dict[str, Any]) -> dict[str, int]:
    counts = {key: 0 for key, _ in PHASE_ORDER}
    for pair in audit["pairs"]:
        phase = pair.get("baseline_phase_at_divergence")
        if phase is None:
            key = "no_sustained_onset"
        else:
            key = PHASE_ALIASES.get(phase, phase)
            if key not in counts:
                key = "other"
        counts[key] += 1
    return counts


def _make_phase_distribution(attn: dict[str, Any], ffn: dict[str, Any], output_dir: Path) -> None:
    phase_keys = [key for key, _ in PHASE_ORDER]
    phase_labels = [label for _, label in PHASE_ORDER]
    attn_counts = _phase_counts(attn)
    ffn_counts = _phase_counts(ffn)
    positions = np.arange(len(phase_keys), dtype=float)
    bar_height = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 7.1), layout="constrained")
    for offset, (module, _filename, color, _marker), counts in zip(
        (-bar_height / 2, bar_height / 2), MODULES, (attn_counts, ffn_counts)
    ):
        values = [counts[key] for key in phase_keys]
        bars = ax.barh(positions + offset, values, height=bar_height, color=color, label=module, alpha=0.9)
        for bar, value in zip(bars, values):
            if value:
                ax.text(
                    value + 0.08,
                    bar.get_y() + bar.get_height() / 2,
                    str(value),
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#333333",
                )

    ax.set_yticks(positions, phase_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, EXPECTED_PAIR_COUNT + 1.3)
    ax.set_xticks(range(EXPECTED_PAIR_COUNT + 1))
    ax.set_xlabel("Number of paired tasks (N=12 per module)")
    ax.set_title(
        "First sustained XYZ divergence by baseline phase",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    ax.text(
        0,
        1.015,
        "Counted by `baseline_phase_at_divergence`; this is a phase label, not a failure classification.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncols=2)

    fig.text(
        0.01,
        -0.025,
        "Heuristic diagnostic, not causal failure onset. Threshold: XYZ > 2 cm for 5 consecutive states. "
        "Only common valid states are included; successful runs may also show sustained divergence.",
        fontsize=9,
        color="#444444",
    )
    fig.savefig(output_dir / "stage_distribution.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "stage_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def _short_pair_id(pair_id: str) -> str:
    pieces = pair_id.split("_")
    suite = pieces[1] if len(pieces) > 1 else pair_id
    task = next((piece for piece in pieces if piece.startswith("task")), "task?")
    return f"{suite} · {task}"


def _make_onset_timeline(attn: dict[str, Any], ffn: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 10.2), sharex=True)
    all_max = max(
        pair["shared_state_count"]
        for audit in (attn, ffn)
        for pair in audit["pairs"]
    )
    signal_handles = None
    for ax, (module, _filename, module_color, _module_marker), audit in zip(axes, MODULES, (attn, ffn)):
        pairs = sorted(audit["pairs"], key=lambda pair: pair["pair_id"])
        ys = np.arange(len(pairs), dtype=float)
        ax.set_yticks(ys)
        ylabels = []
        for pair in pairs:
            baseline_flag = "+" if pair["baseline_success"] else "−"
            variant_flag = "+" if pair["variant_success"] else "−"
            ylabels.append(f"{_short_pair_id(pair['pair_id'])}  [B{baseline_flag}/V{variant_flag}]")
        ax.set_yticklabels(ylabels, fontsize=8.5)
        ax.invert_yaxis()
        for y, pair in zip(ys, pairs):
            ax.hlines(
                y,
                0,
                pair["shared_state_count"] - 1,
                color="#D9DDE2",
                linewidth=5,
                zorder=1,
            )
            ax.text(
                pair["shared_state_count"] - 1,
                y + 0.31,
                f"N={pair['shared_state_count']}",
                fontsize=6.5,
                color="#777777",
                ha="right",
                va="center",
            )

        for field, label, color, marker, y_offset in ONSET_SIGNALS:
            xs = [pair[field] for pair in pairs]
            valid = [(index, value) for index, value in enumerate(xs) if value is not None]
            if not valid:
                continue
            x_values = [value for _, value in valid]
            y_values = [ys[index] + y_offset for index, _ in valid]
            scatter = ax.scatter(
                x_values,
                y_values,
                color=color,
                marker=marker,
                s=58,
                linewidths=0.4,
                edgecolors="white",
                label=label,
                zorder=3,
            )
            if signal_handles is None:
                signal_handles = []
            if label not in [handle.get_label() for handle in signal_handles]:
                signal_handles.append(scatter)
            for x_value, y_value in zip(x_values, y_values):
                ax.annotate(
                    str(x_value),
                    (x_value, y_value),
                    xytext=(3, 2),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=color,
                    alpha=0.95,
                )

        baseline_successes = int(audit["baseline_successes"])
        variant_successes = int(audit["variant_successes"])
        ax.set_title(
            f"{module} intervention · N=12 pairs\n"
            f"baseline successes {baseline_successes}/12; variant successes {variant_successes}/12",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlabel("First sustained onset state index (common valid segment)")
        ax.set_xlim(-5, all_max + 35)
        ax.set_ylim(len(pairs) - 0.4, -0.6)
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Paired divergence onset timelines",
        y=0.987,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.955,
        "XYZ pose, orientation, and gripper-proxy signals · gray bars mark each task's common valid segment",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    if signal_handles:
        fig.legend(
            handles=signal_handles,
            labels=[handle.get_label() for handle in signal_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncols=3,
            frameon=False,
        )
    fig.text(
        0.01,
        0.018,
        "Heuristic diagnostic, not causal failure onset. XYZ > 2 cm, rotation > 5°, gripper width proxy > 0.01; "
        "each requires 5 consecutive states. Onsets use common valid states only. B/V success flags are shown; "
        "divergence in a successful pair is not labeled as an error. Gripper width proxy is not contact force.",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.115, top=0.82, wspace=0.27)
    fig.savefig(output_dir / "onset_timeline.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "onset_timeline.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(
    *, attn_audit_path: Path, ffn_audit_path: Path, output_dir: Path,
) -> dict[str, Any]:
    attn = _read_audit(Path(attn_audit_path).expanduser().resolve(strict=True))
    ffn = _read_audit(Path(ffn_audit_path).expanduser().resolve(strict=True))
    attn_ids = {pair["pair_id"] for pair in attn["pairs"]}
    ffn_ids = {pair["pair_id"] for pair in ffn["pairs"]}
    if attn_ids != ffn_ids:
        raise ValueError("Attention and FFN audits do not contain the same 12 pair IDs")
    attn_thresholds = attn["divergence_definition"]
    ffn_thresholds = ffn["divergence_definition"]
    if any(attn_thresholds.get(key) != ffn_thresholds.get(key) for key in EXPECTED_THRESHOLDS):
        raise ValueError("Attention and FFN divergence thresholds do not match")

    output = Path(output_dir).expanduser()
    if output.name in ("", ".", ".."):
        raise ValueError(f"Output path must name a new directory: {output_dir}")
    output = output.parent.resolve() / output.name
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)

    _make_phase_distribution(attn, ffn, output)
    _make_onset_timeline(attn, ffn, output)
    summary = {
        "N_per_module": EXPECTED_PAIR_COUNT,
        "attention_pair_ids": sorted(attn_ids),
        "ffn_pair_ids": sorted(ffn_ids),
        "phase_counts_by_baseline_phase": {
            "attention16": _phase_counts(attn),
            "ffn2": _phase_counts(ffn),
        },
        "thresholds": EXPECTED_THRESHOLDS,
        "threshold_comparisons": "strictly greater than each threshold for the stated consecutive-state count",
        "consecutive_states": EXPECTED_THRESHOLDS["consecutive_states"],
        "common_valid_states_only": True,
        "interpretation": (
            "Heuristic diagnostic, not causal failure onset. Sustained divergence is not classified as a failure; "
            "successful pairs may also diverge. Gripper width proxy is not a calibrated force/contact metric."
        ),
        "artifacts": [
            "stage_distribution.png",
            "stage_distribution.pdf",
            "onset_timeline.png",
            "onset_timeline.pdf",
        ],
    }
    summary_path = output / "plot_summary.json"
    if os.path.lexists(summary_path):
        raise FileExistsError(f"Refusing to overwrite existing plot summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attn-audit",
        type=Path,
        default=DEFAULT_AUDIT_DIR / MODULES[0][1],
    )
    parser.add_argument(
        "--ffn-audit",
        type=Path,
        default=DEFAULT_AUDIT_DIR / MODULES[1][1],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    summary = make_plots(
        attn_audit_path=args.attn_audit,
        ffn_audit_path=args.ffn_audit,
        output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
