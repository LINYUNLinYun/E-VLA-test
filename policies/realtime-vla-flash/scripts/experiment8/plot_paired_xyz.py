#!/usr/bin/env python3
"""Compare simulator-measured EEF trajectories from paired telemetry JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASELINE_COLOR = "#1f77b4"
VARIANT_COLOR = "#ff7f0e"
MAX_COLOR = "#b2182b"


def read_trajectory(path: Path) -> list[dict[str, Any]]:
    """Read and validate step-indexed simulator EEF positions from telemetry JSONL."""
    rows_by_step: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL record must be an object")

            step_value = row.get("step")
            if isinstance(step_value, bool) or not isinstance(step_value, (int, float)):
                raise ValueError(f"{path}:{line_number}: missing or invalid integer 'step'")
            if isinstance(step_value, float):
                if not math.isfinite(step_value) or not step_value.is_integer():
                    raise ValueError(f"{path}:{line_number}: 'step' must be a finite integer")
                step = int(step_value)
            else:
                step = int(step_value)
            if step < 0:
                raise ValueError(f"{path}:{line_number}: 'step' must be nonnegative")
            if step in rows_by_step:
                raise ValueError(f"{path}:{line_number}: duplicate step {step}")

            position = row.get("eef_pos")
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                raise ValueError(
                    f"{path}:{line_number}: missing or invalid 'eef_pos'; expected 3 simulator coordinates"
                )
            coords: list[float] = []
            for axis, value in zip("xyz", position):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{path}:{line_number}: eef_pos {axis} must be numeric")
                coord = float(value)
                if not math.isfinite(coord):
                    raise ValueError(f"{path}:{line_number}: eef_pos {axis} must be finite")
                coords.append(coord)
            rows_by_step[step] = {"step": step, "eef_pos": coords}

    if not rows_by_step:
        raise ValueError(f"{path}: telemetry contains no EEF positions")
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def _step_extent(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {"first": int(rows[0]["step"]), "last": int(rows[-1]["step"])}


def compare_trajectories(
    baseline_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]],
    *, baseline_label: str, variant_label: str,
    baseline_path: str = "", variant_path: str = "",
) -> dict[str, Any]:
    """Pair on simulator step IDs and summarize positional discrepancy on overlap."""
    baseline_by_step = {int(row["step"]): np.asarray(row["eef_pos"], dtype=float) for row in baseline_rows}
    variant_by_step = {int(row["step"]): np.asarray(row["eef_pos"], dtype=float) for row in variant_rows}
    shared_steps = sorted(baseline_by_step.keys() & variant_by_step.keys())
    if not shared_steps:
        raise ValueError("baseline and variant telemetry have no overlapping environment steps")

    baseline_xyz = np.stack([baseline_by_step[s] for s in shared_steps])
    variant_xyz = np.stack([variant_by_step[s] for s in shared_steps])
    deviations = np.linalg.norm(baseline_xyz - variant_xyz, axis=1)
    maximum_index = int(np.argmax(deviations))
    max_step = int(shared_steps[maximum_index])
    shared_count = len(shared_steps)

    return {
        "baseline": {
            "label": baseline_label,
            "label_source": "caller_supplied",
            "telemetry_path": baseline_path,
            "row_count": len(baseline_rows),
            "step_extent": _step_extent(baseline_rows),
        },
        "variant": {
            "label": variant_label,
            "label_source": "caller_supplied",
            "telemetry_path": variant_path,
            "row_count": len(variant_rows),
            "step_extent": _step_extent(variant_rows),
        },
        "position_source": {
            "field": "eef_pos",
            "origin": "simulator telemetry from experiment6/trajectory_replay.py (robot0_eef_pos observation)",
            "units": "meters",
            "action_command_integration_used": False,
        },
        "baseline_trust": {
            "validated_by_script": False,
            "note": "The baseline label is supplied by the caller. A condition named Control is not automatically treated as a trusted baseline.",
        },
        "step_coverage": {
            "pairing_key": "environment step",
            "baseline_rows": len(baseline_rows),
            "variant_rows": len(variant_rows),
            "shared_step_count": shared_count,
            "shared_step_extent": {"first": shared_steps[0], "last": shared_steps[-1]},
            "baseline_only_step_count": len(baseline_rows) - shared_count,
            "variant_only_step_count": len(variant_rows) - shared_count,
            "baseline_fraction_covered": shared_count / len(baseline_rows),
            "variant_fraction_covered": shared_count / len(variant_rows),
        },
        "paired_deviation_m": {
            "mean": float(np.mean(deviations)),
            "median": float(np.median(deviations)),
            "p95": float(np.percentile(deviations, 95)),
            "maximum": float(deviations[maximum_index]),
            "maximum_step": max_step,
            "baseline_position_at_max": baseline_xyz[maximum_index].tolist(),
            "variant_position_at_max": variant_xyz[maximum_index].tolist(),
        },
        "_plot_data": {
            "baseline_steps": [int(row["step"]) for row in baseline_rows],
            "baseline_xyz": np.asarray([row["eef_pos"] for row in baseline_rows], dtype=float),
            "variant_steps": [int(row["step"]) for row in variant_rows],
            "variant_xyz": np.asarray([row["eef_pos"] for row in variant_rows], dtype=float),
            "shared_steps": shared_steps,
            "deviations": deviations,
        },
    }


def _set_equal_3d_axes(ax: Any, all_xyz: np.ndarray) -> None:
    low = all_xyz.min(axis=0)
    high = all_xyz.max(axis=0)
    center = (low + high) / 2.0
    span = float(np.max(high - low))
    radius = max(span * 0.55, 0.01)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_comparison(
    comparison: dict[str, Any], *, baseline_label: str, variant_label: str,
    png_path: Path, pdf_path: Path,
) -> None:
    data = comparison["_plot_data"]
    bxyz = data["baseline_xyz"]
    vxyz = data["variant_xyz"]
    baseline_steps = data["baseline_steps"]
    variant_steps = data["variant_steps"]
    shared_steps = data["shared_steps"]
    deviations = data["deviations"]
    max_index = int(np.argmax(deviations))
    max_step = shared_steps[max_index]
    bmax = bxyz[baseline_steps.index(max_step)]
    vmax = vxyz[variant_steps.index(max_step)]

    fig = plt.figure(figsize=(14, 9), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.1, 0.9))
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    projections = {
        "xy": (fig.add_subplot(grid[0, 1]), 0, 1, "X (m)", "Y (m)"),
        "xz": (fig.add_subplot(grid[0, 2]), 0, 2, "X (m)", "Z (m)"),
        "yz": (fig.add_subplot(grid[1, 0]), 1, 2, "Y (m)", "Z (m)"),
    }
    deviation_ax = fig.add_subplot(grid[1, 1:])

    mark_every_b = max(1, len(bxyz) // 30)
    mark_every_v = max(1, len(vxyz) // 30)
    ax3d.plot(*bxyz.T, color=BASELINE_COLOR, label=baseline_label, linewidth=2,
              alpha=0.62, linestyle="-", marker="o", markersize=2.5, markevery=mark_every_b)
    ax3d.plot(*vxyz.T, color=VARIANT_COLOR, label=variant_label, linewidth=2,
              alpha=0.78, linestyle="--", marker="^", markersize=2.8, markevery=mark_every_v)
    ax3d.scatter(*bmax, color=BASELINE_COLOR, marker="*", s=100, edgecolor="white", linewidth=0.6, zorder=5)
    ax3d.scatter(*vmax, color=VARIANT_COLOR, marker="*", s=100, edgecolor="white", linewidth=0.6, zorder=5)
    ax3d.plot(*np.stack([bmax, vmax]).T, color=MAX_COLOR, linestyle="--", linewidth=1.5, label=f"max deviation at step {max_step}")
    ax3d.set_title("3D EEF path")
    ax3d.set_xlabel("World X (m)", labelpad=6)
    ax3d.set_ylabel("World Y (m)", labelpad=6)
    ax3d.set_zlabel("World Z (m)", labelpad=4)
    ax3d.legend(fontsize=8, loc="best")
    _set_equal_3d_axes(ax3d, np.concatenate([bxyz, vxyz], axis=0))

    for key, (ax, axis_a, axis_b, xlabel, ylabel) in projections.items():
        ax.plot(bxyz[:, axis_a], bxyz[:, axis_b], color=BASELINE_COLOR, label=baseline_label,
                linewidth=1.8, alpha=0.62, linestyle="-", marker="o", markersize=2.5,
                markevery=mark_every_b)
        ax.plot(vxyz[:, axis_a], vxyz[:, axis_b], color=VARIANT_COLOR, label=variant_label,
                linewidth=1.8, alpha=0.78, linestyle="--", marker="^", markersize=2.8,
                markevery=mark_every_v)
        ax.scatter(bmax[axis_a], bmax[axis_b], color=BASELINE_COLOR, marker="*", s=90, edgecolor="white", linewidth=0.5, zorder=5)
        ax.scatter(vmax[axis_a], vmax[axis_b], color=VARIANT_COLOR, marker="*", s=90, edgecolor="white", linewidth=0.5, zorder=5)
        ax.plot([bmax[axis_a], vmax[axis_a]], [bmax[axis_b], vmax[axis_b]], color=MAX_COLOR, linestyle="--", linewidth=1.4)
        ax.set_title(key.upper() + " projection")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
        if key == "xy":
            ax.legend(fontsize=8)

    deviation_ax.plot(shared_steps, deviations, color="#444444", linewidth=1.8)
    deviation_ax.scatter([max_step], [deviations[max_index]], color=MAX_COLOR, marker="*", s=110, zorder=5)
    deviation_ax.annotate(
        f"max {deviations[max_index]:.3f} m at step {max_step}",
        (max_step, deviations[max_index]), xytext=(8, 10), textcoords="offset points",
        fontsize=9, color=MAX_COLOR,
    )
    deviation_ax.set_title("Same-step Euclidean EEF deviation")
    deviation_ax.set_xlabel("Environment step")
    deviation_ax.set_ylabel("Distance (m)")
    deviation_ax.grid(alpha=0.25)

    coverage = comparison["step_coverage"]
    fig.suptitle(
        f"Simulator EEF trajectory comparison: {baseline_label} vs {variant_label}\n"
        f"paired steps {coverage['shared_step_count']} / {coverage['baseline_rows']} baseline, "
        f"{coverage['shared_step_count']} / {coverage['variant_rows']} variant",
        fontsize=13,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)


def _jsonable_summary(comparison: dict[str, Any], *, png_path: Path, pdf_path: Path) -> dict[str, Any]:
    return {
        key: value for key, value in comparison.items() if key != "_plot_data"
    } | {"artifacts": {"png": str(png_path), "pdf": str(pdf_path)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_telemetry", type=Path, help="baseline simulator telemetry JSONL")
    parser.add_argument("variant_telemetry", type=Path, help="variant simulator telemetry JSONL")
    parser.add_argument("--baseline-label", required=True, help="explicit caller-supplied baseline label")
    parser.add_argument("--variant-label", default="variant")
    parser.add_argument("--output-prefix", type=Path, required=True, help="prefix for .png, .pdf, and .summary.json")
    args = parser.parse_args(argv)

    baseline_path = args.baseline_telemetry.resolve()
    variant_path = args.variant_telemetry.resolve()
    baseline_rows = read_trajectory(baseline_path)
    variant_rows = read_trajectory(variant_path)
    comparison = compare_trajectories(
        baseline_rows, variant_rows,
        baseline_label=args.baseline_label,
        variant_label=args.variant_label,
        baseline_path=str(baseline_path),
        variant_path=str(variant_path),
    )
    prefix = args.output_prefix.resolve()
    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")
    summary_path = prefix.with_suffix(".summary.json")
    plot_comparison(
        comparison, baseline_label=args.baseline_label, variant_label=args.variant_label,
        png_path=png_path, pdf_path=pdf_path,
    )
    summary = _jsonable_summary(comparison, png_path=png_path, pdf_path=pdf_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path), "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
