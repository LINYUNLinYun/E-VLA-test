"""Small CPU-only tests for paired simulator EEF trajectory visualization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiment8.plot_paired_xyz import (
    _jsonable_summary,
    compare_trajectories,
    main,
    plot_comparison,
    read_trajectory,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_paired_xyz_uses_simulator_positions_and_overlap_by_step(tmp_path: Path) -> None:
    baseline = _write_jsonl(
        tmp_path / "baseline.jsonl",
        [
            {"step": 2, "eef_pos": [0.2, 0.0, 0.1], "action": [100, 100, 100]},
            {"step": 0, "eef_pos": [0.0, 0.0, 0.0], "action": [9, 9, 9]},
            {"step": 1, "eef_pos": [0.1, 0.0, 0.0], "action": [-99, -99, -99]},
        ],
    )
    variant = _write_jsonl(
        tmp_path / "variant.jsonl",
        [
            {"step": 1, "eef_pos": [0.1, 0.0, 0.2]},
            {"step": 3, "eef_pos": [0.3, 0.0, 0.2]},
            {"step": 2, "eef_pos": [0.2, 0.0, 0.4]},
        ],
    )

    baseline_rows = read_trajectory(baseline)
    variant_rows = read_trajectory(variant)
    comparison = compare_trajectories(
        baseline_rows,
        variant_rows,
        baseline_label="Full model",
        variant_label="Pruned candidate",
        baseline_path=str(baseline),
        variant_path=str(variant),
    )

    assert [row["step"] for row in baseline_rows] == [0, 1, 2]
    assert comparison["step_coverage"]["shared_step_count"] == 2
    assert comparison["step_coverage"]["baseline_only_step_count"] == 1
    assert comparison["step_coverage"]["variant_only_step_count"] == 1
    assert comparison["step_coverage"]["shared_step_extent"] == {"first": 1, "last": 2}
    assert comparison["paired_deviation_m"]["maximum_step"] == 2
    assert np.isclose(comparison["paired_deviation_m"]["maximum"], 0.3)
    assert comparison["position_source"]["action_command_integration_used"] is False
    assert comparison["baseline"]["label_source"] == "caller_supplied"
    assert comparison["baseline_trust"]["validated_by_script"] is False

    png_path, pdf_path = tmp_path / "paired.png", tmp_path / "paired.pdf"
    plot_comparison(
        comparison,
        baseline_label="Full model",
        variant_label="Pruned candidate",
        png_path=png_path,
        pdf_path=pdf_path,
    )
    summary = _jsonable_summary(comparison, png_path=png_path, pdf_path=pdf_path)
    assert png_path.is_file() and png_path.stat().st_size > 1000
    assert pdf_path.is_file() and pdf_path.stat().st_size > 1000
    assert "_plot_data" not in summary
    assert summary["position_source"]["field"] == "eef_pos"


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"step": 0}], "eef_pos"),
        ([{"step": 0, "eef_pos": [0.0, float("nan"), 0.0]}], "finite"),
        ([{"step": 0, "eef_pos": [0, 0, 0]}, {"step": 0, "eef_pos": [1, 1, 1]}], "duplicate step"),
    ],
)
def test_invalid_or_ambiguous_telemetry_is_rejected(
    tmp_path: Path, rows: list[dict], message: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_trajectory(path)


def test_no_shared_environment_steps_is_rejected() -> None:
    baseline = [{"step": 0, "eef_pos": [0.0, 0.0, 0.0]}]
    variant = [{"step": 1, "eef_pos": [0.0, 0.0, 0.0]}]
    with pytest.raises(ValueError, match="no overlapping environment steps"):
        compare_trajectories(
            baseline, variant, baseline_label="baseline", variant_label="variant"
        )


def test_cli_writes_plots_and_json_summary(tmp_path: Path) -> None:
    baseline = _write_jsonl(
        tmp_path / "baseline.jsonl",
        [{"step": 0, "eef_pos": [0.0, 0.0, 0.0]}, {"step": 1, "eef_pos": [0.1, 0.0, 0.0]}],
    )
    variant = _write_jsonl(
        tmp_path / "variant.jsonl",
        [{"step": 0, "eef_pos": [0.0, 0.0, 0.0]}, {"step": 1, "eef_pos": [0.1, 0.0, 0.1]}],
    )
    prefix = tmp_path / "artifacts" / "pair"
    assert main(
        [
            str(baseline), str(variant),
            "--baseline-label", "Full model",
            "--variant-label", "Variant A",
            "--output-prefix", str(prefix),
        ]
    ) == 0

    summary_path = prefix.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert prefix.with_suffix(".png").is_file()
    assert prefix.with_suffix(".pdf").is_file()
    assert summary["baseline"]["label"] == "Full model"
    assert summary["baseline_trust"]["validated_by_script"] is False
    assert summary["paired_deviation_m"]["maximum_step"] == 1
