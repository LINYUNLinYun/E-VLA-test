"""Synthetic, CPU-only tests for attention/FFN probe metrics and aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.experiment8.analyze_submodules import (
    analyze_observation,
    analyze_probe_files,
    hidden_intervention_metrics,
    task_cluster_summary,
)
from scripts.experiment6.layer_metrics import TokenSpan


def _synthetic_probe(
    path: Path,
    *,
    sample_id: str,
    phase: str,
    shift: float = 0.0,
) -> Path:
    rng = np.random.default_rng(41)
    tokens, hidden = 4, 6
    base = rng.normal(size=(17, tokens, hidden)).astype(np.float32)
    after_attention = base + np.float32(shift + 0.1)
    outputs = after_attention + np.float32(shift + 0.05)
    baseline_actions = np.zeros((1, 15, 9), dtype=np.float32)
    baseline_actions[..., 6] = -1.0
    skip_attn = np.broadcast_to(baseline_actions[None, ...], (17, *baseline_actions.shape)).copy()
    skip_ffn = skip_attn.copy()
    # Deliberately create different intervention magnitudes by layer and module.
    skip_attn[:, ..., :3] += np.arange(1, 18, dtype=np.float32)[:, None, None, None] * 0.001
    skip_ffn[:, ..., 3:6] += np.arange(1, 18, dtype=np.float32)[:, None, None, None] * 0.002
    skip_attn[0, ..., 6] = 1.0
    metadata = {
        "sample_id": sample_id,
        "task_id": int(sample_id[-3:]) if sample_id[-3:].isdigit() else 0,
        "semantic_phase": phase,
        "action_space": "unnormalized policy command; not measured EEF xyz",
        "modality_spans": [
            {"name": "agent_view", "start": 0, "stop": 2},
            {"name": "language", "start": 2, "stop": 4},
        ],
    }
    np.savez_compressed(
        path,
        hidden_inputs=base,
        hidden_after_attn=after_attention,
        hidden_outputs=outputs,
        baseline_actions=baseline_actions,
        skip_attn_actions=skip_attn,
        skip_ffn_actions=skip_ffn,
        metadata_json=np.asarray(json.dumps(metadata)),
        layer_ids=np.arange(1, 18, dtype=np.int16),
    )
    return path


def test_hidden_metrics_report_token_distribution_per_modality() -> None:
    before = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    after = np.asarray([[1.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
    rows = hidden_intervention_metrics(
        before,
        after,
        (TokenSpan("vision", 0, 2), TokenSpan("language", 2, 3)),
    )

    vision = rows[0]
    assert vision["cosine_mean"] == 0.5
    assert vision["cosine_p05"] == 0.05
    assert vision["cosine_p50"] == 0.5
    assert vision["cosine_p95"] == 0.95
    assert vision["relative_residual"] == 1.0
    assert vision["zero_norm_either_tokens"] == 0
    assert np.isclose(rows[1]["cosine_mean"], 1.0)
    assert rows[1]["update_norm_mean"] > 0


def test_task_cluster_summary_weights_task_ids_not_observations() -> None:
    rows = [
        {"sample_id": "task-A", "phase": "initial", "value": 0.0},
        {"sample_id": "task-A", "phase": "approach", "value": 1.0},
        {"sample_id": "task-B", "phase": "initial", "value": 1.0},
    ]
    summary = task_cluster_summary(
        rows,
        group_keys=(),
        metric_names=("value",),
        bootstrap_samples=500,
        seed=7,
    )[0]

    # Task A contributes its phase mean 0.5, and task B contributes 1.0.
    assert summary["task_count"] == 2
    assert summary["observation_count"] == 3
    assert summary["value"] == 0.75
    assert summary["value_ci95_low"] <= summary["value"] <= summary["value_ci95_high"]


def test_synthetic_npz_analysis_writes_phase_and_module_artifacts(tmp_path: Path) -> None:
    probes = [
        _synthetic_probe(tmp_path / "a_initial.npz", sample_id="libero_task001", phase="initial"),
        _synthetic_probe(tmp_path / "a_approach.npz", sample_id="libero_task001", phase="approach", shift=0.1),
        _synthetic_probe(tmp_path / "b_initial.npz", sample_id="libero_task002", phase="initial", shift=0.2),
    ]

    hidden_rows, action_rows = analyze_observation(probes[0])
    assert len(hidden_rows) == 17 * 2 * 2  # layers x modules x modalities
    assert len(action_rows) == 17 * 2 * 2  # layers x modules x action windows
    assert {row["module"] for row in hidden_rows} == {"attn", "ffn"}
    assert {row["action_window"] for row in action_rows} == {"full_chunk", "first_12_actions"}
    assert next(row for row in action_rows if row["module"] == "attn")["full_gripper_flip_rate"] > 0

    output_dir = tmp_path / "analysis"
    report = analyze_probe_files(probes, output_dir, bootstrap_samples=250)
    assert report["task_sample_count"] == 2
    assert report["semantic_phase_observation_counts"] == {"approach": 1, "initial": 2}
    assert report["semantic_phase_task_counts"] == {"approach": 1, "initial": 2}
    assert report["bootstrap"]["resampling_unit"] == "metadata_json.sample_id task cluster"

    expected = (
        "submodule_observation_metrics.csv",
        "submodule_action_observations.csv",
        "submodule_summary.csv",
        "submodule_action_summary.csv",
        "submodule_candidate_ranking.csv",
        "submodule_attn_ranking.csv",
        "submodule_ffn_ranking.csv",
        "submodule_cosine_by_layer.png",
        "submodule_action_sensitivity.png",
        "submodule_candidate_ranking.png",
        "submodule_analysis_report.json",
    )
    assert all((output_dir / name).is_file() for name in expected)

    with (output_dir / "submodule_summary.csv").open(newline="", encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))
    assert {row["semantic_phase"] for row in summaries} == {"initial", "approach"}
    assert max(int(row["task_count"]) for row in summaries) == 2
    with (output_dir / "submodule_action_summary.csv").open(newline="", encoding="utf-8") as stream:
        action_summaries = list(csv.DictReader(stream))
    assert {row["action_window"] for row in action_summaries} == {"full", "prefix12"}
    assert all(row["action_nmse"] != "" for row in action_summaries)
