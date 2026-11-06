#!/usr/bin/env python3
"""Analyze separate VLM attention and FFN interventions from saved NPZ probes.

All analysis is CPU-only. Hidden-state similarity is reported per observation,
layer, module, and modality. Confidence intervals resample task ``sample_id``
clusters so observations from multiple semantic phases of one task are not
counted as independent experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.experiment6.layer_metrics import TokenSpan, action_intervention_metrics
except ImportError:  # direct invocation from a checkout with scripts/ on sys.path
    from experiment6.layer_metrics import TokenSpan, action_intervention_metrics  # type: ignore


MODULES = ("attn", "ffn")
HIDDEN_METRICS = (
    "cosine_mean",
    "cosine_p05",
    "cosine_p50",
    "cosine_p95",
    "relative_residual",
    "update_norm_mean",
    "update_norm_rms",
)
ACTION_METRICS = (
    "action_nmse",
    "action_cosine",
    "translation_rmse",
    "rotation_rmse",
    "gripper_rmse",
    "gripper_flip_rate",
)
SCHEMA_ARRAYS = (
    "hidden_inputs",
    "hidden_after_attn",
    "hidden_outputs",
    "baseline_actions",
    "skip_attn_actions",
    "skip_ffn_actions",
    "metadata_json",
)


def token_cosines(before: np.ndarray, after: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return one cosine per token for matching [tokens, hidden] arrays."""
    x = np.asarray(before, dtype=np.float64)
    y = np.asarray(after, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError(f"Expected matching [tokens, hidden] arrays, got {x.shape} and {y.shape}")
    numerator = np.einsum("th,th->t", x, y)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    result = numerator / np.maximum(denominator, eps)
    return np.clip(result, -1.0, 1.0)


def _validate_spans(spans: Sequence[TokenSpan], token_count: int) -> None:
    if not spans:
        raise ValueError("metadata_json.modality_spans must contain at least one span")
    names: set[str] = set()
    for span in spans:
        span.validate(token_count)
        if span.name in names:
            raise ValueError(f"Duplicate modality span name: {span.name!r}")
        names.add(span.name)


def hidden_intervention_metrics(
    before: np.ndarray,
    after: np.ndarray,
    spans: Sequence[TokenSpan],
) -> list[dict[str, Any]]:
    """Measure one block's hidden-state update separately within each modality.

    ``update_norm_mean`` is the mean token-wise L2 update magnitude. The
    relative residual is the Frobenius norm of the update divided by the input
    Frobenius norm, matching the interpretation used in Experiment 6.
    """
    x = np.asarray(before, dtype=np.float64)
    y = np.asarray(after, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError(f"Expected matching [tokens, hidden] arrays, got {x.shape} and {y.shape}")
    _validate_spans(spans, x.shape[0])
    rows: list[dict[str, Any]] = []
    delta = y - x
    denominator = max(float(np.linalg.norm(x)), 1e-12)
    for span in spans:
        local_x = x[span.start : span.stop]
        local_y = y[span.start : span.stop]
        local_delta = local_y - local_x
        cosines = token_cosines(local_x, local_y)
        token_update_norms = np.linalg.norm(local_delta, axis=1)
        before_norms = np.linalg.norm(local_x, axis=1)
        after_norms = np.linalg.norm(local_y, axis=1)
        rows.append(
            {
                "modality": span.name,
                "token_count": span.stop - span.start,
                "cosine_mean": float(np.mean(cosines)),
                "cosine_p05": float(np.percentile(cosines, 5)),
                "cosine_p50": float(np.percentile(cosines, 50)),
                "cosine_p95": float(np.percentile(cosines, 95)),
                "relative_residual": float(np.linalg.norm(local_delta) / max(float(np.linalg.norm(local_x)), 1e-12)),
                "update_norm_mean": float(np.mean(token_update_norms)),
                "update_norm_rms": float(np.sqrt(np.mean(token_update_norms**2))),
                "global_relative_residual": float(np.linalg.norm(delta) / denominator),
                "zero_norm_input_tokens": int(np.count_nonzero(before_norms <= 1e-12)),
                "zero_norm_output_tokens": int(np.count_nonzero(after_norms <= 1e-12)),
                "zero_norm_either_tokens": int(np.count_nonzero((before_norms <= 1e-12) | (after_norms <= 1e-12))),
            }
        )
    return rows


def _load_probe(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(SCHEMA_ARRAYS) - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing NPZ entries: {', '.join(missing)}")
        metadata = json.loads(str(data["metadata_json"].item()))
        for key in ("sample_id", "semantic_phase", "modality_spans"):
            if key not in metadata:
                raise ValueError(f"{path}: metadata_json is missing {key!r}")
        sample_id = str(metadata["sample_id"])
        semantic_phase = str(metadata["semantic_phase"])
        if not sample_id or not semantic_phase:
            raise ValueError(f"{path}: sample_id and semantic_phase must be non-empty")
        spans = tuple(TokenSpan(**span) for span in metadata["modality_spans"])

        hidden_inputs = np.asarray(data["hidden_inputs"], dtype=np.float32)
        hidden_after_attn = np.asarray(data["hidden_after_attn"], dtype=np.float32)
        hidden_outputs = np.asarray(data["hidden_outputs"], dtype=np.float32)
        if hidden_inputs.ndim != 3 or hidden_inputs.shape[0] != 17:
            raise ValueError(f"{path}: hidden_inputs must be [17, tokens, hidden], got {hidden_inputs.shape}")
        if hidden_after_attn.shape != hidden_inputs.shape or hidden_outputs.shape != hidden_inputs.shape:
            raise ValueError(
                f"{path}: hidden_inputs, hidden_after_attn, and hidden_outputs must match; got "
                f"{hidden_inputs.shape}, {hidden_after_attn.shape}, {hidden_outputs.shape}"
            )
        if not all(np.all(np.isfinite(value)) for value in (hidden_inputs, hidden_after_attn, hidden_outputs)):
            raise ValueError(f"{path}: hidden arrays contain non-finite values")
        _validate_spans(spans, hidden_inputs.shape[1])

        layer_ids_value: np.ndarray | None = None
        if "layer_ids" in data.files:
            layer_ids_value = np.asarray(data["layer_ids"], dtype=int).reshape(-1)
        elif "layer_ids" in metadata:
            layer_ids_value = np.asarray(metadata["layer_ids"], dtype=int).reshape(-1)
        layer_ids = list(range(1, 18)) if layer_ids_value is None else layer_ids_value.tolist()
        if len(layer_ids) != 17 or sorted(int(value) for value in layer_ids) != list(range(1, 18)):
            raise ValueError(f"{path}: layer_ids must contain IDs 1..17 exactly once, got {layer_ids}")

        baseline_actions = np.asarray(data["baseline_actions"], dtype=np.float32)
        skip_attn_actions = np.asarray(data["skip_attn_actions"], dtype=np.float32)
        skip_ffn_actions = np.asarray(data["skip_ffn_actions"], dtype=np.float32)
        if baseline_actions.ndim < 2 or baseline_actions.shape[-1] < 7:
            raise ValueError(f"{path}: baseline_actions must have at least 7 command dimensions, got {baseline_actions.shape}")
        expected_shape = (17, *baseline_actions.shape)
        if skip_attn_actions.shape != expected_shape or skip_ffn_actions.shape != expected_shape:
            raise ValueError(
                f"{path}: skip action arrays must have shape {expected_shape}; got "
                f"{skip_attn_actions.shape} and {skip_ffn_actions.shape}"
            )
        if not all(np.all(np.isfinite(value)) for value in (baseline_actions, skip_attn_actions, skip_ffn_actions)):
            raise ValueError(f"{path}: action arrays contain non-finite values")

        return {
            "path": path,
            "metadata": metadata,
            "sample_id": sample_id,
            "semantic_phase": semantic_phase,
            "spans": spans,
            "layer_ids": [int(value) for value in layer_ids],
            "hidden_inputs": hidden_inputs,
            "hidden_after_attn": hidden_after_attn,
            "hidden_outputs": hidden_outputs,
            "baseline_actions": baseline_actions,
            "skip_attn_actions": skip_attn_actions,
            "skip_ffn_actions": skip_ffn_actions,
        }


def _action_rows_for_window(
    baseline: np.ndarray,
    skips: np.ndarray,
    layer_ids: Sequence[int],
    *,
    prefix_actions: int | None,
) -> list[dict[str, Any]]:
    # The collector stores physical command outputs (and may preserve extra
    # model dimensions). Compare only the first seven robot command channels:
    # translation xyz, rotation xyz, and gripper. These are command-space
    # changes, not reconstructed world-coordinate EEF displacement.
    base = np.asarray(baseline, dtype=np.float64)[..., :7]
    skip = np.asarray(skips, dtype=np.float64)[..., :7]
    if prefix_actions is not None:
        base = base[..., :prefix_actions, :]
        skip = skip[..., :prefix_actions, :]
    return action_intervention_metrics(base, skip, layer_ids)


def analyze_observation(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return detailed hidden and action metric rows for one NPZ observation."""
    probe = _load_probe(path)
    metadata = probe["metadata"]
    sample_id = probe["sample_id"]
    phase = probe["semantic_phase"]
    base_fields = {
        "sample_id": sample_id,
        "task_id": metadata.get("task_id", ""),
        "semantic_phase": phase,
        "observation_id": path.stem,
        "input_file": path.name,
    }
    hidden_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []

    module_states = {
        "attn": (probe["hidden_inputs"], probe["hidden_after_attn"]),
        "ffn": (probe["hidden_after_attn"], probe["hidden_outputs"]),
    }
    for module, (before_all, after_all) in module_states.items():
        for layer_index, layer_id in enumerate(probe["layer_ids"]):
            metrics_by_modality = hidden_intervention_metrics(
                before_all[layer_index], after_all[layer_index], probe["spans"]
            )
            for modality_metrics in metrics_by_modality:
                hidden_rows.append(
                    {
                        **base_fields,
                        "module": module,
                        "layer": int(layer_id),
                        **modality_metrics,
                    }
                )

        skips = probe["skip_attn_actions"] if module == "attn" else probe["skip_ffn_actions"]
        for window_name, prefix in (("full", None), ("prefix12", 12)):
            metrics = _action_rows_for_window(
                probe["baseline_actions"],
                skips,
                probe["layer_ids"],
                prefix_actions=prefix,
            )
            for item in metrics:
                action_rows.append(
                    {
                        **base_fields,
                        "module": module,
                        "layer": int(item["layer"]),
                        **{f"{window_name}_{key}": value for key, value in item.items() if key != "layer"},
                        "action_dimensions_used": "0:7",
                        "action_space_note": str(
                            metadata.get("action_space", "provided policy command space; not measured EEF xyz")
                        ),
                        "action_window": "full_chunk" if prefix is None else "first_12_actions",
                    }
                )

    actions_by_key = {(row["observation_id"], row["module"], row["layer"], row["action_window"]): row for row in action_rows}
    # The combined per-observation table places the two action windows beside
    # each hidden-state modality row; the dedicated action CSV stays one row
    # per module/layer/window and avoids hiding the repeated modality values.
    for row in hidden_rows:
        for window, window_label in (("full", "full_chunk"), ("prefix12", "first_12_actions")):
            action = actions_by_key[(row["observation_id"], row["module"], row["layer"], window_label)]
            for key, value in action.items():
                if key.startswith(window + "_"):
                    row[key] = value
    return hidden_rows, action_rows


def task_cluster_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    metric_names: Sequence[str],
    bootstrap_samples: int = 2000,
    seed: int = 20261003,
    sample_key: str = "sample_id",
) -> list[dict[str, Any]]:
    """Summarize metrics after averaging within task IDs, then bootstrap tasks.

    A task with several phase observations contributes one value to a group.
    This prevents a task with more exported frames from receiving more weight.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if sample_key not in row:
            raise ValueError(f"Rows are missing cluster key {sample_key!r}")
        group = tuple(row.get(key, "") for key in group_keys)
        grouped[group].append(row)
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_task[str(row[sample_key])].append(row)
        task_names = sorted(by_task)
        result: dict[str, Any] = dict(zip(group_keys, key, strict=True))
        result["task_count"] = len(task_names)
        result["observation_count"] = len(group_rows)
        for metric in metric_names:
            task_values = []
            for task in task_names:
                values = [float(row[metric]) for row in by_task[task] if metric in row and np.isfinite(float(row[metric]))]
                if values:
                    task_values.append(float(np.mean(values)))
            if not task_values:
                result[metric] = math.nan
                result[f"{metric}_std"] = math.nan
                result[f"{metric}_ci95_low"] = math.nan
                result[f"{metric}_ci95_high"] = math.nan
                continue
            values_array = np.asarray(task_values, dtype=np.float64)
            result[metric] = float(np.mean(values_array))
            result[f"{metric}_std"] = float(np.std(values_array))
            if len(values_array) == 1:
                ci_low = ci_high = float(values_array[0])
            else:
                boot = rng.choice(values_array, size=(bootstrap_samples, len(values_array)), replace=True).mean(axis=1)
                ci_low, ci_high = (float(value) for value in np.quantile(boot, (0.025, 0.975)))
            result[f"{metric}_ci95_low"] = ci_low
            result[f"{metric}_ci95_high"] = ci_high
        summaries.append(result)
    return summaries


def _bootstrap_cluster_extreme_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    metric_names: Sequence[str],
    extreme: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Task bootstrap after reducing each task's phases/modalities by max."""
    if extreme not in {"max", "mean"}:
        raise ValueError("extreme must be 'max' or 'mean'")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    rng = np.random.default_rng(seed)
    result_rows = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in items:
            by_task[str(row["sample_id"])].append(row)
        task_ids = sorted(by_task)
        result: dict[str, Any] = dict(zip(group_keys, key, strict=True))
        result["task_count"] = len(task_ids)
        for metric in metric_names:
            per_task: list[float] = []
            for task_id in task_ids:
                values = [float(row[metric]) for row in by_task[task_id] if metric in row and np.isfinite(float(row[metric]))]
                if values:
                    per_task.append(float(np.max(values) if extreme == "max" else np.mean(values)))
            if not per_task:
                continue
            array = np.asarray(per_task, dtype=np.float64)
            result[metric] = float(np.mean(array))
            result[f"{metric}_std"] = float(np.std(array))
            if len(array) == 1:
                low = high = float(array[0])
            else:
                boot = rng.choice(array, size=(bootstrap_samples, len(array)), replace=True).mean(axis=1)
                low, high = (float(value) for value in np.quantile(boot, (0.025, 0.975)))
            result[f"{metric}_ci95_low"] = low
            result[f"{metric}_ci95_high"] = high
        result_rows.append(result)
    return result_rows


def _minmax(values: Mapping[Any, float]) -> dict[Any, float]:
    finite = {key: float(value) for key, value in values.items() if np.isfinite(value)}
    if not finite:
        return {key: 0.0 for key in values}
    low, high = min(finite.values()), max(finite.values())
    if high - low <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (float(value) - low) / (high - low) for key, value in values.items()}


def rank_submodules(
    hidden_observations: Sequence[Mapping[str, Any]],
    action_observations: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    min_tasks_for_candidate: int = 3,
) -> list[dict[str, Any]]:
    """Rank module/layer probes using upper-CI action and hidden residual risk.

    Within each task, the maximum over observed phases (and, for residuals,
    modalities) is used before cluster bootstrap. The risk score then takes
    the maximum normalized action component, preventing low average movement
    error from concealing a large translation, rotation, or gripper change.
    Lower scores are more promising probes; closed-loop LIBERO evaluation is
    still required before calling any component redundant.
    """
    hidden_names = ("relative_residual",)
    action_base_names = (
        "full_action_nmse",
        "prefix12_action_nmse",
        "full_translation_rmse",
        "prefix12_translation_rmse",
        "full_rotation_rmse",
        "prefix12_rotation_rmse",
        "full_gripper_flip_rate",
        "prefix12_gripper_flip_rate",
    )
    hidden = _bootstrap_cluster_extreme_summary(
        hidden_observations,
        group_keys=("module", "layer"),
        metric_names=hidden_names,
        extreme="max",
        bootstrap_samples=bootstrap_samples,
        seed=20261003,
    )
    actions = _bootstrap_cluster_extreme_summary(
        action_observations,
        group_keys=("module", "layer"),
        metric_names=action_base_names,
        extreme="max",
        bootstrap_samples=bootstrap_samples,
        seed=20261004,
    )
    by_key = {(str(row["module"]), int(row["layer"])): row for row in hidden}
    for row in actions:
        key = (str(row["module"]), int(row["layer"]))
        by_key.setdefault(key, {}).update(row)
    if not by_key:
        return []

    component_raw: dict[str, dict[tuple[str, int], float]] = {}
    for key, row in by_key.items():
        residual_ucb = float(row.get("relative_residual_ci95_high", row.get("relative_residual", math.inf)))
        component_raw.setdefault("relative_residual", {})[key] = residual_ucb
        for metric in action_base_names:
            metric_ucb = float(row.get(f"{metric}_ci95_high", row.get(metric, math.inf)))
            component_raw.setdefault(metric, {})[key] = metric_ucb
    normalized = {name: _minmax(values) for name, values in component_raw.items()}

    ranked = []
    for key, row in by_key.items():
        action_risk = max(normalized[name].get(key, 1.0) for name in action_base_names)
        residual_risk = normalized["relative_residual"].get(key, 1.0)
        action_task_count = int(row.get("task_count", 0))
        hidden_task_count = int(row.get("task_count", 0))
        task_count = min(action_task_count, hidden_task_count) if action_task_count and hidden_task_count else max(action_task_count, hidden_task_count)
        ranked.append(
            {
                **row,
                "task_count": task_count,
                "normalized_action_risk_ucb": float(action_risk),
                "normalized_hidden_residual_risk_ucb": float(residual_risk),
                "pruning_risk_score": float(0.65 * action_risk + 0.35 * residual_risk),
                "candidate_status": "screen_for_closed_loop" if task_count >= min_tasks_for_candidate else "insufficient_task_coverage",
                "risk_score_note": "upper 95% task-bootstrap bounds; per-task worst observed phase/modality; lower is better",
            }
        )
    ranked.sort(key=lambda row: (float(row["pruning_risk_score"]), str(row["module"]), int(row["layer"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_plot_cosines(
    rows: Sequence[Mapping[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    return task_cluster_summary(
        rows,
        group_keys=("module", "layer", "modality"),
        metric_names=("cosine_mean",),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _aggregate_plot_actions(
    rows: Sequence[Mapping[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    metrics = (
        "full_action_nmse",
        "prefix12_action_nmse",
        "full_gripper_flip_rate",
        "prefix12_gripper_flip_rate",
    )
    return task_cluster_summary(
        rows,
        group_keys=("module", "layer"),
        metric_names=metrics,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _make_plots(
    output_dir: Path,
    hidden_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts: list[str] = []
    cosine_summary = _aggregate_plot_cosines(hidden_rows, bootstrap_samples, seed=20261005)
    if cosine_summary:
        modalities = sorted({str(row["modality"]) for row in cosine_summary})
        fig, axes = plt.subplots(1, len(MODULES), figsize=(7.3 * len(MODULES), 5.2), squeeze=False)
        colors = plt.get_cmap("tab10")
        for module_index, module in enumerate(MODULES):
            axis = axes[0][module_index]
            module_rows = [row for row in cosine_summary if row["module"] == module]
            for modality_index, modality in enumerate(modalities):
                selected = sorted(
                    (row for row in module_rows if row["modality"] == modality),
                    key=lambda row: int(row["layer"]),
                )
                if not selected:
                    continue
                x = np.asarray([int(row["layer"]) for row in selected])
                y = np.asarray([float(row["cosine_mean"]) for row in selected])
                low = np.asarray([float(row["cosine_mean_ci95_low"]) for row in selected])
                high = np.asarray([float(row["cosine_mean_ci95_high"]) for row in selected])
                color = colors(modality_index % 10)
                axis.plot(x, y, marker="o", linewidth=1.7, color=color, label=modality)
                axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
            axis.set_title(f"{module.upper()} residual update")
            axis.set_xlabel("VLM layer (1-based)")
            axis.set_ylabel("token cosine: input vs output")
            axis.set_ylim(-1.02, 1.02)
            axis.set_xticks(range(1, 18))
            axis.grid(alpha=0.25)
            axis.legend(title="token modality", fontsize=8)
        fig.suptitle("Attention and FFN hidden-state cosine by modality\nshading: 95% task-cluster bootstrap CI")
        fig.tight_layout()
        fig.savefig(output_dir / "submodule_cosine_by_layer.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        artifacts.append("submodule_cosine_by_layer.png")

    action_summary = _aggregate_plot_actions(action_rows, bootstrap_samples, seed=20261006)
    if action_summary:
        fig, axes = plt.subplots(1, len(MODULES), figsize=(7.3 * len(MODULES), 5.0), squeeze=False)
        for module_index, module in enumerate(MODULES):
            axis = axes[0][module_index]
            selected = sorted(
                (row for row in action_summary if row["module"] == module),
                key=lambda row: int(row["layer"]),
            )
            if not selected:
                continue
            x = np.asarray([int(row["layer"]) for row in selected])
            full = np.asarray([float(row["full_action_nmse"]) for row in selected])
            full_low = np.asarray([float(row["full_action_nmse_ci95_low"]) for row in selected])
            full_high = np.asarray([float(row["full_action_nmse_ci95_high"]) for row in selected])
            prefix = np.asarray([float(row["prefix12_action_nmse"]) for row in selected])
            prefix_low = np.asarray([float(row["prefix12_action_nmse_ci95_low"]) for row in selected])
            prefix_high = np.asarray([float(row["prefix12_action_nmse_ci95_high"]) for row in selected])
            axis.plot(x, full, marker="o", color="#1769aa", label="full 50-action chunk NMSE")
            axis.fill_between(x, full_low, full_high, color="#1769aa", alpha=0.13)
            axis.plot(x, prefix, marker="s", linestyle="--", color="#f28e2b", label="first 12 actions NMSE")
            axis.fill_between(x, prefix_low, prefix_high, color="#f28e2b", alpha=0.13)
            right = axis.twinx()
            flip = np.asarray([float(row["full_gripper_flip_rate"]) for row in selected])
            flip_low = np.asarray([float(row["full_gripper_flip_rate_ci95_low"]) for row in selected])
            flip_high = np.asarray([float(row["full_gripper_flip_rate_ci95_high"]) for row in selected])
            right.plot(x, flip, marker="^", color="#b23a48", alpha=0.9, label="full-chunk gripper flip rate")
            right.fill_between(x, flip_low, flip_high, color="#b23a48", alpha=0.10)
            right.set_ylabel("gripper sign flip rate", color="#b23a48")
            right.set_ylim(-0.03, 1.03)
            axis.set_title(f"Skip {module.upper()} intervention")
            axis.set_xlabel("VLM layer (1-based)")
            axis.set_ylabel("action NMSE in robot command space")
            axis.set_xticks(range(1, 18))
            axis.grid(alpha=0.25)
            h1, l1 = axis.get_legend_handles_labels()
            h2, l2 = right.get_legend_handles_labels()
            axis.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        fig.suptitle("Fixed-observation attention/FFN skip action sensitivity\nshading: 95% task-cluster bootstrap CI")
        fig.tight_layout()
        fig.savefig(output_dir / "submodule_action_sensitivity.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        artifacts.append("submodule_action_sensitivity.png")

    if ranking:
        ordered = sorted(ranking, key=lambda row: int(row["rank"]), reverse=True)
        labels = [f"{str(row['module']).upper()} L{int(row['layer'])}" for row in ordered]
        scores = [float(row["pruning_risk_score"]) for row in ordered]
        colors = ["#59a14f" if row["candidate_status"] == "screen_for_closed_loop" else "#bab0ab" for row in ordered]
        height = max(5.5, 0.27 * len(ordered))
        fig, axis = plt.subplots(figsize=(9.5, height))
        axis.barh(labels, scores, color=colors)
        axis.set_xlabel("conservative normalized probe risk (lower is better)")
        axis.set_title("Attention / FFN candidate screening\nupper task-bootstrap bounds, worst observed phase and modality")
        axis.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "submodule_candidate_ranking.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        artifacts.append("submodule_candidate_ranking.png")
    return artifacts


def analyze_probe_files(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    bootstrap_samples: int = 2000,
    min_tasks_for_candidate: int = 3,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("At least one probe NPZ is required")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    phases: Counter[str] = Counter()
    phase_tasks: dict[str, set[str]] = defaultdict(set)
    layer_ids_by_file: dict[str, list[int]] = {}
    for path in inputs:
        hidden, actions = analyze_observation(Path(path))
        hidden_rows.extend(hidden)
        action_rows.extend(actions)
        if hidden:
            sample_id = str(hidden[0]["sample_id"])
            phase = str(hidden[0]["semantic_phase"])
            sample_ids.add(sample_id)
            phases[phase] += 1
            phase_tasks[phase].add(sample_id)
            layer_ids_by_file[Path(path).name] = sorted({int(row["layer"]) for row in hidden})

    summary_rows = task_cluster_summary(
        hidden_rows,
        group_keys=("module", "layer", "modality", "semantic_phase"),
        metric_names=HIDDEN_METRICS,
        bootstrap_samples=bootstrap_samples,
        seed=20261007,
    )
    action_summary_rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(("full", "prefix12")):
        source_window = "full_chunk" if window == "full" else "first_12_actions"
        window_rows = [
            {
                **row,
                **{metric: row[f"{window}_{metric}"] for metric in ACTION_METRICS},
                "action_window": window,
            }
            for row in action_rows
            if row["action_window"] == source_window
        ]
        action_summary_rows.extend(
            task_cluster_summary(
                window_rows,
                group_keys=("module", "layer", "semantic_phase", "action_window"),
                metric_names=ACTION_METRICS,
                bootstrap_samples=bootstrap_samples,
                seed=20261008 + window_index,
            )
        )
    ranking = rank_submodules(
        hidden_rows,
        action_rows,
        bootstrap_samples=bootstrap_samples,
        min_tasks_for_candidate=min_tasks_for_candidate,
    )

    # Keep one readable observation table with hidden metrics and both action
    # windows, plus a non-repeated action-only table for downstream analysis.
    _write_csv(output_dir / "submodule_observation_metrics.csv", hidden_rows)
    _write_csv(output_dir / "submodule_action_observations.csv", action_rows)
    _write_csv(output_dir / "submodule_summary.csv", summary_rows)
    _write_csv(output_dir / "submodule_action_summary.csv", action_summary_rows)
    _write_csv(output_dir / "submodule_candidate_ranking.csv", ranking)
    for module in MODULES:
        module_rows = [dict(row) for row in ranking if row["module"] == module]
        for module_rank, row in enumerate(module_rows, start=1):
            row["module_rank"] = module_rank
        _write_csv(output_dir / f"submodule_{module}_ranking.csv", module_rows)
    plot_artifacts = _make_plots(
        output_dir,
        hidden_rows,
        action_rows,
        ranking,
        bootstrap_samples=bootstrap_samples,
    )

    report = {
        "schema_version": 1,
        "input_count": len(inputs),
        "task_sample_count": len(sample_ids),
        "sample_ids": sorted(sample_ids),
        "semantic_phase_observation_counts": dict(sorted(phases.items())),
        "semantic_phase_task_counts": {phase: len(ids) for phase, ids in sorted(phase_tasks.items())},
        "layer_ids_by_file": layer_ids_by_file,
        "bootstrap": {
            "resampling_unit": "metadata_json.sample_id task cluster",
            "method": "average observations within each task, then bootstrap task means",
            "replicates": bootstrap_samples,
            "seed": 20261003,
        },
        "action_metrics": {
            "dimensions_used": "first 7 channels: translation xyz, rotation xyz, gripper",
            "space": "provided policy command space (metadata action_space when available); not measured EEF xyz",
            "windows": {"full": "full action chunk", "prefix12": "first 12 actions"},
        },
        "candidate_ranking": ranking,
        "interpretation": (
            "Lower candidate risk is more promising for a closed-loop follow-up. Ranking uses upper 95% task-cluster "
            "bootstrap bounds, and per-task worst observed phase/modality for hidden residual and action changes. "
            "Probe ranking alone does not establish that a module is safe to prune."
        ),
        "artifacts": {
            "observation_metrics": "submodule_observation_metrics.csv",
            "action_observations": "submodule_action_observations.csv",
            "summary": "submodule_summary.csv",
            "action_summary": "submodule_action_summary.csv",
            "candidate_ranking": "submodule_candidate_ranking.csv",
            "attention_ranking": "submodule_attn_ranking.csv",
            "ffn_ranking": "submodule_ffn_ranking.csv",
            "cosine_plot": "submodule_cosine_by_layer.png",
            "sensitivity_plot": "submodule_action_sensitivity.png",
            "ranking_plot": "submodule_candidate_ranking.png",
        },
        "plots_written": plot_artifacts,
    }
    (output_dir / "submodule_analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Submodule intervention probe NPZ files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--min-tasks-for-candidate", type=int, default=3)
    args = parser.parse_args(argv)
    report = analyze_probe_files(
        args.inputs,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        min_tasks_for_candidate=args.min_tasks_for_candidate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
