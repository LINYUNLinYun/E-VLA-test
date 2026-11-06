#!/usr/bin/env python3
"""Collect and analyze fixed-noise single-layer VLM pruning probes.

The collection function is intentionally importable by the LIBERO evaluator;
the CLI analyzes one or more already-collected ``.npz`` samples without needing
CUDA, Triton, or simulator dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

try:
    from .layer_metrics import (
        LAST_HIDDEN_UPDATE_LAYER,
        NUM_VLM_LAYERS,
        TokenSpan,
        action_intervention_metrics,
        hidden_metrics,
        make_modality_spans,
        rank_candidates,
    )
except ImportError:  # direct script execution
    from layer_metrics import (  # type: ignore
        LAST_HIDDEN_UPDATE_LAYER,
        NUM_VLM_LAYERS,
        TokenSpan,
        action_intervention_metrics,
        hidden_metrics,
        make_modality_spans,
        rank_candidates,
    )


SCHEMA_VERSION = 1


def collect_single_layer_probe(
    *,
    session: Any,
    prepared: Any,
    noise: Any,
    output_path: str | Path,
    sample_id: str,
    view_names: tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    probe_layers: tuple[int, ...] = tuple(range(NUM_VLM_LAYERS)),
) -> Path:
    """Run Full18 plus each one-layer skip with identical inputs and noise.

    ``probe_layers`` uses zero-based runtime indices.  Stored/reporting layer IDs
    are one-based.  Only the Full18 pass captures H0..H18; intervention passes
    capture actions to keep device/host traffic bounded.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_actions, hidden_trace = session.run_layer_probe(
        prepared=prepared,
        noise=noise,
        skip_layers=(),
        capture_hidden=True,
    )
    if hidden_trace is None:
        raise RuntimeError("Layer probe runtime returned no Full18 hidden trace")
    skip_actions = []
    for layer_index in probe_layers:
        actions, _ = session.run_layer_probe(
            prepared=prepared,
            noise=noise,
            skip_layers=(int(layer_index),),
            capture_hidden=False,
        )
        skip_actions.append(actions.detach().float().cpu().numpy())

    trace_np = hidden_trace.detach().float().cpu().numpy()
    num_views = int(prepared.images.shape[0])
    spans = make_modality_spans(
        token_count=trace_np.shape[1],
        num_views=num_views,
        view_names=view_names,
    )
    payload_metadata = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "num_views": num_views,
        "modality_spans": [vars(span) for span in spans],
        "layer_numbering": "one_based_in_artifact_zero_based_runtime",
        "layer18_hidden_update": False,
        **(metadata or {}),
    }
    np.savez_compressed(
        output_path,
        hidden_trace=trace_np.astype(np.float16),
        baseline_actions=baseline_actions.detach().float().cpu().numpy(),
        skip_actions=np.stack(skip_actions),
        skip_layer_ids=np.asarray([index + 1 for index in probe_layers], dtype=np.int16),
        metadata_json=np.asarray(json.dumps(payload_metadata, sort_keys=True)),
    )
    return output_path


def _load_sample(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"hidden_trace", "baseline_actions", "skip_actions", "skip_layer_ids", "metadata_json"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing arrays: {', '.join(missing)}")
        metadata = json.loads(str(data["metadata_json"].item()))
        spans = tuple(TokenSpan(**item) for item in metadata["modality_spans"])
        return {
            "hidden_trace": data["hidden_trace"].astype(np.float32),
            "baseline_actions": data["baseline_actions"].astype(np.float32),
            "skip_actions": data["skip_actions"].astype(np.float32),
            "skip_layer_ids": data["skip_layer_ids"].astype(int).tolist(),
            "metadata": metadata,
            "spans": spans,
        }


def _mean_rows(rows: list[dict[str, Any]], keys: tuple[str, ...], *, bootstrap_samples: int = 1000) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    rng = np.random.default_rng(20261002)
    for group_key, group_rows in sorted(groups.items()):
        out = dict(zip(keys, group_key, strict=True))
        for column in group_rows[0]:
            if column in keys or isinstance(group_rows[0][column], (str, bool)):
                continue
            values = [float(row[column]) for row in group_rows]
            out[column] = float(np.mean(values))
            out[f"{column}_std"] = float(np.std(values))
            if len(values) == 1:
                low = high = values[0]
            else:
                array = np.asarray(values, dtype=np.float64)
                samples = rng.choice(array, size=(bootstrap_samples, len(array)), replace=True).mean(axis=1)
                low, high = np.quantile(samples, (0.025, 0.975)).tolist()
            out[f"{column}_ci95_low"] = float(low)
            out[f"{column}_ci95_high"] = float(high)
        out["sample_count"] = len(group_rows)
        if "layer" in out:
            out["updates_hidden"] = int(out["layer"]) <= LAST_HIDDEN_UPDATE_LAYER
        result.append(out)
    return result


def _plot_outputs(
    *,
    output_dir: Path,
    pairwise_means: dict[str, np.ndarray],
    action_summary: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
) -> None:
    cosine_keys = sorted(key for key in pairwise_means if key.endswith("/cosine"))
    if cosine_keys:
        fig, axes = plt.subplots(1, len(cosine_keys), figsize=(5.2 * len(cosine_keys), 4.6), squeeze=False)
        for axis, key in zip(axes[0], cosine_keys, strict=True):
            image = axis.imshow(pairwise_means[key], vmin=0.0, vmax=1.0, cmap="viridis")
            axis.set_title(key.removesuffix("/cosine"))
            axis.set_xlabel("hidden state index")
            axis.set_ylabel("hidden state index")
            axis.set_xticks(range(0, NUM_VLM_LAYERS + 1, 3))
            axis.set_yticks(range(0, NUM_VLM_LAYERS + 1, 3))
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.suptitle("VLM hidden-state tokenwise cosine")
        fig.tight_layout()
        fig.savefig(output_dir / "layer_similarity_heatmap.png", dpi=170)
        plt.close(fig)

    if action_summary:
        layers = [int(row["layer"]) for row in action_summary]
        nmse = [float(row["action_nmse"]) for row in action_summary]
        flips = [float(row["gripper_flip_rate"]) for row in action_summary]
        fig, left = plt.subplots(figsize=(9, 4.8))
        left.plot(layers, nmse, marker="o", color="#1769aa", label="action NMSE")
        left.set_xlabel("skipped VLM layer (1-based)")
        left.set_ylabel("action NMSE", color="#1769aa")
        left.set_xticks(layers)
        right = left.twinx()
        right.plot(layers, flips, marker="s", color="#d1495b", label="gripper flip rate")
        right.set_ylabel("gripper sign flip rate", color="#d1495b")
        left.set_title("Fixed-noise single-layer intervention sensitivity")
        fig.tight_layout()
        fig.savefig(output_dir / "layer_action_sensitivity.png", dpi=170)
        plt.close(fig)

    if ranking:
        ordered = sorted(ranking, key=lambda row: int(row["rank"]), reverse=True)
        fig, axis = plt.subplots(figsize=(8, 6))
        axis.barh([f"L{int(row['layer'])}" for row in ordered], [float(row["pruning_risk_score"]) for row in ordered])
        axis.set_xlabel("normalized pruning risk (lower is better)")
        axis.set_title("Candidate ranking: representation + action intervention")
        fig.tight_layout()
        fig.savefig(output_dir / "layer_candidate_ranking.png", dpi=170)
        plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_probe_files(
    paths: list[Path],
    output_dir: Path,
    *,
    translation_dims: tuple[int, ...] = (0, 1, 2),
    rotation_dims: tuple[int, ...] = (3, 4, 5),
    gripper_dim: int = 6,
    gripper_threshold: float = 0.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_hidden_rows: list[dict[str, Any]] = []
    all_action_rows: list[dict[str, Any]] = []
    pairwise_sums: dict[str, np.ndarray] = {}
    sample_ids = []
    for path in paths:
        sample = _load_sample(path)
        sample_id = str(sample["metadata"].get("sample_id", path.stem))
        sample_ids.append(sample_id)
        hidden_rows, pairwise = hidden_metrics(sample["hidden_trace"], sample["spans"])
        action_rows = action_intervention_metrics(
            sample["baseline_actions"],
            sample["skip_actions"],
            sample["skip_layer_ids"],
            translation_dims=translation_dims,
            rotation_dims=rotation_dims,
            gripper_dim=gripper_dim,
            gripper_threshold=gripper_threshold,
        )
        for row in hidden_rows:
            row["sample_id"] = sample_id
            all_hidden_rows.append(row)
        for row in action_rows:
            row["sample_id"] = sample_id
            all_action_rows.append(row)
        for key, matrix in pairwise.items():
            pairwise_sums[key] = pairwise_sums.get(key, np.zeros_like(matrix)) + matrix

    hidden_summary = _mean_rows(all_hidden_rows, ("layer", "modality"))
    action_summary = _mean_rows(all_action_rows, ("layer",))
    ranking = rank_candidates(hidden_summary, action_summary)
    _write_csv(output_dir / "layer_hidden_metrics.csv", hidden_summary)
    _write_csv(output_dir / "layer_action_sensitivity.csv", action_summary)
    _write_csv(output_dir / "layer_candidate_ranking.csv", ranking)
    pairwise_means = {key: value / len(paths) for key, value in pairwise_sums.items()}
    np.savez_compressed(
        output_dir / "layer_similarity_matrices.npz",
        **{key.replace("/", "__"): value for key, value in pairwise_means.items()},
    )
    _plot_outputs(
        output_dir=output_dir,
        pairwise_means=pairwise_means,
        action_summary=action_summary,
        ranking=ranking,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(paths),
        "sample_ids": sample_ids,
        "candidate_ranking": ranking,
        "layer18_note": (
            "H18 equals H17 by design: layer 18 produces prefix K/V but does not update hidden. "
            "It is excluded from transformer-block candidate ranking; its action intervention is retained "
            "to detect numerical differences between QKV and KV-only projection paths."
        ),
        "artifacts": {
            "hidden_metrics": "layer_hidden_metrics.csv",
            "action_sensitivity": "layer_action_sensitivity.csv",
            "candidate_ranking": "layer_candidate_ranking.csv",
            "similarity_matrices": "layer_similarity_matrices.npz",
            "similarity_heatmap": "layer_similarity_heatmap.png",
            "action_sensitivity_plot": "layer_action_sensitivity.png",
            "candidate_ranking_plot": "layer_candidate_ranking.png",
        },
    }
    (output_dir / "layer_probe_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parse_dims(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Collected layer-probe .npz files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--translation-dims", type=_parse_dims, default=(0, 1, 2))
    parser.add_argument("--rotation-dims", type=_parse_dims, default=(3, 4, 5))
    parser.add_argument("--gripper-dim", type=int, default=6)
    parser.add_argument("--gripper-threshold", type=float, default=0.0)
    args = parser.parse_args()
    report = analyze_probe_files(
        args.inputs,
        args.output_dir,
        translation_dims=args.translation_dims,
        rotation_dims=args.rotation_dims,
        gripper_dim=args.gripper_dim,
        gripper_threshold=args.gripper_threshold,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
