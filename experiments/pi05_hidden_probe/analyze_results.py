"""Analyze actual probe JSONL output; never synthesizes experiment results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from collections import defaultdict
from typing import Any

import numpy as np

if __package__:
    from .metrics import GROUP_FIELDS, summarize
else:
    # Support both `python -m experiments.pi05_hidden_probe.analyze_results`
    # and direct invocation of this file from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import GROUP_FIELDS, summarize


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each result must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"No result rows in {path}")
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _write_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    columns = list(GROUP_FIELDS) + sorted(set().union(*(row.keys() for row in metrics)) - set(GROUP_FIELDS))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in metrics)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if _finite(value) else "N/A"


def _display(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _figures(metrics: list[dict[str, Any]], directory: Path) -> tuple[str, list[dict[str, Any]]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "Skipped plots: matplotlib is not installed; CSV and summary are complete.", []

    directory.mkdir(parents=True, exist_ok=True)
    index = []
    curve_fields = [field for field in GROUP_FIELDS if field not in ("condition", "source_layer")]
    heatmap_fields = [field for field in GROUP_FIELDS if field not in ("condition", "source_layer", "target_layer")]
    # Candidate scoring is a separate figure set with explicit titles.
    for metric, mode in (("bacc", "generation"), ("score_bacc", "candidate_scoring")):
        curves: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        heatmaps: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in metrics:
            if not _finite(row.get(metric)):
                continue
            curves[tuple(row.get(field) for field in curve_fields)].append(row)
            if row["condition"] == "real" and row["source_layer"] is not None and row["target_layer"] is not None:
                heatmaps[tuple(row.get(field) for field in heatmap_fields)].append(row)

        # Include layer-independent controls in every target-layer curve with
        # otherwise identical metadata. Actual control values are reused only.
        target_index = curve_fields.index("target_layer")
        for key, group in list(curves.items()):
            if key[target_index] is None:
                continue
            generic_key = tuple(None if i == target_index else value for i, value in enumerate(key))
            for control in curves.get(generic_key, []):
                if control["condition"] != "real" and not any(row["condition"] == control["condition"] for row in group):
                    group.append(control)

        for key, group in curves.items():
            source_layers = sorted({row["source_layer"] for row in group if row["source_layer"] is not None})
            if not source_layers:
                continue
            figure, axis = plt.subplots(figsize=(7.5, 4.5))
            for condition in sorted({row["condition"] for row in group}):
                condition_rows = sorted([row for row in group if row["condition"] == condition and row["source_layer"] is not None],
                                        key=lambda row: row["source_layer"])
                if condition_rows:
                    xs = [row["source_layer"] for row in condition_rows]
                    ys = [row[metric] for row in condition_rows]
                    line, = axis.plot(xs, ys, marker="o", markersize=3, label=condition)
                    lows = [row.get(f"{metric}_ci_low", math.nan) for row in condition_rows]
                    highs = [row.get(f"{metric}_ci_high", math.nan) for row in condition_rows]
                    axis.fill_between(xs, lows, highs, color=line.get_color(), alpha=0.12)
                else:
                    baselines = [row for row in group if row["condition"] == condition]
                    if len(baselines) == 1:
                        baseline = baselines[0]
                        line = axis.axhline(baseline[metric], linestyle="--", label=condition)
                        low, high = baseline.get(f"{metric}_ci_low"), baseline.get(f"{metric}_ci_high")
                        if _finite(low) and _finite(high):
                            axis.axhspan(low, high, color=line.get_color(), alpha=0.07)
            metadata = dict(zip(curve_fields, key))
            axis.set(xlabel="Source block boundary (0 = VLM input)", ylabel="Balanced accuracy", ylim=(-0.02, 1.02),
                     title=f"{mode}: {metadata['task_type']} / {metadata['selector']} / target {metadata['target_layer']}")
            axis.legend(fontsize="small")
            axis.grid(alpha=0.2)
            figure.tight_layout()
            slug = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
            name = f"{mode}_layers_{slug}.png"
            figure.savefig(directory / name, dpi=160)
            plt.close(figure)
            index.append({"file": name, "kind": "layer_curve", "metric": metric, "configuration": metadata})

        for key, group in heatmaps.items():
            sources = sorted({row["source_layer"] for row in group})
            targets = sorted({row["target_layer"] for row in group})
            data = np.full((len(targets), len(sources)), np.nan)
            for row in group:
                i, j = targets.index(row["target_layer"]), sources.index(row["source_layer"])
                if np.isfinite(data[i, j]):
                    raise ValueError("Repeated source/target cell in a heatmap configuration")
                data[i, j] = row[metric]
            figure, axis = plt.subplots(figsize=(max(6, len(sources) * 0.35), max(3, len(targets) * 0.4)))
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("lightgray")
            artist = axis.imshow(data, origin="lower", aspect="auto", vmin=0, vmax=1, cmap=cmap)
            axis.set_xticks(range(len(sources)), sources)
            axis.set_yticks(range(len(targets)), targets)
            metadata = dict(zip(heatmap_fields, key))
            axis.set(xlabel="Source block boundary", ylabel="Target block boundary",
                     title=f"{mode}: {metadata['task_type']} / {metadata['selector']}")
            figure.colorbar(artist, ax=axis, label="Balanced accuracy")
            figure.tight_layout()
            slug = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
            name = f"{mode}_heatmap_{slug}.png"
            figure.savefig(directory / name, dpi=160)
            plt.close(figure)
            index.append({"file": name, "kind": "source_target_heatmap", "metric": metric, "configuration": metadata})
    with (directory / "index.json").open("w", encoding="utf-8") as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2)
    return f"Wrote {len(index)} figures from observed data; blank heatmap cells are unmeasured/undefined.", index


def _write_summary(
    path: Path, metrics: list[dict[str, Any]], rows: list[dict[str, Any]],
    input_path: Path, plot_status: str, bootstrap_samples: int, seed: int,
) -> None:
    lines = [
        "# π0.5 hidden-representation probe results", "",
        f"Input: `{input_path.resolve()}`. Observed rows: {len(rows)}; metric groups: {len(metrics)}.", "",
        "Balanced accuracy averages recall over the declared label set. Invalid answers count as errors. "
        "N/A means absent classes, absent controls, no eligible counterfactual/donor groups, or insufficient bootstrap blocks; consult CSV status/count columns.", "",
        f"95% percentile intervals use {bootstrap_samples} bootstrap draws (seed {seed}) of connected "
        "episode/scene/counterfactual blocks. A block includes every linked sample. Real/control deltas "
        "use paired sample IDs and the same resample. Draws missing any declared class are excluded, "
        "with valid draw counts in CSV; fewer than two blocks cannot estimate uncertainty.", "",
        "Generation and candidate-scoring results are separate. These results concern decodability under "
        "the tested reader configuration; they do not identify the first layer to contain information or prove use by the action policy.", "",
        plot_status, "",
        "| Split | Task / probe | Prompt / selector | Source → target | Condition | N / blocks | Gen. BAcc (95% CI) | Invalid | Pair / quad | Donor follow | Δ no-patch / shuffle | Score BAcc |",
        "|---|---|---|---|---|---:|---|---:|---|---:|---|---:|",
    ]
    for row in metrics:
        fields = [row["split"], f"{row['task_type']} / {row['probe_id']}", f"{row['prompt_id']} / {row['selector']}",
                  f"{row['source_layer']} → {row['target_layer']}", row["condition"], f"{row['n']} / {row['n_blocks']}",
                  f"{_fmt(row['bacc'])} ({_fmt(row['bacc_ci_low'])}, {_fmt(row['bacc_ci_high'])})",
                  _fmt(row["invalid_rate"]), f"{_fmt(row['pair_acc'])} / {_fmt(row['quad_acc'])}",
                  _fmt(row["donor_follow_rate"]), f"{_fmt(row.get('delta_nopatch'))} / {_fmt(row.get('delta_shuffle'))}",
                  _fmt(row.get("score_bacc"))]
        lines.append("| " + " | ".join(_display(value) for value in fields) + " |")
    missing = [row for row in metrics if row["missing_labels"]]
    if missing:
        lines.extend(["", f"{len(missing)} groups have missing label classes; their balanced accuracy is undefined."])
    ungrouped = sum(row["ungrouped_samples"] for row in metrics)
    if ungrouped:
        lines.extend(["", f"{ungrouped} result rows lack all block metadata. They are singleton blocks; provide episode/scene/counterfactual metadata before interpreting confidence intervals."])
    # Save an unbiased, reproducible sample of raw records, including failures.
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(rows), size=min(12, len(rows)), replace=False).tolist())
    examples_path = path.parent / "sampled_generations.jsonl"
    with examples_path.open("w", encoding="utf-8") as stream:
        for index in selected:
            stream.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
    lines.extend(["", "Reproducible randomly selected records (including their original outputs) are in "
                  "`sampled_generations.jsonl`. They were selected without filtering for correctness.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="probe_results.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        rows = _read_jsonl(args.input)
        metrics = summarize(rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output_dir / "layer_metrics.csv", metrics)
        plot_status, _ = _figures(metrics, args.output_dir / "figures")
        _write_summary(args.output_dir / "summary.md", metrics, rows, args.input,
                       plot_status, args.bootstrap_samples, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Analyzed {len(rows)} observed rows into {args.output_dir.resolve()}")
    print(plot_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
