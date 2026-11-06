"""Validate and summarize completed 20-condition skip-layer rollouts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[Any]) -> float | None:
    numbers = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return None if not numbers else float(sum(numbers) / len(numbers))


def _run_root(condition: dict[str, Any]) -> Path:
    run_dir = Path(str(condition["run_dir"]))
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    resolved = run_dir.resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"run_dir escapes repository: {resolved}")
    return resolved


def _condition_summary(condition: dict[str, Any], expected_trials: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[int, int], np.ndarray]]:
    condition_id = str(condition["condition_id"])
    run_dir = _run_root(condition)
    episodes = _read_json(run_dir / "episode_log.json")
    if not isinstance(episodes, list) or len(episodes) != expected_trials:
        raise ValueError(f"{condition_id}: expected {expected_trials} episodes, found {len(episodes) if isinstance(episodes, list) else 'invalid'}")
    episode_rows: list[dict[str, Any]] = []
    actions: dict[tuple[int, int], np.ndarray] = {}
    infer_latency: list[Any] = []
    policy_latency: list[Any] = []
    executed_latency: list[Any] = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError(f"{condition_id}: invalid episode record")
        episode_idx = int(episode["episode_idx"])
        infer_path = run_dir / str(episode["infer_path"])
        infer_rows = _read_jsonl(infer_path)
        for infer in infer_rows:
            key = (episode_idx, int(infer["infer_id"]))
            if "chunk_actions" in infer:
                actions[key] = np.asarray(infer["chunk_actions"], dtype=np.float32)
            infer_latency.append(infer.get("client_roundtrip_ms"))
            policy_latency.append(infer.get("policy_time_ms"))
            executed_latency.append(infer.get("avg_policy_time_per_action_ms"))
        episode_rows.append(
            {
                "condition_id": condition_id,
                "vlm_layer_pattern": condition["vlm_layer_pattern"],
                "action_condition_id": condition["action_condition_id"],
                "action_layer_pattern": condition["action_layer_pattern"],
                "denoise_steps": condition["denoise_steps"],
                "is_baseline": bool(condition.get("is_baseline", False)),
                "episode_idx": episode_idx,
                "success": bool(episode.get("success", False)),
                "trajectory_length": episode.get("trajectory_length"),
                "executed_action_count": episode.get("executed_action_count"),
                "infer_calls": episode.get("infer_calls"),
                "client_roundtrip_mean_ms": episode.get("infer_latency_mean_ms"),
                "policy_time_mean_ms": episode.get("policy_time_mean_ms"),
                "avg_policy_time_per_action_ms": episode.get("avg_policy_time_per_action_ms"),
                "video_path": episode.get("video_path"),
                "trace_path": episode.get("trace_path"),
                "infer_path": episode.get("infer_path"),
            }
        )
    successes = sum(int(row["success"]) for row in episode_rows)
    summary = {
        "condition_id": condition_id,
        "vlm_layer_pattern": condition["vlm_layer_pattern"],
        "action_condition_id": condition["action_condition_id"],
        "action_layer_pattern": condition["action_layer_pattern"],
        "denoise_steps": int(condition["denoise_steps"]),
        "is_baseline": bool(condition.get("is_baseline", False)),
        "episode_count": len(episode_rows),
        "successes": successes,
        "success_rate": successes / len(episode_rows),
        "client_roundtrip_mean_ms": _mean(infer_latency),
        "policy_time_mean_ms": _mean(policy_latency),
        "executed_action_amortized_policy_ms": _mean(executed_latency),
    }
    return summary, episode_rows, actions


def summarize_experiment(manifest_path: Path, output_dir: Path) -> list[str]:
    manifest = _read_json(manifest_path)
    errors: list[str] = []
    conditions = manifest.get("conditions", [])
    if len(conditions) != 20:
        errors.append(f"expected 20 conditions, found {len(conditions)}")
    expected_trials = int(manifest.get("num_trials_per_condition", 50))
    summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    action_by_condition: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for condition in conditions:
        try:
            summary, episodes, actions = _condition_summary(condition, expected_trials)
        except Exception as exc:  # noqa: BLE001 - retain all condition errors
            errors.append(str(exc))
            continue
        summaries.append(summary)
        episode_rows.extend(episodes)
        action_by_condition[summary["condition_id"]] = actions

    equivalence_rows: list[dict[str, Any]] = []
    for group in (("V2_B", "V3_B", "V5_B"), ("V2_D", "V3_D", "V5_D")):
        reference = action_by_condition.get(group[0], {})
        for candidate_id in group[1:]:
            candidate = action_by_condition.get(candidate_id, {})
            keys = sorted(set(reference) & set(candidate))
            mismatches = 0
            max_abs_diff: float | None = 0.0
            for key in keys:
                left, right = reference[key], candidate[key]
                if left.shape != right.shape:
                    mismatches += 1
                    max_abs_diff = None
                    continue
                diff = float(np.max(np.abs(left - right))) if left.size else 0.0
                if max_abs_diff is not None:
                    max_abs_diff = max(max_abs_diff, diff)
                mismatches += int(not np.array_equal(left, right))
            equivalence_rows.append(
                {
                    "equivalence_class": "+".join(group),
                    "reference_condition_id": group[0],
                    "candidate_condition_id": candidate_id,
                    "compared_action_chunks": len(keys),
                    "mismatched_action_chunks": mismatches,
                    "max_abs_diff": max_abs_diff,
                    "bitwise_equal": mismatches == 0 and bool(keys),
                }
            )
    if not summaries:
        errors.append("no completed condition artifacts")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "condition_results.csv", summaries, list(summaries[0]) if summaries else ["condition_id"])
    _write_csv(output_dir / "episode_results.csv", episode_rows, list(episode_rows[0]) if episode_rows else ["condition_id"])
    _write_csv(output_dir / "equivalence_checks.csv", equivalence_rows, list(equivalence_rows[0]) if equivalence_rows else ["equivalence_class"])
    summary_payload = {
        "experiment_id": manifest.get("experiment_id"),
        "generated_at": _now(),
        "status": "completed" if not errors and len(summaries) == 20 else "incomplete",
        "condition_count": len(summaries),
        "episode_record_count": len(episode_rows),
        "integrity_passed": not errors and len(summaries) == 20,
        "integrity_errors": errors,
        "conditions": summaries,
        "equivalence_checks": equivalence_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    report = [
        "# π0 VLM × Action Expert × denoising ablation report",
        "",
        f"- experiment_id: `{manifest.get('experiment_id')}`",
        f"- status: `{summary_payload['status']}`",
        f"- conditions: {len(summaries)}/20",
        "",
    ]
    if errors:
        report += ["## Integrity errors", "", *[f"- {error}" for error in errors], ""]
    (output_dir / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    errors = summarize_experiment(args.manifest.resolve(), args.output_dir.resolve())
    if errors:
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"summary written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
