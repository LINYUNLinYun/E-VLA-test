"""Validate and summarize paired multi-task experiment 5 results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    return None if not numbers else sum(numbers) / len(numbers)


def _exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _run_dir(condition: dict[str, Any]) -> Path:
    run_dir = Path(str(condition["run_dir"]))
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"run_dir escapes repository: {run_dir}")
    return run_dir


def summarize(manifest_path: Path, output_dir: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = manifest.get("conditions", [])
    errors: list[str] = []
    if len(conditions) != 5:
        errors.append(f"expected 5 conditions, found {len(conditions)}")
    expected_ids = list(manifest.get("selected_sample_ids", []))
    expected_count = int(manifest.get("selected_sample_count", 0))
    if expected_count != len(expected_ids):
        errors.append(f"selected sample count mismatch: {expected_count} vs {len(expected_ids)} IDs")
    results: list[dict[str, Any]] = []
    episodes_by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    video_rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        try:
            run_dir = _run_dir(condition)
            client_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            episodes = json.loads((run_dir / "episode_log.json").read_text(encoding="utf-8"))
            if client_manifest.get("status") != "completed":
                errors.append(f"{condition_id}: client run status is {client_manifest.get('status')}")
            if len(episodes) != expected_count:
                raise ValueError(f"expected {expected_count} episodes, found {len(episodes)}")
            by_id = {str(row["sample_id"]): row for row in episodes}
            if set(by_id) != set(expected_ids):
                raise ValueError("episode sample IDs differ from manifest")
            episodes_by_condition[condition_id] = by_id
            successes = sum(bool(row.get("success")) for row in episodes)
            for row in episodes:
                for key in ("video_path", "trace_path", "infer_path"):
                    rel = Path(str(row.get(key, "")))
                    artifact = (run_dir / rel).resolve()
                    if rel.is_absolute() or ".." in rel.parts or not artifact.is_relative_to(run_dir) or not artifact.is_file() or artifact.stat().st_size == 0:
                        errors.append(f"{condition_id}/{row.get('sample_id')}: invalid or missing {key}")
                video_rows.append({"condition_id": condition_id, "sample_id": row.get("sample_id"), "suite_name": row.get("suite_name"), "task_id": row.get("task_id"), "success": bool(row.get("success")), "video_path": str(run_dir / str(row.get("video_path", "")))})
            suites = sorted({str(row.get("suite_name")) for row in episodes})
            result = {
                "condition_id": condition_id, "vlm_layer_pattern": condition["vlm_layer_pattern"],
                "num_vlm_layers": len(condition["vlm_layer_indices_1based"]),
                "is_control": bool(condition.get("is_control", condition_id == "Control")),
                "task_count": len(episodes), "successes": successes, "success_rate": successes / len(episodes),
                "policy_time_mean_ms": _mean([row.get("policy_time_mean_ms") for row in episodes]),
                "roundtrip_mean_ms": _mean([row.get("client_roundtrip_mean_ms") for row in episodes]),
                "trajectory_length_mean": _mean([row.get("trajectory_length") for row in episodes]),
            }
            for suite in suites:
                suite_rows = [row for row in episodes if row.get("suite_name") == suite]
                key = suite.removeprefix("libero_") + "_success_rate"
                result[key] = sum(bool(row.get("success")) for row in suite_rows) / len(suite_rows)
            results.append(result)
        except Exception as exc:
            errors.append(f"{condition_id}: {exc}")

    pairwise: list[dict[str, Any]] = []
    control = episodes_by_condition.get("Control")
    if control:
        for condition in conditions:
            candidate_id = str(condition["condition_id"])
            if candidate_id == "Control" or candidate_id not in episodes_by_condition:
                continue
            candidate = episodes_by_condition[candidate_id]
            b = sum(bool(control[key].get("success")) and not bool(candidate[key].get("success")) for key in expected_ids)
            c = sum(not bool(control[key].get("success")) and bool(candidate[key].get("success")) for key in expected_ids)
            pairwise.append({"condition_id": candidate_id, "control_id": "Control", "paired_tasks": len(expected_ids), "candidate_only_failures": b, "candidate_only_successes": c, "success_rate_delta_vs_control": (sum(bool(candidate[key].get("success")) for key in expected_ids) - sum(bool(control[key].get("success")) for key in expected_ids)) / max(len(expected_ids), 1), "mcnemar_exact_p": _exact_mcnemar_p(b, c), "new_failures_vs_control": b})
    else:
        errors.append("Control condition results are missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "condition_results.csv", results, list(results[0]) if results else ["condition_id"])
    _write_csv(output_dir / "episode_results.csv", [{"condition_id": cid, **row} for cid, entries in episodes_by_condition.items() for row in entries.values()], ["condition_id", "sample_id", "suite_name", "task_id", "demo_index", "success", "trajectory_length", "infer_latency_mean_ms", "client_roundtrip_mean_ms", "policy_time_mean_ms", "video_path", "trace_path", "infer_path"])
    _write_csv(output_dir / "paired_results.csv", pairwise, list(pairwise[0]) if pairwise else ["condition_id"])
    _write_csv(output_dir / "video_index.csv", video_rows, list(video_rows[0]) if video_rows else ["condition_id"])
    payload = {
        "experiment_id": manifest.get("experiment_id"), "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "status": "completed" if not errors and len(results) == 5 else "incomplete",
        "integrity_passed": not errors and len(results) == 5, "integrity_errors": errors,
        "condition_count": len(results), "episode_record_count": sum(item["task_count"] for item in results),
        "conditions": results, "paired_comparisons_vs_control": pairwise,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    report = [
        "# LIBERO multi-task VLM pruning experiment", "",
        f"- experiment: `{manifest.get('experiment_id')}`", f"- status: `{payload['status']}`",
        f"- taskset: {expected_count} tasks; conditions: {len(results)}/5", "",
        "| Condition | Layers | Success | Δ vs Control | New failures | McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    pair_by_id = {row["condition_id"]: row for row in pairwise}
    for result in results:
        pair = pair_by_id.get(result["condition_id"], {})
        report.append(f"| {result['condition_id']} | {result['num_vlm_layers']} | {result['successes']}/{result['task_count']} ({result['success_rate']:.1%}) | {pair.get('success_rate_delta_vs_control', 0):+.1%} | {pair.get('new_failures_vs_control', 0)} | {pair.get('mcnemar_exact_p', 1):.4f} |")
    if errors:
        report += ["", "## Integrity errors", "", *[f"- {error}" for error in errors]]
    (output_dir / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    errors = summarize(args.manifest.expanduser().resolve(), args.output_dir.expanduser().resolve())
    if errors:
        for error in errors:
            print(f"- {error}")
    else:
        print(f"summary written to {args.output_dir.resolve()}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
