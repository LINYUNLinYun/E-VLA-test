#!/usr/bin/env python3
"""Audit Experiment 5's Control against the repository's true pi0 baseline.

This is deliberately a read-only analyzer.  It distinguishes an internal
ablation control from a standard baseline and records enough evidence to make
that distinction reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _old_baselines(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*/summary.json")):
        try:
            payload = _json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for condition in payload.get("conditions", []):
            if not isinstance(condition, dict):
                continue
            is_baseline = bool(condition.get("is_baseline")) or str(condition.get("condition_id")) in {
                "full18__full10", "E18_D10", "Control"
            }
            if not is_baseline:
                continue
            count = condition.get("episode_count", condition.get("task_count"))
            successes = condition.get("successes")
            rate = condition.get("success_rate")
            if count is None or rate is None:
                continue
            rows.append(
                {
                    "source": str(path),
                    "condition_id": condition.get("condition_id"),
                    "episode_count": int(count),
                    "successes": None if successes is None else int(successes),
                    "success_rate": float(rate),
                    "scope": payload.get("experiment_id"),
                }
            )
    return rows


def _smoke_result(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None or not run_dir.is_dir():
        return None
    episode_log = run_dir / "episode_log.json"
    manifest = run_dir / "manifest.json"
    if not episode_log.is_file():
        return None
    rows = _json(episode_log)
    return {
        "run_dir": str(run_dir.resolve()),
        "manifest_sha256": _sha256(manifest) if manifest.is_file() else None,
        "episode_count": len(rows),
        "successes": sum(bool(row.get("success")) for row in rows),
        "success_rate": (sum(bool(row.get("success")) for row in rows) / len(rows)) if rows else None,
        "suite_counts": {
            suite: sum(str(row.get("task_suite_name", row.get("suite_name"))) == suite for row in rows)
            for suite in sorted({str(row.get("task_suite_name", row.get("suite_name"))) for row in rows})
        },
    }


def audit(
    *,
    experiment5_results: Path,
    experiment5_config: Path,
    testset_manifest: Path,
    results_root: Path,
    output_dir: Path,
    d10_official_smoke: Path | None = None,
    d10_official_alltasks_smoke: Path | None = None,
    d3_official_smoke: Path | None = None,
    d10_hdf5_smoke: Path | None = None,
) -> dict[str, Any]:
    summary = _json(experiment5_results / "summary.json")
    manifest = _json(experiment5_results / "manifest.json")
    config = _json(experiment5_config)
    testset = _json(testset_manifest)
    condition_rows = _csv(experiment5_results / "condition_results.csv")
    control_row = next(row for row in condition_rows if row["condition_id"] == "Control")
    control_manifest = next(row for row in manifest["conditions"] if row["condition_id"] == "Control")

    standard_defaults = {
        "vlm_layer_pattern": "vlm_full18",
        "action_layer_pattern": "full18",
        "denoise_steps": 10,
        "init_state_source": "task_suite.get_task_init_states(task_id)",
    }
    observed = {
        "vlm_layer_pattern": control_manifest.get("vlm_layer_pattern"),
        "action_layer_pattern": config.get("action_layer_pattern"),
        "denoise_steps": config.get("denoise_steps"),
        "init_state_source": testset.get("selection", {}).get("demo_rule"),
        "resume_used": "--resume" in control_manifest.get("client_command", []),
        "task_count": int(control_row["task_count"]),
        "successes": int(control_row["successes"]),
        "success_rate": float(control_row["success_rate"]),
        "libero_10_success_rate": float(control_row["10_success_rate"]),
        "libero_90_success_rate": float(control_row["90_success_rate"]),
    }
    protocol_mismatches = [
        {
            "field": "denoise_steps",
            "control": observed["denoise_steps"],
            "standard": standard_defaults["denoise_steps"],
            "severity": "critical",
        },
        {
            "field": "init_state_source",
            "control": observed["init_state_source"],
            "standard": standard_defaults["init_state_source"],
            "severity": "critical",
        },
        {
            "field": "rollouts_per_task",
            "control": 1,
            "standard": 50,
            "severity": "statistical",
        },
        {
            "field": "resume_provenance",
            "control": observed["resume_used"],
            "standard": False,
            "severity": "provenance",
        },
    ]
    smoke_runs = {
        "d10_official_init": _smoke_result(d10_official_smoke),
        "d10_official_alltasks": _smoke_result(d10_official_alltasks_smoke),
        "d3_official_init": _smoke_result(d3_official_smoke),
        "d10_hdf5_init": _smoke_result(d10_hdf5_smoke),
    }
    d10_official = smoke_runs["d10_official_init"]
    d3_official = smoke_runs["d3_official_init"]
    d10_hdf5 = smoke_runs["d10_hdf5_init"]
    causal_smoke_interpretation = {
        "denoise_drop_explains_task0_gap": None,
        "hdf5_init_shift_explains_task0_gap": None,
        "evidence_level": "exploratory; causal attribution not established",
    }
    observed_smoke_comparisons = {}
    if d10_official and d3_official:
        observed_smoke_comparisons["d3_rate_lower_than_d10_on_official_task0"] = (
            d3_official["success_rate"] < d10_official["success_rate"]
        )
    if d10_official and d10_hdf5:
        observed_smoke_comparisons["hdf5_rate_lower_than_official_task0"] = (
            d10_hdf5["success_rate"] < d10_official["success_rate"]
        )
    report = {
        "status": "completed",
        "verdict": "internal_ablation_control_not_standard_baseline",
        "experiment5_status": summary.get("status"),
        "control": observed,
        "standard_baseline_definition": standard_defaults,
        "protocol_mismatches": protocol_mismatches,
        "repository_baseline_evidence": _old_baselines(results_root),
        "smoke_runs": smoke_runs,
        "causal_smoke_interpretation": causal_smoke_interpretation,
        "observed_smoke_comparisons": observed_smoke_comparisons,
        "completed_smoke_checks": [
            "task-0 Full18/Full18 D10 on 10 official init states",
            "all 10 LIBERO-10 tasks at Full18/Full18 D10 on official init-state index 0",
            "task-0 D10-vs-D3 on the same 10 official init states",
            "task-0 official-init-vs-HDF5-demo-init at D10 (10-vs-1 exploratory comparison)",
        ],
        "required_followups": [
            "broaden the official-init-vs-HDF5 comparison across tasks with matched rollout counts",
            "run the canonical Full18/Full18/D10 baseline with 50 official rollouts per task",
            "fixed-observation Triton-vs-reference action parity",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Control baseline audit",
        "",
        f"**Verdict:** `{report['verdict']}`.",
        "",
        f"Experiment 5 Control: {observed['successes']}/{observed['task_count']} "
        f"({observed['success_rate']:.1%}); LIBERO-10 {observed['libero_10_success_rate']:.1%}, "
        f"LIBERO-90 {observed['libero_90_success_rate']:.1%}.",
        "",
        "## Protocol mismatches",
        "",
        "| Field | Control | Standard baseline | Severity |",
        "|---|---|---|---|",
    ]
    for item in protocol_mismatches:
        lines.append(f"| {item['field']} | `{item['control']}` | `{item['standard']}` | {item['severity']} |")
    lines.extend(["", "## Controlled smoke runs", "", "| Run | Success |", "|---|---:|"])
    for name, smoke in smoke_runs.items():
        if smoke:
            lines.append(f"| {name} | {smoke['successes']}/{smoke['episode_count']} ({smoke['success_rate']:.1%}) |")
    lines.extend(
        [
            "",
            "Causal attribution is not established. The 10-vs-1 initialization comparison is exploratory; "
            "neither it nor one-task denoising checks explain the 100-task Control result.",
        ]
    )
    lines.extend(["", "The 21% result is valid for paired comparisons inside Experiment 5, but is not a standard π0 baseline."])
    (output_dir / "baseline_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment5-results", type=Path, required=True)
    parser.add_argument("--experiment5-config", type=Path, required=True)
    parser.add_argument("--testset-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--d10-official-smoke", type=Path)
    parser.add_argument("--d10-official-alltasks-smoke", type=Path)
    parser.add_argument("--d3-official-smoke", type=Path)
    parser.add_argument("--d10-hdf5-smoke", type=Path)
    args = parser.parse_args()
    report = audit(
        experiment5_results=args.experiment5_results.resolve(),
        experiment5_config=args.experiment5_config.resolve(),
        testset_manifest=args.testset_manifest.resolve(),
        results_root=args.results_root.resolve(),
        output_dir=args.output_dir.resolve(),
        d10_official_smoke=args.d10_official_smoke.resolve() if args.d10_official_smoke else None,
        d10_official_alltasks_smoke=(
            args.d10_official_alltasks_smoke.resolve() if args.d10_official_alltasks_smoke else None
        ),
        d3_official_smoke=args.d3_official_smoke.resolve() if args.d3_official_smoke else None,
        d10_hdf5_smoke=args.d10_hdf5_smoke.resolve() if args.d10_hdf5_smoke else None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
