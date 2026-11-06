"""Build paired failure cohorts and action/telemetry failure-stage tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .trajectory_analysis import (
        action_summary,
        artifact_index,
        executed_actions,
        first_action_divergence,
        first_unreached_transition,
        proxy_phase,
        read_jsonl,
        telemetry_summary,
        write_csv,
    )
except ImportError:  # direct script execution
    from trajectory_analysis import (  # type: ignore
        action_summary,
        artifact_index,
        executed_actions,
        first_action_divergence,
        first_unreached_transition,
        proxy_phase,
        read_jsonl,
        telemetry_summary,
        write_csv,
    )


def cohort(control_success: bool, candidate_success: bool) -> str:
    if control_success and not candidate_success:
        return "new_failure"
    if not control_success and candidate_success:
        return "candidate_rescue"
    if control_success:
        return "both_success"
    return "both_failure"


def _telemetry_path(root: Path | None, condition: str, sample_id: str) -> Path | None:
    if root is None:
        return None
    value = root / condition / f"{sample_id}.telemetry.jsonl"
    return value if value.is_file() else None


def analyze(
    episode_csv: Path,
    video_csv: Path,
    output_dir: Path,
    control_id: str = "Control",
    telemetry_dir: Path | None = None,
) -> dict[str, Any]:
    episodes = artifact_index(episode_csv, video_csv)
    control = {sample: row for (condition, sample), row in episodes.items() if condition == control_id}
    if not control:
        raise ValueError(f"control condition {control_id!r} is absent")
    conditions = sorted({condition for condition, _ in episodes if condition != control_id})
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for condition in conditions:
        candidate = {sample: row for (cid, sample), row in episodes.items() if cid == condition}
        for sample_id in sorted(set(control) & set(candidate)):
            c_row, p_row = control[sample_id], candidate[sample_id]
            if not c_row["trace_file"].is_file() or not p_row["trace_file"].is_file():
                missing.append(f"{condition}/{sample_id}: missing trace")
                continue
            c_actions = executed_actions(read_jsonl(c_row["trace_file"]))
            p_actions = executed_actions(read_jsonl(p_row["trace_file"]))
            c_summary, p_summary = action_summary(c_actions), action_summary(p_actions)
            divergence = first_action_divergence(c_actions, p_actions)
            phase_source = "action_proxy"
            phase = proxy_phase(
                divergence["first_divergence_step"],
                min(len(c_actions), len(p_actions)),
                p_summary["first_gripper_transition_step"],
            )
            semantic_step = None
            telemetry_path = _telemetry_path(telemetry_dir, condition, sample_id)
            control_telemetry_path = _telemetry_path(telemetry_dir, control_id, sample_id)
            control_semantic = None
            candidate_semantic = None
            first_failed_transition = ""
            if telemetry_path:
                candidate_semantic = telemetry_summary(read_jsonl(telemetry_path))
                phase = str(candidate_semantic["furthest_phase"])
                semantic_step = candidate_semantic["furthest_phase_step"]
                phase_source = "simulator_telemetry"
                if control_telemetry_path:
                    control_semantic = telemetry_summary(read_jsonl(control_telemetry_path))
                    first_failed_transition = first_unreached_transition(
                        phase,
                        control_semantic["reached_phases"],
                        candidate_semantic["reached_phases"],
                    )
                    if not first_failed_transition and bool(control_semantic["success"]) and not bool(candidate_semantic["success"]):
                        first_failed_transition = f"{phase}->success"
            rows.append(
                {
                    "condition_id": condition,
                    "sample_id": sample_id,
                    "suite_name": p_row.get("suite_name"),
                    "task_id": p_row.get("task_id"),
                    "cohort": cohort(bool(c_row["success"]), bool(p_row["success"])),
                    "control_success": c_row["success"],
                    "candidate_success": p_row["success"],
                    "failure_phase": phase,
                    "phase_source": phase_source,
                    "semantic_phase_step": semantic_step,
                    "control_furthest_phase": "" if control_semantic is None else control_semantic["furthest_phase"],
                    "candidate_furthest_phase": "" if candidate_semantic is None else candidate_semantic["furthest_phase"],
                    "control_max_goal_progress": "" if control_semantic is None else control_semantic["max_goal_progress"],
                    "candidate_max_goal_progress": "" if candidate_semantic is None else candidate_semantic["max_goal_progress"],
                    "first_failed_transition": first_failed_transition,
                    **divergence,
                    "control_action_count": c_summary["action_count"],
                    "candidate_action_count": p_summary["action_count"],
                    "control_gripper_transitions": c_summary["gripper_transition_count"],
                    "candidate_gripper_transitions": p_summary["gripper_transition_count"],
                    "candidate_first_gripper_step": p_summary["first_gripper_transition_step"],
                    "control_video": str(c_row["video_file"]),
                    "candidate_video": str(p_row["video_file"]),
                    "candidate_trace": str(p_row["trace_file"]),
                    "telemetry_path": str(telemetry_path) if telemetry_path else "",
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["condition_id", "sample_id"]
    write_csv(output_dir / "paired_failure_analysis.csv", rows, fields)
    new_failures = [row for row in rows if row["cohort"] == "new_failure"]
    write_csv(output_dir / "new_failure_cohort.csv", new_failures, fields)
    counts = Counter((row["condition_id"], row["cohort"], row["failure_phase"], row["phase_source"]) for row in rows)
    aggregate = [
        {"condition_id": key[0], "cohort": key[1], "failure_phase": key[2], "phase_source": key[3], "count": count}
        for key, count in sorted(counts.items())
    ]
    write_csv(output_dir / "failure_phase_counts.csv", aggregate)
    summary = {
        "status": "completed" if not missing else "completed_with_missing_artifacts",
        "control_id": control_id,
        "paired_row_count": len(rows),
        "new_failure_count": len(new_failures),
        "semantic_telemetry_row_count": sum(row["phase_source"] == "simulator_telemetry" for row in rows),
        "action_proxy_row_count": sum(row["phase_source"] == "action_proxy" for row in rows),
        "missing_artifacts": missing,
        "interpretation_warning": (
            "action_proxy phases describe executed-action timing only; they are not evidence of "
            "object approach/contact/grasp. Use trajectory_replay.py telemetry for semantic phases."
        ),
    }
    (output_dir / "failure_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-results", type=Path, required=True)
    parser.add_argument("--video-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-id", default="Control")
    parser.add_argument("--telemetry-dir", type=Path)
    args = parser.parse_args()
    summary = analyze(
        args.episode_results.resolve(), args.video_index.resolve(), args.output_dir.resolve(),
        args.control_id, args.telemetry_dir.resolve() if args.telemetry_dir else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
