"""Shared readers and action-level diagnostics for Experiment 6.

The Experiment 5 traces do not contain simulator state.  This module therefore
labels action-only quantities as *proxies* and keeps them separate from semantic
phases produced by :mod:`trajectory_replay`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


ACTION_PROXY_PHASES = (
    "initial_motion_proxy",
    "approach_proxy",
    "gripper_manipulation_proxy",
    "late_precision_proxy",
)

TELEMETRY_SCHEMA_VERSION = "experiment6.telemetry.v1"
SEMANTIC_PHASES = (
    "initial",
    "approach",
    "contact",
    "grasp_or_manipulate",
    "transport",
    "goal_alignment",
    "release_or_final_action",
    "success",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (rows[0].keys() if rows else ("condition_id",)))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def nonself_target_distances(mapping: dict[str, Any]) -> dict[str, Any]:
    """Drop fixture-to-itself pairs that cannot indicate goal alignment."""

    result: dict[str, Any] = {}
    for key, value in mapping.items():
        if "->" in key:
            left, right = key.split("->", 1)
            if left == right:
                continue
        result[key] = value
    return result


def first_unreached_transition(
    candidate_furthest: str,
    control_phases: Iterable[str],
    candidate_phases: Iterable[str],
) -> str:
    """Name the earliest later phase reached by Control but not candidate."""

    rank = {phase: index for index, phase in enumerate(SEMANTIC_PHASES)}
    current_rank = rank.get(candidate_furthest, -1)
    control_set = set(control_phases)
    candidate_set = set(candidate_phases)
    for phase in SEMANTIC_PHASES:
        if rank[phase] > current_rank and phase in control_set and phase not in candidate_set:
            return f"{candidate_furthest}->{phase}"
    return ""


def artifact_index(episode_csv: Path, video_csv: Path | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """Load episodes and resolve trace/video artifacts.

    Experiment 5's episode CSV stores paths relative to each run directory,
    whereas its video index stores absolute video paths.  The latter lets us
    recover the sibling trace without guessing run names.
    """

    if episode_csv.suffix.lower() == ".json":
        payload = json.loads(episode_csv.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"Expected a JSON list of episode objects: {episode_csv}")
        episodes = [dict(row) for row in payload]
        manifest_path = episode_csv.with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        default_condition = manifest.get("condition_id")
        for row in episodes:
            if not row.get("condition_id"):
                if not default_condition:
                    raise ValueError(
                        f"Episode has no condition_id and sibling manifest supplies none: {episode_csv}"
                    )
                row["condition_id"] = str(default_condition)
    else:
        episodes = read_csv(episode_csv)
    videos: dict[tuple[str, str], Path] = {}
    if video_csv and video_csv.is_file():
        for row in read_csv(video_csv):
            videos[(row["condition_id"], row["sample_id"])] = Path(row["video_path"]).expanduser().resolve()
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in episodes:
        key = (row["condition_id"], row["sample_id"])
        video = videos.get(key)
        if video is None:
            candidate = Path(row.get("video_path", "")).expanduser()
            video = candidate.resolve() if candidate.is_absolute() else (episode_csv.parent / candidate).resolve()
        trace_value = Path(row.get("trace_path", "trace.jsonl"))
        trace = video.parent / trace_value.name
        result[key] = {
            **row,
            "success": bool_value(row.get("success")),
            "video_file": video,
            "trace_file": trace,
            "infer_file": video.parent / Path(row.get("infer_path", "infer.jsonl")).name,
        }
    return result


def executed_actions(trace_rows: Iterable[dict[str, Any]]) -> list[list[float]]:
    actions: list[list[float]] = []
    for row in trace_rows:
        action = row.get("executed_action")
        if isinstance(action, list) and len(action) >= 7:
            actions.append([float(value) for value in action[:7]])
    return actions


def gripper_events(actions: list[list[float]], threshold: float = 0.2) -> list[dict[str, Any]]:
    """Return debounced open/close sign transitions from action dimension 6."""

    events: list[dict[str, Any]] = []
    stable: int | None = None
    for index, action in enumerate(actions):
        value = float(action[6])
        state = 1 if value >= threshold else (-1 if value <= -threshold else 0)
        if state == 0:
            continue
        if stable is not None and state != stable:
            events.append({"step": index, "event": "close" if state > 0 else "open", "value": value})
        stable = state
    return events


def action_summary(actions: list[list[float]]) -> dict[str, Any]:
    if not actions:
        return {
            "action_count": 0,
            "translation_path_proxy": 0.0,
            "rotation_path_proxy": 0.0,
            "gripper_transition_count": 0,
            "first_gripper_transition_step": None,
        }
    translation = sum(math.sqrt(sum(float(v) ** 2 for v in action[:3])) for action in actions)
    rotation = sum(math.sqrt(sum(float(v) ** 2 for v in action[3:6])) for action in actions)
    events = gripper_events(actions)
    return {
        "action_count": len(actions),
        "translation_path_proxy": translation,
        "rotation_path_proxy": rotation,
        "gripper_transition_count": len(events),
        "first_gripper_transition_step": events[0]["step"] if events else None,
        "gripper_events": events,
    }


def first_action_divergence(
    control: list[list[float]],
    candidate: list[list[float]],
    translation_threshold: float = 0.20,
    rotation_threshold: float = 0.20,
) -> dict[str, Any]:
    """Find the first paired executed-action disagreement.

    This is a policy-output divergence, not a state-space divergence.  Once two
    rollouts visit different observations, it must not be interpreted causally.
    """

    length = min(len(control), len(candidate))
    for index in range(length):
        trans = math.sqrt(sum((candidate[index][j] - control[index][j]) ** 2 for j in range(3)))
        rot = math.sqrt(sum((candidate[index][j] - control[index][j]) ** 2 for j in range(3, 6)))
        grip = (candidate[index][6] >= 0) != (control[index][6] >= 0)
        if trans >= translation_threshold or rot >= rotation_threshold or grip:
            return {
                "first_divergence_step": index,
                "first_divergence_fraction": index / max(length - 1, 1),
                "translation_delta": trans,
                "rotation_delta": rot,
                "gripper_sign_mismatch": grip,
            }
    return {
        "first_divergence_step": None,
        "first_divergence_fraction": None,
        "translation_delta": None,
        "rotation_delta": None,
        "gripper_sign_mismatch": False,
    }


def proxy_phase(step: int | None, action_count: int, first_gripper_step: int | None) -> str:
    if step is None or action_count <= 0:
        return "no_detected_divergence"
    fraction = step / max(action_count - 1, 1)
    if fraction < 0.10:
        return ACTION_PROXY_PHASES[0]
    if first_gripper_step is None or step < first_gripper_step:
        return ACTION_PROXY_PHASES[1]
    if fraction < 0.80:
        return ACTION_PROXY_PHASES[2]
    return ACTION_PROXY_PHASES[3]


def telemetry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return evidence-backed phase/progress summary from replay telemetry."""

    rank = {
        "initial": 0,
        "approach": 1,
        "contact": 2,
        "grasp_or_manipulate": 3,
        "transport": 4,
        "goal_alignment": 5,
        "release_or_final_action": 6,
        "success": 7,
    }
    best, best_step = "initial", None
    reached: list[str] = []
    for row in rows:
        phase = str(row.get("semantic_phase", "initial"))
        if phase not in reached:
            reached.append(phase)
        if rank.get(phase, -1) > rank[best]:
            best, best_step = phase, int(row.get("step", 0))
        if bool(row.get("success")):
            best, best_step = "success", int(row.get("step", 0))
            break
    return {
        "furthest_phase": best,
        "furthest_phase_step": best_step,
        "reached_phases": reached,
        "max_goal_progress": max((int(row.get("goal_progress", 0)) for row in rows), default=0),
        "goal_predicate_count": max((int(row.get("goal_predicate_count", 0)) for row in rows), default=0),
        "success": any(bool(row.get("success")) for row in rows),
    }


def telemetry_phase(rows: list[dict[str, Any]]) -> tuple[str, int | None]:
    """Backward-compatible tuple wrapper around :func:`telemetry_summary`."""
    summary = telemetry_summary(rows)
    return str(summary["furthest_phase"]), summary["furthest_phase_step"]
