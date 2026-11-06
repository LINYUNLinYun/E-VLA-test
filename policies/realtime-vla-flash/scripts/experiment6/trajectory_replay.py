#!/usr/bin/env python3
"""Replay saved Experiment 5 actions and recover semantic simulator telemetry."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/spec"))
import spec_client_libero as libero_client  # noqa: E402
from libero.libero import benchmark  # noqa: E402

try:
    from .trajectory_analysis import (  # type: ignore
        TELEMETRY_SCHEMA_VERSION,
        artifact_index,
        executed_actions,
        read_csv,
        read_jsonl,
        nonself_target_distances,
        telemetry_summary,
        write_csv,
        write_jsonl,
    )
except ImportError:
    from trajectory_analysis import (  # type: ignore
        TELEMETRY_SCHEMA_VERSION,
        artifact_index,
        executed_actions,
        read_csv,
        read_jsonl,
        nonself_target_distances,
        telemetry_summary,
        write_csv,
        write_jsonl,
    )


MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def _load_testset(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload.get("tasks", []))
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        item = dict(sample)
        state_path = Path(str(item["init_state_path"]))
        if not state_path.is_absolute():
            state_path = path.parent / state_path
        item["resolved_init_state_path"] = state_path.resolve()
        result[str(item["sample_id"])] = item
    return result


def _as_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_as_list(item) for item in value]
    return value


def _predicate_rows(raw_env: Any) -> list[dict[str, Any]]:
    rows = []
    for state in raw_env.parsed_problem.get("goal_state", []):
        try:
            satisfied = bool(raw_env._eval_predicate(state))
        except Exception:
            satisfied = False
        rows.append({"predicate": [str(value) for value in state], "satisfied": satisfied})
    return rows


def _goal_object_names(raw_env: Any) -> tuple[list[str], list[str]]:
    states = raw_env.parsed_problem.get("goal_state", [])
    object_names = set(raw_env.object_states_dict)
    primary: list[str] = []
    targets: list[str] = []
    for state in states:
        if len(state) >= 2 and str(state[1]) in object_names:
            primary.append(str(state[1]))
        if len(state) >= 3 and str(state[2]) in object_names:
            targets.append(str(state[2]))
    return sorted(set(primary)), sorted(set(targets))


def _object_state(raw_env: Any, name: str) -> dict[str, Any]:
    state = raw_env.object_states_dict[name]
    result: dict[str, Any] = {"name": name, "type": getattr(state, "object_state_type", "unknown")}
    try:
        geom = state.get_geom_state()
        result["pos"] = _as_list(geom.get("pos"))
        result["quat"] = _as_list(geom.get("quat"))
    except Exception:
        result["pos"] = None
        result["quat"] = None
    try:
        result["joint"] = _as_list(state.get_joint_state())
    except Exception:
        result["joint"] = None
    return result


def _robot_contact(raw_env: Any, object_name: str) -> tuple[bool, bool]:
    try:
        gripper = raw_env.robots[0].gripper
        obj = raw_env.get_object(object_name)
        contact = bool(raw_env.check_contact(gripper, obj))
    except Exception:
        contact = False
    try:
        grasp = bool(raw_env._check_grasp(gripper=raw_env.robots[0].gripper, object_geoms=obj))
    except Exception:
        grasp = False
    return contact, grasp


def _distance(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
    except (TypeError, ValueError):
        return None


def snapshot(
    env: Any,
    obs: dict[str, Any],
    *,
    step: int,
    when: str,
    action: list[float] | None,
    primary_names: list[str],
    target_names: list[str],
    initial_positions: dict[str, list[float] | None],
) -> dict[str, Any]:
    raw = env.env
    eef_pos = _as_list(obs.get("robot0_eef_pos"))
    gripper_qpos = _as_list(obs.get("robot0_gripper_qpos"))
    objects = {name: _object_state(raw, name) for name in sorted(set(primary_names + target_names))}
    contacts: dict[str, bool] = {}
    grasps: dict[str, bool] = {}
    eef_distances: dict[str, float | None] = {}
    displacements: dict[str, float | None] = {}
    for name in primary_names:
        contact, grasp = _robot_contact(raw, name)
        contacts[name] = contact
        grasps[name] = grasp
        position = objects.get(name, {}).get("pos")
        eef_distances[name] = _distance(eef_pos, position)
        displacements[name] = _distance(initial_positions.get(name), position)
    target_distances: dict[str, float | None] = {}
    for left in primary_names:
        for right in target_names:
            # A fixture can be both a unary goal object (for example, Open)
            # and the target of another predicate.  Its distance to itself is
            # always zero and must not be treated as placement progress.
            if left == right:
                continue
            target_distances[f"{left}->{right}"] = _distance(
                objects.get(left, {}).get("pos"), objects.get(right, {}).get("pos")
            )
    predicates = _predicate_rows(raw)
    qpos = np.asarray(gripper_qpos if gripper_qpos is not None else [], dtype=float)
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "step": int(step),
        "when": when,
        "action": action,
        "eef_pos": eef_pos,
        "eef_quat": _as_list(obs.get("robot0_eef_quat")),
        "gripper_qpos": gripper_qpos,
        "gripper_width_proxy": float(np.sum(np.abs(qpos))) if qpos.size else None,
        "primary_objects": primary_names,
        "target_objects": target_names,
        "objects": objects,
        "eef_object_distances": eef_distances,
        "object_displacements": displacements,
        "object_target_distances": target_distances,
        "robot_contacts": contacts,
        "robot_grasps": grasps,
        "goal_predicates": predicates,
        "goal_progress": sum(row["satisfied"] for row in predicates),
        "goal_predicate_count": len(predicates),
        "success": bool(predicates) and all(row["satisfied"] for row in predicates),
    }


def _finite_values(mapping: dict[str, Any]) -> list[float]:
    return [float(value) for value in mapping.values() if value is not None and math.isfinite(float(value))]


def assign_semantic_phases(rows: list[dict[str, Any]]) -> None:
    """Assign conservative evidence-backed stages to post-action telemetry."""
    if not rows:
        return
    initial_distances = _finite_values(rows[0].get("eef_object_distances", {}))
    initial_min = min(initial_distances) if initial_distances else None
    initial_goal_progress = int(rows[0].get("goal_progress", 0))
    ever_grasped = False
    ever_transport = False
    previous_command = None
    for row in rows:
        distances = _finite_values(row.get("eef_object_distances", {}))
        min_distance = min(distances) if distances else None
        target_distances = _finite_values(nonself_target_distances(row.get("object_target_distances", {})))
        min_target = min(target_distances) if target_distances else None
        displacements = _finite_values(row.get("object_displacements", {}))
        max_displacement = max(displacements) if displacements else 0.0
        contact = any(bool(value) for value in row.get("robot_contacts", {}).values())
        grasp = any(bool(value) for value in row.get("robot_grasps", {}).values())
        command = None if row.get("action") is None else float(row["action"][6])
        # LIBERO's normalized action convention is +1 close, -1 open.
        released = ever_grasped and previous_command is not None and previous_command > 0.2 and command is not None and command < -0.2
        ever_grasped = ever_grasped or grasp
        ever_transport = ever_transport or (ever_grasped and max_displacement >= 0.03)

        if row.get("success"):
            phase = "success"
        elif released and (min_target is None or min_target <= 0.12):
            phase = "release_or_final_action"
        elif int(row.get("goal_progress", 0)) > initial_goal_progress or (
            ever_transport and min_target is not None and min_target <= 0.10
        ):
            phase = "goal_alignment"
        elif ever_transport:
            phase = "transport"
        elif grasp or (contact and command is not None and command > 0.2):
            phase = "grasp_or_manipulate"
        elif contact or (min_distance is not None and min_distance <= 0.06):
            phase = "contact"
        elif min_distance is not None and (
            min_distance <= 0.15 or (initial_min is not None and min_distance <= initial_min - 0.04)
        ):
            phase = "approach"
        else:
            phase = "initial"
        row["semantic_phase"] = phase
        row["min_eef_object_distance"] = min_distance
        row["min_object_target_distance"] = min_target
        row["max_primary_object_displacement"] = max_displacement
        previous_command = command if command is not None else previous_command


def replay_one(
    *,
    row: dict[str, Any],
    sample: dict[str, Any],
    suite: Any,
    output_path: Path,
    num_steps_wait: int,
    seed: int,
) -> dict[str, Any]:
    task_id = int(sample["task_id"])
    task = suite.get_task(task_id)
    env_seed = int(sample.get("env_seed", seed))
    env, _ = libero_client._get_libero_env(task, libero_client.LIBERO_ENV_RESOLUTION, env_seed)
    trace = read_jsonl(row["trace_file"])
    actions = executed_actions(trace)
    telemetry: list[dict[str, Any]] = []
    replay_done = False
    try:
        env.reset()
        init_state = np.load(sample["resolved_init_state_path"], allow_pickle=False)
        obs = env.set_init_state(np.asarray(init_state, dtype=np.float64))
        for _ in range(num_steps_wait):
            obs, _, replay_done, _ = env.step(libero_client.LIBERO_DUMMY_ACTION)
        raw = env.env
        primary_names, target_names = _goal_object_names(raw)
        initial_objects = {name: _object_state(raw, name).get("pos") for name in primary_names}
        for step, action in enumerate(actions):
            obs, reward, replay_done, _ = env.step(action)
            item = snapshot(
                env,
                obs,
                step=step,
                when="post_action",
                action=action,
                primary_names=primary_names,
                target_names=target_names,
                initial_positions=initial_objects,
            )
            item["reward"] = float(reward)
            item["done_after_step"] = bool(replay_done)
            telemetry.append(item)
            if replay_done:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    assign_semantic_phases(telemetry)
    write_jsonl(output_path, telemetry)
    expected_success = bool(row["success"])
    phase_summary = telemetry_summary(telemetry)
    return {
        "condition_id": row["condition_id"],
        "sample_id": row["sample_id"],
        "suite_name": sample["suite_name"],
        "task_id": task_id,
        "expected_success": expected_success,
        "replay_success": bool(replay_done),
        "success_match": expected_success == bool(replay_done),
        "expected_action_count": len(actions),
        "replayed_action_count": len(telemetry),
        "action_count_match": len(actions) == len(telemetry),
        "max_phase": phase_summary["furthest_phase"] if telemetry else "missing",
        "telemetry_path": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-results", type=Path, required=True)
    parser.add_argument("--video-index", type=Path, required=True)
    parser.add_argument("--testset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", default="Control,V16,V14,V12,V10-Mid")
    parser.add_argument("--sample-ids", help="Comma-separated sample IDs")
    parser.add_argument("--paired-analysis", type=Path, help="Replay only Control/new-failure candidate pairs from this CSV")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    artifacts = artifact_index(args.episode_results.resolve(), args.video_index.resolve())
    samples = _load_testset(args.testset_manifest.resolve())
    conditions = {value.strip() for value in args.conditions.split(",") if value.strip()}
    sample_filter = None if not args.sample_ids else {value.strip() for value in args.sample_ids.split(",") if value.strip()}
    paired_keys = None
    if args.paired_analysis:
        paired_keys = set()
        for pair in read_csv(args.paired_analysis.resolve()):
            if pair.get("cohort") != "new_failure" or pair.get("condition_id") not in conditions:
                continue
            sample_id = str(pair["sample_id"])
            paired_keys.add((str(pair["condition_id"]), sample_id))
            paired_keys.add(("Control", sample_id))
    selected = [
        row for (condition, sample_id), row in sorted(artifacts.items())
        if condition in conditions
        and (sample_filter is None or sample_id in sample_filter)
        and (paired_keys is None or (condition, sample_id) in paired_keys)
    ]
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]
    suites: dict[str, Any] = {}
    benchmark_dict = benchmark.get_benchmark_dict()
    summaries = []
    for index, row in enumerate(selected, 1):
        sample = samples[row["sample_id"]]
        suite_name = str(sample["suite_name"])
        suites.setdefault(suite_name, benchmark_dict[suite_name]())
        output_path = args.output_dir.resolve() / row["condition_id"] / f"{row['sample_id']}.telemetry.jsonl"
        summary = replay_one(
            row=row,
            sample=sample,
            suite=suites[suite_name],
            output_path=output_path,
            num_steps_wait=args.num_steps_wait,
            seed=args.seed,
        )
        summaries.append(summary)
        print(f"[{index}/{len(selected)}] {row['condition_id']}/{row['sample_id']} match={summary['success_match']}", flush=True)
    fields = list(summaries[0]) if summaries else ("condition_id", "sample_id")
    write_csv(args.output_dir.resolve() / "replay_validation.csv", summaries, fields)
    report = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "episode_count": len(summaries),
        "success_matches": sum(bool(row["success_match"]) for row in summaries),
        "action_count_matches": sum(bool(row["action_count_match"]) for row in summaries),
        "all_success_match": all(bool(row["success_match"]) for row in summaries),
        "all_action_count_match": all(bool(row["action_count_match"]) for row in summaries),
    }
    (args.output_dir.resolve() / "replay_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_success_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
