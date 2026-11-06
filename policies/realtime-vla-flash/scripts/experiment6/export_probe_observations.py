#!/usr/bin/env python3
"""Replay Control trajectories and export phase-stratified model observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/spec"))
import spec_client_libero as libero_client  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from trajectory_analysis import artifact_index, executed_actions, read_jsonl  # noqa: E402
from trajectory_replay import _load_testset  # noqa: E402


def _phase_steps(path: Path | None, requested: tuple[str, ...], action_count: int) -> dict[int, str]:
    selected: dict[int, str] = {0: "initial"}
    if path is not None and path.is_file():
        rows = read_jsonl(path)
        for phase in requested:
            if phase == "initial":
                continue
            match = next((int(row["step"]) + 1 for row in rows if row.get("semantic_phase") == phase), None)
            if match is not None and match < action_count:
                selected.setdefault(match, phase)
    if len(selected) == 1:
        for fraction, name in ((0.25, "quarter"), (0.5, "middle"), (0.75, "late")):
            selected.setdefault(min(action_count - 1, round(fraction * action_count)), name)
    return selected


def export_one(
    *,
    artifact: dict[str, Any],
    sample: dict[str, Any],
    suite: Any,
    telemetry_path: Path | None,
    phases: tuple[str, ...],
    output_dir: Path,
    num_steps_wait: int,
    resize_size: int,
    seed: int,
) -> list[Path]:
    actions = executed_actions(read_jsonl(artifact["trace_file"]))
    selected = _phase_steps(telemetry_path, phases, len(actions))
    task = suite.get_task(int(sample["task_id"]))
    env, prompt = libero_client._get_libero_env(
        task,
        libero_client.LIBERO_ENV_RESOLUTION,
        int(sample.get("env_seed", seed)),
    )
    outputs: list[Path] = []
    try:
        env.reset()
        state = np.load(sample["resolved_init_state_path"], allow_pickle=False)
        obs = env.set_init_state(np.asarray(state, dtype=np.float64))
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(libero_client.LIBERO_DUMMY_ACTION)
        for step in range(len(actions)):
            if step in selected:
                image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                image = libero_client.image_tools.convert_to_uint8(
                    libero_client.image_tools.resize_with_pad(image, resize_size, resize_size)
                )
                wrist = libero_client.image_tools.convert_to_uint8(
                    libero_client.image_tools.resize_with_pad(wrist, resize_size, resize_size)
                )
                robot_state = np.concatenate(
                    (obs["robot0_eef_pos"], libero_client._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ).astype(np.float32)
                phase = selected[step]
                path = output_dir / f"{sample['sample_id']}__step{step:04d}__{phase}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "sample_id": sample["sample_id"],
                    "suite_name": sample["suite_name"],
                    "task_id": int(sample["task_id"]),
                    "trajectory_step": step,
                    "semantic_phase": phase,
                    "source_trace": str(artifact["trace_file"]),
                }
                np.savez_compressed(
                    path,
                    agent_image=image,
                    wrist_image=wrist,
                    robot_state=robot_state,
                    prompt=np.asarray(str(prompt)),
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
                outputs.append(path)
            obs, _, done, _ = env.step(actions[step])
            if done:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-results", type=Path, required=True)
    parser.add_argument("--video-index", type=Path, required=True)
    parser.add_argument("--testset-manifest", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--phases", default="initial,approach,contact,grasp_or_manipulate,transport,goal_alignment,release_or_final_action")
    parser.add_argument("--condition", default="Control")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    artifacts = artifact_index(args.episode_results.resolve(), args.video_index.resolve())
    samples = _load_testset(args.testset_manifest.resolve())
    requested = [value.strip() for value in args.sample_ids.split(",") if value.strip()]
    phases = tuple(value.strip() for value in args.phases.split(",") if value.strip())
    benchmark_dict = benchmark.get_benchmark_dict()
    suites: dict[str, Any] = {}
    output_paths = []
    for sample_id in requested:
        artifact = artifacts[(args.condition, sample_id)]
        sample = samples[sample_id]
        suite_name = str(sample["suite_name"])
        suites.setdefault(suite_name, benchmark_dict[suite_name]())
        telemetry_path = None if args.telemetry_dir is None else args.telemetry_dir.resolve() / args.condition / f"{sample_id}.telemetry.jsonl"
        output_paths.extend(
            export_one(
                artifact=artifact,
                sample=sample,
                suite=suites[suite_name],
                telemetry_path=telemetry_path,
                phases=phases,
                output_dir=args.output_dir.resolve(),
                num_steps_wait=args.num_steps_wait,
                resize_size=args.resize_size,
                seed=args.seed,
            )
        )
    manifest = {"observation_count": len(output_paths), "observations": [str(path) for path in output_paths]}
    (args.output_dir.resolve() / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
