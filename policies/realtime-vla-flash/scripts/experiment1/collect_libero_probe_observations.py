"""Replay saved pure-pi0 LIBERO traces and collect raw probe observations.

This script never queries or changes a policy.  It deterministically replays the
already-saved environment actions and snapshots five evenly-spaced observations
from episode 0 of every task in the four LIBERO suites (200 observations total).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    # Keep this byte-for-byte equivalent in semantics to spec_client_libero.py.
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / denominator).astype(np.float32)


def _make_env(task, *, resolution: int, seed: int):
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _episode_zero(run_dir: Path, task_id: int) -> tuple[dict, Path]:
    records = json.loads((run_dir / "episode_log.json").read_text(encoding="utf-8"))
    row = next(r for r in records if int(r["task_id"]) == task_id and int(r["episode_idx"]) == 0)
    return row, run_dir / row["trace_path"]


def _sample_indices(length: int, count: int) -> list[int]:
    if length < count:
        raise ValueError(f"trace length {length} is smaller than requested samples {count}")
    return [int(x) for x in np.linspace(0, length - 1, count).round()]


def collect(*, data_root: Path, output: Path, samples_per_task: int, seed: int, wait_steps: int) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    metadata: list[dict] = []

    for suite_name in SUITES:
        run_dir = data_root / f"pi0_baseline_full_{suite_name}"
        suite = benchmark_dict[suite_name]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            episode, trace_path = _episode_zero(run_dir, task_id)
            trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            trace = [row for row in trace if row.get("executed_action") is not None]
            selected = set(_sample_indices(len(trace), samples_per_task))

            env = _make_env(task, resolution=256, seed=seed)
            try:
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
                for _ in range(wait_steps):
                    obs, _, _, _ = env.step(DUMMY_ACTION)

                for step_idx, row in enumerate(trace):
                    if step_idx in selected:
                        base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        base = image_tools.convert_to_uint8(image_tools.resize_with_pad(base, 224, 224))
                        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ).astype(np.float32)
                        sample_id = f"{suite_name}_task{task_id:02d}_ep000_step{step_idx:04d}"
                        images.append(base)
                        wrist_images.append(wrist)
                        states.append(state)
                        metadata.append(
                            {
                                "sample_id": sample_id,
                                "suite": suite_name,
                                "task_id": task_id,
                                "episode_idx": 0,
                                "task_description": str(task.language),
                                "trajectory_step": step_idx,
                                "trajectory_length": len(trace),
                                "progress": step_idx / max(len(trace) - 1, 1),
                                "episode_success": bool(episode["success"]),
                                "source_trace": str(trace_path),
                            }
                        )
                    obs, _, _, _ = env.step(np.asarray(row["executed_action"], dtype=np.float32).tolist())
            finally:
                env.close()

            print(f"collected {suite_name} task {task_id}: {len(selected)}")

    expected = len(SUITES) * 10 * samples_per_task
    if len(metadata) != expected:
        raise RuntimeError(f"collected {len(metadata)} observations, expected {expected}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image=np.stack(images),
        wrist_image=np.stack(wrist_images),
        state=np.stack(states),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(f"wrote {len(metadata)} observations to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/test"),
        help="Directory containing pi0_baseline_full_<suite> runs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    args = parser.parse_args()
    collect(
        data_root=args.data_root,
        output=args.output,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        wait_steps=args.wait_steps,
    )


if __name__ == "__main__":
    main()
