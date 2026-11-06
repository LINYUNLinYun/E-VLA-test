#!/usr/bin/env python3
"""Send a bounded number of real LIBERO observations to the pi0.5 server."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY_REPO = ROOT / "policies" / "openpi"
sys.path.insert(0, str(POLICY_REPO / "examples" / "libero"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--requests", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=ROOT / "profile_results" / "pi05" / "client.json")
    return parser.parse_args()


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def main() -> None:
    args = parse_args()
    if args.requests < 1:
        raise ValueError("--requests must be >= 1")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    task_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)
    env.reset()
    observation = env.set_init_state(initial_states[args.initial_state])
    dummy_action = [0.0] * 6 + [-1.0]
    for _ in range(10):
        observation, _, _, _ = env.step(dummy_action)

    image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(image, args.resize_size, args.resize_size)
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, args.resize_size, args.resize_size)
    )
    policy_observation = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate(
            (
                observation["robot0_eef_pos"],
                quat2axisangle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task.language),
    }

    client = WebsocketClientPolicy(args.host, args.port)
    result = None
    for _ in range(args.requests):
        result = client.infer(policy_observation)
    assert result is not None
    action = result["actions"][0].tolist()
    _, reward, done, _ = env.step(action)
    env.close()

    output = {
        "server": f"{args.host}:{args.port}",
        "requests": args.requests,
        "task_suite": "libero_spatial",
        "task_id": args.task_id,
        "task_description": task.language,
        "initial_state": args.initial_state,
        "seed": args.seed,
        "action_shape": list(result["actions"].shape),
        "benchmark_step": {"reward": float(reward), "done": bool(done)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
