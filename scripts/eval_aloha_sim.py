#!/usr/bin/env python3
"""Evaluate an OpenPI websocket policy on gym-aloha simulation tasks."""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

import gym_aloha  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8200
    task: str = "gym_aloha/AlohaTransferCube-v0"
    prompt: str = "Transfer cube"
    num_trials: int = 50
    seed: int = 0
    action_horizon: int = 10
    max_steps: int = 300
    resize_size: int = 224
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/baseline_transfer_cube")
    save_video: bool = True
    video_fps: int = 50
    reset_policy_state: bool = False


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _observation_payload(
    obs: dict[str, Any],
    *,
    args: Args,
    reset_policy_state: bool,
) -> dict[str, Any]:
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(obs["pixels"]["top"], args.resize_size, args.resize_size)
    )
    payload: dict[str, Any] = {
        "state": np.asarray(obs["agent_pos"], dtype=np.float32),
        "images": {"cam_high": np.transpose(image, (2, 0, 1))},
        "prompt": str(args.prompt),
    }
    if reset_policy_state:
        payload["__reset_policy_state__"] = True
    return payload


def main(args: Args) -> None:
    if args.num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if args.action_horizon <= 0:
        raise ValueError("action_horizon must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.out_dir / "manifest.json",
        {
            "task": args.task,
            "prompt": args.prompt,
            "num_trials": int(args.num_trials),
            "seed": int(args.seed),
            "action_horizon": int(args.action_horizon),
            "max_steps": int(args.max_steps),
            "resize_size": int(args.resize_size),
            "trajectory_length_definition": "Number of actions applied to the gym-aloha environment.",
        },
    )

    env = gym.make(args.task, obs_type="pixels_agent_pos")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    records: list[dict[str, Any]] = []

    try:
        for episode_idx in range(int(args.num_trials)):
            episode_seed = int(args.seed) + int(episode_idx)
            obs, _ = env.reset(seed=episode_seed)
            action_plan: collections.deque[np.ndarray] = collections.deque()
            frames: list[np.ndarray] = []
            infer_records: list[dict[str, Any]] = []
            max_reward = float("-inf")
            success = False
            error: str | None = None
            reset_pending = bool(args.reset_policy_state)
            executed_steps = 0

            for step in range(int(args.max_steps)):
                if args.save_video:
                    frames.append(np.asarray(obs["pixels"]["top"], dtype=np.uint8))

                if not action_plan:
                    infer_id = len(infer_records)
                    payload = _observation_payload(
                        obs,
                        args=args,
                        reset_policy_state=reset_pending,
                    )
                    reset_pending = False
                    try:
                        start = time.perf_counter()
                        output = client.infer(payload)
                        roundtrip_ms = (time.perf_counter() - start) * 1000.0
                    except Exception as exc:  # preserve partial results for diagnosis
                        error = f"{type(exc).__name__}: {exc}"
                        logging.exception("Episode %d inference failed", episode_idx)
                        break

                    actions = np.asarray(output["actions"], dtype=np.float32)
                    accepted = int(output.get("accepted_prefix_len", args.action_horizon))
                    execute = min(int(args.action_horizon), int(actions.shape[0]), max(0, accepted))
                    if execute <= 0:
                        error = f"policy returned no executable action: accepted_prefix_len={accepted}"
                        break
                    for action in actions[:execute]:
                        action_plan.append(np.asarray(action, dtype=np.float32))

                    timing = output.get("policy_timing", {})
                    if not isinstance(timing, dict):
                        timing = {}
                    infer_records.append(
                        {
                            "infer_id": int(infer_id),
                            "env_step": int(step),
                            "client_roundtrip_ms": float(roundtrip_ms),
                            "accepted_prefix_len": int(accepted),
                            "executed_prefix_len": int(execute),
                            "is_full_pipeline_round": int(round(float(timing.get("is_full_pipeline_round", 0.0)))),
                            "sample_actions_ms": timing.get("sample_actions_ms"),
                            "total_ms": timing.get("total_ms"),
                        }
                    )

                action = action_plan.popleft()
                obs, reward, terminated, truncated, info = env.step(action)
                executed_steps += 1
                max_reward = max(max_reward, float(reward))
                success = success or bool(info.get("is_success", False))
                if terminated or truncated:
                    break

            trajectory_length = int(executed_steps)
            video_path: pathlib.Path | None = None
            if args.save_video and frames:
                suffix = "success" if success else "failure"
                video_path = args.out_dir / "videos" / f"episode_{episode_idx:03d}_{suffix}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_path, frames, fps=int(args.video_fps))

            latency_values = [float(r["client_roundtrip_ms"]) for r in infer_records]
            record = {
                "episode_idx": int(episode_idx),
                "seed": int(episode_seed),
                "success": bool(success),
                "max_reward": None if max_reward == float("-inf") else float(max_reward),
                "trajectory_length": int(trajectory_length),
                "infer_calls": int(len(infer_records)),
                "client_roundtrip_mean_ms": float(np.mean(latency_values)) if latency_values else None,
                "client_roundtrip_median_ms": float(np.median(latency_values)) if latency_values else None,
                "failure_reason": error,
                "video_path": None if video_path is None else str(video_path.relative_to(args.out_dir)),
                "infer_records": infer_records,
            }
            records.append(record)
            _write_json(args.out_dir / "episode_log.json", records)
            successes = sum(int(row["success"]) for row in records)
            logging.info(
                "episode=%d success=%s reward=%s steps=%d aggregate=%d/%d",
                episode_idx,
                success,
                record["max_reward"],
                trajectory_length,
                successes,
                len(records),
            )
    finally:
        env.close()

    successes = sum(int(record["success"]) for record in records)
    failures = len(records) - successes
    success_lengths = [int(record["trajectory_length"]) for record in records if record["success"]]
    failure_lengths = [int(record["trajectory_length"]) for record in records if not record["success"]]
    all_lengths = np.asarray([int(record["trajectory_length"]) for record in records], dtype=np.float64)
    all_successes = np.asarray([bool(record["success"]) for record in records], dtype=np.float64)
    if len(records) > 1 and np.std(all_lengths) > 0 and np.std(all_successes) > 0:
        point_biserial_r: float | None = float(np.corrcoef(all_lengths, all_successes)[0, 1])
    else:
        point_biserial_r = None
    summary = {
        "episodes": len(records),
        "successes": successes,
        "failures": failures,
        "success_rate": successes / len(records),
        "trajectory_length": {
            "definition": "Number of actions applied before success termination or the 300-step time limit.",
            "success_mean": float(np.mean(success_lengths)) if success_lengths else None,
            "success_median": float(np.median(success_lengths)) if success_lengths else None,
            "failure_mean": float(np.mean(failure_lengths)) if failure_lengths else None,
            "failure_median": float(np.median(failure_lengths)) if failure_lengths else None,
            "point_biserial_r_with_success": point_biserial_r,
        },
    }
    _write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
