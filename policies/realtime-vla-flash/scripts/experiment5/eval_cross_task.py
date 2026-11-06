"""Evaluate a frozen multi-suite LIBERO testset against a running policy server."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/spec"))
import spec_client_libero as base  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from openpi_client import websocket_client_policy as websocket_client_policy  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _read_testset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    samples = manifest.get("samples", manifest.get("tasks"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("testset manifest must contain a non-empty 'samples' array")
    normalized: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    sample_ids: set[str] = set()
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            raise ValueError(f"testset sample {index} is not an object")
        suite = item.get("suite_name", item.get("task_suite_name", item.get("suite")))
        task_id = item.get("task_id")
        if not isinstance(suite, str) or not suite.strip() or not isinstance(task_id, int):
            raise ValueError(f"testset sample {index} must have suite_name and integer task_id")
        sample_id = str(item.get("sample_id", item.get("id", f"{suite}:{task_id}")))
        key = (suite, int(task_id))
        if key in keys:
            raise ValueError(f"duplicate suite/task in testset: {suite}:{task_id}")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id in testset: {sample_id}")
        keys.add(key)
        sample_ids.add(sample_id)
        state_data = item.get("init_state")
        state_path = item.get("init_state_path", item.get("init_state_file", item.get("initial_state_path")))
        if isinstance(state_data, dict) and state_path is None:
            state_path = state_data.get("path", state_data.get("file"))
            state_data = state_data.get("data")
        if state_data is None and not state_path:
            raise ValueError(f"sample {sample_id} lacks init_state or init_state_path")
        normalized.append(
            {
                **item,
                "sample_index": index,
                "sample_id": sample_id,
                "suite_name": suite,
                "task_id": int(task_id),
                "demo_index": item.get("demo_index", item.get("demo_idx")),
                "init_state": state_data,
                "init_state_path": state_path,
            }
        )
    return manifest, normalized, hashlib.sha256(raw).hexdigest()


def _load_init_state(sample: dict[str, Any], manifest_path: Path) -> np.ndarray:
    expected_hash = sample.get("init_state_sha256", sample.get("initial_state_sha256"))
    source = sample.get("init_state")
    source_path: Path | None = None
    if sample.get("init_state_path"):
        source_path = Path(str(sample["init_state_path"]))
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"init-state file missing for {sample['sample_id']}: {source_path}")
        loaded = np.load(source_path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = list(loaded.files)
            preferred = next((key for key in ("init_state", "initial_state", "state", "states") if key in keys), None)
            if preferred is None:
                if len(keys) != 1:
                    raise ValueError(f"cannot identify init-state array in {source_path}; keys={keys}")
                preferred = keys[0]
            source = loaded[preferred]
            loaded.close()
        else:
            source = loaded
        if expected_hash:
            file_hash = _sha256(source_path)
            if str(expected_hash).lower() not in {file_hash, hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest()}:
                raise ValueError(f"init-state hash mismatch for {sample['sample_id']}")
    state = np.asarray(source, dtype=np.float64)
    if state.size == 0 or not np.isfinite(state).all():
        raise ValueError(f"invalid init state for {sample['sample_id']}")
    return state


def _max_steps(suite_name: str) -> int:
    return {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}[suite_name]


def _validate_episode_artifacts(run_dir: Path, records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    resolved_run_dir = run_dir.resolve()
    for record in records:
        for key in ("video_path", "trace_path", "infer_path"):
            relative = Path(str(record.get(key, "")))
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{record.get('sample_id')}: invalid {key}")
                continue
            artifact = (run_dir / relative).resolve()
            # Path.is_relative_to() is only available from Python 3.9; the
            # LIBERO client environment is Python 3.8, so keep this check
            # compatible with both runtimes.
            try:
                artifact.relative_to(resolved_run_dir)
                within_run_dir = True
            except ValueError:
                within_run_dir = False
            if not within_run_dir or not artifact.is_file() or artifact.stat().st_size == 0:
                errors.append(f"{record.get('sample_id')}: missing {key}={relative}")
    return errors


def _run_episode(
    *, sample: dict[str, Any], manifest_path: Path, suite: Any, client: Any,
    run_id: str, run_dir: Path, episode_idx: int, seed: int,
    num_steps_wait: int, replan_steps: int, resize_size: int,
    expected_vlm_layer_pattern: str, expected_action_layer_pattern: str,
    expected_denoise_steps: int,
) -> dict[str, Any]:
    sample_index = int(sample["sample_index"])
    task_id = int(sample["task_id"])
    task = suite.get_task(task_id)
    env_seed = int(sample.get("env_seed", sample.get("seed", seed + sample_index)))
    env, task_description = base._get_libero_env(task, base.LIBERO_ENV_RESOLUTION, env_seed)
    action_plan: collections.deque = collections.deque()
    replay_images: list[np.ndarray] = []
    done = False
    episode_error: str | None = None
    t = 0
    infer_calls = infer_latency_sum_ms = client_roundtrip_sum_ms = policy_time_sum_ms = serve_time_sum_ms = accepted_action_sum = 0.0
    policy_time_count = serve_time_count = draft_rounds = full_rounds = 0
    infer_id_next = executed_steps = 0
    need_reset = True
    episode_trace: list[dict[str, Any]] = []
    episode_infers: list[dict[str, Any]] = []
    episode_idx = sample_index
    max_steps = _max_steps(str(sample["suite_name"]))
    try:
        env.reset()
        obs = env.set_init_state(_load_init_state(sample, manifest_path))
        while t < max_steps + num_steps_wait:
            try:
                if t < num_steps_wait:
                    obs, _, done, _ = env.step(base.LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                image = base.image_tools.convert_to_uint8(base.image_tools.resize_with_pad(image, resize_size, resize_size))
                wrist_image = base.image_tools.convert_to_uint8(base.image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
                replay_images.append(image)
                frame_idx = len(replay_images) - 1
                if not action_plan:
                    infer_id = infer_id_next
                    infer_id_next += 1
                    element = {
                        "observation/image": image,
                        "observation/wrist_image": wrist_image,
                        "observation/state": np.concatenate((obs["robot0_eef_pos"], base._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                        "prompt": str(task_description),
                        "__executed_steps__": int(executed_steps),
                        "__trace_run_id__": run_id,
                        "__trace_task_id__": task_id,
                        "__trace_episode_idx__": sample_index,
                        "__trace_infer_id__": infer_id,
                    }
                    if need_reset:
                        element["__reset_policy_state__"] = True
                        need_reset = False
                    send_time = time.time()
                    start_time = time.perf_counter()
                    output = client.infer(element)
                    roundtrip_ms = (time.perf_counter() - start_time) * 1000.0
                    recv_time = time.time()
                    policy_timing = output.get("policy_timing", {})
                    server_timing = output.get("server_timing", {})
                    if not isinstance(server_timing, dict):
                        server_timing = {}
                    infer_calls += 1
                    client_roundtrip_sum_ms += roundtrip_ms
                    sample_ms = policy_timing.get("sample_actions_ms")
                    if sample_ms is not None:
                        infer_latency_sum_ms += float(sample_ms)
                    policy_ms, serve_ms = base._policy_and_serve_time_ms(server_timing)
                    if policy_ms is not None:
                        policy_time_sum_ms += policy_ms
                        policy_time_count += 1
                    if serve_ms is not None:
                        serve_time_sum_ms += serve_ms
                        serve_time_count += 1
                    route = base._route_type_from_timing(policy_timing)
                    full_rounds += int(route == "full")
                    draft_rounds += int(route != "full")
                    accepted = int(output.get("accepted_prefix_len", replan_steps))
                    if bool(round(float(policy_timing.get("include_in_draft_accept_metrics", 1.0)))):
                        accepted_action_sum += float(accepted)
                    exec_len = min(replan_steps, accepted) if accepted > 0 else 0
                    chunk_actions = np.asarray(output["actions"], dtype=np.float32)
                    infer_record = base._make_infer_record(
                        run_id=run_id, task_suite_name=sample["suite_name"], task_id=task_id,
                        episode_idx=episode_idx, infer_id=infer_id, frame_idx=frame_idx, env_step=t,
                        route_type=route, accepted_prefix_len=accepted, chunk_exec_len=exec_len,
                        policy_timing=policy_timing, server_timing=server_timing,
                        client_send_timestamp_s=send_time, client_recv_timestamp_s=recv_time,
                        client_roundtrip_ms=roundtrip_ms, chunk_actions=chunk_actions,
                    )
                    infer_record.update(sample_index=sample_index, sample_id=sample["sample_id"])
                    reported = infer_record.get("server_experiment", {})
                    expected = {
                        "vlm_layer_pattern": str(expected_vlm_layer_pattern),
                        "action_layer_pattern": str(expected_action_layer_pattern),
                        "denoise_steps": int(expected_denoise_steps),
                    }
                    missing_reported = [key for key in expected if key not in reported]
                    if missing_reported:
                        raise RuntimeError(
                            "server response lacks experiment provenance: "
                            + ", ".join(missing_reported)
                        )
                    mismatched = {
                        key: {"expected": value, "reported": reported.get(key)}
                        for key, value in expected.items()
                        if reported.get(key) != value
                    }
                    if mismatched:
                        raise RuntimeError(f"server experiment configuration mismatch: {mismatched}")
                    episode_infers.append(infer_record)
                    if exec_len <= 0:
                        trace_record = base._make_trace_record(
                            run_id=run_id, task_suite_name=sample["suite_name"], task_id=task_id,
                            episode_idx=episode_idx, frame_idx=frame_idx, env_step=t,
                            infer_record=infer_record, action=None, action_offset_in_chunk=None,
                            infer_start_frame=True, reward=None, done_after_step=False,
                        )
                        trace_record.update(sample_index=sample_index, sample_id=sample["sample_id"])
                        episode_trace.append(trace_record)
                        executed_steps = 0
                        continue
                    for offset in range(exec_len):
                        action_plan.append({"action": chunk_actions[offset].astype(np.float32, copy=True), "infer_record": infer_record, "action_offset_in_chunk": offset, "infer_start_frame": offset == 0})
                    executed_steps = exec_len
                action_meta = action_plan.popleft()
                action = np.asarray(action_meta["action"], dtype=np.float32)
                obs, reward, done, _ = env.step(action.tolist())
                trace_record = base._make_trace_record(
                    run_id=run_id, task_suite_name=sample["suite_name"], task_id=task_id,
                    episode_idx=episode_idx, frame_idx=frame_idx, env_step=t,
                    infer_record=action_meta["infer_record"], action=action,
                    action_offset_in_chunk=action_meta["action_offset_in_chunk"],
                    infer_start_frame=action_meta["infer_start_frame"], reward=float(reward), done_after_step=bool(done),
                )
                trace_record.update(sample_index=sample_index, sample_id=sample["sample_id"])
                trace_record.update(
                    measured_eef_pos=np.asarray(obs['robot0_eef_pos'], dtype=float).tolist(),
                    measured_eef_quat=np.asarray(obs['robot0_eef_quat'], dtype=float).tolist(),
                    measured_gripper_qpos=np.asarray(obs['robot0_gripper_qpos'], dtype=float).tolist(),
                    measured_state_when='post_action',
                    measured_coordinate_frame='MuJoCo world',
                    measured_position_units='m',
                )
                episode_trace.append(trace_record)
                if done:
                    break
                t += 1
            except Exception as exc:  # preserve rollout failures as episode records
                logging.exception("Rollout error on %s", sample["sample_id"])
                episode_error = str(exc)
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    if not replay_images:
        raise RuntimeError(f"no rendered frames for {sample['sample_id']}: {episode_error or 'unknown error'}")
    suffix = "success" if done else "failure"
    episode_dir = base._episode_output_dir(
        run_dir=run_dir, task_description=task_description, suffix=suffix, task_id=task_id, episode_idx=episode_idx,
    )
    video_path = episode_dir / f"rollout_{suffix}.mp4"
    imageio.mimwrite(video_path, [np.asarray(frame) for frame in replay_images], fps=10)
    trace_path, infer_path = episode_dir / "trace.jsonl", episode_dir / "infer.jsonl"
    base._write_jsonl(trace_path, episode_trace)
    base._write_jsonl(infer_path, episode_infers)
    executed_action_count = sum(record.get("executed_action") is not None for record in episode_trace)
    return {
        "run_id": run_id, "sample_index": sample_index, "sample_id": sample["sample_id"],
        "suite_name": sample["suite_name"], "task_suite_name": sample["suite_name"],
        "task_id": task_id, "demo_index": sample.get("demo_index"),
        "init_state_sha256": sample.get("init_state_sha256", sample.get("initial_state_sha256")),
        "expert_trajectory_length": sample.get("trajectory_length", sample.get("expert_trajectory_length")),
        "task_description": str(task_description), "episode_idx": episode_idx, "success": bool(done),
        "failure_reason": episode_error, "max_steps": max_steps, "num_steps_wait": num_steps_wait,
        "env_steps_taken": int(t), "trajectory_length": int(executed_action_count),
        "executed_action_count": int(executed_action_count), "infer_calls": int(infer_calls),
        "draft_rounds": int(draft_rounds), "full_rounds": int(full_rounds),
        "infer_latency_mean_ms": infer_latency_sum_ms / infer_calls if infer_calls else None,
        "client_roundtrip_mean_ms": client_roundtrip_sum_ms / infer_calls if infer_calls else None,
        "policy_time_mean_ms": policy_time_sum_ms / policy_time_count if policy_time_count else None,
        "serve_time_mean_ms": serve_time_sum_ms / serve_time_count if serve_time_count else None,
        "avg_policy_time_per_action_ms": policy_time_sum_ms / executed_action_count if executed_action_count else None,
        "video_path": str(video_path.relative_to(run_dir)), "trace_path": str(trace_path.relative_to(run_dir)),
        "infer_path": str(infer_path.relative_to(run_dir)), "frame_count": len(replay_images),
        "trace_record_count": len(episode_trace), "infer_record_count": len(episode_infers),
        "seed": env_seed, "noise_episode_idx": sample_index,
    }


def evaluate(args: argparse.Namespace) -> int:
    testset_path = args.testset_manifest.expanduser().resolve()
    testset, samples, testset_hash = _read_testset(testset_path)
    if args.sample_indices:
        requested = [int(value) for value in args.sample_indices.split(",") if value.strip()]
        by_index = {int(sample["sample_index"]): sample for sample in samples}
        missing = set(requested) - set(by_index)
        if missing:
            raise ValueError(f"sample indices are absent from testset: {sorted(missing)}")
        samples = [by_index[index] for index in requested]
    elif args.sample_limit is not None:
        samples = samples[: args.sample_limit]
    run_root = args.video_out_path.expanduser().resolve()
    run_dir = run_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = run_dir / "manifest.json"
    episode_log_path = run_dir / "episode_log.json"
    existing: list[dict[str, Any]] = []
    if run_manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"run already exists; pass --resume to continue: {run_dir}")
        old_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("testset_sha256") != testset_hash:
            raise ValueError("cannot resume: testset manifest changed")
        existing = json.loads(episode_log_path.read_text(encoding="utf-8")) if episode_log_path.is_file() else []
        errors = _validate_episode_artifacts(run_dir, existing)
        if errors:
            raise ValueError("cannot resume due to invalid completed artifacts: " + "; ".join(errors[:5]))
    existing_ids = {str(record["sample_id"]) for record in existing}
    manifest_payload = {
        "run_id": args.run_name, "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running", "testset_manifest": str(testset_path), "testset_sha256": testset_hash,
        "sample_count": len(samples), "samples": [{"sample_index": s["sample_index"], "sample_id": s["sample_id"], "suite_name": s["suite_name"], "task_id": s["task_id"]} for s in samples],
        "condition_id": args.condition_id, "vlm_layer_pattern": args.vlm_layer_pattern,
        "action_layer_pattern": args.action_layer_pattern, "denoise_steps": args.denoise_steps,
        "global_noise_seed": args.global_noise_seed,
        "seed": args.seed, "num_steps_wait": args.num_steps_wait, "replan_steps": args.replan_steps,
        "resize_size": args.resize_size,
    }
    _atomic_json(run_manifest_path, manifest_payload)
    base._write_json(episode_log_path, existing)
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_cache: dict[str, Any] = {}
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    try:
        for sample in samples:
            if sample["sample_id"] in existing_ids:
                continue
            suite_name = sample["suite_name"]
            if suite_name not in suite_cache:
                if suite_name not in benchmark_dict:
                    raise ValueError(f"unknown LIBERO suite: {suite_name}")
                suite_cache[suite_name] = benchmark_dict[suite_name]()
            record = _run_episode(
                sample=sample, manifest_path=testset_path, suite=suite_cache[suite_name], client=client,
                run_id=args.run_name, run_dir=run_dir, episode_idx=sample["sample_index"], seed=args.seed,
                num_steps_wait=args.num_steps_wait, replan_steps=args.replan_steps, resize_size=args.resize_size,
                expected_vlm_layer_pattern=args.vlm_layer_pattern,
                expected_action_layer_pattern=args.action_layer_pattern,
                expected_denoise_steps=args.denoise_steps,
            )
            existing.append(record)
            existing_ids.add(str(record["sample_id"]))
            existing.sort(key=lambda row: int(row["sample_index"]))
            base._write_json(episode_log_path, existing)
            manifest_payload["completed_sample_count"] = len(existing)
            _atomic_json(run_manifest_path, manifest_payload)
            logging.info("Completed %s (%d/%d): success=%s", record["sample_id"], len(existing), len(samples), record["success"])
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    missing = {sample["sample_id"] for sample in samples} - existing_ids
    if missing:
        manifest_payload["status"] = "incomplete"
        manifest_payload["missing_samples"] = sorted(missing)
        _atomic_json(run_manifest_path, manifest_payload)
        return 1
    errors = _validate_episode_artifacts(run_dir, existing)
    if errors:
        manifest_payload["status"] = "incomplete"
        manifest_payload["artifact_errors"] = errors
        _atomic_json(run_manifest_path, manifest_payload)
        return 1
    manifest_payload["status"] = "completed"
    manifest_payload["completed_sample_count"] = len(existing)
    _atomic_json(run_manifest_path, manifest_payload)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--video-out-path", type=Path, default=REPO_ROOT / "data/test")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--vlm-layer-pattern", required=True)
    parser.add_argument("--action-layer-pattern", default="full18")
    parser.add_argument("--denoise-steps", type=int, choices=(3, 5, 7, 10), default=3)
    parser.add_argument("--global-noise-seed", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=12)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--sample-indices", help="Comma-separated manifest indices for a paired smoke subset.")
    parser.add_argument("--resume", action="store_true")
    return evaluate(parser.parse_args())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
