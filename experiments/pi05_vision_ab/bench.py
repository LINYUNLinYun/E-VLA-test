"""Synchronized vision and full-policy latency measurement (one static config)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

SCOPES = ("vision_all", "vision_single_view", "model", "policy_wall")


def _fixed_order(n, seed=20261020):
    return np.random.default_rng(seed).permutation(n)


def _bench_noise(obs_row, index):
    from experiments.pi05_vision_ab.evaluation_adapter import noise_seed

    rng = np.random.default_rng(noise_seed(obs_row["suite"], obs_row["task_id"],
                                           obs_row["initial_state_index"], index,
                                           stream_tag="bench"))
    return rng.standard_normal((10, 32), dtype=np.float32)


def measure_condition(condition, round_index, observations, output_csv, gpu, run_id,
                      warmup=30, calls=200):
    """Run inside the model environment for exactly one condition."""
    import jax
    import jax.numpy as jnp

    from experiments.pi05_vision_ab.policy_adapter import Pi05VisionPolicy

    policy = Pi05VisionPolicy(condition["mode"], int(condition["keep_tokens"]),
                              condition["id"])
    single, all_slots = policy.make_vision_fns()

    prepared = []
    observations_gpu = []
    raw_policy_inputs = []
    noises = []
    for row in observations:
        _, slots = policy._prepared_inputs(row["raw"])
        prepared.append([jnp.asarray(s) for s in slots])
        observation = policy.make_observation(row["raw"])
        observations_gpu.append(observation)
        raw_policy_inputs.append({
            "observation/image": np.asarray(row["raw"]["agentview_image"]),
            "observation/wrist_image": np.asarray(row["raw"]["wrist_image"]),
            "observation/state": np.asarray(row["raw"]["state"], dtype=np.float32),
            "prompt": row["raw"]["instruction"],
        })
        noises.append(_bench_noise(row, len(noises)))

    # Compilation, measured once per static configuration.
    tick = time.perf_counter()
    jax.block_until_ready(all_slots(prepared[0]))
    compile_vision_s = time.perf_counter() - tick
    tick = time.perf_counter()
    jax.block_until_ready(policy.sample_actions(observations_gpu[0], noises[0]))
    compile_model_s = time.perf_counter() - tick
    tick = time.perf_counter()
    policy.policy.infer(raw_policy_inputs[0], noise=noises[0])
    compile_policy_s = time.perf_counter() - tick

    n = len(prepared)
    for i in range(warmup):
        jax.block_until_ready(all_slots(prepared[i % n]))
        jax.block_until_ready(policy.sample_actions(observations_gpu[i % n], noises[i % n]))
    # Short re-warmup at the start of each measured round.
    for i in range(5):
        jax.block_until_ready(all_slots(prepared[i]))
        jax.block_until_ready(policy.sample_actions(observations_gpu[i], noises[i]))

    order = _fixed_order(n)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "config_id", "backend", "compile_mode", "gpu", "round",
              "iteration", "observation_id", "measurement_scope", "wall_ms",
              "optional_device_ms", "actual_camera_slots", "valid_camera_count"]
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for iteration in range(calls):
            idx = int(order[iteration % n])
            row = observations[idx]
            base = {
                "run_id": run_id,
                "config_id": condition["id"],
                "backend": "jax",
                "compile_mode": "jax_jit",
                "gpu": str(gpu),
                "round": round_index,
                "iteration": iteration,
                "observation_id": row["observation_id"],
                "optional_device_ms": "",
                "actual_camera_slots": 3,
                "valid_camera_count": 2,
            }

            tick = time.perf_counter_ns()
            out = all_slots(prepared[idx])
            jax.block_until_ready(out)
            writer.writerow({**base, "measurement_scope": "vision_all",
                             "wall_ms": (time.perf_counter_ns() - tick) / 1e6})

            tick = time.perf_counter_ns()
            out = single(prepared[idx][0])
            jax.block_until_ready(out)
            writer.writerow({**base, "measurement_scope": "vision_single_view",
                             "wall_ms": (time.perf_counter_ns() - tick) / 1e6})

            tick = time.perf_counter_ns()
            out = policy.sample_actions(observations_gpu[idx], noises[idx])
            jax.block_until_ready(out)
            writer.writerow({**base, "measurement_scope": "model",
                             "wall_ms": (time.perf_counter_ns() - tick) / 1e6})

            tick = time.perf_counter_ns()
            policy.policy.infer(raw_policy_inputs[idx], noise=noises[idx])
            writer.writerow({**base, "measurement_scope": "policy_wall",
                             "wall_ms": (time.perf_counter_ns() - tick) / 1e6})

    meta = {
        "config_id": condition["id"],
        "round": round_index,
        "gpu": gpu,
        "backend": "jax",
        "compile_mode": "jax_jit",
        "dtype": "bfloat16",
        "warmup": warmup,
        "calls": calls,
        "observations": n,
        "compile_vision_s": compile_vision_s,
        "compile_model_s": compile_model_s,
        "compile_policy_infer_s": compile_policy_s,
        "load_s": policy.load_seconds,
        "protocol": policy.protocol,
        "input_residency": "gpu",
        "synchronized_wall_time": True,
    }
    Path(str(output_csv) + ".meta.json").write_text(json.dumps(meta, indent=2))
    return meta
