"""LIBERO episode manifests, paired action noise, and resume-safe rollout logs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_INDEX = {name: i for i, name in enumerate(SUITES)}
# Official evaluator max steps (libero_eval_v21 / openpi examples).
LIMITS = {"libero_spatial": 280, "libero_object": 220, "libero_goal": 300, "libero_10": 520}
NUM_STEPS_WAIT = 10
REPLAN_STEPS = 5
ACTION_HORIZON = 10
ACTION_DIM = 32
ENVIRONMENT_SEED = 7
REPEAT_SEED = 0


def stable_digest(parts) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def noise_seed(suite, task_id, initial_state_index, replan_index, root_seed=7,
               repeat_seed=REPEAT_SEED, stream_tag="policy"):
    """Explicit action-noise key; deliberately excludes config_id."""
    parts = ["pi05_vision_ab", int(root_seed), suite, int(task_id), int(initial_state_index),
             int(repeat_seed), int(replan_index), stream_tag]
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little")


def make_noise(suite, task_id, initial_state_index, replan_index, root_seed=7,
               repeat_seed=REPEAT_SEED):
    rng = np.random.default_rng(noise_seed(suite, task_id, initial_state_index, replan_index,
                                           root_seed, repeat_seed))
    return rng.standard_normal((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)


def episode_key(condition_id, stage, suite, task_id, initial_state_index, repeat_seed):
    return stable_digest([condition_id, stage, suite, int(task_id),
                          int(initial_state_index), int(repeat_seed)])


_SUITE_CACHE = {}
_INIT_CACHE = {}


def init_states(suite_name, task_id):
    from libero.libero import benchmark

    if suite_name not in _SUITE_CACHE:
        _SUITE_CACHE[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
    suite = _SUITE_CACHE[suite_name]
    key = (suite_name, int(task_id))
    if key not in _INIT_CACHE:
        from preparations.idea1_v2_1.runtime_client import official_init_states

        _INIT_CACHE[key] = official_init_states(suite, task_id)
    return suite, _INIT_CACHE[key]


def build_manifest(stage, conditions, screen=(0, 5), confirm=(5, 50), smoke_tasks=1,
                   replicate=1):
    rows = []
    if stage == "smoke":
        tasks = [(suite, task) for suite in SUITES for task in range(smoke_tasks)]
        inits = [0]
    elif stage == "screen":
        tasks = [(suite, task) for suite in SUITES for task in range(10)]
        inits = list(range(*screen))
    elif stage == "confirm":
        tasks = [(suite, task) for suite in SUITES for task in range(10)]
        inits = list(range(*confirm))
    else:
        raise ValueError(stage)
    for condition in conditions:
        for repeat in range(replicate):
            for suite, task in tasks:
                for init in inits:
                    rows.append({
                        "condition_id": condition["id"],
                        "config": {"condition_id": condition["id"], **condition},
                        "stage": stage,
                        "suite": suite,
                        "task_id": task,
                        "initial_state_index": init,
                        "repeat_seed": REPEAT_SEED + repeat,
                        "environment_seed": ENVIRONMENT_SEED,
                        "action_noise_seed_rule":
                            "sha256(pi05_vision_ab,root_seed,suite,task,init,repeat,replan,policy)",
                        "key": episode_key(condition["id"], stage, suite, task, init,
                                           REPEAT_SEED + repeat),
                        "task_rank": SUITE_INDEX[suite] * 10 + task,
                    })
    return rows


def load_done_keys(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("key") and record.get("error_type") != "infrastructure":
                done.add(record["key"])
    return done


def append_jsonl(path: Path, record: dict, lock_path: Path | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = lock_path or path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(lock, fcntl.LOCK_UN)


def run_episode(row, port, root_seed=7):
    """One paired LIBERO episode. Returns a JSON-serializable record."""
    import imageio.v2 as imageio  # noqa: F401
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from preparations.idea1_v2_1.runtime_client import rpc

    from experiments.pi05_vision_ab.policy_adapter import _native_observation

    suite_name = row["suite"]
    task_id = row["task_id"]
    init_index = row["initial_state_index"]
    suite, initial_states = init_states(suite_name, task_id)
    if init_index >= len(initial_states):
        raise ValueError(f"official init index {init_index} unavailable for {suite_name}/{task_id}")
    definition = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / definition.problem_folder / definition.bddl_file

    started = time.perf_counter()
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    policy_calls = 0
    steps = 0
    done = False
    policy_wall_ms = []
    try:
        env.seed(ENVIRONMENT_SEED)
        env.reset()
        obs = env.set_init_state(initial_states[init_index])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        rpc(port, {"op": "reset"})
        protocol = rpc(port, {"op": "health"})["protocol"]
        limit = LIMITS[suite_name]
        while steps < limit and not done:
            replan = policy_calls
            noise = make_noise(suite_name, task_id, init_index, replan, root_seed,
                               row["repeat_seed"])
            model_obs = {k: np.asarray(v) for k, v in obs.items() if isinstance(v, np.ndarray)}
            response = rpc(port, {
                "op": "predict",
                "obs": model_obs,
                "instruction": definition.language,
                "suite": suite_name,
                "noise": noise,
            })
            actions = np.asarray(response["actions"], dtype=np.float32)
            length = min(int(response["execute_length"]), limit - steps)
            if actions.shape != (ACTION_HORIZON, 7) or not np.isfinite(actions).all():
                raise ValueError(f"invalid proposal {actions.shape}")
            policy_calls += 1
            policy_wall_ms.append(float(response.get("policy_wall_ms", 0.0)))
            executed = 0
            for action in actions[:length]:
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                executed += 1
                if done:
                    break
            if executed == 0:
                raise RuntimeError("environment executed zero actions")
        success = bool(done)
        timeout = bool(not done and steps >= LIMITS[suite_name])
    finally:
        env.close()

    images, state = _native_observation(obs)
    first = np.asarray(initial_states[init_index], dtype=np.float32)
    return {
        "key": row["key"],
        "stage": row["stage"],
        "condition_id": row["condition_id"],
        "suite": suite_name,
        "task_id": task_id,
        "initial_state_index": init_index,
        "environment_seed": ENVIRONMENT_SEED,
        "repeat_seed": row["repeat_seed"],
        "root_seed": root_seed,
        "action_noise_seed_rule": row["action_noise_seed_rule"],
        "success": success,
        "environment_steps": steps,
        "policy_calls": policy_calls,
        "timeout": timeout,
        "error_type": None,
        "wall_s": time.perf_counter() - started,
        "policy_wall_ms_sum": float(sum(policy_wall_ms)),
        "policy_wall_ms_mean": float(np.mean(policy_wall_ms)) if policy_wall_ms else None,
        "initial_state_hash": stable_digest(np.round(first, 7).tolist()),
        "protocol": protocol,
        "video_path": None,
    }
