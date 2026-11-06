#!/usr/bin/env bash
set -euo pipefail

benchmark_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${benchmark_root}/calvin/calvin_models"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EGL_VISIBLE_DEVICES="${EGL_VISIBLE_DEVICES:-0}"

timeout 120 ../.venv/bin/python - <<'PY'
from pathlib import Path

import numpy as np
from calvin_env.envs.play_table_env import get_env

dataset = Path("../dataset/calvin_debug_dataset/validation").resolve()
observation_space = {
    "rgb_obs": ["rgb_static", "rgb_gripper"],
    "depth_obs": [],
}
env = get_env(dataset, obs_space=observation_space, show_gui=False)
obs = env.reset()
obs, reward, done, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))

assert obs["rgb_obs"]["rgb_static"].shape == (200, 200, 3)
assert obs["rgb_obs"]["rgb_gripper"].shape == (84, 84, 3)
assert obs["robot_obs"].shape == (15,)
print("CALVIN_SMOKE_OK", reward, done)
PY
