#!/usr/bin/env bash
set -euo pipefail

benchmark_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${benchmark_root}/SimplerEnv"

timeout 120 .venv/bin/python - <<'PY'
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

env = simpler_env.make("google_robot_pick_coke_can")
obs, _ = env.reset(seed=0)
image = get_image_from_maniskill2_obs_dict(env, obs)
obs, reward, done, truncated, _ = env.step(env.action_space.sample())

assert image.shape == (512, 640, 3)
print("SIMPLER_ENV_SMOKE_OK", env.get_language_instruction(), reward, done, truncated)
env.close()
PY
