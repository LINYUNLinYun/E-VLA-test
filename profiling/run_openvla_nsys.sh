#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="$root_dir/profile_results/openvla-oft"
mkdir -p "$result_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$root_dir/shared/caches/huggingface"
export HF_HUB_DISABLE_XET=1
export LIBERO_CONFIG_PATH="$root_dir/policies/openvla-oft/.libero"
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
export MUJOCO_GL=egl

nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=nvtx \
  --nvtx-capture='profile_window@*' \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$result_dir/openvla-oft" \
  "$root_dir/policies/openvla-oft/.venv/bin/python" \
  "$root_dir/profiling/openvla_oft_profile.py" \
  > >(tee "$result_dir/run.log") 2>&1
