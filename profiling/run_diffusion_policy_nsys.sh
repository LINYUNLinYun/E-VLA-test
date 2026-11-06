#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="$root_dir/profile_results/diffusion-policy"
mkdir -p "$result_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=nvtx \
  --nvtx-capture='profile_window@*' \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$result_dir/diffusion-policy" \
  "$root_dir/policies/diffusion_policy/.venv/bin/python" \
  "$root_dir/profiling/diffusion_policy_profile.py" \
  > >(tee "$result_dir/run.log") 2>&1
