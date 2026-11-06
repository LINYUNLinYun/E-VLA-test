#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$root_dir/policies/diffusion_policy"
wheel_dir="$root_dir/shared/caches/wheels"

if [[ ! -x "$repo_dir/.venv/bin/python" ]]; then
  uv venv --python 3.9 --prompt diffusion-policy "$repo_dir/.venv"
fi

uv pip install --python "$repo_dir/.venv/bin/python" \
  "$wheel_dir/torch-1.12.1+cu116-cp39-cp39-linux_x86_64.whl" \
  "$wheel_dir/torchvision-0.13.1+cu116-cp39-cp39-linux_x86_64.whl"
uv pip install --python "$repo_dir/.venv/bin/python" \
  -r "$root_dir/profiling/diffusion_policy_inference_requirements.txt"
uv pip install --python "$repo_dir/.venv/bin/python" --no-deps -e "$repo_dir"

"$repo_dir/.venv/bin/python" -c \
  'import torch, diffusion_policy; print(torch.__version__, torch.cuda.is_available())'
