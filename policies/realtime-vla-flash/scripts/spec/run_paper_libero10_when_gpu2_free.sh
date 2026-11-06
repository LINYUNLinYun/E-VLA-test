#!/usr/bin/env bash
set -euo pipefail

repo=/root/nfs/embodied/benchmark/policies/realtime-vla-flash
cd "$repo"

run_name=pi0_paper_suite_libero_10_50x_d10_20261003
log_dir="$repo/data/test/$run_name"
mkdir -p "$log_dir"

while true; do
  read -r used_mb util_pct < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$1 + 0 == 2 { gsub(/ /, "", $2); gsub(/ /, "", $3); print $2, $3 }')
  printf '%s GPU 2 check: %s MiB used, %s%% utilization\n' "$(date -Is)" "$used_mb" "$util_pct" | tee -a "$log_dir/gpu_wait.log"
  if (( used_mb <= 1536 && util_pct <= 10 )); then
    break
  fi
  sleep 300
done

export OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi
export PYTHONPATH=.:/root/nfs/embodied/benchmark/LIBERO

CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/spec/spec_serve_policy.py \
  --config pi0_libero \
  --base-triton-path /root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted/base \
  --draft-triton-path /root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted/draft_10 \
  --task-suite-name libero_10 \
  --backend triton \
  --max-exec-steps 12 \
  --force-full-each-round \
  --vlm-layer-pattern vlm_full18 \
  --action-layer-pattern full18 \
  --denoise-steps 10 \
  --port 8743 >"$log_dir/server.log" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 1800); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 100 "$log_dir/server.log"
    exit 1
  fi
  if timeout 1 bash -c '</dev/tcp/127.0.0.1/8743' 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if (( ready == 0 )); then
  printf 'Server did not become ready before timeout.\n' | tee -a "$log_dir/server.log"
  exit 1
fi

LIBERO_CONFIG_PATH="$repo/.libero" \
PYTHONPATH=/root/nfs/embodied/benchmark/LIBERO \
MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=2 \
examples/libero/.venv/bin/python scripts/spec/spec_client_libero.py \
  --host 127.0.0.1 \
  --port 8743 \
  --task-suite-name libero_10 \
  --num-trials-per-task 50 \
  --num-steps-wait 10 \
  --replan-steps 12 \
  --seed 7 \
  --video-out-path data/test \
  --run-name "$run_name" 2>&1 | tee "$log_dir/client.log"
