#!/usr/bin/env bash
set -uo pipefail

BENCH_ROOT=/root/nfs/embodied/benchmark
REPO_ROOT="$BENCH_ROOT/policies/realtime-vla-flash"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
RESUME_SCRIPT="$REPO_ROOT/scripts/experiment3/resume_skip_layers_experiment.py"
MANIFEST="$REPO_ROOT/results/pi0_vlm_action_denoise_ablation/manifest.json"
CHECK_INTERVAL_SECONDS=300
MAX_USED_MIB=1024
PORT=8520

timestamp() {
  date '+%F %T %Z'
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

exec 9>"/tmp/pi0_vlm_ablation_monitor.lock"
if ! flock -n 9; then
  echo "[$(timestamp)] another monitor already holds the lock"
  exit 0
fi

echo "[$(timestamp)] monitor started; checking GPU0/1 every ${CHECK_INTERVAL_SECONDS}s"
while true; do
  if pgrep -f '[r]esume_skip_layers_experiment.py' >/dev/null; then
    echo "[$(timestamp)] resume runner is already active; monitor exiting"
    exit 0
  fi

  chosen_gpu=""
  chosen_used=""
  for gpu in 0 1; do
    used=$(gpu_used_mib "$gpu" || true)
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
      if [[ -z "$chosen_gpu" ]] || (( used < chosen_used )); then
        chosen_gpu=$gpu
        chosen_used=$used
      fi
    fi
  done

  if [[ -z "$chosen_gpu" ]]; then
    used0=$(gpu_used_mib 0 || true)
    used1=$(gpu_used_mib 1 || true)
    echo "[$(timestamp)] waiting: GPU0=${used0:-unknown}MiB GPU1=${used1:-unknown}MiB"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  echo "[$(timestamp)] GPU${chosen_gpu} available (${chosen_used}MiB); resuming experiment"
  "$PYTHON_BIN" -u "$RESUME_SCRIPT" \
    --manifest "$MANIFEST" \
    --server-gpu "$chosen_gpu" \
    --client-gpu "$chosen_gpu" \
    --port "$PORT" \
    --max-gpu-memory-mib "$MAX_USED_MIB"
  status=$?
  if (( status == 0 )); then
    echo "[$(timestamp)] experiment and summary completed"
    exit 0
  fi
  if (( status == 1 )); then
    echo "[$(timestamp)] resume stopped after repeated deterministic failures; manual inspection required"
    exit 1
  fi
  echo "[$(timestamp)] resume exited with status $status; returning to GPU wait loop"
  sleep "$CHECK_INTERVAL_SECONDS"
done
