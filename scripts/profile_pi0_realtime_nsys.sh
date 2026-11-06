#!/usr/bin/env bash
# Profile steady-state Triton-pi0 or Realtime-VLA/FLASH server inference.
#
# Usage:
#   bash scripts/profile_pi0_realtime_nsys.sh pi0
#   bash scripts/profile_pi0_realtime_nsys.sh realtime
#
# The server starts under an inactive Nsight Systems session. One LIBERO
# episode warms up model/runtime/CUDA graphs, then collection starts for one
# additional episode. Only the server process is profiled.
set -euo pipefail

MODE=${1:-}
if [[ "$MODE" != "pi0" && "$MODE" != "realtime" ]]; then
  echo "usage: $0 {pi0|realtime}" >&2
  exit 2
fi

BENCH_ROOT=/root/nfs/embodied/benchmark
REPO=$BENCH_ROOT/policies/realtime-vla-flash
CHECKPOINT_ROOT=$BENCH_ROOT/shared/checkpoints/realtime-vla-flash
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-8200}
TASK_SUITE=${TASK_SUITE:-libero_goal}
TASK_ID=${TASK_ID:-0}
PROFILE_ROOT=${PROFILE_ROOT:-$BENCH_ROOT/profiles/nsys}
CUDA_GRAPH_TRACE=${CUDA_GRAPH_TRACE:-node}
READY_TIMEOUT=${READY_TIMEOUT:-2400}
STAMP=$(date +%Y%m%d_%H%M%S)
SESSION="openpi_${MODE}_${STAMP}"
OUTPUT_BASE="$PROFILE_ROOT/${MODE}_${TASK_SUITE}_task${TASK_ID}_${STAMP}"
SERVER_LOG="${OUTPUT_BASE}_server.log"
CLIENT_ROOT="/tmp/openpi_nsys_${MODE}_${STAMP}"

mkdir -p "$PROFILE_ROOT" "$CLIENT_ROOT"

used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
if [[ ! "$used_mib" =~ ^[0-9]+$ ]]; then
  echo "unable to determine memory use for GPU $GPU_ID: $used_mib" >&2
  exit 1
fi
if (( used_mib > 1024 )); then
  echo "refusing to profile: GPU $GPU_ID already uses ${used_mib} MiB" >&2
  exit 1
fi

server_extra=()
if [[ "$MODE" == "pi0" ]]; then
  server_extra+=(--force-full-each-round)
else
  server_extra+=(--tau-radius 0.15 --periodic-full-every-n-draft-rounds 2)
fi

cd "$REPO"
profiler_pid=""
cleanup() {
  if [[ -n "$profiler_pid" ]] && kill -0 "$profiler_pid" 2>/dev/null; then
    kill -TERM -- "-$profiler_pid" 2>/dev/null || true
    wait "$profiler_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

setsid env \
  OPENPI_NVTX=1 \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  OPENPI_DATA_HOME="$BENCH_ROOT/shared/checkpoints/openpi" \
  nsys profile \
    --start-later=true \
    --session-new="$SESSION" \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none \
    --cpuctxsw=none \
    --cuda-graph-trace="$CUDA_GRAPH_TRACE" \
    --export=sqlite \
    --force-overwrite=true \
    --output="$OUTPUT_BASE" \
    .venv/bin/python scripts/spec/spec_serve_policy.py \
      --config pi0_libero --backend triton \
      --base-triton-path "$CHECKPOINT_ROOT/converted/base" \
      --draft-triton-path "$CHECKPOINT_ROOT/converted/draft" \
      --task-suite-name "$TASK_SUITE" \
      --tokenizer-source local --pytorch-device cuda:0 \
      --max-exec-steps 12 --port "$PORT" \
      "${server_extra[@]}" >"$SERVER_LOG" 2>&1 &
profiler_pid=$!

waited=0
while ! curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null; do
  if ! kill -0 "$profiler_pid" 2>/dev/null; then
    echo "profiled server exited before readiness" >&2
    tail -80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( waited >= READY_TIMEOUT )); then
    echo "profiled server did not become ready in ${READY_TIMEOUT}s" >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done

run_client() {
  local run_name=$1
  LIBERO_CONFIG_PATH="$REPO/.libero" \
  PYTHONPATH="$REPO/third_party/libero" \
  XDG_CACHE_HOME="$BENCH_ROOT/shared/caches/xdg" \
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$GPU_ID" \
    examples/libero/.venv/bin/python scripts/spec/spec_client_libero.py \
      --task-suite-name "$TASK_SUITE" --host 127.0.0.1 --port "$PORT" \
      --num-trials-per-task 1 --task "$TASK_ID" --replan-steps 12 \
      --video-out-path "$CLIENT_ROOT" --run-name "$run_name"
}

echo "[warmup] mode=$MODE suite=$TASK_SUITE task=$TASK_ID GPU=$GPU_ID"
run_client warmup >"${OUTPUT_BASE}_warmup_client.log" 2>&1

echo "[profile] starting Nsight Systems session $SESSION"
nsys start --session="$SESSION"
run_client profiled >"${OUTPUT_BASE}_profiled_client.log" 2>&1
nsys stop --session="$SESSION"

cleanup
trap - EXIT INT TERM

echo "[done] ${OUTPUT_BASE}.nsys-rep"
echo "[done] ${OUTPUT_BASE}.sqlite"
echo "NVTX enabled with OPENPI_NVTX=1; normal runs remain uninstrumented."
