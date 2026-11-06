#!/usr/bin/env bash
# Run the official OpenPI pi0 ALOHA-Sim TransferCube checkpoint on one exclusive GPU.
set -euo pipefail

ROOT=/root/nfs/embodied/benchmark
REPO=$ROOT/policies/realtime-vla-flash
GPU=${GPU:-0}
PORT=${PORT:-8200}
TRIALS=${TRIALS:-50}
SEED=${SEED:-0}
ACTION_HORIZON=${ACTION_HORIZON:-10}
SAVE_VIDEO=${SAVE_VIDEO:-true}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
MAX_START_MEMORY_MIB=${MAX_START_MEMORY_MIB:-1000}
RUN_NAME=${RUN_NAME:-pi0_transfer_cube_seed${SEED}_${TRIALS}ep}
OUT_DIR=${OUT_DIR:-$REPO/data/aloha_sim/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$ROOT/logs/aloha_${RUN_NAME}}

mkdir -p "$LOG_DIR" "$OUT_DIR"

gpu_memory_used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if (( gpu_memory_used > MAX_START_MEMORY_MIB )); then
  echo "[fatal] GPU$GPU is not exclusive: ${gpu_memory_used} MiB already in use" >&2
  exit 2
fi

cd "$REPO"
server_pid=""
cleanup() {
  if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[config] GPU$GPU port=$PORT trials=$TRIALS seed=$SEED horizon=$ACTION_HORIZON"
echo "[output] $OUT_DIR"

case "${SAVE_VIDEO,,}" in
  true|1|yes) video_arg=--save-video ;;
  false|0|no) video_arg=--no-save-video ;;
  *) echo "[fatal] SAVE_VIDEO must be true or false, got: $SAVE_VIDEO" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES="$GPU" \
OPENPI_DATA_HOME="$ROOT/shared/checkpoints/openpi" \
XDG_CACHE_HOME="$ROOT/shared/caches/xdg" \
.venv/bin/python scripts/serve_policy.py \
  --env aloha_sim --port "$PORT" \
  >"$LOG_DIR/server.log" 2>&1 &
server_pid=$!

waited=0
while (( waited < READY_TIMEOUT )); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    echo "[ready] policy server after ${waited}s"
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "[fatal] policy server exited before readiness" >&2
    tail -80 "$LOG_DIR/server.log" >&2 || true
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done
if (( waited >= READY_TIMEOUT )); then
  echo "[fatal] policy server did not become ready within ${READY_TIMEOUT}s" >&2
  tail -80 "$LOG_DIR/server.log" >&2 || true
  exit 1
fi

MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES="$GPU" \
XDG_CACHE_HOME="$ROOT/shared/caches/xdg" \
.venv/bin/python "$ROOT/scripts/eval_aloha_sim.py" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --num-trials "$TRIALS" \
  --seed "$SEED" \
  --action-horizon "$ACTION_HORIZON" \
  --out-dir "$OUT_DIR" \
  "$video_arg" \
  2>&1 | tee "$LOG_DIR/client.log"

.venv/bin/python - "$OUT_DIR" "$TRIALS" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
rows = json.loads((out_dir / "episode_log.json").read_text())
assert len(rows) == expected, (len(rows), expected)
assert all(row["trajectory_length"] >= 0 for row in rows)
assert all(row["trajectory_length"] <= 300 for row in rows)
assert all(row["failure_reason"] is None for row in rows), [
    row for row in rows if row["failure_reason"] is not None
]
print(f"[verified] {len(rows)} episodes; trajectory length present for every episode")
PY

echo "[done] summary=$OUT_DIR/summary.json"
