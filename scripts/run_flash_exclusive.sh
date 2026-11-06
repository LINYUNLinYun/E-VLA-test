#!/usr/bin/env bash
# Run all four LIBERO suites concurrently, with one suite exclusively assigned
# to each physical GPU (one policy server + its EGL client per GPU).
set -euo pipefail

ROOT=/root/nfs/embodied/benchmark
REPO=$ROOT/policies/realtime-vla-flash
CK=$ROOT/shared/checkpoints/realtime-vla-flash
TAG=${TAG:-exclusive_full}
TRIALS=${TRIALS:-50}
TAU_RADIUS=${TAU_RADIUS:-0.15}
PF_EVERY=${PF_EVERY:-2}
MAX_EXEC_STEPS=${MAX_EXEC_STEPS:-12}
REPLAN_STEPS=${REPLAN_STEPS:-12}
TASK_SPEC=${TASK_SPEC:-0-9}
READY_TIMEOUT=${READY_TIMEOUT:-2400}
LOGDIR=${LOGDIR:-$ROOT/logs/flash_${TAG}_$(date +%m%d_%H%M)}

SUITES=(libero_spatial libero_object libero_goal libero_10)
GPUS=(0 1 2 3)
PORTS=(8100 8101 8102 8103)
declare -A DRAFT=(
  [libero_spatial]=draft_spatial
  [libero_object]=draft_object
  [libero_goal]=draft
  [libero_10]=draft_10
)

mkdir -p "$LOGDIR"
cd "$REPO"

declare -A SERVER_PIDS=()
declare -A CLIENT_PIDS=()

cleanup() {
  local suite pid
  for suite in "${SUITES[@]}"; do
    pid=${CLIENT_PIDS[$suite]:-}
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
    pid=${SERVER_PIDS[$suite]:-}
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

count_done() {
  local suite=$1
  local path="$REPO/data/test/${TAG}_${suite}/episode_log.json"
  if [[ -f $path ]]; then
    python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$path" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

wait_ready() {
  local suite=$1 port=$2 pid=$3 waited=0
  while (( waited < READY_TIMEOUT )); do
    if curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
      echo "[ready] $suite after ${waited}s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[fatal] $suite server exited before readiness" >&2
      tail -40 "$LOGDIR/server_${suite}.log" >&2 || true
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "[fatal] $suite did not become ready within ${READY_TIMEOUT}s" >&2
  return 1
}

echo "logdir=$LOGDIR"
echo "tag=$TAG tasks=$TASK_SPEC trials=$TRIALS delta=$TAU_RADIUS PF=$PF_EVERY replan=$REPLAN_STEPS"
echo "mapping: spatial->GPU0 object->GPU1 goal->GPU2 10->GPU3"

for i in "${!SUITES[@]}"; do
  suite=${SUITES[$i]}
  gpu=${GPUS[$i]}
  port=${PORTS[$i]}
  CUDA_VISIBLE_DEVICES=$gpu \
  OPENPI_DATA_HOME=$ROOT/shared/checkpoints/openpi \
  nohup .venv/bin/python scripts/spec/spec_serve_policy.py \
    --config pi0_libero --backend triton \
    --base-triton-path "$CK/converted/base" \
    --draft-triton-path "$CK/converted/${DRAFT[$suite]}" \
    --task-suite-name "$suite" \
    --tokenizer-source local --pytorch-device cuda:0 \
    --tau-radius "$TAU_RADIUS" \
    --periodic-full-every-n-draft-rounds "$PF_EVERY" \
    --max-exec-steps "$MAX_EXEC_STEPS" \
    --port "$port" > "$LOGDIR/server_${suite}.log" 2>&1 &
  SERVER_PIDS[$suite]=$!
  echo "[server] $suite pid=${SERVER_PIDS[$suite]} GPU$gpu port=$port"
done

for i in "${!SUITES[@]}"; do
  suite=${SUITES[$i]}
  wait_ready "$suite" "${PORTS[$i]}" "${SERVER_PIDS[$suite]}"
done

for i in "${!SUITES[@]}"; do
  suite=${SUITES[$i]}
  gpu=${GPUS[$i]}
  port=${PORTS[$i]}
  LIBERO_CONFIG_PATH="$REPO/.libero" \
  PYTHONPATH="$REPO/third_party/libero" \
  XDG_CACHE_HOME=$ROOT/shared/caches/xdg \
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu \
  nohup examples/libero/.venv/bin/python scripts/spec/spec_client_libero.py \
    --task-suite-name "$suite" --host 127.0.0.1 --port "$port" \
    --num-trials-per-task "$TRIALS" --task "$TASK_SPEC" \
    --replan-steps "$REPLAN_STEPS" \
    --run-name "${TAG}_${suite}" > "$LOGDIR/client_${suite}.log" 2>&1 &
  CLIENT_PIDS[$suite]=$!
  echo "[client] $suite pid=${CLIENT_PIDS[$suite]} GPU$gpu"
done

started=$SECONDS
while :; do
  running=0
  progress=""
  for suite in "${SUITES[@]}"; do
    pid=${CLIENT_PIDS[$suite]}
    kill -0 "$pid" 2>/dev/null && running=$((running + 1))
    progress+="$suite=$(count_done "$suite") "
  done
  echo "[progress $((SECONDS - started))s] $progress"
  (( running == 0 )) && break
  sleep 60
done

failed=0
for suite in "${SUITES[@]}"; do
  pid=${CLIENT_PIDS[$suite]}
  if ! wait "$pid"; then
    echo "[fatal] $suite client exited unsuccessfully; log tail:" >&2
    tail -40 "$LOGDIR/client_${suite}.log" >&2 || true
    failed=1
  fi
done
(( failed == 0 )) || exit 1

python3 - "$REPO/data/test" "$TAG" "$TRIALS" "$TASK_SPEC" <<'PY'
import json
import pathlib
import re
import sys

base = pathlib.Path(sys.argv[1])
tag = sys.argv[2]
trials = int(sys.argv[3])
task_spec = sys.argv[4]
task_ids = set()
for token in task_spec.split(","):
    token = token.strip()
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
    if match:
        task_ids.update(range(int(match.group(1)), int(match.group(2)) + 1))
    else:
        task_ids.add(int(token))
expected = len(task_ids) * trials
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    path = base / f"{tag}_{suite}" / "episode_log.json"
    records = json.loads(path.read_text())
    assert len(records) == expected, (suite, len(records), expected)
    assert all("trajectory_length" in row for row in records), suite
    assert all(row["trajectory_length"] == row["executed_action_count"] for row in records), suite
    keys = {(row["task_id"], row["episode_idx"]) for row in records}
    assert len(keys) == expected, (suite, len(keys), expected)
    print(f"[verified] {suite}: {len(records)} episodes, trajectory_length present and consistent")
PY

echo "[done] all suites finished in $(( (SECONDS - started) / 60 )) minutes"
echo "summarize: python3 $ROOT/scripts/summarize_flash_runs.py '$REPO/data/test/${TAG}_*'"
