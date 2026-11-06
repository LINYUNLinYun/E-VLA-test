#!/usr/bin/env bash
# Full LIBERO Triton-pi0 baseline: one suite per exclusive GPU, no speculative execution.
set -euo pipefail

ROOT=/root/nfs/embodied/benchmark
REPO=$ROOT/policies/realtime-vla-flash
CK=$ROOT/shared/checkpoints/realtime-vla-flash
TAG=${TAG:-pi0_baseline_full}
TRIALS=${TRIALS:-50}
REPLAN_STEPS=${REPLAN_STEPS:-12}
MAX_EXEC_STEPS=${MAX_EXEC_STEPS:-12}
TASK_SPEC=${TASK_SPEC:-0-9}
READY_TIMEOUT=${READY_TIMEOUT:-2400}
MAX_START_MEMORY_MIB=${MAX_START_MEMORY_MIB:-1000}
LOGDIR=${LOGDIR:-$ROOT/logs/${TAG}_$(date +%m%d_%H%M)}

SUITES=(libero_spatial libero_object libero_goal libero_10)
GPUS=(0 1 2 3)
PORTS=(8200 8201 8202 8203)
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

for gpu in "${GPUS[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used > MAX_START_MEMORY_MIB )); then
    echo "[fatal] GPU$gpu is not exclusive: ${used} MiB already in use" >&2
    exit 2
  fi
done

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
      tail -60 "$LOGDIR/server_${suite}.log" >&2 || true
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "[fatal] $suite did not become ready within ${READY_TIMEOUT}s" >&2
  return 1
}

echo "logdir=$LOGDIR"
echo "tag=$TAG tasks=$TASK_SPEC trials=$TRIALS replan=$REPLAN_STEPS"
echo "mode=Triton-pi0 full path every round; draft artifacts are loaded only because the runtime requires them"

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
    --max-exec-steps "$MAX_EXEC_STEPS" \
    --force-full-each-round \
    --port "$port" >"$LOGDIR/server_${suite}.log" 2>&1 &
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
    --run-name "${TAG}_${suite}" >"$LOGDIR/client_${suite}.log" 2>&1 &
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
    tail -60 "$LOGDIR/client_${suite}.log" >&2 || true
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
    assert all(row["trajectory_length"] == row["executed_action_count"] for row in records), suite
    assert len({(row["task_id"], row["episode_idx"]) for row in records}) == expected, suite
    for row in records:
        infer_path = base / f"{tag}_{suite}" / row["infer_path"]
        infer_rows = [json.loads(line) for line in infer_path.read_text().splitlines() if line.strip()]
        assert len(infer_rows) == row["infer_calls"], (suite, row["task_id"], row["episode_idx"])
        for infer in infer_rows:
            assert int(round(float(infer.get("is_full_pipeline_round", 0.0)))) == 1, (suite, row["task_id"])
            assert float(infer.get("draft_ms", 0.0)) == 0.0, (suite, row["task_id"])
    print(f"[verified] {suite}: {len(records)} episodes, trajectory length consistent, all rounds full path")
PY

echo "[done] all suites finished in $(( (SECONDS - started) / 60 )) minutes"
echo "summarize: python3 $ROOT/scripts/summarize_flash_runs.py '$REPO/data/test/${TAG}_*'"
