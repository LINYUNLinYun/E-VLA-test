#!/usr/bin/env bash
# 两路并行跑 LIBERO 四个 suite 的 Realtime-VLA FLASH 复现。
#
# 超参数按论文对齐（不改任何仓库默认之外的东西）：
#   δ (--tau-radius)                          = 0.15  论文 Table 3 caption / Table 7-8
#   PF (--periodic-full-every-n-draft-rounds) = 2     论文 Table 3 "+FB & PF=2"
#   K = 2（t_list 用仓库默认 (0.10,0.05)，论文只给了 K=|T|=2 未给具体 τ 值）
#   replan every 12 actions (--max-exec-steps 12 / --replan-steps 12)
#   50 trials/task, num_steps_wait 10, resize 224, seed 7
#
# 并行方式：每个 suite 一个服务端进程（各自 draft 权重不同，无法共用），
# 两个服务端同放 SERVER_GPU（推理只占单集 ~4% 墙钟，SM 争抢无影响），
# 两个客户端同放 CLIENT_GPU 做 EGL 渲染。每轮两个 suite 同时跑。
#
# 用法: bash scripts/run_flash_parallel.sh [--dry-run]
set -uo pipefail

ROOT=/root/nfs/embodied/benchmark
REPO=$ROOT/policies/realtime-vla-flash
CK=$ROOT/shared/checkpoints/realtime-vla-flash
LOGDIR=${LOGDIR:-$ROOT/logs/flash_paper_$(date +%m%d_%H%M)}
TAG=${TAG:-paper}                 # run_name 前缀 -> data/test/<TAG>_<suite>
TRIALS=${TRIALS:-50}
TAU_RADIUS=${TAU_RADIUS:-0.15}
PF_EVERY=${PF_EVERY:-2}
MAX_EXEC_STEPS=${MAX_EXEC_STEPS:-12}
REPLAN_STEPS=${REPLAN_STEPS:-12}
SERVER_GPU=${SERVER_GPU:-0}
CLIENT_GPU=${CLIENT_GPU:-0}   # GPU1-3 常被他人任务占用，默认与服务端共用 GPU0（渲染占用极小）
PORTS=(8000 8001)
READY_TIMEOUT=${READY_TIMEOUT:-2400}   # 冷启动要加载 6.5G 权重 + 建 10 个 prompt 的 prefill 缓存

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# suite -> draft 转换产物目录名（goal 的那份就叫 draft）
declare -A DRAFT=(
  [libero_spatial]=draft_spatial
  [libero_object]=draft_object
  [libero_goal]=draft
  [libero_10]=draft_10
)
ROUNDS_DEFAULT=("libero_spatial libero_object" "libero_goal libero_10")
# 单跑某一轮（避免跑完一轮后自动进下一轮、和别人抢同一个 run 目录）:
#   ROUNDS_OVERRIDE="libero_goal libero_10" bash scripts/run_flash_parallel.sh
if [[ -n ${ROUNDS_OVERRIDE:-} ]]; then
  ROUNDS=("$ROUNDS_OVERRIDE")
else
  ROUNDS=("${ROUNDS_DEFAULT[@]}")
fi

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

echo "日志目录: $LOGDIR"
echo "配置: δ=$TAU_RADIUS  PF=$PF_EVERY  K=2(仓库默认)  replan=$MAX_EXEC_STEPS  trials=$TRIALS"
echo "      server GPU$SERVER_GPU (每轮 2 实例)   client GPU$CLIENT_GPU"
(( DRY_RUN )) && echo "      *** DRY RUN，不真正启动 ***"

server_log() { echo "$LOGDIR/server_$1.log"; }
client_log() { echo "$LOGDIR/client_$1.log"; }
pidfile()    { echo "$LOGDIR/$2_$1.pid"; }
alive()      { [[ -f $1 ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

LIVE_SERVERS=()
cleanup() {
  (( DRY_RUN )) && return 0
  for suite in "${LIVE_SERVERS[@]:-}"; do
    [[ -z $suite ]] && continue
    pf=$(pidfile "$suite" server)
    if alive "$pf"; then kill "$(cat "$pf")" 2>/dev/null; fi
  done
}
trap cleanup EXIT

start_server() {
  local suite=$1 port=$2
  local draft=${DRAFT[$suite]}
  echo "  [server] $suite  draft=converted/$draft  port=$port  GPU$SERVER_GPU"
  if (( DRY_RUN )); then return 0; fi
  CUDA_VISIBLE_DEVICES=$SERVER_GPU \
  OPENPI_DATA_HOME=$ROOT/shared/checkpoints/openpi \
  nohup .venv/bin/python scripts/spec/spec_serve_policy.py \
    --config pi0_libero --backend triton \
    --base-triton-path "$CK/converted/base" \
    --draft-triton-path "$CK/converted/$draft" \
    --task-suite-name "$suite" \
    --tokenizer-source local --pytorch-device cuda:0 \
    --tau-radius "$TAU_RADIUS" \
    --periodic-full-every-n-draft-rounds "$PF_EVERY" \
    --max-exec-steps "$MAX_EXEC_STEPS" \
    --port "$port" > "$(server_log "$suite")" 2>&1 &
  echo $! > "$(pidfile "$suite" server)"
  LIVE_SERVERS+=("$suite")
}

wait_ready() {
  local suite=$1 port=$2 pf; pf=$(pidfile "$suite" server)
  (( DRY_RUN )) && return 0
  local waited=0
  while (( waited < READY_TIMEOUT )); do
    if curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
      echo "  [ready ] $suite 就绪（等待 ${waited}s）"
      return 0
    fi
    if ! alive "$pf"; then
      echo "  [FATAL ] $suite 服务端进程已退出，日志尾部："
      tail -20 "$(server_log "$suite")" | sed 's/^/          /'
      return 1
    fi
    sleep 10; waited=$((waited + 10))
  done
  echo "  [FATAL ] $suite 等待 ${READY_TIMEOUT}s 仍未就绪"
  return 1
}

run_client() {
  local suite=$1 port=$2
  echo "  [client] $suite  task 0-9 × $TRIALS 集  run_name=${TAG}_${suite}"
  (( DRY_RUN )) && return 0
  LIBERO_CONFIG_PATH="$REPO/.libero" \
  PYTHONPATH="$REPO/third_party/libero" \
  XDG_CACHE_HOME=$ROOT/shared/caches/xdg \
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$CLIENT_GPU \
  nohup examples/libero/.venv/bin/python scripts/spec/spec_client_libero.py \
    --task-suite-name "$suite" --host 127.0.0.1 --port "$port" \
    --num-trials-per-task "$TRIALS" --task 0-9 \
    --replan-steps "$REPLAN_STEPS" \
    --run-name "${TAG}_${suite}" > "$(client_log "$suite")" 2>&1 &
  echo $! > "$(pidfile "$suite" client)"
}

count_done() {
  local suite=$1 f="$REPO/data/test/${TAG}_${suite}/episode_log.json"
  [[ -f $f ]] && python3 -c "import json,sys;print(len(json.load(open('$f'))))" 2>/dev/null || echo 0
}

# 轮询等两个客户端跑完，顺便报进度
wait_clients() {
  local suites=("$@") t0=$SECONDS
  while :; do
    local running=0 line=""
    for suite in "${suites[@]}"; do
      local pf; pf=$(pidfile "$suite" client)
      if alive "$pf"; then running=$((running + 1)); fi
      line+="$suite=$(count_done "$suite") "
    done
    printf '\r  [进度] %s  已跑 %ds   ' "$line" "$((SECONDS - t0))"
    (( running == 0 )) && break
    sleep 60
  done
  printf '\n'
}

stop_servers() {
  for suite in "$@"; do
    local pf; pf=$(pidfile "$suite" server)
    if alive "$pf"; then
      local pid; pid=$(cat "$pf")
      kill "$pid" 2>/dev/null
      for _ in $(seq 1 30); do alive "$pf" || break; sleep 1; done
      alive "$pf" && kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$pf"
  done
}

round=0
for group in "${ROUNDS[@]}"; do
  round=$((round + 1))
  read -r s0 s1 <<< "$group"
  echo ""
  echo "=== 第 $round 轮: $s0 + $s1  ($(date +%H:%M:%S)) ==="
  start_server "$s0" "${PORTS[0]}"
  start_server "$s1" "${PORTS[1]}"
  ok=1
  wait_ready "$s0" "${PORTS[0]}" || ok=0
  wait_ready "$s1" "${PORTS[1]}" || ok=0
  if (( ! ok )); then
    echo "  第 $round 轮启动失败，跳过"
    stop_servers "$s0" "$s1"
    continue
  fi
  run_client "$s0" "${PORTS[0]}"
  run_client "$s1" "${PORTS[1]}"
  (( DRY_RUN )) || wait_clients "$s0" "$s1"
  echo "  [done  ] $s0 + $s1  ($(date +%H:%M:%S))"
  stop_servers "$s0" "$s1"
done

echo ""
echo "=== 全部结束 ($(date +%H:%M:%S))，共 $((SECONDS / 60)) 分钟 ==="
echo "汇总:  python3 $ROOT/scripts/summarize_flash_runs.py '$REPO/data/test/${TAG}_*'"
