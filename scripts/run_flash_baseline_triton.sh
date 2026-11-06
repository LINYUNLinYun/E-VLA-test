#!/usr/bin/env bash
# 论文 Triton-π0 基线对照（无投机执行），只跑 LIBERO-Goal 的 task 5 / 9 各 50 集。
#
# 目的：分离「π0 本身在这两个 task 上就弱」和「FLASH 的接受链引入了错误」。
#   论文 Table 1：Triton-π0 Goal = 95.0%；FLASH+Triton-π0 Goal = 94.4%
#   我们 δ=0.15/PF=2 下 FLASH 的 Goal 结果见 realtime-vla-flash复现.md。
#
# 实现方式：--force-full-each-round，每轮都走全路径（spec_pi0_pytorch.py:206），
#   因此不会有任何 draft 动作被接受 —— 动作序列等同于纯 π0 + Triton kernel。
#   注意 Triton 运行时仍要求 --draft-triton-path 才能构造（spec_serve_policy.py:882），
#   但该权重在本模式下不会被用到。
#
# 用法: bash scripts/run_flash_baseline_triton.sh
set -uo pipefail

ROOT=/root/nfs/embodied/benchmark
REPO=$ROOT/policies/realtime-vla-flash
CK=$ROOT/shared/checkpoints/realtime-vla-flash
LOGDIR=${LOGDIR:-$ROOT/logs/flash_baseline_triton}
PORT=${PORT:-8004}
GPU=${GPU:-0}
TRIALS=${TRIALS:-50}
TASKS=${TASKS:-5,9}
RUN_NAME=${RUN_NAME:-paper_triton_goal_t59}

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

echo "日志: $LOGDIR   port=$PORT  GPU$GPU  tasks=$TASKS × $TRIALS 集"

CUDA_VISIBLE_DEVICES=$GPU \
OPENPI_DATA_HOME=$ROOT/shared/checkpoints/openpi \
nohup .venv/bin/python scripts/spec/spec_serve_policy.py \
  --config pi0_libero --backend triton \
  --base-triton-path "$CK/converted/base" \
  --draft-triton-path "$CK/converted/draft" \
  --task-suite-name libero_goal \
  --tokenizer-source local --pytorch-device cuda:0 \
  --force-full-each-round \
  --port "$PORT" > "$LOGDIR/server.log" 2>&1 &
SRV=$!
echo "  [server] pid=$SRV"

for i in $(seq 1 180); do
  grep -q "server listening on" "$LOGDIR/server.log" 2>/dev/null && { echo "  [ready ] 等待 ${i}s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "  [FATAL ] 服务端退出"; tail -20 "$LOGDIR/server.log"; exit 1; }
  sleep 1
done

LIBERO_CONFIG_PATH="$REPO/.libero" \
PYTHONPATH="$REPO/third_party/libero" \
XDG_CACHE_HOME=$ROOT/shared/caches/xdg \
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU \
nohup examples/libero/.venv/bin/python scripts/spec/spec_client_libero.py \
  --task-suite-name libero_goal --host 127.0.0.1 --port "$PORT" \
  --num-trials-per-task "$TRIALS" --task "$TASKS" \
  --replan-steps 12 --run-name "$RUN_NAME" > "$LOGDIR/client.log" 2>&1 &
CLI=$!
echo "  [client] pid=$CLI  run_name=$RUN_NAME"

wait $CLI
echo "  [done  ] 客户端结束，关闭服务端"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "结果: $REPO/data/test/$RUN_NAME"
