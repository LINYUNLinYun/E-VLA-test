#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="$root_dir/profile_results/pi05"
port="${PI05_PROFILE_PORT:-8001}"
mkdir -p "$result_dir"

# 请求生命周期：单片参考基线 warmup + 测量，再是拆解路径 warmup + 采集。
# 拆解路径至少要多跑几次，让各组件分别完成 XLA 编译，编译不能落进 capture window。
reference_warmup="${PI05_REFERENCE_WARMUP:-2}"
reference_iterations="${PI05_REFERENCE_ITERATIONS:-5}"
warmup="${PI05_WARMUP:-3}"
iterations="${PI05_ITERATIONS:-10}"
total_requests=$((reference_warmup + reference_iterations + warmup + iterations))

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENPI_DATA_HOME="$root_dir/shared/checkpoints/openpi"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=nvtx \
  --nvtx-capture='profile_window@*' \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$result_dir/pi05-server" \
  "$root_dir/policies/openpi/.venv/bin/python" \
  "$root_dir/profiling/pi05_server_profile.py" \
  --port "$port" \
  --reference-warmup "$reference_warmup" \
  --reference-iterations "$reference_iterations" \
  --warmup "$warmup" --iterations "$iterations" \
  --exit-after-profile \
  > >(tee "$result_dir/server.log") 2>&1 &
server_pid=$!

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 600); do
  if curl --silent --fail "http://127.0.0.1:$port/healthz" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    exit 1
  fi
  sleep 1
done

if ! curl --silent --fail "http://127.0.0.1:$port/healthz" >/dev/null; then
  echo "pi0.5 server did not become ready on port $port" >&2
  exit 1
fi

export LIBERO_CONFIG_PATH="$root_dir/policies/openpi/examples/libero/.libero"
export PYTHONPATH="$root_dir/policies/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
MUJOCO_GL=egl "$root_dir/policies/openpi/examples/libero/.venv/bin/python" \
  "$root_dir/profiling/pi05_libero_client.py" \
  --host 127.0.0.1 --port "$port" --requests "$total_requests" \
  > >(tee "$result_dir/client.log") 2>&1

wait "$server_pid"
trap - EXIT
