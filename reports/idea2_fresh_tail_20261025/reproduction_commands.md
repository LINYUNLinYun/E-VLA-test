# 实际执行命令

重跑前先协调独占GPU，并修改输出目录、port及nsys文件名；历史目录不可覆盖。所有source snapshots与原始命令均保留。计时需按记录的CPU affinity运行。

```bash
cd /root/nfs/embodied/benchmark
export CUDA_DEVICE_MAX_CONNECTIONS=32
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=.:policies/openpi/src
export OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## diagnostic_sm32_tail1_reverse

回执：`results/idea2/fresh_tail_20261025/diagnostic_matrix.execution.json`。
GPU `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，CPU `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/diagnostic_sm32_tail1_reverse --ae-sm-min 32 --fresh-tail-steps 1 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 20 --length 8 --continuous 20 --warmup 20 --verify-count 2 --reverse
```

## formal_sm16_tail0

回执：`results/idea2/fresh_tail_20261025/formal_sm16_matrix.execution.json`。
GPU `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，CPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_sm16_tail0 --ae-sm-min 16 --fresh-tail-steps 0 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2
```

## formal_sm16_tail1

回执：`results/idea2/fresh_tail_20261025/formal_sm16_matrix.execution.json`。
GPU `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，CPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_sm16_tail1 --ae-sm-min 16 --fresh-tail-steps 1 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2
```

## formal_sm16_tail2

回执：`results/idea2/fresh_tail_20261025/formal_sm16_matrix.execution.json`。
GPU `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，CPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_sm16_tail2 --ae-sm-min 16 --fresh-tail-steps 2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2
```

## formal_sm16_tail2_repeat

回执：`results/idea2/fresh_tail_20261025/formal_sm16_matrix.execution.json`。
GPU `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，CPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_sm16_tail2_repeat --ae-sm-min 16 --fresh-tail-steps 2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --reverse
```

## formal_sm16_tail1_repeat

回执：`results/idea2/fresh_tail_20261025/formal_sm16_matrix.execution.json`。
GPU `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，CPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_sm16_tail1_repeat --ae-sm-min 16 --fresh-tail-steps 1 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --reverse
```

## formal_sm16_tail0_repeat

回执：`results/idea2/fresh_tail_20261025/formal_sm16_matrix.execution.json`。
GPU `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，CPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_sm16_tail0_repeat --ae-sm-min 16 --fresh-tail-steps 0 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --reverse
```

## formal_gpu3_sm32_tail0

回执：`results/idea2/fresh_tail_20261025/formal_sm32_gpu3_matrix.execution.json`。
GPU `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，CPU `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_gpu3_sm32_tail0 --ae-sm-min 32 --fresh-tail-steps 0 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2
```

## formal_gpu3_sm32_tail1

回执：`results/idea2/fresh_tail_20261025/formal_sm32_gpu3_matrix.execution.json`。
GPU `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，CPU `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_gpu3_sm32_tail1 --ae-sm-min 32 --fresh-tail-steps 1 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2
```

## formal_gpu3_sm32_tail2

回执：`results/idea2/fresh_tail_20261025/formal_sm32_gpu3_matrix.execution.json`。
GPU `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，CPU `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_gpu3_sm32_tail2 --ae-sm-min 32 --fresh-tail-steps 2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2
```

## formal_gpu3_sm32_tail2_repeat

回执：`results/idea2/fresh_tail_20261025/formal_sm32_gpu3_matrix.execution.json`。
GPU `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，CPU `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_gpu3_sm32_tail2_repeat --ae-sm-min 32 --fresh-tail-steps 2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --reverse
```

## formal_gpu3_sm32_tail1_repeat_retry

回执：`results/idea2/fresh_tail_20261025/formal_sm32_gpu3_remaining_matrix.execution.json`。
GPU `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，CPU `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_gpu3_sm32_tail1_repeat_retry --ae-sm-min 32 --fresh-tail-steps 1 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --reverse
```

## formal_gpu3_sm32_tail0_repeat

回执：`results/idea2/fresh_tail_20261025/formal_sm32_gpu3_remaining_matrix.execution.json`。
GPU `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，CPU `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/formal_gpu3_sm32_tail0_repeat --ae-sm-min 32 --fresh-tail-steps 0 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --reverse
```

## gate_sm16_tail2

回执：`results/idea2/fresh_tail_20261025/gate_matrix.execution.json`。
GPU `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，CPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/gate_sm16_tail2 --ae-sm-min 16 --fresh-tail-steps 2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 2 --length 4 --warmup 2 --verify-count 4 --continuous 0
```

## gate_sm32_tail1

回执：`results/idea2/fresh_tail_20261025/gate_matrix.execution.json`。
GPU `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，CPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/gate_sm32_tail1 --ae-sm-min 32 --fresh-tail-steps 1 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 2 --length 4 --warmup 2 --verify-count 4 --continuous 0
```

## gate_sm16_tail0

回执：`results/idea2/fresh_tail_20261025/gate_matrix.execution.json`。
GPU `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，CPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/gate_sm16_tail0 --ae-sm-min 16 --fresh-tail-steps 0 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/encoder 18/encoder --pairs 2 --length 4 --warmup 2 --verify-count 4 --continuous 0
```

## profile_sm32_tail1

回执：`results/idea2/fresh_tail_20261025/profile_matrix.execution.json`。
GPU `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，CPU `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/fresh_tail_sm32_tail1_lead3_encoder_20261025_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/profile_sm32_tail1 --ae-sm-min 32 --fresh-tail-steps 1 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --profile --profile-lead 3 --profile-variant encoder --conditions 3/encoder --length 8 --pairs 0 --warmup 20 --verify-count 1
```

## profile_sm32_tail2

回执：`results/idea2/fresh_tail_20261025/profile_matrix.execution.json`。
GPU `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，CPU `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/fresh_tail_sm32_tail2_lead3_encoder_20261025_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/profile_sm32_tail2 --ae-sm-min 32 --fresh-tail-steps 2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --profile --profile-lead 3 --profile-variant encoder --conditions 3/encoder --length 8 --pairs 0 --warmup 20 --verify-count 1
```

## matched100_gpu3

回执：`results/idea2/fresh_tail_20261025/matched100_gpu3/process_receipt.json`。

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.matched_quality --execute --count 100 --output /root/nfs/embodied/benchmark/results/idea2/fresh_tail_20261025/matched100_gpu3 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --cpus 96-111
```

## counterfactual_gpu3

回执：`results/idea2/fresh_tail_20261025/counterfactual_gpu3/process_receipt.json`。

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.counterfactual --execute
```

## closedloop_gpu2_eglmap

回执：`results/idea2/fresh_tail_20261025/closedloop_gpu2_eglmap/execution.json`。

```bash
policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.quality --execute --output results/idea2/fresh_tail_20261025/closedloop_gpu2_eglmap --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --egl-device 1 --port 19090 --cpus 0-15
```
