# 两阶段 batch 研究：复现与下载

这些是已经提交的历史实际命令。原输出已存在，不要直接重跑或覆盖；后续运行需重新分配空闲 GPU、使用新输出名，并先跳过已有相容有效结果。实际参数、CPU 绑定、源码快照、数值资格和退出状态分别保存在每个执行回执及运行目录。

本轮正式模型是完整 pi05_libero、D10/H10、lead3；不同 B 独立进程。除显式 input placement / lookahead / SM 配额外沿用相同 Graph、精度和噪声。B>1 显式诊断不会恢复 B1 equivalence 资格。

## 仅准备后续命令

下面的入口只生成文件，不查询或占用 GPU。输出目录必须是新目录。

```bash
python3 -m scripts.idea2_prepare_batch_followup --output results/idea2/my_new_batch_capacity_plan --diagnostic-batch-drift --prepare-stream dedicated --no-profiles
python3 -m scripts.idea2_prepare_batch_followup --output results/idea2/my_new_batch_fresh_plan --input-modes online --disable-lookahead --diagnostic-batch-drift --prepare-stream dedicated --no-profiles
```

## 性能、连续运行和 trace 的实际命令

无 profiler 的正式性能为100对 AB/BA 或 Green 100组，每组8批请求；连续验证是2组×100批，单独报告；profile不参与加速比。每条命令从项目根目录运行，CPU亲和性来自实际回执，环境来自原始模板并替换成实际物理GPU UUID。

### performance

#### stage1_original_b1_full_prepared

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage1_original_b1_full_prepared.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage1_original_b1_full_prepared --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode paired --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4
```

#### stage2_b1_sm16_prepared_full

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_b1_sm16_prepared_full.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b1_sm16_prepared_full --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4
```

#### stage2_b2_full_prepared_full

物理GPU：`GPU-555cde36-5e41-bc44-0c69-ea47358fd703`；回执：`stage2_b2_full_prepared_full.execution.json`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 env CUDA_VISIBLE_DEVICES=GPU-555cde36-5e41-bc44-0c69-ea47358fd703 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b2_full_prepared_full --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode paired --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b4_full_prepared_full

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_b4_full_prepared_full.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b4_full_prepared_full --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode paired --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b2_sm16_prepared_full

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_b2_sm16_prepared_full.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b2_sm16_prepared_full --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode green --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b4_sm16_prepared_full

物理GPU：`GPU-555cde36-5e41-bc44-0c69-ea47358fd703`；回执：`stage2_b4_sm16_prepared_full.execution.json`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 env CUDA_VISIBLE_DEVICES=GPU-555cde36-5e41-bc44-0c69-ea47358fd703 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b4_sm16_prepared_full --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b2_sm24_prepared_full

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_b2_sm24_prepared_full.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b2_sm24_prepared_full --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode green --input-mode prepared --lead 3 --ae-sm-min 24 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b4_sm24_prepared_full

物理GPU：`GPU-555cde36-5e41-bc44-0c69-ea47358fd703`；回执：`stage2_b4_sm24_prepared_full.execution.json`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 env CUDA_VISIBLE_DEVICES=GPU-555cde36-5e41-bc44-0c69-ea47358fd703 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b4_sm24_prepared_full --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode prepared --lead 3 --ae-sm-min 24 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b1_full_online_full

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_b1_full_online_full.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b1_full_online_full --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode paired --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4
```

#### stage2_b4_sm20_prepared_full

物理GPU：`GPU-555cde36-5e41-bc44-0c69-ea47358fd703`；回执：`stage2_b4_sm20_prepared_full.execution.json`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 env CUDA_VISIBLE_DEVICES=GPU-555cde36-5e41-bc44-0c69-ea47358fd703 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b4_sm20_prepared_full --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode prepared --lead 3 --ae-sm-min 20 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b2_full_online_full

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_b2_full_online_full.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b2_full_online_full --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode paired --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b4_full_online_full

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_b4_full_online_full.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b4_full_online_full --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode paired --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b1_sm16_online_full

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_b1_sm16_online_full.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b1_sm16_online_full --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4
```

#### stage2_b2_sm16_online_full

物理GPU：`GPU-555cde36-5e41-bc44-0c69-ea47358fd703`；回执：`stage2_b2_sm16_online_full.execution.json`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 env CUDA_VISIBLE_DEVICES=GPU-555cde36-5e41-bc44-0c69-ea47358fd703 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b2_sm16_online_full --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b4_sm16_online_full

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_b4_sm16_online_full.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b4_sm16_online_full --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b2_sm24_online_full

物理GPU：`GPU-555cde36-5e41-bc44-0c69-ea47358fd703`；回执：`stage2_b2_sm24_online_full.execution.json`。

```bash
taskset -c 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111 env CUDA_VISIBLE_DEVICES=GPU-555cde36-5e41-bc44-0c69-ea47358fd703 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b2_sm24_online_full --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode green --input-mode online --lead 3 --ae-sm-min 24 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_b4_sm20_online_full

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_b4_sm20_online_full.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_b4_sm20_online_full --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode online --lead 3 --ae-sm-min 20 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_tail_b2_sm24_dedicated

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_tail_b2_sm24_dedicated.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b2_sm24_dedicated --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode green --input-mode online --lead 3 --ae-sm-min 24 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift --prepare-stream dedicated
```

#### stage2_tail_b1_sm16_dedicated

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_tail_b1_sm16_dedicated.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b1_sm16_dedicated --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --prepare-stream dedicated
```

#### stage2_tail_b4_sm20_dedicated

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_tail_b4_sm20_dedicated.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b4_sm20_dedicated --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode online --lead 3 --ae-sm-min 20 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift --prepare-stream dedicated
```

#### stage2_tail_b1_full_online_reverse

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_tail_b1_full_online_reverse.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b1_full_online_reverse --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode paired --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --reverse
```

#### stage2_tail_b2_sm16_dedicated

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_tail_b2_sm16_dedicated.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b2_sm16_dedicated --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift --prepare-stream dedicated
```

#### stage2_tail_b4_sm16_dedicated

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_tail_b4_sm16_dedicated.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b4_sm16_dedicated --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift --prepare-stream dedicated
```

#### stage2_tail_b4_sm20_prepared_reverse

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_tail_b4_sm20_prepared_reverse.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b4_sm20_prepared_reverse --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode prepared --lead 3 --ae-sm-min 20 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_tail_b4_sm24_prepared_reverse

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_tail_b4_sm24_prepared_reverse.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b4_sm24_prepared_reverse --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode prepared --lead 3 --ae-sm-min 24 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_latency_b1_caller_matched

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_latency_b1_caller_matched.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_latency_b1_caller_matched --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4
```

#### stage2_latency_b2_caller_matched

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_latency_b2_caller_matched.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_latency_b2_caller_matched --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode green --input-mode online --lead 3 --ae-sm-min 24 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_latency_b4_caller_matched

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_latency_b4_caller_matched.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_latency_b4_caller_matched --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode online --lead 3 --ae-sm-min 20 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift
```

#### stage2_latency_b1_fresh_full

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_latency_b1_fresh_full.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_latency_b1_fresh_full --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode paired --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --reverse --disable-lookahead
```

#### stage2_latency_b2_fresh_full

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_latency_b2_fresh_full.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_latency_b2_fresh_full --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode paired --input-mode online --lead 3 --ae-sm-min 24 --pairs 100 --length 8 --warmup 20 --verify-count 4 --reverse --diagnostic-batch-drift --disable-lookahead
```

#### stage2_latency_b4_fresh_full

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_latency_b4_fresh_full.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_latency_b4_fresh_full --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode paired --input-mode online --lead 3 --ae-sm-min 20 --pairs 100 --length 8 --warmup 20 --verify-count 4 --reverse --diagnostic-batch-drift --disable-lookahead
```

#### stage2_latency_b1_fresh_green

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_latency_b1_fresh_green.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_latency_b1_fresh_green --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4 --prepare-stream dedicated --disable-lookahead
```

#### stage2_latency_b2_fresh_green

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_latency_b2_fresh_green.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_latency_b2_fresh_green --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 2 --mode green --input-mode online --lead 3 --ae-sm-min 24 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift --prepare-stream dedicated --disable-lookahead
```

#### stage2_latency_b4_fresh_green

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_latency_b4_fresh_green.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_latency_b4_fresh_green --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode online --lead 3 --ae-sm-min 20 --pairs 100 --length 8 --warmup 20 --verify-count 4 --diagnostic-batch-drift --prepare-stream dedicated --disable-lookahead
```

### continuity

#### stage2_tail_b1_green_continuous

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_tail_b1_green_continuous.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b1_green_continuous --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 2 --length 100 --warmup 20 --verify-count 4 --prepare-stream dedicated
```

#### stage2_tail_b1_full_continuous

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_tail_b1_full_continuous.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b1_full_continuous --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode paired --input-mode online --lead 3 --ae-sm-min 16 --pairs 2 --length 100 --warmup 20 --verify-count 4 --reverse
```

#### stage2_tail_b4_green_continuous

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_tail_b4_green_continuous.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b4_green_continuous --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode online --lead 3 --ae-sm-min 20 --pairs 2 --length 100 --warmup 20 --verify-count 4 --diagnostic-batch-drift --prepare-stream dedicated
```

#### stage2_tail_b4_full_continuous

物理GPU：`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`；回执：`stage2_tail_b4_full_continuous.execution.json`。

```bash
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 env CUDA_VISIBLE_DEVICES=GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b4_full_continuous --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode paired --input-mode online --lead 3 --ae-sm-min 16 --pairs 2 --length 100 --warmup 20 --verify-count 4 --reverse --diagnostic-batch-drift
```

### profiles

#### stage2_profile_b4_serial_prepared

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_profile_b4_serial_prepared.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_profile_b4_serial_prepared_trace /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_profile_b4_serial_prepared --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode serial --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile --diagnostic-batch-drift
```

#### stage2_profile_b4_pipeline_prepared

物理GPU：`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`；回执：`stage2_profile_b4_pipeline_prepared.execution.json`。

```bash
taskset -c 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 env CUDA_VISIBLE_DEVICES=GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_profile_b4_pipeline_prepared_trace /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_profile_b4_pipeline_prepared --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode pipeline --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile --diagnostic-batch-drift
```

#### stage2_profile_b4_sm16_online

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_profile_b4_sm16_online.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_profile_b4_sm16_online_trace /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_profile_b4_sm16_online --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile --diagnostic-batch-drift
```

#### stage2_profile_b4_sm20_prepared

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_profile_b4_sm20_prepared.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_profile_b4_sm20_prepared_trace /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_profile_b4_sm20_prepared --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode prepared --lead 3 --ae-sm-min 20 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile --diagnostic-batch-drift
```

#### stage2_tail_b1_dedicated_profile

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_tail_b1_dedicated_profile.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b1_dedicated_profile_trace /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b1_dedicated_profile --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile --prepare-stream dedicated
```

#### stage2_tail_b4_dedicated_profile

物理GPU：`GPU-599477fb-0507-40dc-14df-d491f7a7b202`；回执：`stage2_tail_b4_dedicated_profile.execution.json`。

```bash
taskset -c 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 env CUDA_VISIBLE_DEVICES=GPU-599477fb-0507-40dc-14df-d491f7a7b202 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b4_dedicated_profile_trace /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/stage2_tail_b4_dedicated_profile --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 4 --mode green --input-mode online --lead 3 --ae-sm-min 20 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile --diagnostic-batch-drift --prepare-stream dedicated
```

## nsys 下载到桌面

六个下载别名保留简单阶段名 preprocess / encoder / vlm / AE；manifest.json含原始来源、文件大小和SHA。下载目录约30MiB，不包含较大的SQLite导出。

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$HOME\Desktop
sys" | Out-Null
scp -r smc-gpu:/root/nfs/embodied/benchmark/nsys/idea2_experiment/batch_study_20261101 "$HOME\Desktop
sys"
```

macOS/Linux：

```bash
mkdir -p "$HOME/Desktop/nsys"
scp -r smc-gpu:/root/nfs/embodied/benchmark/nsys/idea2_experiment/batch_study_20261101 "$HOME/Desktop/nsys/"
```

先看 B4 serial_prepared / streams_prepared / green20_prepared，再比较 green16_caller_online 与 green20_dedicated_online。最后两者配额不同；输入stream的单变量结论应同时看报告里的同卡同配额控制。B1 dedicated_online用于查看严格B1的真实Encoder重叠。
