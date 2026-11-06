# 已完成实验的实际命令

重跑前修改输出目录与nsys文件名，重新协调独占GPU。
正式队列固定CPU0–15，闭环GPU0固定CPU32–47。

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=32
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=.:policies/openpi/src
export OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
```

## formal_same_task_gpu2

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/formal_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --conditions 0/encoder 1/encoder 3/encoder 6/encoder 18/host_only 18/encoder
```

## gpu0_encoder_after_step4_lead3_screen_clean

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_encoder_after_step4_lead3_screen_clean --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --encoder-after-step 4 --conditions 3/encoder --pairs 20 --length 8 --warmup 20 --verify-count 2 --continuous 20
```

## gpu0_encoder_after_step7_lead3_screen_clean

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_encoder_after_step7_lead3_screen_clean --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --encoder-after-step 7 --conditions 3/encoder --pairs 20 --length 8 --warmup 20 --verify-count 2 --continuous 20
```

## gpu0_full_control_formal

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_full_control_formal --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 0/encoder 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2
```

## gpu0_full_control_formal_repeat

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_full_control_formal_repeat --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 0/encoder 3/encoder 18/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --reverse
```

## gpu0_full_control_repeat_clean

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_full_control_repeat_clean --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/host_only 3/encoder 18/host_only 18/encoder --pairs 20 --length 8 --warmup 20 --verify-count 2 --continuous 20 --reverse
```

## gpu0_full_control_screen

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_full_control_screen --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --conditions 3/host_only 3/encoder 18/host_only 18/encoder --pairs 20 --length 8 --warmup 20 --verify-count 2 --continuous 20
```

## gpu0_full_lead3_profile

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/progressive_kv_full_lead3_encoder_gpu0_20261025_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_full_lead3_profile --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --profile --profile-lead 3 --conditions 3/encoder --length 8 --pairs 0 --warmup 20 --verify-count 1
```

## gpu0_resource_sm16_cleanup_gate

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_resource_sm16_cleanup_gate --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 16 --conditions 3/encoder --pairs 2 --length 4 --warmup 2 --verify-count 1
```

## gpu0_resource_sm16_lead3_screen_clean

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_resource_sm16_lead3_screen_clean --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 16 --conditions 3/host_only 3/encoder 18/host_only 18/encoder --pairs 20 --length 8 --warmup 20 --verify-count 2 --continuous 20
```

## gpu0_resource_sm32_lead3_screen_clean

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_resource_sm32_lead3_screen_clean --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 32 --conditions 3/host_only 3/encoder 18/host_only 18/encoder --pairs 20 --length 8 --warmup 20 --verify-count 2 --continuous 20
```

## gpu0_sm16_lead3_formal

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_sm16_lead3_formal --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 16 --conditions 3/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2
```

## gpu0_sm16_lead3_profile

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/progressive_kv_sm16_lead3_encoder_gpu0_20261025_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_sm16_lead3_profile --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 16 --profile --profile-lead 3 --conditions 3/encoder --length 8 --pairs 0 --warmup 20 --verify-count 1
```

## gpu0_sm16_lead3_repeat

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_sm16_lead3_repeat --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 16 --conditions 3/encoder --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --reverse
```

## gpu0_sm32_exposure_profile

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/progressive_kv_sm32_lead3_encoder_gpu0_20261025_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_adapter --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu0_sm32_exposure_profile --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 32 --profile --profile-lead 3 --conditions 3/encoder --length 8 --pairs 0 --warmup 20 --verify-count 1
```

## gpu3_sm16_slo_33ms

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_slo --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu3_sm16_slo_33ms --bank /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 16 --arrival-period-ms 33 --conditions 3/encoder --pairs 0 --continuous 100 --warmup 20 --verify-count 1
```

## gpu3_sm16_slo_33ms_repeat

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_slo --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu3_sm16_slo_33ms_repeat --bank /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 16 --arrival-period-ms 33 --conditions 3/encoder --pairs 0 --continuous 100 --warmup 20 --verify-count 1 --reverse
```

## gpu3_sm16_slo_62ms

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_slo --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu3_sm16_slo_62ms --bank /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 16 --arrival-period-ms 62 --conditions 3/encoder --pairs 0 --continuous 100 --warmup 20 --verify-count 1
```

## gpu3_sm16_slo_62ms_repeat

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.resource_slo --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/gpu3_sm16_slo_62ms_repeat --bank /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/same_task_bank.pkl --ae-sm-min 16 --arrival-period-ms 62 --conditions 3/encoder --pairs 0 --continuous 100 --warmup 20 --verify-count 1 --reverse
```

## profile_lead0_encoder_same_task_gpu2

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/progressive_kv_lead0_encoder_20261025_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/profile_lead0_encoder_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --profile --profile-lead 0 --profile-variant encoder --conditions 0/encoder --length 8 --pairs 0 --warmup 20 --verify-count 1
```

## profile_lead18_encoder_same_task_gpu2

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/progressive_kv_lead18_encoder_20261025_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/profile_lead18_encoder_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --profile --profile-lead 18 --profile-variant encoder --conditions 18/encoder --length 8 --pairs 0 --warmup 20 --verify-count 1
```

## profile_lead3_encoder_same_task_gpu2

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/progressive_kv_lead3_encoder_20261025_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/profile_lead3_encoder_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --profile --profile-lead 3 --profile-variant encoder --conditions 3/encoder --length 8 --pairs 0 --warmup 20 --verify-count 1
```

## profile_lead6_encoder_same_task_gpu2

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/progressive_kv_lead6_encoder_20261025_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/profile_lead6_encoder_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --profile --profile-lead 6 --profile-variant encoder --conditions 6/encoder --length 8 --pairs 0 --warmup 20 --verify-count 1
```

## repeat_same_task_gpu2

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.performance_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/repeat_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --pairs 100 --length 8 --continuous 100 --warmup 20 --verify-count 2 --conditions 0/encoder 1/encoder 3/encoder 6/encoder 18/host_only 18/encoder --reverse
```

## screen_same_task

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.progressive_kv_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/screen_same_task --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --pairs 20 --length 8 --warmup 20 --verify-count 2
```

## slo_33ms_repeat_same_task_gpu2

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.slo_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/slo_33ms_repeat_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --pairs 0 --continuous 100 --length 8 --warmup 20 --verify-count 1 --arrival-period-ms 33 --conditions 0/encoder 1/encoder 3/encoder 6/encoder 18/no_lookahead 18/host_only 18/encoder --reverse
```

## slo_33ms_same_task_gpu2

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.slo_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/slo_33ms_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --pairs 0 --continuous 100 --length 8 --warmup 20 --verify-count 1 --arrival-period-ms 33 --conditions 0/encoder 1/encoder 3/encoder 6/encoder 18/no_lookahead 18/host_only 18/encoder
```

## slo_62ms_repeat_same_task_gpu2

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.slo_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/slo_62ms_repeat_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --pairs 0 --continuous 100 --length 8 --warmup 20 --verify-count 1 --arrival-period-ms 62 --conditions 0/encoder 1/encoder 3/encoder 6/encoder 18/no_lookahead 18/host_only 18/encoder --reverse
```

## slo_62ms_same_task_gpu2

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.progressive_kv_20261025.slo_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25 --output /root/nfs/embodied/benchmark/results/idea2/progressive_kv_20261025/slo_62ms_same_task_gpu2 --bank results/idea2/progressive_kv_20261025/same_task_bank.pkl --pairs 0 --continuous 100 --length 8 --warmup 20 --verify-count 1 --arrival-period-ms 62 --conditions 0/encoder 1/encoder 3/encoder 6/encoder 18/no_lookahead 18/host_only 18/encoder
```

## 闭环与离线分析

```bash
policies/openpi/.venv/bin/python results/idea2/progressive_kv_20261025/closedloop.py --execute --output <新的目录> --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --egl-device 3 --port 19085 --cpus 32-47
policies/openpi/.venv/bin/python results/idea2/progressive_kv_20261025/summarize.py
```
