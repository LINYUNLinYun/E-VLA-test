# Fixed-batch pipeline commands

Prepared only. Nothing has been launched. Run from the repository root.

Replace the GPU UUID placeholder after a GPU is allocated. Each command checks the shared lease and actual occupancy.

Pass smoke numerical checks before performance; capture selected profiles separately. Do not run all jobs concurrently.

## smoke_b1_paired_prepared

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/smoke_b1_paired_prepared --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode paired --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 2 --length 4 --warmup 2 --verify-count 4
```

## smoke_b1_green_sm16_prepared

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/smoke_b1_green_sm16_prepared --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 2 --length 4 --warmup 2 --verify-count 4
```

## performance_b1_paired_prepared

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/performance_b1_paired_prepared --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode paired --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4
```

## performance_b1_green_sm16_prepared

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/performance_b1_green_sm16_prepared --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4
```

## performance_b1_paired_online

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/performance_b1_paired_online --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode paired --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4
```

## performance_b1_green_sm16_online

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/performance_b1_green_sm16_online --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 100 --length 8 --warmup 20 --verify-count 4
```

## profile_b1_serial_prepared

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/profile_b1_serial_prepared /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/profile_b1_serial_prepared --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode serial --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile
```

## profile_b1_pipeline_prepared

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/profile_b1_pipeline_prepared /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/profile_b1_pipeline_prepared --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode pipeline --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile
```

## profile_b1_green_sm16_prepared

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/profile_b1_green_sm16_prepared /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/profile_b1_green_sm16_prepared --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode prepared --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile
```

## profile_b1_serial_online

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/profile_b1_serial_online /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/profile_b1_serial_online --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode serial --input-mode online --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile
```

## profile_b1_pipeline_online

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/profile_b1_pipeline_online /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/profile_b1_pipeline_online --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode pipeline --input-mode online --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile
```

## profile_b1_green_sm16_online

Numerical policy: strict_b1

```bash
env CUDA_VISIBLE_DEVICES=GPU-SET-ME-BEFORE-RUNNING JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only:nvtx-live --sample=none --cpuctxsw=none --nvtx-domain-exclude=TSL,CCCL --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none --force-overwrite=false --output=/root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/profile_b1_green_sm16_online /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.batch_bench --output /root/nfs/embodied/benchmark/results/idea2/batch_preparation_20261101/strict_b1/runs/profile_b1_green_sm16_online --gpu-uuid GPU-SET-ME-BEFORE-RUNNING --checkpoint /root/nfs/embodied/benchmark/shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --bank /root/nfs/embodied/benchmark/results/idea1_v3_1/input_bank.pkl --batch-size 1 --mode green --input-mode online --lead 3 --ae-sm-min 16 --pairs 1 --length 8 --warmup 20 --verify-count 4 --profile
```
