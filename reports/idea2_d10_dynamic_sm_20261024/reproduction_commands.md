# 实际复现命令

在 `/root/nfs/embodied/benchmark` 下执行。以下来自实际执行收据；**重跑前给 `--output` 和 nsys `-o` 换一个新目录/文件名**，保留原始证据。只在已协调独占的 GPU 上运行，UUID 是本轮设备，不是永久授权。

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=32
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=.:policies/openpi/src
export OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
```

## d10_async_serial_formal

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.d10_dynamic_sm_20261024.async_serial_control --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_async_serial_formal --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective vision_only --warmup 20 --count 100 --continuous-count 100 --reset-every 50 --pairs graph_async_serial:graph_vision_overlap --continuous graph_async_serial graph_vision_overlap
```

## d10_async_serial_gate

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.d10_dynamic_sm_20261024.async_serial_control --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_async_serial_gate --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective vision_only --warmup 20 --count 20 --continuous-count 20 --reset-every 10 --pairs graph_async_serial:graph_vision_overlap --continuous graph_async_serial graph_vision_overlap
```

## d10_async_serial_profile

profile / 不作正式计时

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/d10_async_serial_profile_20261024_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.d10_dynamic_sm_20261024.async_serial_control --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_async_serial_profile --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective vision_only --warmup 20 --count 0 --continuous-count 8 --reset-every 0 --pairs --continuous graph_async_serial --profile
```

## d10_async_serial_repeat

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m results.idea2.d10_dynamic_sm_20261024.async_serial_control --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_async_serial_repeat --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective vision_only --warmup 20 --count 100 --continuous-count 100 --reset-every 50 --pairs graph_async_serial:graph_vision_overlap --continuous graph_vision_overlap graph_async_serial --reverse-order
```

## d10_combined_profile

profile / 不作正式计时

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/d10_combined_profile_20261024_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_combined_profile --num-steps 10 --ae-layout whole --warmup 20 --pairs 0 --continuous-count 8 --reset-every 0 --hoist-modulation --profile --modes combined --profile-mode combined
```

## d10_crossframe_formal

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_crossframe_formal --num-steps 10 --ae-layout whole --warmup 20 --pairs 100 --pair-length 16 --continuous-count 100 --reset-every 50 --hoist-modulation --lifecycle-audit --modes fresh_serial lookahead_serial encoder_overlap combined
```

## d10_crossframe_repeat

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_crossframe_repeat --num-steps 10 --ae-layout whole --warmup 20 --pairs 100 --pair-length 16 --continuous-count 100 --reset-every 50 --hoist-modulation --lifecycle-audit --modes combined encoder_overlap lookahead_serial fresh_serial
```

## d10_crossframe_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_crossframe_screen --num-steps 10 --ae-layout whole --warmup 20 --pairs 20 --pair-length 16 --continuous-count 20 --reset-every 10 --hoist-modulation --lifecycle-audit
```

## d10_dynamic16_gate

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output results/idea2/d10_dynamic_sm_20261024/d10_dynamic16_gate --num-steps 10 --boundary kv_ready --group-size 1 --ae-layout per_step --tail-policy full --ae-sm-min 16 --warmup 3 --count 2 --continuous-count 4 --reset-every 2 --hoist-modulation --lifecycle-audit
```

## d10_dynamic16_profile

profile / 不作正式计时

此早期结果只保存模型子命令。重新采集时在命令前加：`nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop -o <新的trace路径>`。

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output results/idea2/d10_dynamic_sm_20261024/d10_dynamic16_profile --num-steps 10 --boundary kv_ready --group-size 1 --ae-layout per_step --tail-policy full --ae-sm-min 16 --warmup 10 --count 0 --continuous-count 8 --reset-every 0 --hoist-modulation --profile --profile-mode overlap
```

## d10_dynamic16_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_dynamic16_screen --num-steps 10 --ae-layout per_step --tail-policy full --ae-sm-min 16 --warmup 20 --count 20 --continuous-count 20 --reset-every 10 --hoist-modulation --lifecycle-audit
```

## d10_dynamic8_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_dynamic8_screen --num-steps 10 --ae-layout per_step --tail-policy full --ae-sm-min 8 --warmup 20 --count 20 --continuous-count 20 --reset-every 10 --hoist-modulation --lifecycle-audit
```

## d10_encoder64_vlmfull_profile_resumed

profile / 不作正式计时

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/d10_encoder64_vlmfull_profile_20261024_v2 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_encoder64_vlmfull_profile_resumed --num-steps 10 --ae-layout whole --warmup 20 --pairs 0 --continuous-count 8 --reset-every 0 --hoist-modulation --profile --ae-sm-min 64 --vlm-full --modes encoder_overlap --profile-mode encoder_overlap
```

## d10_encoder64_vlmfull_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_encoder64_vlmfull_screen --num-steps 10 --ae-sm-min 64 --vlm-full --ae-layout whole --tail-policy native --modes fresh_serial lookahead_serial encoder_overlap --warmup 20 --pairs 20 --pair-length 16 --continuous-count 20 --reset-every 10 --hoist-modulation --lifecycle-audit
```

## d10_fixed16_profile

profile / 不作正式计时

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/d10_fixed16_profile_20261024_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_fixed16_profile --num-steps 10 --warmup 20 --count 0 --continuous-count 8 --reset-every 0 --hoist-modulation --profile --ae-layout per_step --tail-policy wait_prefix --ae-sm-min 16 --profile-mode overlap
```

## d10_fixed16_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_fixed16_screen --num-steps 10 --ae-layout per_step --tail-policy wait_prefix --ae-sm-min 16 --warmup 20 --count 20 --continuous-count 20 --reset-every 10 --hoist-modulation --lifecycle-audit
```

## d10_fixed8_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_fixed8_screen --num-steps 10 --ae-layout per_step --tail-policy wait_prefix --ae-sm-min 8 --warmup 20 --count 20 --continuous-count 20 --reset-every 10 --hoist-modulation --lifecycle-audit
```

## d10_full_overlap_profile

profile / 不作正式计时

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/d10_full_overlap_profile_20261024_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_full_overlap_profile --num-steps 10 --warmup 20 --count 0 --continuous-count 8 --reset-every 0 --hoist-modulation --profile --ae-layout whole --profile-mode overlap
```

## d10_full_serial_profile

profile / 不作正式计时

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/d10_full_serial_profile_20261024_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_full_serial_profile --num-steps 10 --warmup 20 --count 0 --continuous-count 8 --reset-every 0 --hoist-modulation --profile --ae-layout whole --profile-mode serial
```

## d10_oldkv_encoder_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.cewp_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_oldkv_encoder_screen --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --warmup 20 --count 20 --continuous-count 20 --reset-every 10 --selective vision_only --pairs graph_serial:graph_ae_first --continuous graph_serial graph_ae_first
```

## d10_oldkv_formal

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.cewp_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_oldkv_formal --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective all --warmup 20 --count 100 --continuous-count 100 --reset-every 50 --pairs graph_serial:graph_encoder_overlap graph_encoder_overlap:graph_ae_first --continuous graph_serial graph_encoder_overlap graph_ae_first
```

## d10_oldkv_full_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.cewp_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_oldkv_full_screen --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --warmup 20 --count 20 --continuous-count 20 --reset-every 10 --selective all --pairs graph_serial:graph_ae_first graph_serial:graph_prefill_first --continuous graph_serial graph_ae_first graph_prefill_first
```

## d10_oldkv_graph_ae_first_profile

profile / 不作正式计时

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/d10_oldkv_graph_ae_first_20261024_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.cewp_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_oldkv_graph_ae_first_profile --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective all --warmup 20 --count 0 --continuous-count 8 --reset-every 0 --pairs --continuous graph_ae_first --profile
```

## d10_oldkv_graph_encoder_overlap_profile

profile / 不作正式计时

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/d10_oldkv_graph_encoder_overlap_20261024_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.cewp_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_oldkv_graph_encoder_overlap_profile --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective all --warmup 20 --count 0 --continuous-count 8 --reset-every 0 --pairs --continuous graph_encoder_overlap --profile
```

## d10_oldkv_graph_serial_profile

profile / 不作正式计时

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o nsys/idea2_experiment/d10_oldkv_graph_serial_20261024_v1 /root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.cewp_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_oldkv_graph_serial_profile --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective all --warmup 20 --count 0 --continuous-count 8 --reset-every 0 --pairs --continuous graph_serial --profile
```

## d10_oldkv_mixed_gate

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.cewp_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_oldkv_mixed_gate --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective all --pairs graph_serial:graph_encoder_overlap graph_encoder_overlap:graph_ae_first --continuous graph_serial graph_encoder_overlap graph_ae_first --warmup 20 --count 20 --continuous-count 20 --reset-every 10
```

## d10_oldkv_repeat

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.cewp_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_oldkv_repeat --steps 10 --noise-period 13 --await-previous-prefix --hoist-modulation --selective all --warmup 20 --count 100 --continuous-count 100 --reset-every 50 --reverse-order --pairs graph_encoder_overlap:graph_ae_first graph_serial:graph_encoder_overlap --continuous graph_ae_first graph_encoder_overlap graph_serial
```

## d10_slo_33ms_formal

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_slo_33ms_formal --num-steps 10 --ae-layout whole --warmup 20 --pairs 0 --continuous-count 100 --reset-every 50 --hoist-modulation --arrival-period-ms 33 --modes fresh_serial lookahead_serial encoder_overlap combined
```

## d10_slo_33ms_repeat

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_slo_33ms_repeat --num-steps 10 --ae-layout whole --warmup 20 --pairs 0 --continuous-count 100 --reset-every 50 --hoist-modulation --arrival-period-ms 33 --modes combined encoder_overlap lookahead_serial fresh_serial
```

## d10_slo_62ms_formal

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_slo_62ms_formal --num-steps 10 --ae-layout whole --warmup 20 --pairs 0 --continuous-count 100 --reset-every 50 --hoist-modulation --arrival-period-ms 62 --modes fresh_serial lookahead_serial encoder_overlap combined
```

## d10_slo_62ms_repeat_resumed

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_slo_62ms_repeat_resumed --num-steps 10 --ae-layout whole --warmup 20 --pairs 0 --continuous-count 100 --reset-every 50 --hoist-modulation --arrival-period-ms 62 --modes combined encoder_overlap lookahead_serial fresh_serial
```

## d10_slo_71ms_formal

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_slo_71ms_formal --num-steps 10 --ae-layout whole --warmup 20 --pairs 0 --continuous-count 100 --reset-every 50 --hoist-modulation --arrival-period-ms 71 --modes fresh_serial lookahead_serial encoder_overlap combined
```

## d10_slo_71ms_repeat

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_crossframe_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_slo_71ms_repeat --num-steps 10 --ae-layout whole --warmup 20 --pairs 0 --continuous-count 100 --reset-every 50 --hoist-modulation --arrival-period-ms 71 --modes combined encoder_overlap lookahead_serial fresh_serial
```

## d10_split_full_formal_stable

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_split_full_formal_stable --num-steps 10 --warmup 20 --count 100 --continuous-count 100 --reset-every 50 --hoist-modulation --lifecycle-audit --ae-layout per_step --tail-policy wait_prefix
```

## d10_split_full_repeat

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_split_full_repeat --num-steps 10 --warmup 20 --count 100 --continuous-count 100 --reset-every 50 --hoist-modulation --lifecycle-audit --ae-layout per_step --tail-policy wait_prefix --reverse-order
```

## d10_split_full_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output results/idea2/d10_dynamic_sm_20261024/d10_split_full_screen --num-steps 10 --boundary kv_ready --group-size 1 --ae-layout per_step --tail-policy wait_prefix --warmup 20 --count 20 --continuous-count 20 --reset-every 10 --hoist-modulation --lifecycle-audit
```

## d10_whole_formal_stable

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_whole_formal_stable --num-steps 10 --warmup 20 --count 100 --continuous-count 100 --reset-every 50 --hoist-modulation --lifecycle-audit --ae-layout whole
```

## d10_whole_repeat

正式测量（适用范围见报告）

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/d10_whole_repeat --num-steps 10 --warmup 20 --count 100 --continuous-count 100 --reset-every 50 --hoist-modulation --lifecycle-audit --ae-layout whole --reverse-order
```

## d10_whole_screen

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output results/idea2/d10_dynamic_sm_20261024/d10_whole_screen --num-steps 10 --boundary kv_ready --group-size 1 --ae-layout whole --warmup 20 --count 20 --continuous-count 20 --reset-every 10 --hoist-modulation --lifecycle-audit
```

## d2_regression

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -m preparations.idea2.layer_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output results/idea2/d10_dynamic_sm_20261024/d2_regression --num-steps 2 --boundary kv_ready --group-size 1 --ae-layout per_step --warmup 3 --count 2 --continuous-count 4 --reset-every 2 --hoist-modulation --lifecycle-audit
```

## encoder_pair_curves

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -u -m preparations.idea2.sm_sensitivity_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output /root/nfs/embodied/benchmark/results/idea2/d10_dynamic_sm_20261024/encoder_pair_curves --num-steps 10 --warmup 20 --count 30 --ae-sm-mins 64 32 --hoist-modulation --encoder-pair
```

## iteration_audit

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -m preparations.idea2.d10_iteration_audit --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output results/idea2/d10_dynamic_sm_20261024/iteration_audit --hoist-modulation
```

## sm_curves

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -m preparations.idea2.sm_sensitivity_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output results/idea2/d10_dynamic_sm_20261024/sm_curves --num-steps 10 --warmup 20 --count 30 --hoist-modulation
```

## sm_curves8

诊断或初筛

```bash
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -m preparations.idea2.sm_sensitivity_bench --checkpoint shared/checkpoints/openpi/pi05_libero_pytorch_idea2 --gpu-uuid GPU-599477fb-0507-40dc-14df-d491f7a7b202 --output results/idea2/d10_dynamic_sm_20261024/sm_curves8 --num-steps 10 --warmup 20 --count 30 --ae-sm-mins 8 --hoist-modulation
```

## CPU 离线汇总

```bash
policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/summarize_confirmation.py
CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/offline_trace_matrix.py
CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/summarize_traces.py
CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/oldkv/analyze_oldkv.py
```

闭环运行命令与固定案例见 `results/idea2/d10_dynamic_sm_20261024/closedloop_smoke_README.md`；已有执行目录禁止覆盖，重跑指定新的 `--output`。

## 补充公平对照分析与闭环

```bash
CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/oldkv/analyze_async_control.py
policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/analyze_closedloop.py
policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/analyze_fresh_quality.py
policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/closedloop_smoke.py --execute --output <新的旧KV闭环目录>
policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/fresh_baseline_smoke.py --execute --output <新的新KV闭环目录>
```

闭环默认物理GPU和EGL索引是本机本次配置；独占租约及映射必须重新核对。新KV闭环在GPU0，仅比较固定案例质量，不能与GPU1跨卡排名延迟。
