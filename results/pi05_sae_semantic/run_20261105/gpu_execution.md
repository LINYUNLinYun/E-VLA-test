# 小批闭环续跑入口

离线结果、相对影响先验与 24 条小批案例已冻结。`quality_pilot_manifest.json` 是当前配置，`quality_pilot_manifest.prevalidation.json` 是尚未执行过的格式草稿。

把 `SEMANTIC_GPU_INDEX` 设为本轮可用卡的物理编号。模型进程使用映射后的 `cuda:0`；环境继续使用现有 LIBERO Python 与评测器。以下命令在项目根目录执行。

```bash
export SEMANTIC_RUN=results/pi05_sae_semantic/run_20261105
export PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src:/root/nfs/embodied/benchmark/policies/openpi/packages/openpi-client/src:/root/nfs/embodied/benchmark/LIBERO
export CUDA_VISIBLE_DEVICES="$SEMANTIC_GPU_INDEX"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export LIBERO_CONFIG_PATH=/root/nfs/embodied/benchmark/policies/realtime-vla-flash/.libero

policies/openpi/.venv/bin/python -u -m experiments.pi05_sae_semantic.check_runtime \
  --candidate "$SEMANTIC_RUN/quality_candidate_relative_prior.json" \
  --collection "$SEMANTIC_RUN/local_readouts_cpu" \
  --preflight "$SEMANTIC_RUN/native_preflight_cpu" \
  --assets shared/caches/pi05_sae_semantic/action_atlas_pi05 \
  --output "$SEMANTIC_RUN/runtime_check_gpu" --device cuda:0 --threads 4
```

只读输出与原路径、在线 mask 与直接指定同一 mask，都在同一设备上比较。CPU 捕获仅作跨设备诊断，不能要求 CPU/GPU 逐位相同；实际同设备 no-op 仍要求逐位一致，失败时定位原因，不放宽误差来通过。

检查通过后启动模型服务，保持进程运行：

```bash
policies/openpi/.venv/bin/python -u -m experiments.pi05_sae_semantic.quality_service \
  --manifest "$SEMANTIC_RUN/quality_pilot_manifest.json" \
  --runtime-check "$SEMANTIC_RUN/runtime_check_gpu/runtime_check.json" \
  --port 17605 --device cuda:0 --threads 4 \
  --output "$SEMANTIC_RUN/quality_service_gpu"
```

在继承同样环境变量的另一个进程执行原有 LIBERO runner：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$SEMANTIC_GPU_INDEX"
policies/libero-eval-v21/.venv/bin/python -u -m experiments.pi05_sae_semantic.quality_eval run \
  --manifest "$SEMANTIC_RUN/quality_pilot_manifest.json" \
  --port 17605 --output "$SEMANTIC_RUN/quality_pilot"
```

按任务轮换三个方法的执行次序，每条 episode 显式重置路由计数。原生失败轨迹记为有效失败，不重试到成功；技术中断保留现场，定位后另存技术失败目录再续跑。已有同一 manifest 的完整轨迹自动复用。

每条轨迹保存 `result.json`、`identity.json`、`infer.jsonl`、`telemetry.jsonl`；`quality_pilot/summary.json` 持续汇总成功数、实际跳过数与 policy 时延。这里的闭环时延来自逐渐分叉的观测，GPU 固定输入性能比较仍需单独报告。
