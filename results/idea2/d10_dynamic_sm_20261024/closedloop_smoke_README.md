# D10 旧完整 KV 闭环 smoke

`closedloop_smoke.py` 默认只生成计划和 CPU 包/路径检查，只有显式 `--execute` 才启动模型服务及真实 LIBERO 环境。运行时代码和模型文件未修改。

```bash
# 计划；不会启动 GPU、服务或环境。
policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/closedloop_smoke.py

# 待 GPU1 的正式计时及 nsys 队列结束，由调度 Host 执行。
policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/closedloop_smoke.py --execute
```

默认输出为本目录的 `closedloop_d10_o0_o1/`。默认服务 UUID 是 `GPU-599477fb-0507-40dc-14df-d491f7a7b202`，EGL index 为 `2`，端口 `19082`。EGL index 是这台机器已核对的映射，不是可移植的 GPU 编号。更改端口、设备或顺序时，使用新的 `--output` 生成计划；脚本拒绝覆盖已有 execution。

## 固定条件

- 四套件 `libero_object/spatial/goal/10`，各任务 0、5，均官方 init 0；每模式 8 条，共 16 条。每个案例的 O0/O1 先后顺序交替。
- 同一个常驻 `idea2.server`，`pi05_libero`、D10/H10/R5、AE-first、priority 0、Graph、`vision_only`、`PreparedModulationCore`、`await_previous_prefix`。
- O0 使用 `serial_reuse`，O1 使用 `async_reuse`。服务 `configure` 会排空 Pool 并 reset；同一布局继续保留已有 Graph 和两槽。`vision_only` 不变，只有调度模式切换。
- 原 `runtime_client.run_episode` 直接调用，保持环境 seed 7、官方初始状态、10 次 settling、原 model_seed 和 R5 动作执行规则；不使用未来观测 bank。每 episode 内已有 reset；包装层在成功或异常路径末尾均 drain。

## 记录与判断

原客户端保存 `identity.json`、`native_protocol.json`、`environment_protocol.json`、逐观测 `inputs/*.npz` 及 hash、`chunk*.npz`、`infer.jsonl`、实际执行动作和真实 EEF 的 `telemetry.jsonl` 及 `result.json`。仅在 worker 内将 imageio writer 替换为不保存帧的 sink，保留原 capture、观测、telemetry 和动作调用；不录全量视频。结果显式记录 `video=null` / `video_recorded=false`，不会虚报视频文件。

新增 `smoke_receipt.json` / `complete.json` 保存动作/步数/请求数检查、old-KV age（bootstrap 为 0，此后为 1）、两槽限制、稳态重捕获检查、cold capture 数量、请求延迟分布和含最终 drain 的 wall time。普通任务失败是 `valid / success=false`；技术错误不生成 complete，整组停止并保留日志。

客户端 `CUDA_VISIBLE_DEVICES=''`；仅 EGL 渲染占用已分配 GPU。启动前检查该 UUID 的 compute 和 graphics 空闲；episode 前后要求只有已登记的本服务 compute PID、零 graphics。单客户端运行期间要求 compute 集合不变、至多一个 graphics，并至少观测一次目标卡的 graphics。`nvidia-smi` 使用宿主 PID、`subprocess.pid` 使用容器 PID，二者分开记录，不比较是否相等；归属依据独占 lease 和启动前/运行中/退出后的变化。若 90 秒内未在目标 UUID 看到 EGL context，停止并要求核查映射。清理只操作自己的 subprocess 对象，不向观测到的宿主 PID 发信号。

`plan.json` 记录命令、环境、源文件 SHA256 和 checkpoint 文件元数据；执行前及边界核查源码，结束后再次核查。权重未重新全量 hash；噪声身份保存原 seed、FP32 CUDA `randn` 构造规则及代码/框架身份，没有声称记录服务未暴露的 noise tensor 字节 hash。

## 已完成的 CPU 检查与限制

10 项 CPU 编排测试通过，覆盖 PID 命名空间与禁视频 sink，见 `closedloop_smoke_cpu.log`；两套 Python 的包、checkpoint 文件和 LIBERO 配置路径检查通过，见 `closedloop_smoke_plan_cpu.log`。仅做包发现，不初始化 MuJoCo/EGL，实际驱动与资产载入仍须执行时验证。

这是每模式 8 条探索性闭环 smoke，不是官方全量 LIBERO 评测。自由闭环后两模式输入可以分叉，不应要求后续输入 hash 或动作逐位相同。LIBERO 会等待推理完成，不能据此宣称真实机器人频率或相机 SLO。请求延迟含渲染竞争；episode wall 还含环境创建、动作执行和 I/O。它们用于 smoke 分析，不能替代已隔离的性能计时。
