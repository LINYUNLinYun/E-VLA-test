# π0.5 固定 Batch 流水线实验方案与代码准备

日期：2026-10-31。状态：代码准备和 CPU 检查；**尚未运行任何 B>1 GPU 实验，无新性能结论**。本轮不占卡、不排队等卡、不启动自动监控。后续拿到独占 GPU 后，手动选择已生成的命令执行。

## 要回答的问题

增加 batch 是否提高独立观测请求的总吞吐？在同一 batch 下，流水线是否仍比串行有效？最优 AE SM 配额和 VLM 领先层数是否随 batch 改变？

三个效应需要分别验证：合批可能提高小矩阵计算及权重复用效率；AE/Encoder/VLM 耗时比例变化可能改善隐藏窗口；更大的 GEMM 也可能加重资源竞争。不能预设 B1 的 AE16/Prefix112、lead3 在 B2/B4 下仍最优。

这里 batch 是多个独立机器人/仿真环境的独立观测，不能把同一机器人的未来观测或 H10 动作序列当作 batch。固定轨迹重放只能回答模型/调度性能和数值一致性，不能证明真实多机器人闭环成功率。

## 固定条件与最小矩阵

沿用完整 `pi05_libero` checkpoint、原精度、D10、H10、原 FP32 Euler 时间递推。保持完整模型计算、两个 KV Slot、Lease、epoch/generation 和 Event 协议，不引入第三份完整 KV，也不减少去噪步数。

| 维度 | 首轮配置 | 目的 |
|---|---|---|
| Batch | 1、2、4 | 找到小 batch 的收益与竞争变化 |
| 串行 | 全卡、全新 KV，VLM 完成后启动 AE | 同 batch 的基础对照 |
| 普通流水线 | 全卡共享，lead3，逐轮冻结新 KV frontier；未就绪层用完整旧 KV | 测普通并发收益 |
| SM 隔离流水线 | 同上，AE 请求 16 SM、Prefix 使用独立余下 SM | 对照此前最优配置；记录硬件实际分配 |
| 输入口径 | `prepared` 与 `online` 分开 | 分离模型调度与输入准备的影响 |

全卡串行和普通流水线在同一个运行进程内按 AB/BA 交替，复用同一套 Graph。Green Context 模式单独进程，避免多套捕获同时占显存；它与全卡结果属于跨进程对照，不能标成严格同进程配对。关键候选后续反向进程顺序复测。

首轮仅使用 AE16、lead3。出现有希望的 B 后，再针对该 B 单独测 AE16/32/64 和 lead0/1/3/6；每次只改一个变量，不直接铺满大矩阵。B8 及动态合批不属于本轮。

## 输入、状态和正确性

1. 从既有真实输入 bank 选同任务、固定 mask/layout 的记录，使用 B1/B2/B4 共用且能被 4 整除的样本子集，形成固定 lane 顺序；每个时刻各 lane 使用不同记录，随时刻切换输入。原 bank 同任务 13 条时统一用前 12 条，余 1 条明确记录为未采用。seed 只由原始记录身份与基准 seed 决定，不随 B、lane 或执行顺序变化。
2. 每个 lane 调用原来的输入 transform；CPU 上校验所有叶子的 shape/dtype 后再合批搬运。逐 lane 调用原 `noise(seed)` 后拼接，避免一次 B 维随机采样改变每个请求的噪声。
3. 固定 lane 成员和顺序。首轮整批同步推进/整批 reset；独立机器人 reset、动态换 lane 暂不支持。Prepared 绑定只读 `(lane_ids, epoch)`，旧 episode 的 Prepared 不可重新贴标签使用。
4. Graph 按固定 B、布局和槽地址捕获。batch 改变使用独立运行进程，禁止复用 B1 的静态 Graph 输入/输出。AE 的两图仍串行使用一份单层 scratch，VLM 与 AE 不共享可写 scratch。
5. 先在 Graph 创建前逐 lane 计算 B1 原始 eager 全新 KV oracle，只保留 CPU 动作；临时 KV 释放后才能建立两槽 Graph。全新 KV 的 batch 串行结果对照该 oracle；混合 KV 的流水线结果对照**相同输入、噪声和已观察 frontier 选择日程的串行结果**，不拿新旧 KV 差异冒充调度误差。
6. 所有 lane 分别检查 normalized 和 decoded 动作，预定容差 MAE ≤ 1e-6、max ≤ 1e-5，非有限值直接失败。B 改变可能触发不同 GEMM 路径并出现舍入差异；失败必须保留，不为跑通而放宽阈值。另报混合结果相对全新/全旧 KV 的差异，此项不作为相同日程数值 gate。
7. 数值验证与计时分开。默认每条件 4 个独立验证请求，覆盖两槽轮换、输入变化和整批 reset；全部计时样本保存 normalized/decoded 动作和 frontier。不能把有限验证请求写成所有闭环场景均无损。

## 两种计时边界

`prepared`：相同真实记录已完成输入准备，计时中仍包含静态输入 staging、Graph replay 和输出拷回。可使用下一批已准备好的输入作为 Encoder lookahead。**这是预准备输入的机制/容量测试，不能解释成实时获取了下一帧。**

`online`：在请求序列内逐批执行完整 prepare；串行和流水线均允许相同的 CPU lookahead，只有流水线允许下一批 Encoder 在 AE 尾段执行。prepare 包含 CPU transform、上传及模型观察预处理，不能称为纯 CPU 时间。这里仍是录制数据重放，不模拟相机真实到达/动作执行依赖。

两口径分开出表，不把 prepared 延迟和 online 的端到端延迟直接相除。无真实时间戳的 bank 不报告毫秒观测年龄；仅报告 KV 请求代龄及逐轮新 KV 层数。后续闭环才测 capture→action 和真实观测年龄。

## 后续执行顺序

1. **CPU 准备**：输入/输出样本隔离、确定性 seed、错误 shape、旧 epoch、lane 重排、Graph 布局拒绝、Pool/Lease、租约竞争、命令 dry-run。全程 CUDA 未初始化。
2. **GPU smoke**：先 B1，再 B2/B4；每模式短序列，仅检查模型可执行、显存、全 lane 数值 gate、两槽轮换、原地内容更新和 reset。失败不继续对应性能条件，不重试有效数值失败。
3. **无 profiler 性能**：预热/捕获不计入稳态。全卡两 arm 至少 100 对 AB/BA，每 arm 每组 8 请求；Green 单独同规模运行，明确跨进程比较。组后 drain，再读取 Event 并写盘；不要逐阶段 synchronize。
4. **代表性 NSYS**：只对有代表性的 B/条件另采短 trace。真实 replay 外层仅 `preprocess`、`encoder`、`vlm`、`AE`，同时保留每个 Kernel。使用 `node:host-only:nvtx-live`，禁止再次用 capture-time 名称代替完整运行范围。实际 kernel 区间交集决定是否重叠，NVTX/Event 包围区间重叠只能作线索。
5. **有条件调优**：发现 AE16 限制 batch 后再增配额；发现 VLM 干扰明显再调 lead。先单变量，再复测最优组合。最后才做多环境闭环，报告每机器人 SLO、动作时序/观测年龄及成功率。

## 输出和判定

每行保留 B、资源模式、输入口径、输入/seed/lane 身份。输出 JSON、CSV 和动作 NPZ；性能与 profile 结果分开，异常/OOM/数值失败也保留。

- 总吞吐 = `B × 完成请求批次数 / 整段实际墙钟`。分母覆盖尾部 Prefill/Encoder 完成，不能只看 AE 返回或把阶段耗时相加。
- 同 B 的流水线收益 = 流水线吞吐 / 串行吞吐；合批收益 = 该 B 串行吞吐 / B1 串行吞吐。两者分别报告。
- 报 CPU→CPU mean/P50/P95、输入准备、AE 输出可用时间、Prefill 完成时间、两分支/整段完成时间、Encoder/VLM/AE Event 包围区间、峰值显存、实际 SM 分配、稳态重捕获次数。
- 报逐轮新 KV 层数和被消费 KV 的请求代龄，禁止用降低 KV 更新频率换取表面吞吐。
- 同 B 并发中各模块时长对比串行，检查是否因资源竞争变慢。用 nsys 核对 Kernel 实际交集和同步等待。
- 只有数值 gate 通过、总完成吞吐提高且延迟满足后续 SLO 的配置才作为候选。吞吐上升但单请求变慢要明确列出；无收益和退化保留。

## 代码入口与复现

- `preparations/idea2/batch_backend.py`：固定 lane 输入/噪声/输出与 epoch 合约。
- `preparations/idea2/batch_runtime.py`：完整 B 维输出、逐 lane 数值摘要、局部 replay 标签、全卡/Green stream 选择。
- `preparations/idea2/batch_bench.py`：离线数值 gate、AB/BA、prepared/online、结果保存；`--help` 和 `--dry-run` 不访问 GPU。
- `preparations/idea2/batch_resources.py`：后续显式执行时检查共享租约和真实空闲状态；忙卡立即退出，不抢卡、不终止别人的进程。
- `preparations/idea2/batch_plan.py`：纯 CPU 生成分阶段命令和 manifest，永远不启动命令。

生成命令（现在即可执行，不查询 GPU）：

```bash
python3 -m preparations.idea2.batch_plan \
  --output results/idea2/batch_preparation_20261031/experiment_plan
```

`commands.md` 中保留待分配 GPU UUID 占位符；确认有独占卡后替换，按阶段手动执行。新 output 目录必须不存在，避免覆盖已有记录。

单个后续运行入口示例（**此处没有执行**）：

```bash
env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH="$PWD:$PWD/policies/openpi/src" \
  policies/openpi/.venv/bin/python -m preparations.idea2.batch_bench \
  --gpu-uuid GPU-REPLACE-WITH-ALLOCATED-UUID \
  --batch-size 2 --mode paired --input-mode prepared \
  --pairs 100 --length 8 --warmup 20 --verify-count 4 \
  --output results/idea2/batch_b2_fullgpu_prepared
```

GPU 执行后才填充性能表；本次交付不报告未经测量的 batch 加速比。

## 本次准备结果

联合 CPU 回归 **81/81 通过**，覆盖新增 batch 接口/基准/租约/计划，以及原有 Pool、Kernel 接口和 Graph 兼容检查；测试进程 CUDA 未初始化。SM 隔离桥接库仅用 CPU 编译通过。已生成 **36 条待执行命令**：6 条 smoke、12 条无 profiler 性能、18 条可选 profile；全部通过参数解析，全部未启动。profile 命令是候选清单，后续只选代表条件采集。

这些 CPU 检查验证接口和调度协议，**不等于完整模型在 B2/B4 下已通过 GPU 数值或性能验证**。GPU 捕获、真实数值容差、峰值显存和 Kernel 重叠均等待独占卡后验证。

准备记录：

- `results/idea2/batch_preparation_20261031/cpu_validation.json`：测试数量、CUDA 未初始化和源码摘要。
- `results/idea2/batch_preparation_20261031/cpu_checks.log`：完整 CPU 检查日志。
- `results/idea2/batch_preparation_20261031/input_schedule_preview.json`：真实 bank 的 B1/B2/B4 公共样本/噪声分组。
- `results/idea2/batch_preparation_20261031/experiment_plan/plan.json`：源码冻结及待运行矩阵。
- `results/idea2/batch_preparation_20261031/experiment_plan/commands.md`：直接复制的分阶段命令。

本轮新增 5 个独立 batch 模块与相应 CPU 检查。原 `progressive_kv.py` 仅增加 stream/Graph 工厂、输出下载和验证摘要四个兼容扩展点，默认 B1 行为保留；没有改模型计算、旧 KV 选择算法或 Pool/Lease/Event 生命周期。
