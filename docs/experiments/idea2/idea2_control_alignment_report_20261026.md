# π0.5 控制时序修正：同步执行与动作前缀补偿

状态：32/32 条小批实验全部完成，三次服务均正常退出，GPU3 已释放。本记录独立于此前已完成的实验，不覆盖旧数据。

**同步参考和同步 SM16 流水线均为 8/8；完整异步流水线未补偿为 6/8，跳过过期动作前缀后为 8/8。补偿修正了动作时间索引，GPU 计算耗时基本不变。这是积极的小批结果，不能据此宣称总体无损。**

## 问题与两条路线

此前 continuous300 在执行上一组 R5 动作时推理下一组，这改变了原同步 LIBERO 控制协议。完整组合首动作的输入平均落后 2.78 个已执行控制步，却仍使用预测索引 0。不能把其 84/100 与旧同步 96/100 的差异全部归因于 GPU 并行或混合 KV。

1. **恢复同步执行**：执行完上一组动作后采观测，再计算下一组；计算期间仿真不步进。模型内部保留 lead3、AE16/VLM112 SM、CUDA Graph/Event 与双槽 KV。未来动作后的下一帧尚不存在，因此关闭下一帧 Encoder 提前执行。使用未改动的原 `runtime_client.run_episode`，同步 fresh18 与同步 SM16 lead3 配对。
2. **动作前缀补偿**：保留原 continuous 的全部 GPU/采观测/候选选择规则，只在 R5 交接时按 `d = 当前已执行控制步 - 输入观测控制步` 取 `actions[d:d+5]`；未补偿对照继续取前五步。D10/H10/R5、权重、噪声规则不变。若余下 H10 不足本次 R5，停止并记为 `prediction_horizon_exhausted`，保留在分母；不补零、不延长、不重试有效失败。最后的原生动作上限可以截短 R5。

补偿按仿真实际控制步计算，而非把 RPC 墙钟除以 50ms。暂停、准备和候选交接也影响墙钟时效，单独报告。前缀跳过只修正预测时间索引，不能补偿执行旧动作造成的状态偏差；本次不引入 RTC/inpainting 或其他模型改动。

## 预定小批矩阵与采用边界

四套件 `libero_object/spatial/goal/10`，各 task0、5，init0，共固定八场景；环境 seed7，settling10，原生动作上限 280/220/300/520。四条件各八条，共 32 条：同步 fresh18、同步 SM16 lead3、异步 SM16+Encoder 原始前五步、同路径动作前缀补偿。同步控制和异步控制的耗时不合并为 GPU 加速比。

同步两条件复用原 quality runner。异步两条件在同一张 GPU 上按任务轮换顺序；每次服务最多四个任务。执行前检查独占和租约，先使用核实空闲的 GPU3，不碰其他会话的 GPU。有效失败、预测时域不足均保留；技术中断单列，不覆盖旧目录。先检查门禁是否正常、动作索引是否与原始模型输出逐项相同，再判断是否值得扩大；八场景不能证明总体无损。

原同步 200 例质量结果可作为独立既有证据：fresh18 191/200，SM16 193/200；它关闭下一 Encoder，不能代表完整组合。同协议的本次八例用于重新验证接线与计时，不以挑选已成功案例替代后续扩样。

## 实现与复现

GPU 模型、原 runtime、Pool/Lease/epoch/generation/Event 保持原源码。新增客户端在真实 R5 交接时选取原输出的后续动作，写出跳过步数、真实索引、剩余时域和对齐残差；`off` 是同一客户端的原始前五步对照。新增服务端只增加对齐协议元数据。

```bash
# 只做 CPU 状态机核验，不启动 CUDA/EGL
python3 -m results.idea2.control_alignment_20261026.cpu_selftest

# 同步门禁的实际执行命令保存在 sync_gate/plan.json
python3 -m results.idea2.fresh_tail_20261025.quality --execute \
  --configs fresh18 sm16_tail0 --cases 0 1 2 3 4 5 6 7 \
  --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 \
  --egl-device 0 --cpus 96-111 --port 19630 \
  --output results/idea2/control_alignment_20261026/sync_gate
```

异步入口 `results/idea2/control_alignment_20261026/run_async.py`；完整命令、环境与前后源码摘要写入新输出目录的 `execution.json`。以上历史端口和 GPU 归属不能直接用于今后的重跑；先确认资源并换新目录。性能门禁不开 nsys，避免把 profiler 干扰加入控制时序。

## 小批结果

以下四行均为同八个初始场景。首动作年龄不包括 bootstrap；同步组的控制步年龄由原执行顺序及逐动作核验确定，不能理解成推理的墙钟耗时为零。

| 条件 | 成功数 | 总执行步数 | 首动作输入年龄，控制步均值 | 动作时间索引残差，控制步均值 | 预测时域不足 |
|---|---:|---:|---:|---:|---:|
| 同步全新 KV 参考 | 8/8 | 1,181 | 0 | 0 | 0 |
| 同步 SM16 lead3 流水线 | 8/8 | 1,220 | 0 | 0 | 0 |
| 异步 SM16 + 下一 Encoder，未补偿 | 6/8 | 1,807 | 3.02 | 3.02 | 0 |
| 同路径，动作前缀补偿 | 8/8 | 1,175 | 2.73 | 0 | 0 |

补偿组仍然使用有延迟的观测：首动作输入平均旧 162.91ms、P95 183.22ms。它在 230 个非 bootstrap 动作块中分别跳过 2/3/4 步共 65/163/2 次，从对应预测索引执行，而非让观测变新。未补偿组平均旧 175.96ms、P95 277.14ms。两组自由闭环的轨迹、候选交接相位和请求数量不同，不能把年龄均值差也全部归因于动作索引修改。

| 场景（均 init0） | 同步参考 | 同步 SM16 | 异步未补偿 | 异步补偿 |
|---|---:|---:|---:|---:|
| Object task0 | 成功 /154步 | 成功 /156步 | 成功 /219步 | 成功 /138步 |
| Object task5 | 成功 /130步 | 成功 /124步 | 成功 /143步 | 成功 /124步 |
| Spatial task0 | 成功 /110步 | 成功 /84步 | 成功 /196步 | 成功 /82步 |
| Spatial task5 | 成功 /101步 | 成功 /96步 | 失败 /220步 | 成功 /90步 |
| Goal task0 | 成功 /114步 | 成功 /128步 | 失败 /300步 | 成功 /116步 |
| Goal task5 | 成功 /129步 | 成功 /137步 | 成功 /158步 | 成功 /135步 |
| Long task0 | 成功 /262步 | 成功 /289步 | 成功 /336步 | 成功 /313步 |
| Long task5 | 成功 /181步 | 成功 /206步 | 成功 /235步 | 成功 /177步 |

补偿相对未补偿救回 2 例、损失 0 例；有效失败原样保留，没有重试到成功。Long task0 补偿后仍比同步参考多执行 51 步，因此不能声称两种控制方式已经等价。总步数降低也受失败回合达到上限影响，不能直接换算为模型推理加速比。

### 耗时与并行窗口

稳态 RPC 排除 bootstrap/Graph 准备；异步还排除重复初始观测的首个在线请求。episode 工作墙钟包含准备和记录，且同步仿真没有墙钟 20Hz 节拍限制，不能跨控制协议当作真实机器人任务加速比。

| 条件 | 稳态 RPC mean/P50/P95 ms | episode 工作 mean/P50/P95 s |
|---|---:|---:|
| 同步参考 | 107.65 /106.74 /114.66 | 18.83 /17.11 /25.71 |
| 同步 SM16 | 101.86 /101.37 /105.74 | 17.98 /15.94 /25.29 |
| 异步未补偿 | 60.94 /57.17 /89.01 | 28.16 /27.41 /36.98 |
| 异步补偿 | 57.35 /57.12 /58.57 | 22.75 /20.42 /34.64 |

同步 SM16 本批 RPC 均值少约 5.4%，不是巨大加速。异步两条件 RPC 的 P50 几乎相同，不能因均值和尾延迟不同就称动作前缀跳过加速了 GPU。

| GPU Event 指标，均值 ms | 异步未补偿 | 异步补偿 |
|---|---:|---:|
| Vision Encoder | 11.18 | 11.25 |
| VLM | 38.85 | 38.90 |
| AE 十轮 | 49.21 | 49.50 |
| AE 与本帧 VLM 的 Event 包围区间交集 | 34.43 | 34.47 |
| AE 与下一 Encoder 的 Event 包围区间交集 | 9.88 | 11.16 |

补偿组保留了原 GPU 并行窗口，没有靠减少 AE 迭代获得结果。同步参考的 AE 十轮约 27.12ms，同步 SM16 约 49.82ms，SM 分区仍会减慢 AE 自身。本轮没有重新采 nsys，以上只表示 Event 区间；此前同一模型运行时的真实 Kernel 重叠证据仍见 [独立 trace](../../../nsys/idea2_experiment/online_sm16_prepared_witness_20261025_v1.nsys-rep)，不能将其数值冒充本轮测量。

## 核验与取舍

32 个唯一回合、3,013 次预测、5,383 个实际动作完成核验。跨四条件的同场景 bootstrap 输入 hash、噪声 seed 一致；24 组配对分别校验 decoded/normalized 输出，共 48 项 MAE/max 检查，均为 0，沿用 MAE≤1e-6、max≤1e-5 的容差。后续自由闭环轨迹会分叉，这不代表后续动作在各条件之间逐项相同。

补偿组每个实际动作都对应原模型输出的声明索引；未重写动作值。逐请求检查了两 KV Slot 和实际 16/112 SM 分区；执行期间源码一致，稳态 Graph 重捕获 0 次。原客户端 14 项 CPU 状态机/真实 socket 测试及新索引处理 5 项 CPU 测试通过。

**恢复同步执行已经能保留 VLM×AE 内部流水线，并避免额外的控制步观测滞后；下一 Encoder 提前执行不能在这个因果条件下保留。动作前缀补偿则保留完整异步 GPU 路径，本批修复了两个失败，但它是控制策略修正，仍需扩样。**

后续最有价值的是在固定 100 个场景中配对比较异步未补偿/补偿，重点看夹爪时序、长任务和输入延迟较大的回合。同步路线另有原 200 例证据；本次不把不同控制协议的数据合并，也未开始新的 100 例队列。

## 产物

- [汇总 JSON](../../../results/idea2/control_alignment_20261026/analysis/summary.json)、[逐回合 CSV](../../../results/idea2/control_alignment_20261026/analysis/episodes.csv)、[配对 CSV](../../../results/idea2/control_alignment_20261026/analysis/paired.csv)。
- [首动作年龄/补偿索引 CSV](../../../results/idea2/control_alignment_20261026/analysis/first_actions.csv)、[逐动作 CSV](../../../results/idea2/control_alignment_20261026/analysis/applications.csv)、[数值核验 CSV](../../../results/idea2/control_alignment_20261026/analysis/bootstrap.csv)。
- [客户端实现](../../../results/idea2/control_alignment_20261026/client.py)、[协议元数据服务](../../../results/idea2/control_alignment_20261026/server.py)、[实验编排](../../../results/idea2/control_alignment_20261026/dispatch_execution.json)、[CPU 检查](../../../results/idea2/control_alignment_20261026/cpu_selftest.json)。
- [同步执行记录](../../../results/idea2/control_alignment_20261026/sync_gate/execution.json)、[Object/Spatial 异步记录](../../../results/idea2/control_alignment_20261026/async_gate0/execution.json)、[Goal/Long 异步记录](../../../results/idea2/control_alignment_20261026/async_gate1/execution.json)。

模型和原 runtime 源码未改。测量完成后仅将新 launcher 的环境变量日志改为实验字段白名单，原测量版本保存在 [source archive](../../../results/idea2/control_alignment_20261026/source_archive.json)；没有修改实验结果、GPU 计算或动作选择代码。

纯 CPU 重建汇总：

```bash
python3 -m results.idea2.control_alignment_20261026.summarize
```
