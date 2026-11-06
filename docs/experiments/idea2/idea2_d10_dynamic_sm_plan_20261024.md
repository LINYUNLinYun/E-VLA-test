# π0.5 idea2：十轮 AE 与按阶段切换 SM 的实验方案

> 当前状态：用户再次明确授权自主推进至实验结束。已恢复仅含待完成项目的队列；保留暂停中断记录，不覆盖既有有效结果。

日期：2026-10-24。本文最初为供用户审阅的方案；用户随后明确授权开始推进实验及协调另一会话交卡。**当前执行状态与结果见[执行记录](idea2_d10_dynamic_sm_status_20261024.md)**；本文保留实验设计，不将计划步骤当成已完成结果。

## 1. 要回答的问题

1. AE 恢复十轮后，普通双流、逐层 KV-ready 和 Encoder 重叠各有多少净收益？
2. 固定 Green Context 配额的损失，有多少来自另一分支结束后仍不能使用空闲 SM？
3. “重叠期分区、独占期全卡”能否改善实际请求/连续输出，而不恶化 KV 时效和帧延迟？

**先做最小两阶段切换，再决定是否扩大调度粒度。** 不预设 Green Context 必定有效，不直接开发在线学习调度器。

历史 D2 的普通双流模型跨度约 45.91 ms，AE16/Prefix112、connections32 约 52.12 ms，来自[已有报告](idea2_layer_pipeline_20261024.md)。这些是设计依据，不充当 D10 的新对照。十轮并不意味着本帧 VLM 可与十轮 AE 全程重叠；同帧逐层窗口主要仍在 AE1，后续轮次扩大的是下一帧 Encoder 或旧 KV 跨观测路线的重叠机会。

## 2. 固定边界

| 项目 | 本轮约定 |
|---|---|
| 模型 | 完整 π0.5 `pi05_libero`，相同 checkpoint、BF16、H=10、action_dim=32；不使用 idea1 V14 |
| 去噪 | 主实验 D10；D2 仅保留必要回归和对照，不重跑历史大矩阵 |
| 时间日程 | 沿用原 GPU FP32 `dt=-1/10` 和逐次时间更新；不通过 Python 小数、linspace 或重复 AE1 Graph 改写日程 |
| 优化基座 | 保留当前已通过验证的显式 Graph/固定条件准备；各比较臂配置相同，不同时增加 compile 或算子融合 |
| 运行方式 | 一个调度 Host；预建分区与全卡 stream/context，阶段内最多两条计算分支；不引入两个 Python 进程 |
| 资源协议 | 双完整 KV Slot、Lease、epoch、generation、Event 不变；Pool 只由调度 Host 修改 |
| 不做 | Draft/Verify、跳层、残差/token 复用、改变权重/精度/动作数量、额外减少去噪、隐藏第三份完整 KV |

两种语义必须分别对照和排名：

- **新路线 A/B：当前帧 KV。** A 是本帧 VLM 逐层 KV-ready → AE1；B 是当前 AE 剩余轮次 × 下一帧 Encoder。下一帧 VLM 仍等待当前 AE 最终完成。
- **旧路线 O：完整旧 KV。** AE 使用上一份已完成快照，另一槽构建新 KV。数值参考必须使用相同旧 KV 和当前 state/noise，不能与新 KV 输出比较后解释成计算误差。

## 3. 分阶段实验

### P0：十轮支持与正确性门禁

当前 `LayerGraphs` 明确限定 D2，现有两轮 Graph 分割器不能直接当成十轮。先把指定切点的分段捕获、最终完成 Event、计时字段和参考输出扩展为 D10，保留 D2 回归。**先采用 1+9 两段，不先拆成十张 Graph。**

复用已有 13 个交替观测及与观测绑定的显式噪声，冻结输入 bank/hash。对每个样本检查：

- eager D10、整段 Graph D10、分段 Graph D10 的原始/解码动作；沿用 **max_abs ≤ 1e-5、MAE ≤ 1e-6**，同时记录逐位一致性，失败后不放宽门槛。
- 恰好十次 velocity/Euler 更新；各轮时间 Tensor 值与参考逐位一致；诊断运行保存每轮中间动作和切点状态，不计入性能样本。
- 每轮读取同一 KV identity；两槽轮换、同地址内容更新、episode reset、在途 reader/producer 均覆盖。
- 同时存活的完整 KV allocation 始终只有两份；各配额 Graph 引用同一双槽实际地址。每个配额不得各自实例化一套带两槽的 `LayerGraphs`。
- 并发 Graph 的可写 scratch/静态输出私有；阶段边界允许小型 action/state 交接，费用计入，禁止拷贝整份 KV 来绕开协议。

仅在数值与生命周期门禁通过后进入性能筛选。若已有准备优化在 D10 超差，恢复匹配的原计算路径并对所有臂一致应用，不把失败优化当基线。

### P1：建立 D10 对照，先分清收益来源

| 编号 | 配置 | 作用 |
|---|---|---|
| F0 | 当前帧、整段 Graph、串行，无 Host lookahead | 常规完整路径基线 |
| F1 | 相同 Host lookahead，GPU 仍串行 | 分离 CPU 提前准备的贡献 |
| F2 | 普通双流：剩余 AE × 下一帧 Encoder | 保留已有 Encoder 路线 |
| F3 | 普通双流：本帧逐层 KV-ready → AE1，不提前跑下一帧 Encoder | 单独测逐层路线 |
| F4 | F3 + 下一帧 Encoder 重叠 | 两条新路线组合 |
| O0 | 相同完整旧 KV 的串行 AE/Prefill | 旧语义数值与速度参考 |
| O1 | 旧 KV AE × 新 Encoder，之后新 VLM | 保留此前较优旧路线 |
| O2 | 旧 KV AE × 完整新 Prefill | 复查 D10 全重叠；保留新 KV 更新和消费约束 |

O2 小批比较 `ae_first/prefill_first`，冻结一个顺序后再正式验证。O 路线每次接收观测都构建新 Prefix；消费下一请求前遵守上一轮发布完成规则，禁止通过降低更新频率或重复消费更老快照获胜。

初筛每项 20 对；只有保留的候选与必要基线进入后述正式测量。不能将初筛数字写成最终性能结论。

### P2：SM 敏感度与候选缩减

在本轮实际 GPU 上查询 SM 总量/分配粒度，记录请求配额和实际配额。4090/128 SM 情况下，先测 AE 在 16/32/64/128 SM 的单步和完整 D10；VLM、Encoder 测互补的 112/96/64 SM 与全卡。所有独立测量使用已就绪输入/KV，不混入依赖等待。

输出各模块的 `T(SM)`、相对全卡减速、CPU 提交和显存；再对至多两个分区候选做短并发验证。独立曲线只能筛选候选，不能证明并发收益。历史 16/112 可以保留一个复核点，但不先认定它最优。

### P3：核心实验——AE1 分区，AE2–10 全卡

先关闭下一帧 Encoder 重叠，仅检验资源回收。三个预建 stream 的依赖如下：

```text
Prefix 分区： VLM → layer_ready[0..17] → prefix_done
AE 分区：     逐层等待 KV → AE1 → ae1_done
AE 全卡：     wait(prefix_done, ae1_done) → AE2…10 → ae_done
```

Host 一次排队，等待发生在 GPU；不逐轮 query 决策，不逐层 CPU synchronize。最后一层 KV-ready 不代表 VLM 尾部已经结束，因此全卡尾段必须同时等待两个完成 Event。Lease 的 reader_done 覆盖最后一个全卡阶段，不能在 AE1 结束就释放。

CUDA Graph 的节点执行资源由创建/捕获时 context 决定，**更换 replay stream 不会把已有 Graph 迁移至另一配额**。需要冷启动时预捕获阶段变体，并验证节点 context；稳态不创建 context、不重新捕获。

不可缺少的四组对照：

| 组 | AE Graph 与资源 | 能回答什么 |
|---|---|---|
| G0 | 原整段 D10，全卡普通双流 | 现有实现的实际性能 |
| G1 | 1+9，同阶段 Event，全卡 → 全卡 | 分图及额外等待本身的成本 |
| G2 | 1+9，同阶段 Event，分区 → 原分区 | 固定分区尾段的成本 |
| G3 | 1+9，同阶段 Event，分区 → 全卡 | 动态回收收益及总净收益 |

G1/G2/G3 的 Prefix 配额另作明确字段：G1 为全卡，G2/G3 为同一分区。G3 对 G2 隔离“尾段回收”的效果；G3 对 G1/G0 判断是否真的优于不限制资源的实现，不能只战胜被限速的对照。所有阶段等待、action staging 和额外提交都纳入计时。

P3 最多验证两个配额。即使固定 G2 较差，也保留 G3 的小批验证，因为 G3 正是用来消除固定配额尾部损失的。

### P4：再接入 Encoder 与旧 KV 路线

只有 P3 或独立阶段配对显示明确机会时，才增加组合。先分别测“剩余 AE × 下一帧 Encoder”在普通双流、固定分区和分区后恢复全卡下的差别，再与逐层路线组合。

- 若下一帧 Encoder 在 AE 中段结束，可在预先冻结的 AE 迭代切点恢复全卡。按 P2 的耗时估计只选择一个切点及至多一个相邻点；不遍历所有层/迭代组合。
- 先用已准备输入测硬件上限，再用真实到达/Host 准备测机会命中率。帧未到达或准备迟到时走无重叠路径，不能等待未来帧而隐藏当前动作的延迟。
- 下一帧 VLM 保持等待当前 AE10，不把本轮变成新增多帧 VLM 并行或跨帧逐层混合 KV。
- 旧 KV 路线保留同样的分区/全卡两阶段试验机会，但切点按其 Prefill 完成位置确定，不机械照搬同帧 `k=1`；最多测试一个冻结切点及一个相邻点。

若固定切点恢复全卡需要等待尚未完成的另一分支，这段等待必须单列；不能将其从延迟中扣除。暂不引入逐轮 Host polling 或依赖未来完成状态的复杂 GPU 分支调度。

## 4. 正式测量与 Nsight

筛选和确认数据分开。进入确认前冻结配额、切点、提交顺序及 Host 选帧参数。

- 每个确认候选对匹配基线：预热后 **100 对 AB/BA**；关键结果在同一物理 GPU 上另开进程、反转起始顺序再做 100 对。输入、noise、shape 与初始化顺序绑定，不由完成顺序决定。
- 至少 20 次预热，并检查是否仍在编译/捕获；加载、预热、捕获时间单列。全部正式样本的稳态重捕获必须为 0。
- 跨帧吞吐以 **100 对、每条 16 帧的完整轨迹**比较，计入 fill/drain；关键候选独立复测。每臂另做至少 100 次连续输出，reset 间隔 50；不把单请求延迟倒数当成流水吞吐。
- 资源配置固定 `CUDA_DEVICE_MAX_CONNECTIONS=32`、CPU 线程数和优化开关；记录 driver/runtime/header/PyTorch/源码/输入版本以及 GPU UUID。正式计时期间不运行 profiler。
- nsys 仅采集基线、最佳固定分区、G3、最终组合的短窗口，标签覆盖 frame、slot/generation、SM 档位、AE1…10、Encoder/VLM、KV-ready 和阶段交接。
- 以实际 GPU Kernel 区间交集确认重叠，报告同签名 GEMM/Attention 的并发减速、提交空洞、Event 等待、全卡切换等待、阶段机会/命中数。NVTX 或 Event 跨度不替代 Kernel busy time。

关键输出：AE 输出延迟、Prefill 完成时间、双分支总完成、CPU 提交/等待、CPU→CPU mean/P50/P95、逐轮 AE 时间、吞吐、峰值显存、KV 发布频率、消费快照年龄、Graph 重捕获次数及数值误差。分项跨度存在重叠，不做简单加和归因。

## 5. 到达过程、SLO 与连续运行

先固定 Host lookahead/选帧规则比较 GPU 调度，不把 CPU 策略变化混入 SM 收益。保留历史 33/62 ms 到达场景；另按 D10 F1 实测稳态输出周期的 1.05 倍设置一个接近容量的到达间隔，向上取整到毫秒，选定后所有臂相同。

报告 capture→action P95 在 **0/2/5/10%** 允许退化下的可选方案，同时报告丢帧与实际 source_id。默认实用候选以 **5%** 预算筛选，不能事后只展示最宽的一档。旧 KV 与当前 KV 分列；记录视觉 KV 来源年龄和当前 state 来源，不能用不同观测输出直接做数值差。

每个被接收并服务的观测完成一次 Prefill；fill/drain 单列后核对更新/消费计数。固定输入重放不得额外增加旧 KV 帧偏移，真实到达场景的 P95 KV 年龄也纳入预算。禁止通过饿死 Prefill 宣称动作加速。

最后用四套件各两个预先固定案例做部署 smoke（每种语义至多一个候选及其参考），记录动作、真实执行步数和成功结果，不录全量视频，不据 8 例声称质量等价。若本轮只完成重放，应明确“未验证闭环收益”，不默认扩大为 200 例质量评测。

## 6. 推进与停止标准

- 数值超差、KV identity/Slot 生命周期错误、隐藏额外完整 KV、死锁或稳态重捕获：立即停止该候选，保存失败，不进入速度排名。
- 只比固定限额 G2 快而不比全卡 G0/G1 快：结论为“资源回收有效，但没有净优化收益”。
- 进入推荐的目标：独立复测中 CPU→CPU 延迟下降至少 5%，或在 P95 帧延迟与 KV 年龄预算内吞吐提高至少 5%；需相对**相同 Host 策略、相同 KV 语义**基线成立。
- 1%–5% 的稳定收益保留为小幅收益，不包装成主要突破；统计报告配对差/区间并按轨迹成组，不能把周期重放请求当独立场景样本。
- 两个配额的最小两阶段方案均无净收益，则停止扩大动态 SM 搜索，保留普通 Encoder 重叠主线及全部负结果。不追加每层任意调额、MPS、多 Host Worker 或模型算子改写。

## 7. 实施范围、设备与产物

拟修改范围限于以下模块及直接相关测试；此表不表示已经修改：

- `preparations/idea2/layer_graphs.py`：D10/指定切点捕获、共享双槽的阶段 Graph 变体、各阶段完成事件。
- `preparations/idea2/green_context.py`：预建资源配置和全卡路径元数据；没有运行中 resize。
- `preparations/idea2/layer_bench.py`、`layer_crossframe_bench.py`：D10 参数、匹配分图对照、阶段计时/提交/SLO。
- `preparations/idea2/layer_scheduler.py`、`layer_lifecycle.py`：仅处理跨 stream 最终 reader_done/交接验证等必要差异；不重写 Pool。
- `preparations/idea2/layer_trace.py`、`layer_report.py`：D10 和资源切换标签/汇总，必要的回归测试。

执行阶段只占用一张已授权且确认空闲的 GPU；成组比较和复测使用同一 UUID，不抢占其他会话，不把历史卡号视为永久授权。没有空闲卡时由一个调度者按用户要求每分钟检查；**最初仅审阅方案时未启动检查或实验；现已按用户后续授权执行**。先用有界小批确定配置和捕获成本，再据实给剩余 ETA，不承诺未知编译时间。

拟输出到新的 `results/idea2/d10_dynamic_sm_<run_id>/`、`reports/idea2_d10_dynamic_sm_<run_id>/` 和 `nsys/idea2_experiment/d10_dynamic_sm_<run_id>/`，保留逐请求 CSV、配额/切点选择记录、失败与排除表、完整复现命令和代表性 `.nsys-rep`。起草时未创建这些实验产物；现在的真实命令与结果位于 `results/idea2/d10_dynamic_sm_20261024/`，对应状态见执行记录。

## 8. 参考

- [D2 技术路线、结果及边界](idea2_layer_pipeline_20261024.md)
- [历史 1/5/10 步 CEWP 探索](idea2_cewp_20261021.md)
- [现有分图与层事件实现](../../../preparations/idea2/layer_graphs.py)
- [Green Context 桥接](../../../preparations/idea2/green_context.py)
- [原时间递推与数值验证](../../../preparations/idea2/graph_sampler.py)
- [CUDA 12.6 Green Context API](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)：资源按创建时指定。
- [CUDA Graph 与 Green Context](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#green-contexts-launching-work)：Graph 节点执行 context 在创建/捕获时确定。

本方案复用项目 [embodied-eval-acceleration 技能](../../../.agents/skills/embodied-eval-acceleration/SKILL.md) 的小批筛选、同输入对照和真实流水计时规则；最初“先审阅、暂不运行”的要求已由用户后续“你自己先推进”的授权替代。

## 执行中追加的公平性控制

正式O0/O1结果揭示：O0内部AE约55ms就绪，却等待新Prefill完成后才约100ms返回；O1约63ms返回。因此追加一个有界控制，避免将API提前返回全归GPU重叠：两臂均async_reuse、vision_only同Graph/两槽、AE-first/priority0、await_previous_prefix。`graph_async_serial`将原本Vision之后的同一AE完成Event等待前移到Vision之前，令AE→Vision→VLM完全串行，仍在AE就绪后返回；`graph_vision_overlap`保留原等待位置。只改results薄wrapper，不改运行时/模型/采样规则。先20对门禁，再100对与各100连续、独立反向复测，最后8请求nsys核对控制确无Kernel重叠。它不扩大SM配额或切点搜索。
