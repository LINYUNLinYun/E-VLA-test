# π0.5 Batch 流水线：代码准备与后续实验安排

日期：2026-11-01。本轮按“先准备、后实验”的要求执行：只整理代码、生成命令和运行 CPU 检查，不启动 GPU 实验、不创建等卡任务。历史运行记录保留在 `results/idea2/batch_study_20261031/`，不与本轮准备记录混在一起。

## 主线

增加 batch 可能让 AE 的小矩阵运算更高效，也可能让 Encoder/VLM 的大矩阵更充分占用 GPU，挤掉重叠空间。因此分别回答：

1. **合批收益**：B2/B4 的串行吞吐相对 B1 串行提高多少？
2. **流水线收益**：同一个 B 下，普通双流或 SM 隔离比串行多完成多少观测？
3. **代价**：批次完成延迟、各机器人 KV 更新频率和显存发生什么变化？

batch 指多个独立观测的固定 lane，不把单机器人的未来观测当作当前可用输入，也不把动作 horizon 当成 batch。当前录制输入重放用于测容量与调度；不能据此证明机器人闭环成功率。

## 固定条件

- 完整 `pi05_libero` 权重与模型结构；D10、H10；原精度、显式噪声和原 FP32 Euler 时间递推。
- 每个 lane 保持自己的观测、state、噪声、旧/新 KV；一批固定成员和顺序，只支持整批 reset。
- 保留两个完整的 batched KV Slot、Lease、epoch/generation 和 Event 协议。Pool 仍由唯一调度 Host 修改，没有第三份完整 KV。
- VLM 领先 3 层后启动 AE；每轮 AE 冻结一次可读的新 KV frontier，未就绪层使用同一 lane 的完整上一代 KV。后续 AE 与下一批 Encoder 在可用窗口重叠。
- Graph 按 B、输入布局与 Slot 地址分别捕获；VLM/AE 不共享可写 scratch。不同 B 使用独立进程。

## 第一阶段：建立同口径对照

| 维度 | 配置 |
|---|---|
| B | 1、2、4 |
| 调度 | 全卡串行、普通双流流水线、Green Context 流水线 |
| 首轮 Green 配额 | AE 请求 16 SM，记录实际 AE/Prefix 分配 |
| lead | 3 |
| 输入准备 | `prepared`、允许 lookahead 的 `online`、关闭 lookahead 的 `online` 分开报告 |
| 正式测量 | 20 次预热；至少 100 对 AB/BA，每 arm 每组 8 个请求 |
| 数值验证 | 至少 4 个请求，另覆盖原地更新、两槽轮换与整批 reset |

全卡串行/普通流水线在同一进程、同一套 Graph 上交替；Green 单独运行 100 组，属于跨进程比较。复用已有有效运行，不因新命令清单生成就重复整张矩阵。OOM、数值失败、清理失败保留原始记录。

`prepared` 不计入 bank 的输入准备；`online` 在请求序列内执行 prepare，包含 CPU transform、搬运及模型观察预处理。这里的 `online` 仍指录制输入重放的计时边界，不是实际相机到达或动作执行模拟。串行和流水线使用相同 CPU lookahead 权限。

允许 lookahead 时，当前输入可能已在上一请求期间准备完成，所以 admission→CPU 返回只是一段服务时间，不能直接叫作该输入的完整延迟。新记录另存 `input_prepare_to_cpu_ms`，从这份输入自己的准备开始计时。若要排除提前准备，使用 `--disable-lookahead`：当前输入在自己的请求内准备，同时关闭下一批 CPU 准备和下一批 Encoder。该对照仍允许本批 VLM/AE 并发，但不再有跨批 Encoder 的隐藏窗口。两种实验必须分开解读。

## 正确性先于性能解释

逐 lane 同时检查 normalized 和 decoded 动作：MAE ≤ 1e-6、max error ≤ 1e-5，非有限值失败。混合 KV 路径对照相同输入、噪声和实际 frontier 日程的串行结果；不能把新旧 KV 的语义差异当成调度误差。

已有严格验证发现 B2/B4 相对独立 B1 eager 存在超出上述容差的差异，normalized max error 约 0.00759/0.00748；原始同 B eager 与 Graph 对照在这些样本上一致。相关失败保留于：

- `results/idea2/batch_study_20261031/stage1_smoke_b2_paired_prepared/`
- `results/idea2/batch_study_20261031/stage1_original_b4_numeric/`
- `results/idea2/batch_study_20261031/stage1_batch_shape_diagnosis/`

因此后续有两条明确的执行口径：

- **严格口径**：默认行为不变。B1 equivalence 失败即停止对应性能实验，不放宽容差，也不反复重试有效失败。
- **同 B 调度诊断**：显式传入 `--diagnostic-batch-drift`，仍记录 B1 equivalence 失败，并增加独立的原始同 B eager oracle。相同 frontier 和同 B eager 校验必须通过，才允许分析该 B 的调度性能。`diagnostic_only` 结果不具备上线资格，不能称为相对 B1 无损。

命令生成器现在也支持这个显式开关。即使清单同时包含 B1/B2/B4，诊断开关也只传给 B>1；B1 始终保持严格口径。

## 第二阶段：根据瓶颈调一个变量

先查看第一阶段吞吐、GPU Event 与代表性 nsys。只有出现明确瓶颈才扩展：

1. AE 在 16 SM 上过慢、尾部仍暴露：优先测 24 SM，再按证据选择邻近 20/32 SM；不直接铺满配额网格。
2. VLM 因重叠明显变慢：固定 SM 配额，再测 lead 0/1/3/6，检查整段完成时间和实际新 KV 使用比例。
3. prepare 吃掉下一批 Encoder 的可用窗口：分别记录 CPU 准备、搬运和 Graph 提交时间，再决定是否值得增加 Host worker。不能从 profile 放大的提交时间直接归因正常运行的 CPU 瓶颈。

关键候选做一次同卡反向进程顺序复测，再做较长连续请求序列。保留负收益，不通过减少 Prefill 更新或消费更老 KV 宣称成功。动态 batch、逐 lane reset、多机器人闭环不在本轮代码准备中实现。

## 指标与 nsys

- 总观测吞吐 = `B × 完成请求批次数 / 整段墙钟`，包含最后的 drain；不能只计 AE 返回前的时间。
- 分开报告“同 B 流水线/串行”和“该 B 串行/B1 串行”。批次延迟不能除以 B 后称为单机器人延迟。
- CPU→CPU mean/P50/P95、AE 输出可用时间、Prefill 完成时间、全工作完成时间、峰值显存、Graph 重捕获次数。
- Encoder/VLM/AE 的 GPU Event 区间、CPU 提交/等待、实际 SM 数、每轮新 KV 层数和旧 KV 代龄。每 lane KV 更新频率为请求批次频率，不乘 B。
- 只对代表条件另采 nsys，使用实际 replay 外层标签 `preprocess`、`encoder`、`vlm`、`AE`。`prepared` 没有计时内的 preprocess，属于预期。
- 通过真实 kernel 区间交集确认 AE×VLM、AE×下一批 Encoder 的重叠；Event/NVTX 的区间交叠不能代替 kernel 证据。profile 耗时不参与正式加速比。

## 已准备的代码

| 文件 | 作用 |
|---|---|
| `preparations/idea2/batch_backend.py` | 合批输入、固定 lane、按原始记录生成噪声、保留完整 B 维输出 |
| `preparations/idea2/batch_runtime.py` | 复用两槽调度、全卡/Green 资源选择、逐 lane 验证、完整 replay 标签 |
| `preparations/idea2/batch_bench.py` | 数值门槛、AB/BA、两种输入计时、CSV/JSON/动作保存和安全清理 |
| `preparations/idea2/batch_resources.py` | 启动时原子领取共享租约并检查真实占用；忙卡退出 |
| `preparations/idea2/batch_plan.py` | 纯 CPU 生成基础清单；支持显式诊断口径及逐命令标记 |
| `scripts/idea2_prepare_batch_followup.py` | 复用基础清单，补齐输入提交 Stream 与关闭 lookahead 的开关；不查询 GPU、不启动进程 |

本次补充不改运行时和模型计算。准备适配器放在 `scripts/`，避免修改此前在途实验冻结的源码。实际执行仍调用原来的 `batch_bench`，没有增加 Host worker 或新的调度器。

## 后续入口

以下命令**只生成计划**，不查询或领取 GPU。输出目录必须不存在：

```bash
python3 -m preparations.idea2.batch_plan \
  --batch-sizes 1 \
  --output results/idea2/batch_preparation_20261101/strict_b1

python3 -m preparations.idea2.batch_plan \
  --batch-sizes 2 4 --diagnostic-batch-drift \
  --output results/idea2/batch_preparation_20261101/batch_diagnostic
```

对应 `commands.md` 保留 GPU UUID 占位符。空卡分配后替换 UUID，按 smoke→数值结果检查→性能→代表性 trace 的顺序选择命令，跳过已有有效结果。清单不是自动执行队列。

两份清单合计 36 条候选命令：6 条 smoke、12 条性能、18 条可选 profile。调优时可用 `--green-sm-mins` 和 `--lead` 生成新的独立清单；不要覆盖历史结果，也不要把所有可选 profile 全部作为必跑任务。

初次准备时 CPU 检查 **87/87 通过**，CUDA 未初始化。源码摘要保存于 `results/idea2/batch_preparation_20261101/cpu_validation.json`，完整日志为同目录 `cpu_checks.log`。该版本的 `preparation_check.json` 记录原 36 条命令全部解析通过、全部未启动。

## 补充：后续可直接使用的准备入口

实验按下面顺序执行；本次只完成命令准备和 CPU 核验，不提交新的 GPU 实验：

1. **先核验输入和数值。** 固定 B1/B2/B4、同一批真实记录与每记录噪声，沿用严格门槛。已知 B>1 的 B1 equivalence 失败保持失败；显式诊断只回答同 B 调度问题。
2. **测同 B 的容量收益。** 对照串行、普通流水线、Green AE16；lead3、D10/H10、Graph 和输入提交位置保持一致。分开测 prepared 和 online lookahead。已有相容有效结果直接复用，不重跑已知数值失败。
3. **补输入延迟边界。** 关闭 lookahead，每份输入从开始准备测到动作返回。此处没有下一批 Encoder，因此不能用它否定或证明跨批 Encoder 重叠；它用于说明吞吐与输入延迟之间的代价。
4. **只对有希望的条件采 trace。** 看 Encoder/VLM/AE 单独时长、并发减速和真实 kernel 交集。若 batch 使 AE16 尾部变长，固定其他设置再调配额；若 VLM 竞争恶化，再单独调 lead。不要同时改 batch、配额和 lead。
5. **选满足延迟要求的容量配置。** 总吞吐、每 lane KV 更新频率、完整批次延迟、显存和数值资格共同报告。暂不实现动态合批；真实到达和凑批等待在后续服务/闭环验证中补测。

新的入口只生成文件，使用 stdlib Python 即可：

```bash
python3 -m scripts.idea2_prepare_batch_followup \
  --output results/idea2/batch_preparation_20261101/my_capacity_plan \
  --diagnostic-batch-drift --prepare-stream dedicated --no-profiles

python3 -m scripts.idea2_prepare_batch_followup \
  --output results/idea2/batch_preparation_20261101/my_fresh_plan \
  --input-modes online --disable-lookahead \
  --diagnostic-batch-drift --prepare-stream dedicated --no-profiles
```

两条命令均不检查空卡、不领取租约、不启动模型。正式运行的 GPU UUID 仍是占位符，拿到明确分配的空卡后才能替换。每个输出目录必须是新目录。`--prepare-stream dedicated` 对所有 online 对照一致，prepared 仍保持原路径；与历史 caller 配置比较时，须单独标记这个变量。

已生成的文件位于 `results/idea2/batch_preparation_20261101/followup_ready/`：

- `capacity/plan.json`、`capacity/commands.md`：18 条候选命令，6 条 smoke、12 条性能。
- `fresh/plan.json`、`fresh/commands.md`：12 条候选命令，6 条 smoke、6 条性能。
- `preparation_check.json`：30 条命令全部通过真实 benchmark 的参数解析；未查询 GPU、未启动子进程、CUDA 未初始化；错误计时组合和覆盖已有输出均被拒绝。

这些是候选清单，不是需要全部重跑的队列。fresh 清单的 smoke 仍是原 prepared 数值门槛，可复用同版本的已有有效 smoke。profile 不在这两份正式计时清单中；选定代表条件后，去掉 `--no-profiles` 并用 `--batch-sizes`、`--input-modes` 限定范围生成独立清单，只选其中 profile 条目。

当前运行时已通过的 **93 项 CPU 检查**记录在 `results/idea2/batch_study_20261031/latency_boundary_cpu_validation.json`。本次核对源码摘要与该记录完全一致，复用其结果，没有重复声称本轮又运行了 93 项测试。新适配器的 30 条命令检查单独记录。CPU 准备不代替 GPU 数值、显存、真实重叠和闭环验证。
