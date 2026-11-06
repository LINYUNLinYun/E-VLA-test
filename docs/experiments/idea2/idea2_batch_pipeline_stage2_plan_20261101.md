# Batch 流水线：第一阶段证据与第二阶段计划

日期：2026-11-01。此计划在实际第一阶段 GPU 结果之后制定；不是预设结论。

## 第一阶段已经得到的结果

完整 pi05_libero、D10/H10、原 checkpoint/精度，RTX4090。B1 预准备输入的 100 对 AB/BA、每 arm 共 800 请求：

| 指标 | 全卡串行 | 普通全卡流水线 |
|---|---:|---:|
| 吞吐，样本/s | 14.8553 | 16.5660 |
| CPU→CPU mean，ms | 67.3059 | 60.3542 |
| Encoder Event，ms | 10.2528 | 12.7622 |
| VLM Event，ms | 30.9112 | 33.6423 |
| AE Event，ms | 25.5266 | 54.8028 |

吞吐比 **1.11516**，固定 bank 的配对时序重复 bootstrap95%区间 **[1.11389,1.11639]**，不能解释为任务成功率区间。AE 并发后约慢 2.147 倍，VLM 约慢 1.088 倍。需要分别衡量并发占据窗口和资源竞争，不能把重叠区间直接当作净节省。

B2/B4 原始严格 smoke：同 batch 的 Graph 与同 batch 原始 eager 串行，normalized/decoded 在全部已验证帧上均为零误差；但逐 lane 与 B1 原始计算对照不通过预定 MAE≤1e-6、max≤1e-5。B2 的 normalized max 为 0.00758684，decoded max 为 0.00568191；B4 分别为 0.00747871、0.00629895。原始失败目录保留，未进行这些条件的正式性能计时。

单独诊断进一步确认：Prepared 全部输入、noise、mask 和 position 逐位一致；B2/B4 的 Encoder 输出已不同于 B1；固定完全相同的 KV 后，AE 第一轮 velocity 仍有差异，时间条件 MLP 也有约 9.4e-5 的最大差异。因此证据支持**计算 shape 改变带来的原始 batch 数值差异**，不支持归因为 lane 混合、KV 数据竞争或 Graph 更改了同 batch 计算。没有改动权重或 dtype。

SM16 B1 已记录 100 组、每组约18.1样本/s，但随后在 Tensor 析构时 C++ abort，报 `CUDA error: context is destroyed`。该次标记技术失败，保留全部数据，不能直接列为完成且可采用的性能结果。代码显示 Green Context 在 caller-owned、曾 record_stream 的输入/noise 与工作区释放前关闭，需要先修复生命周期。

## 第二阶段的实施顺序

1. **先修收尾，计算路径不变。** 将 Graph/Pool drain 与 Green 资源销毁分为两阶段。Runtime close 把资源所有权交还调用方；调用方先释放 runtime、输入/noise/model 引用、清理 CUDA 工作区与缓存，再关闭 Green streams/context，最后释放 GPU lease。用短 GPU probe 验证 primary context 在关闭后仍能计算，再跑真实模型短测与正式重复。只有成功退出且租约释放的结果才采用。
2. **分开两个数值问题，保留原严格失败。** 新增默认关闭的 `--diagnostic-batch-drift`。该模式增加独立的同 batch eager CPU 动作 oracle，流水线/Graph 继续按原 MAE/max 门槛校验；跨 B1 的严格等价检查继续记录为失败，不调大容差。此模式的容量/调度测量只能标为 diagnostic，不能标为 B1-equivalent 或直接部署候选。未显式开启时保持原来的失败即停止行为。
3. **补齐容量/调度矩阵。** B1 使用严格模式；B2/B4 仅在同 batch gate 全通过后做诊断测量。分别测 prepared 与包含实际 prepare 的录制输入重放，保持100对/组及既有输入/噪声、双槽、时间日程。先测普通并发和 SM16，不因为有 batch 收益就假定流水线仍有效。
4. **依据竞争做单变量调优。** 对 B2/B4 将 AE16 改为 AE32，保持 lead3；有必要再测 AE64。记录实际 SM 配额、VLM/AE 并发减速、KV frontier 与代龄。如果 quota 调整有效，再单独测领先层数，不直接铺满所有组合。关键结果反向进程/arm 顺序复测。
5. **单独采 NSYS。** 对代表性的串行、普通流水线、SM16/候选配额采短 trace，保留真实 replay 外层标签与每个 Kernel。比较 AE×VLM、AE×下一 Encoder 的 busy-kernel 交集、GEMM/Attention 变化与 CUDA API 等待。计时与 profiler 数据不混用。

## 采用条件与交付边界

- 两份完整 batched KV、单层 scratch、Pool/Lease/epoch/generation/Event 协议不变；不缓存受当前图像影响的 hidden states/KV，不改模型/去噪步数/权重/精度。
- 全部诊断样本也必须通过同 batch、同 KV 选择日程的 normalized/decoded 门槛。额外报告跨 B 的真实误差；失败不重试到通过。
- B2/B4 即使吞吐提高，也**不满足原 B1 严格等价条件**，不能宣称保持闭环成功率。录制 bank 重放只回答容量与调度问题，实际机器人 SLO/观测年龄与多环境成功率仍需独立评测。
- 最终报告两阶段、CSV、复现命令、代表 trace、数值与技术失败，以及最佳已测试配置。无收益/负收益同样保留；不要把 CPU 准备提前或合批本身的收益全部归给流水线。

## 证据入口

- `results/idea2/batch_study_20261031/stage1_original_b1_full_prepared/result.json`
- `results/idea2/batch_study_20261031/stage1_smoke_b2_paired_prepared/result.json`
- `results/idea2/batch_study_20261031/stage1_original_b4_numeric/result.json`
- `results/idea2/batch_study_20261031/stage1_batch_shape_diagnosis/result.json`
- `results/idea2/batch_study_20261031/stage1_original_b1_sm16_prepared.console.log`
- `results/idea2/batch_study_20261031/stage1_original_b1_sm16_prepared.execution.json`
