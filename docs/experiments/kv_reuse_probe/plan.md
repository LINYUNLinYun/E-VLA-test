# 临时探针：V14 后层 KV 与上一重规划点 Full KV

日期：2026-10-24。用户授权一次有界探索，得到结论后停止。原始 idea 文档不修改。

## 问题与证据

V14 在原始 1-based VLM 层 `[3,4,9,11]` 跳过 hidden 更新，但保留每层提供给 AE 的 K/V shell。本实验验证两个不同问题：

1. 相对同观测 Full，V14 的逐层 KV 偏差是否随深度扩大；上一重规划点 Full 的 KV 是否反而更接近当前 Full？
2. 实际将 V14 的后部 KV 替换成上一重规划点 Full KV，AE 生成的动作是否更接近当前 Full？

不能只用余弦相似度判定可靠性，也不能从离线动作接近程度推断任务成功率。

## 数据与边界

- 复用 `results/idea2/run_20261020/episodes/fresh_serial_r1/` 的真实 Full 轨迹；四套件各 task00、task05，init00、init01，共 16 episode。
- 每 episode 按轨迹长度预选 4 个当前 replan，严格配同 episode 的上一 replan，最多 64 对。选择不依赖输出误差或成败。保存输入路径、SHA、prompt、seed、request 与 step。
- 来源 H50/R5、20 Hz，相邻 replan 间隔 5 个实际执行动作，即 250 ms；这里“上一帧”指上一重规划点，不是连续相机帧。
- 当前 Full 是同一当前观测与噪声的参考；旧 Full 使用上一观测的 prefix KV，但 AE 一律使用当前机器人 state、当前噪声和相同去噪日程。
- 原生 Triton 已有路径，完整 vision 和 AE，D2 主测试 64 对；D10 在每 episode 最早／最晚两个固定点做 32 对确认，覆盖轨迹两端，不为找到正结果扩网格。
- 四套件共用同一 checkpoint，prompt 和 token 顺序在一对内不变。不得跨任务混用 KV。

## 条件

1. 当前 Full KV＋当前 AE：参考。
2. 当前 V14 KV＋当前 AE：剪枝对照。
3. 上一 replan Full 的全部 KV＋当前 AE：全旧 KV 对照。
4. 当前 V14 L1–9＋旧 Full L10–18。
5. 当前 V14 L1–12＋旧 Full L13–18。
6. 当前 V14 L1–15＋旧 Full L16–18。

混合发生在提供给 AE 的逐层 prefix KV 上，不重新传播 VLM hidden，因此不能解释为“用旧 hidden 修复后重算后层”。旧 Full 每个采样点都来自真实上一 replan，这也不是连续多轮不刷新的策略。

## 指标与检查

- 每层 K/V 分开比较当前 V14 对当前 Full、旧 Full 对当前 Full：cosine、relative L2、RMS 比；视觉 token 与全部有效 token 分组，排除 padding。
- 动作按前 1–5 步和全 50 步统计，保存原始与 decoded 输出；平移、旋转、夹爪分开。必要 clip 后平移命令乘 50 mm 仅为控制目标增量代理，不是真实 EEF 误差。
- 记录各 hybrid 相对 V14 的逐样本改善比例、均值与 suite 分组，不能只挑最佳案例。重规划点在 episode 内相关，整体均值与 episode 配对结果分别保存。
- 首个 skip 前 L1/L2 当前 Full 与 V14 应一致；同 KV＋同噪声重复结果应一致；混合不得改写原始 context。保存 targeted CPU 检查与 GPU sanity 结果。
- 层间趋势同时看绝对值和归一化偏差，显式展示 skipped/kept；不把 K/V 幅值不同或 Full 最末层 KV-only 实现误判为层重要性。

## 资源、产物与停止条件

代码：`preparations/kv_reuse_probe/`；结果：`results/kv_reuse_probe/<run>/`；总结：本目录 `report.md`。

使用当前授权空闲 GPU，每卡单个持续模型进程，无视频、无新仿真评测。GPU0 首次启动前被其他任务占用，idle guard 在加载模型前拒绝；该未执行尝试保留。D2 改用 GPU2，D10 小确认并行使用 GPU3。GPU1 仍由用户其他会话保留，不扫描／使用。只在尚未获得可用卡时按约 30 秒等待，不重复扫描本实验已经占用的卡。比较均在各进程内同输入配对，不比较两卡绝对耗时。

完成固定样本、必要 sanity、逐层与动作结果分析后停止；是否支持假设均作结论。输出至少一张逐层偏差图和一张各混合方案动作误差图。若结构趋势与动作收益不一致，分别报告。该临时探针不声称闭环成功率改善或已实现加速。
