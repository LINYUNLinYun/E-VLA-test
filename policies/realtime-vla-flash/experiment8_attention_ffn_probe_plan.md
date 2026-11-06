# 实验 8：VLM Attention / FFN 子模块贡献分析

日期：2026-10-03。当前状态：用户已授权启动。第一阶段完成；扩展 seed12345 的 12 任务/64 观测完成并通过审计，seed23456/34567 正在 GPU1/2 并行采集。12 个任务的 baseline、Attention L16、FFN L2 单种子闭环配对已完成并通过状态/噪声/真实轨迹/视频审计，仍属探索性诊断。原始 JAX 模型独立对照已取得首个样本结果，扩大覆盖中。

准备阶段验证：16 项 CPU 合成/接口兼容测试通过；Python 语法检查通过；12 个有效观测的 dry-run 全部通过。真实执行：GPU 0、1、2 完成 3 个任务、12 个观测的 420 次主要前向和 24 次校验前向；全部正常路径与重复 baseline 最大绝对误差均为 0。结果见 `results/pi0_attention_ffn_diagnosis/report.md`，审计见该目录的 `probe_audit.json`。本阶段只提供固定观测下的子模块贡献和候选证据，不提供剪枝闭环成功率。

## 1. 研究问题

在之前整层余弦相似度分析基础上，将 VLM 每一层拆解为 Attention 与 FFN，寻找表示更新小且跳过后动作稳定的子模块。目标不是把余弦相似度直接等同于可剪枝，而是给出可追溯的 Attention、FFN 候选排名及后续闭环验证条件。

对 L1–L17 保存三个真实残差流状态：

```text
h_in → Attention + residual → h_attn → FFN + residual → h_out
```

L18 在当前 Triton 实现中只生成 prefix K/V，不更新 hidden，不列入 Attention/FFN 候选。所有报告使用从 1 开始的层号，运行时索引从 0 开始。

## 2. 实验干预定义

| 条件 | 执行方式 | 保留内容 |
|---|---|---|
| 完整模型 | Full18 VLM、Full18 Action Expert、D10 | 正常 Attention、FFN 和各层 K/V |
| skip Attention Lk | 跳过该层 Attention 聚合及输出投影的残差更新 | 原始 QKV 投影、原始 K/V、该层 FFN |
| skip FFN Lk | 跳过该层 FFN gate/up/down 和残差更新 | Attention、原始 QKV/K/V |

第一阶段有 1 个完整条件和 34 个独立干预。保留原始 QKV 而非改用 KV-only kernel，减少投影数值差异对贡献分析的干扰。后续若要优化延迟，单独测试 KV-only 投影的数值一致性和实际收益。不能把这里的跳过 Attention 当成删除全部投影的耗时收益。

每次干预重新计算视觉编码、重置语言输入、state 和 diffusion noise，不继承前一个干预的 hidden 或动作状态。每个观测的完整模型和所有干预使用同一扩散噪声。

## 3. 数据与样本规模

第一阶段复用之前经过阶段纠正的 12 个观测，涵盖抽屉、多物体/放置和微波炉操作的 3 个任务、初始/接近/操作/搬运/目标对齐阶段。明确排除旧错误标注的 `libero_10_task003__step0001__goal_alignment.npz`。这些状态来自旧实验轨迹，只用于固定观测机制诊断，不作为可信正式 baseline 的成功率证据。

第一阶段计划 12 × 35 = 420 次主要前向；另有每观测正常路径对照、重复完整模型检查，共 24 次校验前向。不能将 12 个观测称作 12 个独立任务。

扩展阶段在可信任务/状态审计后采样至少 12 个不同任务，覆盖 LIBERO-10 与 LIBERO-90、不同初始化和关键操作阶段，固定观测至少使用 3 个独立噪声种子。按语义任务聚合或 cluster bootstrap，避免把同一任务多个阶段当成独立样本。

## 4. 表示指标

Attention 比较 `h_in` 和 `h_attn`；FFN 比较 `h_attn` 和 `h_out`。主相机、腕部相机及语言 token 分别统计：

- 逐 token 余弦相似度均值及 P05/P50/P95，注意低相似度尾部。
- 相对残差更新量 `||after-before|| / ||before||`。
- 绝对更新范数及表示范数，识别余弦很高但幅度变化很大的模块。
- 按任务、阶段提供明细及任务级置信区间。

保存 FP32 hidden 快照，不在 BF16 已有精度基础上额外转为 FP16。保持真实 token 顺序和 modality spans，报告零范数/异常非有限数据。

后续还可对 Attention 更新与 FFN 更新计算方向夹角，检查整层低更新是否来自两个子模块更新互相抵消。该指标是补充机制分析，不代替独立干预。

## 5. 动作敏感度与候选选择

保存原始模型 32 维动作，以及经过与正式 policy 一致的 output transform 后的 7 维机器人动作指令。用于报告的动作误差以反归一化指令为主；不能把归一化动作误差标成真实末端位移，也不能把机器人控制指令直接当世界坐标轨迹。

逐子模块统计完整动作块的 NMSE、平移/旋转指令 RMSE、夹爪 RMSE 和符号翻转率；补充实际执行前 12 步的误差，区分规划块后段差异与真实会执行的动作差异。所有分析记录 action space 和噪声身份。

分别生成 Attention 和 FFN 候选排名。优先候选应同时满足表示更新小、动作误差小、夹爪稳定，检查最差任务/阶段，不仅比较全局均值。综合分数仅帮助排序，不构成非劣效性结论。只有 3 个任务时 bootstrap 区间为探索性统计，不能支撑泛化保证。

## 6. 代码与验收

现有底层入口扩展：

- `scripts/spec/triton/pi0_infer.py`：独立 skip mask 及三状态 trace。
- `scripts/spec/triton/triton_pi0_runtime.py`：eager diagnostics 接口传递。

新增代码：

- `scripts/experiment8/collect_submodules.py`：相同观测/噪声的 34 个独立干预，FP32 快照、输出反归一化、normal/eager parity、干预后的 baseline 重复校验。
- `scripts/experiment8/analyze_submodules.py`：子模块/模态/阶段指标、任务 bootstrap、CSV、图与候选排名。
- `scripts/experiment8/test_submodule_encoder.py`：CPU 合成残差更新验证独立跳过、K/V 保留、快照不别名、L18 排除。
- `scripts/experiment8/test_submodule_metrics.py`：合成表示/动作及聚合指标验证。

正式运行前代码验收：CPU 合成测试通过、语法检查通过、旧诊断接口兼容。采集命令支持 `--dry-run`，只检查观测、prompt manifest、哈希和预计前向数量，不加载模型、不使用 GPU。真实 GPU 路径还需在得到启动指令后做小规模 smoke，再扩大运行；不能把 CPU 合成测试说成真实模型数值一致性已经通过。

GPU smoke 验收：instrumented eager 完整动作与普通完整路径一致（初始 `atol=rtol=1e-4`，不通过时调查而非直接放宽）；所有干预结束后重复完整模型一致；hidden 三状态形状正确、有限、快照独立；同一观测哈希与噪声哈希一致。保留校验误差作为数据元信息。采集器不覆盖已存在结果，避免混合旧配置。

## 7. 运行与交付安排

用户允许使用空闲 GPU；启动时重新检查占用，将不同任务或观测分片分配至空闲 GPU，分析任务可使用 Luna agent。第一阶段已按任务分片执行并完成；以下为采集命令模板，重新运行需使用新输出目录，不能覆盖本次结果。

采集命令模板：

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi \
.venv/bin/python scripts/experiment8/collect_submodules.py <observation.npz ...> \
  --base-triton-path /root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted/base \
  --prompt-manifest <manifest_matching_task_suite> \
  --output-dir results/pi0_attention_ffn_diagnosis/probes/<shard>

.venv/bin/python scripts/experiment8/analyze_submodules.py <probe.npz ...> \
  --output-dir results/pi0_attention_ffn_diagnosis/analysis
```

不同任务使用实际匹配的 prompt manifest；观测内容哈希决定噪声，分片与输入排序不改变随机样本。

交付目录 `results/pi0_attention_ffn_diagnosis/`：原始 probes、每观测模块指标、按任务/阶段聚合、Attention/FFN 余弦图、动作敏感度图、两个候选排名、校验记录和中文报告。

候选确定后进入实验 7 的可信 baseline 闭环体系，先测单独 Attention/FFN 跳过，再测小组合；保存蓝色完整模型与橙色变体的真实 XYZ、空间偏差和可播放视频。成功率、接触/抓持阶段、末端姿态/夹爪以及实际延迟都通过后，才能认定子模块可剪。当前已经启动并完成首批单模块闭环诊断；组合、扩大种子及生产路径延迟验收尚未完成。
