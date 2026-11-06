# 最终实验汇总与分析报告：层裁剪、低步数草稿与恢复/验证机制

更新：2026-10-10。范围：idea1 主线 Experiment 10 的已执行实验，以及 Attention L16/L17 五初始状态补充实验。本文是**已执行实验的最终归档**，不是声称研究方案中的全部扩展验证都已完成。层编号从 1 开始；不修改 idea1.md。

## 1. 核心结论

1. **当前最值得推进的是完整 VLM + Action Expert 降噪步数减少，而不是大幅跳 VLM 层。** D2/D4 为 190/200（95%），D3 为 186/200（93%），D5 为 188/200（94%）；D10 基线为 186/200（93%）。点估计更高不等于显著提高，也没有预注册非劣界限。
2. **高余弦只是提出候选的证据，不是可删证明。** 严格满足启发式的五层组合 [3,5,7,9,12] 只有 70/200（35%）；两层组合 [3,10] 为 184/200（92%），但 Goal 只有 41/50（82%）。因此不能宣称找到了统一最优 skip。
3. **“不同阶段层敏感度不同”目前有描述性线索，但没有足够的统计证据支持主张，更没有验证阶段路由优越性。** 四阶段选出的 mask 完全相同，当前阶段置换检验在样本量与多重比较规则下根本不具备拒绝零假设的分辨率。
4. **并行验证尚未实现可靠的任务级加速。** D3/H8 请求时间优于串行验证，但每任务累计等待为基线的 1.38 倍；D7/H12 接近 1 倍，却只有 32/40 成功，低于同子集基线 35/40。高动作接受率不等于任务成功。
5. **Attention L16 仅可作为继续验证候选；L17 不宜通用删除。** 12 任务 × 5 状态：L16 为 40/60 vs 配对基线 39/60，L17 为 31/60 vs 39/60。这是诊断 eager 模式，不是标准套件成绩。

## 2. 实验过程、协议和数据来源

先从完整模型真实 baseline 轨迹提取阶段观测，再做单层/子模块干预和数值探针；按 search→validation→冻结候选闭环筛选；随后开展恢复前缀长度、Kalman 门控和双 GPU 并行验证。没有使用随机初始化权重或随机输入代替真实机器人观测。

标准闭环：Object / Spatial / Goal / LIBERO-10，官方 init0–4，环境 seed=7、warmup=10、replan=12；训练后同一 checkpoint。search 为 task0–5，validation 为 task6–7，冻结探索评估为 task8–9。任务曾参与历史实验，因此 task8–9 不是严格首次见到的外部测试集。200 条总体成绩含 search/validation 任务，评估子集另外列出。

离线噪声种子为 12345/23456/34567；同一观测与重复噪声不视为独立任务。模型执行 BF16，数值统计转 FP32；权重 hash 不变、baseline 与记录路径 parity 均通过。参数不会随场景改变，“权重非零比例/数值变化”不能衡量场景敏感度；本轮测量的是激活、残差更新和真实跳过干预后的动作变化。

四卡用于相互独立 suite/probe/候选并行；并行验证单 worker 占两卡。无空卡时等待 300 秒；已由本实验持有的卡按任务完成事件派发，不高频扫描。专门计时按套件在同一张 4090 上轮转条件，避免拿不同 GPU 的拥塞时间混作模型差异。


| 证据块 | 实际覆盖 | 解释 |
| --- | --- | --- |
| 真实观测/单层离线探针 | 1619 观测；4857 观测-噪声块 | 399/392/342/486；来自 200 条 baseline，不是 1619 个独立任务 |
| 标准闭环 | 3160 条；124 个条件×套件单元 | 12 条件×200 + 19 条件×40；每条有成功标志/telemetry/视频 |
| 专门计时 | 1920 次 | 8 条件×4 套件×60；少量 validation 阶段观测，非吞吐/任务总耗时 |
| Attention 补充 | 120 对、240 条 rollout | L16/L17 各 60 对；基线重复运行不当作额外独立任务 |
| 视频校验 | 标准 3160 条 + 补充 240 条 | 原始 validator 全帧解码已通过；不是全部人工观看 |


### 2.1 过程中的修正和审计边界

L16 首次 LIBERO-90 运行发生 prompt-cache manifest 不匹配，失败 attempt 单独保留，修复各套件 manifest 后只纳入最终完整配对数据。部分 40 条任务子集曾被旧 validator 当成 50 条而标记失败，后来依据明确 task_ids 重验；运行错误不计作正常任务失败。GPU 归属状态文件发现逻辑扩展为匹配 `*status*.json`，避免漏掉带后缀状态文件。

本报告重新检查 completed validator、episode 唯一 task/init 键、分母、异常标志，并逐对读取 2200 条主条件 XYZ 对比数据验证配对初始 XYZ 一致。阶段/数值统计重新聚合；视频完整解码结论引用原始 validator，不声称这次重新解码了全部视频。历史文字若与原始 validator 冲突，以本报告重新统计的原始数据为准。

## 3. 相似度：怎么测、可说明什么

对真实观测中每层 block 输入 H_in 与输出 H_out，对 agent/wrist 视觉 token、language token 分别计算每个有效 token 的余弦后平均。表中 robust p10：仅 search 数据，先按 suite/task/phase/modality 对观测取中位数，再取这些单元的 10% 分位数。它是**同一层输入→输出的余弦**，不是任意两个层之间的完整 18×18 两两余弦矩阵；原始探针未保存所有 hidden tensor，不能凭现有统计伪造全层两两矩阵。

单层敏感度：固定同一观测和动作噪声，跳过指定 VLM block 的 hidden update，保留每层 KV shell；比较前 4 个动作、前 6 维连续动作在模型归一化坐标中的 RMSE，不包含夹爪维度（夹爪翻转单独记录）。先平均噪声/init，按任务/阶段汇总，再任务等权；先验表为 suite/task 单元均值，各 suite 有同等任务数。它不是米制轨迹误差，米制误差见第 7 节。


![逐层余弦与动作干预](results/experiment10_phase_draft_20261009/final_analysis/layer_cosine_action.png)

| 层 | robust cosine p10 | H4 动作 RMSE | 候选解释 |
| --- | --- | --- | --- |
| 1 | 0.2271 | 0.6686 | 结构/低余弦保护 |
| 2 | 0.9914 | 0.0531 | 结构/低余弦保护 |
| 3 | 0.9937 | 0.0490 | 仅可优先做组合干预 |
| 4 | 0.9938 | 0.0612 | 仅可优先做组合干预 |
| 5 | 0.9946 | 0.0628 | 仅可优先做组合干预 |
| 6 | 0.9912 | 0.0850 | 仅可优先做组合干预 |
| 7 | 0.9913 | 0.1226 | 仅可优先做组合干预 |
| 8 | 0.9867 | 0.0808 | 仅可优先做组合干预 |
| 9 | 0.9917 | 0.0901 | 仅可优先做组合干预 |
| 10 | 0.9911 | 0.0768 | 仅可优先做组合干预 |
| 11 | 0.9882 | 0.0722 | 仅可优先做组合干预 |
| 12 | 0.9835 | 0.0790 | 仅可优先做组合干预 |
| 13 | 0.9657 | 0.1037 | 结构/低余弦保护 |
| 14 | 0.9454 | 0.0840 | 结构/低余弦保护 |
| 15 | 0.9593 | 0.0676 | 结构/低余弦保护 |
| 16 | 0.9449 | 0.0788 | 结构/低余弦保护 |
| 17 | 0.5988 | 0.0281 | 结构/低余弦保护 |
| 18 | 不可定义 | 0.0000 | 结构/低余弦保护 |


L3–L5 是早期候选，但 L7 虽然余弦很高，动作干预更大；L17 余弦很低、动作扰动却较小，说明 hidden 变化幅度与动作影响不是同一件事。L18 是 KV-only 特例：hidden 无更新不能解释为整层可以删除。搜索保留 L1/L2/L17/L18；低余弦 L13–L16 保护；高余弦 L3–L12 仅用于候选提出。

### 3.1 场景/阶段敏感度的证据强度


![层与阶段描述性敏感度](results/experiment10_phase_draft_20261009/final_analysis/phase_layer_sensitivity.png)


阶段比较固定同 task、同 init；对单层影响扣除该 task/phase 的整体层平均变化，以尽量区分“所有层都更难”和“某层相对特别敏感”。408 个阶段比较分为 4 套件、每套件 102 比较。每次只有 5–6 任务，双侧符号置换最小可能 p=0.03125（5 任务为 0.0625），远大于 Holm 首步阈值 0.05/102≈0.000490；全部 Holm 拒绝数为 0，**这是检验分辨率不足，不是证明阶段敏感度相同**。

阶段与轨迹进展、观测分布共同变化，尚未做控制变量的场景扰动；描述性热图不能推出因果阶段路由。后续更合理的是先冻结少量层×阶段主要对比和效应阈值，再对独立任务复现，而非把重复噪声当额外样本来获得显著性。

## 4. Attention / FFN 数值与干预

记录 a=H_attn−H_in、f=H_out−H_attn、b=a+f 的 RMS、分位数、近零比例与 token/channel 能量集中度。相对更新分别为 RMS(a)/RMS(H_in)、RMS(f)/RMS(H_attn)；抵消比为 RMS(b)/(RMS(a)+RMS(f))。由同形状残差 RMS 恒等式推导 cosine(a,f)=(b²−a²−f²)/(2af)，不是额外保存了所有原始残差向量；零分母保留缺失。下图为 search 任务、阶段/任务/套件等权描述。


![Attention FFN 更新和抵消](results/experiment10_phase_draft_20261009/final_analysis/attn_ffn_numerical.png)

| 层 / agent token | Attn 相对 RMS | FFN 相对 RMS | 抵消比 | derived cosine(a,f) |
| --- | --- | --- | --- | --- |
| 1 | 1.721 | 0.321 | 0.974 | 0.861 |
| 3 | 0.019 | 0.038 | 0.776 | 0.091 |
| 5 | 0.030 | 0.046 | 0.733 | 0.027 |
| 7 | 0.055 | 0.059 | 0.701 | -0.020 |
| 12 | 0.080 | 0.260 | 0.667 | -0.493 |
| 16 | 0.071 | 0.226 | 0.812 | 0.079 |
| 17 | 0.528 | 2.169 | 0.829 | 0.033 |

| 层 / agent block 更新 | 精确零比例 | \|值\|≤0.01×RMS | 最高10% token能量 |
| --- | --- | --- | --- |
| 1 | 0.08% | 12.04% | 12.1% |
| 3 | 0.39% | 1.20% | 31.2% |
| 5 | 0.38% | 1.69% | 48.4% |
| 12 | 0.25% | 1.78% | 54.2% |
| 16 | 0.38% | 1.67% | 48.8% |
| 17 | 0.17% | 5.99% | 89.3% |

具体数值表明：agent 的 L3/L5 Attention 相对更新约 1.9%/3.0%，FFN 约 3.8%/4.6%，适合作为轻量干预候选；L12 的 Attention/FFN 更新方向负相关（derived cosine≈−0.49），存在抵消，不能据较小整 block 更新推断两个子模块都无用。L17 的视觉 FFN 相对更新超过 2，最高 10% agent token 集中约 89% block 更新能量，反映集中强更新而非零输出。注意这些是聚合描述，不是对所有任务逐样本的保证。


精确零比例包含 BF16 激活差分的量化效应，不是权重参数稀疏率，也不能直接作为剪枝安全判据。


![子模块单独干预](results/experiment10_phase_draft_20261009/final_analysis/submodule_interventions.png)

数值活跃度可以揭示更新集中或抵消，却不能直接推出“剩余 token/channel 或模块是多余的”。活跃度与干预影响的相关性共有 3264 个可用探索单元，其中 3031 个 task-bootstrap 区间覆盖 0；小样本、重复对比且未经多重校正，不可筛选某个显著单元来证明机制。

子模块图来自各 suite 选定层、search init0 的代表阶段观测与 3 个噪声，不是每层×所有任务的完全均衡设计；缺少某阶段的 task 不能补 0。Attention、FFN、整 block 跳过的作用不同，不能把单模块结论移植为整层安全性。L1 是诊断强干预对照，不是删除建议。尚无完整 QKV entropy、FFN gate 开闭率或权重更新证据。


## 5. 标准套件闭环：成功率、泛化与代价


| 条件 / VLM skip | Object | Spatial | Goal | LIBERO-10 | 总体 |
| --- | --- | --- | --- | --- | --- |
| Full18_D10 / [] | 48/50 (96%) | 47/50 (94%) | 49/50 (98%) | 42/50 (84%) | 186/200 (93.0%) |
| Full18_D1 / [] | 49/50 (98%) | 46/50 (92%) | 41/50 (82%) | 17/50 (34%) | 153/200 (76.5%) |
| Full18_D2 / [] | 49/50 (98%) | 49/50 (98%) | 49/50 (98%) | 43/50 (86%) | 190/200 (95.0%) |
| Full18_D3 / [] | 48/50 (96%) | 48/50 (96%) | 48/50 (96%) | 42/50 (84%) | 186/200 (93.0%) |
| Full18_D4 / [] | 50/50 (100%) | 48/50 (96%) | 48/50 (96%) | 44/50 (88%) | 190/200 (95.0%) |
| Full18_D5 / [] | 48/50 (96%) | 48/50 (96%) | 48/50 (96%) | 44/50 (88%) | 188/200 (94.0%) |
| FrozenStatic01 / [2, 4, 6, 8, 10, 12] | 14/50 (28%) | 10/50 (20%) | 5/50 (10%) | 0/50 (0%) | 29/200 (14.5%) |
| FrozenStatic02 / [2, 12] | 45/50 (90%) | 49/50 (98%) | 47/50 (94%) | 40/50 (80%) | 181/200 (90.5%) |
| FrozenLegal01 / [3, 5, 7, 9, 12] | 31/50 (62%) | 23/50 (46%) | 7/50 (14%) | 9/50 (18%) | 70/200 (35.0%) |
| FrozenLegal02 / [3, 10] | 49/50 (98%) | 50/50 (100%) | 41/50 (82%) | 44/50 (88%) | 184/200 (92.0%) |
| Recover_D3_H8 / [] | 49/50 (98%) | 49/50 (98%) | 49/50 (98%) | 44/50 (88%) | 191/200 (95.5%) |
| Recover_V16_D10_H8 / [9, 10] | 50/50 (100%) | 47/50 (94%) | 49/50 (98%) | 43/50 (86%) | 189/200 (94.5%) |

![套件成功率](results/experiment10_phase_draft_20261009/final_analysis/success_heatmap.png)

全部主条件均为 200 条，不是某些旧记录中的 400 条。Recover 条件表示执行一次短前缀后重新观测并恢复 Full18，不是整条轨迹都用裁剪模型。


| 条件 | 配对差 pp | 任务聚类 95% pp | 新失败 / 恢复 | task8–9 子集 vs baseline |
| --- | --- | --- | --- | --- |
| Full18_D1 | -16.5 | [-24.5, -9.0] | 39 / 6 | 24/40 vs 35/40 |
| Full18_D2 | +2.0 | [-1.0, +5.0] | 6 / 10 | 35/40 vs 35/40 |
| Full18_D3 | +0.0 | [-4.0, +4.0] | 11 / 11 | 37/40 vs 35/40 |
| Full18_D4 | +2.0 | [-1.0, +5.0] | 7 / 11 | 37/40 vs 35/40 |
| Full18_D5 | +1.0 | [-2.5, +4.5] | 8 / 10 | 33/40 vs 35/40 |
| FrozenStatic01 | -78.5 | [-86.5, -69.5] | 157 / 0 | 10/40 vs 35/40 |
| FrozenStatic02 | -2.5 | [-7.5, +2.0] | 13 / 8 | 31/40 vs 35/40 |
| FrozenLegal01 | -58.0 | [-66.0, -49.5] | 116 / 0 | 16/40 vs 35/40 |
| FrozenLegal02 | -1.0 | [-7.0, +4.0] | 13 / 11 | 34/40 vs 35/40 |
| Recover_D3_H8 | +2.5 | [-0.5, +5.5] | 6 / 11 | 37/40 vs 35/40 |
| Recover_V16_D10_H8 | +1.5 | [-3.0, +5.5] | 7 / 10 | 33/40 vs 35/40 |

![配对任务聚类区间](results/experiment10_phase_draft_20261009/final_analysis/paired_success_intervals.png)

统计以 task 为聚类单元、suite 内重采样 10,000 次，四 suite 等权；不是把同任务五状态当 IID。所有区间均探索性、未做跨条件多重校正，也未预注册非劣界限。仅 2 个 task/suite 的 task8–9 子集不支持强泛化主张。

### 5.1 Layer-skip 搜索实际做了什么

采用启发式合法独立集、beam/Pareto、少量随机补充与同预算 local swap；候选排名使用真实组合 forward，不把单层分数相加当组合得分。合法删除数 2/3/4/5 分别只有 36/56/35/6 种，6 层删除不存在合法解。全局实际评估 239 个候选：139 主规则候选、100 单规则挑战；原定预算 90/10，但因合法空间与去重/搜索停止，**实现比例为 58.2/41.8，而不是 90/10**。代码挑战预算取名义预算的 10%，没有按实际主候选数重新缩减。

随后 refined 阶段 18 个候选、validation 阶段 12 个候选。FrozenStatic01=[2,4,6,8,10,12] 与 FrozenStatic02=[2,12] 是刻意放宽 L2 的挑战，不是严格遵守“起始两层保护”的最优解；严格合法对照另有 [3,5,7,9,12] 和 [3,10]。V13 原始随机 mask [3,4,7,12,15] 违反相邻删除及低余弦保护，不应用来支持规则有效。

四阶段最终 mask 都是 [2,4,6,8,10,12]，并未产生差异化阶段路由。未穷举整个参数空间、未证明全局最优；更重要的是闭环显示离线组合排名不能替代成功率。

### 5.2 超过 90% 的条件在哪些 suite

D2/D3/D4/D5 在 Object/Spatial/Goal 都超过 90%；LIBERO-10 分别 86/84/88/88%，没有超过 90%。[3,10] 在 Object/Spatial 为 98/100%，Goal 为 82%，LIBERO-10 为 88%；[2,12] 在 Spatial/Goal 为 98/94%，Object 恰好 90%、LIBERO-10 为 80%。这说明“任务限定轻裁剪”可能成立，但不是全套件统一无损。


![单次推理与成功率](results/experiment10_phase_draft_20261009/final_analysis/latency_success.png)

| 条件 | 专门 pipeline ms | 相对 D10 降幅 | 同轮闭环成功率 |
| --- | --- | --- | --- |
| Full18_D10 | 34.36 | 0.0% | 93.0% |
| Full18_D1 | 24.36 | 29.1% | 76.5% |
| Full18_D2 | 25.48 | 25.8% | 95.0% |
| Full18_D3 | 26.59 | 22.6% | 93.0% |
| Full18_D4 | 27.69 | 19.4% | 95.0% |
| Full18_D5 | 28.82 | 16.1% | 94.0% |

pipeline 为预处理+模型 wall time，排除环境步进/网络/控制，不是整条任务完成时间。D1 虽最短，成功率降至 76.5%，尤其 LIBERO-10 17/50（34%）；D2/D3 更值得评估。冻结 VLM mask 没有对应的专门计时，不用历史请求日志冷启动混造其时延。

## 6. 短前缀恢复、门控与双 GPU 验证


同状态短分支 H8 的 measured EEF 平均偏差（suite 内 task/phase 等权，再 suite 等权）：D3 1.80 mm；V13_D10 7.98 mm；V13_D3 8.41 mm；V16_D10 2.67 mm。它是少量 validation init0 锚点的局部结果，不是任务成功率。


![门控与恢复成本](results/experiment10_phase_draft_20261009/final_analysis/gate_recovery_cost.png)

| 恢复 H（tasks6–7） | 成功 / baseline | 累计 server inference 比 |
| --- | --- | --- |
| 1 | 39/40 vs 39/40 | 1.026× |
| 2 | 38/40 vs 39/40 | 1.068× |
| 4 | 35/40 vs 39/40 | 1.031× |
| 8 | 40/40 vs 39/40 | 0.987× |
| 12 | 38/40 vs 39/40 | 0.984× |

一次早期 D3 前缀后恢复 Full18 在 H8 为 40/40 vs 39/40，200 条补充为 191/200 vs 186/200，但只是初步显示可恢复；H8/H12 的累计 server inference 节省仅约 1–2%，日志受采样和系统扰动影响，没有证明实用净加速。每条仅一次草稿不能摊薄总推理成本，也没有反复进入精细操作阶段的干预。


| Kalman 条件（tasks8–9） | 成功 / baseline | 累计 server inference 比 |
| --- | --- | --- |
| KalmanD3_Q50 | 34/40 vs 35/40 | 1.245× |
| KalmanD3_Q75 | 34/40 vs 35/40 | 1.147× |
| KalmanD3_Q90 | 31/40 vs 35/40 | 1.128× |
| KalmanD3_Q95 | 32/40 vs 35/40 | 1.119× |

Kalman 门控所有阈值均未节省累计推理；Q/R 与方差 floor 敏感性未完成。预测动作一致性不验证目标状态或动作语义正确性，不适合作为安全保证。


![并行验证任务成功与累计等待](results/experiment10_phase_draft_20261009/final_analysis/parallel_verify_tradeoff.png)

| 并行配置（tasks8–9） | 成功 / baseline | 草稿接受比例 | 累计请求等待比 |
| --- | --- | --- | --- |
| ParallelVerifyD3_H4_T89 | 34/40 vs 35/40 | 69.4% | 1.838× |
| ParallelVerifyD3_H8_T89 | 35/40 vs 35/40 | 64.5% | 1.380× |
| ParallelVerifyD3_H12_T89 | 35/40 vs 35/40 | 61.5% | 1.218× |
| ParallelVerifyD5_H4_T89 | 33/40 vs 35/40 | 90.6% | 2.292× |
| ParallelVerifyD5_H8_T89 | 33/40 vs 35/40 | 90.2% | 1.485× |
| ParallelVerifyD5_H12_T89 | 33/40 vs 35/40 | 91.3% | 1.065× |
| ParallelVerifyD7_H4_T89 | 32/40 vs 35/40 | 99.3% | 2.696× |
| ParallelVerifyD7_H8_T89 | 31/40 vs 35/40 | 99.4% | 1.480× |
| ParallelVerifyD7_H12_T89 | 32/40 vs 35/40 | 98.9% | 0.998× |

并行 D3/H8 请求均值 44.51 ms，相对串行验证约 71.2 ms 改善，但仍需要 1052 次 replans，每任务累计等待 1166 ms vs 基线 845 ms。D7/H12 约 843 ms vs 845 ms（0.998×），只差约 2 ms，不能宣称可靠加速，而且损失 3/40 成功。client roundtrip 与上一表 server inference 不同口径；未计入双 GPU 的总吞吐/占用成本，任务累计等待也不等于环境整体 wall time。缩短前缀增加 replans，可抵消单次请求改进。

## 7. 失败阶段与真实 XYZ 轨迹

最高到达阶段回答“失败前进展到哪里”，不回答“错误从哪里开始”。下图保留原阶段分析，并另算 baseline 成功、变体失败的配对轨迹：按相同环境步比较，XYZ 偏差连续 3 状态超过 2 cm 的首次时刻，再相对 baseline 实际接触/抓取事件分类。该阈值是描述性定义，尚未做阈值敏感性分析；未出现阈值不等于无动作/姿态/夹爪错误。


![失败最高阶段](results/experiment10_phase_draft_20261009/final_analysis/failure_highest_stage.png)

![新增失败的首次持续XYZ偏差](results/experiment10_phase_draft_20261009/final_analysis/new_failure_xyz_onset.png)

| 条件 | 新增失败 | 接触前 | 接触后/当时 | baseline无接触事件 | 未持续>2cm |
| --- | --- | --- | --- | --- | --- |
| Full18_D1 | 39 | 35 | 2 | 2 | 0 |
| Full18_D2 | 6 | 2 | 4 | 0 | 0 |
| Full18_D3 | 11 | 4 | 7 | 0 | 0 |
| Full18_D4 | 7 | 2 | 5 | 0 | 0 |
| Full18_D5 | 8 | 3 | 5 | 0 | 0 |
| FrozenStatic01 | 157 | 152 | 0 | 5 | 0 |
| FrozenStatic02 | 13 | 10 | 3 | 0 | 0 |
| FrozenLegal01 | 116 | 111 | 0 | 5 | 0 |
| FrozenLegal02 | 13 | 12 | 1 | 0 | 0 |
| Recover_D3_H8 | 6 | 2 | 4 | 0 | 0 |
| Recover_V16_D10_H8 | 7 | 4 | 3 | 0 | 0 |

新增失败中，[2,4,6,8,10,12] 有 152/157 在 baseline 接触前首次持续偏离，另 5 条 baseline 无接触事件；合法五层组合有 111/116 接触前偏离，另 5 条无接触事件。反之 D3/D4/D5 的接触当时或之后首次偏离为 7/11、5/7、5/8。这支持“大幅跳层并不只是精细操作出问题”，也提示低步数应重点检查接触后控制，但不能把时序相关写成机制因果。baseline 无接触事件可能是非抓取任务或接触标注边界，独立列出，不能硬归为接触前。


不能把“接近物体后才失败”直接等同为“精细操作层被删”：同一步比较包含动作节奏差异，不是动态时间对齐；轨迹偏差只覆盖双方都存在的状态，早结束尾部不补 0。XYZ 看不出完整姿态或夹爪语义，仅支持诊断线索。

以下展示真实 EEF world XYZ（米）而非相机截图；baseline 蓝色实线/圆点，变体橙色虚线/三角点，alpha=0.65，稀疏标记；同时有 XY 投影、共享步偏差与夹爪宽度。示例按新增失败中的最高 p95 偏差确定性选取，属于诊断极端样本，不冒充总体平均。视频与图共享 task/init，原视频 H264/yuv420p 全帧校验；缺少既有同步视频的示例新增左右同步视频，并以 ffprobe 帧数/20fps/H264/yuv420p 和 ffmpeg 全帧解码核查。新左右视频只展示共同步区间，不冻结结尾/不插值，完整尾部保留在原视频链接；左蓝 baseline、右橙 variant，可用步编号对照 XYZ 偏差曲线。


### Full18_D1 / libero_10 task0 init4

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_Full18_D1_libero_10_t00_i04.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task00_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket/ep004_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D1/libero_10/run/episodes/task00_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket/ep004_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/closed_loop_analysis/xyz/Full18_D1/libero_10/task00_init04_synchronized.mp4)


### Full18_D2 / libero_10 task8 init0

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_Full18_D2_libero_10_t08_i00.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D2/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/closed_loop_analysis/xyz/Full18_D2/libero_10/task08_init00_synchronized.mp4)


### Full18_D3 / libero_10 task8 init0

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_Full18_D3_libero_10_t08_i00.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D3/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_Full18_D3_libero_10_t08_i00_synchronized.mp4)


### Full18_D4 / libero_10 task8 init2

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_Full18_D4_libero_10_t08_i02.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep002_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D4/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep002_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/closed_loop_analysis/xyz/Full18_D4/libero_10/task08_init02_synchronized.mp4)


### Full18_D5 / libero_10 task8 init0

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_Full18_D5_libero_10_t08_i00.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D5/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_Full18_D5_libero_10_t08_i00_synchronized.mp4)


### FrozenStatic01 / libero_10 task4 init3

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_FrozenStatic01_libero_10_t04_i03.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task04_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate/ep003_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/FrozenStatic01/libero_10/run/episodes/task04_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate/ep003_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_FrozenStatic01_libero_10_t04_i03_synchronized.mp4)


### FrozenStatic02 / libero_10 task8 init0

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_FrozenStatic02_libero_10_t08_i00.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/FrozenStatic02/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_FrozenStatic02_libero_10_t08_i00_synchronized.mp4)


### FrozenLegal01 / libero_10 task4 init0

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_FrozenLegal01_libero_10_t04_i00.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task04_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate/ep000_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/FrozenLegal01/libero_10/run/episodes/task04_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate/ep000_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_FrozenLegal01_libero_10_t04_i00_synchronized.mp4)


### FrozenLegal02 / libero_10 task8 init1

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_FrozenLegal02_libero_10_t08_i01.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep001_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/FrozenLegal02/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep001_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_FrozenLegal02_libero_10_t08_i01_synchronized.mp4)


### Recover_D3_H8 / libero_10 task8 init0

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_Recover_D3_H8_libero_10_t08_i00.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/Recover_D3_H8/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep000_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_Recover_D3_H8_libero_10_t08_i00_synchronized.mp4)


### Recover_V16_D10_H8 / libero_10 task8 init1

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_Recover_V16_D10_H8_libero_10_t08_i01.png)

[baseline 视频](results/experiment10_phase_draft_20261009/closed_loop/Full18_D10/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep001_success/rollout_success.mp4) · [变体视频](results/experiment10_phase_draft_20261009/closed_loop/Recover_V16_D10_H8/libero_10/run/episodes/task08_put_both_moka_pots_on_the_stove/ep001_failure/rollout_failure.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_Recover_V16_D10_H8_libero_10_t08_i01_synchronized.mp4)


### AttnL16 / libero_90 task43 init4

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_AttnL16_libero_90_t043_i04.png)

[baseline 视频](results/pi0_attention_ffn_diagnosis/closed_loop_multi_init/AttnL16_T12_Init0to4_seed12345/libero_90/pairs/task043_init04/baseline/rollout.mp4) · [变体视频](results/pi0_attention_ffn_diagnosis/closed_loop_multi_init/AttnL16_T12_Init0to4_seed12345/libero_90/pairs/task043_init04/attn16/rollout.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_AttnL16_libero_90_t043_i04_synchronized.mp4)


### AttnL17 / libero_90 task19 init0

![真实末端轨迹](results/experiment10_phase_draft_20261009/final_analysis/xyz_AttnL17_libero_90_t019_i00.png)

[baseline 视频](results/pi0_attention_ffn_diagnosis/closed_loop_multi_init/AttnL17_T12_Init0to4_seed12345/libero_90/pairs/task019_init00/baseline/rollout.mp4) · [变体视频](results/pi0_attention_ffn_diagnosis/closed_loop_multi_init/AttnL17_T12_Init0to4_seed12345/libero_90/pairs/task019_init00/attn17/rollout.mp4) · [同步对比视频](results/experiment10_phase_draft_20261009/final_analysis/xyz_AttnL17_libero_90_t019_i00_synchronized.mp4)


## 8. Attention L16/L17 补充五状态：不要与标准 baseline 混算


| 条件 | 套件 | baseline | Attention skip | baseline-only / candidate-only |
| --- | --- | --- | --- | --- |
| AttnL16 | libero_90 | 22/40 | 21/40 | 5 / 4 |
| AttnL16 | libero_10 | 17/20 | 19/20 | 0 / 2 |
| AttnL16 | pooled_12_tasks | 39/60 | 40/60 | 5 / 6 |
| AttnL17 | libero_90 | 22/40 | 14/40 | 10 / 2 |
| AttnL17 | libero_10 | 17/20 | 17/20 | 1 / 1 |
| AttnL17 | pooled_12_tasks | 39/60 | 31/60 | 11 / 3 |

![Attention配对成功率](results/experiment10_phase_draft_20261009/final_analysis/attention_multi_init_success.png)

| 条件 | 12任务配对差 pp | 任务聚类95% pp |
| --- | --- | --- |
| AttnL16 | +1.7 | [-6.7, +10.0] |
| AttnL17 | -13.3 | [-25.0, -3.3] |

此区间按 suite 内任务重采样，按 8:4 的任务数加权合并，匹配 60 对的总体口径；探索性未校正，不能仅凭区间或点估计宣布 L16 安全。


补充覆盖 LIBERO-90 的 task0/1/19/22/33/38/43/70，以及 LIBERO-10 的 task0/1/3/8，每任务 init0–4。baseline/variant 共享 official initial state hash，env7、warmup10、replan12、noise12345，完整 Action Expert D10；仅跳 Attention，不跳 FFN。运行模式为 triton_eager 诊断路径。

L16 在 LIBERO-90 从 22/40 到 21/40，LIBERO-10 从 17/20 到 19/20，合并净 +1，但新增失败 5、恢复 6，不能称普遍无损或显著改进。L17 在 LIBERO-90 从 22/40 到 14/40，新增失败 10、恢复 2；LIBERO-10 净不变却也有 1 次新增失败与 1 次恢复，说明总正确数会掩盖配对改变。整 block 的离线 L17 低动作扰动不等于 Attention-only 运行可靠。

基线 39/60（65%）偏低这一诊断路径现象未完成根因闭环。与标准套件 186/200 不是同任务集合/同 runtime，不能用两者差值归因于裁剪；历史 21/100 Control 的根因也不能由本轮高标准成绩反向宣布已解决。应分离协议、任务难度、预处理/runtime parity与具体成功判据逐项验证，不把它写成论文 baseline。

## 9. 与上一轮全量标准评测的交叉核对

上一轮 Experiment 9 与本轮不合并分母。其全量报告见 [Experiment 9 最终报告](experiment9_final_analysis_report.md)：标准 Full18 1888/2000（94.4%）；D3/D5 1898/2000（94.9%）/1897/2000（94.85%），支持低降噪步数优先方向。V12 在 Object 为 0/500，不支持“删 6 层但 Object 仍可 90%+”的猜测；V13-Custom 总体 875/2000（43.75%）。

上一轮还做了 Action Expert **单层** L2/L8 裁剪：356/400（89%）/354/400（88.5%），Goal 81%/84%、LIBERO-10 均 80%，不支持直接部署。这与本轮 D2–D5 **减少降噪迭代次数、不删除专家层** 是不同实验。历史数字仅引用既有报告，不冒充本轮重新遍历 28,800 条原始数据。

## 10. 下一步建议与完成边界

优先顺序：先把完整 VLM 的 D2/D3/D4 与 D10 做小规模独立任务/独立 seed 确认和真实端到端计时；再验证任务限定两层 mask，而不是继续扩大五六层 mask。冻结候选、接受标准和非劣界限后再测，避免筛选获胜者后用同一数据证明。

阶段主线保留，但调整证据路线：预注册少量层×阶段对比、同 init 配对、控制场景扰动；以干预动作和闭环失败作为主结果，激活/抵消/能量集中度作为解释。轻量门控应测净收益和误接受率，加入目标状态/姿态/夹爪错误诊断，不能只最大化动作接受率。

并行验证目前更像一致性诊断工具；暂不优先投入更多双卡吞吐预算。短前缀恢复有“错误可恢复”的证据，但净收益小；若探索真正 async，必须单独测 stale observation、执行重叠、取消/回退、系统吞吐及双卡成本，现有 request ledger 单元测试不是已完成异步机器人试验。

| 项目 | 状态 | 不能越界的结论 |
| --- | --- | --- |
| 四套件五状态 probe / 数值 / 单层干预 | 已完成 | 非随机数据，但不是完整任意层两两矩阵 |
| 启发式候选真实组合搜索与冻结闭环 | 已完成本轮范围 | 非全局最优；实际挑战占比偏离原计划 |
| Attention/FFN 代表样本与 L16/L17 五状态 | 已完成本轮范围 | 选层/阶段覆盖不均，诊断模式不可与标准成绩混算 |
| 短前缀恢复 / Kalman / 并行验证 H4/8/12 | 已完成当前筛选 | 没有可靠净加速证明 |
| 逐阶段不同 mask 的动态闭环 | 未完成有效验证 | 当前四阶段 mask 相同，不能支持阶段路由优势 |
| 真实执行重叠 async 与 50–200ms 延迟扰动 | 未完成 | 基础设施不是系统实验 |
| Kalman Q/R、方差 floor、2cm onset 阈值敏感性 | 未完成 | 门控/首次偏差当前定义只作探索描述 |
| 全层两两 hidden 矩阵、完整 QKV/FFN gate 表征 | 未采集或未完成 | 不伪造缺失数据 |
| 历史 Control 21% / eager 65% 的原因闭环 | 尚未定位 | 高标准基线不等于已找到旧 bug |
| 本轮每 suite 50 init 全量扩展 | 按研究筛选目标未开展 | 不是遗漏失败，不能写成 500 条全量 |

## 11. 复核入口

实验方案位于工作区根目录 [idea1 实验方案](../../idea1_experiment_plan.md)。可复现脚本为 [build_final_report.py](scripts/experiment10/build_final_report.py)，执行 `python3 scripts/experiment10/build_final_report.py`。无需 GPU，不启动新实验。全部新 CSV、PNG/PDF、输入文件 SHA256 和轨迹/视频索引保存在本报告的 final_analysis 子目录。


[completed_coverage.csv](results/experiment10_phase_draft_20261009/final_analysis/completed_coverage.csv)
[paired_success_task_bootstrap.csv](results/experiment10_phase_draft_20261009/final_analysis/paired_success_task_bootstrap.csv)
[layer_prior.csv](results/experiment10_phase_draft_20261009/final_analysis/layer_prior.csv)
[numerical_layer_modality.csv](results/experiment10_phase_draft_20261009/final_analysis/numerical_layer_modality.csv)
[submodule_interventions.csv](results/experiment10_phase_draft_20261009/final_analysis/submodule_interventions.csv)
[paired_xyz_and_onset.csv](results/experiment10_phase_draft_20261009/final_analysis/paired_xyz_and_onset.csv)
[failure_highest_stage_counts.csv](results/experiment10_phase_draft_20261009/final_analysis/failure_highest_stage_counts.csv)
[parallel_verify_summary.csv](results/experiment10_phase_draft_20261009/final_analysis/parallel_verify_summary.csv)
[attention_multi_init_results.csv](results/experiment10_phase_draft_20261009/final_analysis/attention_multi_init_results.csv)
[attention_multi_init_pairs.csv](results/experiment10_phase_draft_20261009/final_analysis/attention_multi_init_pairs.csv)
[example_index.json](results/experiment10_phase_draft_20261009/final_analysis/example_index.json)
[provenance.json](results/experiment10_phase_draft_20261009/final_analysis/provenance.json)
