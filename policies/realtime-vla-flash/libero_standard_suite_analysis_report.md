# LIBERO 标准套件与 VLM 裁剪证据分析报告

日期：2026-10-04。层编号均从 1 开始；报告仅分析已有数据，不启动新评测。

## 结论先行

完整模型在标准四套件共 2,000 个 episode 上成功 1,891 次，成功率 **94.55%**。此前的 Control 21/100 来自另一套 LIBERO-90 + LIBERO-10 混合诊断任务、HDF5 初始状态和 D3 设置，不能代表标准套件 baseline。

现有证据支持将 **Attention L16** 和 **完整 block L2–L5 的单层跳过**作为下一轮候选，而不支持直接批量删除中间层。FFN L2 在扩大样本后，前 12 个动作的 NMSE 达到 0.5139，不能因为 12 个诊断 episode 表现不错，就认为它冗余。

**关键缺口：标准四套件目前只有完整模型结果，没有这些层裁剪变体的同协议成绩和配对 XYZ 轨迹。** 下文提供的裁剪成功率、层探针和轨迹来自独立诊断数据，并明确标注；它们是候选筛选证据，不是标准 benchmark 上的非劣证明。

## 1. 数据范围与可比较性

| 数据批次 | 样本与协议 | 可回答的问题 | 不能据此声称 |
| --- | --- | --- | --- |
| 标准四套件 baseline | Spatial / Object / Goal / LIBERO-10，各 10 任务 × 50 episode；共 2,000 | 完整模型标准成功率、任务难点 | 裁剪变体标准成功率 |
| 整层探针 | 分析产物实际纳入 12 个真实观测，来自 3 个诊断任务 | 层输出相似度、固定输入下单层跳过的动作敏感性 | 全标准套件的层冗余规律 |
| 扩展 Attention / FFN 探针 | 12 任务：LIBERO-90 8 个、LIBERO-10 4 个；64 个语义阶段观测 × 3 个动作噪声种子，共 192 次探针 | 子模块表征改变与动作敏感性 | 192 个独立任务的统计结论 |
| 子模块闭环诊断 | 同一批 12 任务，每任务一个官方初始状态；Full18、skip-attn16、skip-ffn2，D10 | 小样本配对结果与真实末端轨迹 | 标准套件上的稳定收益 |
| 历史整层裁剪 | LIBERO-90 90 任务 + LIBERO-10 10 任务，各一个冻结 HDF5 初始状态，D3 | 旧方案的退化与历史轨迹 | 与新标准 D10 的直接因果比较 |

标准运行使用 Full18 VLM + Full18 Action Expert、D10、Triton，每轮强制完整推理；环境 seed=7，warmup=10，replan=12，采用官方初始状态。评测数据目录见文末。

### 1.1 探针不是随机图像或随机初始化模型

输入是已运行轨迹里的真实相机观测、机器人状态和任务文本，使用训练后的模型权重。观测按 initial、approach、contact、grasp/manipulate、transport、goal alignment 等语义阶段选取，属于**有目的的代表性诊断样本**，不是均匀随机覆盖整个标准评测集。

扩展探针的随机性只来自动作扩散的 Gaussian noise；种子为 12345、23456、34567。同一个观测的 baseline 与干预变体共享噪声，以隔离跳过模块造成的差异。不能把多个阶段或多个噪声种子当成独立任务。

整层分析以 [layer_probe_report.json](results/pi0_vlm_pruning_diagnosis/layer_analysis/layer_probe_report.json) 的 12 个样本和 CSV 的 `sample_count=12` 为准。当前采集目录 manifest 列出 13 个探针，包含后续增加的记录；它与已有分析产物不是完全同一版本。不能把旧分析擅自写成 13 样本结果，后续应固定样本清单并重建分析。

## 2. VLM backbone 层间余弦相似度

### 2.1 测量方法

记录 H0（层输入）及 H1…H18（各层输出）。对同一个 token 的隐藏向量计算：

`cos(H_i[t], H_j[t]) = dot(H_i[t], H_j[t]) / (norm(H_i[t]) * norm(H_j[t]))`

先在同一模态 token 内平均，再跨观测汇总。agent-view 和 wrist-view 各 256 个视觉 token，language token 数量可变，分别分析，避免视觉 token 数量掩盖语言变化。热图包含全部层对；裁剪判断重点参考相邻输入/输出 `H_(l-1) → H_l`。

同时测量 `relative_residual = ||H_l-H_(l-1)||_F / ||H_(l-1)||_F`、中心化 linear CKA，以及**实际单层跳过后的动作误差**。高余弦只表示方向接近，并不保证向量幅度、下游 KV 或控制动作不变。

![VLM 全层对相似度热图](results/pi0_vlm_pruning_diagnosis/layer_analysis/layer_similarity_heatmap.png)

以下是 agent-view 的相邻层余弦；不是跨模态综合值：

| 层 | 相邻余弦 | 相对残差 | 解读 |
| --- | ---: | ---: | --- |
| L1 | 0.239344 | 2.216180 | 强表征变换，保留 |
| L2 | 0.998430 | 0.067591 | 低更新，候选 |
| L3 | 0.999135 | 0.049779 | 低更新，候选 |
| L4 | 0.998656 | 0.063518 | 低更新，候选 |
| L5 | 0.998773 | 0.059416 | 低更新，候选 |
| L6 | 0.998505 | 0.081672 | 高相似，但仍需动作验证 |
| L7 | 0.997672 | 0.080673 | 同上 |
| L8 | 0.998114 | 0.071811 | 同上 |
| L9 | 0.997231 | 0.091415 | 同上 |
| L10 | 0.996705 | 0.097384 | 同上 |
| L11 | 0.994738 | 0.122870 | 更新开始增大 |
| L12 | 0.989708 | 0.228489 | 不宜仅按余弦裁剪 |
| L13 | 0.978241 | 0.293577 | 表征变化增大 |
| L14 | 0.950309 | 0.430526 | 表征变化增大 |
| L15 | 0.967298 | 0.871537 | 方向接近不等于幅度变化小 |
| L16 | 0.951933 | 0.230195 | 整层和 Attention 必须分开判断 |
| L17 | 0.579082 | 2.018403 | 强表征变换，谨慎保留 |
| L18 | 1.000000 | 0.000000 | KV-only 实现行为，不是可删证明 |

来源：[完整模态指标 CSV](results/pi0_vlm_pruning_diagnosis/layer_analysis/layer_hidden_metrics.csv)、[全层相似度矩阵 NPZ](results/pi0_vlm_pruning_diagnosis/layer_analysis/layer_similarity_matrices.npz)。

### 2.2 哪些层有裁剪依据？

固定观测与动作噪声，逐次跳过一个完整 block，与 Full18 比较。NMSE 为 `mean((a_variant-a_base)^2) / max(mean(a_base^2), epsilon)`，是动作空间误差，不是实测末端位移。

| 单层候选 | 综合风险排序 | 动作 NMSE | 夹爪命令符号翻转率 | 跨模态汇总相对残差 |
| --- | ---: | ---: | ---: | ---: |
| L3 | 1 | 0.009051 | 0.83% | 0.061687 |
| L2 | 2 | 0.008291 | 0.67% | 0.085316 |
| L4 | 3 | 0.011266 | 0.83% | 0.073469 |
| L5 | 4 | 0.015855 | 0.67% | 0.070759 |
| L1（反例） | 17 | 0.534477 | 40.83% | 1.887557 |

L2–L5 的依据是“高相似度 + 低残差 + 单层跳过低动作扰动”同时成立，而不是仅看热图。L2、L3、L4 的动作 NMSE bootstrap 区间分别约为 [0.0036, 0.0138]、[0.0020, 0.0216]、[0.0032, 0.0250]，重叠明显，不能过度解读排序。样本小，筛选证据仍然有限。

L18 在当前 Triton prefix 实现中只生成 Action Expert 需要的 K/V，没有继续更新隐藏状态，所以 H18=H17。**删除它可能损坏下游 KV，不能据余弦=1 删除。** 单层安全也不意味着多层联合安全，误差与缓存影响可能叠加。

![整层跳过的动作敏感性](results/pi0_vlm_pruning_diagnosis/layer_analysis/layer_action_sensitivity.png)

## 3. Attention 与 FFN 的不同特征

每层分别记录 Attention 前、Attention 残差相加后、FFN 残差相加后的隐藏状态。Attention 测前两者，FFN 测后两者；按照模态和语义阶段分析余弦、残差、更新范数。

Attention 干预保留原始 Q/K/V，只跳过 Attention 输出路径及其投影更新；FFN 干预跳过 FFN 残差更新。因此 **skip-attn16 不等于删除整个 L16**。扩展测试覆盖 L1–L17，L18 不作为普通 Attention/FFN 候选。

扩展探针保存原始 32 维动作，并比较反归一化后的前 7 维控制命令。报告分别给出完整 50 动作 chunk 与前 12 个动作，因为 replan=12，后者更贴近实际执行窗口。普通推理与带探针 baseline 的已审计 parity 最大误差为 0。置信区间按任务聚类：先在任务内平均，再对任务均值做 2,000 次 bootstrap，避免阶段和种子造成伪样本量。

![Attention / FFN 分层余弦](results/pi0_attention_ffn_diagnosis/analysis_expanded_12task_three_seeds/submodule_cosine_by_layer.png)

![Attention / FFN 动作敏感性](results/pi0_attention_ffn_diagnosis/analysis_expanded_12task_three_seeds/submodule_action_sensitivity.png)

| 子模块 | 全部子模块风险排序 | 完整 chunk NMSE | 前 12 动作 NMSE | 排序表相对残差 |
| --- | ---: | ---: | ---: | ---: |
| Attention L16 | 1 | 0.029375 | 0.024081 | 0.101056 |
| Attention L15 | 2 | 0.083580 | 0.090413 | 0.199766 |
| FFN L2 | 18 | 0.159844 | 0.513863 | 0.086963 |
| FFN L16 | 23 | 0.164990 | 0.248535 | 0.344920 |

这里残差来自候选排序的任务/阶段/模态风险汇总口径，不应与上一节单个模态均值直接比较。原始分阶段值见 [submodule_summary.csv](results/pi0_attention_ffn_diagnosis/analysis_expanded_12task_three_seeds/submodule_summary.csv)；排序和区间见 [candidate ranking CSV](results/pi0_attention_ffn_diagnosis/analysis_expanded_12task_three_seeds/submodule_candidate_ranking.csv)。

核心发现：Attention L16 的前 12 动作 NMSE 均值低，但其 95% 区间约 [0.001602, 0.068379]，存在任务差异，只能列为优先验证候选。L1 Attention 在 agent-view 多个阶段的余弦约 0.345–0.353、残差约 1.69–1.74，承担显著变换。FFN L2 虽然隐藏更新小，动作影响却大，直接反驳“高隐藏相似度必然可删”。L16 Attention 和 FFN 的敏感性也不同，应做子模块级筛选，不能把二者一并删除。

## 4. 成功率：标准结果与诊断结果分开看

### 4.1 标准套件完整模型

| 套件 | 成功 / 总数 | 成功率 | 失败数 |
| --- | ---: | ---: | ---: |
| LIBERO-Spatial | 477 / 500 | 95.4% | 23 |
| LIBERO-Object | 496 / 500 | 99.2% | 4 |
| LIBERO-Goal | 484 / 500 | 96.8% | 16 |
| LIBERO-10 | 434 / 500 | 86.8% | 66 |
| 合计 | 1,891 / 2,000 | 94.55% | 109 |

各任务成功数（任务 ID 从 0 到 9，分母均为 50）：

| 套件 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial | 49 | 45 | 50 | 49 | 46 | 46 | 50 | 46 | 48 | 48 |
| Object | 49 | 50 | 50 | 49 | 50 | 49 | 50 | 50 | 50 | 49 |
| Goal | 49 | 49 | 50 | 43 | 50 | 48 | 49 | 50 | 49 | 47 |
| LIBERO-10 | 45 | 49 | 47 | 48 | 42 | 48 | 38 | 48 | 26 | 43 |

LIBERO-10 占全部失败的 66/109=60.6%。其中 task8 “put both moka pots on the stove” 为 26/50=52%，task6 “put the white mug on the plate and put the chocolate pudding to the right of the plate” 为 38/50=76%。这说明长程多物体任务更值得单独诊断，但仅凭成功数不能判定失败发生在抓取还是放置阶段。

**skip-attn16、skip-ffn2、skip-L2/3/4/5、V16/V14/V12/V10-Mid 的上述四套件成绩均为未评测，不能填入推算值。**

### 4.2 子模块闭环诊断：12 个配对 episode，D10

| 条件 | 成功率 | 相对 baseline | 共同成功 | 新增失败 | 恢复成功 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full18 baseline | 7/12 = 58.3% | — | — | — | — |
| Skip Attention L16 | 9/12 = 75.0% | +16.7 个百分点 | 7 | 0 | 2 |
| Skip FFN L2 | 10/12 = 83.3% | +25.0 个百分点 | 7 | 0 | 3 |

两个变体的精确双侧 McNemar 检验分别为 p=0.5 和 p=0.25，均不足以证明提升。零新增失败也不等于风险为零；固定少量任务、一个状态和一个种子存在选择与偶然性。尤其 FFN L2 的闭环小样本结果与扩大后的动作敏感性不一致，需要扩大同协议测试，而不是忽略后者。

来源：[Attention 配对审计](results/pi0_attention_ffn_diagnosis/closed_loop/attn16_12task_paired_audit.json)、[FFN 配对审计](results/pi0_attention_ffn_diagnosis/closed_loop/ffn2_12task_paired_audit.json)。这批 eager 诊断不是标准 Triton 成绩。

### 4.3 历史整层裁剪：100 个混合诊断 episode，D3

| 条件 | 保留层数 | 成功率 | 相对同批 Control |
| --- | ---: | ---: | ---: |
| Control Full18 | 18 | 21/100 = 21% | — |
| V16：保留前 8 + 后 8 | 16 | 16/100 = 16% | -5 个百分点 |
| V14：保留前 7 + 后 7 | 14 | 7/100 = 7% | -14 个百分点 |
| V12：保留前 6 + 后 6 | 12 | 1/100 = 1% | -20 个百分点 |
| V10-Mid：前/中/后组合 | 10 | 1/100 = 1% | -20 个百分点 |

来源：[历史 condition_results.csv](results/pi0_vlm_backbone_fine_pruning/condition_results.csv)。V16 删除 L9–L10，V14 删除 L8–L11，V12 删除 L7–L12；这些不是本次探针优先推荐的 L2–L5。旧结果支持反对激进中层连续裁剪，但不能证明新的单层候选一定有效。

## 5. 真实末端 XYZ：baseline 蓝线，变体橙线

绘图使用 **Matplotlib**，位置来自 simulator 的 `robot0_eef_pos` / telemetry `eef_pos`，是 MuJoCo 世界坐标，单位米。不是相机截图，也不是把动作命令积分成位置。每张图包含 3D 轨迹、XY/XZ/YZ 投影及同环境步的欧氏偏差；配对视频把场景和轨迹一起展示。

偏差定义：`d_t = ||xyz_variant(t)-xyz_baseline(t)||_2`，只统计双方都存在的状态。某一方成功提前结束后的未配对尾段不填充、不混入均值。共同步比较会包含速度/节奏差异，因此大偏差不自动等于空间路径错误；应结合阶段与视频判断。2 cm 持续至少 5 个状态的首次偏离只是诊断阈值，不是失败因果判据。

### 5.1 D10 子模块对比：同任务同官方状态同噪声

下面两图都是 LIBERO-10 task0、official init0、noise seed12345；baseline 与变体均成功：

![Full18 vs Skip Attention L16，真实末端轨迹](results/pi0_attention_ffn_diagnosis/closed_loop_gallery_d10_all24/pairs/libero_10_task000_official0_seed12345/skip_attn16.png)

[Attention L16 配对视频](results/pi0_attention_ffn_diagnosis/closed_loop_gallery_d10_all24/pairs/libero_10_task000_official0_seed12345/skip_attn16.mp4)。平均偏差 1.58 cm，p95 为 4.13 cm，最大 4.61 cm，状态索引 196；最大偏差处双方均标记为 goal alignment。

![Full18 vs Skip FFN L2，真实末端轨迹](results/pi0_attention_ffn_diagnosis/closed_loop_gallery_d10_all24/pairs/libero_10_task000_official0_seed12345/skip_ffn2.png)

[FFN L2 配对视频](results/pi0_attention_ffn_diagnosis/closed_loop_gallery_d10_all24/pairs/libero_10_task000_official0_seed12345/skip_ffn2.mp4)。平均偏差 2.70 cm，p95 为 10.61 cm，最大 13.41 cm，状态索引 259；最大偏差处双方均为 goal alignment。两者最后都成功，说明单一最大偏差不能当作失败标签。

[全部 24 组配对 XYZ 与视频画廊](results/pi0_attention_ffn_diagnosis/closed_loop_gallery_d10_all24/index.html)。画廊 manifest 记录 24/24 渲染成功，既有视频制作流程做过完整 ffmpeg 解码检查；本次报告没有重新编码视频。

### 5.2 历史整层变体对比：baseline 是旧 Control / D3

以下用于展示各变体的真实偏差，不与新标准 D10 baseline 混配。案例任务不同，不能把下面的偏差大小视为严格的变体排序，也不是全样本均值。

| 变体与案例 | 平均偏差 | p95 | 最大偏差 | 对应视频 |
| --- | ---: | ---: | ---: | --- |
| V16 / LIBERO-10 task3 | 2.20 cm | 5.76 cm | 6.60 cm | [视频](results/pi0_attention_ffn_diagnosis/historical_xyz_gallery/videos/V16__libero_10_task003.mp4) |
| V14 / LIBERO-10 task2 | 16.99 cm | 40.28 cm | 41.95 cm | [视频](results/pi0_attention_ffn_diagnosis/historical_xyz_gallery/videos/V14__libero_10_task002.mp4) |
| V12 / LIBERO-10 task1 | 24.12 cm | 43.46 cm | 43.84 cm | [视频](results/pi0_attention_ffn_diagnosis/historical_xyz_gallery/videos/V12__libero_10_task001.mp4) |
| V10-Mid / LIBERO-10 task1 | 20.59 cm | 27.22 cm | 28.01 cm | [视频](results/pi0_attention_ffn_diagnosis/historical_xyz_gallery/videos/V10-Mid__libero_10_task001.mp4) |

![V16 vs 历史 Control，XYZ](results/pi0_attention_ffn_diagnosis/historical_xyz_gallery/plots/V16__libero_10_task003.png)

![V14 vs 历史 Control，XYZ](results/pi0_attention_ffn_diagnosis/historical_xyz_gallery/plots/V14__libero_10_task002.png)

![V12 vs 历史 Control，XYZ](results/pi0_attention_ffn_diagnosis/historical_xyz_gallery/plots/V12__libero_10_task001.png)

![V10-Mid vs 历史 Control，XYZ](results/pi0_attention_ffn_diagnosis/historical_xyz_gallery/plots/V10-Mid__libero_10_task001.png)

[历史全部 XYZ 画廊](results/pi0_attention_ffn_diagnosis/historical_xyz_gallery/index.html)。历史 baseline 标签来自调用者；绘图脚本本身不认证 baseline 正确性，需结合回放审计使用。已有 86/86 simulator 回放匹配原标签与步数的审计，不能因此把旧协议升级为标准协议。

## 6. 为什么之前 Control 只有 21%？查明与未查明的边界

已查明的是**评测口径不同**，不是“标准 baseline 只有 21%”：旧组以 LIBERO-90 为主，使用每任务一个 HDF5 初始状态和 D3；新组是标准四套件每任务 50 个官方状态、D10。任务难度分布、初始状态、扩散步数和执行实现同时改变，不能把 21%→94.55% 的差额归因于单一改动。

同一旧 100 样本改为 D10 后仅 24/100；其中 LIBERO-90 为 16/90，LIBERO-10 为 8/10。D3 vs D10 的配对计数为共同成功 18、共同失败 73、仅 D3 成功 3、仅 D10 成功 6，McNemar p≈0.5078。这反驳“仅增加扩散步数即可解决旧低成功率”。

**尚未找到唯一根因。** 任务/状态分布与低成功率有明显关联，但仍缺少在同任务下、只改变初始状态来源或执行实现的受控比较。不能宣称 HDF5 状态有问题、模型损坏或视频保存导致控制失败，除非有专门证据。

此前 “43/79 接触前失败，34/79 进入抓取或更后阶段” 对应旧 Full18 Control/D3 的 79 个失败：initial 2、approach 41、contact 2、grasp 14、transport 11、goal 4、final 5。剩余 2 个是 contact 阶段。它既不是裁剪组合的失败统计，也不是新标准四套件 109 个失败的阶段分布；阶段标签仍是启发式判定，不是因果证明。

## 7. 下一步建议与额外注意事项

1. **先扩大标准轨迹探针，再正式评测候选。** 从四套件多任务、多状态、成功/失败轨迹及关键阶段分层取样；记录固定清单、模型 hash、代码版本与噪声种子，重新生成整层和子模块指标。当前探针不能代表完整标准分布。
2. **候选分开验证。** 优先 skip-attn16，其次单独 skip-L2/L3/L4/L5；Attention L15 为次级候选。暂不优先 FFN L2，不直接组合删除，也不删除 L18 KV 路径。
3. **同协议配对标准 benchmark。** 使用现有 baseline 的相同任务、官方初始状态、种子、D10、replan 和时间上限；每个变体保存真实 EEF XYZ、夹爪状态、动作与视频。验证 eager 与部署 Triton 的干预语义及 parity。现有两类 baseline 不互换。
4. **成功率与阶段双重验收。** 报告套件级成功率差、共同成功/共同失败/新增失败/恢复成功，任务聚类置信区间及预先约定的非劣阈值；专门检查接触、抓取、目标对齐的新增失败。不要用少量“恢复成功”覆盖新增失败风险。
5. **量化轨迹但不把偏差等同于失败。** 同步步偏差之外，补充按接触/抓取事件对齐的偏差、末端到目标距离、夹爪开合时机及成功任务耗时；首个持续偏离与失败阶段并列展示。
6. **性能必须另做公平计时。** 排除首次编译与热身，固定 GPU 占用和 batch，统计推理调用 p50/p95、真实执行动作吞吐及模块耗时。旧 V12 的延迟异常不能证明层越少越慢；旧 D3 延迟也不能与新 D10 混比。成功率和算力收益都达标才推荐部署。
7. **VLM 与 Action Expert 不混淆。** 另一个 200 真实观测的 early-exit 探针属于 Action Expert，不是本报告的 VLM 相似度证据；不能用其层号或统计替换 VLM 的结果。

## 8. 原始数据与复核入口

标准 baseline 原始日志（每份 500 条 episode）：

- [Spatial episode_log.json](data/test/pi0_paper_suite_libero_spatial_50x_d10_20261003/episode_log.json)
- [Object episode_log.json](data/test/pi0_paper_suite_libero_object_50x_d10_20261003/episode_log.json)
- [Goal episode_log.json](data/test/pi0_paper_suite_libero_goal_50x_d10_20261003/episode_log.json)
- [LIBERO-10 episode_log.json](data/test/pi0_paper_suite_libero_10_50x_d10_20261003/episode_log.json)

各运行目录包含 manifest 与每个 episode 的 MP4 / trace / infer 日志；baseline 视频不是本报告所展示的裁剪配对视频。

方法代码与旧诊断报告：

- [整层指标实现](scripts/experiment6/layer_metrics.py)
- [Attention / FFN 采集](scripts/experiment8/collect_submodules.py)
- [子模块分析与任务聚类 bootstrap](scripts/experiment8/analyze_submodules.py)
- [Matplotlib 真实 XYZ 绘图](scripts/experiment8/plot_paired_xyz.py)
- [完整子模块诊断报告](results/pi0_attention_ffn_diagnosis/final_report.md)
- [整层诊断报告](results/pi0_vlm_pruning_diagnosis/experiment6_report.md)

本 Markdown 在仓库根目录；图、视频和原始实验结果仍在已被 gitignore 排除的 results/data 目录中。单独提交或转发 Markdown 不会自动携带这些图片和视频，需要同时保留相对路径对应的附件。
