# EASE 等待 GPU 期间的中期证据更新

**仍未完成，不作独立验证结论。** Draft 固定截面为 **356/360**（2026-11-04 00:50:32 北京时间）；VLM 发现 **160/160**，确认结果 receipt **0/600**（辅助状态截取 2026-11-03T16:52:56.985954+00:00）。本次只做 CPU 汇总；原 21:21 快照和科学数据均保留。

缺失仅为 **Long 来源纯 Draft → Long task06/init20、Long task07/init20、Long task08/init20、Long task09/init20**。该 run 的 36 条有效 case、899 次请求已保留；canonical run result 当前不存在属于恢复状态，不能把保留结果当作失效或把缺失当策略失败。pipeline 仍处于发现/确认等待阶段；scheduler246695、pipeline255641、observer257177 在辅助截取时存活。目标继续 active，按既定安排等可用 GPU 接续，不标为 blocked。

**方法边界。** 当前训练/serving代码共享 base assets/physical-intelligence/libero 的归一化合同，未发现source专属变换或实际mismatch；但发布head缺少原训练checkpoint、norm hash及config快照，原cache也缺失，因此本轮只能说明固定共同推理合同下的发布head迁移表现，不能称已完全排除历史预处理混淆。teacher_zero_noise仅指训练缓存，官方online默认同样使用标准正态噪声；本轮显式NumPy case/replan配对高斯并非新增zero-vs-random错配，但RNG及CPU→GPU路径不同，不能称官方默认数值或时延的精确复现。policy_wall_ms只包围native original_infer，CPU种子生成和输入hash在计时外；完整client RPC包含wrapper与传输，故分别标为native-policy成本与完整RPC成本，指标值不变。 见[来源归一化与噪声审计](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/source_normalization_audit/README_20261104.md)。

**纯 Draft：同目标比较支持发现集来源依赖，Long 匹配格仍不完整。** 每行固定目标 suite/task/init；四个来源只改变 draft checkpoint，纯 Draft 无 verify，接受率与 Full fallback 为 N/A。

| 目标 | Full | Spatial 纯 Draft | Object 纯 Draft | Goal 纯 Draft | Long 纯 Draft |
| --- | --- | --- | --- | --- | --- |
| Spatial | 10/10 | 9/10 | 0/10 | 0/10 | 0/10 |
| Object | 10/10 | 0/10 | 10/10 | 0/10 | 0/10 |
| Goal | 8/10 | 0/10 | 0/10 | 8/10 | 0/10 |
| Long | 9/10 | 0/10 | 0/10 | 0/10 | 2/6 **未完成** |

Spatial/Object/Goal 匹配来源为 9/10、10/10、8/10，而同一目标的其余三来源均为 0/10。Long 匹配来源只有 task00–05 的 2/6，**不与完整 10-case 格排名、不冻结候选，也不把缺失4例补成失败**。发现结束后仍须按既定规则冻结，再做独立 init21–24 确认。

**FLASH：16格全部完整。** 接受率 = 接受前缀动作数 /（12×verify请求数）；“拒绝Full” = 拒绝触发的 Full 请求 / 全部请求；“全部Full”还包括初始化和周期刷新。native-policy与完整client RPC成本分别累计各自调用耗时并除以实际执行动作数，均包含执行零动作的拒绝请求。

| 目标 | Draft 来源 | 成功/n | 动作接受率 | 拒绝Full | 全部Full | native-policy ms/action | 完整client RPC ms/action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spatial | Spatial（匹配） | 10/10 | 62.62% | 11.97% | 41.88% | 57.57 | 58.66 |
| Spatial | Object | 10/10 | 0.00% | 47.06% | 52.94% | 49.01 | 49.45 |
| Spatial | Goal | 9/10 | 15.37% | 29.65% | 45.23% | 39.87 | 40.36 |
| Spatial | Long | 9/10 | 25.45% | 21.88% | 42.71% | 63.70 | 64.15 |
| Object | Spatial | 10/10 | 0.00% | 47.79% | 52.21% | 28.13 | 28.61 |
| Object | Object（匹配） | 10/10 | 61.37% | 17.12% | 41.78% | 23.12 | 23.46 |
| Object | Goal | 10/10 | 0.00% | 47.79% | 52.21% | 21.02 | 21.48 |
| Object | Long | 10/10 | 6.39% | 33.33% | 45.63% | 27.42 | 27.92 |
| Goal | Spatial | 9/10 | 9.85% | 33.93% | 45.98% | 46.63 | 47.16 |
| Goal | Object | 10/10 | 0.00% | 47.13% | 52.87% | 52.21 | 52.67 |
| Goal | Goal（匹配） | 10/10 | 71.90% | 8.70% | 39.13% | 41.20 | 41.53 |
| Goal | Long | 9/10 | 22.59% | 22.27% | 42.65% | 50.97 | 51.38 |
| Long | Spatial | 9/10 | 1.73% | 45.31% | 48.98% | 22.11 | 22.61 |
| Long | Object | 8/10 | 0.10% | 48.59% | 50.80% | 20.77 | 21.22 |
| Long | Goal | 6/10 | 3.47% | 43.97% | 48.70% | 16.52 | 16.99 |
| Long | Long（匹配） | 9/10 | 59.22% | 13.33% | 37.46% | 25.05 | 25.36 |

固定 Object 目标的 Spatial/Goal 来源接受率都是 0%，FLASH 仍各10/10；两者各108次 verify 全部接受零动作并触发 Full。匹配来源接受率61.37%、成功10/10。这支持“FLASH 最终成功可以掩盖 draft 自身质量损失”，不代表异来源必然降低最终成功，也不构成禁用回退的单独因果消融。

**实际成本仍有限制。** 以下 Full 与各格使用同分支、同目标 case；完整速度比和双方成功数见 [speed_vs_full.json](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim_20261104_waiting/draft_analysis_snapshot/speed_vs_full.json)。

| 目标 | Full 成功/n | Full native-policy ms/action | Full 完整client RPC ms/action | Full episode墙钟均值(s) |
| --- | --- | --- | --- | --- |
| Spatial | 10/10 | 41.24 | 41.50 | 11.53 |
| Object | 10/10 | 18.21 | 18.50 | 8.70 |
| Goal | 8/10 | 26.25 | 26.50 | 10.21 |
| Long | 9/10 | 14.23 | 14.47 | 13.69 |

16个完整FLASH格中，**15/16** 的 native-policy ms/action 高于 Full，**15/16** 的累计 episode墙钟更长；不能宣称 FLASH 已获得实际加速。成本保留全部请求，不以首请求删除或低延迟筛选制造热态；服务准备与潜在 lazy compile/capture、外部GPU争用仍限制解释。请求均值和GPU-stage时间不等于每执行动作成本或端到端完成时间。

实测成本并不随接受率单调变化，匹配来源也并非每个目标里最便宜：固定 Object 目标，匹配来源为23.12ms/action，Goal来源为21.02ms/action，双方均10/10成功，但接受率分别为61.37%与0%。这些发现支持 draft 来源依赖与 Full 回退掩盖质量损失；尚不能把实测成本差全部因果归于泛化下降，也不能声称原本存在的加速收益被跨suite切换消除，因为匹配FLASH条件本身也慢于同目标Full。冷启动、任务轨迹和同卡争用等仍限制解释，现有数据未识别具体主因。

**VLM：原 primary 方向不变，Goal 为确认前登记的补充。**

| 目标 | Full | Jacobian | MSE | Cosine |
| --- | --- | --- | --- | --- |
| Spatial | 10/10 | 9/10 | 8/10 | 5/10 |
| Object | 10/10 | 10/10 | 10/10 | 5/10 |
| Goal | 10/10 | 10/10 | 7/10 | 4/10 |
| Long | 8/10 | 6/10 | 7/10 | 1/10 |
| 合计 | 38/40 | 35/40 | 32/40 | 15/40 |

原 primary 保持 Spatial: Jacobian>MSE +10pp、Long: MSE>Jacobian +10pp；这两项仍只是弱发现信号。Goal: Jacobian>MSE +30pp 在发现后、任何独立确认观测前登记为补充，利用原600例矩阵内的Goal案例；必须同时报告三项及原primary失败情形，按补充协议报告三项Holm结果，不以Goal替换Spatial、不依据确认中途结果改方向。见 [冻结选择](/root/nfs/embodied/benchmark/results/ease_insights_20261103/vlm/analysis/selection.json)、[Goal补充协议](/root/nfs/embodied/benchmark/results/ease_insights_20261103/vlm_confirmation/supplementary_goal_contrast.json)。

VLM三个候选均保留12/18层、周期Full-D10→Fast-D2→Fast-D2、H50/R5及同残差外推。5服务同输入完整policy周期成本为Full326.03ms、Jacobian202.87ms、MSE201.76ms、Cosine201.58ms，Full/候选约1.607/1.616/1.617×；范围不含RPC和环境，也不证明质量非劣或闭环同比加速。Draft/FLASH为H50/R12，不能混用这两个分支的Full分母。

**补充上下文及反证保留。**

[历史扰动探索](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/perturbation_context/README_zh.md)在已验收175例中为V14 89/175、匹配纯draft54/175，配对差+20.0pp（Full138/175）。这是已看过的描述性历史结果，不是本轮确认；不能证明无扰动→有扰动的因果降幅，也不能外推所有trained draft或完整FLASH。V14是D10/H50/R5固定路径，与本轮V12周期路径和纯draft并非同预算比较。本报告只引用既有统计，没有重采样。

[首次Full身份审计](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/target_identity/README.md)覆盖160对：输入/noise/路由及base元数据一致，159对raw/decoded动作前缀精确一致；Spatial来源→Spatial task1/init20一对超过1e-5容差，raw最大差0.0078125、decoded最大差0.00390387135，各28/84元素超限。200条输出独立解码重建全部通过（最大误差5.55e-17），支持共享归一化，但不能声称所有首次Full动作数值一致。[单例边界与敏感性](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/target_identity/materiality.md)保留主分析该例，原因未知；0步夹爪符号变化和双方最终成功都不能证明差异无因果影响。

[Vision负结果](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/vision_pilot/scientific_verdict.md)保持Full39/40、A36/40、B38/40；Goal的预登记A>B没有复现，两个不一致case均B-only，未支持这组可靠互补性。未把该结果改写为VLM或在线selector的确认。

**文件与复现。** [evidence.json](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim_20261104_waiting/evidence.json)保存缺失case、16格指标、固定选择、补充协议和状态；[分析快照](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim_20261104_waiting/draft_analysis_snapshot/audit.json)合并真实两处smoke与discovery，条件无重复、首输入/noise无不一致，但动作数值一致性以单例审计边界为准。[生成脚本](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim_20261104_waiting/build_report.py)只读保存快照重建；重新采集须使用新目录，不能覆盖本截面或旧interim/。

```bash
python3 /root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim_20261104_waiting/build_report.py
```
