# EASE 两条 insight 的中期证据报告

**数据仍不完整，尚不能宣称两条 insight 已验证。** Draft 快照为 **286/360**，截取于 **2026-11-03 21:21:07 北京时间**；尚缺 74 条。VLM 发现集为 **160/160**，冻结后的独立确认仍为 **0/600**。Draft 也还没有独立 init21–24 确认结果。本文件是中期证据记录，不代表实验目标结束；后续由既有调度器和 pipeline 持续接续。

本报告只使用现有结果做 CPU 分析，未运行 GPU、未改变 runtime、冻结选择或队列。分析以 suite/task/init 配对；Spatial、Object、Goal、Long 分别对应四个 LIBERO suite，Long 即 `libero_10`。

**已支持：在当前发现样本内，纯 Draft 的来源依赖很强，FLASH 回退可掩盖这种质量损失。**

下表每一行固定同一个目标 suite 的 10 个 task、官方 init20，只比较来源 checkpoint。前三个来源已完成，Long 来源的纯 Draft 未完成；因此前三行可作同目标匹配来源与异来源比较。

| 目标（同一批 case） | Full | Spatial 纯 Draft | Object 纯 Draft | Goal 纯 Draft | Long 纯 Draft |
| --- | --- | --- | --- | --- | --- |
| Spatial | 10/10 | 9/10 | 0/10 | 0/10 | 缺失 |
| Object | 10/10 | 0/10 | 10/10 | 0/10 | 缺失 |
| Goal | 8/10 | 0/10 | 0/10 | 8/10 | 缺失 |
| Long | 9/10 | 0/10 | 0/10 | 0/10 | 缺失 |

Spatial、Object、Goal 匹配来源分别为 9/10、10/10、8/10；在同一目标上，另外两个已完成来源均为 0/10。配对发现差异为 80–100pp，描述性双侧精确 McNemar p 为 0.00195–0.00781；它们来自探索中的多重比较，不能代替独立确认。Long 目标现有三个外来源都是 0/10，但 Long 自身来源缺失，不能从这行推断“Long 匹配来源更好”。纯 Draft 没有 verify，接受率为 N/A，不能给它赋予 FLASH 接受率。

同一目标上的 FLASH 结果如下。接受率 = accepted prefix 动作数 /（12 × verify 请求数）；“拒绝→Full” = 拒绝触发的 Full 请求数 / 全部请求数；它区别于包含初始化、周期刷新和拒绝回退的“全部 Full”比例。请求只是描述性统计单位，不能当独立 episode 扩大统计样本量。

| 目标 | Draft 来源 | 成功 | 动作接受率 | 拒绝→Full / 全请求 | 全部 Full / 全请求 |
| --- | --- | --- | --- | --- | --- |
| Spatial | Spatial（匹配） | 10/10 | 62.6% | 12.0% | 41.9% |
| Spatial | Object | 10/10 | 0.0% | 47.1% | 52.9% |
| Spatial | Goal | 9/10 | 15.4% | 29.6% | 45.2% |
| Object | Spatial | 10/10 | 0.0% | 47.8% | 52.2% |
| Object | Object（匹配） | 10/10 | 61.4% | 17.1% | 41.8% |
| Object | Goal | 10/10 | 0.0% | 47.8% | 52.2% |
| Goal | Spatial | 9/10 | 9.8% | 33.9% | 46.0% |
| Goal | Object | 10/10 | 0.0% | 47.1% | 52.9% |
| Goal | Goal（匹配） | 10/10 | 71.9% | 8.7% | 39.1% |

例如固定 Object 目标：Object 来源接受率 61.4%，Spatial/Goal 来源都是 0%，但三者 FLASH 均 10/10 成功。两条零接受率条件各有 108 次 verify 全部执行零个接受动作，再触发 Full；这直接支持“FLASH 最终成功并不等于 draft 本身泛化良好”。固定 Goal 目标的 Object 来源也是接受率 0% 而 FLASH 10/10，纯 Draft 则 0/10。回退效果与单独运行的纯 Draft 路由结果一致，但不是禁用回退的单独消融，不能据此给出回退的普遍因果效应大小。

不支持/边界结果也保留：固定 Spatial 目标时 Goal 来源 FLASH 9/10，而 Object 来源 10/10；固定 Goal 目标时 Spatial 来源 FLASH 9/10、Object 来源 10/10。因此“跨 suite 必然降低 FLASH 最终成功率”不受当前数据支持。Long 目标的 Spatial/Object/Goal 来源 FLASH 为 9/10、8/10、6/10，接受率为 1.73%、0.10%、3.47%；仍不能与尚缺的 Long 匹配来源作泛化差异。

完整性：Full 40/40；Spatial/Object/Goal 三来源在 FLASH 与纯 Draft 下均各 40/40；Long 来源 FLASH 6/40、纯 Draft 0/40。Long FLASH 当前部分结果仅列在 [逐格快照](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/draft_analysis_snapshot/summary.json)，不据其部分 task 集合排名。合并分析包含真实 smoke 两目录的 8 条复用数据，没有重复条件；首输入/噪声配对无不一致。缺失的逐 case 列表见 [evidence.json](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/evidence.json)，包含尚未生成 run 目录的 40 条 Long 纯 Draft；仅看分析器的 existing-run 缺失数会漏掉它们。

**探索线索：同预算 VLM 组合质量不同，但双向 suite 偏好仍然很弱。**

VLM 三条候选共享 π0、H50/R5、同残差外推；候选每个周期为 Full-D10→Fast-D2→Fast-D2，18 层中保留 12 层。Jacobian 跳层 [2,3,5,14,16,17]，MSE [2,3,5,11,16,17]，Cosine [2,3,5,6,8,9]。因此候选之间可研究同预算组合差异；相对 Full 的收益同时涉及裁剪、D2 与刷新/补偿，不能只归因于 mask。四 suite 各 10 个 task、init30：

| 目标 | Full | Jacobian | MSE | Cosine |
| --- | --- | --- | --- | --- |
| Spatial | 10/10 | 9/10 | 8/10 | 5/10 |
| Object | 10/10 | 10/10 | 10/10 | 5/10 |
| Goal | 10/10 | 10/10 | 7/10 | 4/10 |
| Long | 8/10 | 6/10 | 7/10 | 1/10 |
| 总计 | 38/40 | 35/40 | 32/40 | 15/40 |

冻结方向仍为 **Spatial: Jacobian > MSE +10pp；Long: MSE > Jacobian +10pp**。Spatial 仅 Jacobian 成功 2 例、仅 MSE 成功 1 例；Long 仅 MSE 成功 1 例、反向 0 例，两组描述性双侧精确 McNemar p 都是 1.0。Goal 的 Jacobian > MSE +30pp 也保留，但冻结的 max-min 目标同为 10pp，按原 suite 顺序选了 Spatial，未因结果重新挑方向。

Jacobian/MSE 相对 Full 总成功率低 7.5/15pp；入选门槛只是发现成功率 ≥60% 且固定输入成本低于 Full，不代表质量非劣。Cosine 虽成本相近，只有 15/40 成功，未过质量门槛。现有数据支持“预算相同也可能质量不同”，只为“存在可复现互补路径”提供弱线索；没有评测在线选择器，也不能声称自适应切换已有收益。

冻结证据见 [selection.json](/root/nfs/embodied/benchmark/results/ease_insights_20261103/vlm/analysis/selection.json)；确认计划为 Full/Jacobian/MSE 各 200 条（40 task × init31–35），合计 600 条，见 [confirmation manifest](/root/nfs/embodied/benchmark/results/ease_insights_20261103/vlm_confirmation/manifest.json)。本次辅助状态截取 2026-11-03T13:25:31.143002+00:00：完成 receipt 为 0，不能用发现集代替独立确认。

**未支持的路径保留：Vision pilot 未复现预登记双向偏好。**

Vision Keep24 A 跳 [8,9,10]、B 跳 [5,14,23]，D10/H50/R5，新 init20 为 Full39/40、A36/40、B38/40。Goal A9/B10，历史的 A>B 方向反向；Long A7/B8，只保留一例弱差。两个 A/B 不一致 case 均为 B 成功，A-only 为 0；这不支持本组可靠互补性。各自闭环平均模型请求 Full108.15ms、A108.55ms、B110.22ms 不是同输入加速比，也没有证明两条路径更快。详见 [保留的不复现结论](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/vision_pilot/scientific_verdict.md)。

**真实速度边界：固定输入有成本优势，不等于整批闭环更快。**

VLM 的 5 个完整服务各自测四条件，先按四 suite 输入等权，再按服务等权。下表是三请求周期的完整 policy cost，包含策略内前处理、残差、模型、解码和输出传输，不含 RPC 与环境：

| 条件 | 同输入三请求周期（ms） | Full / 条件成本 |
| --- | --- | --- |
| Full | 326.03 | 1.000× |
| Jacobian | 202.87 | 1.607× |
| MSE | 201.76 | 1.616× |
| Cosine | 201.58 | 1.617× |

约 1.6× 的结论仅适用于上述固定输入协议。真实发现回合的累计 episode 墙钟依次为 Full497.51s、Jacobian572.27s、MSE488.16s、Cosine553.22s，并伴随不同失败数和轨迹长度；这些数排除准备、不是并行完成时间。它们不能把固定输入收益转换成“整批任务快 1.6×”。

FLASH 请求可能执行 0 动作，主成本采用 **全部完整 policy 请求累计 ms / 实际执行动作数**，同时报告完整 client RPC ms / action、成功数和 episode 墙钟。下表仅列匹配来源与同目标 Full，每行均 10 个相同初始 case；成本是动作加权汇总，墙钟为每 episode 均值，保留所有实测请求及其中可能的冷成本，不含单独的 run 准备：

| 目标 | 条件 | 成功 | policy ms/action | RPC ms/action | episode 墙钟均值（s） |
| --- | --- | --- | --- | --- | --- |
| Spatial | Full | 10/10 | 41.24 | 41.50 | 11.53 |
| Spatial | 匹配纯 Draft | 9/10 | 27.81 | 28.10 | 10.69 |
| Spatial | 匹配 FLASH | 10/10 | 57.57 | 58.66 | 13.44 |
| Object | Full | 10/10 | 18.21 | 18.50 | 8.70 |
| Object | 匹配纯 Draft | 10/10 | 10.96 | 11.25 | 7.25 |
| Object | 匹配 FLASH | 10/10 | 23.12 | 23.46 | 9.03 |
| Goal | Full | 8/10 | 26.25 | 26.50 | 10.21 |
| Goal | 匹配纯 Draft | 8/10 | 17.46 | 17.71 | 8.08 |
| Goal | 匹配 FLASH | 10/10 | 41.20 | 41.53 | 9.73 |

当前 12 个完整 FLASH 来源×目标格中，11 个的 policy ms/action 高于 Full；11 个的累计 episode 墙钟也更长。匹配纯 Draft 的 Full/条件 policy 成本比约 1.48–1.66×，但 Spatial 少成功 1 例，且其异来源质量崩溃。固定 Object 目标时，Goal 来源 FLASH 的接受率为 0%，policy 成本却为 21.02ms/action，低于匹配来源的 23.12ms/action；因此目前连“接受率低必然导致实测成本更高”也不能作普遍结论。全部同 case 费用比与成功数保留在 [speed_vs_full.json](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/draft_analysis_snapshot/speed_vs_full.json)。

准备成本单列：真实 FLASH smoke 为 83.62s、纯 Draft smoke 23.05s；已完整 discovery run 为约 21.05–23.05s，具体值见 [准备成本快照](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/evidence.json)。服务 readiness 之后仍可能发生 lazy graph/compile；没有可靠逐请求 cold/capture 标记，删除每个 episode 首请求不能证明热态。未来已预登记 init21、repeat-task init22–24 子集，但 repeat-task 也不等于部署稳态。已有 [外部 GPU 争用记录](/root/nfs/embodied/benchmark/results/ease_insights_20261103/resource_contention/observed_1793699156.json)，不能把共享负载墙钟归因成独占卡算法速度。

协议不能混合：Draft/FLASH 是 H50/R12、δ0.15、PF2、init20；VLM 是 H50/R5、周期 D10/D2、own-path residual、init30；Vision 则是 H50/R5、固定 vision 剪层、D10、init20。Full 基线必须在本分支和同 case 内配对；即便都叫 Full，也不能把三个分支的成功率或时延合并。

**仍未验证及既定后续。** Draft 需补齐剩余发现矩阵，再按原纯 Draft gap 规则冻结 1–2 对、用同任务新 init21–24 确认；主检验采用冻结 family 的单侧精确 McNemar 与 Holm 校正，完整矩阵之前记为 incomplete_not_tested，并报告 task-cluster 区间。VLM 沿已冻结 Spatial/Long 方向完成独立 600 条，不以新发现重选类别。两者目前都没有未见任务泛化证据；真实稳态成本、端到端完成时间与在线 selector 收益仍未验证。

统计与速度边界已预登记在 [heldout_statistics_protocol.json](/root/nfs/embodied/benchmark/preparations/ease_insights_20261103/draft/heldout_statistics_protocol.json)、[latency_protocol.json](/root/nfs/embodied/benchmark/preparations/ease_insights_20261103/draft/latency_protocol.json)；本报告不修改这些规则。

**数据与复现。**

本报告由 [build_report.py](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/build_report.py) 从 [合并分析快照](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/draft_analysis_snapshot/audit.json) 与 [辅助证据快照](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/evidence.json) 渲染。[纯 smoke 独立分析](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/draft_smoke_snapshot/audit.json) 的 8 条与发现集协议兼容，已按同条件身份合并；[仅 discovery 早一轮快照](/root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/draft_discovery_snapshot/audit.json) 排除 smoke 且截取更早，不作为报告分母。

重新采集必须使用新的报告目录，不能覆盖本时间截面；本次调用的三根输入记录在 evidence.json，使用项目原有 draft/analyze.py。只从现有保存快照重建报告时，运行：

```bash
python3 /root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim/build_report.py
```
