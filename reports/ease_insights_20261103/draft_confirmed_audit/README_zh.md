# Draft独立确认结果审阅

**两条预先冻结的纯draft来源差异均在同任务的新初态上复现；完整FLASH的高成功率仍能掩盖纯draft的跨suite局限。** Discovery360与heldout400已完成。确认集每方向为10个task×init21–24×5条件，共400条；没有遗漏或按成功筛样本。选择于2026-11-04 01:18:19 UTC冻结，早于最早确认run的02:05:22 UTC。

| 固定目标／外来来源 | 匹配纯draft | 外来纯draft | 匹配-only／外来-only | 差值 | Holm单侧p | 母task聚类95%区间 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Object／Spatial | 40/40 | 0/40 | 40/0 | +100pp | 1.81899e-12 | [100,100]pp |
| Spatial／Object | 35/40 | 0/40 | 35/0 | +87.5pp | 2.91038e-11 | [80,95]pp |

主检验是冻结两项family的case-level单侧exact McNemar＋Holm，已独立重算；不混入发现集。任务内4初态可能相关，Holm不能修复该相关；10个task的cluster区间是补充稳健性信息。Object的[100,100]只因全部观测差值相同，不表示总体效果没有不确定性。这支持这两对来源在**相同任务的新初态**上复现，不能外推未见任务、所有trained draft或所有suite。

下表为完整init21–24的FLASH与同case Full。接受率按累计接受动作／(12×verify请求)；两个Full比例均以全部请求为分母，“全部Full”还含初始化及周期Full。

| 目标 | 路由／来源 | 成功 | 动作接受率 | 拒绝→Full | 全部Full | native infer ms/action | RPC ms/action |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Object | FLASH／Object匹配 | 40/40 | 71.894% | 13.465% | 40.754% | 7.185 | 7.512 |
| Object | FLASH／Spatial外来 | 39/40 | 0% | 47.886% | 52.114% | 9.486 | 9.954 |
| Object | Full | 40/40 | N/A | N/A | 100% | 8.234 | 8.504 |
| Spatial | FLASH／Spatial匹配 | 39/40 | 56.417% | 13.949% | 41.061% | 12.492 | 12.907 |
| Spatial | FLASH／Object外来 | 39/40 | 0% | 47.167% | 52.833% | 15.608 | 16.101 |
| Spatial | Full | 37/40 | N/A | N/A | 100% | 11.419 | 11.687 |

两个外来FLASH条件分别有453与333次verify，接受动作均为0，仍分别39/40成功。匹配/外来FLASH成功差仅Object 1/0、Spatial 1/1个discordant case；不能声称FLASH成功率显著下降。回退与每动作成本差是描述性次要结果，也不构成禁用回退的因果消融。

纯draft接受率与回退机制为N/A。其native infer/RPC每动作成本在Object目标为匹配4.132/4.407、外来1.978/2.237ms；Spatial目标为匹配5.202/5.470、外来3.062/3.330ms。外来纯draft虽然该成本低，却0/40成功，不能据成本单列认定效率更好。

成本由**所有原生infer调用累计时间／实际执行动作数**得到，包括拒绝后执行0动作的请求；RPC另列。完整all4集合中，Object匹配FLASH低于Full，而Spatial匹配FLASH高于Full；不能宣布FLASH普遍提速。预先固定的repeat-task init22–24中，匹配/外来/Full native infer成本分别为Object 2.280/4.171/3.397、Spatial 2.560/4.275/3.441ms/action；该子集没有可靠cold/capture标记，不能称完全warm或部署稳态。准备成本未混入此每动作指标，回合墙钟、不同成功数与实际轨迹也必须同时保留。

**两个容易混淆的口径：**统计报告的接受率差73.871pp（Object）、59.855pp（Spatial）是case等权均值；上表71.894%、56.417%是按verify动作预算加权，二者都重算一致。成本统计中的配对case均值差也不等于两个汇总成本相减。报告的policy计时来自`server.py`中完整`original_infer`调用（总403.851s），不含外层noise/hash/logging；原native episode的`policy_time_sum_ms`含额外外层工作（414.627s），是另一计时范围，不可静默替换。RPC覆盖客户端往返。

本次独立CPU审计读取10份请求日志、400个小result，共6,171请求；成功率、接受、各Full路由、zero-action、native infer/RPC累计和执行动作数均与报告一致，30个固定时延子集也全部重算一致。80组×5条件的首input/noise身份一致；纯draft不实际使用所生成噪声。本次没有重复动作数组/控制器hash与数值动作一致性的全面验收。

确认性结论应引用[预注册统计结果](../draft_statistics/statistics.json)；通用`paired_vs_matching_source.json`仍带“independent init replication required”旧模板提示及双侧未校正p，不替代这次单侧＋Holm主检验。secondary区间仅描述性，不能把请求当独立样本。所有旧报告、冻结文件、队列与runtime均未修改。

[审计JSON](audit.json) · [输入hash](input_hashes.json) · [CPU复验脚本](audit.py) · [原始汇总](../draft_confirmation_analysis/summary.json) · [固定时延子集](../draft_confirmation_analysis/latency_subsets.json)

```bash
python3 -B /root/nfs/embodied/benchmark/reports/ease_insights_20261103/draft_confirmed_audit/audit.py
```
