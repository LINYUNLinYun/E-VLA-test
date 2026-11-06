本轮VLM独立复验600/600条通过完整审计，正式主分析和补充分析均已完成。
原Spatial/Long结论：both_prespecified_directions_not_reproduced。
预先登记的补充Goal/Long结论：both_prespecified_directions_not_reproduced。

补充报告中的原主分析逐项与原正式报告一致，保留原Holm2；以下三方向另统一给出Holm3。

| 冻结方向 | A/B成功数 | 差值 | 单侧精确p | Holm3 | Case 95%CI | Task-cluster 95%CI |
|---|---:|---:|---:|---:|---|---|
| libero_spatial: Jac>MSE | 47/46（各50） | +2.0pp | 0.5 | 1 | [-8.0, +12.0]pp | [-8.0, +14.0]pp |
| libero_10: MSE>Jac | 34/34（各50） | +0.0pp | 0.623047 | 1 | [-12.0, +12.0]pp | [-12.0, +10.0]pp |
| libero_goal: Jac>MSE | 39/43（各50） | -8.0pp | 0.945312 | 1 | [-20.0, +4.0]pp | [-20.0, +2.0]pp |

| 条件 | 全部成功率 |
|---|---:|
| Full | 182/200 |
| Jac | 170/200 |
| MSE | 170/200 |

补充注册审计：prospective_supplementary。原identity没有内嵌Unix时间，先后关系使用包括技术重试在内的原始文件mtime；这一限制保留在审计中。
这些是固定路径在init31–35上的结果。不能将非显著解释为等价，也不能据此声称运行时选路、阶段切换或独占GPU加速。外部资源争用不会改写质量成功/失败；成本证据及其限制保存在正式报告中。
后续若使用这些数据重新选择任务或语义类别，只能作为新的discovery，不改写本轮阴性或不确定结论。
完整逐task/init、leave-one-out、Full参照及证据SHA见audit.json和原正式主/补报告。
