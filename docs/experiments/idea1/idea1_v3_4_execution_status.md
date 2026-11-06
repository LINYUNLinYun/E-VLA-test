# idea1 v3.4 执行状态

**已完成并通过最终验收。** E1共920条闭环、E3父系统40条闭环，合计960条；160对真实相邻Full观测机制消融、207个原shadow样本及207个配对噪声补充样本均完成。E2在原文中为空，未执行。

[总报告](idea1_v3_4_report.md) · [E1报告](idea1_v3_4_E1_report.md) · [E3报告](idea1_v3_4_E3_report.md)

## 最终结果

- 周期V14为182/200，L16旧Full KV路径164/200；Spatial为50/50→44/50。成功率点估计下降9个百分点，task聚类探索区间[-16.5,0.0]个百分点，上界触零。
- 同输入请求加速1.081×，VLM模块1.296×；计入周期Full刷新后，每动作策略耗时减少2.27%。回合更长，完整评测吞吐未提高。
- 全160对没有复现L16–18平均误差改善；旧task0/task5集合仍改善，扩展任务集合方向相反。细化网格L17–18离线均值较低，未测试其闭环成功率。
- 被拒draft保留相关信息。配对噪声控制后，两步修复的旋转命令更接近Full参考，平移与夹爪指标较差。修复为shadow，未宣称修复策略闭环质量。

## 验收与可复现记录

41项最终检查通过，技术invalid目录0个。包括所有case键、官方初态/seed/同服务身份、实际执行动作、H/R/D日程、终止上限、缓存年龄、原始数组/动作指标重算，以及207个噪声控制对原ε1路径逐样本复现；相关重算/复现最大差均为0。三张卡的启动真实输入门、跨服务输出与截断对照通过；科学源码和原idea1文档保持冻结SHA。

- [冻结方案](idea1_v3_4_experiment_plan.md)、[噪声补充方案](idea1_v3_4_E3_noise_control_plan.md)
- [原始manifest](../../../results/idea1_v3_4/manifest.json)、[完整验收](../../../results/idea1_v3_4/final_audit.json)
- [完成审查](../../../results/idea1_v3_4/completion_review.json)、[资源退出核验](../../../results/idea1_v3_4/resource_release.json)
- [机器可读分析](../../../results/idea1_v3_4/analysis/analysis_state.json)、[噪声补充验收](../../../results/idea1_v3_4/E3_noise_control/audit.json)

三个最终科学PNG已目视检查，报告本地链接可访问。主队列用时81.86分钟，加载/编译/真实输入门与逐条件墙钟、动作/秒、请求/秒分开记录。原控制、worker、模型服务、噪声补充和收尾进程均已实际退出，GPU0/2/3各为1MiB、0%利用率；GPU1未使用。

执行期间历史状态保存在[历史记录](../../../results/idea1_v3_4/execution_status_history.md)。原idea1.md未改写，所有结果以最终报告和冻结数据为准。
