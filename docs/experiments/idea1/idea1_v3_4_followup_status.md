# v3.4 E1补充执行状态

**已完成并通过验收。** 两项独立验证和按预定门触发的组合测试均已完成，新闭环共440条，无技术invalid。

[实验结论](idea1_v3_4_followup_summary.md) · [详细报告](idea1_v3_4_followup_report.md) · [冻结方案](idea1_v3_4_followup_plan.md)

|条件|初筛|完整结果|Spatial|
|---|---|---|---|
|A：L17–18两轮复用|34/40|未扩样|9/10|
|B：L16–18仅首轮复用|36/40|179/200|47/50|
|C：L17–18仅首轮复用|35/40|180/200|50/50|

历史同case对照：旧L16–18两轮复用31/40、164/200，Spatial44/50；无复用周期V14为37/40、182/200，Spatial50/50。

P=2保持Full→draft→draft；B/C第二轮是完整当前V14，不插入额外Full。A/B独立初筛均比旧方案多成功至少2条且同输入周期成本低于V14，触发C；B/C满足至少35成功的预定扩样门，补init15–18至200。A保留40条，未因结果回调门槛。

B/C新增160条分别143/160、145/160；组合相对B全200恢复8例、损失7例，净增1，task聚类探索区间[-4.5,+4.5]个百分点，尚无稳定叠加收益的证据。组合在Spatial恢复到50/50，但Goal/Long有相反方向，不能泛化为全任务优势。

同输入模型路径周期估算计入Full刷新：无复用V14为68.747ms/请求、原old16两轮66.530ms。A/B/C分别67.251/67.639/67.999ms，较V14节省约2.18%/1.61%/1.09%；相对原old16两轮，耗时增加1.08%/1.67%/2.21%。不将该微基准冒充含RPC/仿真的完整吞吐，跨运行绝对wall不作为加速证据。

19项执行验收通过，包括冻结source/manifest/plan、实际F-H-H/F-H-V路由、所有动作/EEF轨迹对齐、日程/年龄、case全集和历史数值复用门；cutoff17与完整计算再换KV对照通过。另通过触发/扩样决策复核、组合与独立改动的逐case比较、链接及模型退出检查。历史门的参考输出最大差0。

原v3.4实现、manifest与结果未改写。所有本轮controller、GPU0/3服务及review均已实际退出；本轮GPU资源已释放，GPU1/2未使用。既有执行历史保存在results/idea1_v3_4_followup/execution_status_history.md。

[执行验收](../../../results/idea1_v3_4_followup/audit.json) · [完成审查](../../../results/idea1_v3_4_followup/completion_review.json) · [资源退出记录](../../../results/idea1_v3_4_followup/resource_release.json)
