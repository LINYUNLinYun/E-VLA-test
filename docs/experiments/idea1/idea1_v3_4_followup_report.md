# v3.4 E1 两项独立补充实验

原P=2、Full→draft→draft不变。A仅复用L17–18两轮；B保留L16–18但只首轮复用；C为两者组合（条件触发）。

初筛有效性：{'A17_all': True, 'B16_first': True}；组合触发：True；扩样条件：['B16_first', 'C17_first']。

|条件|口径|Spatial|Object|Goal|Long|总成功|策略ms/动作|
|---|---|---|---|---|---|---|---|
|A17_all|first40|9/10|10/10|9/10|6/10|34/40|23.386|
|A17_all|all|9/10|10/10|9/10|6/10|34/40|23.386|
|B16_first|first40|9/10|10/10|10/10|7/10|36/40|24.256|
|B16_first|all|47/50|49/50|46/50|37/50|179/200|18.853|
|C17_first|first40|10/10|10/10|10/10|5/10|35/40|23.468|
|C17_first|all|50/50|50/50|44/50|36/50|180/200|19.582|

原同case对照：periodic_v14首40为37/40、全200为182/200；periodic_old16首40为31/40、全200为164/200。以上历史质量通过当前输入/Full原始动作检查后复用；跨运行绝对wall不直接作为加速证据。

## 同输入周期成本

|条件|周期平均请求ms|相对无复用周期V14加速|
|---|---|---|
|baseline|68.747|1.0000×|
|A17_all|67.251|1.0223×|
|B16_first|67.639|1.0164×|
|C17_first|67.999|1.0110×|

周期估算计入一次Full及两次draft；单次路径实测见decision.json与workers/gpu*/timing.json，准备加载与仿真不属于该微基准。自由闭环每动作策略成本和episode/小时另列quality_summary.csv。

## 配对结果

|候选|对照|口径|n|恢复|损失|成功率差pp|task聚类探索区间pp|
|---|---|---|---|---|---|---|---|
|A17_all|periodic_v14|first40|40|2|5|-7.50|[-20.00,+5.00]|
|A17_all|periodic_v14|all|40|2|5|-7.50|[-20.00,+5.00]|
|A17_all|periodic_old16|first40|40|4|1|+7.50|[-2.50,+17.50]|
|A17_all|periodic_old16|all|40|4|1|+7.50|[-2.50,+17.50]|
|B16_first|periodic_v14|first40|40|2|3|-2.50|[-12.50,+7.50]|
|B16_first|periodic_v14|all|200|8|11|-1.50|[-7.50,+5.00]|
|B16_first|periodic_old16|first40|40|5|0|+12.50|[+2.50,+22.50]|
|B16_first|periodic_old16|all|200|21|6|+7.50|[+2.50,+13.00]|
|C17_first|periodic_v14|first40|40|1|3|-5.00|[-12.50,+2.50]|
|C17_first|periodic_v14|all|200|5|7|-1.00|[-4.50,+2.50]|
|C17_first|periodic_old16|first40|40|5|1|+10.00|[+0.00,+22.50]|
|C17_first|periodic_old16|all|200|25|9|+8.00|[+1.50,+14.01]|

这是探索评测，非官方50-init全量基准；初筛/扩样与组合条件均在运行前冻结，不以接受率或单输入误差代替闭环质量。原始数据保存在results/idea1_v3_4_followup，逐案例、逐请求、真实EEF姿态、动作、路由、seed和年龄均保留。

[冻结方案](idea1_v3_4_followup_plan.md) · [最终验收](../../../results/idea1_v3_4_followup/audit.json) · [触发决策与测速](../../../results/idea1_v3_4_followup/decision.json)
