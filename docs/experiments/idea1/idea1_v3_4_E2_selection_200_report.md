# 周期3余弦/Jacobian：200场景扩样

4 suites×10 task×init19..23，每条件200；原40复用、新160独立列出。8个扩样条件共1600，另有4个随机补偿条件各40，共1760记录；本轮新1440，复用320。Jacobian为原始一次正权Full-D10评分，每Full刷新，不混用后续误差评分/三周期变体。

|方法|预算|无补偿200|外推200|无补偿新160|外推新160|
|---|---|---|---|---|---|
|cosine|V14|149/200|72/200|121/160|56/160|
|jacobian|V14|131/200|176/200|103/160|142/160|
|cosine|V12|104/200|62/200|86/160|53/160|
|jacobian|V12|74/200|150/200|64/160|120/160|

全局固定V12同case既有参考：无补偿134/200，外推169/200。不等同于每回合固定随机。


扩样后，无补偿时余弦成功率点估计高于Jacobian，V14为74.5%对65.5%、V12为52%对37%；V14的配对区间仍跨零。加外推使Jacobian分别提升22.5pp（探索95%区间14至31pp）、38pp（28.5至47.5pp），余弦则下降38.5pp和21pp。上述方向在新增160条中均延续。余弦外推在Long套件仅V14 2/50、V12 0/50；本轮核验确认Full及双历史按计划刷新，具体退化机制尚未做因果消融。

全局固定V12外推169/200高于原始Jacobian外推150/200。由固定切换到Jacobian恢复9例、新增失败28例，固定的配对优势9.5pp，探索95%区间3.5至15.5pp。不能据此将Jacobian概括为普遍优于固定组合或随机选择。

|配对|cohort|n|恢复/新增失败|task等权差值pp|探索95%区间pp|
|---|---|---|---|---|---|
|v14_cosine_none→v14_cosine_extrapolate|all200|200|8/85|-38.50|[-48.50,-27.50]|
|v14_cosine_none→v14_cosine_extrapolate|new160|160|2/67|-37.83|[-47.50,-27.50]|
|v14_jacobian_none→v14_jacobian_extrapolate|all200|200|57/12|+22.50|[+14.00,+31.00]|
|v14_jacobian_none→v14_jacobian_extrapolate|new160|160|48/9|+26.50|[+16.83,+35.67]|
|v14_cosine_none→v14_jacobian_none|all200|200|22/40|-9.00|[-18.50,+0.51]|
|v14_cosine_none→v14_jacobian_none|new160|160|15/33|-9.67|[-19.33,+0.67]|
|v14_cosine_extrapolate→v14_jacobian_extrapolate|all200|200|108/4|+52.00|[+42.00,+61.50]|
|v14_cosine_extrapolate→v14_jacobian_extrapolate|new160|160|89/3|+54.67|[+44.00,+64.84]|
|v12_cosine_none→v12_cosine_extrapolate|all200|200|12/54|-21.00|[-28.50,-13.50]|
|v12_cosine_none→v12_cosine_extrapolate|new160|160|9/42|-21.50|[-30.50,-12.50]|
|v12_jacobian_none→v12_jacobian_extrapolate|all200|200|83/7|+38.00|[+28.50,+47.50]|
|v12_jacobian_none→v12_jacobian_extrapolate|new160|160|62/6|+34.00|[+23.83,+43.83]|
|v12_cosine_none→v12_jacobian_none|all200|200|16/46|-15.00|[-24.00,-6.00]|
|v12_cosine_none→v12_jacobian_none|new160|160|15/37|-13.00|[-22.50,-3.17]|
|v12_cosine_extrapolate→v12_jacobian_extrapolate|all200|200|95/7|+44.00|[+33.50,+54.00]|
|v12_cosine_extrapolate→v12_jacobian_extrapolate|new160|160|73/6|+42.50|[+31.50,+53.17]|
|global_fixed_v12_none→v12_cosine_none|all200|200|8/38|-15.00|[-23.50,-7.49]|
|global_fixed_v12_extrapolate→v12_cosine_extrapolate|all200|200|6/113|-53.50|[-64.51,-42.00]|
|global_fixed_v12_none→v12_jacobian_none|all200|200|8/68|-30.00|[-39.01,-21.00]|
|global_fixed_v12_extrapolate→v12_jacobian_extrapolate|all200|200|9/28|-9.50|[-15.50,-3.50]|

共同前缀异常：相关配对累计0项，完整描述保留、严格配对剔除，all-case敏感性另表；同一个case可在多项比较中重复计数。固定参考异常0项。所有官方初态/控制协议、源hash、Full/Fast路由、单次Full10梯度、随机seed和本路径双历史核验通过。

新160为任务不均初态数（偶task5个、奇task3个），配对点与bootstrap统一suite/task等权，同时保留逐episode差值。总体200每task5个，点估计一致。探索区间未校正多重比较；原40参与多次先前探索，不能当作全新验证。

成本与吞吐见condition_audits、requests.csv与throughput.json。复用40含旧单环境时延、新160可能双环境；双环境RPC含队列、policy内计时不含队列。自由轨迹均值不是同输入加速证明。旧PyTorch D10导数仍为原生BF16近似，不是精确原生Jacobian。

[随机补偿40完整表](idea1_v3_4_E2_selection_random40_report.md)
[冻结方案](idea1_v3_4_E2_selection_200_plan.md)

## 实际执行与成本

本轮新完成1440回合，复用320回合；3张GPU、每卡2个独立环境，共6环境。实际评测窗口90.46分钟，吞吐955.1回合/小时；从控制器启动至最终审计完成96.50分钟。模型/服务与兼容检查准备各卡分别耗时99.06/97.07/99.05秒。评测窗口包含各回合冷启动与环境建立；并发回合wall之和不作为总耗时。
技术重试0回合、worker错误0项。全部已发起有效回合保留，官方初态及控制协议核对通过。

|方法|预算|补偿|稳态完整周期均摊policy ms/请求|平均每回合selector冷捕获秒|
|---|---|---|---|---|
|random_online|V14|extrapolate|75.97|0.00|
|random_fixed|V14|extrapolate|75.32|0.00|
|random_online|V12|extrapolate|74.36|0.00|
|random_fixed|V12|extrapolate|74.09|0.00|
|cosine|V14|none|75.24|0.00|
|cosine|V14|extrapolate|75.55|0.00|
|jacobian|V14|none|123.09|3.88|
|jacobian|V14|extrapolate|123.27|3.84|
|cosine|V12|none|73.80|0.00|
|cosine|V12|extrapolate|74.20|0.00|
|jacobian|V12|none|121.82|3.87|
|jacobian|V12|extrapolate|122.18|3.82|

表中为各自自由轨迹上，排除首个完整周期之后，按完整Full/Fast/Fast周期计时的policy均值；每个扩样条件含40条既有单环境记录和160条本轮记录。policy计时不含RPC排队，两个环境并发时RPC延迟不能当作单机器人延迟。成功轨迹长度不同、输入不同，以上成本是部署量级描述，不构成相同输入的因果加速对照。首周期与冷捕获仍计入实际总吞吐。

## 解释边界

扩样使用原始一次正权Full-D10 Jacobian，每Full评分一次；本轮没有扩样补偿误差方向评分、2/3/4次投影或隔三周期评分方案。原40已参与此前探索，新160单列以便检查新场景是否延续趋势。

四方法的公平直接比较使用相同40个case，见随机补偿报告。逐次随机加外推的成功率点估计为V14 36/40、V12 34/40，相对一次Jacobian的配对差值为+5pp和+10pp，但探索95%区间均跨零。单一预设随机根seed不足以证明随机种子总体更优，随机方法本轮也没有扩到200。

全局固定V12组合[2,3,5,14,16,17]在所有case使用同一mask；每回合固定随机会按case抽不同mask，两者定义不同。200case参考为无补偿134/200、外推169/200，属于同case旧实验复用。

离线汇总首次因旧全局固定记录缺少route.selected_skip字段而中断，原始错误保留在failed.json。review_v1/recover_summary.py仅对400条已固定引用，从result.config.skip读取层编号，并校验静态组合、D10/D2周期与identity.config一致；其余原冻结审计和配对检查均保留。独立适配后汇总通过，没有改动冻结实验源码或重跑评测。详见aggregation_recovery.json。

新增记录中Jacobian外推的非首请求Full平均policy耗时：V14 264.98ms、V12 265.50ms；因此122–123ms是周期均摊，评分Full本身仍超200ms。详见review_v1/new_only_path_timing.json。

## 连续跳层核验

本轮200例从一开始即限制最大连续跳2层。逐条核验8组共1600回合、52687次Fast实际掩码，三连及以上违规0。无补偿Full选层中，含至少一对相邻跳层的比例：Jacobian V14 55.45%、V12 91.16%，余弦两档均100%。允许多组互相分离的二连段；这些自由轨迹频率不能证明两连跳有无害处。详见[连续性核验](idea1_v3_4_E2_selection_adjacency_audit.md)。
