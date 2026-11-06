# Jacobian连续性规则：200例外推补偿对照

两新规则×V14/V12各200，共800新回合。每Full采集自身全部17层真实残差，保留最近两个Full；首周期hold，随后Fast ratio 1/3、2/3。Full-D10→Fast-D2→Fast-D2、原始一次正权VJP、每Full换mask，V14跳4、V12跳6、AE18；仅增加同一外推补偿。

|方法/规则|V14无补偿|V14外推|V12无补偿|V12外推|
|---|---|---|---|---|
|余弦，最多连续2|149/200（74.5%）|72/200（36.0%）|104/200（52.0%）|62/200（31.0%）|
|Jacobian，最多连续2|131/200（65.5%）|176/200（88.0%）|74/200（37.0%）|150/200（75.0%）|
|Jacobian，无连续上限|127/200（63.5%）|181/200（90.5%）|79/200（39.5%）|175/200（87.5%）|
|Jacobian，只许间隔跳|120/200（60.0%）|178/200（89.0%）|69/200（34.5%）|179/200（89.5%）|

全局固定V12外推同case既有参考：169/200（84.5%），固定组合[2,3,5,14,16,17]；不等同于每回合固定随机。


补偿前后成功率明显提升：V14无上限63.5%→90.5%、间隔60%→89%；V12无上限39.5%→87.5%、间隔34.5%→89.5%。同规则的补偿前后严格配对为V14两项200例、V12无上限199例/间隔200例，探索区间均为正。

V12相对旧max2外推75%的改善较明确：无上限的严格task等权提升12.39pp（n194，探索95%区间7.22至17.56pp），间隔跳提升14.83pp（n195，区间8.72至21.83pp）；保留完整200例的敏感性方向也一致。V14相对旧max2外推88%的差异仍不确定，两项区间均跨零。

V12间隔89.5%为本轮最高点估计，无上限为87.5%；两者配对区间仍跨零。相对全局固定V12外推84.5%，间隔的完整200点差为+5pp，但严格task等权差+4.5pp的区间为-1.28至10.61pp，无上限相对固定的区间也跨零。因此尚不能确认新规则优于该固定组合。

|基准→候选|严格配对n|恢复/新增失败|task等权差值pp|探索95%区间pp|
|---|---|---|---|---|---|
|v14_jacobian_extrapolate→v14_jacobian_unrestricted_extrapolate|195|11/6|+2.44|[-2.56,+7.50]|
|v14_jacobian_extrapolate→v14_jacobian_spaced_extrapolate|195|16/14|+0.89|[-4.50,+6.06]|
|v14_jacobian_unrestricted_extrapolate→v14_jacobian_spaced_extrapolate|200|12/15|-1.50|[-6.50,+3.50]|
|v14_cosine_extrapolate→v14_jacobian_unrestricted_extrapolate|195|112/3|+55.39|[+46.22,+63.89]|
|v14_cosine_extrapolate→v14_jacobian_spaced_extrapolate|195|108/2|+53.83|[+45.00,+62.72]|
|v14_jacobian_unrestricted_none→v14_jacobian_unrestricted_extrapolate|200|61/7|+27.00|[+19.00,+35.00]|
|v14_jacobian_spaced_none→v14_jacobian_spaced_extrapolate|200|68/10|+29.00|[+21.50,+36.00]|
|v12_jacobian_extrapolate→v12_jacobian_unrestricted_extrapolate|194|30/6|+12.39|[+7.22,+17.56]|
|v12_jacobian_extrapolate→v12_jacobian_spaced_extrapolate|195|36/7|+14.83|[+8.72,+21.83]|
|v12_jacobian_unrestricted_extrapolate→v12_jacobian_spaced_extrapolate|199|15/11|+2.00|[-3.00,+7.00]|
|v12_cosine_extrapolate→v12_jacobian_unrestricted_extrapolate|194|114/2|+57.28|[+47.00,+66.28]|
|v12_cosine_extrapolate→v12_jacobian_spaced_extrapolate|195|120/3|+59.72|[+50.06,+68.00]|
|v12_jacobian_unrestricted_none→v12_jacobian_unrestricted_extrapolate|199|102/6|+48.13|[+40.00,+57.00]|
|v12_jacobian_spaced_none→v12_jacobian_spaced_extrapolate|200|115/5|+55.00|[+46.50,+63.50]|
|global_fixed_v12_extrapolate→v12_jacobian_unrestricted_extrapolate|194|17/13|+2.06|[-4.06,+7.22]|
|global_fixed_v12_extrapolate→v12_jacobian_spaced_extrapolate|195|21/12|+4.50|[-1.28,+10.61]|

|条件|实际最大连续跳层|选层中三连及以上比例|稳态周期均摊ms/请求|Full ms|Fast ms|平均冷捕获秒|
|---|---|---|---|---|---|---|
|v14_jacobian_unrestricted_extrapolate|4|8.76%|121.99|262.45|51.82|3.77|
|v14_jacobian_spaced_extrapolate|1|0.00%|121.94|262.05|51.88|3.76|
|v12_jacobian_unrestricted_extrapolate|6|38.80%|120.44|262.44|49.50|3.77|
|v12_jacobian_spaced_extrapolate|1|0.00%|120.42|261.92|49.73|3.78|

实际闭环窗口58.14分钟，825.6回合/小时、44.34执行动作/秒、8.95模型请求/秒；含各回合冷启动、环境、排队及落盘，模型准备单列throughput.json。技术重试0回合。
相对max2的基线前缀异常记录21项，相关严格配对异常累计55项；不同条件或比较可重复涉及同case。全部200描述结果保留，严格配对剔除对应异常、完整200敏感性另表。

4 suites×task0..9×init19..23。探索bootstrap采用suite分层task聚类2000次、seed281034，未校正多重比较；已用于探索的case不视作未见验证。存在整task排除时，严格task等权区间不直接对应完整200的逐episode点估计。

规则优化器、单次Full10梯度、真实残差双历史与各层捕获均核验；每个候选使用自己的观测、KV与历史。首个完整周期及冷捕获计入实际wall；稳态均摊从第二Full起按完整三请求块计，policy耗时不含双环境RPC队列。自由轨迹成本差不作同输入因果加速证明。PyTorch Full10近似原生BF16和单加权VJP局限不变。

[冻结方案](idea1_v3_4_E2_adjacency_extrapolate_200_plan.md)

## 前缀异常的独立范围

21条基线异常记录对应6个独立case：Spatial task0的init19–23在四个新条件中重复计为20条，另有Goal task1/init22的V12无上限补偿一条。前者与上轮相同，首观测及梯度匹配、原生首Full动作有差异，根因尚未确认；并未放宽容差或重跑有效回合。

新增Goal例仅首Fast观测hash不同。与旧max2、旧无补偿及新间隔规则相比，其首Full raw与actions逐元素完全一致，前5步动作及前6条EEF位置/姿态也完全一致；具体观测差异原因未确认，记录见extra_prefix_exception_detail.json。该例保留完整描述结果，从相关严格配对排除。

因此，新规则互比为V14 n200、V12 n199；跨max2/余弦/固定的严格配对为n194或195；同规则补偿前后为n199或200。主表完整200成功率与严格task等权区间的样本/估计口径不同，不能混用。所有有效结果均保留，全部200敏感性见paired_sensitivity.csv。

本轮三卡服务已正常结束，800条全部完成，技术重试0。稳态周期均摊约120–122ms/请求，评分所在Full约262ms；补偿与新规则未消除一次Full-D10反传的主要成本。上述时延不含RPC排队与冷捕获，成本表和实际wall已分别列出。
