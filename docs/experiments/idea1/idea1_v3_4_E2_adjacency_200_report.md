# 无补偿Jacobian：连续跳层规则200例对照

每个预算/新规则200条，共800条新回合。Full-D10→Fast-D2→Fast-D2，原始一次正权VJP，每Full更新mask；V14跳4、V12跳6、候选L1–17、AE全18层；其余既有协议保持。unrestricted允许但不强制三连及以上；spaced完全禁止相邻跳过。max2和余弦是同200case旧结果复用。

|方法/规则|V14成功数|V12成功数|
|---|---|---|
|余弦，最多连续2|149/200（74.5%）|104/200（52.0%）|
|Jacobian，最多连续2|131/200（65.5%）|74/200（37.0%）|
|Jacobian，无连续上限|127/200（63.5%）|79/200（39.5%）|
|Jacobian，只许间隔跳|120/200（60.0%）|69/200（34.5%）|


实测中，取消连续限制相对原max2的成功率点估计为V14 -2pp、V12 +2.5pp；完全禁止相邻为V14 -5.5pp、V12 -2.5pp。四项跨旧基线的严格配对探索区间均跨零，没有显示稳定提升；两种新规则在两档预算下的成功率均低于余弦。由此不能把当前无补偿Jacobian落后主要归结为原max2限制，也不能用这轮小幅点估计变化证明哪一种连续性规则普遍更优。

取消上限实际产生三连及以上：V14占Full选层8.65%、最长4连；V12占38.65%、最长6连。间隔规则全部选层最大连续长度1。

|基准→候选|严格配对n|恢复/新增失败|差值pp|探索95%区间pp|
|---|---|---|---|---|
|v14_jacobian_none→v14_jacobian_unrestricted_none|195|7/11|-2.06|[-5.67,+1.67]|
|v14_jacobian_none→v14_jacobian_spaced_none|195|20/31|-5.78|[-12.89,+1.72]|
|v14_jacobian_unrestricted_none→v14_jacobian_spaced_none|200|20/27|-3.50|[-10.50,+3.50]|
|v14_cosine_none→v14_jacobian_unrestricted_none|195|19/40|-10.56|[-19.67,-1.00]|
|v14_cosine_none→v14_jacobian_spaced_none|195|24/52|-14.28|[-23.00,-4.72]|
|v12_jacobian_none→v12_jacobian_unrestricted_none|195|26/22|+2.00|[-4.33,+8.34]|
|v12_jacobian_none→v12_jacobian_spaced_none|195|30/34|-2.61|[-11.89,+6.83]|
|v12_jacobian_unrestricted_none→v12_jacobian_spaced_none|200|26/36|-5.00|[-11.50,+2.00]|
|v12_cosine_none→v12_jacobian_unrestricted_none|195|22/48|-13.22|[-21.94,-3.78]|
|v12_cosine_none→v12_jacobian_spaced_none|195|19/53|-17.83|[-26.61,-9.00]|

|条件|实际最大连续跳层|选层中三连及以上比例|稳态周期均摊ms/请求|Full ms|Fast ms|平均冷捕获秒|
|---|---|---|---|---|---|---|
|v14_jacobian_unrestricted_none|4|8.65%|121.66|261.12|52.01|3.79|
|v14_jacobian_spaced_none|1|0.00%|121.68|261.05|52.00|3.81|
|v12_jacobian_unrestricted_none|6|38.65%|120.41|261.72|49.77|3.79|
|v12_jacobian_spaced_none|1|0.00%|120.40|261.48|49.91|3.80|

从最初模型服务可用到恢复后全部完成的端到端窗口101.30分钟，473.9回合/小时、34.09执行动作/秒、6.84模型请求/秒；包含中断等待与恢复启动，不作为连续活跃评测吞吐。原始服务准备及恢复窗口单列throughput.json。技术重试1回合。
相对max2共同前缀异常20项、相关配对累计异常40项；所有有效结果保留，严格配对剔除对应异常，完整200敏感性见paired_sensitivity.csv。

200case为4 suites×task0..9×init19..23，完整suite细目见summary.csv。探索bootstrap为suite分层task聚类2000次，未校正多重比较；本批case曾用于此前探索，不声称未见验证。每个mask属于各自自由轨迹，不能从三连频率与成功率关系作额外因果归因。

稳态周期从第二Full起按完整三请求块平均；policy计时不含RPC排队，Full/Fast表排除各回合第一个请求，冷捕获单列。不同自由轨迹时延不作为同输入加速证明。评分所在Full的额外反传成本保留，单VJP及PyTorch D10近似原生BF16的局限不变。

[冻结方案](idea1_v3_4_E2_adjacency_200_plan.md)

运行在798/800处发生进程中断；退出原因没有捕获，不能归因为模型OOM或任务失败。原进程与worker已确认退出，旧队列、进度、日志、数值门和状态保存在recovery_01/。原798条有效结果未重跑；仅补两条缺失记录，其中一条已有未完成启动，使用第二次且最后一次技术尝试。恢复仅使用当前空闲卡，未改冻结实验源码、条件或种子。

## 共同前缀异常的独立case范围

20条基线异常记录对应5个独立case：Spatial task0的init19–23，每个新条件均涉及这5例。新规则互相比较的共同前缀全部匹配，严格配对保留200；跨既有max2/余弦基线的严格配对为195。因为Spatial task0整组不进入严格比较，其区间针对剩余task、按suite/task等权，不能把该区间直接套在完整200例的逐episode点估计上。完整200例描述及敏感性结果仍全部公开。

这5例与旧基线的首观测hash和生成噪声seed相同，首Full的17层梯度完全相同；原生Full动作超出预设1e-5容差，首5步平移控制增量差的均值为0.116–0.278毫米，首5步夹爪符号翻转为0。Spatial task0也是数值预热所用prompt；当前仅确认这种集中现象，根因未验证。没有通过放宽容差或重跑有效回合消除异常。量级记录见results/idea1_v3_4_E2_adjacency_200/recovery_01/prefix_magnitude.json。
