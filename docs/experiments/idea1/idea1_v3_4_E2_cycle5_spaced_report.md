# 周期5：完整选层结果，全部只允许间隔跳层

Full-D10→Fast-D2×4，V14跳4层、V12跳6层，原始一次Full10正权Jacobian。外推首周期hold，其后ratio0.2/0.4/0.6/0.8。全部2320条为新评测。

|方法|V14无补偿|V14外推|V12无补偿|V12外推|
|---|---|---|---|---|
|cosine|153/200（76.5%）|91/200（45.5%）|61/200（30.5%）|62/200（31.0%）|
|jacobian|116/200（58.0%）|137/200（68.5%）|53/200（26.5%）|96/200（48.0%）|
|global_fixed|—|—|2/200（1.0%）|70/200（35.0%）|

全局固定V12为新预声明合法组合[1, 3, 7, 10, 12, 15]；旧组合含相邻跳层，未沿用。

相同40条子集：

|方法|V14无补偿|V14外推|V12无补偿|V12外推|
|---|---|---|---|---|
|cosine|29/40（72.5%）|18/40（45.0%）|9/40（22.5%）|15/40（37.5%）|
|jacobian|22/40（55.0%）|26/40（65.0%）|10/40（25.0%）|20/40（50.0%）|
|random_online|2/40（5.0%）|21/40（52.5%）|0/40（0.0%）|18/40（45.0%）|
|random_fixed|22/40（55.0%）|20/40（50.0%）|10/40（25.0%）|24/40（60.0%）|

|条件|稳态均摊ms/请求|Full ms|Fast ms|覆盖回合|完整周期数|平均冷捕获秒|
|---|---|---|---|---|---|---|
|v14_cosine_none|66.42|117.80|53.58|200/200|1442|0.00|
|v14_cosine_extrapolate|66.84|119.36|53.71|200/200|1952|0.00|
|v14_jacobian_none|94.79|263.41|52.63|200/200|1659|3.88|
|v14_jacobian_extrapolate|95.36|265.65|52.79|200/200|1623|3.84|
|v12_cosine_none|64.69|118.05|51.35|200/200|2048|0.00|
|v12_cosine_extrapolate|65.03|119.07|51.52|200/200|2110|0.00|
|v12_jacobian_none|93.64|265.30|50.72|200/200|2077|3.87|
|v12_jacobian_extrapolate|93.89|266.12|50.83|200/200|1904|3.84|
|v12_global_fixed_none|64.01|114.32|51.43|200/200|2334|0.00|
|v12_global_fixed_extrapolate|64.50|115.75|51.69|200/200|2080|0.00|
|v14_random_online_none|65.43|113.36|53.45|40/40|458|0.00|
|v14_random_online_extrapolate|65.38|114.37|53.13|40/40|354|0.00|
|v14_random_fixed_none|65.07|113.33|53.01|40/40|348|0.00|
|v14_random_fixed_extrapolate|65.82|115.11|53.50|40/40|354|0.00|
|v12_random_online_none|63.64|113.69|51.13|40/40|470|0.00|
|v12_random_online_extrapolate|64.01|115.07|51.24|40/40|389|0.00|
|v12_random_fixed_none|63.73|114.16|51.12|40/40|413|0.00|
|v12_random_fixed_extrapolate|64.15|115.80|51.24|40/40|371|0.00|

稳态仅统计第二Full起完整五请求块，冷捕获、首周期、RPC队列单列，不把Full峰值当作均摊时延。

|基准→候选|比较组|严格n|恢复/新增失败|task等权变化pp|探索95%区间pp|
|---|---|---|---|---|---|
|v14_cosine_none→v14_cosine_extrapolate|compensation|200|19/81|-31.00|[-41.00,-20.50]|
|v14_jacobian_none→v14_jacobian_extrapolate|compensation|200|52/31|+10.50|[0.50,20.00]|
|v14_random_online_none→v14_random_online_extrapolate|compensation|40|20/1|+47.50|[30.00,65.00]|
|v14_random_fixed_none→v14_random_fixed_extrapolate|compensation|40|7/9|-5.00|[-25.00,12.50]|
|v14_cosine_none→v14_jacobian_none|main200_methods|200|14/51|-18.50|[-26.00,-11.00]|
|v14_cosine_none→v14_jacobian_none|matched40_methods|40|3/10|-17.50|[-30.00,-5.00]|
|v14_cosine_none→v14_random_online_none|matched40_methods|40|0/27|-67.50|[-80.00,-55.00]|
|v14_cosine_none→v14_random_fixed_none|matched40_methods|40|2/9|-17.50|[-30.00,-5.00]|
|v14_jacobian_none→v14_random_online_none|matched40_methods|40|0/20|-50.00|[-62.50,-40.00]|
|v14_jacobian_none→v14_random_fixed_none|matched40_methods|40|8/8|+0.00|[-12.50,10.00]|
|v14_random_online_none→v14_random_fixed_none|matched40_methods|40|21/1|+50.00|[35.00,65.00]|
|cycle3_v14_jacobian_spaced_none→v14_jacobian_none|cycle3_to_cycle5|200|28/32|-2.00|[-8.00,4.00]|
|v14_cosine_extrapolate→v14_jacobian_extrapolate|main200_methods|200|64/18|+23.00|[14.50,31.00]|
|v14_cosine_extrapolate→v14_jacobian_extrapolate|matched40_methods|40|12/4|+20.00|[5.00,37.50]|
|v14_cosine_extrapolate→v14_random_online_extrapolate|matched40_methods|40|8/5|+7.50|[-7.50,25.00]|
|v14_cosine_extrapolate→v14_random_fixed_extrapolate|matched40_methods|40|8/6|+5.00|[-10.00,20.06]|
|v14_jacobian_extrapolate→v14_random_online_extrapolate|matched40_methods|40|5/10|-12.50|[-30.00,5.00]|
|v14_jacobian_extrapolate→v14_random_fixed_extrapolate|matched40_methods|40|3/9|-15.00|[-30.00,0.00]|
|v14_random_online_extrapolate→v14_random_fixed_extrapolate|matched40_methods|40|6/7|-2.50|[-20.00,12.50]|
|cycle3_v14_jacobian_spaced_extrapolate→v14_jacobian_extrapolate|cycle3_to_cycle5|200|11/52|-20.50|[-27.50,-13.50]|
|v12_cosine_none→v12_cosine_extrapolate|compensation|200|43/42|+0.50|[-12.00,12.50]|
|v12_jacobian_none→v12_jacobian_extrapolate|compensation|200|67/24|+21.50|[10.50,31.50]|
|v12_random_online_none→v12_random_online_extrapolate|compensation|40|18/0|+45.00|[30.00,62.50]|
|v12_random_fixed_none→v12_random_fixed_extrapolate|compensation|40|16/2|+35.00|[17.50,52.50]|
|v12_cosine_none→v12_jacobian_none|main200_methods|200|30/38|-4.00|[-12.50,4.50]|
|v12_cosine_none→v12_jacobian_none|matched40_methods|40|7/6|+2.50|[-10.00,15.00]|
|v12_cosine_none→v12_random_online_none|matched40_methods|40|0/9|-22.50|[-35.00,-12.50]|
|v12_cosine_none→v12_random_fixed_none|matched40_methods|40|8/7|+2.50|[-17.50,22.50]|
|v12_jacobian_none→v12_random_online_none|matched40_methods|40|0/10|-25.00|[-37.50,-15.00]|
|v12_jacobian_none→v12_random_fixed_none|matched40_methods|40|7/7|+0.00|[-17.50,15.00]|
|v12_random_online_none→v12_random_fixed_none|matched40_methods|40|10/0|+25.00|[12.50,37.56]|
|cycle3_v12_jacobian_spaced_none→v12_jacobian_none|cycle3_to_cycle5|200|21/37|-8.00|[-14.50,-1.00]|
|v12_cosine_extrapolate→v12_jacobian_extrapolate|main200_methods|200|53/19|+17.00|[9.50,24.50]|
|v12_cosine_extrapolate→v12_jacobian_extrapolate|matched40_methods|40|9/4|+12.50|[-7.50,32.50]|
|v12_cosine_extrapolate→v12_random_online_extrapolate|matched40_methods|40|5/2|+7.50|[-2.50,17.50]|
|v12_cosine_extrapolate→v12_random_fixed_extrapolate|matched40_methods|40|11/2|+22.50|[7.50,37.50]|
|v12_jacobian_extrapolate→v12_random_online_extrapolate|matched40_methods|40|5/7|-5.00|[-22.50,15.00]|
|v12_jacobian_extrapolate→v12_random_fixed_extrapolate|matched40_methods|40|7/3|+10.00|[-5.00,25.00]|
|v12_random_online_extrapolate→v12_random_fixed_extrapolate|matched40_methods|40|9/3|+15.00|[2.50,30.00]|
|cycle3_v12_jacobian_spaced_extrapolate→v12_jacobian_extrapolate|cycle3_to_cycle5|200|8/91|-41.50|[-50.00,-33.50]|
|v12_global_fixed_none→v12_global_fixed_extrapolate|compensation|199|68/0|+34.00|[26.00,42.00]|
|v12_global_fixed_none→v12_jacobian_none|main200_methods|199|51/0|+26.00|[20.00,32.50]|
|v12_global_fixed_extrapolate→v12_jacobian_extrapolate|main200_methods|200|51/25|+13.00|[4.50,21.50]|

实际评测窗口150.04分钟，927.7回合/小时，71.35动作/秒，14.31请求/秒；模型准备秒数[142.09174878895283, 151.08169086277485, 138.08045213669538]。技术重试0回合。

同一随机根seed，非随机种子总体均值；40条为200条子集。suite分层task聚类bootstrap2000次、seed281034，未校正多重比较；完整样本描述与严格配对范围不同，前缀异常在prefix_audit.json中列出，有效回合全部保留。

仅Jacobian spaced与旧周期3构成相同选层规则的周期比较；余弦和随机的旧周期3使用max2，旧全局固定mask也不同。PyTorch Full10为原生BF16近似回放、一次正权VJP的局限延续。

[冻结方案](idea1_v3_4_E2_cycle5_spaced_plan.md)
