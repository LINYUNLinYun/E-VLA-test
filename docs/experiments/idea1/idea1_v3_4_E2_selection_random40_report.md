# 周期3四种选层：40场景补偿完整对照

随机补偿160条新回合，其他结果复用已完成有效记录。相同case、Full-D10→Fast-D2→Fast-D2、原始一次Full10 Jacobian、随机根seed341101。

|方法|V14无补偿|V14外推|V12无补偿|V12外推|
|---|---|---|---|---|
|cosine|28/40|16/40|18/40|9/40|
|jacobian|28/40|34/40|10/40|30/40|
|random_online|6/40|36/40|1/40|34/40|
|random_fixed|23/40|34/40|12/40|29/40|

相关共同前缀异常记录0项，严格配对按冻结计划排除，完整40条描述保留；细节见random40_prefix_audit.json。一个预设随机根seed，不代表随机种子总体。

## 补偿后方法间配对

同预算、同40case，原始Jacobian一次Full10 VJP。新增逐次随机并未改变Full刷新或补偿历史：相对Jacobian，V14恢复3例/新增失败1例，净+5pp，探索区间[-5,+15]；V12恢复7例/新增失败3例，净+10pp，区间[-2.5,+20.0625]。两项区间均跨零，不宣称随机普遍更优。

|基准→候选|n|恢复/新增失败|差值pp|探索95%区间pp|
|---|---|---|---|---|
|v14_cosine_extrapolate→v14_jacobian_extrapolate|40|19/1|+45.00|[+27.50,+62.50]|
|v14_cosine_extrapolate→v14_random_online_extrapolate|40|20/0|+50.00|[+35.00,+67.50]|
|v14_cosine_extrapolate→v14_random_fixed_extrapolate|40|19/1|+45.00|[+32.50,+57.50]|
|v14_jacobian_extrapolate→v14_random_online_extrapolate|40|3/1|+5.00|[-5.00,+15.00]|
|v14_jacobian_extrapolate→v14_random_fixed_extrapolate|40|4/4|-0.00|[-12.50,+10.06]|
|v14_random_online_extrapolate→v14_random_fixed_extrapolate|40|3/5|-5.00|[-15.00,+5.00]|
|v12_cosine_extrapolate→v12_jacobian_extrapolate|40|22/1|+52.50|[+35.00,+70.00]|
|v12_cosine_extrapolate→v12_random_online_extrapolate|40|26/1|+62.50|[+45.00,+77.50]|
|v12_cosine_extrapolate→v12_random_fixed_extrapolate|40|21/1|+50.00|[+35.00,+65.00]|
|v12_jacobian_extrapolate→v12_random_online_extrapolate|40|7/3|+10.00|[-2.50,+20.06]|
|v12_jacobian_extrapolate→v12_random_fixed_extrapolate|40|7/8|-2.50|[-20.00,+15.00]|
|v12_random_online_extrapolate→v12_random_fixed_extrapolate|40|1/6|-12.50|[-25.00,+0.00]|
