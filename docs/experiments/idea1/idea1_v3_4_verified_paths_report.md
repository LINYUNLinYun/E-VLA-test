# V12/V14外推D2＋验证：周期3/5对Full-D10

选层设置：{'selection': 'jacobian', 'projections': 4, 'adjacency_rule': 'spaced', 'origin': 'Continue the current four-VJP, spaced configuration; assumption stated to user while awaiting optional clarification.'}。验证tau=.10、t=.10/.05、原夹爪保护，无Kalman draft。每项同200例；加速比为同轮新baseline共同163例的完整策略ms/实际执行动作成本比，包含被拒Fast、verify、Full及反推/冷捕获。

|条件|成功率|策略ms/实际动作|相对Full-D10成本加速比|
|---|---|---|---|
|v12_extrapolate_d2_verify_p3|170/200（85.0%）|101.405|0.224×|
|v12_extrapolate_d2_verify_p5|142/200（71.0%）|83.370|0.273×|
|v14_extrapolate_d2_verify_p3|183/200（91.5%）|106.367|0.214×|
|v14_extrapolate_d2_verify_p5|160/200（80.0%）|89.523|0.254×|
|full_d10|187/200（93.5%）|22.739|1.000×|

|条件|请求数|Full调用/请求|零接受/verify|Fast执行动作占比|外推ratio>1占比|
|---|---|---|---|---|
|v12_extrapolate_d2_verify_p3|8052|39.1%|14.1%|60.6%|3.7%|
|v12_extrapolate_d2_verify_p5|9703|28.4%|16.7%|71.2%|10.4%|
|v14_extrapolate_d2_verify_p3|7385|39.6%|14.9%|60.2%|3.6%|
|v14_extrapolate_d2_verify_p5|8801|29.2%|18.4%|70.4%|10.9%|
|full_d10|6756|100.0%|—|0.0%|—|

互斥请求类型占比（每条件合计100%）：

|条件|类别|次数|比例|
|---|---|---|---|
|v12_extrapolate_d2_verify_p3|full_startup|200|2.5%|
|v12_extrapolate_d2_verify_p3|fast_accept_5|4827|59.9%|
|v12_extrapolate_d2_verify_p3|full_periodic|2144|26.6%|
|v12_extrapolate_d2_verify_p3|full_reject_spatial|432|5.4%|
|v12_extrapolate_d2_verify_p3|full_reject_gripper|374|4.6%|
|v12_extrapolate_d2_verify_p3|fast_accept_partial|75|0.9%|
|v12_extrapolate_d2_verify_p5|full_startup|200|2.1%|
|v12_extrapolate_d2_verify_p5|fast_accept_5|6740|69.5%|
|v12_extrapolate_d2_verify_p5|full_periodic|1158|11.9%|
|v12_extrapolate_d2_verify_p5|full_reject_gripper|556|5.7%|
|v12_extrapolate_d2_verify_p5|fast_accept_partial|211|2.2%|
|v12_extrapolate_d2_verify_p5|full_reject_spatial|835|8.6%|
|v12_extrapolate_d2_verify_p5|full_pending_gripper|3|0.0%|
|v14_extrapolate_d2_verify_p3|full_startup|200|2.7%|
|v14_extrapolate_d2_verify_p3|fast_accept_5|4383|59.4%|
|v14_extrapolate_d2_verify_p3|full_periodic|1940|26.3%|
|v14_extrapolate_d2_verify_p3|full_reject_gripper|375|5.1%|
|v14_extrapolate_d2_verify_p3|full_reject_spatial|407|5.5%|
|v14_extrapolate_d2_verify_p3|fast_accept_partial|79|1.1%|
|v14_extrapolate_d2_verify_p3|full_pending_gripper|1|0.0%|
|v14_extrapolate_d2_verify_p5|full_startup|200|2.3%|
|v14_extrapolate_d2_verify_p5|fast_accept_5|5997|68.1%|
|v14_extrapolate_d2_verify_p5|full_periodic|956|10.9%|
|v14_extrapolate_d2_verify_p5|full_reject_spatial|879|10.0%|
|v14_extrapolate_d2_verify_p5|full_reject_gripper|530|6.0%|
|v14_extrapolate_d2_verify_p5|fast_accept_partial|238|2.7%|
|v14_extrapolate_d2_verify_p5|full_pending_gripper|1|0.0%|
|full_d10|full_baseline|6756|100.0%|

同卡同输入Full参考；候选耗时取原自由闭环请求，参考在task组结束后顺/逆两次测量，时间顺序未完全反平衡，作为参考而非严格因果速度证明：

|条件|输入请求数|Full参考ms/请求|候选ms/请求|请求成本比|执行长度归一化成本比|
|---|---|---|---|---|---|
|v12_extrapolate_d2_verify_p3|120|110.76|865.92|0.128×|0.127×|
|v12_extrapolate_d2_verify_p5|120|110.19|808.17|0.136×|0.135×|
|v14_extrapolate_d2_verify_p3|120|110.21|880.54|0.125×|0.125×|
|v14_extrapolate_d2_verify_p5|120|110.64|843.25|0.131×|0.129×|

速度比>1表示候选更快。真实闭环轨迹及请求组成不同；冷捕获、排队、模型准备、同输入参考和实际评测吞吐分别保留。Full/Draft/Verify可在同一拒绝请求并存，调用比例不要求合计100%；互斥请求类别才合计100%。验证零接受与夹爪接管是规则触发因素，不直接等同于任务失败原因。

本轮新测963条、兼容复用37条Full参考；评测窗口119.34分钟，484.2新回合/小时，含独立profiling。技术重试4回合。模型准备单列。

共同前缀配对异常记录0项；全部有效结果保留，严格配对与完整200敏感性分别输出。suite分层task聚类bootstrap2000次、seed281034，探索区间未校正多重比较。

[冻结方案](idea1_v3_4_verified_paths_plan.md)
