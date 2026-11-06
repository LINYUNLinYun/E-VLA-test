# 常用四suite的独立Full健康校准

40条均完成，Full成功39/40：Spatial、Object、Goal各10/10，Long9/10。init44在已知EASE记录和官方物理任务别名范围内未使用；固定全部40任务，未按成功率筛选子集。预设健康门为总至少32/40、各suite至少6/10，已通过。

Pi0权重、归一化、H50/R5和FullD10沿用既有已接受合同；动作上限恢复各suite的220/280/300/520策略步，另加10步settling，seed7。该结果支持在这组任务上继续寻找可用路径，不能证明LIBERO90失败的确定原因，也不支持任何路径互补结论。

[分析](analysis.json)、[事前协议](protocol.json)、[启动前核验](prelaunch_acceptance.json)和[完整只读重建审计](audit.json)均保留。审计覆盖40案例、1,305次请求、两个完成服务，重建动作前缀、物理初态/BDDL/prompt、噪声、路由计数和日程；91个源码身份及数值/resource身份核验通过，科学记录未修改。审计器开发时的JSON整数key与字符串key比较错误及修正前输出保留在audit_development，不是模型失败或重跑。

两张卡的原scheduler/session/lease及64MiB keeper不变，新的资源池空档各保留16GiB、运行时保留8GiB加模型；[交接记录](../../../results/ease_insights_20261103/resource_pool_v2/real_handoff_acceptance.json)包含实际6→16GiB分配与旧大占位正常退出的顺序证据。校准完成后16GiB已恢复，随后接续新的路径发现实验。
