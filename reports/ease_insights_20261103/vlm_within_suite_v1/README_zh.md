# 同一suite内的fast path任务类别对照

状态：432条发现阶段完成，整批案例/动作/数值核验及资源来源核验均通过；冻结选择为`no_candidate`，不启动360条独立复验。Full仅7/72成功，五条路径均未达到绝对可用门槛，当前任务池不能给出所需的可用互补路径证据。GPU0/1各保留16GiB真实CUDA占位，研究目标继续。

| 条件 | 成功数／72 |
| --- | ---: |
| Full | 7 |
| Jac | 6 |
| local_spatial | 8 |
| local_object | 5 |
| local_goal | 7 |
| local_long | 7 |

固定旧输入bank的五条路径周期成本比分别约1.63–1.65，均通过成本门，但不能弥补总体成功率低于.75。Full在两类任务中分别为2/36与5/36；这些结果不能解释为剪层造成了全部失败，也不能据此断言模型不存在互补路径。

结果及证据见[冻结选择](../../../results/ease_insights_20261103/vlm_within_suite_v1/discovery/analysis/selection.json)、[完整432条核验](../vlm_within_suite_v1_preparation/full_discovery_audit/audit.json)和[资源核验](../vlm_within_suite_v1_preparation/reservoir_resource_audit/audit.json)。432条核验覆盖33,384条请求与六个实际服务；有效失败全部保留，原manifest及选择门槛未更改。

[基线诊断](../within_suite_baseline_diagnosis/diagnosis.json)已排除一项输入问题：157个已有prompt缓存均与基座CPU embedding lookup逐值一致，包含本轮全部20条指令。公开训练数据说明及本地配置指向四个常用suite的40任务，不含LIBERO90；由于转换权重的完整训练履历尚未绑定，这只是任务覆盖的线索，不能作为本次低成功率的确定因果解释。后续先核对Full适用任务池，再另立有界路径实验，不在低Full成功率上降低门槛挑候选。

保卡版本先各分配16GiB，在模型运行时保留8GiB占位并容纳模型，结束后恢复16GiB；原64MiB keeper和GPU lease归属保持不变。[实际分配记录](../../../results/ease_insights_20261103/reservoir_v1/first_real_acquisition.json)及资源核验记录保留了PID、nonce、UUID和真实NVML占用的一致性范围。随后已独立版本化交接到resource_pool_v2：新6GiB真实占位成功后旧大占位才正常退出，新占位扩到16GiB；[两卡交接核验](../../../results/ease_insights_20261103/resource_pool_v2/real_handoff_acceptance.json)通过。常用四suite的新Full校准39/40及其后400条新路径发现实验独立记录在[总报告](../README_zh.md)，不修改本分支阴性结果。

本分支在LIBERO90内固定24个任务：12个state-change-only、12个single-object-placement，来自指令与BDDL元数据，不预设难度或路径偏好。任务子集的固定SHA选择早于本轮结果；类别方向是在披露既往阴性结果后的新研究假设，并非完全无历史信息的盲设计。

六个条件沿用Full及五条固定12/18层VLM路径，H50/R5、D10→D2→D2周期、AE与残差补偿预算保持一致。统一400个策略动作，另加10个settling动作，seed7；不在不同类别使用不同步数。24个任务有20条独立文本，其embedding lookup资产已核对；Goal cache标签不表示执行Goal learned draft head。

发现阶段6×24×init36–38，共432条。冻结规则要求路径总体成功率≥.75且≥Full−.15，固定输入周期成本比≥1.15；两个类别的反向差值各≥10pp、至少两个正向task且每task去除后仍为正。按固定排序只选一对；若有合格候选，Full与两条路径使用init39–43复验360条，合计预算上限792。

确认以每类12task为推断单位，枚举4096个单侧task-block sign flips、两方向Holm校正，20k paired-case/task bootstrap seed20261105与线性quantile；方向、任务聚类区间、LOO与两路径总体可用性均须满足冻结条件。无候选或未复现只结束本有界分支，不改写此前阴性结果，也不证明路径等价。多轮探索不宣称全研究整体5%错误率保证。

旧四个真实输入的332项数值门与acceptedJac动作参考继续使用，容差不放宽；这是有限旧bank核验，不代表新24任务、20文本均已做数值oracle覆盖。实际新案例另核官方BDDL/初态向量、prompt、首观测和noise身份。动作检查覆盖导出日程，不把它称为内部实际NFE计数。

冻结清单：`results/ease_insights_20261103/vlm_within_suite_v1/discovery/manifest.json`，manifest内容hash `6db11c5a8f1cdb5eab549f406ece3ba7b14661b2699da407c1ac87073e6c8a7d`，文件SHA `e9dc20f236a74da7e10493488ea644f0050501159f0c2e390e77c8f2ab653ddb`。原GPU会话保留、CPU发布器停用及借用锁证据见[准备目录](../vlm_within_suite_v1_preparation/)。启动曾在模型前遇到NFS共享锁FD与额外GPU负载，两次均0案例；未执行清单、日志和恢复证据完整保留，有效策略失败不重跑。

此前Jac/MSE及Jac/local_long独立复验均为阴性；后者的[完整600条结果](../vlm_local_search_confirmation/README_zh.md)与图表保持不变。第一条insight仍限于共同Pi0推理合同下已发布draft heads的可移植性证据，不因此扩大为训练分布的普遍因果结论。
