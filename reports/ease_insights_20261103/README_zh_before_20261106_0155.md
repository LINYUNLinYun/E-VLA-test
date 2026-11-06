历史快照，包含当时未完成的状态；当前结论以README_zh.md为准。

# EASE 两条 insight：当前证据与未完成项

2026-11-06 接续：[模块发现实验](module_tradeoff_v1/README_zh.md)的320条新路径案例及160条原身份控制引用已全部完成并验收。AE16成功74/80、VLM_c成功70/80；AE16在四suite均不低于VLM_c，冻结`no_candidate`。[等预算相机路径](camera_paths_v3/README_zh.md)的240条新案例及80条Full引用也已完成并验收：Full／双相机D5／agent／wrist为77／73／28／53（各80），两候选未过可用性门，同样冻结`no_candidate`。[动作专家深度与步数组合](denoise_depth_paths_v1/README_zh.md)的320条新案例及160条双来源原控制已通过整批记录和独立统计验收：Full／AE16D2／AE18D2／AE14D3／AE12D4／AE16D3为77／74／74／68／35／76（各80），仍无满足原规则的互补对。一次首次输入配对技术中断按原门恢复，没有重跑有效科学结果；记录及成本处理单独留档。init47–49仍未用。

第二条insight仍待验证。[固定AE16/D2联合层组合筛选](ae_complement_beam_v1/README_zh.md)已完成110条历史观测、40个任务、三组噪声和全部153个跳过两层的组合，筛出[2,13]与[7,14]的离线命令误差偏好。但随后[完整闭环检验](ae_complement_beam_v2/README_zh.md)未支持它：两路径仅60/80与62/80，Object/Spatial均19/20，Long均6/20；质量门和固定方向均不通过，冻结no_candidate，不运行确认。全部160条新案例、160条原控制及独立统计验收通过，代理图不能作为成功率偏好证据。

现已接续[AE14/D3局部层交换](ae_complement_beam_v3/README_zh.md)：以原可用路径68/80作参照，在相同14层、D3预算下测试四个各交换一层的配置，直接比较闭环结果。8条已知Full/AE14D3控制已全部逐位复现，1,728条成本探针验收通过；203个源码与协议哈希固定、26项CPU检查通过，320条新发现案例已提交，原160条控制仍保留双来源身份。init47–49继续留作合格候选的独立确认。原GPU lease持续持有，resource_pool_v4两卡空档各16GiB，模型运行时8GiB占位加模型。下方历史段落保留各自时间范围。

2026-11-05 更新。**固定 draft 的跨 suite 差异已在独立初态上复现；原 Jac/MSE 和后续 Jac/local_long 两组路径的预定互补性均未复现。** 后者720条发现和600条独立复验已完成，数据检查与独立统计复算均通过，阴性结论保留。同一LIBERO90 suite内的[432条新对照](vlm_within_suite_v1/README_zh.md)也已完成并通过整批核验，但Full仅7/72成功，冻结选择为`no_candidate`。[新的常用四suite Full校准](baseline_recovery_v1/README_zh.md)在独立init44达到39/40。随后四条更大层位置差异路径的[400条实验](vlm_diversity_v2/README_zh.md)完成并通过整批记录及独立统计复算：Full77/80，四路径63/62/70/60（各80）。仅spread_c通过预设可用性与成本门，冻结选择为`no_candidate`，不启动其360条确认；init47–49仍未用。第二条insight尚未确立，正准备VLM/AE不同模块组合。两张GPU继续保持原lease，空档各保留16GiB真实CUDA占位，模型运行时保留8GiB占位加模型。下方历史快照保留其原时间范围。

## 固定 draft：支持有限范围内的跨 suite 局限

发现矩阵360例、独立确认400例完整。目标 suite、Full 权重、共同归一化合同、任务和初态固定，仅改变发布 draft head 的来源。独立确认使用未参与筛选的 init21–24。

| 固定目标 | 匹配纯 draft | 外来纯 draft | 匹配 FLASH | 外来 FLASH | Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Object；外来为 Spatial | 40/40 | 0/40 | 40/40 | 39/40 | 40/40 |
| Spatial；外来为 Object | 35/40 | 0/40 | 39/40 | 39/40 | 37/40 |

两条预先冻结的纯 draft 主比较经 Holm 校正仍显著；任务聚类区间及固定任务多初态的相关性限制详见[独立审阅](draft_confirmed_audit/README_zh.md)。完整 FLASH 的外来 head 在两个目标上的动作接受率均为0%，依靠 Full 仍各成功39/40。因此，单看系统成功率会漏掉 draft 自身的失效。

[事后任务聚类敏感性](draft_task_sensitivity/README_zh.md)重读160份原纯draft结果，将四个初态归在同一task内：两目标各10个task均为匹配来源更好，task signflip的Holm2 p均为0.001953125。它不新增验证案例、不替代原预定case检验，也不补齐训练/预处理溯源。[科研图与CSV](draft_confirmation_figures/README_zh.md)另保留完整FLASH、接受率、请求回退及全部初态实际计时口径。

| 固定目标 | 动作接受率：匹配／外来 | 拒绝→Full 请求比例：匹配／外来 | native infer ms/action：匹配／外来／Full | RPC ms/action：匹配／外来／Full |
| --- | --- | --- | --- | --- |
| Object | 71.894%／0% | 13.465%／47.886% | 7.185／9.486／8.234 | 7.512／9.954／8.504 |
| Spatial | 56.417%／0% | 13.949%／47.167% | 12.492／15.608／11.419 | 12.907／16.101／11.687 |

这里是全部 init21–24 的累计量比值，包括执行0动作的拒绝请求。接受率按验证动作预算加权；配对统计中的 case 等权均值是另一口径。native infer 包括原生策略调用，不含外层噪声生成、hash和RPC；RPC列覆盖客户端往返。观测成本受冷编译、轨迹和共享负载影响，不能把全部差异归因于泛化，也不能称匹配 FLASH 在两个 suite 都已快于 Full。预先固定的 repeat-task init22–24 子集中，Object 的匹配／外来／Full native infer 成本为2.280／4.171／3.397ms/action，Spatial为2.560／4.275／3.441：两组都观察到匹配低于Full、外来高于Full，但该子集不能称已证明完全热态。完整计时子集、墙钟、请求数及统计见[确认统计](draft_statistics/README.md)。400份结果和6,171条请求已独立重算通过。

结论限于这些发布 head 在共同 Pi0 推理合同下、固定任务的新初态表现。缺少原始训练缓存的完整归一化与配置溯源，不能排除全部历史预处理混淆；[归一化与噪声审计](source_normalization_audit/README_20261104.md)说明了证据边界。没有独立 Spec-VLA 结果，不能推广到所有额外训练 draft，也没有测量重新训练适配成本。

[历史 LIBERO-Plus 配对结果](perturbation_context/README_zh.md)可作补充：175个扰动案例中，Full/V14/纯 draft 成功138/89/54。原始20例与175例不构成同任务同初态的原始—扰动对照，不能据此估计扰动造成的因果降幅。

## 原 Jac/MSE fast path：可用路径存在，预定互补假设未复现

VLM发现160例和独立确认600例均完整。Full成功182/200，固定选出的Jac/MSE路径分别成功170/200与170/200。两条路径保持相同的12/18保留层预算、Full-D10→Fast-D2→Fast-D2周期、H50/R5与残差补偿。它们的固定输入策略成本优势不等于闭环加速或质量非劣。

| 预定方向 | 独立成功数 | 差值 | Holm3 |
| --- | --- | ---: | ---: |
| Spatial：Jac > MSE | 47/50 对46/50 | +2pp | 1 |
| Long：MSE > Jac | 34/50 对34/50 | +0pp | 1 |
| 事先登记的补充 Goal：Jac > MSE | 39/50 对43/50 | -8pp | 1 |

三方向的case与task聚类区间均跨0，均未满足预先固定的稳定规则。原Spatial/Long主方向及预先登记的Goal/Long补充方向均未复现；不显著也不证明路径等价。[600例正式主/补审阅](vlm_confirmed_audit/README_zh.md)已完成，补充报告中的原主分析与[正式主报告](vlm_confirmation/analysis.md)逐项核对一致；原Holm2保留，三方向另给Holm3。此前vision pilot同样未复现预定互补方向，结果保留在[vision审阅](vision_pilot/scientific_verdict.md)。

Full、Jac与MSE的质量结果全部保留原始成功/失败，没有因资源争用改写。policy/RPC/episode墙钟及固定输入探针成本都可能受外部负载影响；episode墙钟之和也不是并行makespan，因此不宣称独占GPU加速。完整成本、争用记录和逐task/init稳定性见正式报告。

[原600例科研图与CSV](vlm_confirmation_figures/README_zh.md)提供各suite成功数和三个预定方向的task-cluster区间；PNG、PDF、SVG及输入哈希均可复现。

逐40任务及预定义语义类别的[后续探索](vlm_pair_interim/task_semantic_exploration.md)也未找到稳健的反向候选。新候选如使用这批数据筛选，必须标为新发现，并用新的初态独立验证；不得改写本轮预定方向。

## 正在继续

Full确认此前曾在模型初始化阶段因同卡外部占用OOM，未产生有效episode。原失败记录保留；相同科学命令用新队列ID恢复，在GPU3通过连续60秒自然空闲检查后完成Full200/200。已完成的科学结果没有重跑。

原600条已完成正式主分析、事先登记的补充分析和资源争用审阅；这只表示原确认矩阵完成，不表示整个研究目标完成。后续局部搜索720条筛出Jac/local_long，但init39–43的600条独立复验未复现：Object的Jac优势变为−2pp，Long的local_long优势为+4pp，task-cluster区间分别为[−8,4]与[−4,12]pp，任务级Holm2分别0.875和0.625。Full/Jac/local_long总体成功190/168/165（各200条）；全部有效策略失败保留。完整[独立复验与审阅](vlm_local_search_confirmation/README_zh.md)及[图表](vlm_local_confirmation_figures/README.md)不会改写原Jac/MSE阴性结果。

该独立分支固定LIBERO90内24个元数据任务，12个state-change-only、12个single-object-placement；432条发现矩阵现已完成，整批核验覆盖33,384条请求。由于无合格候选，不追加360条独立复验。Full低成功率的原因仍待区分；CPU核对已确认157个既有prompt缓存与基座权重逐值一致，不能把缓存覆盖直接当作模型训练覆盖。常用四suite的独立Full校准达到39/40（Spatial/Object/Goal各10/10，Long9/10），通过事前健康门；它不是互补性或低成功原因的因果证明。后续400条几何路径实验也已完成、通过16,286条请求的整批审计和独立候选排序复算，但只有一条路径合格，未找到可用互补对。选择和复验门槛未降低，原有效失败全部保留。正在[准备不同模块组合](module_tradeoff_v1_preparation/README_zh.md)，现仅有14项CPU检查及原采样源桥接核验，尚无其GPU数值门或科学清单。GPU0/1的原会话、lease及小keeper保持不变，resource_pool_v2已在400条结束后各恢复16GiB；第二条insight的研究尚未完成。

## 补充：AE历史结果复核（2026-11-05）

等待本机GPU期间，复核了AE单层L2/L8历史pilot的800条episode（400对case）。L2为356/400、L8为354/400，差+0.5pp，40-task探索区间[-2.75,3.75]pp。既有metadata语义类别的区间均含0；固定trial0–4选路径、5–9评估仅比训练选择的固定路径多2/200例，条件区间也含0。独立原始记录重建通过，未发现可充当稳定任务偏好的证据。

[完整库存与追溯](ae_existing_evidence_inventory/README_zh.md)保留逐case结果、历史H50/R12/D10配置证据、配对和初态/模型/源码绑定边界。它不能与当前R5局部搜索合并，也不改变其冻结候选、阈值或选择规则。当时局部搜索为120/720，正按15秒连续空闲/2秒检查等待GPU；该段为资源释放前的历史截面，现已完成720条并进入复验。

## GPU恢复后较早进度快照（2026-11-05；已由上方720条结果更新）

本轮已在自然释放后持续获得GPU0/1，真实15秒空闲门已核验。Full与local_long各120案例完成，新增Goal/Spatial正在运行。此快照累计提交561/720（生产提交数，最终全矩阵独立验收仍待完成）；原Jac120保留，已独立抽验6条新condition-case及同case的3条Jac对照。

[资源接续核验](gpu_continuous_handoff_20261105/first_handoff/README_zh.md)确认同一组keeper跨首个任务边界保留CUDA占用。端口复用曾触发技术暂停，随后只用新technical任务ID和显式resume恢复；源文件、科学参数与原valid outcomes均未更改。第二条insight仍待冻结选择及独立确认，以上运行进度不等于互补性结论。
