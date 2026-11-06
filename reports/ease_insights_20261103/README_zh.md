# EASE 两条 insight：当前证据与未完成项

2026-11-06 03:08 UTC 更新。**第一条有界的发布draft跨suite迁移证据可用；第二条尚未找到差异可复现的任务类别互补路径。** 研究目标继续，不能把可用子路径、成本优势、离线误差差异或相反点估计直接当作互补性复现。

[图像位置注意力屏蔽C6](camera_paths_v6/README_zh.md)已完成并独立验收：160条新轨迹加160条原始对照，中央位置52/80、棋盘位置70/80；棋盘路径四个suite点估计均更高，无互补候选，不做确认。另一次对960条历史R5等NFE时间网格轨迹的预定探索也没有候选，见[历史时间网格](historical_time_grid_v1/completed_exploration.json)。

正在运行[固定VLM注意力／FFN组件比较C7](camera_paths_v7/README_zh.md)的160条新闭环探索。在相同6层分别跳过注意力或FFN，保留当前观测K/V、完整新视觉与AE18；两者均为FullD10＋两次FastD2，并采用相同残差补偿。22项CPU检查和[GPU校准](camera_paths_v7/calibration_acceptance.json)通过：8条已知控制逐位一致、1152次成本探测验收；Full/path分别1.544与1.601，配对成本差3.7%。新探索另保留原始Full/DenseD2各80条对照，完整矩阵后再分析。实时计数见[探索进度](camera_paths_v7/progress.json)。条件确认流程另通过19项CPU检查，尚未解锁。

原来两张GPU的session、lease和keeper保持不变：空档每卡16GiB；模型运行时8GiB占位加实际模型。旧失败队列不自动重试，技术恢复绑定原guardian身份。GPU释放检测仍由唯一只读watcher按2秒检查，连续自然空闲门为15秒。

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


## 内部路径：可用性成立，类别互补性仍未成立

[跨模块独立复验](ae_complement_beam_v4/README_zh.md)的720条新rollout、12个服务和27,318次请求完整验收，独立统计无差异。Full/VLMc/AE14D3为224/206/195（各240rollout），两候选均通过质量与成本门，固定输入Full/路径成本比分别1.6048与1.4944。

| 冻结方向 | 成功差 | 十任务Holm2 p | 整任务95%区间 |
| --- | ---: | ---: | --- |
| Spatial：AE14D3 > VLMc | +1/60，+1.67pp | 0.5 | [-3.33,6.67]pp |
| Long：VLMc > AE14D3 | +5/60，+8.33pp | 0.3125 | [-3.33,20.00]pp |

结论为 **not_reproduced**，Spatial的逐任务删除稳健性也未通过。每条件240rollout包含3个初态×2个噪声流；120个物理案例不是240个独立任务，主推断以每类别10个任务为单位。47—49相对当前R5发现集45—46留出，但[扩展历史库存](historical_policy_initials_v1/audit.json)表明所有官方0—49初态都有旧R12记录，不能再称全研究从未使用的初态或未见任务。

[两路新鲜视觉层预算](camera_paths_v4/README_zh.md)的160条新候选和160条原控制引用也完整验收。Full/DenseD2/agent27+wrist18/agent18+wrist27成功77/74/70/73（各80）。两候选总体可用，且成本比1.4085/1.4126相近。Goal的A优势仅2个正任务，Spatial/Long的B优势仅2/3个正任务，未达事先规定的各至少4个任务，冻结 **no_candidate**，不启动该族800例确认。

[紧凑前缀的数值校准](camera_paths_v5/completed_summary.json)在两张GPU上相同地失败：原始/恒等路径先通过，但紧凑存储与独立完整存储屏蔽参考的最大raw/decoded差为0.01218/0.00800，超过预设容差。0条闭环案例、0条控制轨迹；未提高容差，保留失败。后续v6改变实现，保持原始完整存储，仅在softmax屏蔽连接，通过独立参考1e-5及原控制逐位桥接后才进入发现。两版不能合并为一个已通过的数值结果。

更早的有效阴性结果继续保留：

| 分支 | 数据范围 | 结论／边界 |
| --- | --- | --- |
| [Jac/MSE](vlm_confirmed_audit/README_zh.md) | 160发现＋600确认 | 固定方向未复现，Full182/200，两路径各170/200 |
| [Jac/local_long](vlm_local_search_confirmation/README_zh.md) | 720发现＋600确认 | Object/Long任务级Holm2为0.875/0.625，未复现 |
| [LIBERO90内部对照](vlm_within_suite_v1/README_zh.md) | 432新案例 | Full仅7/72，不据此筛选可用互补对 |
| [四种VLM位置组合](vlm_diversity_v2/README_zh.md) | 400新案例 | 只有一条路径合格，无候选 |
| [VLM/AE模块组合](module_tradeoff_v1/README_zh.md) | 320新＋160原控制 | AE16在四suite均不低于VLMc，无候选 |
| [相机缓存路径](camera_paths_v3/README_zh.md) | 240新＋80原Full | 单新鲜相机候选质量未过门 |
| [AE深度／步数组合](denoise_depth_paths_v1/README_zh.md) | 320新＋160原控制 | 无合格互补对 |
| [AE16离线候选闭环](ae_complement_beam_v2/README_zh.md) | 160新＋160原控制 | 离线代理偏好未转成成功率偏好 |
| [AE14局部层交换](ae_complement_beam_v3/README_zh.md) | 320新＋160原控制 | 四邻居质量不足，无候选 |
| [AE单层历史审阅](ae_existing_evidence_inventory/README_zh.md) | 800条旧R12记录 | +0.5pp，任务区间跨0，不能与R5合并 |
| [旧14变体探索](legacy_path_pool_v1/README_zh.md) | 5600历史R12发现记录 | 无候选，后续30—39验证结果未读取 |

当前v6仍按冻结规则前进：全部160新＋160原控制验收后，才决定是否存在两个相反方向各至少10个百分点、至少4个正任务、逐任务删除仍为正的候选。只有合格对解锁10—14初态的800例确认（含Full和DenseD2）。主检验为十任务符号翻转＋Holm2、整任务bootstrap和质量／真实成本门；不得以噪声重复代替更多任务，不重选方向或追加当前矩阵来追求显著性。

所有原始有效成功和失败都保留。固定输入推理成本、模块逻辑工作量与闭环加速分开报告；没有独占GPU、阶段选择、优于最佳固定路径或全研究5%错误率的声明。若DenseD2的质量与成本更好，路径间点估计交叉本身也不能证明自适应路由收益。

[机器可读证据状态](goal_evidence_status_v1/status.json)与各族完成摘要、冻结manifest、逐案例原始动作和独立审计是依据；[更新前快照](README_zh_before_20261106_0155.md)仅用于追溯，含当时已过期的进行中描述。
