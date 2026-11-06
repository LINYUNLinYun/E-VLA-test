# 相机模块 fast path：发现集无合格候选

2026-11-06：240条新案例及80条原Full引用全部完成，通过整批核验和独立统计复算，冻结`no_candidate`。Full／双相机D5／agent／wrist分别成功77／73／28／53（各80）。两条单相机路径均未过可用性门，wrist在四suite均不低于agent，不能支持任务互补；不启动确认，init47–49保持未用。整体研究继续。

Spatial/Object/Goal/Long成功数（各20）：Full为20/20/20/17，双相机D5为20/19/19/15，agent为6/3/12/7，wrist为11/18/16/8。固定输入成本比分别约1.348、1.352、1.350，成本门通过，但不能抵消质量损失。[整批核验](discovery_acceptance.json)覆盖11,430条新请求和2,659条引用请求、六个服务及175个源码哈希；[独立统计](independent_statistics.json)无差异。[完整接续记录](completed_summary.json)保留原身份和证据哈希。

| 配置 | Full 请求 | Fast 请求 | 新发现案例 |
| --- | --- | --- | ---: |
| Full | 双相机、D10 | 无 | 0；引用原80条 |
| 双相机对照 | 双相机、D10 | 双相机、D5 | 80 |
| agent 路径 | 双相机、D10 | 更新外部相机，腕部沿用自身最近Full特征，D5 | 80 |
| wrist 路径 | 双相机、D10 | 更新腕部相机，外部沿用自身最近Full特征，D5 | 80 |

全部保留VLM18、动作专家18层、H50/R5；每次Full之后四次Fast，相机特征在投影前缓存，Fast不更新缓存，episode/prompt之间清空。两条候选只改变更新哪路相机，预算一致；双相机D5用于隔离采样步数的作用，不进入主候选对。仍使用40个任务、init45–46、seed7、10步settling与原suite动作上限。

原生视觉入口的`num_views`参数未被使用，单改参数不能减少工作。实际输入和scratch张量必须变成batch1。首版使用通用batch1投影，和双视角的BF16分块累加顺序不同，在数值门失败、未产生闭环案例；[失败记录](../camera_paths_v1/calibration_rejection.json)保留。独立版本复用原双视角投影kernel、改为单视角序列长度，数值门通过；随后发现旧收据校验只接受D2/D10，因而[未启动发现集](../camera_paths_v2/D5_validator_block.json)。当前版本增加严格D5/D10节点校验，未改变模型、条件、数值容差或选择门槛。

当前两服务各通过632项GPU检查，包含原D5/D10采样器对照、实际AE层访问/NFE、真实batch1、同观测与非平凡混合视角参考、完整prefix与动作等价，以及自身Full缓存的周期和重置。四条已知Full控制的逐请求输入、噪声、动作/raw及结果与原记录逐位相同，[校准验收](calibration_acceptance.json)通过。这是有限输入的运行时校准，不是新的独立质量证据。35项CPU检查通过，175个源码哈希冻结；[协议](protocol.json)和[启动前核验](prelaunch_discovery.json)保留。

原80条Full记录保留其目录、manifest、condition和收据身份，不复制改名，也不算新复验。240条新案例与80条引用构成320个分析单元。全部结束后才重建每条记录并核对物理首输入/噪声，独立重算整数成功数、计时和候选排序。生成报告使用`camera_paths_v3.analysis_entry`，其明确引用范围为原Full80条。

主候选对固定为agent/wrist。两条均须总体≥.75且≥Full−.15，完成服务的固定输入策略成本比≥1.15、彼此成本差≤10%；两个不同suite的反向差值各≥10pp、至少两个正向task，去掉任一task后仍为正。仅合格后才冻结方向并在init47–49运行360条独立确认；十task sign-flip、Holm2和20k whole-task bootstrap按协议判断。没有候选则不启动确认、不降低门槛，并继续整体目标。

固定输入计时是实际策略成本，包含缓存和输出传输，不包括RPC或环境；共享GPU负载仍可能影响比值，不能称为闭环加速。当前设计也不能证明任务内部phase切换或在线选择收益。原GPU lease保持不变，resource_pool_v4在空档各保留16GiB，模型运行时保留8GiB加模型。
