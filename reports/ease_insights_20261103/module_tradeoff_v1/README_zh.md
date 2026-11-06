# VLM／AE模块组合的新发现集

2026-11-06：320条新案例与160条原身份控制引用全部完成，整批记录重建和独立统计复算通过，冻结选择为`no_candidate`。AE16与VLM_c均通过可用性和成本门，但AE16在四个suite均不低于VLM_c，尚不能支持任务互补或自适应选择。整体研究继续，两张原GPU lease保留，空档各恢复16GiB。

Full、VLM_c、AE16、AE14、AE12、混合路径分别成功77、70、74、64、24、62（各80个分析单元）。AE16的Spatial/Object/Goal/Long为20/20/17/17，VLM_c为17/19/17/17。没有启动360条确认，init47–49保持未用；[整批核验](discovery_acceptance.json)覆盖13,736条新请求和5,795条原控制请求，[独立统计](independent_statistics.json)未发现差异。

[科研图与CSV](../module_tradeoff_v1_figures/README_zh.md)保留逐suite成功数、AE16−VLM_c差值和固定输入成本，可导出PNG/PDF/SVG；图不含正在运行的相机实验结果。

| 配置 | 跳过VLM层 | 跳过AE层 | 质量数据来源 |
| --- | --- | --- | --- |
| Full | 无 | 无 | 原Full80条，保留原身份引用 |
| vlm_c | 4,7,10,13,15,17 | 无 | 原spread_c80条，保留原身份引用 |
| ae_16 | 无 | 2,8 | 新80条 |
| ae_14 | 无 | 2,6,7,8 | 新80条 |
| ae_12 | 无 | 2,4,6,7,8,9 | 新80条 |
| hybrid_15_15 | 2,3,17 | 2,7,8 | 新80条 |

层号从1计，AE跳层只发生在FastD2；每三次请求仍以完整VLM18／AE18的D10刷新。H50/R5、视觉27层、自身VLM残差外推、动作预算及seed7一致。空VLM mask也保留周期时间戳，避免AE路径始终退回Full。模块层数不代表同等计算，配对候选另要求实际有限bank策略成本相差不超过10%。

VLM_c是已知400条中唯一合格路径，属于明确的数据后选锚点；AE集合按历史全局单层动作NMSE低值提出，不假定多层可加或具备任务偏好。四个新路径使用已知的init45–46作探索，全部40任务保留。**新增320条与引用160条构成480个分析单元，引用不算新增复验。** 原目录、manifest、condition和收据不复制改名；派生别名`vlm_c`始终绑定原`spread_c`。

启动前两服务各通过488项GPU检查，覆盖原采样器不跳AE时D10/D2的raw／decoded数值等价、VLM残差算术、真实AE helper访问与内部actualNFE。四个已知Full控制案例的动作/raw数组均与原记录逐位相同，包括逐请求输入、噪声及结果；[校准核验](calibration_acceptance.json)通过。这四条是运行时桥接，不是独立质量证据。原生fusedD10与customD2采样代码保持不变，AE层仅通过独立版本的进程内调用包装跳过，每次还原函数绑定。

[协议](protocol.json)、[校准启动前核验](prelaunch_calibration.json)、[发现启动前核验](prelaunch_discovery.json)与冻结manifest均已保留。18项校准CPU检查及3项选择/引用计数检查通过；启动前只读验证了160条控制引用。完成后已逐条重建320条新记录及160条旧记录、核对相同物理首输入/噪声，再冻结阴性选择。

选择保持总体≥.75且≥Full−.15、成本比≥1.15、两个不同suite的反向差值各≥10pp、至少两个正向task和每task去除后仍为正；另加两路径成本比相差≤10%。只冻一对，全部320条前不分析部分成功率挑候选。合格后才用未使用的init47–49跑360条独立确认；十task sign-flip含ties、Holm2、20k whole-task bootstrap、seed20261106及两路径可用性/成本均按预设规则判断。没有候选就不启动确认或降低门槛。

当前发现manifest内容hash：`2d00735fd5f9b8afb06eaadeec47087db7f049eadcffbe2ff92def91ca164e43`。本实验暂不能支持稳定任务偏好、runtime选择或独占GPU加速；固定输入计时与质量结果分开解释。
