# 跨模块 fast path 独立复验 v4：未复现

720 次复验已完成并通过完整案例验收和独立统计复算，但这对路径没有复现预设的任务类别互补性。结果保持 `not_reproduced`，不追加案例，不改 mask、类别、噪声、主检验或阈值；总体研究目标继续。

| 条件 | Spatial /60 | Object /60 | Goal /60 | Long /60 | 总成功 /240 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full D10 | 60 | 59 | 56 | 49 | 224 |
| VLM12 / AE18，Fast D2 | 56 | 54 | 51 | 45 | 206 |
| VLM18 / AE14，Fast D3 | 57 | 53 | 45 | 40 | 195 |

| 冻结方向 | 留出差值 | 任务 bootstrap 95% 区间 | 任务检验 Holm2 p | 复现 |
| --- | ---: | ---: | ---: | --- |
| Spatial：AE14 > VLM | +1.7pp | [-3.3, +6.7]pp | 0.5000 | 否 |
| Long：VLM > AE14 | +8.3pp | [-3.3, +20.0]pp | 0.3125 | 否 |

Spatial 去掉部分单个 task 后差值会变成0，Long 虽然逐task leave-one-out仍为正，但两方向的任务层区间都跨过0。历史发现中 +15pp、+30pp 的差值在新 R5 留出复验中明显缩小，不能用正点估计宣布稳定侧重。

两条路径均通过整体成功率≥.75且≥Full−.15的门槛。12个已完成服务的同服务 Full/path 原生策略成本比中位数分别为 VLM 1.6048、AE14 1.4944，路径成本因子1.0739，均满足 Full/path≥1.15且路径间差≤10%。这是固定输入的原生策略成本，不能当成闭环或独占GPU加速。

实验为40个task × 官方init47–49 × 两组预定模型噪声 × 三条件，共720次rollout。每条件为240次rollout、120个物理案例状态。每个目标suite的10个task始终是主统计单位，每task六个配对观测不可当成六个独立task。主检验枚举1024个整任务符号组合并做Holm2；20000次整任务bootstrap使用seed20261111，整数先求和后除以60、linear percentile分位数。

全部720条记录、27318次请求、12个服务已只读重建；原始动作、实际执行前缀、物理初态、输入/噪声配对、VLM/AE访问、真实NFE/采样节点和资源身份均通过验收。独立统计实现与正式分析没有差异。12条已知控制的校准轨迹此前已逐位复现；这些重复控制不计入720次新复验。

初态使用范围补查覆盖本轮EASE记录及Pi0历史目录979份原生episode_log、40129条身份记录。更早项目使用过全部0–49初态，47–49对应的57组旧实验均为R12；本轮为R5。因此“留出”指相对于本轮R5 init45–46候选筛选，不是全项目首次使用的物理状态，也不是未见任务泛化。冻结方案与主检验未因补查改动。

[完整总结](completed_summary.json) · [案例验收](confirmation_acceptance.json) · [正式分析](confirmation_analysis.json) · [独立统计](confirmation_independent_statistics.json) · [流水线验收](confirmation_pipeline_acceptance.json) · [历史配置补查](prior_configuration_audit.json)

[结果图PNG](../ae_complement_beam_v4_figures_v2/fastpath_replication.png) · [PDF](../ae_complement_beam_v4_figures_v2/fastpath_replication.pdf) · [SVG](../ae_complement_beam_v4_figures_v2/fastpath_replication.svg) · [成功率CSV](../ae_complement_beam_v4_figures_v2/successes.csv) · [配对差值CSV](../ae_complement_beam_v4_figures_v2/paired_differences.csv)

本结果不否定所有内部路径都没有差异，只说明这对路径在该合同下不足以支撑第二条insight。接续工作仍需找到类别偏好并独立复现，不能用两个固定任务的更窄问题替代。本机两张已持有GPU继续在空档保留各16GiB，原租约不变。
