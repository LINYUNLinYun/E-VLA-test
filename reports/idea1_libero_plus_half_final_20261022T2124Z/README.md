# LIBERO-Plus 半量评测最终报告（仅Plus分支）

已完成身份有效条件 585/585；三条件配齐 195/195 case。额外历史 30 条不计入。
这里只比较三条件共同完成的 case，扰动175与原始20分开，不能外推全部10,030任务。

|样本组|配齐case|Full|V14|对应draft|
|---|---:|---:|---:|---:|
|扰动175|175|138/175 (78.9%)|89/175 (50.9%)|54/175 (30.9%)|
|原始20|20|20/20 (100.0%)|16/20 (80.0%)|8/20 (40.0%)|

本次Plus数据矩阵已完成，CALVIN／SimplerEnv／PushT等完整E4补测范围尚未完成，不能将本报告称为整轮E4结束。主代理基于固定CSV／JSON补充以下分析，未重测、剔除或改写科学结果。

## 结论速览

- Plus扰动子集上，Full为78.9%、V14为50.9%、纯draft为30.9%。V14比Full低28.0个百分点，比纯draft高20.0个百分点；这里的draft不包含FLASH验证和Full接管机制。
- V14在Object相对较好（61.4%），Goal最低（31.8%）。各类扰动中，布局下V14最接近Full（17/25对18/25）；相机视角下纯draft略高于V14（13/25对11/25）。这些是小样本描述，不是逐层因果结论。
- 原始20例对照中Full为20/20，本轮Plus并未复现Full挂零；但之前CALVIN等跨环境零分仍未完成排查。当前没有后台实验，未完成分支和视频缺口在下表单独列出。

成功率按母suite、扰动类别和难度见第2—3节；真实末端轨迹与差异口径见第5节；耗时见第6节；数据验收与完整E4剩余范围见第7节。本文数据截至北京时间2026-10-23 05:22:53的Plus完成状态，后续仅离线汇总。

## 1. 测了什么

使用已有LIBERO权重，不做Plus微调。Full与V14均生成H=50动作、每轮执行R=5、D=10；V14仅跳过VLM第3、4、9、11层（1-based），保留14个VLM block，不裁AE。匹配的FLASH纯draft按实例母suite选择四个发布head之一，不含verify、周期Full或夹爪接管，D=N/A，不是完整FLASH系统。

175个扰动实例覆盖7类×5难度、每格5例、全部40个母任务；每实例1次trial，三个条件严格配对。原始无扰动对照20例，每suite前5个母任务各1次trial，另表报告，不混成Plus榜单分数。选择在看结果前冻结；额外30条历史结果保留但不进入本矩阵，43条入选历史条件复用。

官方环境20Hz、5步settling、每回合最多600步，使用实际有效BDDL原文指令。技术失败不记作策略失败。本次585条均有效完成，技术异常0；不是10,030实例全量评测。

## 2. 分母suite结果

| Plus扰动实例的母suite | Full-D10 | V14-D10 | 匹配纯draft |
| --- | ---: | ---: | ---: |
| Spatial | 38/44（86.4%） | 23/44（52.3%） | 21/44（47.7%） |
| Object | 34/44（77.3%） | 27/44（61.4%） | 11/44（25.0%） |
| Goal | 35/44（79.5%） | 14/44（31.8%） | 14/44（31.8%） |
| Long / LIBERO-10 | 31/43（72.1%） | 25/43（58.1%） | 8/43（18.6%） |
| 合计 | 138/175（78.9%） | 89/175（50.9%） | 54/175（30.9%） |

原始无扰动对照：

| 原始suite | Full-D10 | V14-D10 | 匹配纯draft |
| --- | ---: | ---: | ---: |
| Spatial | 5/5（100%） | 5/5（100%） | 4/5（80%） |
| Object | 5/5（100%） | 5/5（100%） | 1/5（20%） |
| Goal | 5/5（100%） | 3/5（60%） | 2/5（40%） |
| Long / LIBERO-10 | 5/5（100%） | 3/5（60%） | 1/5（20%） |
| 合计 | 20/20（100%） | 16/20（80%） | 8/20（40%） |

Full在本次原始对照不低、在Plus也不挂零；但这不能证明之前跨环境零分的原因已经全部找到。两个表的母任务覆盖和初态不完全相同，不能把100%到78.9%直接当作同任务同初态的受控扰动降幅。原始20例也不是LIBERO标准全量成绩。

## 3. 扰动类型与难度

| 扰动类型（各25例） | Full-D10 | V14-D10 | 匹配纯draft |
| --- | ---: | ---: | ---: |
| 物体布局 | 18/25（72%） | 17/25（68%） | 11/25（44%） |
| 相机视角 | 20/25（80%） | 11/25（44%） | 13/25（52%） |
| 机器人初态 | 14/25（56%） | 9/25（36%） | 3/25（12%） |
| 语言指令 | 19/25（76%） | 11/25（44%） | 4/25（16%） |
| 光照 | 25/25（100%） | 18/25（72%） | 12/25（48%） |
| 背景纹理 | 20/25（80%） | 11/25（44%） | 6/25（24%） |
| 传感器噪声 | 22/25（88%） | 12/25（48%） | 5/25（20%） |

| 官方难度标签（各35例） | Full-D10 | V14-D10 | 匹配纯draft |
| --- | ---: | ---: | ---: |
| 1 | 32/35（91.4%） | 24/35（68.6%） | 12/35（34.3%） |
| 2 | 35/35（100%） | 23/35（65.7%） | 17/35（48.6%） |
| 3 | 25/35（71.4%） | 13/35（37.1%） | 8/35（22.9%） |
| 4 | 24/35（68.6%） | 17/35（48.6%） | 11/35（31.4%） |
| 5 | 22/35（62.9%） | 12/35（34.3%） | 6/35（17.1%） |

## 4. 主要发现及能说到什么程度

- 同一175例上，V14比Full低28.0个百分点，纯draft比Full低48.0个百分点；V14比draft高20.0个百分点。配对看，V14有54例“Full成功、V14失败”，也有5例反过来；draft对应87例和3例，不是每条任务都单向退化。
- V14在这批Plus中Object相对较好（61.4%），Goal最低（31.8%）；与纯draft在Goal打平，其余母suite领先。不能据此定位某一个layer的因果责任，本轮没有新增逐层消融。
- 布局扰动下V14与Full只差1例；相机视角下draft反而比V14多成功2例。说明“V14总体更好”不等于每类都更好，但单类25例且任务组合不同，不能把这一局部差异当成已证实的架构优势。
- 机器人初态在三个模型中均是最低类别，Full也只有56%；光照下Full为100%。这是本子集的描述，不证明初态扰动在所有任务上普遍最难。
- 后三档整体更难，但不是单调曲线，例如Full的难度2高于1，V14的4高于3。难度来自官方外部性能分类，并非同任务上单一物理扰动强度逐档增大；本轮不作这种因果解释。
- 纯draft在原始对照也只有40%，因此其Plus低分不能全部归因于新增扰动，更不能当作完整FLASH带verify系统的成功率。

## 5. XYZ轨迹：实际走到哪里，而非同观测动作误差

下面图使用仿真器真实末端世界坐标，不是相机截图。Full蓝色、V14橙色、draft绿色，透明点线／虚线区分，圆点为起点、X为终点。7个代表案例固定为各类别最小source_index，不按成功／失败或图形效果挑选。

数值指标先在每个案例双方都已记录的公共原生时间内计算 `d(t)=1000*||xyz_variant(t)-xyz_full(t)||₂`（mm），再取该案例的时间平均，最后跨案例取中位数；不将先结束的一条补成静止轨迹。

| 样本组 | 对比Full | 各案例平均路径距离的中位数 | 案例间25%—75%区间 |
| --- | --- | ---: | ---: |
| Plus175 | V14 | 108.7 mm | 67.1—188.8 mm |
| Plus175 | 匹配纯draft | 162.3 mm | 75.3—270.1 mm |
| 原始20 | V14 | 69.1 mm | 42.5—83.7 mm |
| 原始20 | 匹配纯draft | 106.2 mm | 52.7—257.8 mm |

这包含各自闭环执行的累计偏离、速度和任务阶段错位，不是同一个obs上预测动作的误差，不是前1—5步误差，也不能解释成机械臂每步定位误差10厘米。即使两者最终都成功，也可能时序不同而有较大同时间路径距离。仅凭这些图和夹爪信号，不能断言失败集中于接触／精细操作；当前telemetry没有物体接触GT，不补造分阶段因果结论。

## 6. 为什么单次推理较快，整条评测反而更慢

以下只描述本批不同闭环轨迹的已有计时，不是同输入独占profile或受控加速比：

| Plus175模型 | 平均执行步数／回合 | 平均回合wall | 逐请求客户端RPC均值 | wall中RPC区间外占比 |
| --- | ---: | ---: | ---: | ---: |
| Full-D10 | 256.3 | 44.7 s | 311.1 ms | 64.1% |
| V14-D10 | 392.7 | 64.3 s | 252.0 ms | 69.1% |
| 匹配纯draft | 461.4 | 72.9 s | 181.6 ms | 77.0% |

压缩条件的请求均值较低，但成功更少、更多回合跑满600步，实际执行和请求更多，因此每回合wall更长。RPC还包括客户端往返；区间外包括环境创建／重置、物理执行、渲染、图像扰动及日志等，不能全部归因于某个噪声算子。这里不能用不同轨迹的wall或请求均值之比宣称纯模型加速比。

## 7. 数据验收、视频与未完成范围

[主代理只读验收记录](root_data_acceptance.json)逐条核验585结果、42,305次请求、210,828步执行：原始／解码动作全部有限，所有请求种子符合冻结规则，R5实际执行动作逐项等于保存的预测，native裁剪与最终成功oracle一致，24个冻结父源码哈希未变。Full／V14所有请求均记录180次AE层调用（18层×D10）；四draft均为1次learned head、无VLM prefix／AE／verify调用。

195组的物理初态、指令和首请求种子一致；194组全部初始NPZ逐位一致。唯一图像差异仍是光照案例210：Full与另外两组相比腕部3像素／4通道、最大1级uint8。原因未证明、影响不宣称为零；保留全部结果，没有择优剔除或重测。2,925个保存初始数组哈希与首请求相符。

现有3个视频均为案例0的真实原始回合，H.264、256×256、20fps，ffmpeg完整解码通过，帧数分别93／98／93，恰为执行步数＋初始帧。其他6个代表案例×3组共18份视频未录制，下面索引如实标缺失；28张XYZ／XY-XZ轨迹图均已生成，但不能将“轨迹齐全”称为“对应视频全齐”。未额外重跑回合来冒充原始视频。

[接续及退出记录](../../results/idea1_v3_2_e4_supplement/half_deadline_guard_v2_20261022T155319Z/execution_1792684603/execution.json)为 `complete_585_verified`，worker／模型退出码均0，本方相关进程已退出。仅同GPU3质量评测，没有第二卡质量上线；没有新的GPU profile、模型／仿真探针或CALVIN恢复。

### 完整E4补测：哪些完成，哪些尚未完成

| 分支 | 已有数据／完成数量 | 当前状态与缺口 |
| --- | --- | --- |
| LIBERO-Plus半量 | 585/585条件；175扰动＋20原始对照，三模型全部配齐 | 质量数据已完成、已验收；28张轨迹图已齐，代表视频3份可用、18份未录制 |
| CALVIN ABC→D原生v3 | 目标100序列×Full／V14＝200条件；仅2/200完整条件，均来自seq0且完成子任务数为0 | 用户要求暂缓；seq1 Full另保存了3个已完成子任务／291步，但整链未完成，不计完整条件，不当作整链零分；没有仍在运行的进程 |
| CALVIN ABCD→D原生 | 固定100序列，尚未完成质量评测 | 缺少已验明训练来源的匹配π0权重；不能用ABC权重或π0.5替代 |
| 更正后跨环境B | CALVIN／Simpler／PushT各100案例×六条件＝1800条件；实际完成0/1800 | 仅准备和接口检查完成，未派发正式质量评测；0/1800是完成数量，不是成功率0% |
| 原生Simpler／PushT正向对照 | 尚无本轮正式质量结果登记 | PaliGemma tokenizer配置访问返回401／GatedRepoError，需合法兼容本地资源或正常HF授权；不以跨环境B或旧零分替代 |

以上分支不是可以合并分母的一组评测：Plus使用LIBERO权重；原生CALVIN使用目标域匹配权重；跨环境B是LIBERO源权重迁移。未完成／未运行条件不能算作策略失败。CALVIN仍按用户要求暂缓，本次整理没有恢复其余分支或启动新实验，旧跨环境零分也尚不能据此宣布全部解释或修复。

### CALVIN已有非零子任务证据：不是整链成绩

对原生ABC→D v3的seq1 Full保存记录进行只读交叉核验：独立`task_0/1/2.json`与汇总记录相符，291步telemetry连续，以下三个子任务各自的末步环境oracle均为成功。

| 顺序 | 已完成子任务 | 本子任务执行步数 | 累计执行步数 |
| --- | --- | ---: | ---: |
| 1 | `turn_off_led`：关闭LED | 132 | 132 |
| 2 | `push_into_drawer`：将方块推入抽屉 | 111 | 243 |
| 3 | `lift_blue_block_drawer`：从抽屉拿起蓝色方块 | 43 | 286 |

第4项`place_in_slider`只保存了5步，此时用户要求暂缓；结果文件的整链`success=null`，并没有完成或失败结论。不能把中断时子任务的`success=false`当作最终超时失败，也不能把这三个成功子任务当作三条独立成功序列。此前seq0的Full／V14都在首项`rotate_blue_block_right`达到360步上限而失败，不能用这一条序列代表100条总体表现。

这项证据仅说明匹配目标域权重、修正接口后的Full已经能够在CALVIN连续完成部分任务，不支持“完全无法工作”；它尚不足以给出总体成功率或证明旧LIBERO权重跨域零分的原因已消除。来源：[seq1原始结果](../../results/idea1_v3_2_e4_supplement/calvin_native100_v3_20261022T1210Z/execution/episodes/seq0001/full_d10/attempt000/result.json)／[逐步telemetry](../../results/idea1_v3_2_e4_supplement/calvin_native100_v3_20261022T1210Z/execution/episodes/seq0001/full_d10/attempt000/telemetry.jsonl)。本项核验未运行模型或仿真，也未改写旧记录。

剩余范围的证据与接续边界见[补测状态](../../docs/experiments/idea1/idea1_v3_2_e4_supplement_status.md)，包括暂缓收据、B冻结登记和依赖访问记录。原始v3.2 E1—E4报告保留历史口径，本报告补充的是后续Plus数据，不覆盖旧结果。

### 固定数据与图表入口

- [最终统计JSON](summary.json)、[成功率与配对差CSV](success_paired.csv)。
- [585条件固定账本](audit_ledger.json)、[逐条数据验收记录](root_data_acceptance.json)。
- [视频可用／缺失索引CSV](videos.csv)；下面按7个预定案例展示全部28张轨迹图和实际可用视频。

## 8. 可视化与视频索引

轨迹采用预先固定的各category最小source_index：0/50/100/150/200/250/300；缺少配齐结果时不挑替代样本。闭环路径距离只在双方已有的公共时间内计算，不是同观测预测误差；没有接触GT，不据夹爪推断接触阶段。

![三条件共同样本成功率](success_paired.png)

### Objects Layout · 固定 source_index=0

![Full vs V14](trajectory_000_v14_d10.png)
![Full vs V14](spatial_000_v14_d10.png)
![Full vs matching draft](trajectory_000_matching_draft.png)
![Full vs matching draft](spatial_000_matching_draft.png)

### Camera Viewpoints · 固定 source_index=50

![Full vs V14](trajectory_050_v14_d10.png)
![Full vs V14](spatial_050_v14_d10.png)
![Full vs matching draft](trajectory_050_matching_draft.png)
![Full vs matching draft](spatial_050_matching_draft.png)

### Robot Initial States · 固定 source_index=100

![Full vs V14](trajectory_100_v14_d10.png)
![Full vs V14](spatial_100_v14_d10.png)
![Full vs matching draft](trajectory_100_matching_draft.png)
![Full vs matching draft](spatial_100_matching_draft.png)

### Language Instructions · 固定 source_index=150

![Full vs V14](trajectory_150_v14_d10.png)
![Full vs V14](spatial_150_v14_d10.png)
![Full vs matching draft](trajectory_150_matching_draft.png)
![Full vs matching draft](spatial_150_matching_draft.png)

### Light Conditions · 固定 source_index=200

![Full vs V14](trajectory_200_v14_d10.png)
![Full vs V14](spatial_200_v14_d10.png)
![Full vs matching draft](trajectory_200_matching_draft.png)
![Full vs matching draft](spatial_200_matching_draft.png)

### Background Textures · 固定 source_index=250

![Full vs V14](trajectory_250_v14_d10.png)
![Full vs V14](spatial_250_v14_d10.png)
![Full vs matching draft](trajectory_250_matching_draft.png)
![Full vs matching draft](spatial_250_matching_draft.png)

### Sensor Noise · 固定 source_index=300

![Full vs V14](trajectory_300_v14_d10.png)
![Full vs V14](spatial_300_v14_d10.png)
![Full vs matching draft](trajectory_300_matching_draft.png)
![Full vs matching draft](spatial_300_matching_draft.png)

真实现有视频（缺失不补造）：

- 0 / full_d10: [MP4](../../results/idea1_v3_2_e4_supplement/libero_plus_20261022T074328Z/perturbed350/case0000/full_d10/attempt000/video.mp4)
- 0 / v14_d10: [MP4](../../results/idea1_v3_2_e4_supplement/libero_plus_20261022T074328Z/perturbed350/case0000/v14_d10/attempt000/video.mp4)
- 0 / draft_spatial: [MP4](../../results/idea1_v3_2_e4_supplement/libero_plus_20261022T074328Z/perturbed350/case0000/draft_spatial/attempt000/video.mp4)
- 50 / full_d10: 缺失（未生成）
- 50 / v14_d10: 缺失（未生成）
- 50 / draft_object: 缺失（未生成）
- 100 / full_d10: 缺失（未生成）
- 100 / v14_d10: 缺失（未生成）
- 100 / draft_goal: 缺失（未生成）
- 150 / full_d10: 缺失（未生成）
- 150 / v14_d10: 缺失（未生成）
- 150 / draft_10: 缺失（未生成）
- 200 / full_d10: 缺失（未生成）
- 200 / v14_d10: 缺失（未生成）
- 200 / draft_spatial: 缺失（未生成）
- 250 / full_d10: 缺失（未生成）
- 250 / v14_d10: 缺失（未生成）
- 250 / draft_object: 缺失（未生成）
- 300 / full_d10: 缺失（未生成）
- 300 / v14_d10: 缺失（未生成）
- 300 / draft_goal: 缺失（未生成）

既有补测[plan](../../docs/experiments/idea1/idea1_v3_2_e4_supplement_plan.md) / [status](../../docs/experiments/idea1/idea1_v3_2_e4_supplement_status.md)。本报告只读离线汇总；当前队列状态见[half live ledger](../../results/idea1_v3_2_e4_supplement/libero_plus_half_20261022T1216Z/live_ledger.json)。
