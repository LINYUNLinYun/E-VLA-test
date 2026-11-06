# idea1 v3.2 最终实验报告

原始报告：2026-10-22；补测数据汇总更新：2026-10-23。原始需求：[idea1.md](../../../idea1.md)；[执行方案](idea1_v3_2_experiment_plan.md)；[运行与技术事件](idea1_v3_2_execution_status.md)。

后续E4补测已直接汇总进本文第4.5—4.13节，包含Plus成功率总表、分suite／扰动／难度分析、真实XYZ轨迹、耗时、视频与剩余范围；不必另开报告看数据。Plus半量已完成585/585条件；CALVIN扩大矩阵及其他跨环境／原生分支仍未完成，具体见第4.12节。单独的[Plus数据与资产报告](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/README.md)保留为可追溯附件。

原始E1—E4范围已完成，后续E4补测的既定Plus三条件分支完成，整轮跨环境补测尚未结束。原始E4的108个条件键全部有有效原生环境闭环结果，历史30次启动技术失败保留，不计为策略失败；这里的“原生环境”不代表使用目标域训练权重。原跨环境补测仍暂缓，新增Plus V14-D2／四draft诊断按最新请求推进，见第4.14节。实验完成不等于预设假设得到支持：尤其本轮不能证明“draft参数少，因此只能泛化到LIBERO”。

| 实验 | 主要发现 | 结论 |
| --- | --- | --- |
| E1 batch | bank吞吐最高1.51×，真实评测最佳动作吞吐仅+0.52%，存在数值偏差 | 保留原B1评测配置 |
| E2 残差复用 | AE约3.91×、请求约1.19×；34/40，与Full同分；D2为35/40 | 相对D10可加速，但没有证明优于直接D2 |
| E3 外推替代 | 51候选均未通过尾部约束，首5步都差于冻结Kalman | 不替换现有Kalman；V14仍明显更接近Full |
| E4 跨环境 | 108条有效结果；CALVIN Full0/8、V14为2/8、Object draft为1/8；Simpler／PushT各条件0/5 | 当前跨域部署整体很弱；不能据此归因模型容量，且已有draft跨域成功 |
| E4补测：LIBERO-Plus | 585条件配齐；扰动175例Full78.9%、V14 50.9%、纯draft30.9%；原始20例另为100%、80%、40% | V14总体优于纯draft，仍明显低于Full；分组与轨迹见第4.5—4.13节，其他跨环境补测未完成 |

新增请求最终更新：V14-D2的195/195例已于北京时间2026-10-23 18:39完成，Plus扰动175例为85/175（48.6%）、原始20例为16/20（80%）；旧585参考不重跑，四条件总览、分suite、扰动及难度表已补齐。三个启动的保存数组已重新核对，全部195例执行日志通过独立核验。四draft的52回合最小诊断及语言、姿态、噪声、背景分析也已完成，详见第4.14—4.15节。本方模型与调度服务均已退出，GPU1仍留给用户的CUDA双流会话；CALVIN等原E4暂缓分支未恢复。此前case0偏差通过恢复历史初始化顺序消除，新服务均通过固定16参考的raw／decoded逐位一致及D2原生门禁，全部新增实验无视频。[最终汇总](../../../reports/idea1_libero_plus_d2_final_20261023T1200Z/summary.json)。

## 1. E1：batch 能否提高评测吞吐

### 1.1 已完成的首个队列

π0 Full、D10/H50/R5、真实100观测、同一样本固定噪声；同一物理GPU3。

| 配置 | bank吞吐，例/秒 | 相对B1 | 数值门禁 |
| --- | ---: | ---: | --- |
| B1 | 12.28 | 1.00× | 对原生B1检查max_abs=0 |
| B2 | 15.30 | 1.25× | 未通过 |
| B4 | 18.56 | 1.51× | 未通过 |

数字是固定bank整段完成吞吐，不是机械臂任务速度。batch增加有吞吐潜力，但输出不严格等价。100点中B4重复输出、倒序后还原和逐样本噪声身份均匹配；这支持结果可重复且未发现样本顺序混淆，不证明已经定位到具体kernel。

多数输出差较小，但发现首5步夹爪输出从B1的约−1.002变为B4的+0.913，不能仅凭均值小就认定无害。原始完整50步最大差约1.93。具体误差分位数、向量通道和输入身份保留在 `e1_20261022T0421_batch/batch_diagnostics.json`。

在明确标记“近似batch”的情况下，做了同服务、同4worker、同8官方场景的小规模闭环：

| 调度 | 成功 | 完成8场景墙钟 | episode吞吐比 |
| --- | ---: | ---: | ---: |
| B1后置对照 | 8/8 | 38.682秒 | 1.000× |
| 最大B4、最多等待40ms | 8/8 | 43.507秒 | 0.889× |

最大B4实际只形成1/2/3大小的批次（114/40/15批），没有大小4。当前配置不采用：既未通过数值门禁，也没有提高实际评测吞吐。**这不是“所有batch都无效”的结论**，随后已在原60分钟预算内补测wait0／wait5。

初始B1还有8条有效成功轨迹，但调度暂停时没有保存完整arm墙钟，不用它计算速度。短闭环都是task0/init0–1，不能当作四套件标准成功率。

原始数据：[首个cohort](../../../results/idea1_v3_2/e1_20261022T0421_batch/rescue.json)；[短等待补测分析](../../../results/idea1_v3_2/e1_wait_scan_20261022/analysis.json)。

### 1.2 短等待补测：仍无可观的端到端收益

以下是**另一独立、同服务cohort**的比较，不将两次B1耗时混为同一对照。

| 配置（同为4环境worker） | 成功 | 8场景墙钟 | episode吞吐比 | 动作/秒 | 请求/秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1／wait0 | 8/8 | 38.293秒 | 1.000× | 31.442 | 6.346 |
| 最大B2／wait0 | 8/8 | 37.682秒 | 1.016× | 31.606 | 6.369 |
| 最大B4／wait5ms | 8/8 | 39.001秒 | 0.982× | 30.717 | 6.205 |

取消等待确实比40ms配置更合理，但目前最优只表现为episode吞吐+1.6%，按动作／请求计分别仅+0.52%／+0.37%，不足以确认实用收益。没有重复cohort或随机交叉顺序，不能把这点差当成稳定加速。两种batch的实际大小仍主要为1，少数为2，没有3/4。

任务成功所需动作数也不同，因此episode吞吐不是完全固定工作量的纯硬件加速；bank固定输入吞吐是另一口径。小样本均成功不代表输出等价，batch产生的闭环后续输入漂移已记录。结论：本轮不切换默认batch评测配置。更多同时就绪环境、更大worker数未测，不能排除其他部署可能有效。

![固定输入bank和真实闭环吞吐是不同口径，batch仍未通过数值门禁](../../../results/idea1_v3_2/e1_batch_evidence.png)

[E1详细报告与48条episode账目](../../../results/idea1_v3_2/E1_BATCH_REPORT_ZH.md)。三种batch调度均不默认采用；本方E1进程已退出。

## 2. E2：十次更新、两次完整Transformer

真实checkpoint门禁、同卡计时、四套件闭环和D2补充对照均已完成。

测试模型是π0 **完整模型**（非V14），vision27层、VLM18层、AE18层，H50/R5。近似严格限定为单次请求内：十次Euler更新仅0、3更新完整18层AE Transformer；这里0/3是0-based采样更新编号，不是模型层号。其余重新构造当前状态／动作／时间embedding，加上缓存的pre-final-norm residual，再计算当前norm和动作头。VLM每次请求刷新，无跨帧KV缓存、无速度场整体复用。

先核对每步刷新版本与原生D10、D2，再测0/3。门禁通过后做四套件task0–9/init0共40条；旧Full40逐条核对实际协议和场景身份后复用。残差复用达到预先约定的有希望质量门槛后，另跑同40场景的原生D2；没有追加超参数搜索。

24个真实输入上，每步刷新版本的raw／解码动作与canonical原生D10完全一致；D2包装与原生D2亦通过。实际计数为10次Euler更新、2次Transformer刷新；跨请求缓存隔离最大差0。

| 同卡预热后p50 | 原生D10 | 残差0/3，仍10次Euler | 原生D2 |
| --- | ---: | ---: | ---: |
| 完整本地请求 | 99.09ms | 83.12ms | 82.37ms |
| AE单独计算 | 21.99ms（等价每步刷新） | 5.63ms | 5.04ms |

残差复用相对D10的AE约3.91×、完整请求约1.19×；**没有比直接D2更快**。十次更新不等于十次昂贵Transformer，两次完整刷新才是主要计算量。是否值得保留，取决于任务质量能否胜过同成本D2，不能仅因AE加速就下结论。计时含同步、变换与解码，排除首次编译；只有24点、3次预热，不冒称长期服务精确分位数。

![AE单独耗时和完整本地请求，不能将前者加速当作后者加速](../../../results/idea1_v3_2/e2_figures/latency.png)

### 2.1 成功率：总体同分，但不是同一条路线

| 配置 | Spatial | Object | Goal | LIBERO-10 | 总体 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full，原生D10 | 9/10 | 10/10 | 9/10 | 6/10 | **34/40，85%** |
| 残差0/3，十次Euler | 9/10 | 9/10 | 10/10 | 6/10 | **34/40，85%** |
| Full，原生均匀D2 | 9/10 | 10/10 | 10/10 | 6/10 | **35/40，87.5%** |

Full与残差：31条双方成功、3条仅Full成功、3条仅残差成功、3条双方失败。残差与D2：33条双方成功、1条仅残差成功、2条仅D2成功、4条双方失败。因此85%相同不意味着样本或动作等价。

每任务只有init0，是快速筛查而非每任务50初态的完整基准；一个suite中1条变化就是10个百分点。D2仅多成功1条，不能据此证明统计优势；但本轮**没有观察到残差复用相较简单D2的质量／成本优势**。也未测其他刷新位置或非均匀D2最优日程。

![四套件及总体成功率，每个suite分母10](../../../results/idea1_v3_2/e2_figures/quality.png)

### 2.2 真实末端轨迹、阶段与视频

四个固定展示任务均在结果揭晓前选定为各suite的task0/init0，不按好坏挑图。蓝色Full、橙色半透明虚线＋点为残差；XYZ来自每次执行后仿真器的真实EEF位置，不是截图，也不是控制命令代替实际位置。

![Object固定样例的真实EEF XYZ，纵轴为米](../../../results/idea1_v3_2/e2_figures/eef_libero_object.png)

Object这条轨迹后半段明显分开，Full成功、残差失败；但单条图本身不足以认定是夹爪或某个模型层导致。其他固定轨迹：[Spatial](../../../results/idea1_v3_2/e2_figures/eef_libero_spatial.png)／[Goal](../../../results/idea1_v3_2/e2_figures/eef_libero_goal.png)／[LIBERO-10](../../../results/idea1_v3_2/e2_figures/eef_libero_10.png)。

残差与Full的**每条episode开头执行后第1–5步**平均末端距离约2.11mm，整条轨迹共同step范围的episode平均约47.82mm；D2分别约1.84mm／40.97mm。前一个量不是沿任务所有重规划的未来1–5步误差，不能据此证明接触阶段安全；后一个量包含路线、进度和累计偏差，不是同观测动作误差。这里也不是“只执行Full”的shadow评测。

残差6条失败的最终启发式阶段为目标对齐3、搬运2、接近1；其中Full原本成功、残差失败的3条最终为目标对齐2、搬运1。该3条的最远进度都曾到达release/final，但不能把进度标签直接当作首次错误或失败原因。

完整视频及实际XYZ的交互入口：[E2图表／视频报告](../../../results/idea1_v3_2/e2_report_zh.html)，含四条旧Full及八条新residual／D2视频；新视频均经完整解码检查，原始索引保留hash。可直接打开HTML浏览，不把无跳转的文字路径当成视频。原始统计：[E2汇总JSON](../../../results/idea1_v3_2/e2_summary.json)／[原生数值门禁和计时](../../../results/idea1_v3_2/e2_probe_gpu2.json)。

## 3. E3：仅凭历史命令能否更好地预测

### 3.1 结论：这三类方法没有替代冻结Kalman

| 方法 | 首1–5步误差 | 第6–12步 | 第13–50步 |
| --- | ---: | ---: | ---: |
| 加权局部多项式WLP | 9.276 | 21.544 | 44.701 |
| 恒转向ConstantTurn | 9.732 | 22.846 | 42.674 |
| 短阶VAR | 9.831 | 22.645 | 40.323 |
| v2.1冻结Kalman | **8.478** | 20.630 | 54.136 |
| 旧Kalman | 9.030 | 23.078 | 61.154 |
| 保持最后命令HoldLast | 10.576 | 25.134 | 43.342 |
| 同观测V14 | **7.446** | **7.264** | **7.903** |

单位均为**平移控制目标增量差代理mm**：归一化命令×50mm，不是机械臂实际末端误差。这是离线预测，不是新方法闭环执行后的任务成功率。

Kalman维护的是6维`[u_x,u_y,u_z,w_x,w_y,w_z]`：前三维是平移控制命令，后三维是这些命令的变化趋势；不是`XYZ＋三个姿态角`，也不是实测末端位置＋速度。旧版趋势按单位时间变化率定义，v2.1按每步变化量定义。旋转和夹爪未纳入该滤波器，仅以已经执行的命令更新，因此本节不能回答完整六自由度位姿或夹爪预测的效果。

复用历史 **D3/H50/R5** Full真实执行轨迹及同观测V14（跳`[3,4,9,11]`）输出，与E2的新D10闭环不是同一组实验。20条task0轨迹／634重规划点校准，180条task1–9轨迹／6345点评估，每任务5个官方init。51候选全部未通过预设校准尾部约束；每类最佳候选仅保留为探索结果。没有看测试集再加参数。

新方法相对冻结Kalman的首5步误差分别多0.798／1.254／1.353mm，按任务簇配对的区间分别为[0.711,0.879]／[1.129,1.375]／[1.191,1.497]mm。不能据此排除所有其他参数或其他轨迹模型，但这轮已无理由替换现有短前缀方案。

部分新方法长段误差小于Kalman，说明阻尼限制了长时外推发散；**不等于实际要用的前5步更好**。第6–50步标签是当前Full预测chunk，不是未来重规划后的真实执行轨迹。

### 3.2 不是单纯“时间早了或晚了”

同一单步命令误差向量分解后，冻结Kalman首5步的平均绝对沿轨分量约4.73mm、横轨分量约5.81mm；WLP约5.29／6.37mm；V14约4.28／5.29mm。横向分量不能用统一提前／滞后来全部解释。分量的均值不应直接代入勾股公式。

同时保留累计控制目标代理的独立分解，两类向量不混用；逐样本勾股恒等式通过。阶段标签只用来事后分组，未作因果判断。

![单步命令与累计目标代理的方向误差，注意两排单位口径不同](../../../results/idea1_v3_2/e3/revision2/direction_curves.png)

读图：上排是逐个未来offset的命令差，下排是假定不断累加这些命令后的代理差；下排不是机器人实际移动了几米。虚线为第5步，右侧长时误差只作诊断。粉色V14基本平稳，而历史外推方法随offset增加明显偏离。

### 3.3 可复查证据

[E3完整中文报告](../../../results/idea1_v3_2/e3/revision2/report_zh.md)包含全部50 offset、分阶段／suite、bootstrap、创新和事后时间偏移诊断；原始预测和冻结参数在其父目录。404个实际消费原日志重新校验；四套件各一条完整轨迹、118点×50步的旧／冻结Kalman和HoldLast独立重放最大差0。

主计算98.5秒，CPU即可，不新跑机械臂或调用模型。最终20项组合CPU测试（E1/E2/E3）已通过；这不能替代E2真实模型门禁。

## 4. E4：纯draft与V14的跨环境表现

### 4.1 实际测了什么，如何保证不是接口报错冒充失败

按[冻结零样本协议](idea1_v3_2_e4_zero_shot_protocol.md)，比较Full、V14和全部四个发布learned head，共18案例×6条件＝108条。没有按目标环境成绩挑选head，也没有训练／微调权重；**做了显式观测与控制接口桥接，不是“零接口适配”**。现有draft head约110.3M参数，不是40M，也不是整个视觉＋draft流水线的总参数量。

Full／V14保持完整视觉编码器和18层AE、均匀D10、H50/R5；V14仅跳VLM的`[3,4,9,11]` hidden更新。纯draft每请求实际调用vision1、head1，VLM／AE／verify／隐藏Full调用均为0；确定性head没有去噪步数，不能给它套用D10。

所有输入来自原生真实任务，而非随机初始化数据。新指令使用完整词表及原生SentencePiece；10个已有LIBERO指令的embedding逐位重现。两次服务启动均先用真实LIBERO输入核对六条件旧／新路径：raw逐位一致，decoded最大差≤`2.3841858e-7`。18案例的初始RGB／state8在六条件间完全一致；模型输出、噪声身份及实际路由逐请求保存。[冻结清单及身份](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/manifest.json)。

CALVIN使用debug validation全部8个语言段起点、原生task checker、360步cap；Simpler使用`google_robot_pick_coke_can`、seed0–4、原生成功判据与80步上限；PushT使用seed0–4和原生覆盖阈值、300步cap。CALVIN／Simpler保存原生实际EEF XYZ；PushT只有真实二维XY像素，不能补造Z或假称机械臂任务。PushT的512px↔0.6m人工投影、黑色缺失腕相机、跨机器人状态／坐标／夹爪映射均已预先声明。

### 4.2 成功率：不支持“draft只会LIBERO”的结论

| 条件 | CALVIN，8案例 | Simpler，5 seed | PushT二维压力测试，5 seed |
| --- | ---: | ---: | ---: |
| Full，D10 | 0/8，0% | 0/5，0% | 0/5，0% |
| V14，D10 | **2/8，25%** | 0/5，0% | 0/5，0% |
| draft_goal | 0/8，0% | 0/5，0% | 0/5，0% |
| draft_object | **1/8，12.5%** | 0/5，0% | 0/5，0% |
| draft_spatial | 0/8，0% | 0/5，0% | 0/5，0% |
| draft_10 | 0/8，0% | 0/5，0% | 0/5，0% |

![跨环境各条件成功率，分母独立，不混算总分](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/report_20261022T061020Z/success.png)

V14成功的是CALVIN case1开灯（230步）和case6向右推粉色方块（38步）；Object draft也完成了case6（29步）。这些是原生判据的实际成功，但不证明稳定的语言理解／迁移能力。V14比Object draft只多一个案例，样本极小，不能证明稳定优势；8个debug语言段也不是CALVIN正式长任务序列基准。

**最重要的限制是Full正对照同样全部失败。** 这里测的是LIBERO权重经固定桥接后的零样本部署，与第2章LIBERO上的85%不是同一组评测。训练分布、相机、机器人/TCP、状态规范和控制频率同时变化：源20Hz，CALVIN30Hz、Simpler3Hz、PushT10Hz。因此只能说本冻结跨域部署整体很弱，不能把失败归因于draft容量不足，也不能把V14的两次成功解释为剪枝提升了泛化。已有Object draft成功，更不能写“换环境必定失败”。

### 4.3 实际轨迹与视频

[预先固定case0的三环境真实XYZ／XY对照图](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/report_20261022T061020Z/trajectories.png)使用蓝色Full、橙色半透明虚线＋点V14，其他head保留独立线型。它们是实际仿真器位置，不是相机截图或动作累加代理。

[打开18个固定视频的索引与播放器](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/report_20261022T061020Z/media_index.html)。每环境case0的六条件全部保存，不按成败挑选；18条均由主代理独立完整解码，H.264/yuv420p、帧数＝执行步数＋初始帧、原生帧率均通过。CALVIN/Simpler/PushT分别30/3/10fps。

下图额外选择有成功对照的CALVIN case6，是**明确的事后选取**，不是无偏代表。上排完整真实轨迹，下排为前50个原生控制步；成功提前终止的轨迹不补造后续点。该事后样例没有补录视频，固定视频仍仅case0；没有为画图重跑模型或环境。

![事后选择的case6，V14与Object draft成功、Full失败的实际XYZ投影](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/outcomes_audit_20261022T061020Z/posthoc_calvin_case6_xyz.png)

[逐案例终止、位置范围和分通道裁剪CSV](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/outcomes_audit_20261022T061020Z/episode_outcomes.csv)保留所有条件。平移、旋转和夹爪裁剪分开；夹爪在±1附近的小幅裁剪不能当成失败原因。请求延迟仅限本实验同卡Triton实现，不能与E2的JAX延迟横比。

### 4.4 技术事件没有抹去，也没有变成策略0分

首轮PushT有30次第0步`diffusion_policy`导入失败，没有策略质量调用；仅补全父进程的本地`PYTHONPATH`后，以**同一不可变manifest**续跑。原有78条有效结果全部跳过，失败attempt000保留，PushT使用attempt001。最终为108条有效结果、138次总尝试，不是138条有效episode。

[全部尝试历史](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/outcomes_audit_20261022T061020Z/attempt_history.csv)和[完整审计](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/outcomes_audit_20261022T061020Z/audit.json)记录这一过程。自动主表的“技术失败0”只针对每键选定的最终有效结果，不表示历史从未报错。两次服务的preflight／quality分类计数分别保存并合并核对；未因续跑重做前78条质量评测。

E4从05:00 UTC准备至约06:10 UTC计算完成，约70分钟，在自定90分钟探索边界内；这不是将E1–E3的用户原始预算重新计时。[E4详细报告](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/report_20261022T061020Z/report_zh.md)／[逐案例审计说明](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/outcomes_audit_20261022T061020Z/report_zh.md)。

### 4.5. E4后续补测汇总（2026-10-23更新）

本节将后续补测的数据、分析与图表直接纳入v3.2报告正文。原始108条件与已完成Plus三条件分开记账，缺失条件不计作策略失败；新增D2／四draft无视频排查的当前进展见第4.14节。

已完成身份有效条件 585/585；三条件配齐 195/195 case。额外历史 30 条不计入。
这里只比较三条件共同完成的 case，扰动175与原始20分开，不能外推全部10,030任务。

|样本组|已配齐三条件case|Full-D10|V14-D10|对应纯draft|新增V14-D2|
|---|---:|---:|---:|---:|---:|
|扰动175|175|138/175 (78.9%)|89/175 (50.9%)|54/175 (30.9%)|85/175 (48.6%)|
|原始20|20|20/20 (100.0%)|16/20 (80.0%)|8/20 (40.0%)|16/20 (80.0%)|

本次Plus数据矩阵已完成，CALVIN／SimplerEnv／PushT等完整E4补测范围尚未完成，不能将本节数据视为整轮E4补测结束。主代理基于固定CSV／JSON补充以下分析，未重测、剔除或改写科学结果。

#### 结论速览

- Plus扰动子集上，Full为78.9%、V14为50.9%、纯draft为30.9%。V14比Full低28.0个百分点，比纯draft高20.0个百分点；这里的draft不包含FLASH验证和Full接管机制。
- V14在Object相对较好（61.4%），Goal最低（31.8%）。各类扰动中，布局下V14最接近Full（17/25对18/25）；相机视角下纯draft略高于V14（13/25对11/25）。这些是小样本描述，不是逐层因果结论。
- 原始20例对照中Full为20/20，本轮Plus并未复现Full挂零；但之前CALVIN等跨环境零分仍未完成排查。当前没有后台实验，未完成分支和视频缺口在下表单独列出。

成功率按母suite、扰动类别和难度见第4.7—4.8节；真实末端轨迹与差异口径见第4.10节；耗时见第4.11节；数据验收与完整E4剩余范围见第4.12节。本文数据截至北京时间2026-10-23 05:22:53的Plus完成状态，后续仅离线汇总。

### 4.6. 测了什么

使用已有LIBERO权重，不做Plus微调。Full与V14均生成H=50动作、每轮执行R=5、D=10；V14仅跳过VLM第3、4、9、11层（1-based），保留14个VLM block，不裁AE。匹配的FLASH纯draft按实例母suite选择四个发布head之一，不含verify、周期Full或夹爪接管，D=N/A，不是完整FLASH系统。

175个扰动实例覆盖7类×5难度、每格5例、全部40个母任务；每实例1次trial，三个条件严格配对。原始无扰动对照20例，每suite前5个母任务各1次trial，另表报告，不混成Plus榜单分数。选择在看结果前冻结；额外30条历史结果保留但不进入本矩阵，43条入选历史条件复用。

官方环境20Hz、5步settling、每回合最多600步，使用实际有效BDDL原文指令。技术失败不记作策略失败。本次585条均有效完成，技术异常0；不是10,030实例全量评测。

### 4.7. 分母suite结果

| Plus扰动任务：按来源suite分组 | Full-D10 | V14-D10 | 匹配纯draft | 新增V14-D2 |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 38/44（86.4%） | 23/44（52.3%） | 21/44（47.7%） | 25/44（56.8%） |
| Object | 34/44（77.3%） | 27/44（61.4%） | 11/44（25.0%） | 26/44（59.1%） |
| Goal | 35/44（79.5%） | 14/44（31.8%） | 14/44（31.8%） | 19/44（43.2%） |
| Long / LIBERO-10 | 31/43（72.1%） | 25/43（58.1%） | 8/43（18.6%） | 15/43（34.9%） |
| 合计 | 138/175（78.9%） | 89/175（50.9%） | 54/175（30.9%） | 85/175（48.6%） |

原始无扰动对照：

| 普通LIBERO无扰动小样本对照 | Full-D10 | V14-D10 | 匹配纯draft | 新增V14-D2 |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 5/5（100%） | 5/5（100%） | 4/5（80%） | 5/5（100%） |
| Object | 5/5（100%） | 5/5（100%） | 1/5（20%） | 5/5（100%） |
| Goal | 5/5（100%） | 3/5（60%） | 2/5（40%） | 3/5（60%） |
| Long / LIBERO-10 | 5/5（100%） | 3/5（60%） | 1/5（20%） | 3/5（60%） |
| 合计 | 20/20（100%） | 16/20（80%） | 8/20（40%） | 16/20（80%） |

Full在本次原始对照不低、在Plus也不挂零；但这不能证明之前跨环境零分的原因已经全部找到。两个表的母任务覆盖和初态不完全相同，不能把100%到78.9%直接当作同任务同初态的受控扰动降幅。原始20例也不是LIBERO标准全量成绩。

### 4.8. 扰动类型与难度

| 扰动类型（各25例） | Full-D10 | V14-D10 | 匹配纯draft | 新增V14-D2 |
| --- | ---: | ---: | ---: | ---: |
| 物体布局 | 18/25（72%） | 17/25（68%） | 11/25（44%） | 13/25（52%） |
| 相机视角 | 20/25（80%） | 11/25（44%） | 13/25（52%） | 11/25（44%） |
| 机器人初态 | 14/25（56%） | 9/25（36%） | 3/25（12%） | 7/25（28%） |
| 语言指令 | 19/25（76%） | 11/25（44%） | 4/25（16%） | 11/25（44%） |
| 光照 | 25/25（100%） | 18/25（72%） | 12/25（48%） | 21/25（84%） |
| 背景纹理 | 20/25（80%） | 11/25（44%） | 6/25（24%） | 12/25（48%） |
| 传感器噪声 | 22/25（88%） | 12/25（48%） | 5/25（20%） | 10/25（40%） |

| 官方难度标签（各35例） | Full-D10 | V14-D10 | 匹配纯draft | 新增V14-D2 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32/35（91.4%） | 24/35（68.6%） | 12/35（34.3%） | 19/35（54.3%） |
| 2 | 35/35（100%） | 23/35（65.7%） | 17/35（48.6%） | 22/35（62.9%） |
| 3 | 25/35（71.4%） | 13/35（37.1%） | 8/35（22.9%） | 14/35（40.0%） |
| 4 | 24/35（68.6%） | 17/35（48.6%） | 11/35（31.4%） | 17/35（48.6%） |
| 5 | 22/35（62.9%） | 12/35（34.3%） | 6/35（17.1%） | 13/35（37.1%） |

### 4.9. 主要发现及能说到什么程度

- 同一175例上，V14比Full低28.0个百分点，纯draft比Full低48.0个百分点；V14比draft高20.0个百分点。配对看，V14有54例“Full成功、V14失败”，也有5例反过来；draft对应87例和3例，不是每条任务都单向退化。
- V14在这批Plus中Object相对较好（61.4%），Goal最低（31.8%）；与纯draft在Goal打平，其余母suite领先。不能据此定位某一个layer的因果责任，本轮没有新增逐层消融。
- 布局扰动下V14与Full只差1例；相机视角下draft反而比V14多成功2例。说明“V14总体更好”不等于每类都更好，但单类25例且任务组合不同，不能把这一局部差异当成已证实的架构优势。
- 机器人初态在三个模型中均是最低类别，Full也只有56%；光照下Full为100%。这是本子集的描述，不证明初态扰动在所有任务上普遍最难。
- 后三档整体更难，但不是单调曲线，例如Full的难度2高于1，V14的4高于3。难度来自官方外部性能分类，并非同任务上单一物理扰动强度逐档增大；本轮不作这种因果解释。
- 纯draft在原始对照也只有40%，因此其Plus低分不能全部归因于新增扰动，更不能当作完整FLASH带verify系统的成功率。

### 4.10. XYZ轨迹：实际走到哪里，而非同观测动作误差

下面图使用仿真器真实末端世界坐标，不是相机截图。Full蓝色、V14橙色、draft绿色，透明点线／虚线区分，圆点为起点、X为终点。7个代表案例固定为各类别最小source_index，不按成功／失败或图形效果挑选。

数值指标先在每个案例双方都已记录的公共原生时间内计算 `d(t)=1000*||xyz_variant(t)-xyz_full(t)||₂`（mm），再取该案例的时间平均，最后跨案例取中位数；不将先结束的一条补成静止轨迹。

| 样本组 | 对比Full | 各案例平均路径距离的中位数 | 案例间25%—75%区间 |
| --- | --- | ---: | ---: |
| Plus175 | V14 | 108.7 mm | 67.1—188.8 mm |
| Plus175 | 匹配纯draft | 162.3 mm | 75.3—270.1 mm |
| 原始20 | V14 | 69.1 mm | 42.5—83.7 mm |
| 原始20 | 匹配纯draft | 106.2 mm | 52.7—257.8 mm |

这包含各自闭环执行的累计偏离、速度和任务阶段错位，不是同一个obs上预测动作的误差，不是前1—5步误差，也不能解释成机械臂每步定位误差10厘米。即使两者最终都成功，也可能时序不同而有较大同时间路径距离。仅凭这些图和夹爪信号，不能断言失败集中于接触／精细操作；当前telemetry没有物体接触GT，不补造分阶段因果结论。

### 4.11. 为什么单次推理较快，整条评测反而更慢

以下只描述本批不同闭环轨迹的已有计时，不是同输入独占profile或受控加速比：

| Plus175模型 | 平均执行步数／回合 | 平均回合wall | 逐请求客户端RPC均值 | wall中RPC区间外占比 |
| --- | ---: | ---: | ---: | ---: |
| Full-D10 | 256.3 | 44.7 s | 311.1 ms | 64.1% |
| V14-D10 | 392.7 | 64.3 s | 252.0 ms | 69.1% |
| 匹配纯draft | 461.4 | 72.9 s | 181.6 ms | 77.0% |

压缩条件的请求均值较低，但成功更少、更多回合跑满600步，实际执行和请求更多，因此每回合wall更长。RPC还包括客户端往返；区间外包括环境创建／重置、物理执行、渲染、图像扰动及日志等，不能全部归因于某个噪声算子。这里不能用不同轨迹的wall或请求均值之比宣称纯模型加速比。

### 4.12. 数据验收、视频与未完成范围

[主代理只读验收记录](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/root_data_acceptance.json)逐条核验585结果、42,305次请求、210,828步执行：原始／解码动作全部有限，所有请求种子符合冻结规则，R5实际执行动作逐项等于保存的预测，native裁剪与最终成功oracle一致，24个冻结父源码哈希未变。Full／V14所有请求均记录180次AE层调用（18层×D10）；四draft均为1次learned head、无VLM prefix／AE／verify调用。

195组的物理初态、指令和首请求种子一致；194组全部初始NPZ逐位一致。唯一图像差异仍是光照案例210：Full与另外两组相比腕部3像素／4通道、最大1级uint8。原因未证明、影响不宣称为零；保留全部结果，没有择优剔除或重测。2,925个保存初始数组哈希与首请求相符。

现有3个视频均为案例0的真实原始回合，H.264、256×256、20fps，ffmpeg完整解码通过，帧数分别93／98／93，恰为执行步数＋初始帧。其他6个代表案例×3组共18份视频未录制，下面索引如实标缺失；28张XYZ／XY-XZ轨迹图均已生成，但不能将“轨迹齐全”称为“对应视频全齐”。未额外重跑回合来冒充原始视频。

[接续及退出记录](../../../results/idea1_v3_2_e4_supplement/half_deadline_guard_v2_20261022T155319Z/execution_1792684603/execution.json)为 `complete_585_verified`，worker／模型退出码均0，本方相关进程已退出。仅同GPU3质量评测，没有第二卡质量上线；没有新的GPU profile、模型／仿真探针或CALVIN恢复。

#### 完整E4补测：哪些完成，哪些尚未完成

| 分支 | 已有数据／完成数量 | 当前状态与缺口 |
| --- | --- | --- |
| LIBERO-Plus半量 | 585/585条件；175扰动＋20原始对照，三模型全部配齐 | 质量数据已完成、已验收；28张轨迹图已齐，代表视频3份可用、18份未录制 |
| CALVIN ABC→D原生v3 | 目标100序列×Full／V14＝200条件；仅2/200完整条件，均来自seq0且完成子任务数为0 | 用户要求暂缓；seq1 Full另保存了3个已完成子任务／291步，但整链未完成，不计完整条件，不当作整链零分；没有仍在运行的进程 |
| CALVIN ABCD→D原生 | 固定100序列，尚未完成质量评测 | 缺少已验明训练来源的匹配π0权重；不能用ABC权重或π0.5替代 |
| 更正后跨环境B | CALVIN／Simpler／PushT各100案例×六条件＝1800条件；实际完成0/1800 | 仅准备和接口检查完成，未派发正式质量评测；0/1800是完成数量，不是成功率0% |
| 原生Simpler／PushT正向对照 | 尚无本轮正式质量结果登记 | PaliGemma tokenizer配置访问返回401／GatedRepoError，需合法兼容本地资源或正常HF授权；不以跨环境B或旧零分替代 |

以上分支不是可以合并分母的一组评测：Plus使用LIBERO权重；原生CALVIN使用目标域匹配权重；跨环境B是LIBERO源权重迁移。未完成／未运行条件不能算作策略失败。CALVIN仍按用户要求暂缓，本次整理没有恢复其余分支或启动新实验，旧跨环境零分也尚不能据此宣布全部解释或修复。

#### CALVIN已有非零子任务证据：不是整链成绩

对原生ABC→D v3的seq1 Full保存记录进行只读交叉核验：独立`task_0/1/2.json`与汇总记录相符，291步telemetry连续，以下三个子任务各自的末步环境oracle均为成功。

| 顺序 | 已完成子任务 | 本子任务执行步数 | 累计执行步数 |
| --- | --- | ---: | ---: |
| 1 | `turn_off_led`：关闭LED | 132 | 132 |
| 2 | `push_into_drawer`：将方块推入抽屉 | 111 | 243 |
| 3 | `lift_blue_block_drawer`：从抽屉拿起蓝色方块 | 43 | 286 |

第4项`place_in_slider`只保存了5步，此时用户要求暂缓；结果文件的整链`success=null`，并没有完成或失败结论。不能把中断时子任务的`success=false`当作最终超时失败，也不能把这三个成功子任务当作三条独立成功序列。此前seq0的Full／V14都在首项`rotate_blue_block_right`达到360步上限而失败，不能用这一条序列代表100条总体表现。

这项证据仅说明匹配目标域权重、修正接口后的Full已经能够在CALVIN连续完成部分任务，不支持“完全无法工作”；它尚不足以给出总体成功率或证明旧LIBERO权重跨域零分的原因已消除。来源：[seq1原始结果](../../../results/idea1_v3_2_e4_supplement/calvin_native100_v3_20261022T1210Z/execution/episodes/seq0001/full_d10/attempt000/result.json)／[逐步telemetry](../../../results/idea1_v3_2_e4_supplement/calvin_native100_v3_20261022T1210Z/execution/episodes/seq0001/full_d10/attempt000/telemetry.jsonl)。本项核验未运行模型或仿真，也未改写旧记录。

剩余范围的证据与接续边界见[补测状态](idea1_v3_2_e4_supplement_status.md)，包括暂缓收据、B冻结登记和依赖访问记录。本报告第1—4.4节保留原始实验范围与历史口径，本节补充后续Plus数据，不覆盖旧结果。

#### 固定数据与图表入口

- [最终统计JSON](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/summary.json)、[成功率与配对差CSV](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/success_paired.csv)。
- [585条件固定账本](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/audit_ledger.json)、[逐条数据验收记录](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/root_data_acceptance.json)。
- [视频可用／缺失索引CSV](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/videos.csv)；下面按7个预定案例展示全部28张轨迹图和实际可用视频。

### 4.13. 可视化与视频索引

轨迹采用预先固定的各category最小source_index：0/50/100/150/200/250/300；缺少配齐结果时不挑替代样本。闭环路径距离只在双方已有的公共时间内计算，不是同观测预测误差；没有接触GT，不据夹爪推断接触阶段。

![三条件共同样本成功率](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/success_paired.png)

#### Objects Layout · 固定 source_index=0

![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_000_v14_d10.png)
![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_000_v14_d10.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_000_matching_draft.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_000_matching_draft.png)

#### Camera Viewpoints · 固定 source_index=50

![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_050_v14_d10.png)
![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_050_v14_d10.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_050_matching_draft.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_050_matching_draft.png)

#### Robot Initial States · 固定 source_index=100

![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_100_v14_d10.png)
![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_100_v14_d10.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_100_matching_draft.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_100_matching_draft.png)

#### Language Instructions · 固定 source_index=150

![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_150_v14_d10.png)
![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_150_v14_d10.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_150_matching_draft.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_150_matching_draft.png)

#### Light Conditions · 固定 source_index=200

![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_200_v14_d10.png)
![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_200_v14_d10.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_200_matching_draft.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_200_matching_draft.png)

#### Background Textures · 固定 source_index=250

![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_250_v14_d10.png)
![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_250_v14_d10.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_250_matching_draft.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_250_matching_draft.png)

#### Sensor Noise · 固定 source_index=300

![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_300_v14_d10.png)
![Full vs V14](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_300_v14_d10.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/trajectory_300_matching_draft.png)
![Full vs matching draft](../../../reports/idea1_libero_plus_half_final_20261022T2124Z/spatial_300_matching_draft.png)

真实现有视频（缺失不补造）：

- 0 / full_d10: [MP4](../../../results/idea1_v3_2_e4_supplement/libero_plus_20261022T074328Z/perturbed350/case0000/full_d10/attempt000/video.mp4)
- 0 / v14_d10: [MP4](../../../results/idea1_v3_2_e4_supplement/libero_plus_20261022T074328Z/perturbed350/case0000/v14_d10/attempt000/video.mp4)
- 0 / draft_spatial: [MP4](../../../results/idea1_v3_2_e4_supplement/libero_plus_20261022T074328Z/perturbed350/case0000/draft_spatial/attempt000/video.mp4)
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

既有补测[plan](idea1_v3_2_e4_supplement_plan.md) / [status](idea1_v3_2_e4_supplement_status.md)。本报告只读离线汇总；当前队列状态见[half live ledger](../../../results/idea1_v3_2_e4_supplement/libero_plus_half_20261022T1216Z/live_ledger.json)。

### 4.14. 四draft失败原因排查与D2执行历史（已完成）

最终状态：V14-D2的195/195例已于北京时间2026-10-23 18:39完成，四draft最小诊断也已完成；新增结果和验收见第4.15节。以下资源等待、双卡启动、单卡接续段落保留为历史过程，不代表当前仍在运行。

最新资源调整（08:24 UTC）：按用户要求GPU1已释放给另一个CUDA双流会话，本轮只继续使用GPU0，不再抢占GPU1。GPU1已完成39例，58个未完成key将以独立GPU0接续补齐；case154因资源切换在250步中断、success=null，保留为技术中断而非策略失败。GPU0原98例队列仍正常推进，完整目标仍195，已完成数据不重跑。[资源交接与后续边界](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_quality_gpu01_20261023_run2/runtime_gpu1/user_card_release.json)。

当前正式评测状态（2026-10-23 07:56 UTC快照）：已不再等待GPU3。GPU0／1各自通过历史六路线初始化、16条真实参考各两次全raw／decoded逐位一致门禁及D2原生／封装逐位检查；新run2已正式双卡运行，195例按全局ordinal%2固定分为98／97例，互不重叠、不做batch。首两例99步／184步成功，root全流验收及两卡D2字节对照通过后已发布精确marker，继续剩余193例。快照已完成5例、另2例运行中，不把这个小样本当最终成功率。[当前manifest](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_quality_gpu01_20261023_run2/manifest.json)身份`c823d3ccea915a4cca78a6282b479f290f1cfff7a0c58052d14081a7efad3290`／[首例全流验收](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_quality_gpu01_20261023_run2/root_first_cases_audit.json)。

旧GPU3空等队列已在核实仅本方PID、无服务／回合子进程后安全停止，原CPU检查器自然退出；没有动外部作业。质量run1两卡严格门禁已通过，但actor解释器缺顶层依赖而在0质量步退出；仅固定新增sentencepiece0.2.0／safetensors0.5.3，原生MuJoCo／Torch／NumPy／robosuite未升级，Plus及original实际解释器dryrun和隔离原生导入均通过后以同一冻结v3源注册run2。技术日志保留，不记策略失败。[环境与队列交接](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_quality_gpu01_20261023_run2/environment_and_queue_handoff.json)。以下“只等GPU3／仍0条件”的记录为先前历史，不再代表当前运行状态。

最新用户覆盖：已授权在其他真正空闲GPU做严格逐位等价诊断，不再将“拒绝跨卡数值差异”误解为“必须等GPU3”。初次检查GPU0／1空闲；固定16条真实Full／V14参考，先复现生产路径、重复和保存有效中间数组，逐位一致才允许新卡D2质量评测。当前新增工作是诊断准备，未产生D2质量成绩；下文GPU3资源阻塞为该新指令之前的历史状态。细则见[当前设备与数值边界](idea1_v3_2_plus_v14_d2_supplement.md)。

其他卡实测更新（07:12 UTC）：GPU0与GPU1各完成16条参考、每条重复；两卡当前raw／decoded16/16逐位相同，两条捕获参考的归一化输入、vision、H0、KV、10次Euler latent也全部跨卡逐位相同。同输入重复、捕获／未捕获、同raw32输入D2 wrapper／原生路径均逐位一致。与旧GPU3记录相比，14/16条完整raw／decoded逐位相同；仅扰动case0的Full／V14两条不符（raw最大差0.03125／0.02001953125，decoded最大差0.01026654／0.00963295）。这不支持“换卡普遍不一致”的解释，但两条例外仍未过原门槛，不据14条成功删除异常或放行质量评测。两probe进程已自然退出，GPU0／1回到1MiB。[root数组对照](../../../results/idea1_v3_2_e4_supplement/d2_cross_device_bitwise_20261023_run1/root_cross_card_audit.json)。

CPU追溯发现：历史case0的Full是恢复服务第一个quality请求，之前已执行六路线legacy preflight；V14之前有19个Full请求。新probe从case0直接开始，初始化调用史不同；旧half只是原文件复用，没有找到独立的同输入历史Full／V14参考。当前仅能提出初始化分支假设，不能认定GPU或某个layer为根因。下一最小对照固定GPU0先执行同bank、同顺序legacy preflight，再case0 Full／V14各重复两次；不模拟缺失的19条演化图像，不将这个V14对照称为完整历史复现。[历史来源审计](../../../results/idea1_v3_2_e4_supplement/d2_cross_device_bitwise_20261023_run1/case0_provenance_cpu.json)。

用户最新要求：不接受跨卡数值差异；V14-D2在原GPU3继续，四个draft分别排查初态、指令、噪声和背景，全部新增实验不录视频、只保留必要数据。本节与上面585已完成结果分开，不能把准备工作或未派发条件算作成功率。

#### 已查明：无扰动对照也存在协议变化，不能把历史降幅全归因Plus

| 因素 | 历史纯draft 200例 | 本次无扰动20例／Plus175例 |
| --- | --- | --- |
| 初始状态 | 每任务init6—10 | 固定init0 |
| 环境seed | 7 | 每case派生seed |
| settling | 10步，夹爪命令−1 | 5步，全零动作 |
| 语言 | benchmark任务定义，来自任务名 | 实际BDDL原文 |
| 原版任务覆盖 | 每suite全部10任务 | 无扰动每suite前5任务，Plus覆盖40母任务 |
| 回合上限 | Spatial220／Object280／Goal300／Long520 | 统一600 |
| 预测／执行 | H50／R5 | H50／R5，不变 |

独立读取全部195个Full结果的`native_info`，20个无扰动对照中14个、175个Plus实例中150个的实际BDDL指令与benchmark定义不同。因此，即使没有Plus语言扰动，draft输入的文字也可能已经偏离历史训练／评测常用表达。例如原版Spatial task0从“pick up the black bowl …”变为“pick the akita black bowl …”，Goal task0从“open the middle drawer of the cabinet”变为“open the middle layer of the drawer”。这些表达语义可能相近，但不能在没有同观测干预前认定模型输出等价，更不能直接认定语言变化已导致任务失败。

将历史200也限定到新无扰动对照相同的前5个母任务，仍有明显差异，说明不只是任务覆盖变少；但初态、seed、settling和文字仍同时不同，尚未分离原因：

| head | 历史全部10任务 | 历史同前5母任务、init6—10 | 新无扰动同5母任务、init0 |
| --- | ---: | ---: | ---: |
| Spatial draft | 46/50（92%） | 24/25（96%） | 4/5（80%） |
| Object draft | 40/50（80%） | 19/25（76%） | 1/5（20%） |
| Goal draft | 29/50（58%） | 16/25（64%） | 2/5（40%） |
| Long draft | 22/50（44%） | 9/25（36%） | 1/5（20%） |

#### 四个head在各类Plus扰动中的实际表现

每格为该head自身的成功数／案例数。不同head对应不同suite任务，并非同任务换head的公平消融；每格仅6—7例，不把百分比波动当成稳定排序。

| 扰动类别 | Spatial draft | Object draft | Goal draft | Long draft |
| --- | ---: | ---: | ---: | ---: |
| 机器人初态 | 1/6 | 1/6 | 1/7 | 0/6 |
| 语言指令 | 2/6 | 1/6 | 0/6 | 1/7 |
| 传感器噪声 | 3/6 | 0/6 | 0/7 | 2/6 |
| 背景纹理 | 3/6 | 0/7 | 3/6 | 0/6 |
| 物体布局 | 5/7 | 3/6 | 2/6 | 1/6 |
| 光照 | 4/7 | 3/6 | 3/6 | 2/6 |
| 相机视角 | 3/6 | 3/7 | 5/6 | 2/6 |

可以定位的线索：初态类别四head都低；语言下Goal为0/6；噪声下Object、Goal都为0；背景下Object、Long为0。但Goal在相机视角下反而5/6，不能说四head对所有扰动统一失效。布局与视角等类别任务组合也不同，目前这些是分组关联，不是已证明的致败机制。

源码还确认：官方噪声包装只处理`agentview_image`，不处理腕相机。因此噪声配对可以在同一物理状态保留相同腕相机，仅比较干净／扰动外部视角；这比重新换母任务与整套初态更容易隔离视觉噪声效应。对这批25个机器人初态案例逐类审计，实际Panda变体全部仅覆盖`init_qpos`，未发现变体类修改基座／相机；这是源码层结论，不等于重置后两套仿真状态自动完全相同。[实际机器人类审计](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/robot_class_audit.json)。

CPU证据：[协议审计](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/protocol_audit.json)／[历史同母任务比较](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/historical_comparison.json)／[四head×类别×难度原始计数](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/head_category_difficulty.csv)／[逐case审计](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/case_audit.csv)。

#### 正在执行／准备的最小补测

V14-D2：195例无视频清单已冻结。旧watcher PID3712613在05:41 UTC检查时已消失，没有派发质量样本，旧状态文件保留但不再当存活证据；随后以独立tmux会话恢复原执行器，不改冻结源码或门槛。[当前GPU3专属watcher](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery/watcher.json) PID3752472，新清单身份`f35a09c66577443fcc19d8c7fea0127003ce405267afcd74898dd3c0ad3273c4`，仍为相同195例、原GPU3、每30秒等待空闲，不抢占外部作业。开跑必须通过同卡旧D10严格对照及D2原生parity，不接受此前GPU0数值差异。目前处于等待设备阶段，尚无新D2质量成绩。[中断与恢复记录](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/runtime_recovery_0543.json)。

为避免空卡后再等待人工放行，06:19 UTC上线独立CPU首例检查器（PID3775288），不改原模型／runner／清单。root通读代码并独立通过5项CPU测试后，预批准它复核原严格数值数组、服务身份、首例全部请求／种子／噪声、实际AE36／NFE2路由、R5执行与裁剪、初态、XYZ及oracle；全部通过才原子发布原watcher要求的精确`continue.json`。600步的合法策略失败与成功同样放行，不按成败挑样本；技术异常不发续跑文件、不改结果。不额外扫描GPU或调用推理。[审核收据](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery/first_case_auto_review.json)／[root预批准与源码身份](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery/automatic_review_authorization.json)。

四条件只读汇总器也已独立通过7项CPU测试并核对585旧条件的身份／SHA；[当前待测汇总](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery/analysis_pending_root_20261023/summary.json)明确D2完成0、待测195、成功率null。固定585参考表与D2已完成案例交集的配对表分开，避免部分D2与全量基线冒充同分母比较；这只是分析准备，不是新增质量结果。

06:33 UTC资源阻塞核验：至少连续3个goal轮次确认原GPU3仍被非本任务分配占用约16GiB，D2服务未加载、质量0/195；所有可独立完成的四draft诊断与分析准备已结束，继续模型评测需要原GPU3释放。已向用户请求协调，不停止未知作业、不换卡或放宽门槛。后台watcher及自动首例检查器均实际存活，仍按既定时限等待并可在设备释放后自动接续；目标未完成与后台等待不是同一件事。[资源阻塞核验](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery/blocked_resource_audit_20261023T0633Z.json)。

06:10 UTC额外CPU排查：两条旧D10启动参考的5组原观测、指令输入与噪声哈希均匹配实际首请求；native入口FP64状态与raw入口FP32状态经归一化最大差约1.27e-7，但进入BF16的状态相同，实际CPU输入准备器的图像／state／noise也均逐位相同。因此当前不能用输入或这项状态精度差解释旧raw偏差，更不能直接定性为“不同GPU舍入”。这没有执行GPU快速输入核或VLM／AE，且旧中间K/V与包版本未保存，仍需真实同卡验证。[CPU输入／精度审计](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/d2_cpu_input_precision_audit.json)。

draft：40回合原版协议小对照已完成（每head固定前2任务，init0/6×旧/新settling共32，再加8个实际BDDL prompt臂）；4回合启动动作拆分、24次语言探针、8回合闭环恢复及12个输入对照也已完成，最后一个背景对照为明确标注的辅助近似臂。原GPU0服务PID3715967对旧GPU3四head真实输入通过raw与decoded全部逐位一致门禁，8组最大差均为0，见[原启动门禁](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/draft_service_gpu0.json)。中断后仅为最后4请求恢复的新服务PID3752040也再次通过同一逐位门禁，现已正常RPC关闭、PID消失，GPU0回到1MiB，见[恢复服务及收尾](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/draft_service_gpu0_recovery_0543.json)。没有接受跨卡数值差异；新配对自身始终同卡。

40/40协议对照现已完成，[配对验收](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/original_protocol40/analysis.json)无物理初态／同模型身份配对失败。首次实际执行已独立核验预测前5步与执行动作一致、实际vision1／head1、AE／VLM／verify／Full均0、无视频；[事前40条件清单](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/original_protocol40/protocol_plan.json)没有按结果修改。以下每格只有各head原先固定的2个母任务，不是新suite成功率：

| head | init0／5步零动作／旧指令 | init0／10步开夹爪／旧指令 | init6／5步零动作／旧指令 | init6／10步开夹爪／旧指令 | init0／5步零动作／实际BDDL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| Object | 1/2 | 2/2 | 1/2 | 2/2 | 1/2 |
| Goal | 2/2 | 2/2 | 1/2 | 1/2 | 2/2 |
| Long | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |

在这8个固定任务里：Object体现settling整组配置的影响，Goal体现官方init向量的影响，Spatial和Long的成功数未随这些配置改变。init0／6会同时改变场景和物体状态，不能称为仅改变机械臂姿态；旧／新settling同时改时长和夹爪命令，不能仅凭此表把影响归于夹爪。更换原版任务的benchmark／实际BDDL指令，没有改变这8个任务的成功数，不支持“所有原版低分都是文字变化导致”。[逐格数据](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/original_protocol40/factorial_cells.csv)／[逐任务配对](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/original_protocol40/paired_comparisons.csv)。

实际初始化也解释了为什么不能把settling视作纯等待：Object task0/init0的5步全零两指关节位置约±0.02068m，而10步开夹爪约±0.03872m，按两指关节位移合计约41.35mm／77.45mm；这不是末端XYZ误差，也未校准为指尖真实几何间距。两组末端位置仅差约0.56mm。4条事后小探针已补齐这两个Object任务的“5步开夹爪／10步零动作”，拆开时长和夹爪因素；不会把这个事后探针假称事前独立验证：

| 固定Object任务／init0／旧指令 | 5步零动作 | 5步开夹爪 | 10步零动作 | 10步开夹爪 |
| --- | --- | --- | --- | --- |
| task0，milk→basket | 成功，154步 | 成功，181步 | 失败，600步 | 成功，139步 |
| task1，cream cheese→basket | 失败，600步 | 失败，600步 | 失败，600步 | 成功，128步 |

四种配置初始化后的两指关节位移合计分别约41.35／67.88／41.09／77.45mm。task0延长等待但保持零动作反而从成功变失败；task1只有10步开夹爪成功。因此存在时长、夹爪与任务的交互，不能宣称“多等待”或“打开夹爪”单独普遍有效。新增4例的同模型与参考初态门禁全部通过，见[汇总原始数据](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/diagnostic_summary.json)。历史相同suite／task／init6的8个旧配置回合与新对照8/8成败一致，但成功步数部分有差异，只能称小样本结局复现，不能称历史轨迹逐位复现。

#### 同画面只换语言：已经观测到什么

固定4个Plus语言案例，各head一个。使用旧真实保存观测，实际Plus／母任务BDDL／旧benchmark三指令按ABC-CBA次序各重复，24次请求的同输入重复均逐位一致；模型、图像和state不变。下表为实际Plus指令对母任务BDDL的差异：

| head／source_index | 首5步平移命令L2均值 | 首5步平移命令L2最大 | 首5步夹爪符号不同步数 |
| --- | ---: | ---: | ---: |
| Spatial／151 | 1.13343 | 2.27215 | 2/5 |
| Object／152 | 0.17108 | 0.28541 | 0/5 |
| Goal／153 | 0 | 0 | 0/5 |
| Long／150 | 0.07627 | 0.19987 | 0/5 |

单位是解码后的原生控制命令，不是米、毫米或实际EEF误差，且尚未按[-1,1]执行裁剪。不能把其中的2.27解释成末端移动2.27米。Goal的首5动作虽相同，全50步有效7维动作并不相同，最早在第8个预测动作出现差异；该轮只执行前5步，所以也不能用第8步的偏差直接证明本轮实际执行错误。需要后续重规划及闭环结果。[动作分通道CSV](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/language_same_observation/action_differences.csv)／[重复与来源记录](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/language_same_observation/receipt.json)／[全chunk核验](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/language_probe_full_chunk_audit.json)。

Spatial／151已有局部闭环恢复证据：[只恢复母任务BDDL指令](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/restore8/case0151/result.json)后88步成功，旧实际Plus改写条件为600步失败。恢复前逐位复现了旧初始5组观测和首请求raw／decoded，改变指令前后物理字段及画面不变、成功oracle未变。这个案例支持长描述改写确实影响该draft任务执行，不证明所有语言类别失败均由这一机制引起；Long样例把caddy描述成wheeled utility cart还存在指称漂移风险，不能一概称为严格同义改写。

#### 闭环恢复：语言两例有效，四例去噪未改变成败

按head／类别事先固定的8个案例均已完成；所有案例在恢复前通过旧初始观测、指令、init向量与首请求raw／decoded逐位复现门禁。恢复臂仍用该suite自身draft，不用Full／verify接管。下表中的步数是实际执行步数，600为达到上限：

| head | 只恢复母任务指令：实际Plus→恢复臂 | 只移除外部相机噪声：实际Plus→恢复臂 |
| --- | --- | --- |
| Spatial | 151：失败600→成功88 | 302：成功95→成功95 |
| Object | 152：失败600→成功147 | 303：失败600→失败600 |
| Goal | 153：失败600→失败600 | 300：失败600→失败600 |
| Long | 150：失败600→失败600 | 301：失败600→失败600 |

这是2个具体语言案例的局部恢复证据；Goal的put／place替换和Long的长描述恢复都没有救回任务。四个固定噪声案例中，去噪没有改变成功结局，但不代表动作或轨迹逐位相同，也不能外推为全部噪声等级／任务无影响。样本按metadata选择，不是挑选失败再统计“救回率”；其中Spatial噪声案例本来就成功。每格只有1个case，不是suite成功率。

首次去噪尝试在0质量步被拦截：用`force_update`重新获取干净画面会同时改变腕相机及缓存状态观测，尽管仿真qpos／qvel未变，不是合格的“只移除噪声”。旧技术失败记录保留、不计策略失败；新版本捕获官方噪声函数真正输入的同帧干净agentview，保持腕相机、state及物理状态逐位不变。每例记录1次reset与5次settle的真实算子调用，移除首帧噪声后后续步也关闭同一官方算子，不增加随机调用或仿真步。[同帧核验](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/capture_frame_audit.json)／[8例恢复原始汇总](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/diagnostic_summary.json)。

#### 同状态输入对照：姿态、噪声、背景分别影响多少动作

12个固定输入对照已完成，合计48次ABBA请求，同输入raw／decoded重复均逐位相同。每格为**首5步平移控制命令差的L2均值（夹爪符号不同步数／5）**，不是米、毫米或预测正确率：

| head | 机械臂关节姿态／速度干预 | 去除外部相机噪声 | 替换静态背景纹理 |
| --- | ---: | ---: | ---: |
| Spatial | 0.23025（2/5） | 0.05449（1/5） | 0.09066（1/5） |
| Object | 0.17392（1/5） | 0.03306（0/5） | 0.17259（0/5） |
| Goal | 0.24097（3/5） | 0.00496（0/5） | 0.11261（0/5） |
| Long | 0.05418（0/5） | 0.00760（0/5） | 0.10458*（0/5） |

*Long背景为预登记的近似母纹理外观辅助臂，详见下方边界。每head／类别只有1个不同任务，不能据跨格大小给扰动或head作稳定排序。[分通道完整CSV](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/input_sensitivity12.csv)另含旋转三轴RMS／L2、最大差及输入状态变化；[主代理独立48请求核验](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/root_input_audit.json)。

这张表支持什么：背景保持物理与非图像观测不变，仍能改变动作，Spatial这例甚至有1步夹爪异号，说明存在对无关外观的敏感性；但尚未为背景做闭环恢复，因此不能声称它造成了失败。机器人干预将7个臂关节置回继承的默认初姿、相应速度归零，保留非机械臂状态；其视觉和自身状态共同变化，本来就可能要求不同动作，所以动作差不等于动作错误。

4个机械臂输入对照的真实EEF位置变化分别为Spatial20.73、Object33.49、Goal38.25、Long8.50mm，SO(3)姿态差约0.073／0.164／0.271／0.038rad。实际使用的native轴角输入未出现接近2π的表示跳变，不支持将本轮失败归于这一数值分支问题，也不据此擅改发布模型的状态表示。

技术记录与边界：首次背景纹理上传被MuJoCo底层model／robosuite wrapper类型不匹配拦下，属于技术失败、未发模型比较请求；修复直接使用对应底层model／render context，旧失败路径不覆盖。Long实际静态家具含无名geom，已用其实际场景XML、mesh／material／body层级和静态祖先确认归属；但实际512×512纹理与母任务1024×1024纹理尺寸不同，原样精确替换仍被拦截，不能将空对照记为“模型不敏感”。另[预登记辅助臂](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/capture_v5_auxiliary_plan.json)仅用确定性PIL11.3.0 LANCZOS将母任务2D RGB纹理降采样到实际512×512，再保持所有物理／非图像输入逐位不变；它是近似母纹理外观，不是原分辨率精确恢复。其余3个背景对照无需重采样。

一个仪表字段已明确作废：服务附带的`visual_preencoder` RMS/L2读取了draft路径未初始化的`encoder_x`缓冲，而实际draft视觉前缀走另一缓冲。该字段无论有限或inf都不可用于特征稳定／爆炸归因；不影响真实模型动作、首请求逐位门禁或闭环结果。`normalized_state`与raw／decoded动作独立有效，详见[指标有效性说明](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/metric_validity.json)。不为补此非必要字段追加模型重启或大规模profile。

主代理已逐项检查52个完整闭环回合（协议40＋事后拆分4＋恢复8）的3,125次请求、15,540步执行：所有raw／decoded有限且形状正确，每请求实际vision1／head1、AE／VLM／verify／Full均0；执行命令精确等于对应预测前缀及[-1,1]裁剪，无新增MP4。[完整执行核验记录](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/root_closed_loop_audit.json)。另独立复跑14项CPU测试、739条响应检查（恢复667＋语言24＋输入48，含已计入上述质量请求的恢复部分，不另累加）、8项同帧噪声检查，均通过；12项输入及所有复用SHA完整，背景3精确＋1近似辅助。

当前归因结论：至少有两类已被局部闭环支持的因素——启动／初态协议差异，以及部分语言改写。背景存在动作敏感性证据，但没有背景闭环致败证据；四例去噪没有救回原失败，不支持把大跌统一解释成噪声。代码及实际路由表明纯draft使用视觉投影token、语言embedding与state经一个learned draft block生成动作，不运行完整VLM或AE；这与它在部分改写指令上的脆弱性相容，但只是结构解释，不能单凭本轮数据证明“参数少”就是原因。[实际draft输入与block实现](../../../policies/realtime-vla-flash/scripts/spec/triton/pi0_spec_infer.py)。

结论边界：四draft的这批最小诊断已完成，但仍不能给出“175例整体下降由各因素贡献多少”的分解。V14-D2现已完成全部195例，最终结果见下节。

### 4.15. V14-D2最终结果：195例完成

固定175个Plus扰动实例和20个原始对照，新增V14-D2全部完成，与旧585条件构成195组四条件配对。D2使用积分节点1→0.5→0、H50/R5、完整视觉和18层AE，仅跳过VLM的L3、L4、L9、L11。旧Full-D10、V14-D10和四个母suite匹配draft均复用原有效数据，无重测。

| 样本组 | Full-D10 | V14-D10 | V14-D2 | 匹配纯draft |
| --- | ---: | ---: | ---: | ---: |
| Plus扰动175例 | 138/175（78.9%） | 89/175（50.9%） | 85/175（48.6%） | 54/175（30.9%） |
| 原始无扰动20例 | 20/20（100%） | 16/20（80%） | 16/20（80%） | 8/20（40%） |

**总体上，D2比V14-D10少成功4例（−2.3个百分点），仍比纯draft高17.7个百分点，低于Full-D10约30.3个百分点。** 原始20例两种V14均80%，但其中有1例D10成功而D2失败，也有1例反过来，并非成功任务完全相同。

分suite看，D10→D2分别为Spatial 52.3%→56.8%、Object 61.4%→59.1%、Goal 31.8%→43.2%、Long 58.1%→34.9%。Long下降10例，抵消了Spatial与Goal的改善；所以不能将总体接近解读成所有任务类型都不受去噪步数影响。每suite仅43—44例，是探索性结果。

175例中D10／D2共同成功69例、仅D10成功20例、仅D2成功16例、共同失败70例。布局17→13、初态9→7、噪声12→10，光照18→21、背景11→12，相机和语言成功数不变；成功数相同也不等于逐例相同。完整四条件分suite、类别和难度已填入第4.5、4.7、4.8节。

执行上保留父GPU0的98例、已释放GPU1的39例，再由GPU0接续58例，合计195个唯一案例，无遗漏或重复有效结果。case154因释放卡中断的旧attempt保留为技术失败、success=null，后续重试已正常完成，不计作额外策略失败。三个服务启动保存的102份NPZ／204对raw与decoded数组重新核验逐位一致；195例全部14,754次请求、73,579步实际命令、73,774条真实XYZ记录通过独立全流核验，全部无新增视频。模型服务和调度进程已退出，GPU1仍留给另一个会话。本次没有新增独占延迟测试，不能从成功率或AE调用次数推算新的端到端加速比。

[最终汇总JSON](../../../reports/idea1_libero_plus_d2_final_20261023T1200Z/summary.json)／[四条件成功率CSV](../../../reports/idea1_libero_plus_d2_final_20261023T1200Z/paired_four_conditions.csv)／[逐组配对变化CSV](../../../reports/idea1_libero_plus_d2_final_20261023T1200Z/paired_changes.csv)／[195例执行验收](../../../reports/idea1_libero_plus_d2_final_20261023T1200Z/root_all195_execution_audit.json)／[启动数组复核](../../../reports/idea1_libero_plus_d2_final_20261023T1200Z/root_saved_startup_arrays_audit.json)。CALVIN等原先暂缓分支仍未恢复，此处完成仅指本次授权的Plus D2与四draft诊断。

## 5. 数据账目与完成核验

以下第一部分保留原始E1—E4账目，不包含后续Plus条件。两批不合并计算成功率，也不把未完成CALVIN条件或未派发B条件记为零分。

新增质量执行记录为E1的48条、E2的80条、E4的108条，共236条；E1重复固定场景，且跨实验任务／接口不同，**不能混成236条独立任务的总成功率**。E2的Full40、E3的200条Full轨迹为旧数据复用，不冒充新跑。E4另有30次已保留的技术失败尝试，不计策略分母。用户原始idea文档和旧模型源码没有修改。

主代理独立检查了45项E1–E4核心CPU测试、E2真实门禁和任务分母、E3冻结输入与方向口径、E4全部108个唯一有效条件键。E4共5,580次质量请求、27,897个执行动作与预测chunk逐项匹配，28,005个真实位置点完整；18案例初态与956个重规划索引的跨条件噪声身份一致。两个服务的质量计数与全部原始推理日志相等；13个冻结源码及快照hash一致，78个已完成键确实跳过。

E2的12条固定视频和原始E4的18条固定视频均经过独立完整解码检查；E2图表／索引、E3源码和输入hash再次核对一致。原始E1—E4本方模型服务和rollout子进程均已退出，原始队列无待派任务；这不代表后续补测范围已完成。详细时间、失败与资源身份保留在[执行记录](idea1_v3_2_execution_status.md)。

后续Plus原585/585有效条件、195组三条件配齐，其中43条入选历史条件复用、542条新增完成；额外30条历史结果保留但排除在本矩阵分母外。42,305次请求、210,828步执行的核验及媒体缺口见第4.12节。新增V14-D2现已完成195例，构成195组四条件配对，独立核验14,754次请求、73,579步执行，见第4.15节。四draft诊断完成52回合；本次授权分支全部完成，相关模型与调度进程均已退出。CALVIN原生v3仅2/200完整条件，另1条用户暂停记录；更正后跨环境B未派发，原生Simpler／PushT和ABCD→D仍有资源缺口，均未恢复。这些异构数据不混成总体成功率。

## 6. 下一步建议（本轮不继续开跑）

- 工程评测仍用B1；本轮batch没有实用端到端收益，且存在夹爪数值翻转，不能直接上线。
- 动作生成先保留简单D2作为成本基准。残差0/3确实能加速D10，但本轮未胜过更简单的D2，不值得默认增加复杂度。
- 不用这三类历史命令外推替换冻结Kalman；实际所需的短前缀仍不如同观测V14。长50步更稳不能抵消前5步变差。
- 若下一轮要检验draft相对泛化，先建立Full能稳定成功、机器人与动作规范一致的域外正对照，再比较全部draft和V14。不要用本轮异构桥接的低分证明容量因果，也不按期望结论挑head。
