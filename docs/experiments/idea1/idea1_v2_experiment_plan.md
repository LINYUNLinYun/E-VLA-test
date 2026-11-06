# idea1 v2：pi 周期交替、阶段路由与 Kalman 预测实验方案

> 文档导航：[实验文档索引](README.md)。

版本：2026-10-14 v1；代码就绪修订：2026-10-15。状态：**实现及24项CPU测试已完成；正式E2/E3未运行，GPU验收待执行。** 用户2026-10-15将等待外部GPU时的共享扫描间隔改为30秒，科学评测协议未改。第8、12节的“未实现”描述为初始设计时的记录，当前可执行入口、历史CPU E4与新attempt要求以[Luna交接单第5节](idea1_v2_luna_handoff.md#5-命令与代码状态2026-10-15-代码就绪修订)为准。

本文件是 v2 主线的详细执行规范，配套入口为 [Luna 交接单](idea1_v2_luna_handoff.md)。用户原始 [idea1.md](../../../idea1.md) 保持不动。[历史总方案](idea1_experiment_plan.md)第16节保存讨论决定，本文件补全实施细节；v2 执行以本文件为准。历史 v1 的实验4是跳层 Shadow，本文件的实验4是 Kalman 比较，不能混用编号。

## 1. 本轮要回答什么

| 实验 | 问题 | 优先级与边界 |
| --- | --- | --- |
| E2：固定交替 | 每两轮 V14 后接一轮 Full，是否改善成功率并节约推理时间？ | 主实验，新增真实闭环；不验证回滚或恢复因果机制 |
| E4：Kalman | 基于已执行动作历史的外推，与 V14 相比，谁更接近当前 Full 的动作预测？分歧发生在哪里？ | 主实验，优先纯 CPU 复用已有预测；不让 Kalman 控制环境 |
| E3：阶段路由 | 仿真器判断自由空间接近时允许交替，其余由 Full 接管，有什么收益和成本？ | E2 完成后的条件实验；使用额外仿真状态，不训练阶段识别器 |
| E1：更深 backbone | Xiaomi 等是否更容易跳层？ | 后置，不属于本轮必跑范围 |

用户接受先做快速验证，不因既有毫米级命令差阻塞 E2；90%成功率是探索目标，不是完成门槛。成功率改善与“Full消除了累计误差”分开报告。固定 V15、1:1/其他交替比例、同状态恢复分支、夹爪替换、在线 Kalman 门控均后置，不作为本轮前置要求。

本轮不新增 mask、不训练/蒸馏、不改 Action Expert、不扫描降噪步数、不扩大到每 task 50个初始状态、不部署异步/双GPU推测验证。收到明确启动指令后才运行 GPU smoke、环境 episode 和正式矩阵；本次仅编写方案。

## 2. 路径与已核实的可复用数据

以下路径相对工作区 `/root/nfs/embodied/benchmark`。执行时在 manifest 中解析为绝对路径并记录 hash，不能依赖 shell 的模糊 glob 选最新目录。

```text
REPO = policies/realtime-vla-flash
OLD  = REPO/results/experiment11_layer_skip_prefix_20261010
NEW  = REPO/results/idea1_v2_pi_alternation_kalman_20261014/<attempt_id>
```

`NEW` 是拟建目录，此文档不创建结果或队列；attempt 使用实际启动时间和唯一后缀。历史结果只读，失败重试另建 attempt/episode 输出并关联原记录，不覆盖成功、失败或 partial。

| 来源 | 路径（相对 OLD） | 用途 |
| --- | --- | --- |
| 固定协议/候选 | `protocol/closed_loop_config_formal.json`、`protocol/candidate_manifest.json` | checkpoint、mask、D3与条件身份 |
| Full闭环 | `closed_loop/Full18_D3_R5_FORMAL/<suite>/run/` | 200条 episode、动作、真实XYZ、视频和 Full 50步输出 |
| V14闭环 | `closed_loop/V14_A03_D3_R5_FORMAL/<suite>/run/` | 200条真实执行基线，不用 Shadow 推导成功率 |
| V14配对预测 | `shadow_supplement_v14/round2_20261011/V14_A03/<suite>/predictions.jsonl` 与 `validated.json` | E4的Full/V14同观测、同噪声50步预测 |
| Full输入 | 原infer元数据中的 `full_input_path` | 缓存输入及校验，不从视频反解 |
| 历史报告/审计 | `analysis/next_round/round2_20261011/next_round_final_report.md`、`final_handoff_audit.json` | 结果与范围参照，不作为新轮完成标志 |

本次只读核对：V14 Shadow四suite共6979点；Full与V14既有D3成绩分别185/200、148/200。旧审计 `experiments_complete=true`、`report_checked=true`；旧交接单的“0/4”“probe未启动”等是历史状态，不能据此重启。

代码接入前将现有基线 manifest/episode/infer/telemetry、预测文件及配置 hash 纳入 `source_manifest.json`。检查 Full/V14 每个 key 唯一、预测 `(50,7)` 有限、同输入/噪声标识匹配、时间线完整。本轮基数为200闭环episode/条件、6979配对点；实际审计不符时列缺口，不猜测或补造计数。

## 3. 冻结环境与模型配置

| 项目 | 固定值 |
| --- | --- |
| 模型 | 当前转换的 `pi0_libero` checkpoint，读取旧manifest的hash |
| 已记录权重hash | `c423a508742f8461fa17d8fb42951837b4419804842ed27a4890a9903042f891`；启动时核实，不自动改成新权重 |
| Full | Full18，skip=[] |
| Variant | V14_A03，1-based skip=[3,4,9,11]，保留各层KV shell |
| Action Expert | 完整18层，D3覆盖完整积分区间，不能只截断D10前三步 |
| 输出/执行 | 每次预测50动作；执行前5，终止时不足5则按实际数量记录 |
| suite | `libero_object`、`libero_spatial`、`libero_goal`、`libero_10` |
| 闭环样本 | task0–9 × 官方init0–4；每suite50、每条件200 |
| 环境seed/等待 | seed=7；初始10个dummy动作，仅稳定物体，不计正式策略动作 |
| 动作上限 | Object280、Spatial220、Goal300、LIBERO-10 520；不含10步等待 |
| 噪声 | `default_rng(SeedSequence([12345, official_init_index, replan_idx])).standard_normal((50,32)).astype(float32)` |
| 控制周期 | 原样本实测0.05s；启动/复用时逐轨迹核验时间戳，不用重规划周期0.25s替代 |
| 动作单位 | 控制命令各轴clip[-1,1]；平移×0.05m，旋转增量向量×0.5rad；核对实际控制器配置 |
| 视觉/状态 | 沿用原双相机旋转180°、resize与归一化；状态保持实际客户端精度，不额外float32化后冒充相同输入 |

噪声沿用原规则，不额外加入suite/task salt，以保持历史可比；相同init和replan在不同task可能使用同一噪声，这是既有协议，不是独立噪声重复实验。条件间按task/init配对；轨迹分叉后，同replan编号不表示相同物理状态。

正常环境超时/任务未完成计失败；进程崩溃、配置错误、缺失数据计 invalid。不能把异常当任务失败，也不能因为成功率低删掉episode。已有baseline复用须通过第8节；复用不可行时先报告，不擅自启动额外400条基线。

## 4. E2：周期交替的逐步流程

条件名建议 `ALT_V14x2_FULLx1_D3_R5`。两种模型共享同一权重和单GPU后端，通过mask选择；缓存各自计算图，每轮只推理被选中的模型，不同时计算两者再选择。

```text
reset官方init → 执行10步dummy → 建立初始telemetry，r=0
while 未成功且正式动作数未达上限:
    读取当前真实obs/state
    model = V14 if r % 3 in {0,1} else Full
    按原规则生成本轮noise
    根据当前obs重新准备输入与条件，推理50动作
    保存实际model/mask/noise/input身份及50步输出
    逐个执行前5动作，每步保存真实状态/视频/事件
    若done或达到上限立即停止，不补齐动作块
    丢弃未执行后缀，r += 1
```

因此完整周期是3次推理/15个动作，连续两轮V14之间仍获取新观测。episode重置循环索引；重试从初始状态重新开始，不能从丢失的中间状态接续。不得使用旧Full输出作为“纠错指令”，也不在Full接管时恢复旧环境状态。

Full和V14图都在测量前按相同预热规则准备；预热不推进环境或循环索引。必须验证两种图共享缓冲区时输入、噪声、KV和输出不会残留上一mask数据。新prompt长度触发后端重建的开销单列；不能每次切换重新加载权重，也不能删除重建日志掩盖成本。

### 4.1 E2分析与成功定义

- 真实任务成功率：总体、suite、task；与Full及V14配对，分双方成功、交替新增失败、交替恢复成功、双方失败。
- 记录GPU事件计时、含输入准备/切换的服务耗时、客户端请求耗时、每episode推理总耗时、实际动作数与墙钟时间。报告p50/p95与均值，不将加权p50当实测周期延迟。
- Full调用比例按真实执行轮次及动作数分别给出；不能直接假设恰好1/3，因为终止可截断周期。
- 对Full接管前后画实际轨迹、目标物体/抓取状态和命令变化。保留夹爪符号切换、掉落候选、首次持续偏离等事件；不把改善直接称为“恢复已累计误差”。
- 90%是用户预期，低于90%仍是有效结果并须完成全部冻结样本；不事后调整循环比例追分。

## 5. E4：Kalman、V14与Full的同起点动作预测

### 5.1 数据划分与复用规模

Kalman单独使用以下划分，E2/E3闭环仍评测全部task。先冻结清单再调参，不根据结果重新选择task。

| suite | 校准：task0、init0–4 | 评估：task1–9、init0–4 | 合计 |
| --- | ---: | ---: | ---: |
| Object | 144点 | 1322点 | 1466 |
| Spatial | 82点 | 1046点 | 1128 |
| Goal | 120点 | 1098点 | 1218 |
| LIBERO-10 | 288点 | 2879点 | 3167 |
| 总计 | 20episode / 4task / 634点 | 180episode / 36task / 6345点 | 200episode / 6979点 |

这些数量来自本次逐行读取现有预测的task_id。启动时重算并核验。校准集不混入主评估均值。该划分只隔离本次滤波参数调优；V14候选和历史数据以前已经分析过，不能声称新模型的独立盲测。

仅复用已有Full/V14预测、动作日志和阶段标签，默认新增模型调用=0、环境动作=0。缺少某个原始文件时停止对应数据核验，不自动重跑Shadow或采新环境。报告预测型指标包含原Full成功与失败episode，不只留成功路线。

### 5.2 索引、历史范围和动作空间

令正式动作从1编号：`a_j`执行后得到状态`s_j`；初始状态是`s_0`。重规划记录`frame_idx=t`表示当前状态`s_t`，允许使用的动作历史最多是`a_1…a_t`，新chunk的offset1对应待执行`a_(t+1)`。原telemetry的`step=t, post_action`携带的是已执行`a_t`。首轮`t=0`无历史，dummy动作不补入滤波器。

先用`trace/infer/telemetry`核实上述对应关系，不仅凭字段名猜测。检查chunk前缀与实际执行action一致；每条已执行命令仅同化一次。主滤波器不能看到当前Full尚未执行的50步chunk、V14预测或未来telemetry。

观测 `z_j=clip(a_j[:3], -1,1)` 是实际送入控制器的平移命令经相同clip后的三维值。它不是仿真器真实XYZ，也不是位置传感器测量。本实验的“真实数据更新”特指已真实执行的动作记录；若未来研究XYZ滤波，需要独立状态模型与真实位置观测。

预测副本输出未来50个三维平移命令；比较Full、V14及Kalman各自clip后的命令，乘50转换为毫米控制目标增量。保留Kalman原始未clip输出及饱和比例，不能因clip掩盖发散。有限但过大的预测保留为结果；NaN/Inf是数值实现异常。

### 5.3 首版滤波模型与固定调参范围

使用float64、线性常变化率模型，状态 `x=[u_x,u_y,u_z,v_x,v_y,v_z]`，`v`是命令值每秒变化率，不是机械臂物理速度。`dt=0.05s`，矩阵为：

```text
F = [[I, dt*I], [0, I]]        H = [I, 0]
Q = q0 * [[I/3, I/(2*dt)], [I/(2*dt), I/(dt*dt)]]
R = r0 * I
```

这是连续白噪声驱动变化率模型离散化的一种参数化，等价于谱密度`q=q0/dt^3`；q0是一个动作周期下的尺度参数，避免误把q0当实际位置噪声。三个轴共享一组q0/r0，四suite使用同一组参数，不做每task/每阶段拟合。

首版仅搜索25组：`q0,r0 ∈ {1e-6,1e-5,1e-4,1e-3,1e-2}`。固定F、dt和初始化，不同时搜索运动模型/执行周期。Q/R表示外推模型的噪声及失配假设，不是对实际传感器误差的测定。

每episode初始化`x=0`，`P=diag(1,1,1,1/dt²,1/dt²,1/dt²)`。收到每条真实历史命令先predict再correct，协方差用Joseph形式或数值等价稳定实现并保持对称。无历史首轮预测为0；冷启动点纳入总体，另列已同化0、1–4、≥5条历史的分层，禁止删除冷启动大误差样本。

校准目标：前1–5步clip后平移L2的均值，先replan内平均、再episode、再task、再四suite等权平均。只用原Full确实执行的有效前缀，终止后不足5步按实际长度；零长度标缺失。误差相差≤1e-12时按较小q0、再较小r0确定唯一赢家。输出完整25组得分和参数hash，冻结后评估task1–9；不能查看评估结果再调参。最优值落边界也如实报告，不自动扩展网格。

### 5.4 预测副本与真实更新流程

```text
对每条episode初始化主滤波器，last_assimilated=0
对按时间排序的每个replan t:
    逐条同化(last_assimilated, t]中的已执行命令
    last_assimilated = t
    复制主滤波器的x和P（不得引用共享可变数组）
    在副本上连续50次predict，保存每步状态均值与协方差
    不调用correct，不将自己的预测当成测量
    保存本次预测，不推进主滤波器
    下一replan到来后，用新到的真实已执行命令更新主滤波器
```

连续50次预测每次都会递推状态和协方差；“不校正”不等于状态不变。未来真实数据到来时，更新主滤波器并创建新副本，不将旧副本的第50步状态当成当前真实时刻继续更新。每个episode重置，无跨episode历史。

### 5.5 参考、指标与解释

增加无需训练的HoldLast：未来各步保持最近一条已执行平移命令，首轮无历史用0。两种方法具有相同历史截止时间。主表包含Kalman–Full、HoldLast–Full、V14–Full，另给Kalman–V14分歧。

交付offset1–50曲线、1–12放大、前缀h=1/2/3/4/5/8/12统计；每轴有符号差、L2毫米、P50/P95、最大值、clip比例及历史覆盖。连续指标按episode→task→suite等权聚合，样本分位数另标采样权重。协方差是模型内部不确定性，未经覆盖率检验不称置信保证。

前1–5步可对照真实Full执行命令并使用动作前状态阶段；offset6–50只对照当前Full的同一预测块，阶段标当前replan阶段。Full随后重规划出的真实动作不能替代原chunk后缀，未来终止后不存在的状态不填充。原50步预测仍可作预测块比较，和有效执行前缀统计分表。

阶段分层至少包括接近、接触/抓取、搬运、对齐/释放、unknown。夹爪沿用V14–Full方向/持物条件分析，不把Kalman连续预测硬阈值成新夹爪策略；旋转本轮不新增Kalman拟合。

分歧解释使用三方指标：若Kalman–V14很大而V14–Full很小，是“V14与Full共同偏离历史外推”的证据；若V14–Full也大，再描述关联。按校准集Kalman–V14前5步误差分布五分位冻结分箱，在评估集报告每箱V14–Full和Kalman–Full误差、样本/task覆盖。它不是V14错误分类器，也不产生在线门控阈值。

## 6. E3：仿真器辅助的阶段路由

### 6.1 启动条件

E2与E4主结果先完成。为落实“E2远低预期先不做E3”，本方案将默认资源门槛具体化为：E2全200条有效完成且总成功率≥80%时进入E3；90%仍是探索目标。80%是本方案的实施选择，不是用户给定数值，也不是显著性/安全门槛。低于80%时标记`deferred_by_preregistered_gate`并报告，不自动换比例或只选高分suite；用户可在运行前调整。这一门槛在运行前写入manifest，禁止看到成绩后改动并仍称预先设定。

### 6.2 首版路由规则（保守的自由空间接近验证）

首版只检验“未发生抓取等操作前的自由空间接近可交替，其余Full”。不声称已覆盖所有简单平移；尤其持物搬运统一Full，避免把夹爪保持误当简单任务。

每个真实环境步都采集状态，但只在重规划点决策，不中断已发出的5步。使用当前及过去的primary对象距离、contact/grasp、物体位移和目标谓词；不使用Full未来轨迹、未来成功、事后完整phase标签、尚未执行的动作。

优先级由高到低：

1. 核心状态缺失/NaN/对象身份无法解析：该轮Full，记录`unknown`，不伪造距离。
2. 设置episode内`precision_latch`：任一已观测grasp成立、任一primary物体相对初始位置位移≥0.03m，或已观测目标进度超过初始值。置位后本episode剩余轮次全部Full。多物体长任务因此可能更保守，完整报告Full比例。
3. 任一当前contact/grasp成立，或最近5个已执行状态中发生contact，或最小EEF–primary距离≤0.06m：Full。
4. 距离0.06–0.08m的过渡带：Full。只有当前距离>0.08m、已有最近3个连续状态均距离>0.08m且无contact/grasp、未置位latch时，判为`free_approach`，允许2:1循环。episode初始历史不足3帧时Full，不用未来补历史。
5. 每次由精细/unknown回到free时循环重新从V14开始；连续free的循环按V14,V14,Full推进。进入Full保护区立即清零循环计数。Full周期槽位仍计为free阶段，不能反过来用模型名定义阶段。

距离使用已核验的对象/部件位置；fixture中心不代表可操作把手表面，无法可靠映射时使用unknown→Full并报覆盖。上述阈值用于固定启发式，不是安全保证，不在评估成功率上调参。可复用旧snapshot、目标对象解析和历史阶段分析，但在线路由状态单独实现/记录，不能从完整轨迹CSV查未来标签。

条件名建议`PHASE_ALT_V14x2_FULLx1_D3_R5`，同四suite、200episode。报告每suite free/precision/unknown轮数、Full调用比例、latch触发类型/时间、成功率与总推理时间。若某task全程Full仍保留，并说明该任务上未检验变体路由收益。

## 7. 数据落盘、统计和可视化

建议新目录结构：

```text
NEW/
  protocol/ source_manifest.json execution_manifest.json kalman_split.json
            kalman_parameters.json route_rules.json code_provenance.json
  status/ queue.json gpu_leases.json work_ledger.json
  smoke/  CPU检查、缓存切换检查、基线复现及视频检查
  closed_loop/<condition>/<suite>/taskXX/initYY/<attempt>/
  kalman/ calibration_scores.csv predictions.npz records.jsonl
  analysis/ success.csv latency.csv phase_counts.csv failure_events.csv
            kalman_offsets.csv kalman_prefixes.csv divergence_bins.csv
  figures/ trajectories/ videos/ media/index.md media/viewer.md media/gallery.html
  report.md completion_audit.json missing_or_deferred.json
```

### 7.1 必须保存的字段

| 粒度 | 内容 |
| --- | --- |
| 配置/来源 | 代码hash、dirty差异摘要、checkpoint与源文件hash、环境/库版本、GPU、控制器、协议、attempt |
| episode | suite/task/init/seed/condition、success、done/timeout/invalid、实际步数、总耗时、文件路径 |
| replan | 索引、动作前step/time、循环槽位、选中model/mask、D3、noise hash、精确输入hash、50步输出、实际执行长度、计时、route原因 |
| 实际输入 | 原dtype的双相机/state/prompt与noise缓存；新数据不得把float64 state转float32后称精确可重放 |
| 每个真实状态 | 初始及每步post-action的XYZ、四元数、夹爪qpos/命令、对象位姿、contact/grasp、目标谓词、时间戳、来源replan/offset |
| Kalman预测 | split、源key/hash、历史截止step/最后同化命令、历史长度、参数hash、预测均值/协方差、未clip/clip结果、对应Full/V14引用 |

输出预测后、环境执行前即持久记录重规划元数据；后续补真实执行长度，保证崩溃时可辨别“预测过”与“执行过”。数组必须copy，避免后端缓冲区在切换后覆盖已保存值。

### 7.2 统计

闭环按task/init配对；给分子/分母、suite/总体成功率和差值pp，2000次task聚类bootstrap、总体按suite分层、固定seed=20261014。所有方法配对重采样，不能把每个replan或动作当独立任务。E4先在episode内聚合再task/suite等权，校准与评估分开；阶段缺失不填0，必须给覆盖数。分位数和均值加权方式注明。

正常成功/失败均纳入冻结样本；invalid列原因和预算消耗，最终缺项时不能只报剩余有效样本为全量完成。不因区间跨0认定等效，不将低频夹爪分歧直接认定致败。

真实闭环失败保留“最终最远阶段”和“首次持续同step XYZ差≥2cm、持续3步”两种诊断，不能混成失败起点；抓取丢失需要前3步连续grasp再后3步连续无grasp，释放可能正常，标候选事件。比较是否在Full接管附近发生，不能据相关性声称恢复机制。

### 7.3 图和视频

Matplotlib输出PNG+SVG：成功率与成本、分阶段失败计数、Full调用比例、E4 offset曲线/1–12局部、阶段热图/误差尾部、真实XYZ三维和XY/XZ/YZ投影。Full蓝色实线点标记，交替橙色虚线三角，alpha约0.6–0.7；加时间点和Full接管标记，重叠时提供分轴差值。E3另出配对图，不把三四条曲线强挤在一起。

保存全部新episode的完整视频（初始帧+每个真实动作后帧），fps取真实控制频率，H264/yuv420p；ffprobe及全帧解码检查。索引同时列完整视频和事件片段，片段明确标事件±25步而非“完整视频”。终止后不延长真实轨迹，配对视频若留最后一帧需标已终止。

案例按固定规则每新增条件、每suite最多3例：Full成功而该条件失败的同step最大XYZ差最大者；V14失败而该条件成功的task/init字典序首例；双方成功的平均XYZ差最接近该成功集合中位数者。去重，不存在则留空，不用反向案例顶替；额外标明是否低频正常释放。全量episode索引均可查，精选案例不估计发生率。

Markdown图片使用相对路径；MPE通过`viewer.md`内嵌图和video播放，另提供无外部依赖的`gallery.html`。不只给无法预览的二进制链接，不以相机截图替代XYZ。链接/解码验证与实际插件播放测试分开报告。

## 8. 现有代码、待实现工作与验证顺序

**本方案没有修改或新增实验代码。以下新入口是实现要求，不是已存在的可执行命令。** 用户要求代码修改由 **GPT-6.1-sol、medium** 负责；Luna负责调度/验收/实验，后续需补代码时继续交给该模型，不能静默换模型。若环境无法调用指定模型，先说明，不冒充已完成指定模型实现。

| 现有模块（相对 REPO） | 可借用部分 / 注意事项 |
| --- | --- |
| `scripts/experiment10/backend.py` | 任意mask+D3、每mask计算图、copy输出；prepare会因prompt长度重建，必须检查切换缓冲区与精度 |
| `scripts/experiment10/serve_policy.py` | 现有noise规则、输入转换、计时；旧mode不是新固定2:1调度 |
| `scripts/spec/spec_client_libero.py`、`scripts/experiment10/client_libero.py` | 官方init、50/5执行、真实telemetry、视频；旧输入缓存state会转float32，新增精确缓存须避免 |
| `scripts/experiment6/trajectory_replay.py` | snapshot、对象/接触解析、历史阶段；在线router另写并限制历史 |
| `scripts/experiment11/shadow_risk_metrics.py` | 单位/夹爪差异，留意其旧聚合权重与本轮episode等权定义区别 |
| `scripts/experiment11/replay_state.py` | 旧输入精度恢复并验证float32 round-trip；不能随意换预处理 |
| `scripts/experiment11/render_failure_event_clips.py`、`scripts/experiment8/render_xyz_video.py` | 实际XYZ和有效H264视频输出；不要沿用硬编码旧结果目录 |
| `scripts/experiment11/gpu_discovery.py` | 只借用共享发现/租约思路；本轮按用户更新采用30秒间隔。旧模块写OLD状态，不能原样调用造成旧目录污染 |

建议新增隔离包 `scripts/idea1_v2/`：配置/manifest、交替策略与阶段路由、client/server适配、队列和审计、Kalman CPU分析、绘图/媒体索引及必要测试。优先新增薄包装，确需修改共享文件时保留用户dirty改动并列清影响。不要直接调用旧`launch_closed_loop --launch`，它会排旧矩阵。

### 8.1 GPU/环境之前完成的CPU检查

1. 序列模拟验证V,V,F循环、episode重置、提前终止、不足5步、重复请求不推进索引。E3测试优先级、latch、0.06/0.08边界、unknown、状态窗口及循环重置。
2. Kalman测试恒值/线性命令、多步预测与解析递推一致、副本不污染主滤波器、P对称半正定、无真实观测时不correct。不能测试“所有情况下P每个分量都单调上升”这种不成立的泛化。
3. 改写未来命令不影响过去时刻预测；改变task1–9数据不影响校准参数选择；每条历史只同化一次；t=0、t=5、终止短块的索引单测。
4. 冻结划分634/6345点、200episode身份、控制周期、源hash和实际执行前缀；CPU入口不可创建环境/CUDA推理。
5. 检查输出覆盖保护、断点续跑不重置账本、invalid不混作失败、媒体路径可解析。

### 8.2 明确启动之后的GPU smoke

单个空闲卡先运行，其他卡不预先拉起全矩阵。使用四suite task0/init0和task5/init0各一个保存观测，共8点；恢复原输入精度，按既有后端准备流程构造同状态噪声。

每点比较静态与交替调用顺序下Full和V14输出，顺序包含F→V→V→F→V→F；重放同一点时重置相同噪声/输入。要求有限 `(50,7)`、正确mask/D3，静态与切换结果 `allclose(rtol=1e-4, atol=1e-5)`。记录最大差值。若缓存历史导致parity失败，保留现场并参考`OLD/task_type_layer_probe/round2_20261011/parity_investigation.md`；不提高容差，也不无边界重放历史试错。额外逻辑预测共用256次诊断上限，内部forward另记。

随后用新client/server静态Full和静态V14分别复现四suite task0/init0，共8条诊断episode，保持原suite预热与task顺序；与旧相同key核对实际动作、XYZ、终止/成功及图视频。动作容差沿上，XYZ每轴绝对差≤1e-5m作可重现检查，不是任务容差。匹配后才复用历史基线；不匹配时暂停GPU矩阵定位实现差异，E4独立CPU工作可继续。

最后运行E2四suite各task0/init0，配置不变且验收通过则复用为200条正式结果，不能额外算4条重复正式样本。每条检查V,V,F实际来源、N+1状态/视频、夹爪方向、计时、终止与日志完整。

## 9. 工作量、调度与失败边界

### 9.1 固定工作量

| 项目 | 科学样本 / 新增工作 | 上限口径 |
| --- | --- | --- |
| E4 | 6979旧配对点；25组CPU校准、一次冻结评估 | 新增模型/环境=0 |
| 基线复现smoke | 8条诊断episode | 最多2640正式动作、528次预测；另80步dummy |
| E2 | 200条新episode，前4条smoke合并计入 | 最多66000正式动作、13200次动作块预测；另2000步dummy |
| E3 | 过第6.1节门槛后200条 | 同E2；不过门槛列deferred，不伪报完成 |
| 额外GPU诊断 | 缓存/切换parity等 | 最多256次逻辑动作块预测 |
| invalid重试 | 主实验与smoke合计最多8次episode重试，每key最多1次 | 保守最多4160正式动作、832次预测；另80步dummy |

若E3执行，合计最多416次环境episode attempt（400正式+8静态smoke+8invalid重试），正式动作预测调用加诊断保守上限28016。若E3不执行，上限216次attempt、14816次逻辑预测。早停任务实际小于上限。预热/图捕获/后端内D10校验不计为D3科学预测，但要另记内部forward与初始化耗时，不能称它们为零成本或额外评测D10。

不存在自动补跑400条旧基线的预算；固定V15/1:1等后置项目不占本表、不随主实验启动。需要扩大样本、预算或改变协议时先说明新范围。

### 9.2 并行与续跑

- E4 CPU与E2代码/运行独立并行；只限制CPU线程以免影响模拟器/计时。CPU资料不完整只阻塞相关分析，GPU实现问题不阻塞完整的E4。
- 通过单卡smoke后，E2按suite分成4个单GPU worker，每worker顺序task0–9、init0–4，以保持已有运行历史；快suite结束先做CPU分析，不为填满卡重复episode或启动后置项目。
- 四卡均空闲可全部使用；只有部分空闲就按队列运行。自己的worker完成以进程退出/心跳事件续派；等待外部空闲卡才每30秒最多扫描一次（用户2026-10-15更新），多个调度器共用发现时钟和租约。不得抢占/终止他人进程。
- 单GPU内同一后端切换Full/V14，不以双卡隐藏切换成本。记录GPU型号、独占情况、CPU/video并发；非独占延迟标注，不将其当独占benchmark。
- 有效完成键不重复；重启从首个未完成episode继续。普通失败不重试。异常保存traceback、partial、消耗与code/config hash；重试前先定位并修复，超过预算或同类问题再次出现则停止相应阶段并交付缺口，不能静默丢样本。

## 10. 验收与报告

完成不是达到90%，而是协议正确、样本与记录完整、结果可复查。交付 `completion_audit.json` 分开标记E2/E4/E3；E3门槛未满足使用deferred，不能把它标为已跑完。后置项目注明未运行原因。

- E2：200/200唯一有效key，所有轮次model来源符合循环；baseline复用审计通过；XYZ/实际动作/输入/视频/计时齐全；异常与重试另列。
- E4：634校准、6345评估配对点，25组校准得分和唯一冻结参数；历史截止检查通过，offset1–50完整，阶段/终止短块分母明确；新GPU与环境计数为0。
- E3（若运行）：200/200，在线路由仅依赖历史状态，与冻结规则逐条复核；报告实际free覆盖和Full比例，不能将全Full的task声称为交替验证成功。
- 所有完整视频解码、XYZ来自实际telemetry、图片/视频相对路径解析通过；不能用旧报告audit为新媒体背书。

中文 `report.md` 采用四部分：E2成功率与成本；E4预测比较与分歧解释；E3条件结果/暂缓原因；限制、未完成/后置项目与建议。每部分先给通俗结论，再给表图和源数据入口。必须注明：动作目标增量≠实际末端位移；Full路线Shadow≠V14实际闭环；相关性≠恢复/致败因果；90%是目标；本轮不以误差阈值宣称安全。

## 11. 给代码实现者的交付要求

代码实现由GPT-6.1-sol medium负责，Luna应向其提供本文件与指定文件所有权，并说明共享工作区已有用户修改，不得回退。实现结束交付：改动清单、CPU测试结果、实际CLI的`--help`、dry-run任务清单、配置样例、运行/续跑/分析/审计命令和已知限制。不得在代码实现阶段自行启动GPU或环境。

CLI统一要求：默认只dry-run；显式`--launch`才运行；所有路径来自参数/新manifest；不存在的CLI不得在交接单中声称已验证。最终实现后把真实命令补入[Luna交接单](idea1_v2_luna_handoff.md)，并注明代码来源的模型、reasoning及commit/文件hash。若测试发现本规范矛盾，先修正文档和版本记录，不暗中改变科学协议。

## 12. 来源与状态记录

本次读取过的实现：旧protocol/backend/serve/client、snapshot与phase函数、Shadow worker、单位换算、GPU发现和视频渲染代码。旧代码具有复用价值，但新交替调度、Kalman分析和在线router尚未实现，不得误读为已有可运行系统。

Kalman预测/校正定义参照[官方predict/correct说明](https://www.mathworks.com/help/control/ref/extendedkalmanfilter.predict_method_state_estimation.html)。模型结构、成功率、执行协议与数据数量来自本地现有产物，不重新引用外部论文成绩替代基线。

本版没有改 `idea1.md`，没有修改实验代码，没有调用GPU或环境，也没有启动Luna/代码代理。文档交接之后，用户明确启动时才按上述顺序实施。
