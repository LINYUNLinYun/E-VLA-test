# idea1 v3 补充实验：质量约束下的加速搜索

日期：2026-10-20。状态：**下列冻结范围已执行完成，实际1680有效episode与7配置计时见[执行状态](idea1_v3_supplement_status.md)及[最终报告](idea1_v3_supplement_report.md)。** 本文保留设计协议，不修改用户idea1、已完成v3参数或旧结果。原轮1976有效episode与终稿保留；补充结果独立，不覆盖旧分母。

## 1. 目标与不变边界

E1只研究π0 V14-D2＋双时间点verify的tau、周期Full、夹爪保护与拒绝接管成本；E2重点搜索π0.5跳4／5层。回答90%、85%、80%质量档内实测最快的候选，而不是只报告更快但失败更多的条件。本轮不训练、不加E3历史机制、不延长Rmax5、不用离线代理筛选淘汰闭环候选。

角色：GPT-6.1-sol low独占模型／执行代码，medium负责方案／状态／报告与必要窄验收，Luna Max为唯一GPU执行owner。允许四张可用卡独立分片；不抢占外部进程。只有待派任务且无可用卡时每30秒查卡；已有活跃任务以完成／异常事件推进。旧结果只读复用，不重跑Full成功率，不热改运行控制路径。

沿用本地checkpoint、真实输入／原生transform、H50（π0）或H10（π0.5）、D2 draft／D10 π0.5、原生timeout、种子规则和任务初态。π0保持安全QK revision与真实模型子进程安装／参考receipt，不回到已知越界旧核。Full KV独立clone，不被V14覆盖；拒绝draft不能先执行；reset清cache；P按draft轮而非环境步计数。

## 2. E1：分批直接闭环，最多16个新条件

每个新条件先四suite×10任务×init6＝40条。既有严格τ=.10/P2与τ=.15/P2在init6–10的各200条直接复用；旧校准8条不冒充新首40或扩大主样本分母。

### A：tau与周期（8个新条件）

固定strict夹爪保护与所有Full事件D10，V14 draft D2。P表示连续draft轮上限，periodic Full后归零；更大的P允许更旧Full KV，并不取消每轮双分支verify。

| tau | 周期P |
| --- | --- |
| .10 | 4、8 |
| .25 | 4、8、none |
| .40 | 4、8、none |

none禁周期Full，但保留startup／拒绝Full，不是纯draft。8×40＝320条，全部直接闭环，无离线淘汰。报告与旧P2严格条件同init6表。阶段A结束才冻结B父配置，不预先展开所有交叉组合。

### B：两个父各两种gripper，共4项

父条件选自A与可复用旧严格P2，**排名统一只用共同init6各40条**（旧条件不能用200条与新40混排，校准stage排除）：一为成功数最高者，二为success≥80%中闭环累计policy wall／实际动作最低者；重复则取次优独立条件。质量并列按较低成本，再按较小tau／P（none排序在有限P后）；速度并列按较高成功数再同一参数顺序。记录排序与引用，不称最优。包括被拒draft＋verify＋Full，不用episode/hour；共享负载下只作筛选，正式ratio另用第4节同卡计时。

| 规则 | 冻结可执行定义 | 其余不变 |
| --- | --- | --- |
| G1 confirmed_disagreement | 在空间可接受前缀内，若同一步两个K重建夹爪sign均反于draft，截断至该步前；没有独立的自身切换否决或pending Full | Full均D10 |
| G2 off | 关闭夹爪否决、截断及pending Full；夹爪动作仍由真实draft输出，不钳为上一符号 | Full均D10 |

strict旧语义：任一重建分支未来窗口的夹爪符号切换使接受0；draft接受前缀符号切换则截断并pending Full。上述旧切换检查不是预测不一致率。B为2父×2规则×40＝160条，空间零接受仍接管；两个新规则不继承strict的pending机制。

### C：最多4项仅拒绝Full-D2组合

每个父各取strict和该父最佳relaxed（G1／G2中按质量—成本排序选一个），将**仅拒绝接管**改Full-D2，共最多4项。拒绝包括空间零接受、夹爪拒绝及strict的pending_gripper接管；startup／periodic仍D10。每次接管仍fresh Full prefix并更新独立Full KV，不执行被拒draft。不得新增runner-up tau/P或所有Full事件D2条件。

重复语义／与已有条件相同则引用，不占新任务。C的具体父／组合先写冻结选择JSON后派发，最多160条。A8＋B4＋C4≤16新条件，不为了用满16条扩网格。

首40后至多2个E1 Pareto候选只补init7–10各160条至200。优先≥90%质量档中最高speed者和≥80%档中最快Pareto者；若重复则不凑数，没有满足者如实保留、不加搜。最终必须单列选择集40与新增160，不把选择集当独立确认；质量档以实际样本率定义，附不确定性，不承诺总体真实成功率下界。

## 3. E2：5个跳4层＋5个跳5层

均为原18层1-based索引，保留1、2；允许相邻簇，不预设连续层必然失败。每个mask保持prefix hidden-update skip＋原层号KV shell，完整AE suffix；本补充应使用独立revision让suffix直接真实original scanned native路径，避免原轮masked AE展开的计时混杂。先CPU parity／真实首episode验证，所有新mask用同一revision，旧质量只作既有参照、不同revision不伪称完全受控速度对照。

| 层数 | 名称 | skip列表 | 结构 |
| --- | --- | --- | --- |
| 4 | s4_spread_a | [3,6,10,14] | 非相邻分散 |
| 4 | s4_spread_b | [4,8,12,16] | 非相邻分散 |
| 4 | s4_transfer | [3,9,11,15] | 从既有强mask扩展 |
| 4 | s4_early_cluster | [3,4,5,6] | 早段相邻簇 |
| 4 | s4_mid_cluster | [9,10,11,12] | 中段相邻簇 |
| 5 | s5_spread_a | [3,6,9,12,15] | 非相邻分散 |
| 5 | s5_spread_b | [4,7,10,13,16] | 非相邻分散 |
| 5 | s5_transfer | [3,9,11,14,17] | 从既有强mask扩展 |
| 5 | s5_early_cluster | [3,4,5,6,7] | 早段相邻簇 |
| 5 | s5_mid_cluster | [8,9,10,11,12] | 中段相邻簇 |

10×40＝400条直接闭环。旧V14 [3,4,9,11]＝33/40、旧Custom [3,4,7,12,15]＝32/40复用，不重复。既有Full／跳2／3层完整结果亦复用成功率。跳4与跳5各至多1个新候选补init7–10至200，最多新增320条：每层数在首40成功率≥80%者中按**最高成功数→最低policy_ms／实际动作→稳定mask名**选择；没有≥80%则不扩、仍报全40与旧对照。E2质量优先，不套用E1质量档内最高速度的扩样规则，也不因层数更多自动优先；最终仍报告各质量档Pareto。

## 4. 速度：相同硬件／后端／输入边界，冷启动单列

只补Full-D10与Full-D2的小批量推理计时，不补Full成功率episode。π0与π0.5分别比较，不能跨backend做模型优劣排名。为保持最多7配置，纯V14-D2本批不另计时，旧质量参照仍可复用。

- E1五点reset不能触发P8，**不得用旧五点微基准作本次周期搜索最高加速结论**。在已计划选中候选扩样的init7／task0、四suite原闭环内，各至多保存25个连续真实predict输入／种子一次为小pkl；不新增episode，不修改已运行服务。短episode保留真实短序列，不补造、拼接或补跑。
- 正式比较统一一个共同bank：稳定取优先90%档候选的上述四suite输入；无该档则取实际首选候选。两Full与两个候选都用同一bank，其他候选自身采集输入仅留参考，不新增计时；报告采集策略依赖与实际覆盖，不能各用自身bank再横排最高speed。
- π0 Full-D10、Full-D2及至多2个最终候选在同一物理GPU、同Triton安全revision、同一冻结长序列、相同预处理／decode边界和线程配置串行计时。每suite≤5次warmup后reset路由状态，完整≤25连续点measure一次；每配置总≤20warm＋100测量，预算不增。全router包括draft、verify和拒绝Full；报告实际periodic触发数与cache-age范围，未触发的情况不冒称覆盖。
- π0.5 Full和最终跳4／跳5候选在同一物理GPU、同JAX／native scanned suffix revision／原生H10D10条件计时；可复用旧20真实输入、各5点同序列。既有旧revision timing不直接作正式ratio。除必要新revision同卡Full计时，不重跑既有成功率。
- setup／模型加载／编译、测量中runtime重建分开；若100测量仍混入setup，按setup×route分组并展示混合总量，不称100纯热态。记录mean／p50／p95，不对各组p95加权造总p95。
- E2采用suite／prompt外层：每段5点先预热一次，再同段5次measurement、各段reset，仍总20setup＋100测量；E1按上列长序列协议。不因追求热态扩大预算；记录真实runtime事件，不因循环调整自动宣称消除编译。
- **主speedup为单位执行动作成本比**：`(Full同边界总时延/Full实际动作数)/(router同边界总时延/router实际动作数)`，分别以Full-D10主参照、Full-D2次参照。固定重放的动作数明确为execute_length／virtual ack累计，真实闭环用实际executed长度；候选L<5不能忽略。每请求时延比仅作次列，不能用它替代主加速式。
- 序列重放仅是固定输入协议，不把virtual ack历史或其路由频率当闭环。
- 另用真实闭环sum(policy wall)/sum(actual actions)与每请求总成本、Full/拒绝率、R实际分布呈现。冷启动另列；多进程episode/hour不是算法加速，提前失败／短轨迹不能冒充更快。
- 报告90/85/80%质量档Pareto；不得只凭不同card／共享CPU/NFS背景下的绝对毫秒宣称最高加速比。

计时规模上限：π0两Full＋最多2候选，π0.5 Full＋最多2候选＝最多7配置×(20setup＋100测量)，不得再对全部tau／mask逐一micro测速。若已有同硬件同revision数据满足边界，直接复用，不为整数预算重跑。

## 5. 总预算、保存和必要验收

| 分支 | 首批上限 | 扩展上限 | 新闭环上限 |
| --- | ---: | ---: | ---: |
| E1 | 16×40＝640 | 2×160＝320 | 960 |
| E2 | 10×40＝400 | 2×160＝320 | 720 |
| 合计 | 1040 | 640 | **1680** |

1680是科学样本硬上限、不是必须填满的任务量；不含至多840个小批计时请求，不另加校准网格或Full闭环。墙钟以真实吞吐估计并随阶段更新，不把GPU空闲等待计成模型性能；预算／网格变化须root明确冻结后再准备新manifest。

科学identity必须包含tau/P、gripper规则、各Full原因的D、mask、backend revision与安全revision。新补充attempt独立保存，引用旧同语义对照key；新增量只init7–10，不手改清单，不重跑valid失败。技术invalid与科学失败分开，修复先诊断，episode边界新revision，不做无界retry。

必要验收限于：E1零／部分接受、每种夹爪规则、P计数、每种Full原因D、cache隔离、客户端只执行最终允许前缀；E2 prefix skip与KV shell、suffix真实scan、与原生Full有限CPU数值parity、首真实mask接口；计时确认同GPU／输入／边界与完整拒绝路径。首episode直接计入科学清单，不额外大smoke，不重复全旧raw hash或全视频解码。

交付：独立补充status／报告，条件与source引用表、逐suite／overall成功率、首40与追加160分段、真实路由／每动作成本、同卡计时ratio、质量档Pareto、必要固定案例索引。既有v3终稿保持完成快照，本补充结论不得静默改写原轮。
