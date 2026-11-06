# v2.2 最终汇总：两步采样、非均匀日程与 VLM 跳层

日期：2026-10-19。**这是用户批准的快速验证阶段总结，不是原 10800 条全量评测完成声明。** 数据、图表及报告已完成独立针对性复核。全部实验文档见[索引](README.md)，原始 idea 文档未修改。

## 1. 先说结论

1. **最值得继续跟踪的是 Xiaomi 的 skip12＋均匀两步。** 共同样本成功率 120/120；相对 Full-D5，动作生成约快 2.45×，服务总推理约快 1.55×，客户端往返约快 1.49×。这是当前最清楚的质量／延迟收益组合，但不意味着总体成功率必然为 100%。
2. **π0 的 V14 没有靠调整步长补回主要损失。** 均匀两步 Full 为 90.8%，V14 为 70.8%；V14 本批最高日程也只有 71.7%。与此同时，V14 的服务总推理只快约 1.11×，这个交换不理想。
3. **DB-CogACT 不能直接照搬 Xiaomi 的两步方案。** 原生 D2 的 Full 成功率从 93.3% 降到 66.7%。固定高端起点重新分配 DDIM 日程后，Full-m25 达到 86.7%，skip9-m10 达到 79.2%；说明应联合考虑跳层与采样日程，而不是只改迭代次数。
4. **没有观察到统一的“预测越远误差越大”或“只有精细操作才受损”。** 不同模型、阶段和指标的变化不同。动作空间差、真实末端轨迹差、任务失败是三种不同证据，不能互相替代。

优先建议：先保留 Xiaomi skip12-u50；DB 跟踪 Full-m25 与 skip9-m10；π0 暂不为仅一两条成功差异继续追逐“最优节点”。以下给出依据。

## 2. 实际测了什么，分母是多少

| 模型 | 固定 VLM 变体：跳过层号均为原始 1-based | 预测 H／执行 R | 唯一条件数 |
| --- | --- | --- | ---: |
| π0 | V14_A03：3、4、9、11 | H50／R5 | 16 |
| Xiaomi | skip12_short_runs：4、5、6、8、9、10、29、30、31、33、34、35 | H10／R10 | 18 |
| DB-CogACT | skip9_adjacent_pairs：5、6、8、9、21、22、23、24、25 | H16；Spatial R12、Object R16、Goal R16、LIBERO-10 R15 | 20 |

- 成功率来自各条件**独立闭环执行自己的动作**，覆盖 Spatial、Object、Goal、LIBERO-10，各 10 个任务，统一取官方初态索引 6、7、8（0-based）。每条件 120 条、每 suite 30 条，共 **6480 条**。不是只执行 Full 的 shadow 成功率。
- 另有 **2833 条有效补充结果**原样保留，未混入主表分母。主批次与补充共 9313 条有效结果；两条历史技术 invalid 保留，均有合法有效重试，不因任务失败而重跑。
- 同观测动作分析：每模型 **32 个冻结的真实 Full 历史观测**，不是随机初始化张量。它们按已有场景／阶段选取，不是全任务均匀随机抽样；不同模型的阶段分布不完全相同。总计 1728 个条件／观测输出，保留 π0 的全部 50 步、Xiaomi 的 10 步、DB 的 16 步。
- 延迟：每模型均保留 32 个真实观测。π0／Xiaomi 每观测每条件 1 次预热＋5 次测量，即每条件 160 个正式测量；DB 继承已验收的 5＋20，即每条件 640 个。测速逻辑请求共 **22528 次**，后端构建／启动预热另记，不混入正式测量。
- 配对区间按 suite 分层、task 为簇，保留每任务三个初态，2000 次 bootstrap，seed=1701。它们是探索性、未做多候选校正的区间，不能把本批最高分宣布为最优。

同模型科学配置保持既定协议，但 π0 闭环数据保留了混合吞吐执行版本的历史；不能把“配置一致、账本有效”理解为全部生产请求逐位等价，或把汇总差异全部因果归于 mask／日程。该限制见第 6 节。独立延迟测试另有其单请求执行与验证记录。

均匀两步为 u50。Flow matching 用剩余噪声节点 `1→m→0`；π0 实际时间下降，Xiaomi 实际时间上升。**第一段长度为 1−m，因此 m70 是前小后大，m30 是前大后小。** 这里是两次 Euler 更新，不是二阶中点法。π0 本轮没有另测原生 D10 对照，不能从本轮量化 π0 的 D10→D2 收益。

DB 使用 DDIM：原生 D2 为实际 `50→0→clean`，custom-u50 为 `90→45→clean`，custom-m10 为 `90→9→clean`。原生 D10→D2 同时改变起点网格，不是纯粹的 NFE 因果消融；也不能将 DDIM 的 m 标签当成 flow matching 步长。

## 3. 成功率：哪些结论清楚，哪些只是候选

![全部54条件成功率][heat]

图中蓝色行名为 Full、橙色为跳层变体；每 suite n=30，总体 n=120。图中文字取整数，精确分子／分母以表格和[全部成功率 CSV][success]为准。

### 3.1 两步日程：不是统一的“前小步更好”

| 模型／mask | 均匀两步 u50 | 本批最高非均匀日程 | 相对 u50 |
| --- | ---: | ---: | ---: |
| π0 Full | 109/120（90.8%） | m70：111/120（92.5%） | +1.7 pp |
| π0 V14 | 85/120（70.8%） | m60：86/120（71.7%） | +0.8 pp |
| Xiaomi Full | 120/120（100%） | 多个日程同为 120/120 | 无可分辨领先 |
| Xiaomi skip12 | 120/120（100%） | 没有非均匀日程超过 u50 | 无收益证据 |
| DB Full，custom 日程 | 97/120（80.8%） | m25：104/120（86.7%） | +5.8 pp |
| DB skip9，custom 日程 | 76/120（63.3%） | m10：95/120（79.2%） | +15.8 pp |

π0 的两个领先区间分别为 −3.3 至 +5.8 pp、−6.7 至 +8.3 pp，不能据此确认优于均匀两步。m10 反而让 π0 Full 降到 80.8%、V14 降到 62.5%。

DB 有较明确的探索信号：Full-m25 相对 custom-u50 的区间为 +0.8 至 +10.8 pp；skip9-m10 为 +5.0 至 +25.8 pp。同一个 m10 对 Full 相对 custom-u50 净少成功 5 条，对 skip9 却多 19 条，交互差为 +20.0 pp，区间约 +6.7 至 +33.4 pp。**日程可能需要随 mask 调整**，但候选来自本批多日程搜索，仍需新样本确认。

### 3.2 减少迭代：Xiaomi 可行，DB 原生 D2 损失明显

| 模型／mask | 原生步数 | 均匀／原生 D2 | 配对成功率差及 95% 区间 |
| --- | ---: | ---: | --- |
| Xiaomi Full | D5：119/120 | u50：120/120 | +0.8 pp［0，+2.5］ |
| Xiaomi skip12 | D5：120/120 | u50：120/120 | 0 pp［0，0］ |
| DB Full | D10：112/120 | D2：80/120 | −26.7 pp［−36.7，−16.7］ |
| DB skip9 | D10：99/120 | D2：80/120 | −15.8 pp［−26.7，−5.0］ |

Xiaomi 的退化区间不能用来证明总体等价：本批样本接近天花板，skip12 两组全成功使 bootstrap 差值退化为零。已有完整 200 条补充结果中，Full D5／D2 均为 198/200，skip12 为 197/200、198/200；相同总成功数也不代表逐初态结果完全相同。

DB 的 Full 原生 D2 相对 D10 有 4 条改善、36 条退化；skip9 有 10 条改善、29 条退化。两种 mask 的少步损失之差为 +10.8 pp，区间约 −3.3 至 +25.9 pp，**不能确认少步一定放大或减小跳层损失**。完整配对结果见[配对统计 CSV][paired]。

### 3.3 任务类型差异：Goal 更脆弱，但不能定位到单层

以下严格比较相同日程的 Full／变体，数值是成功率下降的百分点：

| 匹配比较 | Spatial | Object | Goal | LIBERO-10 |
| --- | ---: | ---: | ---: | ---: |
| π0 u50：Full→V14 | −23.3 | 0 | −46.7 | −10.0 |
| DB D10：Full→skip9 | −3.3 | −6.7 | −26.7 | −6.7 |

π0 V14-u50 在 Object 仍为 30/30，Goal 却为 13/30，对照 Full 为 27/30。DB skip9-D10 的 Goal 为 21/30，对照为 29/30。这是“同一个固定 mask 对任务类型影响不同”的线索，而不只是 Goal 原本较难。

但本轮没有验证“Goal 保持得好、Object 明显变差”的互补 mask；也没有单层消融，不能进一步宣布某个具体层专门负责 Goal。

## 4. 延迟：动作头确实变快，但端到端不会同比例下降

![独立延迟比较][latencyfig]

以下均为独立请求的 mean，单位 ms，不是八日程共享 VLM 后的摊销时间。

| 条件 | 成功率（/120） | 动作生成 | 服务总推理 | 客户端往返 |
| --- | ---: | ---: | ---: | ---: |
| π0 Full-u50 | 109 | 14.16 | 52.73 | 88.95 |
| π0 V14-u50 | 85 | 14.07 | 47.43 | 73.73 |
| Xiaomi Full-D5 | 119 | 109.26 | 214.78 | 246.89 |
| Xiaomi Full-u50/D2 | 120 | 44.03 | 150.15 | 176.18 |
| Xiaomi skip12-D5 | 120 | 108.69 | 203.22 | 230.56 |
| Xiaomi skip12-u50/D2 | 120 | 44.51 | 138.22 | 165.25 |
| DB Full-D10 | 112 | 68.35 | 171.74 | 228.49 |
| DB Full-D2 | 80 | 14.26 | 117.77 | 176.19 |
| DB skip9-D10 | 99 | 68.41 | 150.70 | 196.27 |
| DB skip9-D2 | 80 | 14.19 | 96.84 | 140.55 |
| DB Full-m25 | 104 | 14.18 | 120.20 | 177.59 |
| DB skip9-m10 | 95 | 14.27 | 96.80 | 143.76 |

- Xiaomi Full D5→D2：动作生成 **2.48×**，服务 **1.43×**，客户端 **1.40×**。Full-D5→skip12-D2 联合为 **2.45×／1.55×／1.49×**。
- DB Full D10→原生 D2：动作生成 **4.79×**，服务 **1.46×**，客户端 **1.30×**。若换成更值得跟踪的 skip9-m10，联合服务 **1.77×**、客户端 **1.59×**，但成功率仍比 Full-D10 低 14.2 pp。
- π0 u50 跳层：VLM 均值 21.15→16.39 ms，约 1.29×；动作头几乎不变，服务仅 1.11×，不能忽略其 20 pp 的成功率损失。

原因不是把“少一半迭代”自动换算成“总时间少一半”：例如 Xiaomi Full 的视觉约 21.55 ms、VLM 约 59.67 ms，并未随 D5→D2 消失；DB Full 的视觉约 13.02 ms、VLM 约 65.31 ms 也保留。输入处理、适配及客户端传输／持久化同样有成本。

服务计时在预测持久化前截止，客户端包含往返与持久化；各计时边界并非封闭加和账本，不能把分项中位数相加当总延迟。测速仅证明实际采样及调用边界的 GPU 隔离，不代表整机独占、零观察开销或 episode 吞吐提升。不同模型／实现间不据绝对 ms 排名。

全部分项的 mean／p50／p95 见[延迟明细][latency]及[加速比表][speed]。例如 Xiaomi Full-D5→skip12-D2，服务 p50 为 213.56→137.00 ms，p95 为 231.93→150.78 ms；客户端 p95 为 306.30→227.83 ms。客户端长尾仍明显，这不是最终部署 SLA 测试。

## 5. 动作、末端轨迹与失败阶段分别说明什么

### 5.1 同观测动作差：前缀不等于零误差，远期也不必单调增加

同一点、同一初始噪声比较预测输出，参考预测不是动作真值。下表平移量是原生动作前三维差的 L2 均值，**不是毫米**。每行有 32 个真实观测；前五步分母为 160 个动作位置，不是 160 个独立任务。

| 条件相对参考 | 前1–5步平移差 | 第6–H步平移差 | 前1–5步夹爪方向不一致率 |
| --- | ---: | ---: | ---: |
| π0 V14-u50 vs Full-u50 | 0.1454 | 0.1261（H50） | 2/160＝1.25% |
| Xiaomi skip12-u50 vs Full-u50 | 0.0496 | 0.0561（H10） | 0/160 |
| DB skip9-m10 vs Full-m10 | 0.1637 | 0.1603（H16） | 0/160 |
| DB skip9-D2 vs 同 mask D10 | 0.1698 | 0.1647（H16） | 6/160＝3.75% |

![V14全50步动作差][actionfig]

实线为各 offset 的观测均值，阴影是 32 个观测的 p05–p95，不是成功率置信区间。虚线标第 5 步。π0 本批前五步均值反而较高，不能支持“越往后必然越偏”的统一假设；夹爪在后续部分 offset 仍有不一致，不能只看前缀就称全 H 相同。

阶段也并非“抓取最大”：π0 V14 前五步平移差在 approach 为 0.1382（8 观测）、抓取／操作 0.0840（7）、搬运 0.1606（8）、目标对齐 0.1964（8），contact 为 0.1037，但只有 **1 个观测**。阶段标签来自冻结观测所属的历史 Full 轨迹，不是当前变体闭环；场景／任务组成也可能混杂。

夹爪指标统计所有已有阶段，不只接触／搬运。低不一致率不等于无致命风险，零观测不一致也不等于总体安全；这些 32 个离线点不能与 120 条闭环失败一一归因。L2 非负不代表凭空产生误差，相同向量的 L2 为零；未做同模型重复精度基准，不能把差值直接叫数值噪声。

另从均匀两步日志看，π0 Full 预测场 RMS 约 1.236→1.031，V14 为 1.233→1.028；Xiaomi Full 约 1.040→1.045、skip12 为 1.033→1.023。**幅值不是变化率或曲率**，更不是末端速度。π0 该字段由更新量／dt 得到，Xiaomi 为直接速度输出，绝对值不作跨模型比较。这不足以证明“早期场变化更剧烈，因此必须早期小步”。

全部 offset、旋转／有符号各轴、夹爪及实际执行 R 窗口见[动作逐步表][actions]和[分阶段／窗口汇总][errors]。

### 5.2 实际末端 XYZ：这是执行后的累计轨迹差

以下先在每对轨迹的共同仿真步计算真实 EEF 距离的 p95，再对 120 对取中位数；不是把动作指令 L2 换单位得到的数值。

| 实际执行轨迹配对 | 配对数 | 每对 p95 距离的中位数 | 连续5个采样点均超过50mm的配对数 |
| --- | ---: | ---: | ---: |
| π0 V14-u50 vs Full-u50 | 120 | 139.6 mm | 109 |
| Xiaomi skip12-D2 vs Full-D2 | 120 | 42.8 mm | 50 |
| Xiaomi skip12-D2 vs 同 mask D5 | 120 | 12.1 mm | 7 |
| DB skip9-m10 vs Full-m10 | 120 | 112.4 mm | 109 |

这里的阈值是 **50 mm＝5 cm，不是5%**。同时间距离混合了路径形状与动作快慢；共同前缀在较短轨迹结束时截断，不能据此说后续误差消失。轨迹差不直接等于失败：Xiaomi 两步的 Full 与 skip12 本批都全部成功，但仍有厘米级累计差异。

下面均为仿真遥测保存的真实 XYZ：蓝色半透明点线为 Full，橙色半透明虚线为变体，前三幅是 XY／XZ／YZ 投影，最后是共同时间距离。灰线是**变体**的阶段转换；密集标签不能替代阶段统计。

![V14新增失败的真实XYZ][xyzpi]

π0 u50，LIBERO-10 task0/init7：Full 成功、V14 失败。图中路径出现明显分叉，同时间距离峰值接近 250 mm，不只是几毫米扰动。对应[Full完整视频][vpfull]、[V14完整视频][vpvar]。

![Xiaomi双成功的真实XYZ][xyzx]

Xiaomi u50，LIBERO-10 task0/init6：两者都成功，轨迹接近但并非完全相同。对应[Full完整视频][vxfull]、[skip12完整视频][vxvar]。

选图规则固定为 suite/task/init 排序的首个“双成功”和首个“Full成功、待比较条件失败”；不存在相应类型时明确留空，不换成别的类型。不是挑最大偏差。全部条件的图与完整视频见[图库][gallery]、[视频索引][media]、[XYZ源索引][trajectory]和[选图记录][selection]。图库的“同 mask 原生步数 vs D2”图按标题识别蓝色原生／橙色少步，不将通用图例字样误当成架构 Full／剪层对照。

### 5.3 失败阶段：不是“35条都死在夹爪”

![只统计有效失败的最远阶段][failurefig]

每配置总计 120 条；下表后三列仅划分该配置的有效失败，未混入成功轨迹。初期／接近是 initial＋approach，接触／抓取是 contact＋grasp_or_manipulate，后期是 transport＋goal_alignment＋release_or_final_action。

| 条件 | 有效失败 | 初期／接近 | 接触／抓取 | 搬运或更后 |
| --- | ---: | ---: | ---: | ---: |
| π0 Full-u50 | 11 | 0 | 2 | 9 |
| π0 V14-u50 | 35 | 8 | 17 | 10 |
| Xiaomi Full-D5 | 1 | 0 | 0 | 1 |
| Xiaomi skip12-D2 | 0 | 0 | 0 | 0 |
| DB Full-D10 | 8 | 0 | 1 | 7 |
| DB Full-D2 | 40 | 4 | 15 | 21 |
| DB skip9-D10 | 21 | 1 | 8 | 12 |
| DB skip9-D2 | 40 | 2 | 12 | 26 |
| DB skip9-m10 | 25 | 2 | 7 | 16 |

V14 有 8 条甚至最远只到初期／接近，不能说只影响精细操作。另一方面，Full 本身的失败也常在后期，不能仅凭“变体失败偏后期”证明跳层特别损害精细操作。这里统计的是**失败任务最远到过哪里**，不是错误首次出现的位置，更不是故障原因。

阶段依据真实物体、接触和抓取遥测的规则标签，缺证据保留 unknown。完整阶段到达数、到达后最终成功数与分母见[阶段统计][stages]；该原表包含成功 episode 的阶段到达／最远阶段，不能不筛 success 就当失败表。上图与本表另按主批次 success=False 过滤。当前数据不足以把失败统一归为夹爪松开或特定精细环节。

## 6. 完成边界、数据入口与下一步

**快速阶段已完成的范围**：6480 主样本、2833 保留补充、1728 同观测输出、22528 测速请求、全 H 动作统计、真实 XYZ／视频索引与汇总图。原 10800 全量计划已按用户指令停止补量，不能写成全部跑完。

独立媒体记录覆盖主样本 6285/6480，其余 195 条尚未独立解码检查，不等于视频丢失。本阶段复用逐 episode worker 验证与已有独立记录，采用代表性原始数据抽查和必要结果复核，**没有重新独立全审所有 raw／视频，也不声称旧 C01–C08 全通过**。

还需保留一项执行历史限制：π0 正式结果混合原冻结 worker、`two_env_v1`、`pi0_prompt_cache_order_v2`，随后回到原 worker；全部有效记录保留，未为此重跑或筛出新的主结果。先前 `pi0_prompt_cache_v1` 的失败 gate 有 8 对中的 3 对 raw 不一致，未获接受且没有正式 episode；后续 order-v2 在双方相同 ABAB 顺序、prompt length 14／16、两个 mask 与两个 seed 的 8 对受控请求中均 exact。**后者只证明该有限门禁范围，不证明所有生产长度／请求顺序／正式轨迹全局逐位等价**，也未证实 compile 为根因。相关记录见[混合执行历史与失败门禁的复核](idea1_v2_2_single_env_rollback_review.md)。因此 π0 成功率是保留真实执行历史的探索性汇总；吞吐加速尝试／回退窗口也不能充当模型延迟的因果证据，第 4 节速度另取已验收的独立 benchmark。

收尾遇到两项分析软件问题：v1 相对／绝对目录表示矛盾；v2 在每条统计记录上重复计算大文件 SHA，造成严重重复读取。旧源及失败／中止 view 保留；v3 保持统计数值、行、分母与证据绑定，只修正路径处理及循环不变量哈希，CPU 重跑不涉及重跑模型实验。具体过程见[执行记录](idea1_v2_2_execution_status.md)与[独立复核记录](idea1_v2_2_single_env_rollback_review.md)。

下一步只建议做小规模确认，不自动开跑：

1. Xiaomi skip12-u50：优先用尚未看过的初态／任务验证天花板下的稳健性，并继续关注轨迹和夹爪风险，而非为 1 条成功排名补大规模评测。
2. DB Full-m25／skip9-m10：保留两个候选，用新样本确认 mask×日程交互，避免再次把原生 D2 起点变化误当纯 NFE 效应。
3. π0：优先寻找质量更好的 mask；现有 V14 仅靠步长没有补回损失。若要判定精细操作致因，应做局部受控干预，而不是继续只看范数或最远失败阶段。

可复查入口：

- [快速阶段完成记录][finalreceipt]、[覆盖与产物检查][completion]、[汇总图输入／输出绑定][figreceipt]、[独立快速阶段结果复核][review]。
- [补充样本独立分母表][extras]；不与主批次混分母。
- [全部图与完整视频入口][autogallery]；CSV 在前述各表链接中。最终 view：`policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z`。

最终阶段记录 SHA256：`69ea4cc6be4fbcb2a4fcc145dbf304eae54164a9b594a453f51ad718322d81a3`。汇总图记录 SHA256：`b623da0de02f16ae4db2789a409a6b5bd8d98634df684e079d6d7562f8f26f62`。链接均相对本文目录，图片／视频保留原位，未复制媒体或修改原始结果。

[heat]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/summary_figures/01_success_heatmaps.png
[latencyfig]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/summary_figures/02_independent_latency_comparison.png
[failurefig]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/summary_figures/03_valid_failure_furthest_stage.png
[actionfig]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/figures/actions_pi0_V14_A03_u50_vs_pi0_full_u50.png
[xyzpi]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/figures/pi0_V14_A03_u50_new_failure.png
[xyzx]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/figures/xiaomi-robotic0_skip12_short_runs_u50_double_success.png
[success]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/success_by_condition_suite.csv
[paired]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/paired_comparisons.csv
[latency]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/latency_breakdown.csv
[speed]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/action_gen_speedup.csv
[actions]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/action_error_by_offset_phase.csv
[errors]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/action_error_summary.csv
[gallery]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/gallery.html
[autogallery]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/gallery.html
[media]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/media_index.md
[trajectory]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/trajectory_index.csv
[selection]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/representative_selection.json
[stages]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/failure_stage_counts.csv
[finalreceipt]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/quick_stage_final_receipt.json
[completion]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/completion_audit.json
[figreceipt]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/summary_figures/reporting_receipt.json
[review]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/medium_quick_stage_review.json
[extras]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/supplement_extras.csv
[vpfull]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/episodes/pi0_full_u50/libero_10/task00/init07/4060595-1792319009431577458/video.mp4
[vpvar]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/episodes/pi0_V14_A03_u50/libero_10/task00/init07/62327-1792324995818475366/video.mp4
[vxfull]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/episodes/xiaomi-robotic0_full_u50/libero_10/task00/init06/3363185-1792291738282191012/video.mp4
[vxvar]: ../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/episodes/xiaomi-robotic0_skip12_short_runs_u50/libero_10/task00/init06/3363185-1792293826058726445/video.mp4

