# idea1 v3 补充实验最终报告

2026-10-20。**1680条唯一有效episode全部完成**：E1为16×40＋2×160＝960，E2为10×40＋2×160＝720；7个正式同卡计时配置已齐，运行已收尾。E1最终Q为179/200（89.5%）、F为173/200（86.5%），**两个正式计时点均未达到90%观测质量档**；F相对Full-D10最高1.588×，但相对Full-D2仅0.788×、没有加速。E2跳4为187/200（93.5%）与1.121×，跳5为177/200（88.5%）与1.125×。

[最终1680质量／成本表](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/summaries/final_1680/overall_quality_cost.csv)／[汇总收据](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/summaries/final_1680/summary_receipt.json)／[方案](idea1_v3_supplement_plan.md)／[执行状态](idea1_v3_supplement_status.md)。本报告独立于[原v3终稿](idea1_v3_final_report.md)，不改写原1976条数据，不补跑旧Full成功率。

本轮直接闭环搜索tau／周期／夹爪规则／拒绝Full-D2与π0.5跳4／5层，所有冻结条件首40完整保留、没有离线筛选或按中间输赢停掉正常任务。最终扩样只补init7–10；质量与速度是不同测量边界，最高只限正式测速的两个入选点，而非全局最优。

## E1-A首40：严格规则下tau／周期搜索

各条件init6、四suite各10条。成本为完整policy wall累计／实际动作，含被拒draft＋verify＋Full及首次形状准备；模型startup另列。不同批次共享负载／revision限制保留，此列用于预定筛选，不是正式加速比。

| tau | P | 成功数 | ms／实际动作 |
| --- | --- | ---: | ---: |
| .10 | 2（旧复用） | 37/40 | 27.696 |
| .15 | 2（旧复用） | 37/40 | 30.939 |
| .10 | 4 | 35/40 | 26.438 |
| .10 | 8 | 31/40 | 30.775 |
| .25 | 4 | 33/40 | 26.471 |
| .25 | 8 | 31/40 | 25.772 |
| .25 | none | 30/40 | 27.374 |
| .40 | 4 | 33/40 | 25.740 |
| .40 | 8 | 31/40 | 25.434 |
| .40 | none | 30/40 | 25.638 |

A的新配置未达到90%档；更长／无周期未显示首40质量改善。按冻结规则选择旧τ=.10/P2为质量父，τ=.40/P4为≥80%档最快父，分别测试confirmed／off。首40小样本不能证明普遍趋势；不因P8或none当前质量较低补新网格。

来源：[A冻结清单](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e1_phase_a_init6_20261020T0302Z/manifest.json)；旧共同40来源为[原E1-main](../../../policies/realtime-vla-flash/results/idea1_v3/e1_main_init6_20261019T1514Z/manifest.json)，只取init6且排除calibration，不用旧200混排名。

B的[冻结选型收据](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e1_phase_b_init6_20261020T0349Z/selection.json)及[4条件160清单](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e1_phase_b_init6_20261020T0349Z/manifest.json)已保存；现已160/160有效完成，C按其结果冻结。

## E1-B：夹爪规则首40

| 父tau/P | 规则 | 成功数 | ms／实际动作 |
| --- | --- | ---: | ---: |
| .10/P2 | confirmed_disagreement | 37/40 | 27.813 |
| .10/P2 | off | 37/40 | 29.376 |
| .40/P4 | confirmed_disagreement | 33/40 | 26.666 |
| .40/P4 | off | 33/40 | 23.880 |

B160/160有效。放宽夹爪规则没有改变这两个父的首40成功数；同质量下按成本选τ=.10/P2的confirmed与τ=.40/P4的off进入C，同时保留各父strict做仅拒绝D2比较。off在一个父上较慢、另一个较快，单批共享负载成本不能替代正式同卡比率，也不能推断某条夹爪规则普遍更优。

confirmed只在空间已接受前缀中两个K同一步sign均反draft时截断，无旧切换／pending；off关闭夹爪否决／截断。各40均按原timeout完成，正常失败未重跑。

C的[冻结选择](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e1_phase_c_init6_20261020T0410Z/selection.json)与[160清单](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e1_phase_c_init6_20261020T0410Z/manifest.json)已落，160/160有效完成；正式同卡计时另节，不把拒绝D2的预期便宜直接当部署加速。

## E1-C：仅拒绝接管Full-D2的首40

| tau/P | 夹爪规则 | 成功数 | ms／实际动作 |
| --- | --- | ---: | ---: |
| .10/P2 | strict | 38/40 | 27.668 |
| .10/P2 | confirmed | 38/40 | 26.413 |
| .40/P4 | strict | 34/40 | 23.574 |
| .40/P4 | off | 33/40 | 21.553 |

C160/160有效。真实日志已核startup／periodic仍D10、gripper／zero_accept接管D2且不执行被拒draft。相对各父rejectD10，首40成功数有0–1条变化，不将此小差异当独立确认；完整请求成本也包含拒绝前已花的draft／verify。

联合16新条件，按预定规则扩样**τ=.10/P2 confirmed＋rejectD2**（选择集≥90%档最快、38/40）和**τ=.40/P4 off＋rejectD2**（选择集≥80%档最快、33/40），各只补160到200。前者的init7/task0四suite连续真实predict输入固定为共同bank，两个Full及两候选都用此bank计时。选择集标签仅用于冻结候选；现扩样与计时均完成，正式质量档按最终89.5%／86.5%重归。

[最终选择收据](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e1_phase_c_init6_20261020T0410Z/summaries/e1-selection.json)与[增量320清单](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e1_expand_init7_10_20261020T0431Z/manifest.json)已保存；4静态分片各80，先GPU2／3运行owner0／1，后续自然接卡。采集仅原定扩样中的真实predict输入，不新增质量episode。

E1测速的[共同长bank有限核验收据](idea1_v3_supplement_bank_receipt.json)已通过：按首40选择的优先90%候选init7/task0四suite真实输入，spatial17／object25／goal25／libero_10 25，共92点；逐文件hash、完整来源及真实infer seed顺序一致。20次warm后每suite路由reset再连续重放，不把短spatial补到25；四项计时均已使用同一有序bank、GPU3及安全后端完成。

### E1同卡长bank：完整路由成本

四项已真实完成、比较门通过：GPU3 UUID `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`、相同bank／顺序／seed／线程／revision，每配置20warm＋92测量／460个virtual replay动作。本次均R5，因此每请求比恰等于单位动作比；主式仍使用累计实际返回长度，不能推广为任意候选都可忽略短前缀。

| 条件 | ms／请求 | ms／重放动作 | 对Full-D10主比 | 对Full-D2次比 | 模型加载／warm累计s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-D10参考 | 115.046 | 23.009 | 1.000× | — | 20.143 / 10.865 |
| Full-D2参考 | 57.077 | 11.415 | 约2.016× | 1.000× | 21.459 / 7.925 |
| Q：.10/P2 confirmed／rejectD2 | 76.674 | 15.335 | **1.500×** | **0.744×** | 20.241 / 12.201 |
| F：.40/P4 off／rejectD2 | 72.428 | 14.486 | **1.588×** | **0.788×** | 21.347 / 12.649 |

**相对Full-D2没有加速。** F比Full-D2每动作慢约27%，Q慢约34%；不能只选择D10参照报“最高加速”。参考Full-D2本身约2.016×于D10，但不是新增验证器候选、其成功率仍为旧历史来源，不能用本轮计时冒称新Full-D2质量评测。Q/F质量档按下节最终200重新归类，不沿用首40的95%／82.5%。

测量中Q为4次startup Full、24次periodic Full、7次zero_accept即时Full-D2、57次接受draft；F为4次startup、15次periodic、73次接受draft。成本包含被拒draft＋verify＋Full，不只取接受请求。**F此bank无拒绝事件，不能把其1.588×独立归因于D2 fallback**，它仍含tau／P／夹爪策略的整体差异。Q/F入口Full KV年龄分别为rounds1–3／1–5、env steps5–15／5–25；年龄是当次Full刷新前，P仍按draft轮数2／4。长序列确实触发周期，不以五点重置微基准冒充覆盖。此bank未观测夹爪拒绝，不能据此证明全部夹爪分支的速度；真实闭环路由另表。

四项own-QK参考检查均passed、unsafe_legacy=false；测量期记录runtime setup为0，冷加载／warm在表中分列。重放ack是virtual，bank状态仍来自Q原闭环，其他候选不改变后续bank观测；共享CPU／NFS背景未控制，不能把该比率叫部署闭环加速或跨任务保证。

证据：[四比率CSV](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/summaries/e1_same_card_ratios.csv)、[Full-D10计时](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/bench_e1_fulld10_20261020T1249Z/benchmark_summary.json)、[Full-D2计时](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/bench_e1_fulld2_20261020T1250Z/benchmark_summary.json)、[Q计时](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/bench_e1_confirmed_tau01_p2_rejectd2_20261020T1251Z/benchmark_summary.json)、[F计时](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/bench_e1_off_tau04_p4_rejectd2_20261020T1252Z/benchmark_summary.json)。最高仅限这两个正式计时入选点，另外14个首40探索条件没有同卡正式比率。

### E1完整200质量确认与真实路由

| 候选 | 选择集init6 | 新增init7–10 | 完整200 | spatial/object/goal/10（各50） | 闭环ms／实际动作 |
| --- | ---: | ---: | ---: | --- | ---: |
| Q .10/P2 confirmed／rejectD2 | 38/40 | 141/160 | **179/200（89.5%）** | 46/50、49/50、46/50、38/50 | 22.053 |
| F .40/P4 off／rejectD2 | 33/40 | 140/160 | **173/200（86.5%）** | 45/50、50/50、39/50、39/50 | 18.656 |

Q从首40的95%降至89.5%，不能四舍五入归入90%档；F最终86.5%归85%与80%档。旧严格τ=.10/P2为185/200（92.5%）独立复用，不用它的质量替新策略填90%档；本次放宽／廉价接管整体未改善该旧质量点。Q/F二项Wilson95区间分别约[84.5%,93.0%]／[81.1%,90.6%]，不是task聚类配对因果区间，也不把观测档位当总体成功率保证。

| 完整200路由 | Q | F |
| --- | ---: | ---: |
| 全请求／实际动作 | 7309 / 35934 | 7548 / 37402 |
| startup Full-D10 | 200 | 200 |
| periodic Full-D10 | 1983 | 1383 |
| zero_accept Full-D2 | 587 | **0** |
| gripper Full-D2 | 17 | **0** |
| 已接受draft请求 | 4522 | 5965 |

Q接受draft中94次只接受1–4步，实际执行长度另受success／timeout截断；闭环成本一律除真实动作，不能默认所有请求R5。F的5965次draft均空间接受5步，**在完整200及本次重放均没有拒绝接管**：rejectD2字段在F实际没有触发，不能把较低成本独立归于便宜fallback；整体表现更接近较松阈值、无夹爪保护、周期Full的混合方案。这里Full与verify可同请求发生，不把调用counter累加当独立请求数。

来源：[最终分段表](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/summaries/final_1680/success_by_segment.csv)、[逐suite](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/summaries/final_1680/success_by_condition_suite.csv)、[完整路由](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/summaries/final_1680/routing.csv)。闭环policy成本含冷形状准备及所有被拒开销，模型startup另表；共享负载下不能直接对旧批次成本给因果速度比。

## π0.5：全部10个新mask首40

全部400/400有效，原生H10／D10／R5，prefix原索引KV shell、AE suffix真实native scan。以下成本含首请求编译／输入、共享负载，失败任务动作更多也会改变每动作分母；不把低质量条件较低此成本解释为部署加速。

| 跳层数 | mask | 跳过的原层号 | 成功数 | ms／实际动作 |
| --- | --- | --- | ---: | ---: |
| 4 | spread_a | [3,6,10,14] | 39/40 | 37.744 |
| 4 | spread_b | [4,8,12,16] | 35/40 | 35.220 |
| 4 | transfer | [3,9,11,15] | 39/40 | 36.198 |
| 4 | early_cluster | [3,4,5,6] | 18/40 | 29.009 |
| 4 | mid_cluster | [9,10,11,12] | 20/40 | 29.200 |
| 5 | spread_a | [3,6,9,12,15] | 32/40 | 31.838 |
| 5 | spread_b | [4,7,10,13,16] | 18/40 | 27.725 |
| 5 | transfer | [3,9,11,14,17] | 35/40 | 32.544 |
| 5 | early_cluster | [3,4,5,6,7] | 8/40 | 27.056 |
| 5 | mid_cluster | [8,9,10,11,12] | 8/40 | 27.008 |

跳4的transfer与spread_a同为39/40，按较低每动作成本选transfer；跳5选最高成功数35/40的transfer，而非更快但质量较差者。每个仅补init7–10的160至200。首40中连续簇质量明显较低，但spread_b跳5也仅18/40，不能简单概括“非连续就好”或只由层数推断质量。旧V14四层33/40、Custom五层32/40仅复用质量参照，不重跑。

来源：[10mask冻结首筛清单](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e2_masks_init6_20261020T0301Z/manifest.json)。选择按预定质量优先规则，40与追加160已在下节分列；正式同卡比率独立列于计时节。

[冻结选择收据](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e2_masks_init6_20261020T0301Z/summaries/e2-selection.json)已保存，[增量320清单](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e2_masks_expand_init7_10_20261020T0401Z/manifest.json)只含init7–10，现已320/320有效完成，未重跑源40。

### 两选中mask完整200确认

E2全部720条唯一有效已完成（10×40＋2×160）。纯CPU一次[合并质量／成本表](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e2_masks_expand_init7_10_20261020T0401Z/summaries/all_720_cpu/overall_quality_cost.csv)与[分段表](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/e2_masks_expand_init7_10_20261020T0401Z/summaries/all_720_cpu/success_by_segment.csv)已生成。

| 选中mask | 选择集init6 | 新增init7–10 | 完整200 | spatial/object/goal/10（各50） | ms／实际动作 |
| --- | ---: | ---: | ---: | --- | ---: |
| 跳4 [3,9,11,15] | 39/40 | 148/160 | **187/200（93.5%）** | 45/50、50/50、46/50、46/50 | 25.953 |
| 跳5 [3,9,11,14,17] | 35/40 | 142/160 | **177/200（88.5%）** | 46/50、50/50、43/50、38/50 | 24.355 |

跳4首40的97.5%降至完整93.5%；跳5完整88.5%，主要较弱在libero_10。旧本轮前阶段π0.5 Full为196/200、跳2为196/200、跳3为188/200；本次跳4／5不是等质量替代。这里旧质量参照直接复用，AE编译结构／revision差异保留，不把这些成本直接相除作正式加速。下节同卡新Full与两mask计时采用统一native suffix，不把扩样后冷编译摊薄导致的成本下降归于新算法改进。

### π0.5同卡、同输入的正式重放计时

三组均GPU0 UUID `GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`，同P0真实20点／顺序／seed、同benchmark revision与线程环境；每prompt先5次warm、再5轮同5点，合计20warm＋100测量／500次virtual replay动作。新mask的suffix均原生scan，Full为真实native Full；不存在原轮masked AE展开的总量混杂。JAX重编译未独立计数，不凭startup事件字段0就称100次完全纯热态。

| 条件 | 新测量均值ms/请求 | ms/重放动作 | 相对本轮Full单位动作加速 | 模型加载s | 20次warm请求累计s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 102.105 | 20.421 | 1.000× | 24.538 | 22.149 |
| 跳4 [3,9,11,15] | 91.081 | 18.216 | **1.121×** | 24.233 | 54.402 |
| 跳5 [3,9,11,14,17] | 90.733 | 18.147 | **1.125×** | 25.689 | 50.647 |

按最终200实测质量，90%档内已正式测速者最快为跳4（93.5%，1.121×）；85%／80%档内数值最大为跳5（88.5%，1.125×）。但两mask均值仅差约0.35ms／请求（约0.4%），不足以在共享CPU／温度等未控制背景下证明跳5稳定更快；跳5相对跳4损失5个百分点质量，跳4是更稳妥的后续候选。未对全部10mask逐一独立测速，不能将这两个计时点称全局最优。

模型加载与warm／首编译成本显著，短任务不保证端到端获益；warm累计为原predict计时，不含另行四次diagnostic split-stage检查的额外工作，不能将表中三列简单相加冒称完整作业墙钟。三组native stage checks各4/4在预定atol／rtol=.02内通过（最大差分别.001953／.002441／.003234），分项独立编译／同步，不相加替代native fused总延迟。

证据：[同卡比率CSV](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/summaries/e2_same_card_ratios.csv)、本轮[Full计时](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/bench_e2_full_20261020T1233Z/benchmark_summary.json)、[跳4计时](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/bench_e2_s4_transfer_20261020T1235Z/benchmark_summary.json)、[跳5计时](../../../policies/realtime-vla-flash/results/idea1_v3_supplement/bench_e2_s5_transfer_20261020T1238Z/benchmark_summary.json)。质量来自上节720表、旧Full成功率另引原v3，不把本轮仅计时的Full冒称新闭环质量评测。

## 最终质量档与Pareto

质量用入选者各200条观测成功率，不沿首40标签；速度只在各模型两个正式计时入选点比较，不混入仅40探索点或旧协议测速。

| 模型 | 质量门槛 | 本次正式测速点中的最快合格者 | 最终质量 | 对对应Full-D10 | 对Full-D2 |
| --- | --- | --- | ---: | ---: | ---: |
| π0验证器 | ≥90% | **无** | Q89.5%、F86.5%均不达 | — | — |
| π0验证器 | ≥85% | F .40/P4 off／rejectD2 | 173/200（86.5%） | 1.588× | 0.788× |
| π0验证器 | ≥80% | F .40/P4 off／rejectD2 | 173/200（86.5%） | 1.588× | 0.788× |
| π0.5 | ≥90% | 跳4 [3,9,11,15] | 187/200（93.5%） | 1.121× | 不适用 |
| π0.5 | ≥85% | 跳5 [3,9,11,14,17] | 177/200（88.5%） | 1.125× | 不适用 |
| π0.5 | ≥80% | 跳5 [3,9,11,14,17] | 177/200（88.5%） | 1.125× | 不适用 |

Q质量高于F、速度较低，二者仍是质量—成本非支配点；≥85%／≥80%档的最高速度选F，不表示Q没有用途。π0.5跳4与跳5也分别体现质量／成本取舍，但0.4%测量速度差不能当稳定收益，优先考虑质量较好的跳4。不同backend的绝对毫秒不作模型优劣排名。

结论：本次有界搜索没有找到达到90%观测质量且比Full-D10快的新π0验证器确认点；85%档最快为F的1.588×，但对Full-D2仍慢，不能称两基线均加速。扩大tau、减少Full或放宽夹爪没有带来已确认的无损替代；较便宜Full接管也不能与其他改动独立归因。π0.5的跳4保留93.5%质量、约1.12×重放收益，比跳5仅快约0.4%却损失5pp质量更值得后续验证。全部16＋10首筛结果保留，不为得到更好结论追加第三扩样候选。

运行收尾：Luna确认科学四owner均exit0、各80有效，扩样320/320；7计时配置均exit0，own safety／比较门通过。一次最终CPU summary session61339 exit0，26条件／1680有效（E1960、E2720），没有新GPU任务待派；owner锁文件保留但无活动flock。本轮一次`.json`误传pickle loader的技术启动错误发生于模型／GPU前，纠正后未增加测量预算，不计科学失败。旧结果／视频／raw未重验或覆盖；本补充不新增图片代码，有限表格作为可视化交付。

## 对照与“最高加速”的证据边界

旧质量参照从[原v3终稿](idea1_v3_final_report.md)与其原始索引引用：π0 Full-D2为109/120、纯V14-D2为85/120（旧共同init6–8）；Full-D10为180/200、纯V14-D10为148/200（旧init0–4，与本补充init6–10不配对）。π0.5 Full为原轮196/200；旧严格验证器τ=.10/P2为185/200。不同初态、backend／revision不合并成新成功率分母，不把历史结果伪称本轮Full闭环，也不拿旧参照质量替新Q/F填90%档。

本轮同卡Full-D10、Full-D2与最终候选计时的输出路径／GPU UUID／revision已在两模型计时节列出。主比为单位动作成本比，每请求比次列；候选重放使用执行长度累计和virtual ack，不冒充真实闭环执行。最大值仅称**各模型本次两个正式测速入选点、给定质量档与计时协议下的最高实测加速**，不是全局最优或质量无损保证。

冻结的10个π0.5 mask全部直接实测完整40，正常科学失败不会导致中途淘汰或停派；扩样选择只在规定首40覆盖齐后进行。
