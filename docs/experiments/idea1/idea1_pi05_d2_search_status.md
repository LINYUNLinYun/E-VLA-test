# π0.5 跳n层＋D2搜索执行状态

2026-10-20。**本轮完成**。[方案](idea1_pi05_d2_search_plan.md)冻结的20条件首筛800条＋480增量＝1280/1280有效，四项同卡计时完成。n6b为149/200（74.5%），对FullD10／FullD2分别1.394×／1.176×；n6a118/200（59%）不合格。本轮D2与旧D10结果独立；65%只约束总体，各suite原样报告。

| 阶段 | 状态 | 证据／下一步 |
| --- | --- | --- |
| 初批6 | complete | 240/240有效、4session48682/73254/88654/45591均exit0／0invalid；各owner60 |
| 固定后13 | complete | 520/520有效、各owner130；session55075/29858/9661/75154均exit0，invalid0，与初批零重复 |
| 唯一奇数 | complete | n7_adaptive=[3,4,6,9,12,15,18]；40/40有效、17成功，四session均exit0 |
| 确认／计时 | complete | 480/480有效，session41976/59061/88092/71554均exit0；4项同GPU0计时全部exit0，正式比率检查通过 |

角色：low唯一代码owner、medium文档／必要验收、Luna Max唯一GPU执行owner。NFS有空间，根overlay约98%满，不要求复制模型到本地scratch；新结果根为`policies/realtime-vla-flash/results/idea1_pi05_d2_search/`。

## 交接记录

- 管理者已读新policy／CLI／server／checks：真实`create_trained_policy(num_steps=2)`并assert实际_sample_kwargs；节点1/.5/0、nativeH10R5；prefix shell保留1/2、suffix真实original scan，未改旧源。
- low CPU检查通过：直接调用原Pi0.sample_actions的runtime callback观察t=[1,.5]／dt=-.5／至0恰两次suffix；tiny18 skip0/6/16 KV形状／完整native suffix exact parity／jaxpr scan；fresh子模型server分派与D2 key。管理者独立复跑dispatch-only及初批manifest，通过240唯一、各40、4owner各60、init6、D2节点显式。
- 已放行初批四卡真实启动。精确attempt为`policies/realtime-vla-flash/results/idea1_pi05_d2_search/pi05_d2_init6_screen_20261020T1340Z`，revision=`pi05-d2-search-native-scan-v1`；执行者启动前13:41:29确认四卡空闲，随后四session均alive、正常task order，无闲卡频扫。
- 首真实Full-D2 spatial/task0/init6已由root／执行者实读valid success，74动作／15预测，实际num_steps2、t[1,.5]、H10/R5／完整18层AE正确；计入首40，不加smoke。`counts.full`仍表示native policy调用，不叫fallback。
- root另实读n2／spatial/task0/init6 valid success，skip[3,9]、实际num_steps2／t[1,.5]、完整native scanned AE18正确，真实跳层＋D2接口亦通过，不新增smoke。
- root冻结后11＋n14／n16合并13条件。管理者独立CPU核520唯一、4owner各130，与初批240 key无交集，合计19语义mask无重复，名称n等于实际skip长度且首2层保留。已通知Luna可立即prepare immutable新清单，首卡自然释放即接，不等完整advance或初批全结束。
- 后13精确attempt为`policies/realtime-vla-flash/results/idea1_pi05_d2_search/pi05_d2_followup_init6_20261020T0544Z`，执行者prepare后再核520唯一／13×40／各owner130，四初批仍live所以未launch后批，不抢占或热改。时间戳说明：初批目录后缀`T1340Z`误用上海本地钟，实际UTC约05:40；保留原目录不改，后13使用正确UTC后缀`T0544Z`，实际运行时间以日志为准。
- 全部后续按冻结规则由medium与Luna直接推进，无需逐阶段等root重批；异常／实质定义变化升级。完整19后才可按规则准备odd，不新增质量门槛或无限fallback。
- **后处理／扩样／计时全链放行**：管理者读advance／summary／bench／compare与README，独立post_checks通过三类排序／去重、整体65而低suite允许、odd固定优先序、语义重筛拒绝、仅init7–10新keys、实际动作成本。Full-D2空skip已从剪枝候选资格先排除，仍独立无条件确认；fixture覆盖Full-D2最快时仍不吞剪枝类别。初批4文件未改。
- 后13无source_attempts链接，后续odd／select／expand／summary必须显式传初6＋后13；若odd只产无manifest决策则不作为source。禁止改冻结清单或重prepare同13。最终只≥130/200总体者进入最高速度／fallback，最大确认n与最快实测点分列；最多3剪枝候选、不填第四。
- 13:46上海本地低频快照：初批68/240 valid标记，owner0–3为19/18/14/17，四session仍live、无invalid；E2 worker无episode events，不能凭仅startup误判未执行。标记快照不是正式全条件完成／成功率。
- **初批240完成**：Luna逐owner核60/60 valid×4、session48682／73254／88654／45591均exit0、invalid0。管理者运行一次初表summary（session80128 exit0），输出初批`summaries/initial_240`，仅新小result/infer：FullD2 39/40、n2 38/40、n3 38/40、n4 37/40、n5 35/40、n6a 26/40。n6a四suite为7／10、10／10、6／10、3／10（spatial/object/goal/10）；整体65%达screen门，但不是各suite门、更非最终200确认。
- 自然接卡后后13四owner均live：GPU0 session55075、GPU1 session29858、GPU2 session9661、GPU3 session75154；首n6b／spatial/task0/init6 valid success，实际D2／H10／R5、skip[3,6,9,12,15,18]、native完整18层AE scan正确。没有接口偏离或额外smoke，不依据初表停后批任务。

报告入口：[D2搜索报告](idea1_pi05_d2_search_report.md)。

## 固定19完成与奇数边界冻结

后13四session均exit0，逐owner核130/130 valid，invalid0，无剩余该attempt worker。与初批240合计760，不重跑。管理者有限CPU调用advance.build(odd)，最高通过总体26/40门的层数为6；同层父n6b为33/40，完整闭环policy成本43.140426718ms/实际动作。按冻结优先序，首个尚未跳层为4，故唯一奇数为n7_adaptive=[3,4,6,9,12,15,18]。已通知Luna双source prepare独立40／4shards并立即自然接卡，不需root再批。该40齐后才冻结最终三类去重候选；FullD2仍独立无条件扩至200。

固定19一次CPU汇总session97337 exit0，输出结果根`summaries/fixed19_760`。odd精确attempt为`pi05_d2_odd_n7_init6_20261020T0655Z`，prepare session52798 exit0；双source、40唯一、owner各10及父／extra4 receipt一致。执行者确认四卡无compute owner后启动四worker session25711/42757/50269/31682；不新增mask或smoke。

## 全20选型冻结

odd四worker均exit0，40/40 valid、invalid0，成功17/40，不达总体65%。三source select一次CPU session80750 exit0，收据为结果根`summaries/final_selection_800.json`。最高passing n的质量点为n6b（33/40、43.140426718ms/动作）；≥65最快点为n6a（26/40、36.371003790）；≥75最快点仍为n6b，故去重只保留两个剪枝候选，不补第三。已放行FullD2／n6b／n6a各补init7–10新增160，共480，建议4shards各120；最终总1280而非填满1440。正式same-card bench仅FullD10／FullD2／n6b／n6a四配置。最终通过仍要求各候选总体≥130/200，不按各suite65筛，也不为确认失败加新候选。

扩样精确attempt=`pi05_d2_expand_selected_init7_10_20261020T0704Z`，prepare session93151 exit0。Luna核480唯一new keys、与原800交集0、每条件160、仅init7–10、三source齐、4owner各120；condition排列不同但集合与selection收据完全一致。启动前四卡无compute owner，四worker session41976/59061/88092/71554已接GPU0–3。自然空卡后以同一物理GPU串行四项正式bench，不额外全网格测量。

## 最终验收／交接

- 扩样四session41976／59061／88092／71554全部exit0，480/480 valid、invalid0；总20条件1280新episode，未用满上限、不补新候选。
- 正式bench位于扩样attempt的`benches/`：FullD10／FullD2／n6b／n6a四目录后缀`G0_20261020T0732Z`，session58417／2017／75448／62303均exit0。每项20warm＋100measure、500虚拟动作，同GPU0 UUID／四输入SHA／线程／revision一致，模型load单列。bench_compare实际exit0，结果根`summaries/same_card_ratios.csv`四组正式比率门通过；JAX trace未知、分项未测，不能称绝对纯热或旧D10profiler分项。
- 最终一次summary session79346 exit0，结果根`summaries/final_1280`：FullD2 196/200，n6b149/200，n6a118/200；其余17条件各40，合计1280。每suite／40与160分段均独立保存，原生实际D2协议检查通过。
- 最高确认n为6；合格正式测速最高点n6b对FullD10 1.394254887×、对FullD2 1.176046123×。n6a更快但总体59%不合格，不进入最高结论、不追加候选。最终门是观察总体≥130/200，不要求每suite65%。
- Luna确认本轮不再派GPU；科学和bench句柄均已退出。文档与表格交付，旧idea1／旧源／旧结果未改，不新增图片或重验旧视频。
