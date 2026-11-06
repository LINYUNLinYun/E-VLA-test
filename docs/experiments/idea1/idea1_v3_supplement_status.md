# idea1 v3 补充实验执行状态

更新：2026-10-20。**本补充既定范围全部完成**：1680条唯一有效episode（E1960／E2720）、26条件、7配置正式同卡计时、一次最终CPU汇总齐，所有本轮运行与活动owner flock已退出。E1 Q179/200／F173/200均未达90%观测档；E2跳4 187/200、跳5 177/200。结果见[最终报告](idea1_v3_supplement_report.md)，原v3完成状态／1976结果不变。

| 阶段 | 当前状态 | 冻结范围／下一事件 |
| --- | --- | --- |
| E2首筛／扩样 | complete（720有效） | s4_transfer 187/200、s5_transfer 177/200，40／160分表；三同卡bench完成且比率门通过 |
| E1-A | complete | 320/320唯一有效，session26367／90111均exit0；8条件各40，结果已用于冻结B父 |
| E1-B/C | complete | B160/160、C160/160有效，两stage所有session exit0；16新条件首40已齐 |
| E1扩展 | complete（320有效） | 4owner各80有效、均真实exit0；合并源640共960；Q179/200、F173/200，40／160分段 |
| 同卡速度／Pareto | complete | E1观测90档无合格，85／80档F为1.588×于D10、0.788×于D2；E2 90档s4 1.121×，85／80档s5 1.125× |

角色：GPT-6.1-sol low唯一实现owner；medium管理／必要验收与文档；Luna Max唯一GPU执行owner。四卡空闲为一次readiness观察，实际启动前仍由Luna核实绑定，不预占外部卡；已有作业用完成事件调度，无空卡且有待派才30秒查询。

## 交接与验收记录

最终验收：Luna核四个E1扩样session16428／96095／30827／48327均exit0、各owner80逐key有效、无invalid；320/320齐，owner锁文件保留但无held flock，无本轮活动科学／bench进程。管理者一次最终summary session61339 exit0，读取两个expand根递归新源、26条件总1680（E1960／E2720），没有旧raw／视频复验。Q首38＋新增141＝179/200（89.5%），F首33＋新增140＝173/200（86.5%）；最终档位已重归，未追加第三候选。正式same-card两组CSV／7bench safety与比较门过，完整探索表／Pareto／成本解释见终稿；无待派实验。下列按时间保留过程快照，其中“待／running”不代表当前状态。

- 管理者已读现有Policy的Full／draft／verify／ack、严格夹爪符号切换及pending、π0.5 prefix shell／AE suffix路径、QK安全安装、补充CLI／server；原Policy只允许冻结旧网格，补充必须独立实现，不能热改旧入口。
- 新E2 CLI显式mask写科学config／key、四suite/task0–9/init6、静态owner、checkpoint/norm identity；worker使用原锁／持久化流但fresh eval与model子进程入口均替换为补充包。native H10、D10、R5；prefix skip保留原层号KV shell、suffix回真实original scan。代码窄审未见旧Full质量重跑；CPU tiny parity／真实首条待验。
- low已收到root冻结全部10mask及A8/B4/C4语义；执行者收到先E2、E1就绪可2＋2卡的交接。不等全模块、不离线淘汰候选、不扩大网格。
- **E2启动放行**：low在openpi venv以CPU tiny18通过4／5层含相邻mask的suffix exact original parity、jaxpr scan、prefix／KV稳定；dispatch mock确认子模型模块替换新server。我独立CPU验证冻结10mask manifest＝400唯一key、init6 only、各40、2owner各200、无Full质量任务。已向Luna发真实prepare／worker入口并放行；真实接口尚待首条，不把CPU通过当成功率结果。
- **E1-A启动放行**：新e1_cli／e1_server／Policy入口源码窄审通过，真实模型子进程先安装原passed安全QK。管理者独立复跑e1_checks通过strict旧fake差分、none禁周期／P按draft轮、各Full原因D、confirmed/off、KV独立／拒绝不推进／ack／reset；冻结manifest为8条件320唯一key、init6、strict／rejectD10，none两条件。已交Luna另两卡双分片启动，不等计时入口；尚待实际QK与首episode证据。
- 速度口径经root冻结为单位实际动作成本比，每请求比次列；计时最多7配置，pureV14不另测，suite／prompt外层仍20setup＋100测量，不追加热态预算。
- **E1-A真实启动**：独立attempt `e1_phase_a_init6_20261020T0302Z`，320唯一key／8×40／2owner各160；GPU2 shard0 session26367，GPU3 shard1 session90111，沿用passed QK guard。E2 session24521／34539仍占GPU0／1；四路已启动，首真实episode／revision／绑定由执行者核验后补记，不将启动当结果。
- E2精确attempt为 `policies/realtime-vla-flash/results/idea1_v3_supplement/e2_masks_init6_20261020T0301Z`，四卡均为本方对应worker持有。E1 startup已进首episode_start，完成证据待执行者报。
- **长序列计时修正**：旧5点reset不足覆盖P8，root已让low在未来自然选中扩样init7/task0每suite原闭环最多记录25连续输入；最多20warm＋100measure预算不增，候选／两Full用同一冻结序列，短episode不补造。E2可旧5点输入。此为尚待实现／测量的协议，不追溯重跑当前首筛。
- **E1-A首真实接口通过**：执行者核首episode valid／success，H50／R5、τ=.10/P4、strict／reject10；qk-row-bounds-v1参考receipt已生成，构造startup full_AE_calls=0、首Full为D10／startup。此只验证首路径，none与新G／rejectD2仍待自然阶段实测，不据一条success做质量结论。
- **E2首真实接口通过**：root实读s4_spread_a／spatial task0 init6 complete valid／success，80动作／16预测、H10／D10／R5、native scanned suffix协议正确。branches.worker没有episode_start／complete事件，完成以result／complete与profile为准；只有startup日志不能推断未运行。
- E2扩样规则已在结果出齐前冻结：每个skip数从≥80%首40者按最高成功数、最低policy_ms／实际动作、稳定mask名选至多1；无达标不扩。E1继续质量档内速度Pareto规则，二者不混用。
- **后续独立入口放行**：管理者源码窄审advance／summary／capture_worker／bench／bench_compare并独立复跑post_checks通过：共同40父排序、init7–10仅160且同task相邻owner、捕获最多25个实际调用不加预测、sum毫秒／实际动作、保留合法失败、不同bank拒ratio。自然阶段满足完整覆盖后可直接prepare新manifest，不再等root逐阶段确认。E1扩样必须capture_worker；真实长bank／same-card速度接口仍待实际测量，不能凭此CPU通过宣称加速。
- 03:21 UTC低频快照：E2有151/400个complete标记、E1-A137/320；这是执行者轻量标记数，未逐个重读valid，不据此做正式成功率或提前选型。四session仍在运行、无已报退出／异常；全部冻结条件继续，不因途中输赢停正常任务。
- **A完成及B冻结**：Luna确认A两session26367／90111均exit0，320唯一complete均valid。管理者只读advance.build联合A与旧E1-main，显式init6／排除calibration复算。质量父为旧τ=.10/P2（37/40，27.695982ms/action）；速度父新τ=.40/P4（33/40，25.740075ms/action）。旧τ=.15/P2同37但更慢30.939426；P8最快τ=.40为31/40，低于80%不入速度父。已放行B两父各confirmed/off共160、2shards在GPU2／3接卡，不等E2；selection.json由执行者prepare冻结留存。
- **B真实启动**：`e1_phase_b_init6_20261020T0349Z`，selection.json与manifest父／成本一致，160唯一key、4×40；GPU2 shard0 session28094，GPU3 shard1 session64654，fresh e1_cli worker复用passed QK guard。执行者启动前单卡确认无compute进程；首confirmed/off真实行为待核，不将启动当质量结果。
- **B首正常接口通过**：confirmed_disagreement／τ=.10/P2／reject10，首task0/init6 valid success；H50／R5／D10、18层与router revision正确，实际QK参考passed且SHA匹配。16请求中Full6（startup1／periodic5）、verify10，R5；该例三个gripper布尔事件均未触发，因此仅证明新配置正常路径，不声称实际触发了夹爪截断，也不为此加smoke。
- **E2首筛完成／扩样冻结**：Luna确认session24521／34539均exit0、400/400唯一valid，10mask各40全部保留。管理者只读advance.build复算选择4层s4_transfer [3,9,11,15]＝39/40、36.197726ms/action（spread_a同39但37.744476较慢）；5层s5_transfer [3,9,11,14,17]＝35/40、32.543992，质量最高。已放行新320清单仅init7–10、2shards在GPU0／1接卡，原40引用；B两卡不动。稳定选择收据由执行者保存，不重跑旧Full／V14／init6。
- **E2扩样真实启动**：`e2_masks_expand_init7_10_20261020T0401Z`，selection收据为首筛`summaries/e2-selection.json`，320唯一key、init7–10 only、两condition各160、两owner各160；GPU0 shard0 session12644、GPU1 shard1 session1724。启动前无compute进程，无重跑init6；最终选中条件分母200需引用源40，当前不能称已完成200。
- E2扩样首task0/init7已由执行者实核valid success，s4_transfer／原生H10D10R5／native suffix正确；首请求约56s含cold编译，不当热态延迟。E2的counts.full仍仅native policy调用，不能解释为fallback。
- **B完成及C冻结**：session28094／64654均exit0，160唯一complete全部valid。管理者只读advance.build复算：τ=.10/P2的confirmed/off同37/40，confirmed成本27.813083低于off29.376079；τ=.40/P4两relaxed同33/40，off23.879800低于confirmed26.665833。C按预定规则取两父strict与各自best relaxed共4×40，rejectD2（含strict pending），startup／periodic仍D10；已放行GPU2／3接卡。E2扩样此时99/320标记，仍在运行，不重分其清单。
- **C真实启动**：`e1_phase_c_init6_20261020T0410Z`，selection.json核对best relaxed／成本一致；160唯一key、rejectD2、4×40，GPU2 shard0 session40699、GPU3 shard1 session93101，安全gate沿用passed。实际首拒绝Full-D2路径尚待自然日志核，不增加smoke。
- **C真实拒绝路径核验**：首task0/init6 valid success，QK reference passed；round0 startup Full10，round7 gripper Full2与round12 zero_accept Full2均accepted_length=0，执行最终Full而非被拒draft；protocol列startup／periodic10、其余拒绝2。没有为触发此路另加smoke。
- **C完成与最终E1选择冻结**：session40699／93101 exit0，160唯一valid。管理者联合A/B/C只读复算全16条件：≥90%档最快为τ=.10/P2 confirmed／rejectD2，38/40、26.413209ms/action；≥80%档最快τ=.40/P4 off／rejectD2，33/40、21.553364。已放行这两条件仅补init7–10共320、新4shards各80，先在GPU2／3接owner0／1，E2释放后接未跑owner，不重分旧清单。E1必须capture_worker；同一优先90%候选init7/task0四suite的真实连续bank作两Full／两候选共同测速输入。当前E2扩样308/320标记，尚未正式完成。
- **E1最终扩样真实启动**：`e1_expand_init7_10_20261020T0431Z`，选择receipt为C`summaries/e1-selection.json`，320唯一key、init7–10、4静态owner各80。GPU2 owner0 session16428、GPU3 owner1 session96095，均capture_worker＋passed QK gate；owner2／3待自然释放卡，不改清单。每suite捕获最多25个原闭环实际predict，不补造短episode或额外预测。
- **E2全720完成**：session12644／1724均exit0，320增量全部valid，合并源首400共720。管理者运行一次纯CPU新summary（session65753 exit0）到`e2_masks_expand_init7_10_20261020T0401Z/summaries/all_720_cpu`，无XYZ／视频／GPU。s4_transfer首39/40＋追加148/160＝187/200；s5_transfer首35/40＋追加142/160＝177/200，均按预定质量优先选型后确认。已授权GPU1接E1owner2，GPU0同卡串行3个既定E2bench，随后未跑owner3接自然空卡，不重跑质量对照。
- **E2同卡计时启动**：GPU0确认无compute进程后，Full参考`bench_e2_full_20261020T1233Z` session93506已真实启动，新bench／pi05-check、原P0固定真实输入；后续s4/s5必须同GPU／bank／线程／revision。当前仅启动，不引用旧bench代替本次正式比率。
- E1扩样owner2已在GPU1用capture_worker接起，session30827；与GPU2／3 owner0／1并行，不等待E2三组计时。owner3仍ready，将接下一自然释放卡，不重复owner或重分片。
- **E2三同卡bench完成**：Full session93506、s4 session8896及s5均exit0，GPU0同UUID／input hashes／线程／revision，各20warm＋100测量／500virtual动作。管理者实际bench_compare通过并生成`summaries/e2_same_card_ratios.csv`：Full20.420951ms/动作，s4 18.216118→1.121037×，s5 18.146605→1.125332×；三组native-stage checks各4/4过预设.02/.02。模型加载／warm分列，JAX重编译未独立计数；此为重放非闭环部署速度。与最终质量187／177各200结合，90档s4，85／80档数值最快s5但二者仅约0.4%速度差，不作稳定优劣结论。
- GPU0随后启动E1 owner3／shard3，session48327，capture_worker＋同gate；四owner分别GPU2／3／1／0，无重复owner，最后320科学扩样继续。
- **E1共同长bank及空卡门通过**：owner1 session96095 exit0，80逐key有效／0 invalid、GPU3无compute进程。管理者独立实读4个新pkl与metadata/result/complete：source为优先90%候选τ=.10/P2 confirmed／rejectD2，init7/task0，key／valid／SHA匹配，seed逐项等于实际infer前缀。固定顺序spatial17／object25／goal25／libero_10 25，共92 measurement／20warm，不补短例；[有限收据](idea1_v3_supplement_bank_receipt.json)已落。已放行GPU3四E1bench（两Full＋两selected）同bank／顺序／seed／线程／安全revision；其他science继续，实际ratio待四run比较门。
- **计时启动前技术错误**：首次FullD10命令误把`.json`传给pickle loader，load_inputs即UnpicklingError，尚无模型／GPU／benchmark数据、不消费测量预算。已按源码诊断只需四个同名`.pkl`（模块自动读`.json`验证）；允许保留命令错误日志、fresh输出目录纠正一次，不改代码／输入bank／科学参数，不计科学invalid。
- **E1四项计时齐／比较门通过**：纠正后的FullD10 session79173、FullD2 session30007、Q session85040及F全部exit0；GPU3同UUID／bank hashes／顺序／线程／revision，每项20warm＋92测量／460virtual动作。管理者实读4个ownQK passed／unsafe=false，并运行bench_compare生成`summaries/e1_same_card_ratios.csv`：D10/D2参考23.009103／11.415458ms/动作，Q15.334884→1.500442×／0.744411×，F14.485691→1.588402×／0.788051×。周期Full实测Q24／F15次，入口KV年龄rounds1–3／1–5；费用包含拒绝Full-D2。7配置测量预算已齐，不再bench；最终质量档等E1科学200重新归档，不沿用选择集95%／82.5%。

结果和技术异常将在实际启动／完成事件后补记；旧v3无待恢复队列。本补充报告入口：[补充报告](idea1_v3_supplement_report.md)。
