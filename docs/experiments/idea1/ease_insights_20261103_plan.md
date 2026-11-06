# EASE 两条 insight：预注册实验与执行记录

日期：2026-11-03。目标是检验假设，接受支持、不支持或证据不足的结果；不以获得正结果为停止条件。

## 1. 固定训练 draft 的跨 suite 泛化

固定 π0 LIBERO target 权重、normalization、仿真协议和目标 task/init，只改变四个已发布 FLASH draft 的来源 suite。先测四来源 × 四目标 suite × 十个 task × init20。完整 FLASH 与纯 draft 分别各 160 episode，Full 基线 40 episode。共同 H=50、R=12、settling=10、环境 seed=7；FLASH δ=0.15、PF=2。最终运行参数以冻结 manifest 为准，任何差异须单列。

比较必须在同一个目标 suite 内进行。记录成功率、逐请求接受情况、Full 调用/回退、真实推理和闭环耗时、执行步数，分开报告纯 draft 与完整系统。跨条件 request noise 由 suite/task/init/request 决定，不受调度顺序影响。不同条件自由闭环后的观测会分叉，因此接受率是各自闭环指标，不伪称同观测动作误差。

发现矩阵完成后，按预定规则选择最需要核实的 source-target 差异，在 init21–24 上独立复验；若没有初步差异，同样报告，不强行找显著结果。复验选择及方向须在新数据生成前冻结。Spec-VLA 另查本地资产和固定 target 兼容性；若可运行则补独立模型证据，不把 FLASH 单模型结果扩大为所有 draft 的普遍结论。

历史扰动证据可引用，但不可把不同母任务/初态的数据混成严格配对退化：LIBERO-Plus 报告的 30.9% 是纯 draft；其原始20例只有40%，与早期200例68.5%不是同一批 case。已有175例扰动的 Full、V14、draft 是同case配对，可以比较扰动下相对表现。

## 2. 同预算固定路径的任务偏好

第一对候选来自已结束的 v3.1 init0 发现数据：π0 vision27保留24层，A=front_middle，跳[8,9,10]；B=uniform，跳[5,14,23]，层编号1-based。两者都完整保留VLM18和AE，D10/H50/R5，不做Full混合、残差补偿或在线切换。历史成功为A35/40、B36/40，Goal的task0/5偏A，Long的task3/6偏B；这只是弱线索。该候选是在本轮新数据生成前从历史结果筛出，seeded_random作为备选保留记录。

独立验证固定四suite全部40个task，init20–24，共200例/条件，另有同协议Full参考。先用init20检查通路及耗时，计入200总量，不根据这批结果改候选、类别或删除困难任务。主要方向事先固定为Goal A>B、Long B>A；报告全部suite和全部逐task结果。成功率配对差、A-only/B-only成功数、95%区间、每task一致性是核心分析，不能只看总体平均值。

并行只读筛查已有VLM同预算结果；若第一对候选差异不复现，进行单独标记的固定预算局部搜索，并使用未用于搜索的新初态复验。发现阶段与验证阶段不可混算。阶段选择、runtime选择器和切换工程暂不属于这两步实验范围。

判断边界：A全面优于B支持固定用A；只有一种类型差异支持路径有影响；在独立数据上两个方向的偏好反转才直接支持互补。200例/条件仍是探索性验证，不是官方50初态全量基准。验证不显著不能自动解释为等价。

## 3. 执行与数据约定

- 代码写入 preparations/ease_insights_20261103/；结果写入 results/ease_insights_20261103/；不修改历史结果。
- GPU由一个调度者每30秒检查自然空闲，使用UUID固定、项目共享flock锁；不终止外部任务。默认最多同时使用两张空闲卡，后续按真实吞吐调整。
- 正式运行前冻结checkpoint标识、实现hash、mask、采样/控制参数和case清单。保留官方init向量hash、BDDL、prompt、首次观测hash与控制器标度，配对身份不一致则不合并。
- 持续服务，优先少量必要I/O，保留逐case结果、逐request时延/路由、实际动作和末端轨迹；不默认录全量视频。
- 冷编译与加载单列；热推理加速、每episode成本和整批评测墙钟分别报告。技术中断不记为策略失败，已有效失败不反复运行到成功。
- 收尾检查所有预定case及条件，报告缺失、技术失败、独立复验边界与两条insight各自能支持的结论。

## 4. 已核对的输入

- idea1.md v4.1 与原 EASE PDF：两个待验证假设。
- reports/idea1_libero_plus_half_final_20261022T2124Z/README.md：扰动证据与口径限制。
- results/idea1_v3_1/manifest_v2.json 与 final_artifacts_20261021_0524/：vision候选、mask与发现数据。
- shared/checkpoints/realtime-vla-flash/：四个draft checkpoint和converted/base均已存在。
- 模型解释器和libero-eval-v21解释器的CPU导入检查通过；GPU占用来自本容器外的进程，等自然释放或用户指定设备。

## 5. 启动后的证据与调整

1. 用户明确选择等待本机自然空闲卡。GPU0/3随后释放，使用UUID租约启动；没有终止外部进程。
2. FLASH四suite各task0/init20的4例通路检查有效完成，全部成功；Spatial来源在Spatial接受率100%，Object/Goal/Long为0%/3.7%/0%。纯draft同4例有效完成，成功1/4，100请求均为draft调用，Full/verify调用为0。两模式初态、真实sim状态、BDDL、controller、首观测及noise哈希一致。每suite仅1任务，不能据此宣称跨suite结论已经成立。8例计入360例发现矩阵。
3. vision历史A/B没有正式同输入端到端提速证据；另一Keep24路径曾测得0.982×native Full。A/B首先作为裁剪路径的偏好线索检验，完成40例/条件pilot后再决定是否值得扩展，不把它们称为已经实测更快的路径。该调整源于启动时追加的历史时延核查，不是根据新成功率挑选。原200例/条件的独立验证目标保留在vision manifest中，但不自动排队。
4. 为检验真正较便宜的fast path，准备三条固定V12 VLM路径：Jacobian校准mask[2,3,5,14,16,17]、单层交换MSE mask[2,3,5,11,16,17]、cosine mask[2,3,5,6,8,9]。所有条件保持Full-D10→Fast-D2→Fast-D2、H50/R5、同残差外推。init30四suite全部40 task做发现，候选与类别冻结后用init31–35验证。MASK必须真正进入执行与残差捕获，不能仅改配置名；数值门禁完成后才启动闭环。
5. 首次vision Full服务在生成episode之前遇到系统NumPy2与模型NumPy1的pickle兼容问题，服务已退出，随后将runner改为OpenPI自带解释器重启。该技术失败单独保留，不计入成功率，也不需要改变科学配置。
6. 固定draft发现矩阵独立复验选择规则：按同目标suite内“匹配来源成功率−外来来源成功率”排序，选最大正差的source-target对；如有另一个不同target的正差，再选其最强对。并列按固定suite顺序解决。选择和方向必须在init21–24生成前冻结；这属于发现后的独立验证，报告全部发现比较而非只报被选中的差异。
7. vision init20 pilot已完整完成并通过独立协议审计：Full39/40，A36/40，B38/40。Goal为Full10/A9/B10，Long为Full9/A7/B8；A-only成功为0，B-only为Goal task3和Long task6。预注册的Goal A>B没有复现，不能将此pilot称为互补证据，也不重挑类别。报告在reports/ease_insights_20261103/vision_pilot/。这批不扩为vision200，后续把验证资源用于有真实成本优势的VLM路径；历史发现与本次不复现结果都保留。
8. VLM首个服务通过116项实际GPU数值/执行检查。四个固定真实输入、各条件预热6轮后测12轮的首次成本测试：Full平均109.09ms；Jacobian/MSE/cosine固定路径的Full-Fast-Fast平均为68.59/68.17/66.72ms，比例约1.59/1.60/1.64倍。这里包含策略内部预处理、残差、模型、解码和输出传输，不含RPC/环境；不能当成闭环评测加速或质量证据。闭环发现矩阵仍在运行。
9. VLM首个episode原生结果已经完成，但receipt代码把积分节点list当audit dict读取，触发三次相同案例的技术尝试。已按最早complete时间恢复第一份结果；另外两份原样保留、不计分母。新增runner_v2/worker_v2/recovery技术修订，科学manifest、模型与全部29个冻结源文件不变；收尾异常不再触发已完成科学episode重跑。独立审计确认真实worker_v2 argv、后续task正常完成、首例原始hash不变，见reports/ease_insights_20261103/vlm_technical_recovery_audit.json。
10. FLASH拒绝回合可能执行0动作。正式实际成本采用完整policy/RPC累计耗时除以实际执行动作数，并保留同case episode墙钟与成功率；平均每请求耗时仅为诊断，不单独称实际加速。启动与每个新prompt的冷成本另列，不能用GPU-stage时间冒充端到端热态性能。
11. 已启动独立pipeline接续进程：它只观察现有GPU调度器，待发现集严格完整后执行冻结选择、创建独立确认矩阵、原子追加队列和统计；不另扫描GPU、不终止外部任务、不标记用户目标完成。VLM确认按4个task分片保留每task连续5初态，总计600例；FLASH确认按冻结1–2个source-target对生成200–400例。无合格候选或技术问题分别交回科学/技术审查。实验队列使用无额外停留的即时空闲检查＋租约＋再检查，仍每30秒查卡。
12. 后续成本分析预先固定all init21–24、首次task init21、repeat-task init22–24三种口径；repeat-task不等于已证明完全热态。记录外部GPU争用，不能将共享负载下的墙钟直接称为独占卡性能。语义taxonomy只从官方语言/BDDL建立，未读成败，主suite-only选择规则未改；类别与suite相关，任何语义分析须明确探索性质。
13. 用户明确更新目标：没有空闲GPU时持续等待并间隔轮询，不擅自停止目标。此前两个前台工具会话对应的进程实际退出；恢复时确认Object纯draft原40例已经完整结束，只修正滞后队列状态，零重跑。调度器与pipeline现通过nohup、独立session、文件日志及断开stdin启动，已确认PPID=1且心跳正常。空闲扫描维持30秒。当前发现状态为draft280/360、VLM160/160；VLM冻结Jac/MSE及Spatial/Long两个方向，600例独立确认已入队，尚无独立验证结论。
14. 在独立确认仍无观测时，将VLM运行粒度由12个50例分片合并成3个连续200例条件，以减少模型重载和重复释放GPU；逐case集合、seed、配置与科学manifest字节完全不变，当前GPU任务未被打断。登记见process_recovery/coarsen_pending_confirmation_*。未来就绪门槛使用新的all200任务与confirm_shard00of01_complete.json，旧pending任务状态仅是历史。
15. 在任何独立确认观测产生前，另登记Goal方向Jac>MSE作为补充比较（发现10/10对7/10）；原Spatial/Long主方向保留。固定三方向同时报告、Holm3及case/task聚类区间，原主结论和Goal/Long补充结论分列。协议为vlm_confirmation/supplementary_goal_contrast.json，不改变600例实验或冻结selection。
16. Long来源纯draft在36/40有效结果后因同卡另一个进程占18.02GiB而OOM，失败case没有成功模型请求。36例及899请求全部原样保留；归档技术尝试后仅续跑剩余4例，真正收齐40例并通过审计才发布完整收据；两次准备/活动wall与历史GPU来源分别记录，不把技术中断记成策略失败。目标保持active并继续等待空闲卡。
17. 首次Full输出独立CPU审计覆盖160/160 FLASH–Full配对：159对raw/decoded精确一致，Spatial来源→Spatial task1/init20一对超过1e-5（raw最大0.0078125，decoded最大0.00390387）；输入、噪声、target元数据一致，200条归一化独立重建全部通过。该例保留，原因未定位，不宣称全部数值严格一致；主结果与明确标注的删除同case敏感性同时可查target_identity审计。

18. Long纯draft补跑再次在首个新请求前OOM，VLM Full确认也在模型准备阶段OOM；分别新增有效case为0，原有科学数据全部保留。Long v2恢复保留36case/396文件/899请求及两次leaf成本、逐case来源；累计技术case成本17.060232s不另加总wall。失败证据见resource_contention/recovery_attempt_failures_20261103.json。所有补跑及后续确认新增assigned-GPU启动等待：分配后每10秒复查该卡，连续60秒无compute进程且显存低于500MiB才exec原实验命令，busy/查询异常继续等待。该策略降低短暂窗口风险，不能保证加载后无外部竞争。现有scheduler未停止；两技术分支失败导致旧pipeline自然退出，核实/proc absent后启动新pipeline继续自动接续。冻结科学配置、样本和选择规则不变。

19. 等卡期间补充历史Plus175的V14–匹配纯draft配对统计：V14-only51、draft-only16，差20.0pp，40母task聚类bootstrap95%区间[8.3,31.4]pp。全部为已看过数据的探索性分析；相机类差为-8pp，不能称各类均优势；原始20例非相同母任务/初态受控对照，不能用作扰动因果降幅。585个小result身份/hash复核通过，case210的V14与draft初始NPZ一致，Full涉及的细微图像差异仍保留披露。见reports/ease_insights_20261103/perturbation_context/。

20. 后续实验在运行时增加只读资源观察：每15秒仅查询已分配且/proc确认身份的实验GPU，保留显存、利用率、host compute PID与查询错误；无实验或仍在启动等待时不查询GPU，不参与空闲发现/调度。主调度器继续每30秒等待自然空闲。记录用于解释共享负载，有限采样不能证明全程独占；不据争用删除或重测有效案例。12项CPU测试及当前无实验零样本检查通过，输出resource_contention/observer/。

21. 2026-11-04 00:50:32固定截面的CPU汇总已另存interim_20261104_waiting/，原21:21快照保留。Draft356/360，16个FLASH格完整，Long纯draft匹配格2/6明确不完整。15/16 FLASH格policy每执行动作成本高于Full；Object匹配来源23.12ms/action、接受率61.37%，Goal来源21.02ms/action、接受率0%，双方10/10，显示实测成本不随接受率单调变化。不能把成本差全部因果归于泛化，也不能声称跨suite消除了已经成立的加速收益；仍需独立确认，冷启动、轨迹和资源争用边界保留。

22. 后续只读source归一化/噪声合同审计见source_normalization_audit/README_20261104.md：当前cache训练与serve都使用base的assets/physical-intelligence/libero归一化，未发现source专属映射；但发布head缺原cache manifest、训练checkpoint/norm hash/config快照，不能称已完全排除历史预处理混淆。当前结果限定为共同Pi0推理合同下的发布head迁移表现。teacher_zero_noise仅训练缓存，官方online默认本就用标准正态；显式配对NumPy Gaussian未新增zero/random错配，但RNG和CPU/传输路径有差别。policy_wall_ms只包native original_infer，CPU seed/hash在外，完整client RPC包含wrapper与传输；新中期报告已分开标明。未改变任何已执行或冻结条件。


23. 2026-11-04 恢复工作时已确认后台取得新进展：draft发现360/360、独立确认400/400均完整，Jac/MSE确认各200/200完整。恢复前严格重验VLM receipt artifacts与冻结scientific sources通过，Full仍0/200。Full第二次启动在模型初始化阶段遭遇另一进程20.62GiB占用而OOM；没有有效episode。原scheduler/pipeline均已自然退出且/proc确认absent，保留原队列、状态与失败日志后以新ID vlm_confirm_full_d10_all200_retry2继续相同科学命令；无进程被终止，原资源观察器保持运行。新调度器335208、pipeline335209；GPU3通过连续60秒空闲检查后已启动Full服务。恢复记录位于process_recovery/full_confirmation_retry2_1793779702133338058。平台goal元数据此时仍为usageLimited，用户明确请求继续；后台工作实际恢复，不能将平台状态误称active或实验完成。

24. Draft独立确认支持共同Pi0推理合同下的发布head跨suite差异：Object目标匹配纯draft40/40、Spatial来源0/40；Spatial目标匹配35/40、Object来源0/40。原冻结主检验Holm校正p分别1.81899e-12、2.91038e-11。FLASH成功Object为40/40对39/40、Spatial双方39/40；动作加权接受率匹配为71.894%和56.417%，外来双方0%。所有400例及80组五条件的首次input/noise配对通过。全部init动作加权native policy ms/action：Object匹配FLASH7.185、外来9.486、Full8.234；Spatial12.492、15.608、11.419。接受率降低、Full回退增加与观测成本增加得到独立复验，但不能称匹配FLASH在两个suite都已快于Full，也不能将所有成本变化因果归于泛化。case等权统计与动作加权汇总分别标注；固定任务的多初态不能当未见任务泛化证据。详见draft_statistics、draft_confirmation_analysis与draft_confirmed_audit。

25. Jac/MSE的400条独立成对结果均有效，两者总成功均170/200。原Spatial Jac>MSE方向为47/50对46/50；原Long MSE>Jac为34/50对34/50；预登记补充Goal Jac>MSE为39/50对43/50。三方向case/task聚类区间均跨0、Holm3均1，原主及补充的双向偏好均未复现，不能据此宣称互补或等价。Full尚未齐时只发布pair-only interim，不绕过600条正式完整性门槛。任何基于这批逐task结果的新候选都只能视为新发现，必须使用新初态且另冻结协议，不能修改本轮主方向。


26. 有限局部搜索已在新观测产生前另冻结：以旧init14的20个观测的单层predicted/actual扰动排名重建Jac单交换邻域，51个合格未测邻居按各suite校准分数选4个；不宣称其来源标签已代表闭环偏好。旧三候选分数复现，但旧source_manifest hash链接仍有缺口，完整保留。新协议vlm_local_search_v1/protocol.json规定4邻居+Jac anchor+Full在init36–38的720例完整发现，再按可用性、同输入成本及双向/逐task稳定性规则决定是否冻结2候选与方向，用init39–43作600例独立确认。最多1320个新科学episode；无候选或未复现即报告该有界搜索的阴性结果，不自动重复寻找显著性。确认主检验使用10task均值差的exact block sign-flip+Holm2，case McNemar作为敏感性；旧主假设/结果不变。目前新runtime仍在准备和CPU校验，尚未启动这一轮GPU实验。


27. 局部搜索正式清单已冻结，manifest_hash=032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9。Root联合复跑33项CPU检查通过（22.713s），含真实旧episode输入格式校验与1320条临时artifact的全流程测试；独立只读审阅无阻断。新mask尚须每次服务启动时通过实际GPU数值/执行门，CPU通过不替代GPU验证。6个连续120例任务已原子追加至原queue；新supervisor PID338645每10秒接续，只采用原GPU allocator335208，不自行找卡。它共享调度器重启锁、仅在OS确认退出后允许重启，no_candidate/未复现不会无限扩展。日志与launch_audit在results/ease_insights_20261103/vlm_local_search_v1/。原Full确认继续运行，旧冻结源码和主假设均未改变。


28. 原VLM确认600/600已严格完整，missing/invalid/unexpected均0：Full182/200、Jac170/200、MSE170/200；原Spatial/Long与事前登记Goal/Long正式结论均未复现。主统计与补充报告中的原主副本逐项一致，Holm2保留、三方向另给Holm3，正式报告见vlm_confirmed_audit与总README。原pipeline自然完成退出；GPU scheduler与局部搜索supervisor继续运行，新的Full120任务已分配GPU2但因其再次出现外部compute进程而仍处于CPU启动等待，不能称新mask已开始GPU实验。没有因无卡停止整体目标或标为完成。


29. 局部搜索Jac anchor在GPU0通过连续60秒自然空闲检查后实际启动；全部6条件共332项GPU数值/捕获/执行/调度检查通过，固定输入成本与服务protocol生成，随后进入新init36–38闭环。初始15条真实episode经新load_stage严格验证，missing705、invalid0、unexpected0，未执行选择。Full120任务仍是GPU2的CPU等卡进程，未开始模型运行。根目标已由用户重新设为active，并明确无卡持续等待、直到两个insight证据齐全才结束；局部协议的no_candidate仅是本轮分支终点，不代表总目标完成。原600例导出科研图见vlm_confirmation_figures。


30. 为利用其它卡的短暂空闲窗口，新增仅重排自有CPU等待gate的协调器，最终源码70ce6ef8...，root与独立审阅均通过90项fake测试，真实default inspect通过。检查涵盖已审queue字节到最终rename的守卫、pending同audit恢复、scheduler真实relative python3重启的kernel-exe/raw-argv身份、共享队列历史script忽略及槽容量规则。两次手动尝试均因目标卡在fresh检查时重新占用而在STOP前被拒绝，队列和进程未改变。协调器以nohup/独立session/DEVNULL启动PID356386并核实PPID1/live/waiting；实际GPU分配仍由原scheduler335208执行，模型任务不是迁移对象。首次真实事务另做只读审计，不能把脚本启动称为已成功迁移或新科学case。状态见gate_opportunity_coordinator/。


31. 首次真实CPU等待任务迁移已完成并经独立只读核验：事务opportunity_1793793271326258_000001仅替换Full的job ID与gate状态路径；其余35个任务、科学命令、complete_file及46项runtime/6项analysis源码hash均不变。队列提交后，原scheduler335208持久化新ID且同一supervisor338645完成postcommit tick，随后才清理旧CPU gate339341。原scheduler分配新gate361717至GPU3；该卡约10秒后再次busy，未达到连续60秒空闲，因此仍未启动Full模型。旧job历史记录的returncode=-15是调度清理，不是科学episode失败；当前队列failed为空。首次迁移审计见reports/ease_insights_20261103/gate_opportunity_live_audit/first_transaction_1793793271326258.json。Jac已完成120例，整体新发现仍120/720；持续等卡，总目标保持active。


32. 用户要求提高GPU释放通知频率后，新增独立只读gpu_release_watch：每2秒查询全部4卡的显存与compute进程，只有零进程且显存<500MiB标为idle；查询失败、缺卡或无效值标unknown。最终源码4314b4b5...，11项CPU测试通过；真实单例后台PID385842/start2618445988/PPID1与启动hash已核对。state变化写事件，新idle事件等待命令带run/event cursor，通知同时给原事件时间、当前状态及新鲜度；子agent gpu_release_watch持续守候并通知root。原scheduler335208、coordinator356386和所有模型gate未改变，连续60秒启动门槛保持。运行记录results/ease_insights_20261103/gpu_release_watch/。这只提高通知及时性，不构成新的科学episode或insight证据。


33. 2秒监控已实测捕获GPU0/3的30秒释放窗口（event9/10→11/12），但旧60秒门槛未触发模型启动。资源门槛不在冻结科学source内，因此新增gate-only env：EASE_GPU_GATE_IDLE_SECONDS=5、EASE_GPU_GATE_POLL_SECONDS=2，默认60/10仍保留，exec前同UUID额外fresh复查，两个env在exec前移除。新旧gate15项CPU检查与独立复核通过，gate源码b4b5ca1b...。独立一次性refresh工具b63b6a03...经18项CPU fake和独立审阅后，将原Full361717和local_spatial344836分别通过pidfd STOP/精确身份复核→原子更名及资源env→原scheduler与同一supervisor双ack→TERM/CONT接续；两个事务均complete，旧进程absent，历史技术清理保留，科学argv/完成目标/35个其他任务及冻结manifest未变。另三个pending任务仅加相同资源env/metadata。原scheduler335208和supervisor338645持续运行；coordinator356386通过stop正常退出，事务结束后同源码恢复415450/start2619222121/PPID1，当前waiting_scheduler_capacity、pendingNone。全部5个未完成任务采用5/2策略，等待原scheduler分配；尚未运行新episode。记录gate_policy_refresh/与gate_policy_refreshes/；该调整不构成insight证据，也不保证共享GPU持续独占。


34. 新5秒门槛捕获下一次GPU0/3释放（watch event13/14，1793818650.263），两个gate均在约6.6–6.9秒后通过fresh空闲复查并exec。Full/GPU0与local_spatial/GPU3的服务随后在模型构造pi0_infer1633报CUDA OOM，未进入任何科学episode；两failed job、identity/server日志及资源观察均保留。观察显示CPU加载期间两卡仍曾1MiB/0进程，随后持续18GiB进程在我方服务退出数分钟后仍在；无NSpid映射，外部重占与OOM因果按强推断而非完整归属确证披露。轻量context/显存claim与冻结runner82–85的任意compute拒绝冲突，按有界工程边界未实现、不改科学源码。恢复采用idle60秒/poll2秒与原fresh复查，所有5个未完成任务统一设置；2秒全卡监控及即时通知继续。strict120/720有效（Jac99成功+21有效策略失败全部保留），Full/spatial均0，invalid/unexpected0。经独立审核的stable60s恢复包原子更名两个0case failed job为__startup_retry1并保留原科学命令/输出，原scheduler335208持续运行；新Sup437956/start2619934847与新coor438128/start2619938459已OS、状态、绑定验收，当前waiting_gpu/active[]/current_failed[]。准备/归档/执行收据见process_recovery/local_search_startup_20261105_retry1_stable60s_review/，诊断与恢复方案见reports/ease_insights_20261103/startup_oom_20261105/。整体目标仍active，技术启动失败不计作策略失败，不重测已有有效case。
