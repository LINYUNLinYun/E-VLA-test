# idea1 v3.2 执行记录

2026-10-22；[方案与边界](idea1_v3_2_experiment_plan.md)。本轮已获启动授权，用户短暂暂停后已恢复。原始 `idea1.md` 不修改。

## 最终完成：2026-10-23 10:39 UTC（18:39 北京时间）；12:00 UTC后离线验收

- 新增V14-D2有效结果195/195，175个Plus扰动＋20个原始对照，无待测key。来源为父GPU0 98例＋冻结GPU1 39例＋GPU0接续58例；没有重复有效key。
- Plus D2成功85/175（48.6%），对照旧V14-D10为89/175（50.9%）、Full-D10为138/175（78.9%）、母suite匹配纯draft为54/175（30.9%）。原始20例D2为16/20，与D10同分；具体分suite和配对变化已填入主报告4.5、4.7、4.8、4.15。
- case154因资源释放中断的技术attempt保留且不计策略失败，接续有效重试完成。三个启动的102份NPZ、204对数组重新加载逐位一致；全部195例的14,754次请求、73,579步动作和73,774条真实XYZ通过独立验收。汇总器独立6项CPU测试通过。
- 父GPU0与接续均queue_scanned，两个对应模型服务均complete；已核实原watcher／service和接续watcher进程退出。GPU1保持留给另一个CUDA双流会话，没有重新扫描或占用；全部新增实验无视频。四draft诊断也已完成，未恢复CALVIN等暂缓分支。
- [最终统计及195例来源](../../../reports/idea1_libero_plus_d2_final_20261023T1200Z/summary.json)／[执行全流验收](../../../reports/idea1_libero_plus_d2_final_20261023T1200Z/root_all195_execution_audit.json)／[启动数组复核](../../../reports/idea1_libero_plus_d2_final_20261023T1200Z/root_saved_startup_arrays_audit.json)。以下为执行历史，不代表仍在运行。

## 最新资源交接：2026-10-23 08:24 UTC（16:24 北京时间）

- 用户要求释放一张卡给另一个CUDA双流会话。本轮GPU1的已核实模型服务3837647已通过RPC正常关闭，GPU1显存4MiB／0%；当前本轮只继续使用GPU0，不再扫描或重新抢占用户已让出的GPU1。
- GPU1保留39条有效complete，剩余58个key待接续。资源切换使case154在250步中断（technical_failure／success=null），原记录完整保留、不计策略失败；不是模型预测失败。
- GPU0原98例分片继续正常评测。6.1-sol medium正在准备独立GPU0-only接续清单：待父GPU0自然跑完98例并释放服务后，仍在GPU0重过原严格零差门禁，补这58个未完成key；不改冻结双卡分片、不覆盖或重跑已完成，最终仍合并195例。
- 接续已完成root独立3项CPU测试、Plus／original两实际解释器episode dryrun并上线：manifest `fbaf319821ae7ad12b2b74d2f6b1004c35b536b1216d7168893844cdbd14671c`，tmux `goal_d2_gpu0_continuation`、CPU等待PID3873549、后续端口19166。它只观察父GPU0；父98正常排空、父模型退出后才检查GPU0空闲并重过同严格门禁，顺序补58，不查询或占用GPU1／2／3。[接续清单](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu0_continuation_20261023_run1/manifest.json)／[接续状态](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu0_continuation_20261023_run1/continuation.json)。
- [用户资源交接记录](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_quality_gpu01_20261023_run2/runtime_gpu1/user_card_release.json)。以下双卡ETA与运行快照是这次资源调整前的历史状态，新的单卡结束时间需按后续实际进度更新。

## 当前正式运行：2026-10-23 07:56 UTC（15:56 北京时间）

- GPU0／1各自严格零差门禁全部通过，双B1固定98／97分片已正式运行，不再等GPU3；不batch、不录视频。当前run2身份`c823d3ccea915a4cca78a6282b479f290f1cfff7a0c58052d14081a7efad3290`，截止1792762731，原195案例／585参考不变。
- watcher PID3837565／3837568，service PID3837648／3837647，端口19164／19165，tmux `goal_d2_quality_run2_gpu0/1`。07:56快照5 complete、2 running、0 technical；持续进度以各runtime_gpu{card}/watcher.json和有效result为准。
- 首例card0/case0成功99步20请求、card1/case1成功184步37请求；root重载全部startup数值数组重算全字节等价，两卡同输入D2也逐位相同；首2回合完整57请求／XYZ／实际动作／oracle／初态／无MP4验收通过，已原子发布各精确marker，剩余193继续派发。[首例验收](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_quality_gpu01_20261023_run2/root_first_cases_audit.json)。
- 运行源为6.1-sol medium实现，root独立3项quality-v3测试通过；未更改冻结v1/v2或旧结果。run1导入失败仅补固定两个依赖，以--no-deps保留numpy1.26.4/mujoco3.3.6/torch2.7.1+cpu/robosuite1.4.0；实际两解释器actor dryrun及隔离原生import通过。旧失败记录保留、0策略分母。
- root按旧同195例单卡累计3.29小时及当前D2成本估计：双卡正式开跑后约1.5–2小时、保守2.5小时，约北京时间17:30–18:00，保守18:30；不是已完成时间承诺，后续昂贵扰动和成败路径会影响用时。数据持续自动落盘，最终195全流验收仍必需。

## 当前：2026-10-23 06:59 UTC，用户授权其他空闲GPU严格等价诊断

- 用户明确允许换卡，但不接受跨卡数值差异；此前硬绑原GPU3是执行方自加限制，现不再将其作为永久阻碍。
- 初次检查GPU0／1均1MiB无compute，GPU2／3各约16GiB占用。GPU0主诊断、GPU1固定复制，启动前与首次CUDA前复核UUID和空闲。旧GPU3 watcher／reviewer仍存活，诊断期间保留，质量派发必须单一协调避免重复。
- GPT-6.1-sol medium正在新增隔离probe及CPU测试，16条按metadata固定真实参考（扰动／原版各四suite×Full／V14），每条重复并要求全raw／decoded逐位一致。额外核真正生产路径、有效中间数组与D2原生parity，不放宽门槛，不录视频，不派发质量条件。
- 新指令后的实际工作已推进；下方06:33资源阻塞记录为旧设备限制下的历史审计，不能用它阻止这次已授权诊断。
- 双卡probe已实际完成：[固定manifest](../../../results/idea1_v3_2_e4_supplement/d2_cross_device_bitwise_20261023_run1/manifest.json)，身份`00bc9f0cad3b4e8b3a82a674859493add471fe1dbcd5f69ae69f5acb0d17d65f`，原probe PID3801072／3801084均自然退出，GPU0／1回到1MiB。每卡16参考×2，加2捕获／4D2 parity调用，共38 preflight；零质量、零视频。root独立复跑5项CPU测试并逐文件比较数组。
- 实测GPU0vsGPU1所有16 raw／decoded逐位相同；对旧记录14/16逐位相同，只有同一扰动case0的Full／V14例外。所有当前捕获stage也跨卡相同，因此不能将异常直接归因GPU。未放宽gate、未移除case0、未开始新卡质量评测。
- CPU来源追溯后下一分支固定单卡GPU0：精确legacy preflight六route原顺序后，先未capture Full×2、V14×2，再额外capture。绑定同bank和旧／新输出，保留V14历史19条演化观测缺失边界；当前由GPT-6.1-sol medium准备独立新模块，不修改冻结v1。

## 最新：2026-10-23 05:58 UTC（13:58 北京时间）

- 当前授权仅推进Plus同卡V14-D2与四draft失败归因，不恢复旧CALVIN等分支；全部新增实验无视频。
- 四draft最小诊断已完成：40协议回合＋4启动动作拆分＋8闭环恢复；另24次语言请求、48次同状态输入请求。输入为11严格对照＋1明确标注的Long背景降采样辅助臂，不将其冒充精确原纹理恢复。
- root全流核验52回合、3,125次质量请求、15,540步执行通过；CPU分析复跑14项测试、739条恢复／语言／输入响应核验、8项同帧噪声核验均通过。所有无效visual_preencoder统计剔除。
- Spatial／Object各1个语言改写案例恢复原指令后由失败变成功；Goal／Long语言恢复仍失败；4个固定去噪案例成败均未改变。背景有动作敏感性，没有背景闭环致败证据。完整表及边界见[主报告第4.14节](idea1_v3_2_report.md)。
- 原watcher／draft服务意外消失的收据保留。用独立tmux恢复后，新draft服务再次通过四head历史raw／decoded逐位一致，最后辅助4请求已完成；健康核对PID／UUID／源码后正常RPC关闭，PID3752040消失，GPU0实际回到1MiB。
- 当前后台GPU实验队列是GPU3专属等待队列：tmux `goal_d2_gpu3_recovery_0543`、PID3752472、manifest `f35a09c66577443fcc19d8c7fea0127003ce405267afcd74898dd3c0ad3273c4`。每30秒等待原GPU3，目前仍有外部16GiB计算分配，不抢占；V14-D2质量结果0/195，不记作0%成功率。严格同卡启动门禁及全部无视频保持不变。
- [当前watcher](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery/watcher.json)／[诊断汇总](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/diagnostic_summary.json)／[12输入分通道CSV](../../../results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/input_sensitivity12.csv)。整个新增目标仍未完成，不标记complete。
- 06:10 UTC追加CPU排查：两条D10启动参考的原观测／模型输入／噪声哈希匹配旧首请求；FP64／FP32状态差经实际CPU准备器后BF16 state相同，图像与noise也逐位相同。没有GPU模型请求，不能据此定位GPU快速输入核或首个VLM／AE分歧层，也不把未证实的“跨卡舍入”当作结论；同卡195条件队列继续等待。
- 06:19 UTC：root通读GPT-6.1-sol medium新增独立首例检查器、独立复跑5项CPU测试通过，已用tmux `goal_d2_first_review`上线，PID3775288、GPU可见性为空。只监测原watcher PID＋start_ticks与状态文件，每30秒读一次、不扫GPU；待首例技术／数值／执行全流核验通过后原子写精确续跑对象，首例成或败不影响放行。该自动检查代替等待root手工写marker，科学门槛未变。[审核收据](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery/first_case_auto_review.json)。
- 四条件CPU汇总器由GPT-6.1-sol medium另行实现，root审阅并独立7项测试通过；首份root核验输出在[analysis_pending_root_20261023](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery/analysis_pending_root_20261023/summary.json)。旧585全部绑定SHA，重新得到扰动138／89／54成功（各175），无扰动20／16／8成功（各20）；D2完成0、195待测，所有配对率null。后续汇总只比较D2已完成案例交集，技术失败／未完成不计0分。汇总器不代替最终195例全流独立执行验收。
- 06:33 UTC：连续至少3个goal轮次的同一资源条件核验已满足阻塞审计；原GPU3仍约16GiB非本任务分配，D2服务未加载，0/195。独立CPU／draft工作已完成，当前推进质量评测只能等待原设备释放，已请求用户协调；不放宽数值约束。watcher PID3752472和reviewer PID3775288均存活，不发送任何停止／暂停信号，仍按原deadline1792774358继续等待并自动接续。[阻塞核验原始记录](../../../results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery/blocked_resource_audit_20261023T0633Z.json)。

## 2026-10-22 04:14 UTC（12:14 北京时间）

- 已读取原文 v3.2 全部四项并与 v3.1／v2.1 真实代码和数据核对。新代码、结果和文档均隔离于历史结果。
- E1：GPT-6.1-sol medium 正在实现真正 B1/B2/B4 原生 batch 及并发 rollout 聚合，已授权使用初次检查空闲的物理 GPU3，启动前需再检查。真实 GPU 推理尚未开始。
- E2：GPT-6.1-sol medium 实现十次更新内的 AE pre-final-norm residual 复用，仅完整刷新 0/3；等待 E1 释放 GPU 或其他卡空闲。真实 GPU 验证尚未开始。
- E3：GPT-6.1-sol medium 已确认读取原 v2.1 的 634/6345 校准／评估点，正在实现 CPU 外推比较。尚无新方法成绩。
- E4：本地已有 CALVIN debug、SimplerEnv 和 PushT 代码；FLASH 发布和本地权重仅发现 LIBERO 四 suite draft。正在核对接口，已向用户询问是否接受有明确局限的 CALVIN 零样本迁移，或等待匹配权重。前三项不因此停止。
- 第一次 GPU 检查：0/1/2 已被现有计算进程占用，未干预；3 空闲。不得将 0% 利用率但高显存占用的卡视为空闲。

外部资格参考：[FLASH 官方仓库](https://github.com/dexmal/realtime-vla-flash)、[发布权重](https://huggingface.co/Dexmal/RealtimeVLA-Flash/tree/main)。外部信息仅用于资格判断，不替代本地实际执行证据。

## 04:18 UTC 更新

- E1 第一轮因 JAX AOT `.lower` 的 ArgInfo 触发运行时类型标注检查而退出；失败日志保留在 `results/idea1_v3_2/e1_20261022T_batch/`。仅在 AOT tracing 时使用原生已有的 typechecking-disable 上下文修复，不修改网络或放宽原生 B1 一致性门禁。
- 当前 E1 重启目录：`results/idea1_v3_2/e1_20261022T0418_batch/`；controller PID3193599、service PID3193667 已通过 `ps` 确认活跃，仍在加载／编译，未产出吞吐结论。每样本 batch 对照同时记录严格 `1e-5` 和宽松 `2e-3`，不将宽松通过称为完全一致。
- E2 standalone 入口和数值门禁已实现，等待 GPU；增加 native D2 对照、实际10次更新／2次Transformer刷新计数和跨请求缓存隔离检查。CPU门禁测试不等于真实模型通过。
- E4 已完成[具体资格检查](idea1_v3_2_e4_compatibility_audit.md)：本地真实 draft 为110.3M参数；目标环境接口与新语言存在适配缺口，正式策略评测仍待定义确认。

## 04:23 UTC 更新

- E1 又保留了一次原生门禁失败：封装 sampler 将 `num_steps=10` 变为闭包常量，和原生通过参数追踪的实现出现约0.003差。将参数追踪方式对齐后的第三次运行在真实 B1 检查中 **max_abs=0**；当前路径为 `e1_20261022T0421_batch/`。
- 100点 bank 初测吞吐 B1/B2/B4 约12.28/15.30/18.56例每秒，但 B2/B4 数值门禁失败（最大 raw／decoded 误差约1.93），**不能认定为可直接替换的加速**。正在检查首5步／尾部误差、batch顺序与噪声身份；不将“更快”与“等价”合并。
- 原 runner 选择通过门禁的B1，已在首个8场景B1臂完成后暂停调度，避免再跑两次相同B1冒充batch验证。初始8场景全部成功；暂停点未完整记录arm墙钟，不用它计算吞吐。将明确标注近似的B4探索臂与后置B1臂各测8场景，单独计时并保留所有结果。controller暂停不表示GPU服务已退出。
- E3 第一版已在98.5秒完成51候选与6345×50评估：三种新方法首5步宏均误差均高于冻结Kalman。发现继承历史指标命名将单步L2与累计along/lateral放在同表，正在保留原产物另做revision2，补齐同一误差向量的分解及勾股一致性检查，避免混淆。
- 主代理检查旧 `manifest_v2.json` 的原有源码及checkpoint身份文件仍匹配冻结SHA；原历史实现未被修改。新的组合测试发现E3脚本式相对导入不支持模块入口，已交实现代理修复，尚不宣称全套CPU测试通过。

## 04:28 UTC 更新

- E3 已完成，最终入口：[方向口径修订2报告](../../../results/idea1_v3_2/e3/revision2/report_zh.md)。6345个评估点、全部50 offset与阶段表已齐；两类向量各自的勾股一致性检查通过。独立重放四套件各一条完整轨迹共118点，旧／冻结Kalman和HoldLast与保存预测最大差0。主代理组合运行19项CPU测试通过，并实际查看方向图确认可读。
- E1 首个调度队列已停止、服务已回收。权威补充收据为 `e1_20261022T0421_batch/rescue.json`：近似最大B4配置8/8、43.507秒；B1后置对照8/8、38.682秒，episode吞吐比0.889。实际batch大小为1/2/3（114/40/15次），没有凑满4。该配置不采用。
- E1 仍在60分钟原始预算内，针对40ms等待这个明确未测试的旋钮开启新的独立cohort：`results/idea1_v3_2/e1_wait_scan_20261022/`。GPU3启动前空闲检查通过；依次最大B2／wait0、最大B4／wait5、B1／wait0，同服务同4worker同8场景，单列结果，不与旧cohort冒充严格同服务比较。
- E2 GPU3 首次加载被保守中止，日志保存在 `e2_probe_gpu_race_blocked.*`。先前不同PID的占用随后随本方退出消失，**可能是host/container PID映射，而非已证明的外来进程**；保留为资源身份不确定事件，不算策略失败。
- 04:27:16 UTC 主代理检查GPU2、3均无compute PID、1MiB；分别授予E2和E1。E2新路径 `e2_probe_gpu2.{json,log}`、本方PID3208554，构造模型前重新检查指定GPU2空闲。已核对实际UUID；不以GPU利用率或PID命名空间差异单独判断可抢占。
- E2 原生控制修正为直接调用原 `Policy._sample_actions` 的动态 `num_steps` 参数，另增加 canonical `Policy.infer` 对照，避免只用两个新封装彼此比较。

## 04:42 UTC 更新

- E1 全部结束：48条有效执行episode（固定8场景在不同臂重复，不是48独立场景），最快短等待配置实际动作／请求吞吐增益不足1%；不切换默认B1。详细报告与图：[E1报告](../../../results/idea1_v3_2/E1_BATCH_REPORT_ZH.md)。全部本方E1服务／controller的8个PID由主代理再次检查均不存在，原始失败收据和中断调度器未伪装成完成。
- E2 native门禁完成24点：每步刷新对canonical D10和D2的raw／decoded最大差均为0，实际10次Euler／2次Transformer，跨请求隔离最大差0。主代理已从原始JSON重新汇总确认。
- E2 residual0/3新闭环40条：34/40；与历史Full40逐条核对初态／BDDL／控制器／原生源／checkpoint身份／第一帧观测／噪声和seed全部兼容，Full为34/40。31条双方成功，3条Full独有成功，3条residual独有成功，3条双方失败。
- 达到预先约定的有希望质量触发条件后，补充原生均匀D2同40场景，不另搜参：35/40。三个条件的Spatial/Object/Goal/LIBERO-10分别为Full9/10/9/6，residual9/9/10/6，D2 9/10/10/6（每格分母10）。旧Full不计新增episode。
- E2新增80条质量episode、8个新视频；two-worker even/odd分片共用串行模型服务。初次compile单列于RPC运行过程，不混入独占稳态计时。
- 04:42:04 UTC E2最后D2服务完成shutdown并reap，worker退出0，owner确认GPU2无本方compute进程；主代理检查7个已知E2模型／客户端PID均不存在。当前只有CPU图表／媒体索引汇总，无本方GPU实验运行。
- E3源数据实际为历史Full18_D3_R5_FORMAL路径（原始infer中denoise_steps=3），并非本轮E2的D10；报告已明确两个实验口径。
- E4仍只有资格调查，无任何策略episode，不计成功率；用户尚未确认零样本迁移定义。

## 04:49 UTC 汇总与独立检查

- E1–E3 的GPU／CPU计算、汇总表、图和原始数据均已交付：[当前完整报告（E4尚缺）](idea1_v3_2_report.md)。20项组合CPU测试由主代理独立运行通过。
- E2详细交付：[中文HTML图表／视频入口](../../../results/idea1_v3_2/e2_report_zh.html)、[JSON汇总](../../../results/idea1_v3_2/e2_summary.json)、[生命周期与预算](../../../results/idea1_v3_2/e2_lifecycle.json)。12条固定视频为4条旧Full＋8条新residual/D2，不把旧Full算新跑。
- 主代理已核对原始门禁与实际80条新result的唯一场景键、分suite成功数及8个新视频数量；实际查看E1对照图、E2成功率及Object XYZ图、E3方向图，未使用相机截图冒充轨迹。
- 主代理另对12个固定视频逐条完整ffmpeg解码，核对文件SHA、H.264/yuv420p、20fps及帧数＝执行动作数＋初始帧，全部通过；E2 gate当前源码及E3修订2源码均匹配各自收据SHA。报告／方案／状态／资格文档与HTML当前本地链接检查无缺失。
- E2“开头1–5步真实EEF差”修正为仅执行后5点，不再把初始零差也平均进去；残差2.114mm、D2 1.844mm。明确不是每次重规划的短前缀安全验证。
- E4阻碍仍是比较定义：是否采用带清晰局限的CALVIN零样本迁移（加入Full），或等待匹配checkpoint。未获得用户反馈；没有擅自新增训练、随意截取PushT动作或记作0%任务成功率。
- **整轮goal尚未完成，不应标记complete。** 当前没有本方GPU实验在后台继续运行；等待E4定义确认后才能继续该项。

## 05:00 UTC：E4进入适配准备

- 新的真实CPU证据：[指令预检](../../../results/idea1_v3_2/e4_prompt_preflight/cpu_native_sp_v1/preflight.json)。使用本地完整BF16词表和原生SentencePiece，10个发布LIBERO指令与旧embedding逐位一致，全部8个CALVIN真实指令完整有限；5项测试通过，只materialize49个词表行，无GPU／模型／环境执行。
- 原语言阻碍是旧manifest适配器限制，不是缺模型词表。主代理重新审查授权边界：用户原文已要求跨环境实验及补充设计，先前询问属于口径偏好而非权限阻碍，因此采用[冻结零样本协议](idea1_v3_2_e4_zero_shot_protocol.md)在原范围继续，不凭自己的早期确认门禁永久缩小目标。
- 没有收到用户对额外选项的回复，不作虚假确认声明。保留Full对照、全部四个发布draft head，不训练／换权重；机械臂环境做物理接口对齐，PushT单列人工二维投影压力测试，禁止因低成功率倒推模型容量因果。
- 三个原6.1-sol medium实现代理分别负责新policy/server、三环境bridge/client、冻结manifest／调度／汇总；现阶段仅CPU代码，尚未授权GPU构造或策略动作。
- E1–E3结果及原时间预算不改变；E4另设90分钟探索边界。现在仍不能宣称v3.2全部完成，但已不再仅等待先前的可选口径问题。

## 05:10 UTC：E4原生接口检查与资源分配

- CALVIN全部8个标注初态和PushT全部5个固定seed均已在各自原生环境完成reset／一步接口检查；全部检查点的原生成功判定为false，未把初始即成功计为策略能力。此检查没有运行模型，仍不是任务评测成功率。
- 无相机、无GPU的实际LIBERO环境提取了MountedPanda的基座坐标`[-0.66, 0, 0.912]`和真实state8；对齐使用该事实参考，不以末端位置猜测机器人基座。证据在`results/idea1_v3_2/e4_cpu_adapters/source_reference.json`。
- 新策略接口CPU元数据检查完成，保留原Full／V14和全部四个发布draft head；新指令按原生词表构造，后续还需真实GPU旧路径输出一致性检查。
- 主代理05:10实际查询：GPU0／1／2均有已有compute进程和高显存占用，GPU3为1MiB且无compute PID。只授予GPU3先做Simpler渲染绑定检查，再交给模型门禁；没有使用已有其他任务的卡。调度器改为支持一张或两张空闲卡，不因只有一张卡而删减108个预定条件键。
- 发现调度器初稿直接比较NVIDIA host PID与容器PID，会错误拒绝本方服务；已要求在冻结前修正为保留实际身份与设备证据的检查。所有进程清理仍限定精确的本方子进程及启动时间，不发送信号给NVIDIA所列未知PID。

## 05:14 UTC：E4冻结并启动

- 最终三环境全部18个初态接口预检通过；CALVIN corrected base、Simpler现版SAPIEN实际指尖pose及PushT原生成功阈值的收据已统一到最终源码hash。初始／一步noop均未成功。
- 冻结根目录：`results/idea1_v3_2/e4_transfer_20261022T051417Z/`；manifest hash `3eeeb50e43b1078a61d9015bbb0fe371ff5f4bd98809e0242699de700369222f`，固定108个案例／条件键，GPU3单分片。
- 启动前GPU3再次验证无compute PID、1MiB。controller PID3250832、policy service PID3250838、端口18641；主代理用实际`ps`确认二者活跃。服务先执行全部六条件的真实LIBERO旧／新路径门禁，未通过不得进入跨环境质量调用。
- PID检查修订保留容器PID、NSpid、NVML host PID和UUID；不能直接匹配时限定为启动前空闲、启动后唯一新compute进程、精确本方活子进程及相同设备健康响应的联合证据。仅清理本方Popen子进程。
- 主代理独立组合CPU测试44项通过，后另独立运行修订后的编排7项测试通过（新增host/container PID测试与单卡冻结测试）。这些测试不替代真实GPU数值门禁或闭环任务结果。

## 05:16 UTC：E4真实模型门禁通过

- 主代理独立读取真实GPU服务收据：六个条件的raw `[50,32]`动作均与旧路径逐位一致；解码`[50,7]`最大差不超过`2.3841858e-7`，通过`1e-5`门槛，但不称解码逐位一致。
- 实际函数路由计数：Full／V14每请求vision1、VLM1、AE layer180、Euler10；四个纯draft均vision1、VLM0、AE0、learned head1、verify0、隐藏Full录制0。启动／门禁和质量类别分别记录。
- 服务状态ready，首条CALVIN Full闭环已开始。此刻仍没有完成的跨环境成功率；不会用门禁通过替代任务结果。

## 05:26 UTC：E4首批配对记录独立核对

- 本方controller／模型服务仍活跃。主代理重新校验冻结源码及所有消费收据hash，检查当时已完成的13条有效episode、910次质量请求。
- 已完整跑完的CALVIN case0和case1，六条件各自的初始原始RGB／state8 hash完全相同；case2当时仅Full完成，不将未齐的分母冒充完整配对。
- 逐请求核对mode、quality类别、首帧hash、实际vision/VLM/AE/head/verify路由、10次或0次Euler计数、同case/request噪声种子及hash，异常为0；真实遥测点数等于执行步数＋1。
- case0六条件均未成功；case1仅V14在230步由原生任务checker判定成功，其他条件均到360步cap。这是小样本观察，不是已完成的环境总体成功率，更不是模型容量因果证据。

## 05:49 UTC：CALVIN完整矩阵完成，转入SimplerEnv

- CALVIN的8案例×6条件全部为有效原生闭环结果，无技术失败：Full0/8，V14为2/8，draft_object为1/8，draft_goal／draft_spatial／draft_10各0/8。
- V14的成功为case1开灯（230步）和case6向右推粉色方块（38步）；Object draft也在case6成功（29步）。成功使用原生任务checker，不是动作相似度或reward替代。其他条件同一case的初态与指令保持不变。
- 不能得出“draft换环境必定失败”：已有一个draft原生成功；同时Full正对照自身0/8，且只有本地debug 8个语言段、存在跨机器人／视角／控制频率等混杂，不能据此归因模型容量。没有为改变结果追加调参或训练。
- SimplerEnv已开始并产出有效80步原生截断结果；PushT尚待后续运行。不能把CALVIN完成替代整个E4完成。

## 05:58 UTC：SimplerEnv完成，PushT启动路径失败后续跑

- CALVIN48条和SimplerEnv30条均为有效闭环；Simpler六条件各0/5，所有失败由原生80步终止／截断记录，不是技术异常。
- PushT的30个attempt000在环境构造、第0步、首次策略调用之前全部报`ModuleNotFoundError: diffusion_policy`。主代理从全部原始result聚合确认错误一致；这是编排器没有继承CPU预检使用的本地源码路径，不能计为PushT策略0分。
- 原controller PID3250832和service PID3250838已正常退出。原lifecycle的`complete`仅表示派发／清理结束，不证明108条质量评测完成；当时有效质量结果只有78条。
- 授权仅修正启动环境的`PYTHONPATH`，指向现有`policies/diffusion_policy`，以同一冻结清单续跑；不修改任何冻结代码、模型、接口映射、任务或seed。新启动前须重新核对GPU3空闲；保留所有失败attempt000，仅新增attempt001，跳过原有78条匹配有效结果。
- 最终汇总需同时呈现108个有效条件键和30次历史技术失败尝试，模型启动／重复门禁不计质量样本。不能把续跑覆盖成从未失败。

## 06:00 UTC：PushT续跑已产生有效策略结果

- 使用原清单和补全后的继承`PYTHONPATH`启动：新controller PID3277552、service PID3277568，生命周期`lifecycle_1792648770542312134.json`。GPU3启动前重新验证空闲；没有修改13个冻结源码文件。
- 主代理确认新进程实际存活，第二次六条件真实模型门禁通过，raw逐位一致、decoded最大差仍为`2.3841858e-7`。PushT首个案例的Full／V14分别完成300步，均为有效任务失败；不再是环境导入异常。
- 当时有效episode80/108；原有78条未重跑，30个失败attempt000完整保留，PushT新增attempt001。科学配置、坐标映射及种子规则均未改变。
- 同期主代理重新核对E2的1个源码／1个输入／14个图表媒体索引产物hash，及E3的6个源码／5个父级输入hash，全部一致；这不替代尚待完成的E4全量核验。

## 06:17 UTC：整轮完成与主代理最终核验

- E4全部108个唯一有效条件键完成；总138次尝试，另30次历史PushT第0步导入技术失败保留。CALVIN Full0/8、V14为2/8、Object draft为1/8、其余head各0/8；Simpler与PushT六条件各0/5。未将PushT二维压力测试与机械臂任务混算总分，不支持模型容量因果结论。
- 主代理独立读取全部108个初态NPZ及完整遥测／推理日志：18案例六条件初态一致；5,580次质量请求、27,897个执行动作与预测chunk逐项精确匹配，28,005个实际位置点覆盖完整；956个请求索引的跨条件噪声身份一致。D10节点、实际AE／VLM／head路由和无verify／回退逐请求检查通过。
- 两次服务计数与全部选定质量日志一致；续跑确实跳过78条旧有效结果。全部冻结源码／快照及消费收据hash仍一致，新只读post-audit源文件hash匹配；独立审计异常为0。
- 主代理完整解码18个E4固定视频，核对H.264/yuv420p、帧数＝动作数＋1、30/3/10原生fps，全部通过。E2此前12个固定视频独立解码已通过，当前图表／索引及E3源码／输入hash再次核对一致。实际查看E4成功率图、固定XYZ／XY和事后case6投影，未用截图或动作累加冒充位置。
- [最终报告](idea1_v3_2_report.md)及新E4两份报告／状态／方案／README的当前本地链接均无缺失；E4 HTML中40个本地媒体链接全部解析到存在文件。核心CPU组合测试45项通过，不冒称这些测试替代真实模型门禁。
- 本方两轮E4全部子进程以及已知E1／E2控制器、模型和客户端PID均不存在；GPU3最终compute列表空、1MiB。无剩余科学计算或待派任务，没有向外来PID发送信号。
- [主代理最终验收收据](../../../results/idea1_v3_2/e4_transfer_20261022T051417Z/parent_acceptance_final.json)逐项对应原始E1–E4要求及证据。实验计算与交付已完成，后续建议仅写入报告，没有另行开跑。
