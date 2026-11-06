# idea1 v3 执行状态

更新日期：2026-10-20（Asia/Shanghai；下文attempt时间戳保留原执行记录）。**本轮既定实验与交付已完成**：联合1976条唯一有效episode、P0有界测量、七代表测速、88固定新视频检查、完整父对照、两张总览与中文终稿齐。所有科学／bench进程和活动owner flock已退出；无待派实验。设计依据为 [v3 方案](idea1_v3_experiment_plan.md)，结果见[中文终稿](idea1_v3_final_report.md)及[衍生指标](idea1_v3_derived_metrics.md)。goal最终状态由root验收后更新。

## 角色与边界

- GPT-6.1-sol low：模型／执行代码实现，提供真实 CLI 与有限 CPU 检查。
- GPT-6.1-sol medium：本记录、轻量需求覆盖与必要验收；不编写模型／执行代码。
- Luna Max：GPU 唯一执行 owner，负责真实启动、设备绑定、调度和实验结果。
- 保留用户 `idea1.md`、旧 v2.2 冻结源及结果；不恢复旧队列。已有 π0 Full/V14 对照只复用，不为了配齐新对照重跑。

## 当前阶段与放行

| 阶段 | 状态 | 下一步／证据门槛 |
| --- | --- | --- |
| 实现准备 | complete | 代码／增量／分析／安全续跑／总览入口齐；CPU与真实接口通过，总览实际渲染及有限视觉验收通过 |
| P0 | complete（有界范围） | A/B 8/8、共同4条、micro 100及残差产物齐；保留A、不宣称提速。仅NPZ候选，活动/磁盘等待缺测，micro非纯热态 |
| E1 | complete（安全revision） | 校准96独立；主800有效，D2选中185/200、参考178/200，D10参考176/200、选中172/200；唯一invalid同key一次有效恢复，不重跑valid |
| E4 | complete | 200/200有效、137成功，Full/verify均0；历史8对XYZ／视频索引、新8视频各一次decode、代表bench与中文解释齐 |
| E2 | complete | 680有效齐；Full/transfer16各196/200、transfer15 188/200，V14 33/40、V13 32/40；3代表bench／固定40视频／32对本轮Full真实XYZ齐 |
| E3 | complete | 200有效184成功，各suite49/47/45/43（各50）；完整父E1为185/200，差−0.5pp、恢复8／新增失败9；独立bench／固定媒体齐 |

首阶段真实命令已实现：在工作区根目录运行 `python3 -m preparations.idea1_v3.cli prepare --attempt <new-attempt> --stage P0 --launch`，再由GPU执行owner运行 `python3 -m preparations.idea1_v3.cli worker --attempt <new-attempt> --shard 0 --gpu <physical-index> --launch`。`stop --attempt <attempt> --launch`仅停止新派发；`resume --attempt <attempt> --launch`后重跑同worker恢复；`summary --attempt <attempt> --launch`生成摘要。具体attempt、GPU、PID和启动时间由执行者实测后补记。

## 需求覆盖与必要验收

| 范围 | 必要检查 | 当前证据／结论 |
| --- | --- | --- |
| 历史对照 | 限时索引；H/R、D、mask、checkpoint与key匹配；不重跑π0 | `preparations/idea1_v3/readiness/historical_controls.csv`已冻结；D10非配对、D2共同120仅描述参照 |
| E1 Full KV | 最近Full独立存储；V14不覆盖；reset清空 | 源码clone／恢复、CPU Full→draft→verify与reset检查通过；真实P0/校准已执行 |
| E1 verifier | t=.10/.05共享噪声并行；最新state；前6维双分支前缀 | 实际Triton batched verify与exact时间输入源码、CPU检查通过；真实调用与误差日志已产生 |
| E1路由 | R≤5；零接受不执行被拒draft，最终Full才推进 | server/client实控流及CPU拒绝／ack检查通过；校准96有效，真实主样本Rmax5 |
| E1计数/RNG | P按draft轮；Full重置；generation/verify域隔离 | 源码与CPU通过；累计counter用末值／差分，不sum各请求累计值 |
| E3历史 | 前轮state锚、已执行r步、H50、只比未来5、不重执行历史 | r1..5／拒绝排除／reset CPU通过；真实首条r5、前轮anchor、dist[2,5]核实 |
| E3坐标 | 原decoder归一化／absolute-delta可逆重锚；实际clip记录 | 真实output_transform仿射逆CPU通过；首history与telemetry clip后命令精确一致 |
| E4纯draft | suite权重；真实vision→PrefixContext→learned；Full/verify=0；R≤5 | 全200有效且计数均零；正确四suite meta与无history依赖核实；D=N/A |
| E2 | 真实π0.5／原生H/D／18层mask／完整AE | native H10/D10/R5；tiny JAX suffix parity与五条件共200真实首筛通过；AE编译结构性能混杂已标 |
| P0 | 真实输入reset；≤4配置；原生timeout；完整成本／残差 | 2配置8闭环＋100微基准＋8行残差齐；保留A，不宣称加速／纯热态 |
| 调度/保存 | 独立分片日志／GPU-EGL绑定／唯一key／持久后完成 | 真实identity和CLI已落；NFS同步合并写、complete后提交；新增量只7..10，冻结manifest不改 |
| 结果 | 有效失败保留；首40汇报；固定视频；轻量记录 | 联合1976有效覆盖、各stage独立分母、最终父配对／历史描述表、固定XYZ、88新视频各一次decode、两总览与终稿均齐 |

验收采用必要的新源码检查、有限CPU状态测试和首批结果结构检查；不建立多级审计、不全量重hash旧raw、不重解码旧视频。

## 交接记录

最终验收记录：Luna确认联合analysis session97003退出0、1976/1976有效且9个attempt覆盖完成；overview同输入最终重渲session72119退出0，root实际查看两图通过。完整E3父配对184对185（−0.5pp，恢复8／新增失败9，探索性区间[−3.5125,2.5]pp），不沿用首40提升结论。七代表各20setup＋100测量、88固定新视频各一次全帧检查齐；固定遥测事件表及共同120历史描述已交付，未重跑旧π0基线。所有科学／bench进程和活动owner flock退出，保留锁文件／原invalid日志作为历史证据。唯一技术invalid已按同key一次恢复计入有效覆盖，无待补科学样本。最终入口为[中文终稿](idea1_v3_final_report.md)和[衍生小表](idea1_v3_derived_metrics.md)。

以下按发生顺序保留过程快照；其中“待启动／暂停／待修”等是当时状态，当前事实以上表及末尾新增记录为准。

- 管理者已完整读取 v3 方案及本目录 README，并将方案10.2角色行按本次用户指令修正。
- 已向实现者通知所有权边界和优先验收点。等待首个可运行交付；没有证据支持称P0或任一E已启动／完成。
- 只读确认复用路径为 `preparations/idea1_v2_2/pi0_adapter.py` 的 converted Triton π0 adapter；其 `prepare` 已clone K/V，`sample`恢复K/V及state。此确认不替代新实现状态切换／客户端路由检查，不扩展旧provenance审计。
- 执行者报告历史D10已定位至 `policies/realtime-vla-flash/results/experiment11_layer_skip_prefix_20261010/analysis/d10_followup_final/report.md`：Full 180/200、V14_A03 148/200，task0–9/init0–4、H50/R5，与新init6–10不配对；仅历史参考。D2沿用方案指定v2.2入口，后续核共同key。以上为历史结果，绝非本轮新增实验。
- 用户暂停后，执行者确认未启动任何v3 GPU任务／episode，无session或在途排空；只读readiness证据已交root。实现者保留新草稿 `preparations/idea1_v3/{__init__,policy,io,server,episode}.py`，尚无CLI调度与CPU测试，**不能启动**。
- 暂停前有限源码阅读发现待恢复问题：`TritonBackend.spec`以suite/seq缓存spec，构造器复制language embedding，而`learned`读取该缓存；同suite同token长度但不同prompt需要刷新语言或细化key。已交实现者记录，暂停期间不修不测。新实现尚未完成验收，无P0放行。
- 用户已恢复active，管理者已通知实现者和执行者解除暂停。low确认将每次spec调用刷新当前runtime语言embedding，继续补worker CLI、manifest、P0摘要和fake-backend路由检查。首个E1/P0路径可运行即窄审，不等待全模块。
- root转交readiness：π0.5与四suite learned draft权重均本地具备；当前E2/E4不能以缺checkpoint记资源阻塞，仍需实现与运行验证。根盘约98%满且无专用scratch，首版可用轻量NFS合并写入；不把本地缓存列为放行前提。
- 首阶段新增8个模块 `preparations/idea1_v3/{__init__,policy,io,server,episode,cli,profile,checks}.py`。实现者报告py_compile、两venv导入、默认CLI dry-run及8项CPU检查通过；管理者独立复跑8项CPU检查通过。
- 管理者已窄审并放行E1/P0：实际converted Triton路径、Full K/V独立clone、最新state验证、generation/verification RNG域、P2按draft轮、零接受只返回最终Full动作、客户端R≤5与ack、静态owner、同步写入排空及完成标记。此为启动放行，尚非真实GPU接口或科学结果验收。
- P0首版A=合并压缩NPZ、B=合并不压缩NPZ；单worker串行四suite task0/init0，只有共同完整4条、精确数组一致且B至少快10%才选B，否则保留A。micro计入累计60分钟预算，未完成则注明，不延期补齐。后续E2/E3须隔离实现，已运行控制路径不得热改，必要修复在episode边界切revision。
- **真实P0启动**：执行者报告attempt `policies/realtime-vla-flash/results/idea1_v3/p0_20261019T143600Z`，exec session 21961，launcher PID 1504326、eval worker 1504327、model server 1504332／port40837。物理GPU0 UUID `GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`，CUDA/EGL均0。prepare退出0，manifest为8个A/B episode；服务仍在启动，尚无首predict／完成episode。GPU1他方进程未触碰。
- **P0分析限制（待独立分析修正）**：首版只测NPZ压缩候选，不代表搜索了全部优化；request摘要尚未按execution分组，自动selection暂只算episode成本，不能直接当最终工程选型。须合并startup／首编译／预热与drain总成本；CPU/GPU活动、disk-wait缺测如实标注。已交实现者独立修分析并提醒执行者，A真实诊断不等待此修正。
- **首个真实执行检查**：管理者只读worker events确认startup 30.02s、A前三个episode正常complete（success=true/false/true），第四已开始，server日志READY。任务长度不同，单条墙钟不能作优化加速比较；有效失败保留，P0不混入E1科学成功率。
- 独立`profile.py`已修request按manifest execution分组、补startup与episode总成本、注明嵌套区间和CPU/GPU/disk-wait缺测；首A承担lazy setup且无反向确认，保留A不宣称B更快。micro同一budget已加段边界守卫，未热改正在运行的E1路径。
- 管理者轻量读取A四suite的result/infer：spatial成功81动作/17请求，object有效timeout280/56，goal成功155/31，libero_10成功271/55；均实际Rmax5、Full/V14+verify调用存在、valid=true。四条仅工程诊断，不计E1科学成功率。
- E4首40源码窄审放行：四suite checkpoint身份已入科学key，真实vision→project→PrefixContext→learned block、语言缓存刷新、构造Full图预热被抑制；待首条真实日志核Full/verify=0及startup full_AE_calls=0。E1校准可在P0保留可用配置后启动，不等E2/E3；并行micro的共享CPU/I/O干扰须记录。
- P0 A/B闭环已8/8完成，`profiling/selection.json`保留A：共同4条但output_equal=false、动作数一致性false，首用成本不公平；A/B episode合计77.097/49.550s，加共享startup为107.117/79.570s，只作观察，**不报告压缩优化提速**。管理者极窄定位仅spatial有差：A81/B80动作、均成功，首次Full normalized maxabs .015625／decoded actions .009633；其余三suite首两轮精确相等且长度／成败相同。已交low分析首用初始化／数值状态，尚未归因，不追加规模。
- 同输入有限定位：管理者与low分别确认spatial第0 case保存的所有obs数组、相机、state、prompt和seed精确相等。low只读源码未发现明显KV污染，Full/V14前显式重写并clone；首Full A的vision/VLM/AE=1969.6/426.5/153.2ms，B=13.4/18.9/71.2ms，vision内存在torch.compile SDPA，首用编译数值路径只是候选解释，未证根因。既有micro未存输出；下次必要修订可加hash/maxdiff，不为此扩大GPU补跑，也不要求所有闭环bitwise。
- P0有限micro已保存 `micro_summary.json`：20次预热、100测量，mean247.883ms、p50 63.096ms、p95 1003.561ms。这是四suite首5个真实重规划点、按同序列reset的微基准，阶段多样性不保证，不是闭环平均；low读取repeat1–4完整route（含dist/signed_error）相等，热态验证／路由稳定，Full原始动作未存，不能声称Full全位相等。root已同意继续A科学，不新增大一致性门槛。
- 执行者确认P0有界测量总预算<10分钟，微基准venv session6087退出0，load30.573s；首次system-python调用因缺jax在GPU前失败，未计为科学episode。micro含prompt构建与startup/periodic Full，不能称纯热态服务延迟。E1/E4准备启动，P0不延期扩搜索；独立分析补非重叠区间与未归属残差。
- **科学实验真实启动**：E1 `policies/realtime-vla-flash/results/idea1_v3/e1_calibration_20261019T1442Z`，12条件×8=96，物理GPU2，session19992，PIDs1507908/1507909/1507914，port54485。E4 `.../results/idea1_v3/e4_init6_20261019T1442Z`，首40，GPU3，session16693，PIDs1507941/1507942/1507947，port58539。执行者报告均在模型启动，尚无科学完成条数；待E4首结果核查。
- **E4首条真实检查通过**：spatial task0/init6成功，72动作／15请求，所有route为learned_draft，最后累计counter draft=15、Full=verify=0，startup full_AE_calls=0，draft_spatial权重对应。累计counter不能逐请求求和（15请求累加会虚增为120）；E1各调用率亦用末值或差分。E4结果继承base adapter的D10字段，报告标D=N/A（单learned head），不解释为E4扩散采样。
- **E2窄审放行**：真实π0.5 checkpoint/native transform、H10/D10/Rmax5、明确18层mask、prefix-only隐藏更新绕过并保留original KV shell、suffix完整AE；low tiny JAX CPU检查确认suffix对原生数值parity。GPU0 firstscreen为Full＋transfer16 `[3,9]`／transfer15 `[3,9,11]`／V14 `[3,4,9,11]`／V13 `[3,4,7,12,15]` 各40，总200。首episode直接计入清单，不另重复smoke；初版仅总延迟，VLM/AE分项缺口不阻塞成功率。
- E3已有独立CPU证据：H50/r1..5、前次锚点命令逆变换、拒绝draft不入history、部分执行后anchor更新、reset，及加载真实pi0_libero output_transform的有限affine inverse roundtrip。待E1-D2父配置冻结再启动；已向low核实controller input_clip后的真实历史命令语义，不影响E2/E1/E4。
- E2真实启动attempt `policies/realtime-vla-flash/results/idea1_v3/e2_screen_200_20261019T1444Z`，session89670，launcher1510204／eval1510205，GPU0 UUID同上；worker verify通过，首Full／mask尚无实际结果。
- 轻量进度快照：E1 calibration已有7/96有效、6成功（仅首条件部分覆盖）；E4已有11/40有效、10成功。不是各实验最终结论。E3独立clip修正准备中；P0现存候选窗口9510个前六维值无超界，不能据此假定未来永不clip。
- 后续扩展仅新增init7..10并引用首40，E4补160；E2 Full＋最多2选mask各补160；E1各D校准best＋reference去重，校准和主样本不混分母。已交low补真实增量／选型CLI；禁止手改冻结manifest或原`--expand`另attempt重跑init6。
- E4动作history协议窄核：low提供released `src/openpi/models_pytorch/draft.py:111–113` 的`forward(last_actions)`立即`del last_actions`；converted `pi0_spec_infer.py:_run_draft_block`仅使用prefix/state/query、无history输入。v3沿用该released转换head，不人为新增history；外围history_len6缓存不能称为此head实际依赖。不需要更改正在运行的E4。
- 独立`advance.py`／`advance_checks.py`已交付并由管理者复跑CPU通过：E1校准全96有效后按成功数→policy_ms/实际动作→tau/P选择各D best＋reference去重，先各40；独立e1-expand仅7..10。E4/E3 expand强制源首40实际完成，仅新增160并引用源；E2完整screen选Full＋最多2mask，仅新增7..10。原科学manifest不改。
- E3独立clip修正通过CPU：HistoryPolicy保存sent和实际OSC input clip后的effective命令，专属history_client验证bounds=[-1,1]；实际hist prefix使用effective，保留可逆重锚。管理者复跑r1..5/拒绝排除/reset通过，待父配置冻结后可启动。
- 最新轻量覆盖：E1校准16/96（D2 tau.10 P2与P4各7/8成功）；E4 21/27有效成功，尚未首40；E2 Full12/40有效且成功，未进入跳层条件。没有提前冻结选型。模型启动按配置重载开销（E1约30s／E2首103s）单列为工程成本，当前不为常驻重构中断科学执行。
- **E4首40平衡初表完成**：session16693退出0、analysis退出0，40/40有效；spatial9/10、object8/10、goal6/10、libero_10 4/10，合计27/40（67.5%）。`routing.csv` Full=0／verify=0，draft实际请求分别236/337/368/857；D=N/A。8张固定init6真实XYZ图已生成；初表尚未历史配对，不能声称相对Full恢复／下降或延迟ratio。GPU3自然释放，已授权新160只补init7..10。
- 独立analysis历史XYZ reader窄审通过：仅从v2.2 trajectory_index读取固定8key Full-D2遥测作蓝色视觉参考，标题明确不是Full-D10／revision差，历史rows不进入新成功率或CI。不扫描旧raw，不为缺图重跑。E2用本轮Full、E3用冻结父条件。
- root补充只读历史参照：v2.2 trajectory_index筛选`condition=pi0_full_u50,init=6`得到同40任务初态Full-D2=35/40（spatial10/10、object10/10、goal9/10、libero_10 6/10）。可与E4新27/40列“同任务／初态历史参照”，历史revision／limits差仍存在，不进新主表、不作正式配对CI或因果速度ratio。low独立报告保存filter与来源；Full-D10旧180/200另为init0..4非配对参照。
- **E4增量实际启动**：`policies/realtime-vla-flash/results/idea1_v3/e4_expand_init7_10_20261019T1454Z`，160key仅init7..10，source仅首40attempt；manifest SHA `a9ec31635eec9bc8213c92bc81f5cf948f871725c8dac4ca1de8bded5fafe914`。GPU3 UUID `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，session69641，PIDs1516756/1516757/1516780，port36773。未重跑init6，待新首episode结果。
- E4增量首条spatial task0/init7有效成功，86动作／18 learned_draft请求，末计数Full0／verify0／draft18；GPU3 CUDA/EGL均3。后续按队列继续，不新增重复smoke。
- **遥测解释边界**：low只读确认复用`trajectory_replay._robot_contact`遇异常会返回False但未保留availability，因此contact/grasp负值不能单独断言真阴性。真实XYZ／物体距离可用，phase有源码规则但最远阶段不等于致败阶段；最终中文报告注明，不为此补跑。XYZ图层后续用稀疏stage短码／markers，保留原telemetry，不反复重绘旧8图。
- 独立phase分析已补真实availability未知分母：episode_phases保留telemetry_observations、unknown_steps与contact/grasp未知步数；phase_routing标未知request，缺证据阶段映unknown、保留原heuristic telemetry。图／表明确负值不是真阴性、不作精细因果。low以真实E4首40 CPU分析通过。
- **后续有限测速**：自然释放卡后，最多8个新增代表条件、每条件约100独立测量；不逐tau扫全表、不重测已有π0基线。优先E1各D选中／必要参考、E2本轮Full＋选mask、E3、E4；low补独立E2 VLM/AE分项后窄审再由Luna统一执行。分模型初始化／请求路由／每实际动作成本／闭环吞吐报告，注明共享CPU/I/O。当前三路既有科学worker继续，不为新增测速中断。
- 独立`bench.py`／`pi05_bench.py`已窄审放行待自然空卡：20真实输入、20setup＋100测量、五点段reset、action/raw hash及参考测量maxdiff／route／长度检查。E1/E3 virtual ack只解释微基准，不冒充实际执行历史／闭环路由；E2 image/VLM/完整AE分解是额外JIT同步边界diagnostic，不替native总延迟、不采用科学执行，预先给定atol/rtol .02与native decoded比较。超容差仅暂停分解分析，不使既有成功率无效。测量仍可含prompt/shape runtime重建，不称100纯热态。
- E4已加载权重meta进一步确认四suite均chunk_m50/out_dim7/use_last_actions=False/use_seed_actions=False；spatial的draft_history_len6不代表实际head启用history。meta中的eval_exec_steps12只是训练验证设置，真实客户端仍Rmax5；没有history初始化遗漏需补跑。
- **P0残差产物已实际补齐并窄审通过**：仅summary CPU重跑退出0，未重跑worker/micro。`profiling/episode_wall_decomposition.csv`共8行，episode wall126.647s＝known_nonoverlap124.241s＋未归属2.406s，单条残差0.093–0.498s。envcreate/reset/predict RPC/step含capture/video drain/save是主线程顺序区间；telemetry和GPU嵌套项未重复相加。ack RPC／结构检查／result+complete原子同步等留在残差，startup30.02s另列。P0在有界范围内完成，限制仍保留，不触发新GPU测量。
- P0这8例已知分解：RPC62.834s（49.6%）、envcreate14.980s（11.8%）、reset13.387s（10.6%）、step+capture32.052s（25.3%）、save0.988s（0.8%）、残差2.406s（1.9%）。存储不是这8例主瓶颈；RPC包含cold setup，不能等同GPU计算，模型部分即使理想2倍也不使整轮2倍。后续自然新attempt可考虑同task相邻init减少prompt重建、单活跃env复用减少create；这是建议，当前immutable队列不改，不追加长优化工程，采用需low实现后有限验证。
- 分解覆盖限制：P0为init0，按预定固定视频规则不写MP4，因此video_drain=0只是未覆盖视频，不能说视频没有成本；科学首init6的task0/5各suite固定8/40才保存视频。save0.8%仅这8例新合并数组／日志流程，不泛化到v2.2全视频／全hash流程。
- 最近执行者事件快照：E1校准80/96（剩D10 tau.25 P2/P4）、E4增量73/160、E2首筛105/200（Full40＋transfer16 40＋transfer15 25）；GPU0/2/3均科学worker持有，GPU1他方，不轮询本方GPU等待。E2真实transfer16全40完成，首mask live接口通过。
- **E2 counter语义**：`Pi05Policy`对所有mask返回`counts.full=round+1`，其含义是该mask的native policy调用数，不是未剪Full或失败回退。已交low仅修analysis为native_policy_calls＋实际mask，Full fallback=N/A；不能用该列推跳层的Full接管率。E1/E3/E4仍按各自真实路由counter解释。
- **E2性能归因限制已确认**：nonempty mask的patched llm对suffix虽执行完整18层、CPU parity通过，但采用Python展开Block；Full空mask返回原native nn.scan，AE编译结构不同。端到端差混合VLM跳层与AE编译开销，不能全归VLM剪枝。analysis／pi05_bench已独立标ae_compilation_structure和latency_attribution；当前科学服务不热改、有效质量样本不重跑。未来suffix直接original可作独立revision，本次未实施。
- root随后批准有界bench-only修正，现已交付并窄审：`pi05_bench_v2.py`通过显式Flax `method=gemma.Module._v3_original_call`仅suffix走真实native scan，prefix skip／KV shell不变，无全局方法切换、不改科学服务。low tiny18 CPU检查native suffix精确parity、prefix/cache不变且jaxpr有scan。Luna自然空卡测速指定`--pi05-stages --pi05-stage-revision v2-native-scanned-suffix`替代旧诊断AE分项，仍同一20setup＋100测量预算，不双跑两rev／不补质量episode；旧native总时延及其unrolled结构继续保留，新分项不称已部署总加速。
- **E1校准冻结与主样本启动**：session19992退出0，96/96有效（12×8）；已保存全12候选`e1_calibration_20261019T1442Z/summaries/e1-select.json`。管理者只读复核排序：D2 best tau.10/P2（7/8，31.941ms/action），D10 best tau.15/P4（7/8，35.357ms/action），各保留tau.15/P2 reference，去重后4条件。校准与主样本分母分开，选中不称统计最优。
- E1主样本attempt `e1_main_init6_20261019T1514Z`已新manifest启动160key＝4×40、仅init6、shards1，GPU2 session6296，launcher1529711／eval1529712，manifest SHA `01d1d19824b355bb4dd33b9f14eeeaad69114d4225f2a03d48b04c2e78da8eb7`。E3 attempt `e3_parent_d2_init6_20261019T1515Z`父配置D2 tau.10/P2已prepare首40、shards2并引用E1主manifest，待下一空卡先启动shard0、再空卡接shard1，不等齐卡空置；不必等E1父结果完成才生成本轮科学数据。
- **E4全部预定200有效数据完成**：新160自然结束，独立analysis递归引用源40，输出`e4_expand_init7_10_20261019T1454Z/summaries/all_init6_10/`。成功数spatial46/50、object40/50、goal29/50、libero_10 22/50，合计137/200（68.5%）；Full=verify=0，D=N/A。该数据范围完成，固定媒体一次检查／图和最终中文解释仍随整轮交付，不代表其他E完成。
- E3首40 shard0在GPU3 session54727真实启动，首条spatial task0/init6有效成功、H50/Rmax5、74动作／15预测，父E1-D2 tau.10/P2；待单个history request字段实核，不增加科学key。E1main/E2仍运行，最近执行者覆盖分别44/160与185/200，非最终表。
- **E3真实history接口检查通过**：首条15请求中9次history_verified、6次Full；9次r5、anchor为前轮chunk起点、未来窗口5、接受／执行5，draft/verify每次各增1。管理者只核首history请求（step5、anchor0、r5），history_commands与遥测前5步`controller_command_after_input_clip`精确相等，distance形状[2,5]、future_reanchored[5,32]，历史不计接受误差。该条无zeroaccept样例，不能称真实测过拒绝路径（已有CPU/控制流检查）；不为此额外GPU补样。route缺独立sent来源标签，但同期telemetry保存sent与clip后命令可追溯。
- **E4共同120历史参照已实际保存**：`e4_expand_init7_10_20261019T1454Z/summaries/all_init6_10/historical_quality_overlap.{csv,md}`及逐key表；仅旧索引与新小result。旧π0 Full-D2为109/120，新E4为84/120；四格为双成功79／旧成功新失败30／旧失败新成功5／双失败6。每suite共同30：spatial旧30新28、object30/24、goal27/18、libero_10 22/14。主表仍137/200，历史mixed revision／limits与D2非D10限制保留，四格仅描述，不给正式因果CI／速度ratio。
- **E4历史XYZ对比已实际交付**：执行者CPU调用已审historical flag，`e4_expand_init7_10_20261019T1454Z/summaries/with_history/xyz_index.md`含固定8key蓝Full-D2／橙E4对比，新增`historical_trajectory_references.csv`及共同120参照表；未重跑episode／视频解码。管理者实看spatial task0：3D、三投影、Z-time/stage短码、共同时间距离齐，标题明确D2非D10／视觉参考；原no_baseline图保留，最终引用with_history索引。
- **E2完整首筛与扩展启动**：session89670退出0，200/200有效，五条件各40，Full40成功、transfer16/transfer15各39、V14 33、V13 32。`summaries/e2-select.json`冻结Full＋transfer16＋transfer15；管理者核best／距best≤4/40且跳层最多规则通过，V14距best6/40不入扩展。新attempt `e2_expand_init7_10_20261019T1530Z`，480key仅init7..10、3shards，GPU3 shard0 session37761。init6选型样本与init7..10未参与该次选型的结果须单列，不宣称统计最优／π0.5比不同checkpoint的π0更耐剪。
- E3首40 shard0 session54727已20/20退出0，shard1已GPU0 session56418；E1主样本最近68/160仍GPU2。当前已有manifest不重新分片；后续E1新640按4shards准备以接续释放卡。
- **E4固定视频交付齐**：新E4 init6 task0/5×四suite共8视频各全新decode一次，8份`media_checked.json`均`full_decode_checked=true`，ffprobe帧数=actions+1；历史仅链接、没有再解码。执行者只重生`e4_expand_init7_10_20261019T1454Z/summaries/with_history/media_index.md`／`.html`，paired8行两侧相对／绝对路径及历史source齐，未重绘XYZ或覆盖其他结果。E4闭环／媒体已完成，代表bench及最终中文解释随整轮交付。
- **E3首40与增量**：`e3_parent_d2_init6_20261019T1515Z/summaries/first40_parent_snapshot/`已交40有效、39成功，父E1-D2 tau.10/P2同40为37成功，恢复2／新增失败0，探索性差＋.05、CI[0,.125]。实际每动作policy成本四suite父36.43/25.89/29.58/24.43ms，新E3 39.54/31.50/41.88/27.15ms，质量和成本一起解释，不称加速。8张父XYZ已产。新`e3_expand_init7_10_20261019T1537Z`仅新增160、2shards，GPU0 shard0 session58645，launcher1543735／eval1543736；E1/E2仍继续。
- 六块[中文报告](idea1_v3_final_report.md)已建立进行中版本，已完成证据可查，完整200／bench未齐的分支明确待测，自动英文原始report保留。
- **误差／阶段分析补齐已窄审**：新增future_error_by_phase_step_branch及episode_equal／per_episode表，粒度current heuristic phase×future1..5×K2；仅未来模型归一化误差，非mm／真实EEF误差，历史prefix不入样本。episode等权先episode内均值／RMS，再跨episode均值，不补零；夹爪只报已记录verify/post检查事件和Full原因，不伪造逐步不一致率。low真实E3＋父80条CPU检查receipt为`preparations/idea1_v3/readiness/verification_details_real80_cpu_receipt.json`：2016次verify、20160向量、分K／future计数及已核axis0/RMS/distance等权一致；原临时CSV已清理，不补造原hash。Luna后续既定summary自然产表，不专为此重图／旧视频或GPU。
- **E1主init6完整与增量启动异常**：session6296退出0、160/160有效。新attempt `e1_expand_init7_10_20261019T1551Z`已冻结640key／4shards，仅新增init7..10。首启动session77845退出1，诊断为执行者误用`branches worker`，日志`ValueError: branch server only handles E2/E3`；没有生成episode，不算有效科学失败。保留wrong-launch日志／manifest，改用既有正确`preparations.idea1_v3.cli worker`同attempt做一次已诊断恢复；不改科学代码、不重建manifest、不重跑已完成key。待恢复真实PID后更新。
- E1增量正确cli恢复session76042，root实读identity／ps确认eval PID1552228、物理GPU2 UUID83fd…、inference/EGL均2，worker live；尚待首增量结果，不把仅启动写作已产有效数据。
- E1恢复已过真实模型：startup29.018s、首条成功，随后shard0已有3complete、第四开始。E1首init6 summary `e1_main_init6_20261019T1514Z/summaries/init6_with_history_20261019/`退出0，覆盖独立E1 160／calibration96；四条件各40均衡，D2selected .10/P2=37、D2ref .15/P2=37、D10ref .15/P2=36、D10selected .15/P4=35。已真实生成蓝历史Full-D2／橙各E1图及新future1..5/K/phase、gripper事件CSV；管理者读取新CSV字段和样例核实存在，模型归一化单位／历史排除／obs及episode分母齐。旧D2同init6 35/40仅描述参考，不作正式CI／因果速度。
- E3增量shard0已80有效并自然退出0，shard1在启动前收到调度调整：GPU0先插一条既定代表E2 Full bench（20setup＋100 measurement，v2-native-scanned-suffix），session4083真实启动，之后接E3剩余80；没有中断活跃科学worker／新增smoke／重复两rev。此举早核诊断GPU接口，仍需真实完成证据，其他两卡科学继续。
- **首代表bench真实通过（仅E2 Full）**：session4083退出0，产物`e2_expand_init7_10_20261019T1530Z/benchmarks/representative_E2_Full_v2_20261019T1556Z/`；管理者实读两JSON确认20setup＋100测量，重复action差0、route/length exact，model_load25.835s。native total mean113.452／p50 115.000／p95 123.068ms；v2stage decoded最大差.003234、预定.02abs/rel内，adopted=false。stage均值input51.28／image12.67／VLM35.56／AE20.87／decode1.93ms，sum不替native total；input含host transform＋JAX preprocess/noise及同步，不等同纯CPU。masked代表尚待同样真实容差检查，不能称所有mask已验。随后GPU0已接E3增量shard1 session34715，identity eval1555167通过；E2Full原100预算已用，不重复。
- **首masked代表bench通过（transfer16）**：E2增量shard0 session37761已160有效退出0；GPU3自然边界执行transfer16的20setup＋100测量，session91898退出0，产物`benchmarks/representative_E2_transfer16_v2_20261019T1602Z/`。native mean100.064／p50 96.758／p95 110.153ms，repeated action差0、route/length exact；v2stage decoded最大差.003173、预定.02abs/rel通过，adopted=false。分项input51.281／image12.520／VLM28.090／AE20.553／decode1.852ms，load26.479s，setup20请求均值约3446ms；不将分项sum当native，不跨不同card／共享负载／AE结构把差归少2层。仅此mask实测通过，transfer15待测。
- 随后GPU3已接E1增量shard1，session67945，eval1559529／server1559534，identity CUDA/EGL3、UUID `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`；E1现两卡并行，E2剩shard1/2保持ready，现有manifest不改。
- **E1增量shard0真实技术异常**：session76042退出1，owner0已有87/160 complete＋1 invalid；failed key `c59370bc436ea04da2bffacf7395ce1dc04a70036c998893020ee4600b2d4c39`，D10 tau.15/P2／libero_goal task8/init7。predict的Full startup经`pi0_adapter.sample` Triton `matmul_k8_256_n_softmax_mask0`报CUDA illegal memory access。原failed目录／日志保留，不算有效任务失败、不盲重试；GPU2暂不派新worker，已followup low有限只读诊断。GPU3活跃sh1不热改、不受此暂停。
- E3增量shard1 session34715退出0，160/160新增complete，200总范围数据齐，待合并结果。GPU0按原计划唯一transfer15代表bench20setup＋100/v2继续，独立GPU2故障不阻塞其他分支，未新增smoke。
- GPU2安全只读检查：出错context全退出、4MiB无owner，remap0/pending=no/failure=no；dmesg权限拒绝未绕过，journal无可见NVRM/Xid。无可见硬件故障证据，按root授权已放行不相关JAX E2增量shard1 GPU2/session96411；E1failedkey仍未重试。
- **第三E2代表bench通过（transfer15）**：session54643退出0，`benchmarks/representative_E2_transfer15_v2_20261019T1619Z/`，20setup＋100测量；native mean98.28／p50 95.83／p95 111.77ms，load24.63s，repeat action差0、route/length exact。v2stage最大decoded差.004042、预定.02abs/rel通过，adopted=false；分项input50.91/image12.57/VLM26.16/AE20.82/decode1.78ms不sum当native。三个E2代表各一次100预算齐，不外推不同card／编译结构的因果加速。
- GPU0随即接E2增量shard2 session57393，GPU2 shard1 session96411继续，GPU3 E1 shard1继续；没有因GPU2旧Tritoncontext错误暂停独立JAX或其他GPU。
- **E1有界修复窄审放行（待GPU）**：low新建`safe_qk.py`／`safe_server.py`／`recover_one.py`及CPU边界检查，不改live／冻结源。原QK tile load缺Q行<M、K行<N mask是确证缺陷，失败形状M408/N571/K256推导有额外Q8/K5行读取；是否此次异步illegal根因仍是候选。新核仅加行mask、保持tile32/32/64、grid、FP32累加、scale、BF16/store，合法输出数学不变；CPU地址界限／整数tiled dot通过，管理者源码对照通过。首真实QK一次与TF32关Torch FP32参考比，预定rtol .0078125／atol .0009765625，失败停止不回unsafe核。单原failedkey新context、原invalid核验、拒valid、一次guard、旧dir保留，已放行待自然空卡；再失败不第二重试。剩余E1续跑需独立safe worker入口，不直接unsafe cli resume，活跃sh1不热改。
- **E3完整200已纯CPU汇总**：`e3_expand_init7_10_20261019T1537Z/summaries/all_init6_10_cpu_20261020T1625Z/`退出0，E3为184/200（92%）：spatial49/50、object47/50、goal45/50、libero_10 43/50。source引用产生的E1 main160／calibration96按stage另列，不能把stdout总valid456当E3分母。父配对当前仅init6 exact40 snapshot（＋.05／探索CI[0,.125]），未伪配init7..10；最终200common等待E1expand齐后显式联合两expand。phase/future/gripper实际CSV已产，无--xyz／视频check，既有固定图不重绘。
- **安全续跑入口已条件放行**：新增safe_worker／safe_worker_server／safety_runtime与bench显式安全开关，真实模型子进程factory前安装bounded核。门槛是单failedkey retry状态valid＋QK参考passed＋同safe_qk SHA＋实际complete；原manifest/owner/锁/skipvalid不变，未解决invalid阻止自动续跑。E4不为触发检查额外调用Full，无QK则not_exercised。旧三条JAX E2bench不动／不重测。
- root批准E1活跃sh1在episode边界停止新派发：确证旧QK越界UB后不再继续unsafe整shard，但不kill／不中断在途／不热改，保留所有有效结果。Luna用既有STOP待自然退出释放GPU3，先单failedkey一次恢复；成功后既有resume清STOP，仅safe_worker同owner续余key，不能误启旧CLI。E2两JAX卡不受影响；若恢复再invalid即停，不第二盲试。
- **单key安全恢复真实通过**：E1sh1 session67945在当前episode完成后自然退出0；GPU3无owner后`recover_one` session15607退出0。管理者与root实读QK参考receipt：实际Q[408,256]／K[571,256]／seq520，TF32关Torch FP32参考，maxabs .015625、预定rtol .0078125／atol .0009765625通过，unsafe legacy未执行；guard status=valid／success=true／同核心SHA，complete/result valid=true，旧目录保存`.invalid_before_qk_bounds_…`。这说明已确认越界UB被排除且本key恢复成功，不声称复现证明唯一crash根因。已解除安全续跑条件，Luna清STOP后仅safe_worker owner0 GPU3续剩余有效清单，科学参数/key未改，技术invalid旧尝试单列。
- safe_worker已实际产有效结果；root低频实读其独立首QK receipt `qk_reference_safe_worker_1792427624577976749.json`，实际Q408/K577、maxabs .03125、预设容差passed=true／unsafe=false。该receipt继承safe_qk硬编码scope“one failed-key recovery”，但实际是safe_worker首请求，以文件名／上层safe_worker_revision界定；仅文档说明，不改核心SHA／不重做验证以免使已通过gate失效。
- **其余固定媒体尾项已提前完成**：单路依次E1主init6四条件32视频（session45689）、E2首筛五条件40（80117）、E3首init6固定8（59018），均exit0、ffprobe＋全帧ffmpeg decode，分别32/32、40/40、8/8新receipt。各源／增量attempt的`summaries/media_tailcheck_fixed_20261020/media_index.md/.html`实际存在；E3视频源在`e3_parent_d2_init6_20261019T1515Z`。加E4既有8，共88个新固定视频各一次检查齐，E4未重做、历史Full仅链接未解码，不遍历全episode视频。
- **最终总览代码已交、尚未运行**：独立overview.py仅读每stage唯一完整summary的CSV与7代表bench小摘要，已窄审stage过滤／sumS/sumN／sum policy_ms/actual_actions／replay按request加权／不同模型和计时边界分panel。E2诊断input/image/VLM/AE/decode五项独立点，不sum为native total；input含预处理/noise/同步非纯CPU。fixture与compile通过，数据齐才实际一次产两组合PNG和输入receipt，不把代码存在当图已交付。
- E1安全owner0 session10558退出0，manifest唯一key覆盖160；原恢复key已在其中，不额外加一。GPU3首D2代表bench session60249已实际有20setup＋100测量，ownQK实际Q408/K577、seq526、maxabs .015625、固定容差passed／unsafe=false。scope硬编码仍按上层bench路径说明，不改核心SHA；root实读route CSV显示100测量含20个序列startup/runtime（均值约1027ms）、55 verified（62.14ms）、20 periodic（113.89ms）、5 zeroaccept（177.92ms），不称全热态、不用重放频率代闭环。其real_closed_loop_cost为E1init6父40快照27.696ms/action，非完整200，最终取全量analysis。
- E2增量shard1 session96411已160/160有效退出0；GPU2自然接safe E1 owner1 session93341，跳过原94有效、仅余66。GPU0 E2 shard2继续，其他未跑E1 owners按原清单接续。
- **E2全680实际汇总完成**：last shard2 session57393退出0；`e2_expand_init7_10_20261019T1530Z/summaries/all_680_cpu_20261020/`session19404退出0、680unique。Full196/200、transfer16 196/200、transfer15 188/200，未选V14 33/40、V13 32/40。init6选型40/39/39，新增init7..10分子由聚合减screen且coverage=480确认：156/160、157/160、149/160。配对200 t16净0／恢复3新增3／探索CI[-.02,.02]；t15净-.04／恢复3新增11／CI[-.075,0]。低层因果／跨checkpoint耐剪不由此证明。固定40视频已查，管理者有限查两个E2summary目录未见XYZ索引／PNG，正核另有产物或补原计划固定32对图，不重绘既有图／不再视频decode。
- E1 D10代表bench session95400已退出0、20setup＋100测量，ownQK passed、repeat action差0／route/length exact；GPU3随后E3代表session61434。E1其余safe科学队列继续，不重复已完成两个E1代表。
- **七代表测量全部完成**：E3 session61434、E4 pure session72546均退出0，各20setup＋100测量，repeat action差0／route/length exact；E3 ownQK passed，E4明确not_exercised（无Full AE QK），没有为检查加Full。E4测量100仍含20 runtime/cold（约876.8ms均值）与80非setup learned（21.14ms），不称100纯热态。三个E2＋两个E1＋E3＋E4各一次原预算齐，不再bench；真实闭环cost快照不代最终全量。
- **E2 XYZ缺口已实际补齐**：一次all680 --xyz退出0，`e2_expand_init7_10_20261019T1530Z/summaries/all_680_xyz_20261020/xyz_index.md`列32张新Full-vs四mask固定图，无视频check／decode。管理者实看libero_10 task0 transfer15，root实看同key transfer16，蓝Full／橙mask、3D＋三投影＋Z-time／阶段＋共同时间距离齐，单位m、无padding；大距离含路线与快慢差，不用小动作重建误差概括真实EEF毫米一致。
- GPU3随后启动safe E1 owner3 session51855；GPU0 safe owner2 session8279、GPU2 safe owner1 session93341按冻结640清单继续。剩余工作为E1有效覆盖、完整父对照／历史描述、最终有界事件表与两总览／中文报告，不新增科学样本。
- **固定遥测事件表已一次实际完成**：`policies/realtime-vla-flash/results/idea1_v3/final_fixed_telemetry_events_20261020/`session25342退出0，96唯一路径缓存、missing0，仅既定init6/task0或5。管理者实读88行＝80 available variant＋8 E2 baseline_self/no_variant；事后固定>5cm持续3步的首次偏离观测E1 27/32、E2 22/32、E3 5/8、E4 7/8，分别记录起点／确认点／双方phase与common/full长度，不补零；是固定样本几何差异，不是统计显著／任务失败／致因率。108对象轨迹grasp证据行全部loss=undetermined，86有正观测；正证据消失与False仅未知可用性，不能确认为丢失。索引重复视图去重保留sources、冲突路径拒绝；无GPU／视频／重图，不重复最终运行。
- safe E1owner1 session93341已退出0，160个episode_complete key无重复；owner0/1两分片完成，owner2/3各由唯一owner继续。GPU2空闲但无未占owner的待派任务，停止不必要GPU1轮询，不为占卡扩条件／重分片。待两session结束后显式四expand根一次analysis，目标1976＝calibration96＋E1800＋E2680＋E3200＋E4200，source visited去重、stage分母独立；不--xyz／不--check-media，随后overview一次，避免各分支重复重读。
