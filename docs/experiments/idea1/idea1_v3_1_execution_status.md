# idea1 v3.1：执行进度

2026-10-21。**91条件首筛、15条件确认和全部既定性能计时已完成；最终科学／视频审计通过，汇总报告与全量文件索引已生成。** idea2已先行全部完成并通过112条最终审计，两轮数据分开保存。

## 最新状态（05:42，最终完成）

- 首筛：91条件×40＝3640条全部通过canonical检查，全局promotion已封存；既定质量总预算6040条。
- 确认：15个条件全部完整200条，合计6040个唯一质量episode，完整键／身份与冻结选型检查通过。四个Full成功数分别为π0 176、π0.5 195、XR0 195、CogACT 188（各200）；无待运行质量任务。
- 性能：两π模型各4档vision代表、XR0主要跳18及高质量跳15参考、CogACT历史跳9及新跳9探索点均已完成同卡20预热＋100真实输入计时，5个cohort均通过来源与漂移检查。全部NN进程／GPU等待器已结束，不继续扫描卡、不追加实验。
- 最终科学审计 `final_scientific_audit_20261021_0527.json` 已通过：6040质量episode，errors/missing均0，严格视频解码开启；idea2的112条最终复核与四模型E2共128条同服务复核均通过。102项CPU单测通过。
- 已生成[最终汇总报告](idea1_v3_1_final_report.md)、[全部已存视频HTML](../../../results/idea1_v3_1/final_episode_inventory_20261021_0545/videos.html)与[全部episode原XYZ索引CSV](../../../results/idea1_v3_1/final_episode_inventory_20261021_0545/episodes.csv)。索引将6040质量、128采用profile、32旧未采用profile分别标记；404个视频中质量364、采用32、归档8，归档不进新质量／采用分母。
- 最后[逐要求完成审计](../../../results/idea1_v3_1/completion_audit_20261021_0542.json)通过：idea2与v3.1分开、冻结promotion完整重算一致、5个正式性能cohort复核一致、全部原始quality result SHA再核对、报告43个相对链接存在、原始目录无technical invalid／incomplete、无残留两轮实验进程。必需未完成项0；条件扩展未触发，可选相机候选未采用，没有启动下一轮。
- 图表：[评测吞吐](../../../results/idea1_v3_1/diagnostics/performance_figures_20261021_0434/e2_throughput.png)、[vision与整体RPC速度](../../../results/idea1_v3_1/diagnostics/performance_figures_20261021_0434/vision_native_p50.png)、[最终60组固定XYZ及双侧视频索引](../../../results/idea1_v3_1/final_artifacts_20261021_0524/media.md)。最终图表目录含229项产物，60组真实末端XYZ对比，缺失证据0、379个相对链接已核对；阶段图另已生成42张。

最新主数据索引：`results/idea1_v3_1/dataset_bundle_v3_all_models/index.json`。旧v2及早期v3索引保留为切换前快照，不能用它们单独统计后续新增模型根目录的数据。以下按时间保留历史进度。

## 冻结身份与修订

- 正式使用 `results/idea1_v3_1/manifest_v2.json`，身份 `70f0d281af056a9716a1861e2d4b8f4b2c843ad8f6902089a10adc65564df4ec`。
- 原始 `manifest.json` 身份 `759fdea316c8125254ec3f1bab94ba1b9dd7a7675bd5f973884ea53526facd16` 保留；修改前全部源码按旧哈希核对，归档 `source_archive_v1.tar.gz`，SHA256 `6533d0a2d8c59b2f09ccf35078442dbafa9bac87e817033f5015a724ccc5ec2c`。
- v2仅纠正尚未运行的质量评测默认环境复用问题：默认关闭；显式开启必须通过同模型8场景E2初态／输入／噪声／动作／结果一致性及实际吞吐收益门槛。E2四个变体设置不变。未改mask、任务、预算、H/R/D、checkpoint或用户idea文档。
- gpt-6.1-sol medium完成代码；主代理重新验证冻结源码与checkpoint身份，15项CPU单测全部通过。新增只读最终审计，缺数据时不会判完成。
- 真实输入bank仍为100个唯一观测、四套件各25，SHA256 `2c2e90a3e1fcac3f02226a433c6db9e19a0f3c169deea77707a9a1fbe8996d55`。

## 当前三路（00:03左右启动）

以下表格是首批启动记录，均已正常完成；当前任务见本节末更新。

| GPU | 任务 | 控制会话／PID | 模型服务PID／端口 |
| --- | --- | --- | --- |
| 0 | π0.5 Full，E2 full_io，8场景 | 62083／2343482 | 2343561／9130 |
| 2 | π0 Full，原生request＋vision 20warmup/100真实输入 | 82340／2343832 | 2343904／9132 |
| 3 | 小米Full-D2，原生适配验收＋20/100同卡测速 | 80499／2344096 | 2344370／9133 |

原始数据：`results/idea1_v3_1/run/<condition>/`。服务日志位于各condition目录，源码身份受manifest检查。GPU1仍有他人PID1793831，占约11.9GB，未使用。已有自有任务时按进程／会话结束调度，不频繁扫描这些卡。

### 首批实际进展

- 小米Full-D2原生适配parity：同显式噪声误差0，20warmup＋100唯一真实输入已完成，GPU3单请求P50=150.42ms、vision P50=20.84ms；这不是闭环成功率。
- π0原生Full同样完成20/100，GPU2单请求P50=101.12ms、三路vision合计P50=12.40ms。不能将不同模型／不同卡的绝对毫秒直接作为模型优劣排名。
- GPU2随后启动all27 selected-scan Full参考，会话36704、控制PID2347616／服务2347684、端口9135；原始Full benchmark不重跑。须与native Full同输入／噪声比对。
- GPU3小米已正常exit0，接着测CogACT Full：会话66017、控制PID2347116／服务2347184、端口9134，20/100。首次复用9133在模型分配前被bind preflight拒绝；`ss`确认无listener、前服务已退出，仅TIME_WAIT，改用独立端口9134，没有停止他人服务。这属于启动技术事件，不是策略失败。
- 启动时物理GPU身份：0=`GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`，2=`GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`，3=`GPU-555cde36-5e41-bc44-0c69-ea47358fd703`。已对当时运行的三个服务 `/proc/<pid>/environ` 仅读取CUDA／EGL选择项，分别确认0／2／3。其进程内逻辑cuda:0并不表示都在物理GPU0。

### 后续启动与门槛发现

- GPU0：E2 full_io与lean_io各8条均完成、成功8/8；随后会话47268顺序执行reuse_env（9146）与two_workers（9147）。不同profile原生进程中首请求输入／噪声相同，动作仍有约0.001–0.0027差异，不能把所有差异归为I/O或环境复用。严格采用门槛尚未通过，不提前开启质量评测的复用／双worker优化。
- GPU2：π0原生／selected-scan两份20/100已完成。跨进程动作门槛失败的原始 `pi0_vision_parity.json` 保留；已用同一加载模型定位到native重trace、v2 selected-scan及无gather辅助路径逐层／动作均误差0，支持跨进程数值变化而非wrapper损坏。正在补两个模型的多suite同进程E0，方法澄清见方案3.1。**未启动vision mask闭环。**
- GPU3：CogACT原生适配parity0，Full20/100完成，P50 request178.21ms／vision12.31ms。随后会话42577启动E1真实闭环队列：小米27条件＋CogACT22条件，依次各40场景，共1960个首筛键。队列 `run/batch_receipts/e1_screen_gpu3.json`，首控制PID2358017、首端口9400，后续逐job独立端口至9448。每个condition保留独立模型进程，不改科学参数；默认单worker且不复用环境。
- 新增queue只做已有controller的顺序编排与精确已完成键跳过，6项CPU测试通过，源码SHA写入batch receipt。它不修改v2生产源码或manifest，避免已退出端口TIME_WAIT导致队列在第一项后误停。遇非零exit停止并保留数据，不能把排队当完成。

### 00:20前后的验证与排队

- 两个同进程E0均完成，5条真实输入索引 `[0,1,25,50,75]` 覆盖四suite；每模型各3模式、27层×4节点×3相机，以及完整raw／decoded动作均逐位0误差。π0 gate SHA `4dd5206df421fd6bcc19d3c601db9ce47fb34a75b8d476697cd5df2c66480452`；π0.5 gate SHA `f47789be06df36fe375d627537809f3a4ee0e3ad30aa65e03ebfe33d7ffafaef`。主代理独立CPU核对source／bank／checkpoint／coverage／原N10递推，4项新增negative gate测试通过。
- π0.5另完成原生／matched Full两份20/100（会话71183正常exit0）；其跨进程比较同样约0.0020／0.00244差异，原failed `pi05_vision_parity.json` 保留，与同进程肯定证据分列。
- GPU2正式进入π0 E3 screen：会话55599，队列PID2368448、首控制2368449、首服务2368517，端口9600–9620，21条件×40＝840键。显式使用 `pi0_vision_same_process_parity.json`；生产v2源码与manifest不变。
- GPU3小米Full40/40有效、39/40成功（97.5%），仍只是首筛；队列已自然进入historical skip12，不能把Full首40替代后续200条确认。
- E2原始四组各8/8成功。118.94／93.88／88.67／57.31秒分别对应full_io／lean／reuse／two_workers，描述性吞吐比1／1.267／1.341／2.075。因跨进程数值及自由轨迹差异，原严格采用检查均不通过，质量评测仍默认单worker／不复用环境。GPU0保留用于一次同服务的受控吞吐补充验证，不把2.075×先写成已部署收益。
- 新增只读审计另发现两个旧verifier元数据假设错误：benchmark保存的是parity之前的health snapshot，以及Torch shard digest序列化与旧common.digest不同。正在加肯定证据适配；不改NN／原始结果，不把verifier错误记作策略失败，也不在缺数据时判完成。

### 00:42更新：π0.5正式采用双worker，三卡质量评测均已启动

- π0.5同一持续服务的四组×8条均有效、各8/8成功。各优化组254/254请求与基线输入／噪声／decoded动作完全一致，8个初始图像与数值状态完全相同，40个固定控制误差0。严格视频与数据审计 `e2_shared_service_validation/audit.json` 通过（SHA `87aac904d16694a72d74400adc7150af5348c636edc6956742376e66535be85a`）。
- 同服务墙钟113.870／92.083／87.140／55.453秒，lean／reuse／two-workers分别1.237／1.307／2.053×。这只是离线评测吞吐，不是单请求延迟或机器人加速比。
- 已完成有来源标记的派生计时登记与原benchmark精确副本；主代理独立核对哈希，并确认**原冻结** `quality_reuse` 门槛返回true，无修改NN或绕过门槛。
- GPU0：π0.5 Full正式40条首筛，会话88210／控制PID2395365／服务2395434、端口9800；显式2worker＋环境复用，使用原始根目录的π0.5同进程vision gate。先运行Full一个完整condition，随后安排有限的其他模型E2小批验证，再继续20个vision mask。
- GPU2：π0原队列PID2368448仍在，单worker、不复用；GPU3：小米／CogACT队列PID2358016仍在，同样原配置。未向其他模型外推π0.5的收益，也未中断在途episode。
- π0.5质量数据写入 `results/idea1_v3_1/e2_shared_service_validation/<condition>/episodes/`；其余模型仍在 `results/idea1_v3_1/run/`。不可覆盖的组合索引是 `results/idea1_v3_1/dataset_bundle_v2/index.json`，实际数据不移动／不复制，分析显式读取真实目录而不依赖递归跟随软链。
- 组合审计当前errors=0但missing>0、passed=false，正确表示尚未完成；37项CPU测试通过不能代替真实矩阵交付。其他三模型的32条/模型E2追加边界已在方案登记，尚未运行。

## 接下来（按00:42更新）

1. GPU0完成π0.5 Full首40；其他模型E2各32条仅做一次，逐模型决定是否采用，再继续π0.5 vision矩阵。原队列在此期间继续。
2. 两π模型原生／matched Full测速及同进程肯定parity均已完成；旧跨进程差异单列，原始失败文件保留。
3. E3每模型Full＋4预算×5mask、E1两模型各预算×5mask，均直接40条标准四套件筛选；随后冻结promotion，按规则补齐选中候选与Full至200。
4. 每vision预算至少一个代表测速，原生／匹配Full比率分列；E1前沿若仍满足相对70%门槛，按方案另立不可覆盖扩展清单。
5. XYZ、失败最远进展、视频与完整数据审计后再写最终结果。代码／准备通过不代表全实验完成。

### 00:57更新：π0也通过优化验证，已切换后续队列

- π0.5 Full首40已正常exit0，40/40成功；不是最终200条率。其正式质量数据与E2的32条profile分开计数。
- π0追加E2四组各8/8成功，246/246配对请求输入／噪声／decoded动作完全一致，初始图像／数值状态一致，40次固定控制误差0，严格独立数据／视频审计通过。full／lean／reuse／two-worker墙钟110.627／89.676／82.706／52.782秒，吞吐1／1.234／1.338／2.096×。原冻结reuse gate返回true，因此显式采用2worker＋复用。
- GPU2新π0队列会话4996／PID2424053，首控制2424055／服务2424130，端口9904起；新receipt是 `e2_shared_pi0/batch_receipts/pi0_screen_gpu2_two_workers.json`。已完成Full与3个Keep24条件共160条均精确跳过，通过4个可核对的目录／episodes别名保留原始数据，未复制或重跑有效episode。
- 边界切换技术记录：旧queue PID2368448在首4条件完成后收到仅发给编排父进程的SIGINT，exit130；信号与下一condition启动竞态，下一控制进程在产生任何quality episode、parity或loaded-protocol文件前已退出，空启动日志保留。该尚未测量condition在新根目录／端口重新启动，不计任务失败。以后切换先暂停编排父进程、等待当前controller完整终止，再退出父进程，避免再次与自动启动下一项竞态。
- 采用／别名／保留科学键的追加凭据：`diagnostics/parent_pi0_adoption_0057.json`。v3索引里的 `resume_aliases_activated=false`描述planner自身不激活别名；实际父代理激活见该凭据，不改写不可覆盖的索引快照。
- π0的2.096×是在GPU0同服务测得；质量队列回到原GPU2，不把该比值宣称为GPU2已经测得的实际部署加速。NN、D/H/R、mask和episode键均未改动。
- 速度比解释：π0的lean／two=1.699×、reuse／two=1.567×；π0.5对应1.661×、1.571×。2.096／2.053×均以全录像＋保存观测的full_io为分母，包含I/O、复用与并发合并收益，不是双worker单项收益。四组固定顺序、每模型仅一次8场景测量，计时含环境创建，NN预热不等于所有环境／文件缓存均同样预热；因此这是快速采用证据，不是排除了所有缓存／顺序影响的稳定性能估计。正式首筛每task仅init0，环境复用本身通常不会触发，主要收益来自并发。
- GPU0现运行小米E2（会话42986／PID2418659、服务2418725、端口9811）；随后仍有CogACT一次32条验证待运行。GPU3原小米／CogACT质量队列继续；未先替小米或CogACT开启尚未验证的配置。
- 已产生一次只供核对的真实数据可视化快照 `diagnostics/report_progress_0054/media.md`：8张真实XYZ对比、两侧视频与遥测链接，无缺失链接；主代理查看了实际图像。该快照不是最终报告。距离包含路线与执行进度快慢差，不能单靠同step／同仿真时刻两种距离拆开这两类误差。
- CPU验证：原协议／vision／队列／审计／E2等51项通过，新的3项v3索引与8项报告绘图测试通过。严格数据审计仍因剩余矩阵未完成而为false，不以测试通过宣称实验完成。

### 后续执行注意

1. v3索引按condition显式选择唯一episode源，E2来源与benchmark来源另列；分析用 `dataset_bundle_v3`，最终审计同时检查profile凭据与原始vision gates。旧数据与失败观察继续保留。
2. 对已在原根目录完成Full首40、后续使用 `episodes`别名的模型，Full确认优先在原根目录以2worker、`environment-reuse off`运行（须先有同模型并行采用证据），避免冻结controller的递归计数不跟随内部episodes别名而产生错误吞吐计数。其他mask在通过门槛的新根目录继续2worker＋复用。科学样本／完整NN不变；Full额外环境创建成本披露，不写成同一吞吐配置。
3. 未完成91条件首筛前不冻结全矩阵promotion；之后按规则补齐Full与入选候选至200，并处理实际触发的前沿扩展。最终仍需每vision预算代表测速、四套件表、阶段进展、XYZ／视频及逐项完成审计。

### 01:05更新：小米已安全切换；CogACT质量矩阵仍须单独启动

- 小米四组同服务E2各8/8成功，每个优化组119个请求与基线输入／噪声／动作完全一致，初始图像／状态一致，固定控制最大误差0，无技术invalid；独立数据／严格视频审计通过，原reuse gate返回true。
- full／lean／reuse／two墙钟92.336／80.594／72.114／49.373秒；相对full_io的合并吞吐1.870×，相对lean约1.632×、相对reuse约1.461×。仍是GPU0固定小批、固定顺序的证据，不冒称GPU3质量队列的实测同条件加速。
- 旧编排父PID2358016先SIGSTOP，当前controller2423538继续完成40条并自然退出；实际观察其`/proc`状态Z、exit_code0，日志`complete:true`，之后仅向编排父发SIGINT＋SIGCONT，退出130。没有打断在途episode，也没有启动下一旧controller。
- 原小米Full、历史参照及4个新mask共240条完整有效数据，用6个可核对别名原位保留；凭据 `diagnostics/parent_xiaomi_adoption_0105.json`。新小米队列会话52879／PID2451918，首controller2451926、端口10006起，显式2worker＋复用，自动跳过已有6条件。
- 新主索引 `dataset_bundle_v3_xiaomi/index.json` 显式合并π0、π0.5、小米的新根目录与原始数据；旧索引不删除。原v3审计接口的bool位置参数错误已由gpt-6.1-sol medium修正并加回归测试；4项v3测试通过，实际CPU严格部分审计errors=0、missing=169、passed=false。创建源码hash与当前分析／审计源码hash分列，原索引未改写。
- **原49项合并队列已退役，新队列只负责小米27条件；CogACT全部22个质量条件尚未启动，不能遗漏。** GPU0正在跑CogACT独立E2：会话15689／PID2445607、服务2445678、端口9812；该32条profile不能计入未来880条首筛。
- 当前GPU2是π0优化后质量队列，GPU3是小米优化后质量队列；GPU0完成CogACT小批后恢复π0.5剩余20个vision mask。CogACT质量矩阵需另派空闲卡，若其他卡都由本实验占用，仅检查候选GPU1是否释放，不频繁扫描自有忙卡。

### 01:10更新：四模型E2全部通过，三路质量矩阵＋一路待卡

- CogACT同服务四组各8/8成功，三个优化组各98次请求与基线输入／噪声／动作一致，初始图像／数值状态一致，固定控制最大差0。严格独立审计、派生计时登记、原冻结reuse gate均通过。
- CogACT full／lean／reuse／two墙钟97.354／87.372／78.152／55.151秒；相对full_io合并吞吐1.765×、相对lean1.584×、相对reuse1.417×。不是flow matching积分步长实验，也不是同模型单请求加速比。
- 四模型同服务验证共128条（每模型32），与原π0.5跨进程32条观察、正式质量episode分开。四模型均保持原D/H/R，原始失败观察与所有科学结果不覆盖。
- 最新主索引是 `dataset_bundle_v3_all_models/index.json`，93个condition定位项＝91个质量条件＋2个仅测速参考。分析显式读取各模型实际数据，质量分母不包含benchmark-only参考或E2 profile。

| 卡／状态 | 当前用途 | 会话／编排PID | 数据根目录／receipt |
| --- | --- | --- | --- |
| GPU0，运行 | π0.5 Full已完成40，继续20个vision mask，2worker＋复用 | 78826／2455743；首controller2455744，端口10101起 | `e2_shared_service_validation/batch_receipts/pi05_screen_gpu0_two_workers.json` |
| GPU2，运行 | π0余下vision矩阵，2worker＋复用 | 4996／2424053 | `e2_shared_pi0/batch_receipts/pi0_screen_gpu2_two_workers.json` |
| GPU3，运行 | 小米剩余VLM矩阵，2worker＋复用 | 52879／2451918 | `e2_shared_xiaomi/batch_receipts/xiaomi_screen_gpu3_two_workers.json` |
| GPU1，等待他人释放 | CogACT22条件质量首筛，尚未启动 | 等待器40873／PID2461723 | 将写 `e2_shared_cogact/batch_receipts/cogact_screen_gpu1_two_workers.json` |

01:08实际只检查候选GPU1：仍有他人PID1793831，占11876MiB，即使利用率0也不算空闲。等待器只在**尚未启动自有CogACT队列时**每30秒检查GPU1的compute PID，释放后exec原冻结queue并停止扫描；controller仍执行GPU空闲二次检查，竞态不会抢占他人任务。它不扫描自有GPU0／2／3。如果这些自有队列先完成，先核对等待器实际仍为等待shell、再结束该自有等待器并把CogACT派到新空闲卡；若等待器已exec成正在跑的CogACT队列，则不得因旧等待状态而终止它。

本轮还未完成：三模型剩余首筛、CogACT全部质量首筛、不可覆盖promotion、入选200条确认、实际触发的前沿扩展、每vision预算代表测速和最终报告／严格完成审计均不能省略。当前优化验证完成，不等于整轮v3.1实验完成。

### 01:14初步数据分析：布局差异显著，但尚非最终排名

以下只取已完整完成首40的预算组，固定4套件×10任务×init0；均未扩到200。完整矩阵仍在运行，不据此提前冻结promotion，也不据此省略其余mask。

| 模型／裁剪模块 | Full | uniform | front_middle | back_middle | short_runs | seeded_random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| π0，vision跳3／保留24 | 34/40 | 36/40 | 35/40 | 34/40 | 1/40 | 38/40 |
| 小米，VLM跳12／保留24 | 39/40 | 32/40 | 0/40 | 1/40 | 38/40 | 31/40 |

小米历史skip12参照另为40/40。π0的short_runs恰为跳过vision原始L1–3，而其他同预算布局大多保持较高首筛成功数；这里只能说**这个早段三层组合**的消融非常有害，不能定位其中哪一层单独重要。小米反而是短连续布局38/40、两个集中中段布局近乎全失败，说明“连续跳层一定不好”不是通用规则。模块、具体位置和组合都要分开讨论。

这两个预算组支持继续直接测不同布局，而不是仅按裁剪层数或离线分数推断成功率。π0部分mask比Full首40更高也不是已经证明剪枝提高了总体能力，仍需按预定规则补量确认。失败最远进展只表示已经到达哪里，不归因为某个精细操作。

不可覆盖快照：`diagnostics/screen_initial_complete_budgets_0113.json`（13个完整条件、520条）；补充诊断图选择上述两个预算各一个高／低表现mask，mask选择为事后说明性选择，episode仍固定各suite task0/init0，不按轨迹结果挑场景。新图目录 `diagnostics/report_progress_0113/`，不是最终报告。此前全模型严格部分审计已正常结束：errors=0、missing=161、passed=false。

01:20图表核对：诊断XYZ／视频／遥测索引已完成，71个产物的链接无缺失；新增16张阶段计数＋失败条件占比图，主代理已检查真实输出。第一版长标签重叠，另存v2仅压缩显示标签并加完整ID映射；原始CSV、计数、占比和旧版图均不覆盖。v1核心5项、v2样式3项CPU测试通过，不据此宣称主矩阵完成。

- [固定场景XYZ、双侧视频与遥测索引](../../../results/idea1_v3_1/diagnostics/report_progress_0113/media.md)
- [π0跳3个vision block：失败最远进展对比](../../../results/idea1_v3_1/diagnostics/phase_progress_0119_v2/phases_pi0_screen_init0_budget3_page0.png)
- [小米跳12个VLM block：失败最远进展对比](../../../results/idea1_v3_1/diagnostics/phase_progress_0119_v2/phases_xiaomi-robotic0_screen_init0_budget12_page0.png)

图中F为失败条数，后面的分数是已观察／预期样本数；右图仅在各组失败轨迹内部归一化。不能将右图颜色比例直接当作该组总体失败率，也不能把阶段标签解读为失败原因。

### 01:38更新：闭合预算的补量调度已验收，尚未启动confirm

- 本次显式按真实数据索引收集到1196/3640个有效首筛键：π0为434、π0.5为276、小米为486、CogACT为0。这是采集瞬间的进度，三个质量队列仍在运行；E2 profile不计入分母。最新组合summary已刷新，仍为partial，不能据此宣称全矩阵完成。
- 原编排PID2424053／2455743／2451918均实际存活，分别负责GPU2的π0、GPU0的π0.5和GPU3的小米。GPU1等待器PID2461723实际仍为等待shell而非CogACT Python队列；没有额外扫描自有忙卡，也没有抢占他人进程。
- gpt-6.1-sol medium新增的性能cohort与局部confirmation dispatch由主代理运行14项CPU测试，全部通过；真实bank dry run核对100个唯一输入、7个π系计时phase，未生成结果目录、未查询GPU或启动NN。旧NN／controller／manifest与原始benchmark均未修改。
- 首个不可覆盖局部调度凭据是 `results/idea1_v3_1/dispatch/closed_budgets_20261021_0140.json`（文件名仅为标识）；主代理已在实际数据上生成并重新验证，`global_promotion=false`。它固定了23个完整首筛条件的result／identity／complete源路径及SHA，仅包含已闭合的4个预算和3个Full，不代表全局promotion。

| 已闭合预算 | 冻结规则选中的候选 | 首40成功数 | 对应Full首40 |
| --- | --- | ---: | ---: |
| π0 vision跳3 | `pi0_keep24_5_seeded_random` | 38/40 | 34/40 |
| π0.5 vision跳3 | `pi05_keep24_1_uniform` | 40/40 | 40/40 |
| 小米VLM跳12 | `xiaomi-robotic0_skip12_4_short_runs` | 38/40 | 39/40 |
| 小米VLM跳15 | `xiaomi-robotic0_skip15_5_seeded_random` | 36/40 | 39/40 |

π0.5跳3有多个40/40，按原condition ID平分规则选uniform，未根据后续速度或轨迹改排名。小米跳15其余4个布局为2／0／0／0成功，与seeded_random的36/40反差很大；仍只能说明组合差异，不能把一条布局规律外推成个别层的因果解释。上述候选尚未补至200。

主代理还以真实数据运行了5组原queue CLI的confirm dry run，共7个条件，全部得到合法pending计划：π0 Full原根目录／复用off、π0入选mask新根目录／on；π0.5 Full及入选mask均新根目录／on；小米Full原根目录／off、两个入选mask新根目录／on。均为2worker、显式原vision gate（π系），未分配GPU、未写batch receipt、未启动NN。每条件已有40条，正式运行应仅补160条，共1120条；运行后仍须按实际完成键验收，不能把dry run计入进度。

首张释放的卡优先派CogACT的880条长首筛；其余空档可用已验证局部dispatch安排既定确认。正式性能cohort须等待各预算代表确定，执行Full-before／matched Full／代表／Full-after，同物理卡、20warm＋100真实输入；RPC或vision的Full前后P50漂移超过10%即标不稳定，不自动重测挑最好值。所有这些实际确认／测速仍待运行，CPU测试通过不是交付完成。

### 01:44追加：π0 vision跳6预算已闭合

π0的五个Keep21条件首40均已完成；原规则选中 `pi0_keep21_1_uniform`，33/40（82.5%），相对Full首40的34/40保留97.06%。追加的不可覆盖调度凭据为 `results/idea1_v3_1/dispatch/closed_budgets_20261021_0144.json`，主代理重新验证了旧凭据仍有效、新凭据的原始来源／SHA，以及旧选中集合确为新集合子集。新增该条件的原queue confirm dry run通过，未启动NN。

目前局部dispatch包含5个完整预算、3个Full，共8个待确认条件；均尚未补量，若按当前集合需各新增160条，共1280条。其他预算仍须按原计划完整首筛，尚不能封存全局promotion。实际轮询三个原会话均仍在运行，小米已进入跳18层第二种布局；GPU1等待器仍为shell，无新分配或重启。

### 01:47中途可视化与阶段检查

新快照 `diagnostics/report_progress_0146/` 显式记录采集时1346/3640个首筛键、33个完整条件；这不是当前实时总数。四个说明性mask各取固定四suite task0/init0，生成16组真实EEF XYZ／同step及近邻仿真时间距离CSV，以及双侧视频／遥测索引。77个产物、123个链接均核验无缺失。另生成22张阶段计数／失败条件占比图 `diagnostics/phase_progress_0146_v2/`；主代理已查看真实XYZ与π0.5跳3阶段图，而不只核对文件存在。

π0.5 vision保留24层的五布局首40成功数依次为40／37／40／0／40，Full为40/40。0/40的short_runs恰跳原始vision L1–3；其40条失败的最远heuristic进展为initial23、approach16、grasp_or_manipulate1，即39/40尚未记录到接触／抓取及更后阶段。固定Spatial示例也可看到明显早期位置偏离。因此**不能把所有裁剪失败都概括为只在精细操作时发生**；该结果是vision早段三层组合消融的反例，不是定位了某一单层作用，也不是VLM backbone的同一种干预。阶段仍是进展描述，不证明视觉表征损坏等具体致败机理。

01:56定向验收追加：对这个0/40条件逐条调用原始episode验收及既有v2协议schema修正，40条均通过manifest／checkpoint协议、官方初态与Full配对、预测动作到实际执行动作重建、NFE、轨迹长度／数值与时间检查；4个固定视频完整解码、20FPS及动作数＋1帧检查通过。凭据 [pi05_vision_L1_3_full40_data_audit.json](../../../results/idea1_v3_1/diagnostics/pi05_vision_L1_3_full40_data_audit.json) 固定了每条result／identity／complete／infer／telemetry及视频SHA。没有重跑模型或覆盖原始数据。其`passed=true`严格只指这个完整首40条件的数据一致性，不代表全矩阵完成，也不能排除所有可能的实现问题或证明具体失败机理。

小米跳15层的两个对比mask为uniform与seeded_random（首40为2与36），均为事后说明性选择；场景不按成败挑选。random的固定Goal示例路线较接近Full，但同step距离仍包含进度快慢差，不能把末端距离直接解释为纯横向误差或失败原因。所有候选尚未补到200，图表不表示全矩阵或最终报告已完成。

- [本次真实XYZ、视频与遥测索引](../../../results/idea1_v3_1/diagnostics/report_progress_0146/media.md)
- [π0.5跳vision L1–3的固定Spatial轨迹](../../../results/idea1_v3_1/diagnostics/report_progress_0146/pi05_keep24_4_short_runs_libero_spatial_task00_init00.png)
- [π0.5保留24层五布局的失败最远进展](../../../results/idea1_v3_1/diagnostics/phase_progress_0146_v2/phases_pi05_screen_init0_budget3_page0.png)
- [小米跳15层random固定Goal轨迹](../../../results/idea1_v3_1/diagnostics/report_progress_0146/xiaomi-robotic0_skip15_5_seeded_random_libero_goal_task00_init00.png)

### 02:09追加：π0.5跳6、XR0跳18的预算已闭合

02:02左右刷新主索引summary时收集到1657/3640条有效首筛（45.5%），这是首筛进度，不是包含补量／测速／报告的总进度。其后三路队列实际仍存活，继续推进；CogACT等待器实际仍为shell，尚未开始质量首筛。

新的局部调度凭据 `results/idea1_v3_1/dispatch/closed_budgets_20261021_0209.json` 在真实数据上生成并重新验证，保留之前全部选中条件及原始screen文件SHA；`global_promotion=false`。新增两个完整预算的选择如下：

| 模型／模块 | 选中的候选 | 首40成功数 | 对应Full首40 | 相对保留率 | 当前意义 |
| --- | --- | ---: | ---: | ---: | --- |
| π0.5 vision跳6／保留21 | `pi05_keep21_1_uniform` | 39/40（97.5%） | 40/40 | 97.5% | 按原规则进入200条补量，尚非确认结果 |
| XR0 VLM跳18／保留18 | `xiaomi-robotic0_skip18_5_seeded_random` | 26/40（65%） | 39/40 | 66.67% | 未达到70%目标；因达到原60%近阈值补量门槛，进入补量以判断，不称已可用 |

当前局部dispatch累计7个闭合预算、3个Full，共10个既定确认条件；尚未启动confirm，按目前集合需各新增160条，共1600条。其余预算与CogACT仍须完整首筛，全局promotion不能提前封存。没有新增NN评测范围、修改采样参数或为得到较好排名重跑已有结果。

### 02:17追加：π0 vision跳9／保留18的预算已闭合

五布局首40均完成后，原规则选中 `pi0_keep18_1_uniform`：30/40（75%），对应Full为34/40（85%），相对保留88.24%。不可覆盖凭据为 `results/idea1_v3_1/dispatch/closed_budgets_20261021_021706.json`，已重新验证所有已承诺screen文件及旧选中集合，当前累计8个闭合预算、3个Full，共11个待确认条件。尚未开始补量，不把这30/40视为最终估计。

原三路会话均仍运行：π0已进入最后的Keep15档；π0.5处于Keep18第三布局；小米处于跳21层档。CogACT等待会话仍存活、尚未转为质量队列。未重启在途任务、未重复已有40条、未改NN或采样参数；实际卡释放前继续会话级等待。

### 02:26进度快照：首筛超过一半，CogACT仍待卡

主索引summary已重新从唯一实际数据源生成：2135/3640个有效首筛键（58.65%），51个完整条件、另3个部分条件；π0为717、π0.5为594、小米为824、CogACT为0。质量分母没有混入128条同服务E2、原π0.5跨进程profile、benchmark-only参考或微基准请求。仍无confirm数据，不能将首筛比例当作整轮目标完成比例。

编排PID2424053／2455743／2451918仍实际存活，分别在π0 Keep15第二布局、π0.5 Keep18第四布局、小米跳21第四布局。GPU1等待器PID2461723仍为等待shell；没有新扫描自有忙卡、抢占他人任务或重启在途episode。首张释放卡仍优先安排CogACT的880条长首筛；若之后有多张卡可用，可在完整condition边界将剩余CogACT条件拆为互不重叠的原queue子集，避免所有后续工作串行等待单卡，不改变科学键或NN源码。

### 02:28工具回归检查

主代理运行整个v3.1 CPU测试目录，85项全部通过，耗时28.403秒；命令为 `env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu nice -n 10 policies/openpi/.venv/bin/python -m unittest discover -s preparations/idea1_v3_1 -t . -p 'test_*.py'`。覆盖协议、CPU vision、原始／v2验收、同进程gate、四模型E2辅助、登记及索引、queue、局部dispatch、性能cohort、报告与两版阶段图。测试中的PID12345等为mock输出，不是真实模型启动；没有占用显卡。JAX／Flax仅有弃用提示，未为此修改冻结依赖或NN源码。

这批测试包含小型随机初始化CPU单测，其作用仅是代码回归；它不是正式实验输入，也不是训练checkpoint上的质量／余弦结论依据。真实科学数据仍来自已加载checkpoint在官方LIBERO初态上的闭环；真实同进程E0另有五条真实观测证据。85项通过不代表尚未完成的91条件首筛、200条确认、性能cohort或最终审计已交付。

同时实际会话输出显示π0已进入Keep15第三布局、π0.5进入Keep18第五布局；其余任务继续，无在途重启。

### 02:38追加：π0.5跳9、XR0跳21预算闭合

局部凭据 `results/idea1_v3_1/dispatch/closed_budgets_20261021_023752.json` 已在实际数据上生成及复核，旧选中集合与screen源文件哈希保持有效。新增闭合预算如下：

| 模型／模块 | 首筛选型 | 成功数 | Full首40 | 决定 |
| --- | --- | ---: | ---: | --- |
| π0.5 vision跳9／保留18 | `pi05_keep18_5_seeded_random` | 35/40（87.5%） | 40/40 | 进入200条补量，尚非最终质量 |
| XR0 VLM跳21 | 五种布局均0/40；按ID平分规则代表为uniform | 0/40（各布局） | 39/40 | 未达60%近阈值补量门槛，不补该档200条 |

五种XR0跳21布局均失败，只能说明**本次预定五种布局**均不可用，不证明全部跳21组合不可能，也不能定位是哪一个具体层造成的。初始跳24档仍按计划评测，不因跳21全失败而取消。当前局部dispatch累计10个闭合预算、3个Full，共12个待确认条件，`global_promotion=false`；确认仍未启动。

三路原队列及GPU1等待器均实际存活；π0处于Keep15第四组，π0.5处于Keep15档，小米处于跳24档。首张释放的卡仍优先转给CogACT，不为追加中途图表占用显卡或重跑旧质量数据。

### 02:54重要交接：π0首筛完成，CogACT已在GPU2真实开跑

- π0原会话4996正常exit0，原编排PID2424053及末controller2586385均已退出。主代理独立检查原batch receipt为completed、所有新job exit0／已有job合法跳过，并逐条件调用原canonical validator：21条件×40＝840条完整有效首筛。其200条确认与正式代表测速仍未完成。
- π0 Keep15档的最佳为 `pi0_keep15_1_uniform`，5/40（12.5%），Full为34/40；质量很差，不称可用或等质量加速。由于原冻结vision补量门槛为至少4/40，它仍进入既定确认；每档至少一代表的性能覆盖也仍执行。追加凭据为 `results/idea1_v3_1/dispatch/closed_budgets_20261021_025222.json`，旧选型及screen哈希复核通过，当前13个选中条件，仍非全局promotion。
- GPU2实际compute列表已空后，使用pidfd先暂停并重新确认旧等待器PID2461723仍为自有pending bash，再只终止该等待shell（会话40873 exit143），没有停止任何CogACT NN队列。凭据 `diagnostics/cogact_gpu2_handoff_waiter_retirement.json`。openpi虚拟环境缺少pidfd_open的首次编排检查在发出任何信号前失败；改用系统Python的标准pidfd接口完成，未修改代码或模型，不计科学invalid。
- CogACT队列已在GPU2启动：会话98119／编排PID2595138，首controller2595139／模型服务2595207，端口10200起，共22条件×40＝880条。新receipt为 `e2_shared_cogact/batch_receipts/cogact_screen_gpu2_two_workers.json`，原GPU1 receipt未生成。采用已验证的2worker＋复用，D10原生DDIM、H16、各suite原R不变。实际RPC health核对同PID及Full condition、真实输入parity=true，并观察到真实quality complete标记，不能只把排队当启动。

02:53刷新主summary的瞬时计数为2635/3640（72.39%）：π0 840、π0.5 760、小米1000、CogACT35；65个完整条件、1个部分条件。该计数是首筛快照，不含尚未进行的confirm／正式cohort，也不代表整轮完成。CogACT的35条不是旧E2 profile，而是新Full质量episode。

| 卡／状态 | 当前工作 | 会话／编排PID |
| --- | --- | --- |
| GPU0，已有任务 | π0.5末档vision首筛 | 78826／2455743 |
| GPU2，已有任务 | CogACT质量首筛22条件 | 98119／2595138 |
| GPU3，已有任务 | 小米末档VLM首筛 | 52879／2451918 |
| GPU1，待可用 | π0四预算正式同卡performance cohort | 新等待器72710／PID2598025 |

新GPU1等待器仅在有待办且尚未exec自有工作时每30秒检查GPU1；空闲后exec既有 `performance_cohort run`，不再扫描自有卡，也不重复启动CogACT。拟写独立目录 `performance_cohorts/pi0_gpu1_20261021_after_screen`，端口12000–12006；四个真实首筛代表为Keep24 random、Keep21 uniform、Keep18 uniform、Keep15 uniform，已完成100唯一真实输入／7phase／四预算覆盖的CPU dry run，未创建cohort目录。原生Full-before／matched Full／四代表／Full-after各20warm＋100，原始gate显式绑定。该测速尚未实际启动；若其他自有卡先释放，应先核对并安全撤销仍为shell的这个等待器，再调到空闲卡，不能重复测或误杀已经exec的cohort。

接下来继续完成剩余首筛，并优先用后续空闲卡按互斥条件拆分CogACT长队列、安排确认与短测速；只在完整condition边界切换编排，保留所有有效数据。全局promotion、200条确认、条件触发前沿、实际同卡性能、最终可视化／报告／严格审计仍须完成。

### 03:12调度中：三模型首筛均完整，CogACT拆双卡、π0正式测速

- 小米会话52879正常exit0，原编排2451918和末controller2604072均退出；27条件逐一通过canonical检查，共1080条。跳24档五布局均0/40，不满足近阈值补量／扩展条件，初始cap没有触发更高预算扩展；这不证明未测组合不可能。仍需Full及已入选跳12／15／18条件的200条确认。
- π0.5会话78826也已正常结束，原receipt为completed，21条件×40＝840条逐一验证。其最后Keep15预算及最终代表仍须按原规则封存，确认与测速尚未完成。至此π0／π0.5／小米原始质量首筛合计2760条均完整；CogACT另计，不能因此封存全局promotion。
- GPU3已空后，安全撤销仍为pending bash的π0性能等待器PID2598025（会话72710 exit143），凭据 `diagnostics/pi0_performance_gpu3_handoff_waiter_retirement.json`。没有终止已开始的cohort。实际正式测速现运行于GPU3：会话90065／编排PID2610819，根目录 `performance_cohorts/pi0_gpu3_20261021_after_screen`、端口12000–12006，原生／matched／四预算代表／原生after共7phase；真实gate显式绑定，各20warm＋100，不复用旧跨卡分母。状态以本批cohort.json为准，完成前不报告加速比。
- CogACT Full首40为37/40（92.5%），Spatial／Object／Goal／LIBERO-10分别9／10／9／9成功（各10）；历史参照及两新布局随后完成。其Full仍需补到200。

**当前有意暂停的只是旧CogACT编排父PID2595138，不是goal，也不是NN。** 系统pidfd确认其命令与身份后SIGSTOP，原当前controller2616419（`db-cogact_skip9_3_back_middle`）继续自然完成。不可覆盖分片计划 `diagnostics/cogact_multigpu_split_plan.json` 固定了4个已完成条件、这个在途条件、剩余9＋8个互斥条件，三者并集等于原22条件，未改科学键或NN。

GPU0新队列已启动：会话9523，receipt `e2_shared_cogact/batch_receipts/cogact_screen_gpu0_shard.json`，首controller2618382、端口10400起，9个显式条件。GPU2的剩余8条件暂未启动：**须先观察controller2616419实际终止／Z状态及退出信息，核对该条件完整40，再只退出已暂停的旧编排父进程，之后在GPU2用计划里的8个ID启动独立queue。不能因父PID处于T而误判NN停止，也不能恢复旧父进程让它再次遍历已分给GPU0的条件。** 所有有效结果保留原位置。

GPU1现有新的待办等待器：会话67548／PID2616363，只在尚未exec自有任务时每30秒检查GPU1；空闲后运行原queue的π0 Full confirm，原根目录`run`、2worker、复用off、显式原vision gate、selection为`closed_budgets_20261021_025222.json`、端口13000，拟写 `run/batch_receipts/pi0_full_confirm_gpu1.json`，已有40条会跳过，仅新增160。该确认截至登记时尚未启动；以后若改派其他卡，同样先核对等待器实际状态防重复，不得误杀已exec的确认队列。

后续仍需CogACT全部首筛、全局promotion与已局部选型的兼容检查、所有入选200条、实际触发前沿、同卡性能完整审计／登记、最终XYZ／视频／报告与逐项完成验收。以上队列切换不代表全轮完成。

### 03:22交接已完成：无暂停父进程遗留，CogACT双卡／π0.5测速并行

03:18已实际观察原controller2616419为Z、Linux退出码0，并重新核对 `db-cogact_skip9_3_back_middle` 完整40条；之后只向旧编排父2595138发送SIGINT＋SIGCONT，旧会话98119 exit130，保留原receipt中的有意编排中断及所有有效数据。没有给NN进程发信号。凭据 `diagnostics/cogact_gpu2_parent_retired_after_child.json` 同时保存了退休前batch内容及SHA。**上节有意暂停父进程的过渡状态现已结束，不应再恢复旧父进程。**

| 卡 | 当前工作 | 会话／编排PID | 数据／receipt |
| --- | --- | --- | --- |
| GPU0 | CogACT显式9条件分片，2worker＋复用 | 9523／2618376 | `e2_shared_cogact/batch_receipts/cogact_screen_gpu0_shard.json`，端口10400起 |
| GPU2 | CogACT显式8条件分片，2worker＋复用 | 63871／2626956 | `e2_shared_cogact/batch_receipts/cogact_screen_gpu2_shard.json`，端口10600起 |
| GPU3 | π0.5正式同卡7phase性能cohort | 76071／2627034 | `performance_cohorts/pi05_gpu3_20261021_after_screen`，端口12100–12106 |
| GPU1，仍待可用 | π0 Full既定200条确认 | 67548／PID2616363 | 与上节相同等待器；尚未启动时只扫描GPU1，每30秒一次 |

上述CogACT两个分片互不重叠，与已完成／过渡条件的并集正好覆盖原22条件。GPU2新分片首controller2626957真实启动；GPU0已推进到跳12 uniform。π0.5性能已推进到第三phase，尚未最终完成或判定稳定；实际以各receipt／cohort.json为准。

π0最后Keep15档首筛最佳5/40已经登记。π0.5对应档最佳为 `pi05_keep15_1_uniform`，7/40（17.5%），Full40/40；按原至少4/40门槛仍将补量，不称可用。其四预算测速代表为Keep24 uniform、Keep21 uniform、Keep18 seeded_random、Keep15 uniform，未按速度改选型。

### π0正式同卡性能已完成，整体请求收益很小

π0会话90065正常exit0，7phase各20warm＋100唯一真实观测、同一GPU3物理RTX4090；本批 `audit.json` 为passed=true、stable_for_primary_comparison=true，四预算完整。Full前后P50变化：RPC2.60%、独立vision3.26%，均低于预定10%不稳定门槛。Native-before完整RPC P50为100.911ms，独立所有相机vision P50为12.254ms；两项是独立计时，不相减／相加归因。

| vision保留层数／代表 | 首40成功数 | 对本批Native Full的RPC P50 speedup | 对本批Native Full的独立vision P50 speedup |
| --- | ---: | ---: | ---: |
| Keep24 seeded_random | 38/40 | 0.982× | 0.979× |
| Keep21 uniform | 33/40 | 0.995× | 1.095× |
| Keep18 uniform | 30/40 | 0.996× | 1.248× |
| Keep15 uniform | 5/40 | 1.016× | 1.469× |

完整RPC变化约为−1.8%到＋1.6%，与Full前后2.6%变化同量级，不能据此宣称有明确整体加速。单独vision改善不能直接当作完整模型提速；Keep15的高vision比率同时伴随很差质量。Matched-all27参照的比率另存原audit，不用较好看的matched比率冒充对原生Full的实际收益。也不从独立vision与RPC简单相减推出VLM／AE分项耗时。

严格cohort重审后已用既有registrar仅填入4个此前缺失的候选benchmark精确副本及来源sidecar，原Full／matched benchmark、旧gate及质量数据均不改写。原生主比率必须引用 `performance_cohorts/pi0_gpu3_20261021_after_screen/audit.json`；组合索引里的旧原生Full只是legacy覆盖源，不与新mask跨卡／跨批次重新计算主比率。正式确认、全部其他模型性能及最终完成审计仍未完成。

### 03:39更新：两个vision性能cohort均完成，首个200条确认已实跑

π0.5性能会话76071正常exit0，7phase均完整、同GPU3的20warm＋100唯一真实观测；`performance_cohorts/pi05_gpu3_20261021_after_screen/audit.json` 为passed=true、stable_for_primary_comparison=true。Full前后RPC P50漂移0.598%、vision0.688%；Native-before完整RPC P50为107.281ms，独立所有相机vision为12.598ms。

| π0.5 vision代表 | 首40成功数 | 对本批Native Full的RPC P50 speedup | 对本批Native Full的独立vision P50 speedup |
| --- | ---: | ---: | ---: |
| Keep24 uniform | 40/40 | 1.006× | 0.983× |
| Keep21 uniform | 39/40 | 1.029× | 1.123× |
| Keep18 seeded_random | 35/40 | 1.041× | 1.277× |
| Keep15 uniform | 7/40 | 1.031× | 1.483× |

该批完整请求的改善至多约4%，不是vision分项约1.48×的整体收益。Keep15质量很差，不能因为分项较快就称可用。其4份候选benchmark精确副本／来源sidecar均已由严格registrar写入此前不存在的位置；旧Full／matched文件及旧gate未改写。主比率引用本cohort，不跨批次重算。

### CogACT跳9五新布局不及历史参照，原规则保持不变

Full首40为37/40，历史 `reference_skip9` 为33/40；五个新布局依次1／0／0／2／19成功（各40）。新方案最高为seeded_random 19/40，相对Full51.35%，未达到60%近阈值补量门槛，不补该档200条。历史参照相对Full为89.19%，但按预先规则不参与新布局选型，**不能因它被流程排除，就把它误报为质量低或不可用**。报告将把“本轮新方案没有改进”和“既有参照依然较好”同时呈现，且参照本轮仍只有40条，不能冒充200条确认。未悄悄更改selection或重测失败以取得更好结果。

CogACT跳12五布局也已全部0/40；跳15／18仍在双卡首筛。不能由这些预定布局推断全参数空间不可行，其他未试组合仍未知。

最新不可覆盖局部dispatch为 `dispatch/closed_budgets_20261021_033758.json`：新增π0.5 Keep15、XR0跳24、CogACT跳9／12四个闭合预算及CogACT Full，累计15个闭合预算，15个选中条件。原首40哈希在π0确认已经新增样本后重新验证仍一致；`global_promotion=false`，尚不能替代最终全局promotion。

### 当前运行分配

- GPU0：CogACT9条件分片，会话9523／PID2618376，原参数2worker＋复用。
- GPU2：CogACT8条件分片，会话63871／PID2626956，原参数2worker＋复用；旧父进程退休已经完成，无T状态遗留。
- GPU3：π0 Full确认已真实启动，会话90623／编排2645199／controller2645200、端口13010。原根目录`run`、2worker、复用off、原vision gate；receipt `run/batch_receipts/pi0_full_confirm_gpu3.json`。只补init1–4，03:38实际新增75条，init0仍40条，尚未完整200。
- GPU1：新的π0.5 Full待卡等待器，会话68681／PID2647851，每30秒仅查GPU1，释放后exec原queue；原新根目录`e2_shared_service_validation`、2worker＋复用、端口13020，拟写 `batch_receipts/pi05_full_confirm_gpu1.json`，已有40条跳过。此确认尚未实际启动。

旧π0 Full GPU1等待器PID2616363已先停止／重新核对仍为pending shell，再安全退出（会话67548 exit143），凭据 `diagnostics/pi0_full_confirm_gpu3_handoff_waiter_retirement.json`。没有误杀或重复启动确认模型。后续改派π0.5等待任务时同样先核对shell／已exec状态。

**以后统计首筛进度必须只取init0。** π0 Full已有init1–4数据，主summary的canonical_quality_counts是全部合法init总数，不能再直接求和称为“/3640首筛”；使用显式collect后按init过滤，或report_artifacts的screen_init0 CSV。确认与首筛分开呈现。

仍待：CogACT最后两个预算首筛、全局promotion与早期dispatch兼容、15个已选条件补齐200、实际触发的前沿、E1合格点的同卡性能、最终图表／报告及严格逐项审计。两份vision性能已交付不代表整轮完成。

### 04:08更新：首筛队列全部退出，两个Full已完成200条

CogACT两个互斥分片会话9523／63871均正常exit0，原编排和模型进程已退出；91条件的canonical首40验证、全局promotion封存及全部旧局部dispatch的哈希兼容检查正在独立执行。不能把进程结束直接替代严格数据验收，也不把新增confirm样本算进3640条首筛分母。

π0 Full会话90623与π0.5 Full会话10563均正常exit0，原canonical validator确认各200条完整，按官方init0–4计数：

| 模型／原采样基线 | Spatial | Object | Goal | LIBERO-10 | 总体 |
| --- | ---: | ---: | ---: | ---: | ---: |
| π0 Full，D10 | 46/50 | 48/50 | 45/50 | 37/50 | 176/200（88%） |
| π0.5 Full，D10 | 50/50 | 50/50 | 49/50 | 46/50 | 195/200（97.5%） |

π0原首40为34/40，新增160为142/160；π0.5原首40为40/40，新增160为155/160。两checkpoint与原生H不同，不据这张表将模型差异归因于单个模块。

实际确认原任务终止、三张卡compute列表为空后，已继续调度原已选mask的200条确认，不重跑任何完整科学键：

| 卡 | 当前队列／会话 | 条件及receipt |
| --- | --- | --- |
| GPU0 | XR0已选跳12／15／18，11712，端口13100起 | `e2_shared_xiaomi/batch_receipts/xr_selected_confirm_gpu0_20261021_0407.json` |
| GPU2 | π0四个vision预算代表，30051，端口13200起 | `e2_shared_pi0/batch_receipts/pi0_selected_confirm_gpu2_20261021_0407.json` |
| GPU3 | π0.5四个vision预算代表，20211，端口13300起 | `e2_shared_service_validation/batch_receipts/pi05_selected_confirm_gpu3_20261021_0407.json` |
| GPU1，仍待可用 | XR0 Full确认等待器24695／PID2660508 | 每30秒仅查GPU1；空闲后exec原`run`根目录、复用off的Full确认，端口13040 |

三条新队列均2worker＋已验证复用、仍引用033758局部dispatch；实际RPC health确认首条件分别为XR0 skip12 short_runs／π0 Keep24 seeded_random／π0.5 Keep24 uniform，同PID且真实parity=true。尚不能把这些启动的队列报成完整确认。

π0.5原GPU1等待器PID2647851先经pidfd停止并重新确认仍是pending bash，之后只退出该shell，再将其确认交给GPU3；凭据 `diagnostics/pi05_full_confirm_gpu3_handoff_waiter_retirement.json`。当前XR0等待器不同，不误杀已exec模型。CogACT Full200仍待调度，其他确认与E1性能／最终图表审计尚待完成。

另由6.1 sol medium准备三个**独立新模块**：`sparse_camera.py`、`sparse_camera_probe.py`、`test_sparse_camera.py`。主代理已阅读实现并重跑6项CPU测试全部通过；测试使用固定原生Observable计时源码，但物理与渲染为mock，不冒充真实仿真证据。候选只缓存不会被策略消费的中间RGB，保持原生采样时钟；视频／reset／初始10步不变。正式方案2.1已在真实probe前登记每模型16条的上限与严格输入／物理／RNG配对要求。没有启动此probe、没有改NN或冻结源码，也没有生产迁移；现有复用批准不等于批准新相机策略。优先完成在途既定质量确认。

### 04:10全局首筛验收及选型已封存

独立进程会话27755正常exit0：91个质量条件逐一通过原canonical scene／identity validator，首筛共3640条。原冻结 `controller.promotion` 的17个预算均闭合，无incomplete decision；生成不可覆盖的 `dataset_bundle_v3_all_models/view/promotion.json`，全部7份旧局部dispatch的首40文件哈希及选型兼容复核通过。凭据 `diagnostics/global_promotion_20261021_0407.json` 为passed=true。

最终入选15条件：4个Full、π0与π0.5各4个vision代表、XR0跳12／15／18三个代表。原先局部名单没有变化，当前三路确认仍合法。既定质量总预算现可明确为3640首筛＋15×160新增确认＝6040条，E0／E2 profile及性能请求不混入这个分母。

CogACT跳15、跳18各五布局也均为0/40；其新跳9最佳19/40（相对Full37/40为51.35%），跳12五布局均0/40。因此CogACT没有新的mask进入补量，仍须完成Full200；历史reference_skip9为33/40，单列保留，不伪称新赢家。XR0跳21／24各五布局均0/40。两模型初始最高预算都未达到扩展触发条件，故不启动更高预算前沿；这不是未测布局不可行的证明，也没有事后调低门槛以选入更多点。

### 04:13新增分析快照与后续排队

- [完整首筛表、质量图、20张固定XYZ及双侧视频索引](../../../results/idea1_v3_1/diagnostics/screen_complete_20261021_0411/media.md)：共109个产物，missing_evidence=0，所有生成链接目标校验通过。包含91条件完整init0结果；combined_init0_4列是生成时确认进度快照，不能当最终200条表。XYZ场景统一四suite的task0/init0，覆盖CogACT历史跳9参照与新seeded_random、XR0跳18以及两π模型Keep18，并非按成功／失败挑轨迹。主代理实际查看CogACT两张Spatial XYZ，蓝色原生Full、橙色点虚线可区分，使用真实遥测、毫米单位、无补零；同时间距离仍混合路径与行进快慢，不能解释为纯横向误差。
- [42张失败最远阶段图索引](../../../results/idea1_v3_1/diagnostics/phase_screen_complete_20261021_0413/plot_index.json)：由同一快照生成，筛选与确认列分开，左绝对数、右仅失败样本内比例；阶段不是致败原因。
- 加入6项新相机模块测试后，主代理CPU隔离重跑 `unittest discover -s preparations/idea1_v3_1 -t . -p 'test_*.py'`，91/91通过、41.508秒；只有已知JAX／Flax deprecation warning，没有加载真实GPU模型。这不替代新相机策略的真实仿真验收。
- CogACT Full200已有明确后续调度：会话8471／等待器PID2676917持有当前自有XR0队列PID2671303的pidfd，只等待该进程实际退出，**不扫描自有忙卡**；只有原receipt为completed且全部job成功／合法跳过，才exec GPU0的CogACT Full confirm。使用全局promotion、2worker＋已批准复用、端口13400，拟写 `e2_shared_cogact/batch_receipts/cogact_full_confirm_gpu0_after_xr_20261021_0415.json`。此时仍是等待，不冒称已开始；controller实际启动前仍会检查分配卡是否空闲。XR0 Full的GPU1等待器独立保留，不能重复调度。

当前整轮仍未完成，剩余包括全部入选变体／另外两个Full的200条确认、E1确认合格点的同卡性能、最终数据汇总与严格审计／报告。新相机候选尚未真实运行或上线。

### 04:15冻结首筛的描述性分析（不是最终确认结论）

由完整首筛CSV独立生成 `diagnostics/screen_findings_20261021_0414.json`，保存源CSV SHA、17预算×5布局的原始1-based跳层列表、四suite成功数与布局间极差；455行scope／suite记录均完整。后续确认数据不能改写这张首筛快照。

关键观察：**同样剪层数，布局差异非常大，不能只按剪层比例预估质量。** 下表每格是40条成功数，布局次序固定为uniform／front_middle／back_middle／short_runs／seeded_random。

| 模型／跳层预算 | 五布局成功数 | 解释边界 |
| --- | --- | --- |
| π0 vision跳3 | 36／35／34／1／38 | 最好与最差相差37条，但首40排名不是独立确认 |
| π0.5 vision跳3 | 40／37／40／0／40 | 原始L1–3一起跳过严重退化，不能定位单层因果 |
| π0.5 vision跳9 | 32／0／30／0／35 | 保留18层仍有多个有希望的布局，正在补量最高者 |
| π0.5 vision跳12 | 7／0／1／0／3 | 本轮五布局均质量很差；不证明所有Keep15都不可能 |
| XR0 VLM跳15 | 2／0／0／0／36 | seeded_random明显优于另外四个既定布局，仍待200条 |
| CogACT VLM跳9 | 1／0／0／2／19 | 全部不及历史参照33/40，不能把参考布局忽略掉 |

π0.5 vision的实际对照进一步限制归因：跳`[17,18,19]`仍40/40，跳`[8,9,10]`为37/40；因此“连续跳层一定有问题”不成立。跳`[2,4,25]`也40/40，故`[1,2,3]`的0/40不能扩展成“所有靠前层、或L2单层绝不能剪”。需要单层／成组受控消融才能拆开重要层和组合交互，本轮没有为这些解释临时增加新的质量条件。

### 04:23确认推进与性能证据整理

XR0跳12 short_runs完整200条已通过canonical确认：185/200（92.5%），首40为38/40，新增160为147/160；四suite为47／48／46／44（各50）。XR0 Full200尚待执行，因此此时不拿其Full首40的97.5%去当最终200条相对保留率分母。凭据 `diagnostics/confirmed_200_20261021_0420.json`。GPU0队列已自然进入跳15 seeded_random，当前controller2679167／端口13101，没有中断或改写前一条件。

π0与π0.5队列也均已自然进入Keep21 uniform，controller分别2681516／2679696、端口13201／13301；Keep24的最终质量与配对将在独立完整200校验后登记，不沿用正在执行时的部分计数。

两份vision性能cohort已再次调用严格完整cohort审计，结果与原audit逐项一致：passed、稳定、四预算覆盖均为true。归集凭据 `diagnostics/performance_findings_20261021_0418.json` 保存原audit SHA、实际GPU、before／after数值及native／matched两种比率。没有重跑请求或质量样本；未用新mask与旧跨卡Full重算速度。

另从先前独立审计的128条同服务E2 profile整理 `diagnostics/e2_throughput_findings_20261021_0422.json`，没有冒充新的原始审计。8个相同episode的墙钟与双worker比率如下：

| 模型 | full_io单worker秒 | lean_io单worker秒 | reuse_env单worker秒 | two_workers秒 | 双worker／lean吞吐比 | 双worker／reuse吞吐比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| π0 | 110.627 | 89.676 | 82.706 | 52.782 | 1.699× | 1.567× |
| π0.5 | 113.870 | 92.083 | 87.140 | 55.453 | 1.661× | 1.571× |
| XR0 | 92.336 | 80.594 | 72.114 | 49.373 | 1.632× | 1.461× |
| CogACT | 97.354 | 87.372 | 78.152 | 55.151 | 1.584× | 1.417× |

四模型优化组与原生full_io的配对请求数分别246／254／119／98，各组固定控制最大动作差为0，初始图像／数值状态／结果配对已由原audit检验；原始raw不存在时不伪称raw动作完整比对。四arm是固定顺序、小批8场景，公共首输入观察I/O计入arm墙钟；只表示这个profile的整体评测吞吐，不能当单机器人请求延迟、batch推理或独立反平衡因果实验。双worker共享串行RPC，NN与D/H/R未改。

### 04:25两个Keep24的完整质量及配对分析

原canonical validator已分别确认Keep24完整200条、官方task／init及初态hash与各自200条Full配对一致，凭据 `diagnostics/vision_keep24_confirmed_20261021_0423.json`：

| vision候选 | Spatial | Object | Goal | LIBERO-10 | 总体 | 首40／新增160 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| π0 Keep24 seeded_random | 49/50 | 49/50 | 45/50 | 41/50 | 184/200（92%） | 38/40；146/160 |
| π0.5 Keep24 uniform | 50/50 | 50/50 | 48/50 | 49/50 | 197/200（98.5%） | 40/40；157/160 |

π0与其Full的四格为双成功170／Full独赢6／variant独赢14／双失败10，净＋4个百分点；π0.5为194／1／3／2，净＋1个百分点。这些不是“恢复了14条所以提升7个百分点”：还必须扣掉新增失败6条，并使用完整配对分母。

为避免将小样本正差称为稳定提升，事后增加一次只读CPU描述性分析：四suite内分别按10个task成组有放回抽样，保留每task的全部初态、20000次、seed1701，配对成功率差的95% percentile区间见 `diagnostics/keep24_paired_bootstrap_20261021_0425.json`，保存全部800份原result的SHA：

| 候选 | 合计200相对Full差／区间（百分点） | 新增160相对Full差／区间（百分点） |
| --- | --- | --- |
| π0 Keep24 | ＋4.0；[-0.5, 8.5] | ＋2.5；[-1.875, 7.5] |
| π0.5 Keep24 | ＋1.0；[-0.5, 2.5] | ＋1.25；[-0.625, 3.125] |

两个区间都包含零，不证明剪枝稳定提升总体质量。该分析是事后探索性补充，不是新选拔标准、不改原门槛，也不把bootstrap当全面统计保证。200条包含参与选型的40条；新增160未用于mask选型，但仍是相同任务的新官方初态，不称独立新任务泛化。

目前15个既定确认条件中已有5个完整200（两个π Full、两个Keep24、XR0跳12），其余仍在执行／等待。Keep24数值质量较好不等于部署加速：其native分母的请求比率分别约0.982×／1.006×，当前几乎没有速度收益。完整结论须等待其他预算。

### 04:33小米跳15完整确认、正式性能图已交付

XR0跳15 seeded_random的完整200条已通过canonical检查：183/200（91.5%），Spatial／Object／Goal／LIBERO-10为47／50／44／42（各50）；首40为36/40，新增160为147/160。凭据 `diagnostics/xiaomi_skip15_confirmed_20261021_0433.json`。与跳12的185/200相差2条，但两个mask不嵌套，不能将差异单归于“多剪3层”；Full200尚待执行，暂不报告最终相对保留率。GPU0已进入跳18 seeded_random，controller2686311／端口13102。15个既定确认条件中现有6个完整200，整轮仍未完成。

6.1 sol medium仅新增 `performance_figures.py`／`test_performance_figures.py`，没有修改冻结源码或在途运行数据。主代理完整阅读实现、独立重跑6项CPU测试（1.648秒）全部通过，再生成以下真实数据图并实际查看：

- [E2整体评测吞吐与双worker增量](../../../results/idea1_v3_1/diagnostics/performance_figures_20261021_0434/e2_throughput.png)：左图以full_io为参照，右图另列相对lean_io／reuse_env的增量，明确不是单请求推理或batch加速。
- [vision剪枝：完整RPC与独立vision两种P50比率](../../../results/idea1_v3_1/diagnostics/performance_figures_20261021_0434/vision_native_p50.png)：只画同卡native Full-before主分母，横向为Keep24／21／18／15；matched分母单独保留在原值CSV。可直观看出分项改善并未变成同等的整体收益。
- [原值CSV、输入／源／产物SHA及原审计引用索引](../../../results/idea1_v3_1/diagnostics/performance_figures_20261021_0434/plot_index.json)。两张PNG、三份CSV共5个产物，索引另列。

绘图校验了两份findings及其26个原audit／receipt／benchmark引用的哈希、manifest、GPU身份、四预算、100行P50与比率算术，**不是新raw审计**，也不代替最终质量验收。图中文字／数值无明显重叠，参考线1清楚；所有真实新图只用CPU生成，不加载模型。若未来输入不稳定批次，模块会保留并显式标记UNSTABLE，不挑稳定结果；本次两vision批次均稳定。

### 04:38确认进度与计算语义澄清

π0.5 Keep21 uniform完整200条通过canonical检查，并与自身Full200的官方task／init初态hash逐一配对：194/200（97%），四suite为50／50／48／46（各50）；首40为39/40，新增160为155/160，Full195/200。同卡native-before完整RPC比率约1.029×，不是vision独立计时的1.123×。凭据 `diagnostics/keep21_confirmation_progress_20261021_0437.json`。本次快照中π0 Keep21尚不完整，未用部分数去排名；π0.5队列已进入Keep18 seeded_random，controller2687761／端口13302。现有7/15个既定确认条件完整200。

包含新增性能图模块的全CPU隔离测试已完整重跑：97/97通过、37.602秒，仅已知JAX／Flax deprecation warning；没有加载真实GPU模型。测试中的假PID／随机toy参数不属于科学评测样本，科学E0仍以原trained checkpoint和真实bank门槛为准。

源码复核另存 `diagnostics/vision_and_kv_source_review_20261021_0438.json`（五个已查看源文件SHA），供最终报告正确解释计时：

- **图像分支≠物理相机数。** raw观测来自agentview与robot0_eye_in_hand两路；OpenPI原生LiberoInputs建立base／left-wrist／right-wrist三个分支，其中right-wrist为zeros_like填充、非FAST模型的mask=False。`embed_prefix`源码对各图像分支调用编码器后拼接mask；本轮standalone vision明确返回三个分支、每分支256 token。此前“3相机”计时口径应理解为模型三个输入分支，不能写成三台真实相机。没有为该发现修改预处理、减少token或新跑推理；也没有单独量到填充分支的真实硬件成本，不能直接据此宣称缓存能节省1/3时间。
- **VLM跳层并非全部计算物理删除。** XR0与DB-CogACT既有adapter在跳过hidden更新时仍保留K/V投影、rotary／cache更新与原层索引，audit中的fully_eliminated_layers为空。它们跳过Q／attention／O／FFN hidden更新，原AE与readout不改；因此不能按“跳15/36层”直接计算整机FLOPs或速度同比例降低。CogACT此实现也保留原生use_cache=True，虽然动作路径读最终cognition特征。本轮只解释既有路径，不擅自另造无KV版本。

两条均为只读源码及既有协议解释，不冒充新GPU profile／运行时张量捕获。填充分支缓存、删除完全mask token等只能列为未来需独立parity验证的方向，未实现／未测，不加入本轮收益。

### 04:41 π0 Keep21确认下降，任务类型差异明显

π0 Keep21 uniform的完整200条canonical检查及对Full官方初态hash配对通过，凭据 `diagnostics/pi0_keep21_confirmed_20261021_0440.json`：156/200（78%），首40为33/40（82.5%），新增160为123/160（76.875%）。这是完整确认结果，不能继续引用首筛82.5%来代表最终质量。GPU2已自然进入Keep18 uniform，controller2691644／端口13202。现有8/15个既定确认条件完整200。

| π0条件 | Spatial | Object | Goal | LIBERO-10 | 总体 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 46/50（92%） | 48/50（96%） | 45/50（90%） | 37/50（74%） | 176/200（88%） |
| Keep21 uniform | 44/50（88%） | 48/50（96%） | 33/50（66%） | 31/50（62%） | 156/200（78%） |
| 差值（百分点） | −4 | 0 | −24 | −12 | −10 |

同一个mask在Object保持原成功数，但Goal损失12条；这提示该**层组合在该checkpoint／原采样配置下**对任务类型有不同影响，不宜仅看总体78%。不能据此定位“Goal专用层”，也不能把跨checkpoint的π0.5 Keep21 97%与π0 78%差异全归架构，因为checkpoint、原生H及完整下游模块不同。本批是每suite10任务×5init，非50个独立新任务。

该mask的同卡完整RPC比率约0.995×，即本批没有明确整体速度收益，不能将vision分项1.095×当作质量损失换来的整机加速。剩余Keep18、Keep15及两个Full仍须完成，不能据当前部分结果提前封存报告。

### 04:48小米跳18确认完成，CogACT Full已真实接续

XR0三条件确认会话11712已正常exit0，原receipt为completed、三个job均exit0；原编排2671303已退出。跳18 seeded_random完整200通过原canonical检查：139/200（69.5%），首40为26/40，新增160为113/160；Spatial／Object／Goal／LIBERO-10为38／44／31／26（各50）。凭据 `diagnostics/xiaomi_skip18_confirmed_20261021_0447.json`。至此9/15个既定确认条件完整200，仍不是全轮完成。

**绝对69.5%不能直接判定不满足相对70%目标。** 目标分母是XR0 Full200的实际成功数，而非固定200；139/198已超过70%，139/199略低于70%。Full首40固定39成功，仍须等待其新增160结果，不能拿39/40外推或先行宣布通过／失败。XR0跳12／15／18质量数据均保留，最高可用预算稍后按最终基线判断。

原等待器会话8471／PID2676917通过自有XR0队列pidfd实际退出事件、completed receipt及全部exit0，已exec CogACT Full确认；没有重新扫描自有忙卡或给NN发信号。当前controller2695783、服务2695856、端口13400，两个真实rollout worker2696273／2696274已启动。独立RPC health确认condition=db-cogact_full、同服务PID且real_input_parity_passed=true；D10原生DDIM、H16、原suite R保持。receipt为 `e2_shared_cogact/batch_receipts/cogact_full_confirm_gpu0_after_xr_20261021_0415.json`，只新增init1–4的160条，已有init0跳过。

当前GPU0为CogACT Full、GPU2为π0 Keep18、GPU3为π0.5 Keep18；GPU1的XR0 Full等待器仍是pending bash PID2660508／会话24695，每30秒仅查GPU1，不是已启动的NN。如果其他卡先完成，必须先确认该等待器仍为bash并安全撤销后再改派，不能重复运行Full。后续仍需XR0 Full、剩余四个vision确认、E1同卡性能、最终全部图表／严格审计与报告。

E1性能边界已在方案3.2登记：XR0主要测最终合格的最高跳层新候选；CogACT无新候选通过补量门槛，可用一个有界cohort单列历史reference_skip9与新最佳skip9的探索性速度，质量仍各只有首40，不称200条确认的等质量收益。没有为此改选型或补跑参考质量。

本次另只读复核了idea2完整交接方案、最终报告和现存final_audit：E0/E1/E2、同卡Graph补充与112闭环均有对应交付，receipt passed=true、missing/errors空、graph_required=true；动态Graph闭环、真实相机采集年龄和持续时钟deadline明确未验证而非记0。没有重跑idea2，也没有修改用户idea文档。

### 04:52 π0.5 Keep18完整确认，已进入最后Keep15预算

π0.5 Keep18 seeded_random完整200条canonical检查及自身Full200官方初态hash配对通过，凭据 `diagnostics/pi05_keep18_confirmed_20261021_0451.json`：183/200（91.5%），首40为35/40，新增160为148/160。队列已进入最后Keep15 uniform，controller2697052／服务2697144、端口13303。当前10/15个既定确认条件完整200。

| π0.5条件 | Spatial | Object | Goal | LIBERO-10 | 总体 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 50/50（100%） | 50/50（100%） | 49/50（98%） | 46/50（92%） | 195/200（97.5%） |
| Keep18 seeded_random | 48/50（96%） | 50/50（100%） | 47/50（94%） | 38/50（76%） | 183/200（91.5%） |
| 差值（百分点） | −4 | 0 | −4 | −16 | −6 |

主要质量损失集中于LIBERO-10，并非所有suite平均退化；Object原成功数保留。本批完整RPC同卡P50比率约1.041×，独立vision约1.277×，两者不能互换或相减归因。相对Keep21的97%，Keep18下降5.5个百分点，而完整请求比率只从1.029×到1.041×，当前不是很有吸引力的质量—整体速度取舍；该判断限于本批候选与原生采样配置，未搜索全布局空间。

05:00前后接下来应优先核对CogACT Full会话8471的实际结束及其200条，再在有卡时安排XR0 Full。如果GPU1等待器已exec则不得重复；若仍pending bash，按已有pidfd停止／复核／退出shell流程安全改派。CogACT探索性计时已做只读CPU dry run：四phase、100唯一真实输入、端口14000–14003、未来新目录 `performance_cohorts/cogact_gpu0_20261021_after_full_confirm`；未创建该目录、未查GPU或启动NN。原生Full-before／历史skip9参照／新最佳skip9／Full-after，均各20warm＋100，仍须在必要质量确认之后实际执行，dry run不计完成。

### 04:57已完成200条条件的失败进展与配对分析

两个只读派生表已保存：`diagnostics/confirmed_failure_progress_20261021_0455.json`（10个完整200条件，含2000份result SHA）与 `diagnostics/paired_failure_progress_20261021_0457.json`（仅与自身Full200配对的5个完整π候选，引用前表源SHA）。未将缺少Full200的小米条件误当作已完成反事实配对。

**总失败数不能直接当剪枝新增失败数。** π0 Keep21的44条失败中有12条Full也失败；与Full相比，32条Full成功但variant失败，同时12条Full失败但variant成功，故净少20条成功。π0.5 Keep18则是15条Full成功但variant失败、3条反向恢复，净少12条成功。

| 完整200候选 | Full成功、variant失败的配对数 | 这部分最远仅初始／接近 | 这部分有接触或更后进展记录 | 此部分在四suite的数量 |
| --- | ---: | ---: | ---: | --- |
| π0 vision Keep21 uniform | 32 | 7 | 25 | Spatial5／Object2／Goal13／LIBERO-10 12 |
| π0.5 vision Keep18 seeded_random | 15 | 1 | 14 | Spatial2／Object0／Goal2／LIBERO-10 11 |

因此π0 Keep21并非所有配对退化样本都到了精细阶段；π0.5 Keep18的该子集多数已有后续操作进展，且主要是LIBERO-10。以上只是**结果分歧样本的最远进展**，不能推出哪个阶段的瞬时失败风险更高、夹爪造成失败或剪枝是唯一原因；并未沿Full轨迹做同状态动作反事实。unknown阶段计数为0也不证明contact／grasp负值都可靠。

XR0跳18的61条总失败单列：13条最远仅接近，48条记录到接触或更后（接触3／抓取或操作14／搬运3／对齐4／释放或后期24）。这不是32／15配对子集的共同分母，也不是“48条在这些阶段被夹爪弄失败”。Full200未完成前只保留该描述，不提前归因。

π0 Keep21的Object虽与Full同为48/50，也不是同一批成功完全没变化：Object有2条Full独赢及2条variant独赢，净值相抵。最终报告会同时呈现成功率与配对分歧，避免“总数相同＝所有轨迹都不受影响”的误读。所有阶段图仍须注明最远进展而非致败阶段。

### 04:59 XR0 Full接续已排队，避免CogACT结束后等待空转

新增运行编排会话17824／PID2702853，持有当前自有CogACT队列2676917与原GPU1等待器2660508的pidfd。当前只是等待，不是已启动的Full；不扫描自有忙卡。触发后先验证CogACT receipt正常completed、全部job exit0／合法跳过、Full200 canonical完整，再判断XR0 Full是否已完成／GPU1是否已exec。

- 若XR0 Full已完整200，直接退出，不重复；若等待器已exec为Python队列，则不终止、不重复启动。
- 只有仍是原pending bash时才STOP，停止后重新核对cmdline；若发生exec竞态，只CONT恢复已执行队列，不TERM。确认仍为同一原shell才TERM＋CONT并等其终止，然后exec GPU0上的XR0 Full确认。旧会话24695若因此exit143为有意退休等待shell，不是NN失败。
- 原等待器若意外终止而Full不完整，或身份／状态不能确认，自动接续停止并保留人工检查要求，不能仅因handle丢失就重启模型。

拟写handoff凭据 `diagnostics/xiaomi_full_gpu0_handoff_after_cog_20261021_0459.json`；新质量receipt为 `run/batch_receipts/xiaomi_full_confirm_gpu0_after_cog_20261021_0459.json`，使用全局promotion、原`run`根目录、2worker／复用off、端口13500、原真实bank，仅新增init1–4。controller实际启动前仍检查分配卡空闲。原GPU1每30秒等待器在接续触发前继续保留机会，若该卡先释放则照常在那里开跑。

后续不要再手工重复改派同一Full，应先核对17824的真实输出／PID实际cmdline／上述receipt。三路现有NN仍正常执行，科学键／源码／D/H/R均未改变。

### 05:03两个确认完成，XR0 Full已在GPU0真实开跑

原会话8471正常exit0，CogACT Full200 canonical检查通过：188/200（94%），Spatial／Object／Goal／LIBERO-10为45／50／48／45（各50）；首40为37/40，新增160为151/160。π0 Keep18 uniform也完整200：133/200（66.5%），四suite为37／39／37／20（各50）；首40为30/40，新增160为103/160。两项凭据 `diagnostics/pi0_keep18_and_cogact_full_confirmed_20261021_0501.json`，原有效结果均未重试。12/15个既定确认条件现已完整。

π0 Keep18的完整请求同卡比率约0.996×，独立vision约1.248×，即明显质量损失没有换来明确整体速度收益。相对Keep21，本批Goal从33/50升到37/50、Object从48/50降到39/50，存在**两个已测组合之间的suite排名交叉**；但预算与具体mask同时变化、非嵌套、Goal只差4条，不能归因为单层任务专属性或证明总体更优。应与本轮严格同预算的首40布局表分开解释。

接续会话17824／PID2702853已实际执行：确认CogACT completed及canonical Full200后，STOP并复核原GPU1等待器2660508仍为同一pending bash，再仅TERM＋CONT该shell；旧会话24695 exit143，原PID已退出，没有暂停父进程或NN残留。handoff凭据 `diagnostics/xiaomi_full_gpu0_handoff_after_cog_20261021_0459.json` 保存前／停止后cmdline及no_nn_signalled=true。

XR0 Full新controller2703016／服务2703084、端口13500，独立RPC health核对同PID及condition=xiaomi-robotic0_full、real_input_parity_passed=true、D2／H10／R10。receipt `run/batch_receipts/xiaomi_full_confirm_gpu0_after_cog_20261021_0459.json`。原根目录`run`、2worker／复用off，只补已有40之外的160条；没有重复启动GPU1 Full。当前GPU2已进入π0 Keep15 uniform（controller2700837／端口13203），GPU3仍π0.5 Keep15 uniform（2697052／13303）。

为保留第四张卡释放时的机会，新增**CogACT性能**等待器会话44426／PID2703826；这不是已退休的XR0等待器，也不是NN。仅在pending时每30秒检查GPU1，空闲后exec原 `performance_cohort run`，端口14100–14103，拟写 `performance_cohorts/cogact_gpu1_20261021_after_full_confirm`。模型Full质量已完整，四phase仍Full-before／历史skip9／新最佳skip9／Full-after，各20warm＋100真实输入；不新增质量样本。不扫描自有忙卡；若别的卡先可用，须先检查该等待器实际bash／已exec状态，再安全改派，不可重复同cohort。此前GPU0 dry run目录未创建，不代表该计时已完成。

### 05:12正式计时的边界接续与任务交互不确定性

为避免卡释放后等待空转，两项**所需E1计时**已排入自有进程退出接续；都不是已完成的NN测量，未改变质量预算：

1. CogACT：会话43718／PID2705496持有π0.5队列2671448与现GPU1待卡shell2703826的pidfd。π0.5四条件全部完成、原receipt正常exit0且四个canonical200及CogACT Full200复核后，若CogACT等待器仍为pending bash，先STOP／重新核对／TERM＋CONT并等其退出，再在GPU3运行四phase计时。若原等待器已exec，保留GPU1任务，不重复；意外终止则停止并留人工检查要求。拟写 `diagnostics/cogact_performance_gpu3_handoff_after_pi05_20261021_0506.json`，目标 `performance_cohorts/cogact_gpu3_20261021_after_full_confirm`、端口14300–14303。原GPU1等待器在此之前仍仅每30秒检查GPU1。
2. XR0：会话70829／PID2706133持有当前Full队列2702853的pidfd。Full正常结束并确认200后，从**原全局promotion的非参照、完整200候选**中，按原70%相对保留率门槛选择最大跳层数；不是先测速度再选赢家。核对全部官方task／init及初态hash，另存Full／候选成功数、原result SHA与选型凭据 `diagnostics/xiaomi_highest_confirmed_timing_selection_20261021_0508.json`，再在同GPU0执行Full-before／candidate／Full-after各20warm＋100真实输入。目标 `performance_cohorts/xiaomi_gpu0_20261021_after_full_confirm`、端口14400–14402。最终Full结果尚未知，因此此时没有提前指定通过门槛的候选或填入比率。

两接续都只等自有进程事件，不扫描自有忙卡；下一步手工操作前须先核对实际cmdline和各会话输出，不能重复启动。模型／NN源文件、D/H/R、已完成数据与全局promotion不变。

另将π0 Keep18与Keep21的Goal／Object排名交叉做事后CPU不确定性检查，保存 `diagnostics/pi0_vision_suite_interaction_20261021_0510.json`（400份result SHA、同官方初态、每suite内10个task成组bootstrap20000次、seed1701）。200条中，Keep18相对Keep21的Goal差＋8个百分点，95%描述区间[-6,24]；Object差−18，区间[-36,-2]。Goal−Object的差中差为＋26，区间[4,50]；仅用未参与选型的新增初态后为＋22.5，区间[0,50]。因此**不能单凭点估计断定Keep18真的提高Goal**；交互有探索性迹象，但新增初态区间已碰到零，且样本仍是相同任务。

这是看过结果后的描述分析，没有多重比较／选型校正，也没有控制两个mask的剪层数与具体位置变化；不得据此声称定位任务专属层、证明单层因果或改变选拔标准。完整首筛中未找到“总体至少20/40且Goal比Object多至少2/10”的候选，这只是对现有表的事后筛查，不是新的实验或不可行性证明。

具体执行边界见[方案](idea1_v3_1_experiment_plan.md)与 `preparations/idea1_v3_1/README.md`。原[准备状态](idea1_v3_1_preparation_status.md)保留为启动前快照，以本文件为当前进度。
