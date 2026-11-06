# D10 实验完成性审计

最终审计快照：2026-10-24T14:54:55.988189+00:00。本轮仅使用 CPU 读取结果、NPZ、CSV、只读 SQLite 和源码快照，没有运行模型、GPU 探测或环境，也没有修改运行时代码。

对照依据：[实验方案](../../../docs/experiments/idea2/idea2_d10_dynamic_sm_plan_20261024.md)。**39 项机器检查全部通过，异常 0；必要门禁、正式反向复测、SLO、匹配 trace、公平接口控制和 24 例有限闭环均已完成。** 机器记录见 [requirements_audit.json](requirements_audit.json)，保存检查计数、边界和 55 个证据文件 SHA。审计通过表示完成性与证据一致，不表示所有候选有收益，也不将历史失败改为通过。

## 逐项状态

| 要求 | 状态 | 已核对证据与边界 |
|---|---|---|
| D2 必要回归 | 已通过 | `d2_regression/result.json` 为 passed、lifecycle passed；12 条实测，performance_eligible=false。只作回归。 |
| D10 原时间递推、恰好十次更新、逐轮原始输出 | 已通过 | `iteration_audit/result.json`：13 个真实同任务输入，52 请求、520 轮，all_bitwise=true；max≤1e-5、MAE≤1e-6 未放宽。诊断捕获额外保存中间量，不列入性能。 |
| 逐轮产物独立复核 | 已通过 | `iteration_artifacts_independent_audit.json`：52 请求、520 轮，全部数组与 SHA 通过。审查者另以 CPU 重读 13 oracle＋52 Graph NPZ，11 类数组/位字节均匹配。 |
| 两槽、原地更新、reset、在途生命周期 | 已通过指定覆盖 | 同一诊断中 slots 0/1 各26次，epochs 1/2各26次；两套各36个KV地址不交叉并保持稳定；52次复用同一外部输入地址、recaptures=0。四请求生命周期路径覆盖0→1→0→reset后0；活lease拒绝reset、Encoder pending drain、旧epoch拒绝均通过。 |
| 没有第三份完整KV、Graph可写空间分离 | 代码/门禁已支持 | Pool双槽地址和生命周期稳定；Oracle先释放再捕获，性能组关闭Graph后才构建参考KV。`additional_full_kv_allocations=0`是实现声明；当前证据不等于完整分配器追踪。 |
| G0整段与G1分段对照 | 正式与独立反向复测已过 | `d10_whole_formal_stable` / `d10_whole_repeat`、`d10_split_full_formal_stable` / `d10_split_full_repeat` 均passed。每运行20 warmup、100对、两臂各100连续输出，共400实测行。repeat反转首臂及连续组次序。 |
| 分图/额外gate成本一致 | 已具备对照 | G0=whole/native/full；G1=per_step/wait_prefix/full；G2=per_step/wait_prefix/固定分区；G3=per_step/full/分区→primary。不能把G3只胜G2写成相对G0/G1净收益。 |
| 独立SM曲线与候选缩减 | 已完成筛选 | `sm_curves`、`sm_curves8`：各阶段20 warmup、30次，full重复点；另有`encoder_pair_curves`及报表CSV/PNG。它们明确performance_eligible=false，不是端到端性能。 |
| G2/G3两配额小批对照 | 已完成并停止扩展 | 8/120、16/112固定/动态四组各20对＋两臂20连续输出、80行，均通过数值/lifecycle/清理。均为筛选数据；当前无净优势不要求机械扩为100对，更不追加任意SM搜索。 |
| 真实资源切换 | 已通过 | `d10_dynamic16_trace_audit.json` 8帧：AE1 8,432 kernels@16SM；VLM 7,072@112SM；AE2–10 75,888@primary；8个尾段边界通过，model-kernel gap 4.288–5.376µs。primary表示不受配额限制，不表示占满128SM。 |
| P4固定Encoder配额控制 | 已完成小批、未推荐扩大 | `d10_encoder64_vlmfull_screen`：20×16帧×3臂＋各20连续，共1,020行；lifecycle通过。配套resumed trace实证Vision64、AE64、VLM primary。不能把约1.6ms阶段微基准收益当整请求收益。 |
| O0/O1/O2当前旧KV公平对照 | 正式与反向复测已过 | `d10_oldkv_formal`、`d10_oldkv_repeat`各700行：两个100对混合组＋三模式各100连续。全部age=1、每请求Prefill=1、steady recapture=0、数值通过；每混合组capture_engines=1且same_graph_selective_passed=true。两运行均记录源码前后未变。 |
| 相同提前返回接口的GPU串行控制 | 正式与反向复测已过 | `d10_async_serial_formal` / `repeat`各400行，100对＋每臂100连续；20对gate及8请求trace另列。两臂同vision_only Graph、每组engine=1、两槽地址稳定，只将gate从VLM前移至Vision前；age1/Prefill1/recapture0/误差0。wrapper SHA/快照相同，all_work=响应wall＋到排空起点间隔＋drain，统计Event开销不混入drain。 |
| 提前返回串行控制的真实Kernel验证 | 已通过 | `trace_analysis/oldkv/async_control_analysis`：第四条116,172个Kernel CSV与SQLite逐条相同；8请求AE×Vision/VLM交集严格0，8/8在动作返回后Prefill仍继续，Vision在本请求AE后、VLM在Vision后；与O1的136种signature及数量完全匹配。 |
| 当前帧跨帧F0/F1/F2/F4完整轨迹 | 正式与独立反向复测已过 | `d10_crossframe_formal` / `d10_crossframe_repeat`各6,800行：100组×4臂×16帧＋4臂各100连续。每轮100组输入签名、1,600组按(pair,index)匹配的观测/seed均一致，原始/解码误差0、age0、recapture0；两槽、reset和生命周期通过。每组Prefill数=消费数=请求数，fill/drain纳入。frame是13条bank的循环索引，不能误当16帧轨迹位置。 |
| 新KV路线真实Kernel重叠/竞争 | 已完成6条短trace离线分析 | `trace_analysis/trace_summary.json`、`resource_audits.json`、CSV与README。每条8帧；全部Graph阶段/请求映射通过，5个候选各匹配7,738种signature、未匹配0。已覆盖serial、layer overlap、固定16、动态16、combined、Encoder64。 |
| 当前旧KV配置的O0/O1/O2匹配trace | 已完成且独立复算 | 三条各8请求，实际103,160个模型Kernel无归属缺失/歧义。审计逐条检查每条116,172个CSV Kernel的时间/设备/context/stream与只读SQLite相同；重新求区间交集：O0 AE×Vision/VLM=0/0，O1=88.821898/0ms，O2=86.209196/83.172738ms（每条8请求合计）。两候选136种signature及数量均匹配。后台Kernel不按Host返回截断。 |
| 33/62/近容量到达与SLO | 六组正式与反向复测已过 | 33/62/71ms各两轮、每轮4臂×100=400，共2,400行。独立从200条F1原始轨迹重算内部周期67.01452889945358ms，ceil(1.05×周期)=71ms；预算0/2/5/10%，默认5%未改变。逐行source/seed/到达/丢帧与capture延迟公式匹配，24行汇总、426源码快照及624 same-source smoke通过。62ms原repeat启动失败保留，使用resumed。 |
| 有限真实闭环 | 旧KV 16例、新KV 8例均有效完成 | O0/O1各4/8成功、2137步、428请求；428对输入与动作逐位一致，排唯一cold capture为427对，排全部8次bootstrap为420对。fresh为8/8成功、1181步、239请求且全部age0。8场景分别与O0/O1比较的16个bootstrap，初始完整输入SHA/seed/协议（除mode）及动作全部相同。任务失败保留且与技术失败分开；跨GPU不排名时延。 |
| 复现/CSV/trace产物 | 已具备 | 每结果含命令、配置、源码/bank hash；公平控制还保存wrapper自身SHA与快照。实际命令覆盖正式/反向/SLO/十条有效trace/两组闭环；`reports/idea2_d10_dynamic_sm_20261024/`包含原始指标派生CSV、配对区间、SLO及新旧闭环对照。设备释放由调度Host确认，本审计核验两闭环server退出0与shutdown成功，不另行探卡。 |

## 已解决的执行阻碍

1. **宿主/容器 PID 不能直接比较。** `nvidia-smi`给宿主PID，`subprocess.pid`给容器PID。闭环脚本已改为单GPU独占、客户端前后零G/唯一已登记C、单客户端期间唯一G的归属证据；同时记录两种PID，清理仅操作自己的subprocess，不向观测到的宿主PID发信号。目标GPU始终看不到G仍会报EGL映射错误。
2. **方案不录全量视频，原客户端固定录制。** results包装层已使用worker范围的DiscardVideoWriter，只丢弃编码/视频输出，保留capture、观测、inputs、actions、EEF/telemetry。结果显式video=null。CPU检查覆盖这一行为，没有修改runtime源文件。

## 必须保留的排除与解释边界

- `d10_whole_formal`、`d10_split_full_formal`因冷加载期间源码变化已标记`excluded_source_drift`。旧队列文件仍可能显示历史passed，选表应服从各result当前排除状态；使用对应`*_formal_stable`替代。
- `d10_encoder64_vlmfull_profile`被用户暂停中断，不能作有效profile；使用`d10_encoder64_vlmfull_profile_resumed` / `..._v2.nsys-rep`。
- `d10_slo_62ms_repeat`保持**failed**，0测量/0组：启动空闲守卫看到43%利用率而拒绝；没有放宽隔离条件，使用`d10_slo_62ms_repeat_resumed`的400条有效结果。
- `trace_audit_historical_check.json`为**passed=false**：历史D2固定16 trace的8个尾段均早于两个生产者结束（约−1.67至−1.80ms）。这是历史数据不满足当前full-Prefix尾段门禁，不是D10失败，也不等于D2数值错误。
- `prepare_cpu_ms`包含deepcopy、输入变换、H2D与预处理的Host墙钟。没有独立prepare NVTX时，不能把整条trace所有memcpy/synchronize时间都归到输入准备，也不能把它叫纯CPU算术开销。
- Kernel busy为实际CUPTI区间并集，交集是真重叠；不是SM占用率。Attention区域含RoPE/reshape/mask等。Profile时间不进入正式端到端排名。
- 逐轮诊断使用same-frame KV；旧KV数值参考另用相同旧KV＋当前state/noise。Graph逐轮输入由noise和前轮Euler输出推得，不冒充独立输入hook。
- 本次闭环含旧KV O0/O1与新KV fresh_serial；当前帧combined仍为观测重放，不能宣称已验证其闭环部署收益。8例每模式不能证明质量等价。fresh与old跨GPU，仅比较质量及匹配bootstrap数值，不比较绝对时延。
- 闭环没有保存原始noise张量字节，证据是相同seed及构造协议；checkpoint为同路径和文件metadata，没有新增全权重SHA证明。fresh顶层execution.protocol记录启动health的默认async_reuse，8份episode protocol及239条实测infer均已独立核验为fresh_serial，不能误用启动health代表执行模式。
- SLO实测行没有逐请求KV digest；624条same-source smoke有实际digest与oracle/槽地址对应校验。周期到达重放的各臂可能选到不同source，不能当逐帧跨臂配对。

## 最终核验与解释约束

- 闭环独立重哈希旧路线856个input＋856个chunk、fresh239个input＋239个chunk。旧路线4,274条实际动作与chunk消费相符、2,145对EEF行逐位相同；fresh1,181条动作相符、1,189条EEF有限。全部24例无mp4，video=null；原观察和动作规则保留。
- 旧路线65个源码SHA一致，fresh66个SHA一致，其中共同65个相同；两执行记录均passed、source_unchanged、shutdown成功、server退出0。每例reset和末尾drain完成。父调度已确认GPU释放；本审计未对其他会话设备作探测。
- 当前帧六条trace的归属/资源审计通过；旧KV与新控制四条trace额外作独立SQLite逐区间复算。四条合计464,688个Kernel行与源表完全一致，不以NVTX重叠替代Kernel交集。
- 旧KV O0→O1的约1.59×包含接口提前返回和后续Host准备隐藏。相同提前返回控制后，连续吞吐仅增加约8.2–8.5%，单请求动作约慢4–5ms；不能把两种速度比分母混用。
- 62ms SLO下Encoder正式轮满足2%预算，反向轮须5%；不能声称2%稳定。33ms下Encoder/combined连10%预算也不满足；71ms主要受输入到达率限制。预算和输入语义分开报告。
- fresh 8/8与旧KV各4/8构成明确的小样本风险信号；初始条件一致、后续输入因策略轨迹分化，不能强求后续新旧输入相同，也不能从两种旧KV调度逐位一致推导旧KV语义无损。

没有待补的必需实验或未解决产物异常。负收益候选保持筛选身份，历史排除/失败与任务失败均保留。最终性能解释和建议见[主报告](../../../docs/experiments/idea2/idea2_d10_dynamic_sm_report_20261024.md)；本文件及JSON负责完成性与证据边界，不替代报告。
