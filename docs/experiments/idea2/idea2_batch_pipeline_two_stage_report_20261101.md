# π0.5 Batch 流水线两阶段实验报告

状态：**两阶段测量完成**。34个有效正式性能进程、4个连续验证进程、6份独立nsys trace均已正常退出。原失败和负收益保留；最终数据与交付核验见文末链接。本轮后续代码准备没有另提交GPU实验。

## 当前能确认的主线

增加 batch 确实提高了录制输入重放的总吞吐，但幅度有限，单批延迟明显增加。普通流水线相对同 B 串行的额外收益，从 B1 的约11.5%下降到 B4 的约3.3%。VLM 随 B 增大的成本接近线性，AE 的串行成本增长较慢，可隐藏的相对份额随之下降。

SM 配额需要随 B 调整。预准备输入时，B2 的 AE16 比普通双流更慢，AE24 可以恢复吞吐；B4 的 AE20/24 比 AE16 有利。20 相对24的约1.1%优势已经反向进程顺序复测，方向一致，但不是全局最优证明。

实际 prepare 暴露了另一个关键问题：旧 Green 路径中，下一批上传出现等待当前 AE 的长同步，导致 AE×下一批 Encoder 的真实 Kernel 重叠消失。仅把 prepare 放到 Prefix stream 未解决问题；独立输入 stream 已完成正式对照，恢复了真正的 AE×下一批 Encoder 重叠。

吞吐收益明显大于单输入延迟收益。B1 关闭 lookahead 后，输入开始准备到 CPU 动作返回为95.51→91.28ms，下降约4.4%；允许 lookahead 的58.93ms是稳态服务段。二者计时边界不同，不能把58.93ms直接当作新观测的完整延迟。

B2/B4 没有通过相对独立 B1 的预定数值容差。它们的结果只回答同 batch 的容量/调度问题，不能宣称相对 B1 无损，也不能证明闭环成功率不变。

## 实验范围与计时口径

完整 `pi05_libero`，Gemma2B VLM + 300M Action Expert，D10/H10、action_dim32、原 checkpoint 精度和 FP32 Euler 时间递推；RTX4090。各对照均使用相同的显式 CUDA Graph 配置，预热、捕获和模型加载不计入稳态。没有改权重、结构、去噪步数或采样规则。

每个模型进程只有一个调度 Host。普通模式使用 Encoder、Prefix、AE stream；Green 模式中 Encoder/Prefix 使用同一分区 stream，AE 使用另一个分区 stream。独立输入方案另加一条仅处理原输入准备的普通 stream，没有引入双 Host 发射。

固定 B1/B2/B4 lane，统一选同任务12条录制观测，保留原始记录身份和逐记录噪声 seed。输入交替变化；整批 reset；不支持动态换 lane。该 bank 没有可用的真实相机到达时间，本轮不报告机器人观测年龄或成功率。录制replay中允许下一批输入已可用；单机器人同步控制若还没有下一观测，跨批Encoder窗口不能直接兑现，必须在真实输入到达和动作执行协议下重新验证。

- `prepared`：输入 bank 预准备；计时仍包括静态输入 staging、Graph replay、CPU 动作下载与解码。
- `online`：请求序列内执行原 prepare，包含 CPU transform、上传和 GPU 观察预处理。这里仍是录制输入重放，不是实时相机或动作执行模拟。
- 吞吐 = `B × 完成请求批次数 / 整段墙钟`，包括组末所有分支 drain。每 lane KV 更新频率为批次请求频率，不能乘 B。
- 允许 lookahead 的 CPU→CPU 字段从本轮 admission 计时，可能包含下一输入准备，而当前输入已在上一请求准备；这是服务段时间。新记录另有每份输入自己的 `input_prepare_to_cpu_ms` 与入队时的准备年龄。历史未记录这两个字段的运行保持缺失，不反推。
- `online --disable-lookahead`：在当前请求内准备当前输入，关闭未来 CPU 准备和下一 Encoder；此时报告完整 prepare→CPU 动作延迟。它仍可有本批 VLM/AE 重叠，不能当作跨批 Encoder 重叠实验。批次延迟均不能除以 B 后当作单机器人延迟。
- Event 完成时间从当前 Encoder 的 GPU 开始计算；Encoder 可能已在上个 Host 请求中提前运行，因此 Event 时间与当前 Host 延迟不能直接相减。

全卡串行/普通流水线同进程、同 Graph 交替100对 AB/BA，每 arm 共800请求。Green 单独进程100组、每组8请求，属于跨进程比较。组间 reset、旧KV bootstrap、bank setup在原始groups中单列，吞吐统计组内请求及末尾drain；报告范围为预热后的请求服务时间。首次基础矩阵18格全部完成；初次配额条件共17次有效正式运行、23行模式结果；最终加上输入修复、同卡复测和新输入边界，共34次正式运行、44行模式结果。不同 GPU 的比较在 CSV 中保留 UUID 与标记。

## 第一阶段：合批与普通流水线

下表为预准备输入的吞吐，单位观测/s：

| B | 全卡串行 | 普通流水线 | 普通流水线/同B串行 |
|---|---:|---:|---:|
| 1 |14.855|16.566|1.115×|
| 2 |16.474|17.418|1.057×|
| 4 |18.732|19.354|1.033×|

合批本身的容量收益与流水线额外收益应分开计算。例如 B4 串行相对 B1 串行约1.261×；在这个 B4 基础上，普通流水线再增加约3.3%。不能将两者全部归给并行调度。

同口径的稳态 GPU Event 均值，单位ms：

| B/模式 | Encoder | VLM | AE | CPU提交AE |
|---|---:|---:|---:|---:|
| B1串行 |10.25|30.91|25.53|0.34|
| B1普通流水线 |12.76|33.64|54.80|0.34|
| B4串行 |30.18|131.56|50.88|0.39|
| B4普通流水线 |38.59|138.65|186.64|0.39|

串行时 B 从1增至4，VLM 约增至4.26倍，AE 约增至1.99倍。普通并发后，AE 的区间进一步拉长；正常运行的 Graph 提交只有亚毫秒量级，不能仅凭慢 AE 区间归因为 Python 来不及发射。不同 GPU/CPU 绑定的其他运行提交时长也单独保留于 CSV。

## 第二阶段一：修复生命周期并调整 SM 配额

首次 Green B1 测量在清理时发生 `context is destroyed`，该次没有作为成功结果采用。原因是关闭 Green context 时，调用方 Tensor 和 cuBLAS 工作区仍保留 stream 生命周期标记。

修复采用两阶段关闭：Runtime drain/销毁 Graph 后把 Green 资源交回调用方；调用方释放输入、noise、模型和工作区、清理缓存，最后关闭 Green streams/context，再释放 GPU 租约。合成生命周期 probe 和后续完整模型运行均检查了成功退出及租约释放。

预准备输入的正式配额结果：

| B | AE16吞吐 | 调优配额/吞吐 | 调优CPU→CPU mean | 每lane KV更新Hz |
|---|---:|---:|---:|---:|
|1|18.074|16 /18.074|55.32ms|18.074|
|2|16.973|24 /19.507|102.51ms|9.754|
|4|19.721|20 /20.706|193.17ms|5.176|

B4 AE24 为20.472观测/s、CPU mean194.54ms。AE32短筛选更慢，负结果保留；不能仅通过缩短 AE 自身耗时选择配额。AE 配额增加会减少 Prefix 的 SM，也可能让 AE 更早读取旧 KV。

以 B4 预准备条件为例，新 KV 层使用比例约为：普通流水线88.2%、AE16为65.6%、AE20为58.9%、AE24为52.8%。旧 KV 的请求代龄均为1，但不同配额的逐轮 frontier 并不相同。这些配置的质量不能仅由相同日程数值校验推出。

## 第二阶段二：实际输入准备暴露的等待

原 caller-stream 输入路径，吞吐/CPU→CPU mean：

| B | 串行 | 普通流水线 | Green AE16 | 调优Green |
|---|---:|---:|---:|---:|
|1|13.990 /71.47ms|15.545 /64.32ms|11.238 /88.97ms|同AE16|
|2|15.814 /126.45ms|16.750 /119.39ms|12.850 /155.63ms|AE24：14.469 /138.21ms|
|4|18.300 /218.57ms|18.904 /211.58ms|16.201 /246.89ms|AE20：17.406 /229.80ms|

![基础吞吐对照](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/base_complete_analysis/original_input_comparison.png)

Green 的预准备收益不能直接外推为实际输入路径收益。B4 AE16 在线 trace 中，下一批 prepare 的一次 `cudaStreamSynchronize` 持续184.305ms，到上一批 AE 完成才返回。7个连续窗口的该等待均值181.978ms；AE×下一批 Encoder 的真实 Kernel 交集为0。该等待将下一批 Encoder 的提交推迟了。

先试了 `--prepare-stream prefix`：数值/生命周期通过，但短测未恢复性能。随后增加 `--prepare-stream dedicated`：原 prepare 放到独立输入 stream，每个 Prepared 持有就绪 Event，调用流加入该 Event 依赖；Graph staging 继续受输入就绪和原 Slot/Lease 协议保护。异常清理也先 drain 输入 stream。所有数据转换和模型运算保留原样。

独立 stream 的 B2/AE24短测最初为同B门槛通过、prepare约32ms、吞吐18.1观测/s；它只有2组，单独保留为筛选。之后完成下列100组正式实验。B1/B4初次启动和B2首次正式启动曾被外部占卡守卫拒绝，模型未执行；这些准入失败保留，未作为数值失败或性能结果。

三个有界微基准没有复现实际模型的长等待，因此不支持“普通 Green 工作必然阻塞默认拷贝”“所有 Graph 都串行”及“尾部 D2D 单独导致”的笼统归因。当前直接证据定位到实际模型的输入提交/等待链，尚未将其归结为某个驱动内部机制。

独立输入 stream 的100组正式结果如下。该列计时从本轮入队开始，当前输入可在上一请求准备，属于稳态服务段，不能称为新观测从prepare开始的完整延迟。

| B/配额 | 吞吐，观测/s | 入队→返回mean/P50/P95，ms |
|---|---:|---:|
|B1 /16|16.965|58.93 /54.29 /93.55|
|B2 /24|18.928|105.65 /100.22 /148.45|
|B4 /20|20.034|199.65 /193.15 /252.66|

同配额、同物理卡的 caller 对照已经完成，每个条件均为100组×8请求：

| B/配额 | caller吞吐 | dedicated吞吐 | 吞吐比 | caller服务段mean | dedicated服务段mean |
|---|---:|---:|---:|---:|---:|
|1/16|11.373|16.965|1.492×|87.91ms|58.93ms|
|2/24|14.498|18.928|1.306×|137.93ms|105.65ms|
|4/20|17.196|20.034|1.165×|232.60ms|199.65ms|

B2/B4的独立输入AE16消融分别为16.469/19.055观测/s；同卡调到24/20后为18.928/20.034。这说明恢复输入窗口和调整SM配额是两个独立因素。相对旧 caller 的大幅收益首先是修复该路径的退化，不能全部写成胜过普通全卡路径的收益。

B1全卡online反向顺序复测为串行14.006、普通流水线15.576观测/s，与原14.0/15.5接近。独立输入Green为16.965，相对这组同卡串行/普通流水线的吞吐比分别为1.211×/1.089×；这仍是有lookahead的容量比较。

B4配额反向顺序复测（同一张GPU，20→24）为20.794/20.559观测/s，方向与原24→20测量一致，AE20优势约1.1%。B1/B4的2×100请求连续运行及匹配串行/普通流水线控制均成功退出，数值门槛和生命周期检查通过。

## 补充：新输入延迟与连续序列

新输入对照每批在自己的请求内准备输入，不使用未来输入或下一 Encoder；每个B的全卡串行/普通流水线和Green均在同一张物理卡上完成。Green配额沿用容量实验选出的16/24/20，未额外搜索单输入延迟最优配额。下表为mean/P50/P95，单位ms：

| B | 全卡串行 | 普通流水线 | Green |
|---|---:|---:|---:|
|1|95.51 /94.95 /98.46|93.71 /92.91 /96.31|91.28 /90.80 /95.10|
|2|152.86 /152.40 /155.50|150.45 /149.96 /153.42|144.39 /143.84 /147.19|
|4|254.50 /254.04 /257.50|252.65 /252.17 /255.90|244.34 /244.04 /247.56|

相对同卡全卡串行，Green的完整输入延迟下降分别为4.43%、5.54%、3.99%，对应吞吐比分别为1.046×、1.059×、1.042×。因此本轮支持“有容量收益、单输入延迟改善较小”，不支持“AE完全隐藏后单帧加速约2倍”。

![输入Stream修复与完整输入延迟](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/final_analysis/input_stream_and_fresh_latency.png)

这组边界包含CPU处理、上传、模型、等待、下载与解码，结束于CPU动作可用；没有相机采集、网络、凑批等待或机器人执行。全卡串行仍使用同一Graph优化配置，不能把它叫作未经优化的native JAX baseline。混合KV流水线相对全新KV串行具有已授权的输入语义差异，数值校验使用相同frontier的独立串行oracle。

连续序列独立验证如下，每模式2组×100批请求。它们用于检查较长运行和每组启动/尾部影响，不替代正式100对/100组统计：

| B | 串行吞吐 | 普通流水线吞吐 | 独立输入Green吞吐 | Green服务段mean/P95 |
|---|---:|---:|---:|---:|
|1|14.643|16.532|18.301|54.64 /54.52ms|
|4|18.520|19.251|20.737|192.88 /194.20ms|

吞吐单位为观测/s，B1用AE16，B4用AE20。每个B的连续控制同卡；Group首请求较慢，可使B1的mean略高于P95，不是分位数算错。各模式均保持逐轮frontier和输出归档，稳态重捕获为0。

## nsys：实际重叠与减速

已采集并完成官方完整 replay 范围核对的 B4 trace：

| 条件 | AE×VLM真实交集/请求 | AE×下一Encoder真实交集/窗口 |
|---|---:|---:|
|串行prepared|0|0|
|普通流水线prepared|17.30ms|14.11ms|
|Green20 prepared|138.80ms|22.65ms|
|Green16 online/caller|133.06ms|0|
|B1 Green16 online/dedicated|34.82ms|9.00ms|
|B4 Green20 online/dedicated|138.64ms|22.62ms|

VLM 每条 trace 有8个窗口，下一 Encoder 有7个；这里计算的是实际 Kernel 区间交集，不能把两列相加当作请求节省时间。旧B4 caller trace用AE16，新dedicated trace用AE20，不能把这两份trace的全部差异单独归给输入stream；同配额同卡性能控制与AE16消融另见上表。

当前实现让AE按依赖和frontier连续推进，并没有硬编码“只有第一轮AE与VLM重叠、后9轮全部与Encoder重叠”。真实时间线是多轮AE与VLM交错，尾部再与下一Encoder重叠。无profiler条件下，Green B1的AE平均比下一Encoder晚3.55ms结束，B2/B4则提前约2.80/13.76ms结束；这些是700个有下一Encoder的窗口均值。AE被隐藏得更多，也不会消除VLM本身及其并发减速。

普通流水线的 AE GPU 包围区间约186.68ms，实际 Kernel 区间并集约65.20ms；串行分别约51.85/51.24ms。VLM 并发时约137.38/135.93ms，几乎连续执行。AE 的长间隙与单个 Kernel 变慢共同存在。

相同 stage、Kernel 名、grid/block 且调用次数一致的代表组：Encoder 的 flash attention 均值约15.38→31.55μs，AE 的代表 GEMM 约11.68→15.08μs；VLM 两个主要 GEMM 约1.014×、1.041×。详见 `kernel_slowdown.csv`，不把这些匹配组推广为全部算子的统一减速倍数。

每条 trace 保留逐 Kernel，且8个 Encoder/VLM/AE 完整外层范围均与官方 `nvtx_gpu_proj_trace` 首尾 GPU 操作一致。`prepared` 无计时内 preprocess，`online` 有8个。nsys 节点采集将 CPU Graph launch 放大到约20ms，正式性能表完全来自无 profiler 测量。当前证据尚未区分SM计算、显存带宽和缓存压力各自的份额，不把竞争进一步归结为单一硬件资源。

## 正确性与可采用边界

预定逐 lane、normalized/decoded 容差为 MAE≤1e-6、max≤1e-5，未放宽。B2/B4相对独立B1 eager的 normalized max 分别约0.007587/0.007479，decoded max约0.005682/0.006299；原始失败目录保留。

准备输入/noise/mask/position已逐位核对。Encoder 输出、固定相同KV下的AE velocity以及时间MLP都显示 batch shape 敏感性，支持原始数值路径差异的诊断；没有证据将其归因为 lane 混合或 Graph 改变了同 B 计算。

默认严格模式仍失败即停止。显式诊断模式增加原始同 B eager oracle，并保留 B1 equivalence失败；只有同 B、相同实际 frontier 日程的校验通过后，才运行诊断计时。最终独立核验覆盖44个采用运行、41,256项逐lane误差项；相同日程的 normalized/decoded MAE/max 均为0。这里包含各模式和误差种类的重复核验，独立输入仍只有固定12条，不能当成大规模质量样本。计时内混合输出会归档，但不会对每个计时请求额外跑 eager oracle；数值验证单独进行。

双槽、Pool/Lease/epoch/generation/Event 管理保持，读取结束前不覆盖 Slot；没有第三份完整KV。覆盖输入交替、两槽轮换、整批 reset、相同 Tensor 地址原地更新。CPU完整协议检查93/93通过，CUDA未初始化；此前GPU已启动诊断标记的修正也保留了针对性26项检查。

正式结果稳态重捕获均为0。PyTorch稳态峰值allocated为7236.0～7478.3MiB，reserved为7768～9418MiB；包含模型/Graph建立的setup峰值allocated为7341.8～7917.3MiB。CSV保留每进程两个阶段的计数；paired两arm共享进程计数，不当作分别测得的arm显存。这是PyTorch分配器口径，不是包含全部驱动资源的全进程峰值显存。

“B1严格”表示B1批次计算与基准一致，并且流水线实现匹配相同frontier日程；不表示混合新旧KV等价于完整新KV。最终输出质量仍需真实闭环验证。

吞吐提高不会自动证明闭环质量。当前录制 replay 假定下一批输入可用；真实多机器人需另测到达/排队、每机器人动作时序、相机年龄和成功率。更大 B 的193ms批次延迟不能称为约48ms的单机器人延迟。

## 代码、数据和复现

- [第二阶段原计划](/root/nfs/embodied/benchmark/docs/experiments/idea2/idea2_batch_pipeline_stage2_plan_20261101.md)，以及结果目录内的 `stage2_quota_amendment.md`、`stage2_sm20_rationale.md`、`stage2_input_stream_amendment.md`，均在对应改进测试前记录。
- [最终完整CSV](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/final_analysis/table.csv)、[最终JSON](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/final_analysis/summary.json)、[同卡输入对照](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/final_analysis/input_stream_comparison.csv)、[完整输入延迟](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/final_analysis/fresh_latency.csv)。
- [连续序列CSV](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/final_analysis/continuous_validation.csv)、[真实Kernel交集CSV](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/final_analysis/trace_overlap.csv)、[匹配Kernel减速CSV](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/base_complete_analysis/kernel_slowdown.csv)。
- [独立交付核验](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/final_audit.json)：实际退出、源码前后一致、动作归档、逐lane数值、两槽轮换/原地内容更新、计时分母、完整replay范围与GPU租约释放。
- [逐运行数值CSV](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/final_analysis/numerical_audit.csv)、[交付文件检查](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/delivery_check.json)：44个运行租约均释放，6份下载别名核对通过，原队列均退出。
- [实际复现命令及scp下载](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/reproduction_commands.md)：44条采用运行的实际命令、环境与CPU绑定；历史输出禁止覆盖。
- [实验索引](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/study_catalog.json)：有效正式结果、失败和trace分开；每次运行另有执行回执、实际命令、源码快照、逐请求数据、动作NPZ、租约释放记录。
- [输入适配](/root/nfs/embodied/benchmark/preparations/idea2/batch_backend.py)、[运行时](/root/nfs/embodied/benchmark/preparations/idea2/batch_runtime.py)、[基准入口](/root/nfs/embodied/benchmark/preparations/idea2/batch_bench.py)、[有限尾部队列](/root/nfs/embodied/benchmark/results/idea2/batch_study_20261031/run_stage2_tail.py)。

单次运行通过 `python3 -m results.idea2.batch_study_20261031.run_one` 选择原始 job模板，显式传入空闲物理GPU UUID、索引和新名称；B>1诊断加 `--diagnostic-batch-drift`。输入方案使用 `--prepare-stream caller/prefix/dedicated`，配额用 `--ae-sm-min`；详细实际命令保存在各 `.execution.json`，新结果目录必须不存在。

六份已验证trace的清晰下载名位于 `/root/nfs/embodied/benchmark/nsys/idea2_experiment/batch_study_20261101/`，对应manifest记录原始来源与SHA。完整GPU计算的比较优先看 `pi05_b4_serial_prepared_d10`、`pi05_b4_streams_prepared_d10`、`pi05_b4_green20_prepared_d10`；输入准备问题另看caller/dedicated online两份，注意其配额差异。详细下载命令见上述复现文档。

[后续代码准备方案](/root/nfs/embodied/benchmark/docs/experiments/idea2/idea2_batch_pipeline_preparation_20261101.md) 与 `scripts/idea2_prepare_batch_followup.py` 可生成下一轮命令，保留物理GPU占位符。新清单仅做CPU参数核验，没有自动等卡或执行模型。

## 后续选择

1. 单机器人延迟优先使用B1；先把输入可用时机和完整prepare→action边界固定，再谈SLO，不能用容量表的服务段代替新输入延迟。
2. 多独立环境容量可继续看小batch，但B4相对B1的收益有限，而且单批延迟与每lane更新间隔明显增加；先解决B>1的数值资格与闭环评测，再考虑动态凑批。
3. 当前调度改进优先保留独立输入stream。配额/lead在有证据时单变量微调，结合KV frontier选择；不再假设更大SM隔离或更高重叠率必然更快。
