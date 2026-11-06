# π0.5 可视化探索实验执行记录

日期：2026-10-26。对应方案：[idea1_可视化探索性实验.md](../../../idea1_可视化探索性实验.md)。

**本轮采集、特征提取和完整分析已完成。** 使用 GPU 1、2 两个持续模型服务，结束后服务和调度器均已退出。结果见[最终报告](../../../results/idea1_visualization/run_20261026/analysis/report_zh.md)和[交互 PCA](../../../results/idea1_visualization/run_20261026/analysis/pca_explorer.html)，完整输出核验见 [final_audit.json](../../../results/idea1_visualization/run_20261026/final_audit.json)。运行清单见 [manifest.json](../../../results/idea1_visualization/run_20261026/manifest.json)。

**轨迹采集已完成 120/120 条，成功 117 条、有效失败 3 条，均保留。** 其中 Spatial 30/30、Object 29/30、Goal 29/30、LIBERO-10 29/30。按既定规则共选出 **2326 个不同真实推理请求，全部完成全层激活提取和分析**。证据：[采集完成记录](../../../results/idea1_visualization/run_20261026/collection_complete.json)／[特征完成记录](../../../results/idea1_visualization/run_20261026/features_complete.json)／[轨迹汇总](../../../results/idea1_visualization/run_20261026/rollout_summary.json)。这些成功数描述本轮探索采集，不作为官方 50-init 基准成绩。

全量采集链路检查已通过：120 条轨迹共记录 3784 次推理；每个所选请求的 RGB／状态／指令输入哈希、显式噪声哈希、当前 EEF 状态和已执行动作前缀均与原始记录一致。证据：[collection_audit.json](../../../results/idea1_visualization/run_20261026/collection_audit.json)。本检查不代表全层激活或最终分析已经完成。

**全部 2326 份特征归档已通过核验，pending 为 0**：逐 NPY header 检查覆盖所有组件名称、层数／去噪步数／shape／dtype、有效 token 布局，并核对原始噪声、最终动作、样本和服务来源。原始／解码动作与对应 rollout 记录的实际最大差均为 0，预设 `atol=rtol=1e-5` 未调整。证据：[feature_audit.json](../../../results/idea1_visualization/run_20261026/feature_audit.json)。归档结构对应约 712.5 GB 数据；大型 hidden payload 未重读或全量重哈希，阶段标签准确率和分析正确性不属于此项核验。

## 本轮协议

- π0.5 官方 `pi05_libero` JAX checkpoint，完整 Vision／VLM／AE，无跳层、无跨请求 KV 复用。
- D10、H10、R5；四套件各 10 个 task，官方 init0、init1、init2，共 120 条完整 rollout。环境 seed7、10 步 settling、20Hz，套件控制步上限依次为 220／280／300／520。
- 每个推理请求保存实际观测、指令、显式初始噪声、原始与解码动作；逐控制步保存末端、物体、接触／抓取和目标谓词 trace。有效失败正常保留，技术中断独立存档。
- 每条轨迹最多 20 个不同请求：12 个均匀点＋最多 8 个事件附近补点；不足时按方案补齐或保留全部实际请求。预期约 2400 个观测，实际数量可能因短轨迹少于此值。
- 同一 task 的采集与特征重放绑定同一持续模型服务：偶数 task 在 GPU1，奇数 task 在 GPU2；每个套件保持此分片方式。

## 启动核验与已确认边界

1. trace 选点和 phase 标签的 7 项检查通过，包括短轨迹、事件前后实际请求、缺失证据、夹爪命令不冒充抓取，以及多子目标和关节操作情形。
2. 小型真实 Gemma 模块的 prefix、KV cache、AE suffix 与 gated residual 检查通过。
3. 四套件各一条固定试采轨迹均完成；同服务直接原生采样与官方 `Policy.infer` 的 float32 控制动作一致。
4. 四套件首尾共 8 个真实请求的完整激活重放通过预设 `atol=rtol=1e-5` 核验；这 8 个请求的原始／解码动作实际最大差均为 0。证据：[pilot_gate.json](../../../results/idea1_visualization/run_20261026/pilot_gate.json)。正式提取仍会逐请求核对原始／解码动作。
5. 初始插桩的去噪标量提升规则曾造成约 0.002 的偏差，已保留原生动态 `num_steps` 的 JAX 类型规则并通过上述核验，未放宽阈值。
6. 本机原生策略跨卡／重启也观察到 BF16 数值差，因此保留此前试采为历史记录，当前正式轨迹与提取在对应同一服务内完成。旧试采位于运行目录的 `pilot_previous_service/`，不计入本轮 120 条或特征分析。
7. 已修正控制进程带有 CUDA mask 时的 EGL 枚举问题：在不带 mask 的子进程查询物理映射，再为环境绑定实际设备。已完成轨迹保留；技术中断轨迹独立存档后续跑。

## phase 与选点

phase 是 trace 推断的离线规则标签，保留证据与 `unknown`，不作为阶段真值。抽屉等任务可用目标区域与末端共运动作为“操作”的线索；多物体任务优先围绕当前抓取／接触对象或未完成子目标判断，避免用已放置物体的位置给后续操作误标。

2326 个所选请求的标签为：initial 345、approach 771、抓取／操作 322、搬运 574、目标对齐 173、contact 91、释放／最终操作 13、unknown 37。释放类支持量较少，分析会报告训练／留出分布和类别覆盖；这些是选点后的计数，不是自然阶段时长比例，规则标签的可标注比例也不代表正确率。

每条轨迹保存 `phase_trace.jsonl` 和 `selection.json`。选点来源区分 `uniform`、`event`、`uniform_fill`，重合时保留多种来源。固定 task0/init0 的四段视频用于阶段抽查；[试采画面抽查](../../../results/idea1_visualization/run_20261026/pilot_phase_review.png) 的初始标签为规则第一版，后续标签以各轨迹的 `phase_trace.jsonl` 为准。

## 数据与分析输出

每个所选请求保存：

- `activations.npz`：Vision 各层、VLM 各层 attention／FFN 状态和实际残差分支更新、AE 全部去噪步与层、flow 和最终动作；原生 BF16 以 uint16 位模式无损保存，语义 dtype 记录在 `metadata.json`。
- `pooled.npz`：用于分析的 float32 pooled 特征，两路有效相机分开、VLM 视觉／有效 prompt token 分开；排除 padding。已核对本轮官方 `pi05_libero` 的实际配置为 `discrete_state_input=False`，语言池为指令 prompt token；不套用 π0.5 通用默认的离散状态输入假设。
- `residual_stats.npz`：各 token 组的逐层更新相对范数及输入输出余弦。
- `parity.json`、`sample.json`、`metadata.json` 和原子写入的 `complete.json`：数值核验、样本身份、phase／事件证据、GPU 服务归属和实际存储信息。

单个试采观测全层数据约 308 MB；热态提取 GPU 结果约 0.24 秒，单份落盘约 2.7–4.4 秒，另含主机 pooling／统计成本。当前以完整科学输出为主，后续 ETA 按实际双 worker 吞吐更新。

调度顺序为：120 条 rollout → trace 标注与选点 → 双服务特征提取 → PCA／probe／聚类／残差分析。AE 本轮采集但暂不分析。

完整分析已完成：每个 task 的 init0/1 为训练轨迹、init2 为留出轨迹；标准化和 PCA 仅在训练集拟合。产出 1568 行逐层 probe、2016 行条件 probe、784 行同服务对照、1656 个聚类条件、184 组 PCA 投影、68 张静态图和独立交互 HTML。证据：[analysis_complete.json](../../../results/idea1_visualization/run_20261026/analysis/analysis_complete.json)。AE／flow 已采集，本轮未分析。

## 分析检查与固定小样本预览

- 已用固定的 Spatial task0/1、各 init0–2，共 6 条轨迹的 **112 个真实观测**生成[预览报告](../../../results/idea1_visualization/run_20261026/preview_first2_spatial/report_zh.md)和[交互 PCA](../../../results/idea1_visualization/run_20261026/preview_first2_spatial/pca_explorer.html)。训练 76 个观测，留出 36 个观测；留出观测来自两条完整 episode，不能按 36 次独立试验解释。预览不用于调整最终参数或选样。
- 输入层语言特征在每个 task 内最大差为 0，符合本 checkpoint 仅使用指令 prompt 的实际配置。float32 PCA 在零方差轴上的舍入差曾被自动坐标缩放放大，因此 PCA 改为 float64；仅按训练矩阵的标准机器精度秩阈值把未确定零空间坐标置零，保留有效低方差方向。原始特征 probe 不使用该坐标处理。
- 正式输出增加 `conditional_probes.csv`：同 task 内读出 phase、同 phase 内读出 task，同时报告训练／留出样本数、类别覆盖和多数类基线。分类器与 PCA 均保留原定 episode 划分。
- task 与持续服务分片绑定，且实际观察到跨服务 BF16 差异，因此增加 `backend_control_probes.csv`，在同一服务内做每 suite 的 5-task 读出，保留 init2 留出规则。合并的 10-task 读出不作为跨后端语义泛化证明；固定前两个 task 的预览也存在服务绑定，phase 的同 task 对照保持同服务。
- 结构边界：原生采样器使用 VLM 的逐层 prefix KV，并忽略 prefill 的最终 hidden 输出。因此末层 H18 的表征变化不直接等于动作敏感度；这来自原生代码依赖关系，不能作为这批图表的因果发现。
- k-means 同时记录请求 k、实际拟合 cluster 数及原始训练特征的不同向量数；k 超过不同训练向量数时标为不可估计，避免对固定指令输入拟合数值伪簇。
- 选点、分析与 padding／相机 pooling 的 13 项 CPU 检查通过，另有启动时的小型 Gemma 检查。静态预览图已目视检查，交互 HTML 的内嵌脚本通过 Node 语法检查。

## 本轮有界耗时诊断

- 单个真实全层样本的无损 Zstandard 流式压缩逐数组回读一致，但写入耗时 5.01 秒，原 NPZ 为 2.95 秒；本轮保持原 NPZ 格式。证据：[存储 profiling](../../../results/idea1_visualization/run_20261026/diagnostics/storage_probe/profile.json)。
- 主机统计存在重复 float32 转换和重复范数计算，已实现缓存。两份不同真实输入的旧／新统计耗时分别为 5.88→3.68 秒、6.61→3.83 秒；pooled 和残差统计实际最大差均为 0。统计预设容差为 `atol=rtol=1e-12`，未在失败后调宽。证据：[缓存 profiling](../../../results/idea1_visualization/run_20261026/diagnostics/pool_cache_profile.json)。
- 缓存版已在请求完成边界切入，两个原生模型服务 PID 保持一致，逐请求原始／解码动作核验继续执行；新元数据标记 `pooling_impl=cpu_cached_v2`，并分别记录 GPU 提取、主机统计、打包、保存及总时长。上线记录见 [host_pooling_adoption.json](../../../results/idea1_visualization/run_20261026/host_pooling_adoption.json)。小批统计提速不直接当作整轮吞吐提速，后续以实际进度计算 ETA。

## 首轮观察与最终核验

- VLM 视觉池 H18 的 task 线性读出仍为 97%–100%，同服务 5-task 对照为 95%–100%，而 k=10 聚类的 task ARI 接近 0；不能把聚类混合解释为任务信息消失。
- 同 task 内，语言池 H0 的 phase 读出等于多数类基线；H18 的四套件平均分为 60.8%／59.7%／72.6%／54.9%。这描述规则阶段标签的可读性，不是动作重要性。
- 阶段读出没有统一的单调上升规律：部分中层高于末层，suite 和 token 池之间存在差异。均匀子集也支持输入层到末层的阶段读出提升，但样本预算不同，不能据此估计选点的因果效果。

详细数值与限制见[最终报告](../../../results/idea1_visualization/run_20261026/analysis/report_zh.md)和 [findings.json](../../../results/idea1_visualization/run_20261026/analysis/findings.json)。表格支持量和 episode 划分已逐行对照固定 selection；184 组投影和交互页面内嵌数据一致，68 张 PNG 可解码，交互脚本通过 Node 语法检查。代表 PCA、probe、阶段热图和残差图已目视检查；未声称在浏览器中验证交互运行。

静态图仅在完成分析后补全阶段图例、区分阶段热图的零差值和缺失值颜色，从冻结坐标／CSV 重绘，不重新拟合 PCA、分类器或聚类。原分析源码哈希保留在 `analysis_complete.json`，最终源码清单在 `source_identity_final.json`，重绘记录在 `pca_render_complete.json` 和 `conditional_render_complete.json`。

## 执行入口

2026-10-29 增加具体任务对的可分性检验：[任务对交互散点](../../../results/idea1_visualization/run_20261026/analysis/task_pair_separability_20261029/task_pair_explorer.html)／[五页任务对报告 PDF](../../../results/idea1_visualization/run_20261026/analysis/task_pair_separability_20261029/task_pair_separability_report_zh.pdf)。用户指定 Spatial task0 vs task5、Spatial task0 vs Goal task0，另补 Spatial task0 vs task4 同服务对照。每对仅取对应任务的真实 pooled 特征，PCA、k=2、簇到任务的颜色对应及线性分类器均在 init0/1 拟合，init2 留出。在 raw VLM 语言 H12，Spatial0/5 三个 k-means 种子分别正确划分 34/34、33/34、34/34 留出点；Spatial0/Goal0 为三次 36/36，L2 后两对均三次全部正确。同服务 Spatial0/4 对照也三次 36/36。该结果是两任务单独比较，不等于在全部40个task中各自独占一簇；每个task仅一条留出轨迹。Spatial0/5 原采集服务不同，报告明确此混淆；语言 H0 本已含不同指令，不能把 H12 可分直接解释为新形成任务语义。全部168个配置、504个聚类、same-phase 对照和数据／指标核验见该目录的 CSV 与 [validation.json](../../../results/idea1_visualization/run_20261026/analysis/task_pair_separability_20261029/validation.json)。

2026-10-28 增加四套件**真实 hidden state 联合散点与 k-means**：[交互散点入口](../../../results/idea1_visualization/run_20261026/analysis/joint_suite_scatter_20261028/joint_suite_explorer.html)／[六页散点图 PDF](../../../results/idea1_visualization/run_20261026/analysis/joint_suite_scatter_20261028/joint_suite_scatter_report_zh.pdf)。四个 suite 的 2326 个原始 pooled 特征在每个模块／层联合拟合 PCA，按 suite 与 k-means 簇并排着色；覆盖 28 个层位置、原始／L2 两种预处理、k=2／4／8、seed0/1/2，共 504 次聚类。PCA／中心仅在 init0/1 的 1556 点拟合，770 个 init2 点留出评价。每个 k 改变时散点坐标保持不变，交互页面提供每簇的 suite 组成。保存坐标、中心、簇标签和 ARI／NMI 表；[核验记录](../../../results/idea1_visualization/run_20261026/analysis/joint_suite_scatter_20261028/validation.json)覆盖中心分配、留出指标和交互数据／逻辑对应。

2026-10-28 增加四套件模块／层偏好补充分析：基于原固定 probe 表，比较 24 个 H>0 hidden-state 位置的模块比较优势和模块内部深度倾向，并分别按 task／phase 指纹对 suite 和层聚类。结果见 [6 页专题 PDF](../../../results/idea1_visualization/run_20261026/analysis/suite_preferences_20261028/suite_preferences_report_zh.pdf)、[模块偏好图](../../../results/idea1_visualization/run_20261026/analysis/suite_preferences_20261028/module_suite_preferences.png)、[层深度曲线](../../../results/idea1_visualization/run_20261026/analysis/suite_preferences_20261028/phase_depth_profiles.png)和[聚类及重采样记录](../../../results/idea1_visualization/run_20261026/analysis/suite_preferences_20261028/summary.json)。task 首次合并 Object／LIBERO-10，在逐一去掉一个模块和改变层权重的检查中保持；phase 首次合并只有约 38% task-bootstrap 频率，112 个偏好单元的同时区间均跨零，暂作候选倾向。该补充未重跑模型、未重拟合 probe，不把可读出偏好当作动作因果重要性。

代码：[preparations/idea1_visualization/](../../../preparations/idea1_visualization/README.md)。后台调度器及两个模型服务的 PID、命令和日志保存在运行目录。`controller.py` 的阶段错误会写入 `status.json`，已有效完成的任务不重跑；技术中断目录保留后可恢复未完成任务。
