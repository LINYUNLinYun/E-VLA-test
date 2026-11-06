# EASE fast path 后续实验：原始文献与有限方案

**整理日期：2026-11-04。**本备忘录只用于规划后续探索，不修改当前冻结实验、队列、推理代码或模型。当前上下文来自主实验负责人：FLASH 跨 suite 的 400 个独立复验案例已完成；VLM 的 600 个独立复验案例中，Jacobian 与 action-MSE 各成功 170/200，Full 成功 182/200，尚未复现 suite 间互补；720-case 局部 mask 搜索已启动，Jac anchor 已有实际运行。该局部搜索须按原 manifest 继续；本文的分支/子模块建议只是后续候选机制。

**当前本地 mask 的实际作用范围（只读核对代码与 manifest 后补充）：**本轮与前一轮 V12 mask 都作用于观察前缀的 18 层 VLM transformer。例如 Jac mask 是 `[2,3,5,14,16,17]`，action-MSE mask 是 `[2,3,5,11,16,17]`。`TritonBackend.prepare` 先无 mask 地运行视觉编码，再把 mask 传给前缀 `transformer_encoder`；前缀序列含图像 tokens 和语言 prompt tokens，不含 action/state suffix tokens。被 mask 的层仍计算该层 K/V（KV-only shell），跳过 Q、attention 输出与 FFN 的 hidden-state 更新，随后用本路径此前 Full refresh 保存的残差作补偿/外推。mask 不作用于图像编码器的 27 层，也不作用于 AE/action expert 的 18 层。`sample` 不接收 mask；AE 每个 flow step 都完整执行 18 层，Fast 的 AE 使用 D2、Full refresh 的 AE 使用 D10。整个 Fast/Full 对照同时包含 VLM prefix 更新/补偿方式差异与 D2/D10 步数差异。周期配置是一次 Full-D10 refresh 后最多两次 Fast-D2。

这个合同由本地 [VLM backend](/root/nfs/embodied/benchmark/preparations/ease_insights_20261103/vlm/backend.py:8)、[VLM policy](/root/nfs/embodied/benchmark/preparations/ease_insights_20261103/vlm/policy.py:33)、[基础 prepare/sample](/root/nfs/embodied/benchmark/preparations/idea1_v3/policy.py:56) 和 [Pi0 adapter](/root/nfs/embodied/benchmark/preparations/idea1_v2_2/pi0_adapter.py:95) 的调用链确认。VLM confirmation manifest `0fa449c833c888abf93ea82e5ccd203350ea0528a60e9a0c8683addfdae3d6da` 对应的 backend/policy/pi0_infer/adapter SHA256 分别为 `33cbc9bd…561c6`、`91312c72…3845`、`c554399a…56b8`、`5048a703…a623`；当前 720-case local-search manifest 为 `032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9`，并明确复用该 backend/policy。实验实际运行以这些本地 manifest 和文件哈希为准。下文引用的 upstream OpenPI 代码是架构参考，不代表它就是当前 Triton runtime。

文献只提供可检验的动机。它们没有证明 π0/VLA 内部已有互补 fast path，也不能把语言模型或图文问答的结果外推成机器人操控结论。

## 文献证据

### 1. π0 的分支结构与重复计算位置

[π0 原论文](https://www.physicalintelligence.company/download/pi0.pdf)把图像和指令送入预训练 VLM，把状态及 noisy action chunk 送入较小的 action expert；两组权重通过 self-attention 交互。论文给出的 Gemma VLM 深度为 18，action expert 宽度较小；π0 每次动作预测使用 50 个 action tokens 和 10 步 flow matching。论文的 RTX 4090 计时把图像编码、观察前缀和 10 次 action forward 分开列出，分别约 14、32、27 ms。

[Physical Intelligence 官方 OpenPI 实现的固定快照](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)可以核对架构：[pi0.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)构造 SigLIP So400m/14 图像编码器与 Gemma VLM/action expert；[siglip.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/siglip.py)将 So400m 深度设为 27，并把每层写成 self-attention 与 MLP 两个残差子块；[gemma.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py)把 Gemma 2B 与 300M expert 都设为 18 层。该快照不同于本地转换后的 Triton 推理实现；固定 VLM mask 也不是在这段 upstream 代码中执行。

**对本实验的推论：**同样“跳过一层”不是同一预算。SigLIP 的层在一次图像编码中执行；VLM prefix 会生成供动作后缀使用的 K/V；AE action suffix 会随 flow integration 重复执行。将预算从这些路径中移走，可能产生比只交换一个 VLM prefix 层索引更清楚的路径对照。预算应按实际调用次数和实际耗时或 FLOPs 匹配，不能只匹配跳过层数。论文中的计时只适用于其硬件和实现，不能当作本地速度预测。

### 2. 单层影响可能弱，也可能非平滑

[ShortGPT](https://arxiv.org/abs/2403.03853)用相邻层 hidden state 的变化定义 Block Influence，并在 Llama 2、Baichuan2 等语言模型上观察到层重要性不均匀：不少层可移除而影响较小；随着删层增加，某个位置又可能出现明显性能下跌。论文还报告 attention head 的冗余模式随模型而变，并没有简单的统一顺序。

**对本实验的推论：**若两个同预算 mask 只差一个层，且该层处在冗余区，两个路径可能几乎等价；若交换恰好涉及关键层，差异也可能是普遍损伤，而不是“路径 A 擅长空间、路径 B 擅长物体”这样的互补。这个结论是由语言模型上的层影响分布推到 π0 的待验证假设，不是 ShortGPT 在机器人任务上的结果。

### 3. 视觉通路可有不同于文本通路的冗余

[FastV](https://arxiv.org/abs/2403.06764)在 LLaVA、QwenVL-Chat 和 Video-LLaVA 中报告：深层对视觉 tokens 的注意力存在低效现象；其免训练方法在指定层根据注意力给视觉 tokens 排序并裁剪，在多种图像/视频理解任务上报告了约 45% FLOPs 降低且整体任务分数没有下降。方法调节的是视觉 token 数量，不是删除 transformer 层。

**对本实验的推论：**可把视觉编码、图文联合前缀和 action suffix 当作不同候选预算池来测；也可以把“裁剪视觉 token”作为一种不同机制的后续对照。但 FastV 不证明这些机制适用于 Pi0，也不证明视觉/语言路径会在操控任务类别上互补。

### 4. 任务专用剪枝可免训练，但证据边界很窄

[Task-specific Compression for Multi-task Language Models using Attribution-based Pruning](https://arxiv.org/abs/2205.04157)在多任务 NLU 上，用 attribution 找出某个任务的重要神经元，再免训练地为该任务剪枝。论文报告了六个数据集，并包含 unseen-domain 评估。

**对本实验的推论：**已知任务指令或 suite 标签可以作为一个事先确定的静态路由信号，值得测量“按任务类型选择 fast path 是否胜过一个固定路径”。但该论文剪的是神经元而非 Pi0 层，也没有证明不规则剪枝一定带来实际推理加速；更不能拿发现集的最佳 mask 直接当成验证证据。

## 为什么只交换一层不容易得到稳健偏好

第一，单层差异的作用量没有保证。很多层可能是冗余的，交换它们几乎不改变输出；少数关键层则可能让所有任务一起变差。两种情况都不自然地产生类别互补。把 layer index 当作功能标签，容易过度解读一次小样本差异。

第二，本地 mask 仅跳过图像与语言组成的 VLM observation prefix 中的 hidden-state 更新；图像编码器本身仍完整运行，action expert 也仍全层运行。因 prefix block 对整段图像与语言 tokens 共同做更新，换一层会同时扰动这两类前缀表示；但这不等于剪掉 AE/action token 的 transformer 层。现有路径对照因此测试的是「prefix mask + D2/D10 flow-step 数」的组合，不能据此推断 AE 路径是否互补。更细的预算对照也必须把 KV-only 计算和残差补偿算进去。

第三，任务类别偏好、flow timestep 偏好和闭环任务 phase 偏好是三个不同假设。已知 flow timestep 可用于静态的积分步路由；抓取/放置等环境阶段则需要可靠的阶段定义与切换开销审计。发现“Spatial suite 选择 A、Object suite 选择 B”不能说明同一 episode 中应在线切换 A/B。

因此，当前 720 个单交换候选是一个有边界的局部搜索；如果它没有给出稳定复现，不应得出“模型内没有互补性”的普遍结论，也不应扩大到无约束搜索。可把后续范围扩展到不同计算通路或子模块，但须单独冻结搜索空间和预算。

## 三个供后续评估与预注册的实验提案

以下方案都等当前冻结队列与计划内独立确认完成后再评估，不动当前 mask 搜索或确认集。它们是新机制的实验提案，不是已批准的 runtime revision 或已注册的实验。下方的预算比例、样本量与成功率阈值均为提案值，不能当成既有 manifest 的 preregistration。若之后执行，需为新机制、实现、compute budget 和阈值单独冻结 manifest。建议共用控制为固定 checkpoint、预处理/归一化、action horizon、flow 步数、随机噪声种子规则、运行时版本和环境参数；同一 task/init 在各路径间配对。每个方案只用发现集挑选候选，确认集不再调 mask、类别定义或阈值。

### 方案 A：预算放在视觉编码、联合 observation prefix 还是 action expert

**新机制，当前代码未执行。**比较三种静态路径：现有 VLM observation-prefix mask；新增 SigLIP image-encoder layer mask；新增 action-expert suffix layer mask，另保留 Full。当前 VLM mask 同时作用于图像与语言组成的 prefix，不能被描述成 language-only mask；要再拆分 prefix 内视觉 token 与 prompt token 的更新，需要单独设计 token-group-specific 机制。后两种新 mask 都需要独立改造/冻结 runtime。按实际执行次数和 KV shell/残差补偿计量各分支成本，调整 mask 数使预算相配（约 90% reference FLOPs 只是候选设定，提速仍须本机测量）。不改变 flow 步数，不重新训练权重。

若未来启动，发现集在四类任务（Spatial、Object、Goal、Long）各取 10 个 task、每 task 2 个初态；只从 3 条路径中选最多 2 条候选。冻结后，在新初态上确认：每类至少 10 个 task × 4 个未见初态。主指标是按 task 聚类的配对成功率差；同时报告每个已执行 action 的 native-policy wall time、端到端 RPC 时间、Full 成功率差，以及可用时的接受率/Full 回退比例。相对 Full 的 −10 pp 非劣界只是提案值，不是当前已登记阈值。

**支持互补的候选标准：**若未来正式冻结该方案，预注册的两类任务上，路径 A−B 的成功率差需方向相反，两个方向各至少 10 个百分点，按 task 聚类的 95% 区间均排除 0；每条路径还需满足当时预先设定的 Full 非劣与实际延迟门槛。10 pp 及相关门槛目前都只是提案。没有反向优势、只有 Full 差距、或 latency 没下降，都不能支持“按任务类型选路径有收益”。

### 方案 B：同一层里的 attention 与 MLP 分配不同预算

**新机制，当前 mask 不是 sub-layer bypass。**在固定 VLM-prefix 层位置上比较两种无训练 residual bypass：路径 A 跳过 attention 子块，路径 B 跳过 MLP 子块。底层 `pi0_infer.transformer_encoder` 有 attention/FFN 子块接口，但当前 EASE `Backend.encoder` 的 mask 是 whole-block KV-only + residual compensation，未使用这些接口；将来需以独立 runtime revision 暴露并审计它们。按 profiler 的执行 FLOPs 或执行时间匹配预算（约 10% 也只是提案值），不要按子块个数匹配。发现空间暂限 12 个层区段组合，发现集最多冻结一对 A/B。

使用与方案 A 相同的发现/确认拆分、四类 suite 和配对初态。唯一主要检验是预先指定的两类任务之间的 `path × category` 交互；同样报告成功率、实际每 action 延迟和 Full 对照。若只有 attention/MLP 的总体差，而类别交互在独立确认中消失，就支持“子模块选择影响总体能力”，不支持“不同类别需要不同路径”。如果某一路径虽省了 FLOPs 却没有实测提速，也不算实用 fast path。

### 方案 C：用已知任务类型在已确认的 prefix masks 间选路径

**当前 mask 机制可以支撑的静态路由提案。**只有在当前 720 搜索或其他独立实验已给出一对可复现的 prefix masks 后才运行。发现集预先指定一个类别对并冻结映射（例如 `Spatial → A, Object → B`）；另两类只作外部检验。确认中，对每个 task/init 跑 Full、固定 A、固定 B 与映射路由，共 4 条条件；每类至少 10 个 task × 4 个新初态。路由只读测试前已知的 task/suite ID，不看图像、状态、成功历史或 episode 阶段。主指标是路由相对“发现集选出的最佳单一路径”的配对成功率差，按 task 聚类给 95% CI；另报告总延迟、路由开销和每类结果。5 pp 成功率门槛是提案值。

**支持路由的标准：**确认集路由相对最佳单一路径的平均成功率提升至少 5 个百分点，95% task-cluster bootstrap 区间下界大于 0，且 per-case 计算预算与固定路径匹配。若映射不改善确认集，或 A 在所有类别都不差于 B，则目前应选择固定路径；不能声称有自适应收益。即便此方案通过，它也只支持“执行前按任务类型选固定路径”，不支持 episode 内动态切换。

## phase 切换应放在之后

如果静态路径互补已经被独立确认，再单独测已知 flow timestep 的路径日程：固定总 flow 步数与总执行预算，比较同一路径贯穿所有步、A/B 依预注册的早/晚 timestep 分配、以及反向日程。它不需要 phase detector，因此能先隔离“去噪阶段需要不同路径”这一假设。后续若要按抓取/运输/放置等环境阶段切换，则需在新实验中冻结阶段标注规则、selector、错误代价和切换延迟，并与“整段 episode 固定最佳路径”比较。无这些证据时，不应把任务类别互补写成在线自适应能力。

## 共用统计与报告约束

- 先登记唯一主要类别对、最小有用差异（以上暂取 10 pp）、Full 非劣界（暂取 −10 pp）、确认案例和分析代码；这些阈值是本实验的操作定义，不是文献的通用标准。
- 发现与确认使用不同初态；以 task 为聚类单位作 paired bootstrap，报告每个 task 的差异、类别交互和区间。若样本量不够区分预注册差异，结论写“证据不足”，不从多个类别中事后挑最显著一对。
- 同步报告每 action 模型 native wall time（包含请求被拒、未产生动作的请求）、完整 RPC wall time、成功率及可用时的接受/回退率。GPU 有并发时保留资源观测；不要把共享卡计时写成独占吞吐。
- 结论按证据层级表述：有稳定反向偏好 → 支持这对路径具有类别互补；路由在独立确认胜过最佳固定路径 → 支持任务级选择有收益；这两者都不自动支持同一 episode 的在线 phase 切换。
