# π0.5 × RTX 4090：SAE 语义驱动动态跳过探索实验

## 0. 目标与边界

验证如下假设：**从 SAE 中识别出的高置信语义，可以提供有用的计算需求先验，用轻量规则选择不同的跳过模式。**

本实验的“置信度”指 SAE 特征对语义的识别分数，而不是跳过后的成功率保证。不得把任务改写成预测最终动作偏差、雅可比敏感性分析或逐特征闭环消融。

首轮冻结 π0.5；优先复用 Action Atlas 的预训练 SAE；进行离线语义统计、有限的 VLM 前向干预和分批闭环验证。闭环只评测完整策略，不评测每个特征。SAE 只读 hidden state，不把重构结果写回 VLA。

这是实验设计，不是已完成实验；下文层号、阈值、样本数均为探索性默认配置，不是论文验证过的最优参数。

按三个问题分别积累证据：①语义是否对应不同的可跳过性；②SAE 路由是否比固定 mask 或指令路由更有用；③当前实现能否获得净加速。首轮先寻找条件冗余信号并做小批闭环，不要求三项同时达标才允许继续。某个默认层位、阈值或实现失败，只限定该配置的结论。

## 1. 固定配置

| 项目 | 首轮设置 |
|---|---|
| 硬件 | 单张 RTX 4090，24 GB |
| 模型 | 用户当前可运行的 π0.5 LIBERO checkpoint；不换权重 |
| 框架 | 保留当前 OpenPI/PyTorch 环境，不为接入 SAE 重装整套 Action Atlas 环境 |
| 精度 | 保留原有 BF16/FP32 混合精度策略；不新增量化 |
| 推理 batch size | 1 |
| 路由探测点 | VLM L5 的完整层输出，即 attention、FFN 和残差相加之后 |
| 离线语义读出点 | VLM L16 的完整层输出 |
| SAE | 上述两个位置对应的 per-token PaliGemma SAE |
| 候选模块 | VLM L6、L8、L10、L12、L14、L15 的 FFN |
| 每次跳过数量 | 0 或 2 个 FFN；禁止同时跳过相邻的 L14/L15 |
| 部署路径 | FULL + 最多 3 种固定 skip mask |
| 保持不变 | SigLIP、所有 Attention、所有 prefix KV 生成接口、Action Expert、flow-matching 步数、action horizon、replan interval、输入预处理 |

全部层号从 0 开始。L5 是第六个 Transformer block。

选择 FFN 是为了降低第一版工程复杂度，不是断言只有 FFN 有语义分工。OpenPI 的动作生成会读取 VLM 的 prefix KV，因此首轮不把语义实验与 Attention/KV 路径改造混在一起。[5][6]

不要硬编码序列长度、相机数量、action chunk 长度：从现有模型运行时读取。特别不要把先前讨论中的示例长度 978 当作当前配置。

## 2. 接入校验：先保证“读的是同一类表征”

### 2.1 使用的公开资产

数据集标识：`bag100/action-atlas-pi05`。

需要的 SAE 文件：

```text
saes/per_token/paligemma/sae_layer5.safetensors
saes/per_token/paligemma/sae_layer16.safetensors
concepts/concept_features_pi05.json
```

公开说明中的 PaliGemma SAE 输入维度为 2048、字典大小为 16384、TopK 为 64；实际加载时以 metadata 和 tensor shape 为准。公开概念索引包含 feature indices 和概念相关统计，不应把这些数值直接当作校准后的概率。[1]

### 2.2 必须核对

核对当前 checkpoint 与 SAE 的来源、权重转换、tokenizer/图像预处理、hook 位置、归一化统计和 SAE 实现。Action Atlas 的当前 π0.5 adapter 使用 LeRobot checkpoint 接口；架构相同不代表与你的 OpenPI checkpoint 表征完全一致。[2]

优先借用官方 `load_release_sae`、`sae_encode` 的逻辑；正确恢复 `mean/std`、bias、TopK 规则。不要把 decoder 方向投影当作 encoder 输出，也不要额外加一个原实现不存在的 ReLU。[3]

用 32～64 个本地观测做：

- SAE 重构误差、激活稀疏度、特征激活分布的检查；重构仅用于离线诊断，不写回模型。
- 对已知概念检查正负任务中的激活是否仍有区分。
- no-op 路径校验：只读取 hidden state、不跳模块时，与原始路径数值一致；需要比较动作时固定 flow-matching 初始噪声。这仅是实现正确性检查，不参与规则学习。

SAE 必须读取完整层输出，不能把训练在 residual stream 上的 SAE 直接套在 FFN 的独立输出上。[4]

若公开 SAE 在本地完全失去语义区分度，先排查接口与预处理。仍不兼容则停止复用，不沿用原 feature ID 的语义标签。可以另开一个补充阶段，仅训练 L5/L16 两个本地 SAE，再重新做概念关联；这不属于首轮默认工作量。训练时卸载 VLA、逐个 SAE 训练即可，不需要同时驻留全部模型；不可把新 SAE 的相同数字 ID 当成原来的概念。

### 2.3 先复用公开成果，再补本地实验

执行时先完成不依赖空闲 GPU 的公开数据分析，再在当前 checkpoint 上补必要的读出和模块干预。[9][10][11]

| 公开资产 | 具体复用方式 |
|---|---|
| 预训练 SAE 与官方 loader | 直接加载 L5/L16 权重及归一化统计；先用公开激活做编码/重构检查，再检查本地迁移，不重新实现或训练 SAE。 |
| 概念—特征索引 | 整理 PaliGemma、Action Expert 各 18 层的概念分布，直接复用对应层的候选 ID、数量与 max d。当前发布索引未附逐特征 d/频率时，不把 ID 列表顺序当作效应量排名；需要的统计从公开原始结果或小批激活补算。 |
| 原始激活与已有层统计 | 优先复用已计算的层统计；按需下载相同轨迹的 L5/L16 激活，用固定小样本验证 loader 和候选特征，避免全量下载或重新采集。 |
| 网站、特征说明与实验代码 | 用网站及其 processed/descriptions 数据检索候选概念，用官方代码和原始文件做计算；自动生成的文字说明作为线索，不代替实测。 |
| 已有 feature 消融与轨迹索引 | 复用已有任务/层/概念覆盖和可评分结果；缺失 success 的记录保留为缺失。原有 feature 消融提供候选线索，计划中的 FFN 跳过影响仍需实际测量。 |

公开统计与本地测量分别标记来源；同一个数字 feature ID 仍只属于其对应的 SAE。三个 Hugging Face 仓库固定 revision，缓存可复用。实施入口为 `experiments/pi05_sae_semantic/`，首轮产物写入 `results/pi05_sae_semantic/run_20261105/`；运行事实以该目录的 manifest 和检查结果为准。

## 3. 离线样本：不用逐特征采集

优先复用已有 baseline 的 RGB、instruction、proprioception 记录。无需复用旧 hidden state；可以在固定观测上重算。

建议从 LIBERO-Spatial、Object、Goal 中选 24 个任务，按概念覆盖分为：

- 12 个规则发现任务；
- 4 个离线校准任务；
- 8 个预留闭环任务，用于小批试验及后续扩样。

概念划分要尽量让发现集和校准集都有正、负任务。不要随机拆相邻帧后宣称跨任务泛化。

前 16 个任务每任务取 2 条完整轨迹，每条均匀抽取 24 个重规划观测，合计最多 768 个观测。记录 episode/task/step ID，避免把同一条轨迹的相邻帧视为独立证据。没有现成轨迹时，仅为此收集最多 32 条 full-path baseline 轨迹，不按 feature 重复采集。

这些数量是本轮上限。先复用接入校验的 32～64 个观测，检查少量概念是否有可用信号；需要更多任务覆盖或触发样本时再补数据，不以凑满 768 个观测作为开始分析的前提。

任务指令/公开 concept-task mapping 用于概念标签。首轮只研究任务目标、物体、空间关系等静态标签，**不把 `PICK` 指令标签等同于“此刻处于抓取阶段”**。阶段识别是后续独立问题。

这些标签也可能由指令直接提供。Prompt-only 用于区分“语义有助于计算分配”和“SAE 提供额外信息”；二者效果相近，不自动否定前一个假设，也不能据此声称 SAE 识别出了在线操作阶段。

### 3.1 保存与 token 选择

只保存 L5/L16 的采样 token hidden state、选中特征分数和观测元数据，不保存全部 18 层的完整序列。

默认每个观测取固定布局的最多 64 个有效 token：最多 32 个指令 token，其余位置均匀覆盖各路相机的视觉 token；不足的指令额度可分配给图像。根据 tokenizer 确定真正的 instruction span，不能混入 padding 后按普通文本处理。长指令按固定规则覆盖头尾。离线与在线使用完全相同的采样规则。[8]

分别记录图像 token 和指令 token 的激活。若所谓高置信特征只跟着任务文本变化，应将其解释为任务语义，而不是视觉状态或当前操作阶段。

对 SAE 先逐 token 编码，再聚合稀疏激活；不要先平均 hidden state 再输入 per-token SAE。

## 4. SAE 高置信语义筛选

### 4.1 离线关联分数

复用公开的概念候选特征，并在本地发现集重新计算：

$$
R_{f,c}=\max(d_{f,c},0)\cdot\mathrm{freq}_{f,c}^{+}.
$$

其中 $d$ 是概念正、负样本的激活差异效应量，$\mathrm{freq}^{+}$ 是正样本中的激活频率。这沿用 Action Atlas 的频率加权概念关联思路；这里只保留正向概念证据，负相关特征暂不用于“概念存在”的判断。[7]

各任务/episode 等权，避免某个长轨迹主导统计。按概念筛选，首轮保留 $R_{f,c}>0$ 的候选配对，而不是只挑全局激活最大的特征。

先用 **2～3 个有支持的概念、每个概念每层 1～4 个特征**形成最小规则；有信号再扩展到 4～8 个概念，不设凑数门槛。L5 和 L16 的 feature ID 分别保存，通过概念标签对应，不能直接按 ID 对齐。

仅保留有足够正、负任务支持的概念：发现集建议至少 2 个正任务和 2 个负任务；校准集必须仍可检查正负区分。若不满足，不要硬凑概念数量。

### 4.2 在线语义分数

定义当前观测的单特征激活：

$$
a_f(x)=\max_{i\in\mathcal T(x)} z_{i,f}(x).
$$

对每个 feature–concept 配对，先在发现集上设置候选触发阈值：

$$
\tau_{f,c}=Q_{0.95}(a_f\mid \neg c),\qquad
T_{f,c}(x)=\mathbf 1[a_f(x)>\tau_{f,c}].
$$

分别统计触发后的经验命中率 $P(c\mid T=1)$、正样本覆盖率 $P(T=1\mid c)$、负样本误触发率 $P(T=1\mid\neg c)$，以及触发观测所覆盖的 task/episode 数；沿用任务等权口径。优先保留触发后能区分正负概念、且不只由单个轨迹片段支撑的特征。经验命中率受当前任务组成影响，不当作部署概率。

**高置信与高覆盖率分开判断。**例如，30% 的正样本激活、负样本均不激活的特征仍可能有用；不能因为全体正样本中位数为零而剔除。严格使用 $a_f>\tau$，避免零激活因等于阈值而被记为触发。

仅在已触发的正样本上确定参考幅度，并固定用于 full/skip 两条路径：

$$
r_{f,c}=Q_{0.50}(a_f\mid c,T=1),\qquad
b_{f,c}=r_{f,c}-\tau_{f,c}.
$$

没有正样本触发或支持不足时，该配对记为待补样本，不据此否定整个 SAE 路线。$b_{f,c}$ 需有可分辨的正值；过小时先检查激活尺度，避免用极小分母放大数值噪声。

定义未截断的归一化激活 $u$ 和在线使用的语义证据 $s$：

$$
u_{f,c}(x)=\frac{a_f(x)-\tau_{f,c}}{b_{f,c}},\qquad
s_{f,c}(x)=\operatorname{clip}(u_{f,c}(x),0,1).
$$

$$
C_c(x)=\sum_{f\in F_c}\bar R_{f,c}s_{f,c}(x),\qquad
\bar R_{f,c}=\frac{R_{f,c}}{\sum_{g\in F_c}R_{g,c}}.
$$

首轮以 `C_c >= 0.8` 作为候选高置信触发规则，并在校准集检查该概念级规则的命中率、误触发率、覆盖率及支持量；单特征可用不自动代表聚合后仍可用。这个 0～1 分数是实验定义的语义证据强度，**不是 80% 的语义正确概率，更不是 80% 的跳过成功概率**。

候选特征与初始阈值来自发现集；仅在离线校准集上根据语义区分做少量阈值调整，不使用闭环 SR 挑特征或拟合阈值。若调整 $\tau$，用发现集重算对应的 $r,b$，冻结后再计算模块影响矩阵。低覆盖率先作为局限报告；只有当前配置无法形成有支持的规则时才暂停该配置，区分接口不兼容、样本不足和语义分离失败，不把一次阈值失败扩大为路线否定。

## 5. Concept–FFN 关联：固定观测上的有限前向干预

### 5.1 具体计算

从发现集选最多 128 个任务/概念覆盖较均衡的观测，可从已有小批开始。每个观测先缓存正常执行后的完整 L5 hidden state、attention mask、position IDs。注意这里要缓存完整序列，不能只缓存 SAE 抽样的 64 个 token。

在这个共同起点上运行：

- 正常的 L6→L16 尾段；
- 分别跳过 6 个候选 FFN 的 L6→L16 尾段。

满额时共 $128\times(1+6)=896$ 次 VLM 尾段前向；小批结果缓存复用。无需重复 SigLIP 和 L0～L5；无需执行 Action Expert、flow matching 或环境 step。

对每次得到的 L16 hidden state，用同一个 L16 SAE 同时读出全部已选语义特征。**一次模块干预得到全部 feature 的变化，计算量不乘 feature 数量。**

跳 FFN 的定义是保留 attention 之后的 residual stream，不计算 FFN 分支；不是将整层输出置零。分析阶段可以先实现正确的开关；部署计时阶段必须真正绕过 FFN kernel，不能在 FFN 已算完后用 hook 把输出乘零。

### 5.2 关联矩阵

对概念 $c$，按与在线相同的 L5 触发规则定义当前离线观测池中的样本集合 $\mathcal D_c=\{x:C_c^5(x)\ge\theta_c\}$，其中 $\theta_c$ 为冻结后的概念阈值，初始值为 0.8。**不再要求 L16 full path 也有证据，不按真实概念标签剔除 L5 的误触发样本。**L16 支持率另行记录，用于理解两层读出的一致性；发现集和校准集使用同一选样口径。

L16 是离线响应探针，不是第二个在线语义分类器。有限干预可以在 L5 语义已校准、L16 候选在多个发现任务有关联的条件下开始，不额外要求 L16 在全局校准集达到某个分类 precision。后续规则条目需检查 **L5 实际触发区域内**的晚期基线证据覆盖；所有 L5 选中样本仍计入 D/E，不能通过删掉晚期弱样本来美化结果。若整个适用群体缺乏晚期基线证据，则标记该条目缺少支撑，不能把全零读出当作“无影响”。

每个候选模块 $m$ 用未截断激活计算：

$$
D_{m,c}=Q_{0.90,\,x\in\mathcal D_c}
\left[
\sum_{f\in F_c^{16}}\bar R^{16}_{f,c}
\left|u^{16,\mathrm{full}}_{f,c}(x)-u^{16,\mathrm{skip}(m)}_{f,c}(x)\right|
\right].
$$

它表示“L5 触发该概念时，跳过模块对 L16 所选概念特征激活的归一化影响”。full/skip 共用第 4 节冻结的尺度。它不涉及动作输出、动作误差、SR 或成功概率，也不把激活幅度下降直接等同于语义丢失。

同时保留将上式 $u$ 换成 $s$ 得到的截断证据变化 $E_{m,c}$，作为辅助解释，复用同一次前向即可。若 $E$ 很小但 $D$ 较大，说明证据等级可能仍然保持，而激活已有明显变化；不能因分数均饱和到 1，就把模块影响判为零。路由表使用 $D$，不靠饱和后的 $E$ 排序。

使用逐特征绝对差再聚合，避免不同特征的增减在概念平均分里相互抵消。每个概念的样本支持不足时标为 unsupported；不能把缺失统计记成零影响。

输出热力图：行是 6 个 FFN，列是当前有支持的概念。分别检查模块排序、影响的绝对大小，以及在同一变化阈值下的可跳过性是否随概念变化，不要求一个 FFN 只负责一个概念。

排序相同仍可能需要动态路由：同一 mask 在概念 A 下影响为 0.02、在 B 下为 0.20，阈值为 0.1 时应分别选择 skip 和 FULL。只有候选 mask 的选择及 skip/FULL 的可行性均未显示有用的概念依赖时，才优先保留静态跳过，不继续扩展本轮路由复杂度。

### 5.3 压缩成最多三个 skip mask

根据单模块矩阵，先为常见的高置信概念组合提议 1～2 组 mask，有必要时增加至最多 3 组；每组跳 2 个非相邻 FFN。mask 由数据决定，不预写“L8 是 PICK 模块”等结论。

在同一批发现集观测上实际运行这些组合：满额时最多 $128\times3=384$ 次 VLM 尾段前向。计算组合的 $D_{M,c}$ 和辅助指标 $E_{M,c}$，不要直接把两个单模块分数相加当作组合结果。

然后在校准集取最多 128 个观测做 full + 最多 3 个组合：最多 512 次尾段前向。每版冻结配置的总预算上限约 1792 次尾段前向；小批计入扩样总数，仍然没有逐特征闭环。校准集用于规则选择，不将其结果称作独立测试。

只保留在校准集中有支持的语义—mask 条目；在线规则表的 $D$ 使用校准集组合实测值。`delta_semantic = 0.1` 可作为初始探索值，单位是未截断归一化激活的加权变化，归一化尺度为第 4 节的 $b$；它不是截断分数下降 0.1，也不是动作安全性阈值。结合校准集的 $D/E$ 分布做少量调整后冻结，不能因默认阈值下全部回退 FULL 就直接认定没有条件冗余。

如果初始阈值没有覆盖，可保留该负结果，另存一版“相对影响先验”：仅用离线校准结果，在已实测组合中选择能产生非刷新触发的最低代价配置，再冻结后进入小批闭环。明确报告采用的阈值及其支持量；较宽阈值不能称为通过了原阈值下的语义保真检验，也不能用闭环测试成功率反向选择它。

同时输出被跳过路径在离线观测中的概念支持分布，防止只看到少数有利样本。

## 6. Runtime 规则

每次 policy 重规划调用：

```text
当前观测
  → SigLIP + VLM L0～L5 完整计算
  → L5 SAE 编码固定采样 token
  → 得到高置信概念集合 C_active
  → 选择 FULL 或三种固定 mask 之一
  → 执行 VLM 剩余层，生成所有需要的 prefix KV
  → 原样执行 Action Expert 与动作后处理
```

具体规则：

```python
# 逻辑示意，不是可直接运行的项目代码
active_concepts = {c for c in concepts if C5[c] >= theta[c]}
if first_two_policy_calls or periodic_full_call:
    mask = FULL
elif not active_concepts:
    mask = FULL
else:
    eligible = []
    for mask in candidate_masks:  # 最多 3 个
        if not all_have_offline_support(mask, active_concepts):
            continue
        semantic_cost = max(D[mask, c] for c in active_concepts)
        if semantic_cost <= delta_semantic:
            eligible.append((semantic_cost, mask))
    mask = min(eligible)[1] if eligible else FULL
```

默认每 10 次 policy 调用执行一次完整路径，作为首轮保守规则。该计数按 policy 调用，不按物理仿真步。

同一观测的多个高置信概念同时参与保护，不使用“只保留最高分概念”的 winner-take-all 路由。低分代表证据不足，不据此断言相关功能不重要。

路由只读取 L5 及之前已经获得的信息；L16 SAE 只用于离线，不在在线路径上用于决定前面是否跳过。

第一版不需要训练决策树：阈值判断加小型规则表已经能表达逻辑。以后可以将规则压缩成浅树，但无需为“有一个 router”而引入额外训练。

## 7. 4090 实现与性能控制

### 7.1 SAE 开销

首轮在 L5 的采样 token 上使用完整 SAE encoder + 全字典 TopK，不运行 decoder。公开实现的 TopK 是在全部字典坐标中选取；仅取几个 encoder 行再做 TopK，不等价于原 SAE 的选中特征。[3]

后续可以研究 selected-row 投影加校准阈值，但必须单独验证与完整 SAE 路由的一致率，不能默认为完全等价。

BF16 的 2048×16384 encoder 权重约为 64 MiB（不含 bias/统计量）；若为兼容性保留 FP32 则约 128 MiB。优先保证 loader 数值正确，再测试压低精度，不为了省这点内存改变语义结果。

768 个观测、2 层、每层 64 个 2048 维 BF16 token，纯 hidden-state 缓存约 384 MiB。尾段前向缓存另计，可放 CPU/磁盘并分批搬运。不要保存大块 `[全部观测, 全部token, 16384]` 稠密 SAE 输出。

### 7.2 编译与路由

分析阶段允许 eager + hooks；小批闭环使用已通过 no-op 校验的真实 FFN 开关，不以完成多分支编译为前提。先做一次简短回放，估计可省的 FFN 时间与 SAE/router 成本；确认质量信号和优化空间后再投入编译。声称加速时必须使用真实跳过实现。

优化阶段可编译一个共享 L0～L5 前缀和实际需要的固定尾段：FULL + 1～3 masks。权重共享，不复制多套模型。先验证分段后的 FULL 与原 FULL 等价。只核验本次改动影响的路径，已有有效校验可复用。

避免在每层 `.item()`/CPU 判断，最多在 L5 后选择一次路径；这次同步和调度开销必须计入延迟。避免新 mask 导致运行中重复编译。若多个编译/CUDA Graph 内存池使显存过高，将候选 mask 降为 2 个，仍不扩大到全部组合。

正式比较时分别报告原始 FULL、分段 FULL、FULL+SAE但不跳、以及真实 SAE 跳过的延迟；早期尚未实现的编译路径标为未测。不能把编译基线改慢后产生的差距当作加速。

### 7.3 计时

固定观测回放做性能测试：早期仅需小批估算成本；优化阶段各路径充分预热后采样，例如预热 50 次、计时 300 次，仍有冷编译则单列。

GPU 计算用 CUDA events，完整 policy 调用用同步后的 wall-clock 时间。不要逐个 kernel 强制 synchronize。所有方法采用一致的 batch size、dtype、denoising steps、输入形状和编译策略。

记录：VLM p50/p95、SAE+router 开销、完整 policy p50/p95、GPU 峰值 allocated/reserved、平均跳过 FFN 数、FULL 比例、mask 切换频率。环境仿真耗时另列，不混入纯推理加速比。

不预先承诺加速率；必须验证省下的 FFN 时间能覆盖 SAE、同步和图切换开销。

## 8. 分批闭环：只评测完整策略

冻结规则与参数后，在预留的 8 个任务上分批比较以下方法；同批各方法使用相同的初始状态/seed：

| 方法 | 用途 |
|---|---|
| 原始 FULL | 质量和延迟基准 |
| 固定 mask | 只使用一个固定 mask，可按预定非语义周期回退 FULL，检验是否只是静态冗余 |
| SAE 语义路由 | 本方案 |
| Prompt-only 路由 | 用任务指令中的概念选择同一 mask 集，检验 SAE 是否提供额外信息 |

**先获得闭环反馈，再决定扩样：**

1. 小批试验：8 个任务 × 1 个 init × 3 个方法（FULL、固定 mask、SAE 路由），共 24 条轨迹。检查成功/失败、实际触发覆盖率、mask/FULL 分布和粗略时延；不要求少量成功数差异达到统计显著。
2. 有希望则扩样：上述 3 个方法各补 4 个 init，新增 96 条；同版规则的小批结果直接计入每方法 40 条的总数。
3. 补充归因对照：Prompt-only 在相同的 8 × 5 个案例上运行，新增 40 条。它用于判断 SAE 的额外价值，不作为开始小批试验的门槛。

完成全部阶段合计 $24+96+40=160$ 条闭环轨迹。没有既有数据时，加上离线采样的最多 32 条 baseline，本轮最多 192 条完整轨迹；若小批后停止，则闭环仅 24 条，含离线采样最多 56 条。这个数量不随 feature 数量增长，原始 FULL 数据在协议兼容时可复用。

固定 mask 用发现集上的全局语义保真度选择，不能用测试 SR 选最优 mask。Prompt-only 使用预先冻结的概念词表和同一张语义—mask 表，不读取测试 task ID 或动作标签。

若小批先检验“相对影响先验”，固定对照可直接使用同一个离线选出的 pair，以集中比较是否需要语义触发；将这一协议选择写入本版 manifest。固定周期仍只在闭环前由校准回放确定，不把回放上的预算匹配当作自由闭环也已匹配。

不同路由的 FULL 回退频率可能不同。先报告 SR—延迟—平均跳过数；需要比较相近计算预算时，在闭环前用校准集确定固定周期/不依赖语义的调度，尽量配平平均跳过数量，测试时仍报告实际预算。预算不匹配不阻止探索，但不能据此宣称“同预算更好”，也不为追求精确配平反复补跑。

实现上尽量固定 episode 初始状态和每次 policy 调用的采样随机数规则。不同策略的后续观测分布自然会分叉，不要求观测保持相同。

8 episodes/方法的小批用于尽早发现信号或失败模式，40 episodes/方法的扩样仍是探索性规模，不能支撑 1～2 个百分点差异的精确结论。报告原始成功数、逐任务结果与不确定性，不挑 seed。小批结果只决定本版是否扩样，不回流到本版特征选择或规则拟合；如据此修改策略，另记探索版本，不混合不同版本的结果，也不把已用于改规则的案例称作独立验证。上述预算针对一版冻结配置，不自动累加新一轮实验。

## 9. 结果解释与停止条件

### 分阶段判断是否继续

| 阶段 | 本阶段要回答的问题 | 支持下一步的证据 |
|---|---|---|
| 离线探索 | 语义是否对应不同的计算需求？ | 本地概念可读，且影响大小、候选 mask 或 skip/FULL 的可行性呈现条件差异；排序变化不是必要条件。 |
| 小批闭环 | 这种先验能否形成可用策略？ | 路由有实际覆盖，且出现质量与计算量之间的有希望权衡，值得扩样；不要求同时胜过所有对照。 |
| 扩样与归因 | 动态语义和 SAE 各自增加了什么？ | 用固定 mask 区分静态冗余，用 Prompt-only 判断 SAE 额外价值；分别报告支持程度。 |
| 性能优化 | 当前实现能否把可跳过性兑现为净加速？ | 真实 policy 延迟下降，且质量代价符合本轮研究目标；准备/编译成本另列。 |

这四项逐步积累证据，不构成首轮必须同时通过的验收清单。有条件冗余但当前实现未加速，可以继续做有界工程优化；有语义路由收益但 SAE 未胜过 Prompt-only，可以保留语义分配结论，并收缩对 SAE 独特价值的主张。

### 负结果的范围与停止条件

- **本地读出失败：**区分接口不兼容、触发支持不足和语义分离失败。做一次针对性排查或补样本后仍无可用规则，就暂停当前层位/采样/阈值配置；不沿用无证据的 feature ID 标签，也不据此否定所有 SAE 路线。
- **模块排序相同：**继续检查影响大小及 skip/FULL 的可行性；只有这些也没有有用的概念依赖时，才停止增加本轮语义路由复杂度，保留静态跳过结果。
- **SAE 路由与 Prompt-only 基本相同：**当前实验尚未证明 SAE 的额外价值或在线状态识别收益；语义计算分配本身是否有效，仍由其与固定 mask 的对照判断。
- **语义指标保留但闭环性能下降：**当前特征及其影响指标不足以支持该跳过策略，可能涉及未覆盖的控制需求或闭环分布变化。记录失败模式；不能宣布语义保持等于动作保持，也不直接推断全部概念都无用。
- **小批结果尚不明确：**报告支持量与覆盖率；有明确待检验的信号才扩样，不为凑出显著差异反复增加 seed。
- **推理未变快：**分解 SAE、同步及编译开销；若存在可兑现的时间余量，可做有界优化，无合理余量则停止当前实现的加速投入。该结论与语义关联是否成立分开报告，不只报告跳过率或 FLOPs。

一个模块可以承担多种功能。本实验只要求存在可用的条件冗余规律，不要求发现严格的一模块一概念分工。

## 10. 交付物与代码边界

建议实现以下脚本；这些名字是新建文件建议，不是声称现有仓库中已存在：

```text
collect_semantic_data.py     # 固定观测、两层 hidden state 和 token span
select_sae_concepts.py       # 语义分数、阈值和高置信 feature 清单
map_concept_ffn.py           # 最多 896 次 full/单模块尾段前向与组合检查，不运行 AE
build_semantic_rules.py      # 最多 3 个 mask 的规则表
benchmark_semantic_skip.py   # 小批回放/24 条闭环，有希望再扩样和补对照
```

输出：

```text
run_manifest.json            # checkpoint/commit/软件/config/随机种子
concept_features.json        # 分层 feature ID、概念、统计、正负样本支持
semantic_thresholds.json     # tau/r/b、C 阈值、触发统计和 token 采样规则
concept_ffn_matrix.csv       # FFN/组合的 D/E、L5 触发支持量与 L16 支持率
skip_rules.json              # 固定 mask、支持概念、回退规则
routing_trace.parquet       # 每次调用的 C、mask、FULL 原因
latency_results.csv          # 原 FULL/分段 FULL/SAE-only/skip
rollout_results.csv          # 方法、task、seed、success、latency
```

不要实现逐 feature 闭环、动作影响回归器、额外 draft model 或全层大规模 SAE 训练。不要因高置信规则未找到而悄悄把筛选目标改成 SR。

## 参考与核对入口

[1] Action Atlas π0.5 数据卡：`https://huggingface.co/datasets/bag100/action-atlas-pi05`

[2] Action Atlas 发布说明（checkpoint/adapter）：`https://github.com/CWRU-AISM/action-atlas/releases`

[3] SAE 编码与加载实现：`https://raw.githubusercontent.com/CWRU-AISM/action-atlas/main/experiments/sae_hooks.py`

[4] 论文 v6，Appendix D.4 hook 协议：`https://arxiv.org/html/2603.19233v6`

[5] OpenPI policy 推理：`https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models_pytorch/pi0_pytorch.py`

[6] OpenPI VLM/expert 和 prefix KV 接口：`https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models_pytorch/gemma_pytorch.py`

[7] 官方概念识别分数：`https://raw.githubusercontent.com/CWRU-AISM/action-atlas/main/experiments/concept_id.py`

[8] 官方 instruction span 处理：`https://raw.githubusercontent.com/CWRU-AISM/action-atlas/main/experiments/prompt_spans.py`

[9] 原始 π0.5 激活、轨迹与层统计：`https://huggingface.co/datasets/bag100/action-atlas-pi05-activations`

[10] 网站预计算数据：`https://huggingface.co/datasets/bag100/action-atlas-viz`

[11] 官方数据下载、匹配与复用协议：`https://github.com/CWRU-AISM/action-atlas/blob/main/setup/DATA.md`
