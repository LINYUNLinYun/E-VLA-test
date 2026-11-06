# π0.5 隐藏表征语义探测实验设计

**目标：** 检查正常动作推理过程中，π0.5 的 VLM 各层、各类 token 的表示中，哪些信息能够被一个冻结的文本解码器读出。

**主路线：** 冻结模型、离线提取、Patchscopes 式表征注入、属性问答与成对反事实对照。

**版本：** 2026-11-05。本文是待实施的实验方案，不是已完成的实验结果。样本量、层配置和门槛均为建议默认值；执行时需锁定代码 commit 与 checkpoint。

---

## 1. 研究问题与实验边界

本实验回答三个问题：

1. 当前 π0.5 检查点能否作为文本解码器，可靠完成本实验所需的简单任务？
2. 正常动作推理产生的隐藏表示，是否携带可解码的视觉属性、空间关系、任务条件化信息？
3. 这些信息的可解码性，如何随 VLM 层数、token 位置和读出方式变化？

第一轮不做跳层、不训练 π0.5、不修改动作策略、不研究动作误差或在线调度，不探测 Action Expert，也不把注意力图、聚类图或余弦相似度当成语义证据。

### 1.1 结果的正确表述

可以报告：

> 在冻结的目标模型、提示和注入配置下，第 l 层的某组表示能够支持某属性的解码，且这种能力优于无注入和打乱源表示对照。

不能直接报告：

> 该层首次产生了这种信息；解不出来就说明不存在；后面的层没有作用；模型一定使用这项信息生成动作。

Patchscopes 的原始实验证据主要来自自回归语言模型。迁移到 π0.5 是需要验证的新实验，而不是已知可行的直接复现。[1]

## 2. 执行前的事实核对

OpenPI README 当前仍把 π0.5 的标准训练／推理支持限定在 flow-matching head；因此本实验不能直接假定公开检查点已具备可用的文本生成流程。[2]

JAX 实现保留共享词嵌入的词表解码操作，并在模型返回时执行最终 RMSNorm；PyTorch 实现包含 PaliGemma 模块。但这些只提供实现文本解码的线索，不构成文本能力已经成立的证据。[3][4]

执行时输出 `model_audit.json`，至少记录：

| 字段 | 要求 |
|---|---|
| 代码和权重 | 仓库 commit、checkpoint 路径／版本、后端、依赖版本、权重加载缺失项 |
| 模型结构 | VLM 层数、hidden size、视觉 token 数、词表大小 |
| 输入契约 | 相机名称与拼接顺序、image mask、文本模板、tokenizer、state 是否实际进入 VLM |
| 数值实现 | dtype、attention 实现、位置编号、embedding 缩放、最终归一化位置、词表投影权重来源 |
| 原始动作配置 | 为重现原推理保留记录；不将其作为本次探测变量 |

常见 OpenPI `gemma_2b` 配置是 18 层、2048 维，但实现应从配置读取，不写死。[3]

**特别注意：** 当前官方 `pi05_libero` 配置将 `discrete_state_input` 设为 `False`；LIBERO 适配器通常有两个有效相机和一个被 mask 的填充相机。因此不能默认存在三路有效图像，也不能因为 observation 字典有 state 就声称 VLM 使用了 state。[5][6]

## 3. 总体执行顺序

```text
A. 审计实际模型与输入，检查提取／注入不破坏原计算
                  ↓
B. 验证候选 Target 的文本输出和基础问答能力
                  ↓
C. 验证注入机制：自替换 + 同模型 layer-0 图像替换
                  ↓
D. 小规模成对语义探测，校准提示／位置／目标层
                  ↓
E. 冻结配置，在独立测试集扫描所有源层
                  ↓
F. 输出可解码性曲线、对照结果和明确的结论边界
```

优先用当前 π0.5 检查点同时充当 Source 和 Target。只有经排错仍不能胜任 Target 时，才转入外部 Target 分支；Source 始终保持为待研究的 π0.5。

## 4. 阶段 A：提取接口与计算一致性

### 4.1 Source 必须保持正常动作推理输入

Source 输入沿用现有策略的图像预处理、原任务指令、相机 mask、文本格式，以及实际存在的 state 编码。不额外加入“物体是什么颜色”等探测问题，不提前附上答案。

允许构造合法的反事实观测／任务指令，但必须标记其来源。人为更改指令构造的诊断样本，不能伪装成官方 benchmark 的原生轨迹。

只运行原动作推理的 prefix 路径即可提取目标表示；先对少量样本检查它与完整策略实际执行的 prefix 结果一致。OpenPI 原动作采样路径本身会先计算视觉语言 prefix，再运行动作专家。[7]

### 4.2 统一 hidden state 定义

使用：

$$
H^{(0)} = \text{VLM 第一层之前的输入表示},\qquad
H^{(l)} = \text{第 }l\text{ 个 Transformer block 完整残差更新后的输出}.
$$

最后一层的 block 输出和经过最终 RMSNorm 的输出分开保存、分开命名。不要混用 block 输出、attention 输出、MLP 输出、KV 或最终归一化表示。

`H^(0)` 的视觉部分已经经过视觉编码器，不是原始像素；它可以作为“进入 VLM 前”的参考，但不是“尚未完成任何视觉处理”的基线。

层编号固定为 0…L，并在结果中明确编号规则。

### 4.3 最低工程检查

关闭随机增强和 dropout，使用 eval／inference 模式。先用 eager 路径调通，避免编译或图捕获影响 hook 行为。

只安装提取 hook、不修改张量时，prefix 输出应与原路径一致；将同一张量读出后写回同一位置，输出也应一致。记录最大绝对误差、均方误差以及必要的 logits 差异。数值容差依据同一后端重复运行的误差基线设定，而非为通过测试任意放宽。

任何工程一致性检查失败，都先停止语义解释。

## 5. 阶段 B：验证 Target 的文本解码能力

### 5.1 最小解码实现

独立新增文本探测入口，不修改正式策略接口。冻结所有模型参数，不新建随机 LM head。

```text
图像 + 问题前缀 → 视觉编码器／VLM → 最后有效位置的输出
               → 原检查点的词表投影 → 生成下一个文本 token
```

首版采用 batch size 1、贪心解码、最多 16–32 个新 token，遇 EOS 停止。正确性优先，可以不使用增量缓存。

必须核对：

- 原检查点的词表投影／共享 embedding 已正确加载；没有静默随机初始化。
- 最终 RMSNorm 恰好执行一次；embedding 缩放也恰好执行一次。
- logits 来自最后一个有效输入位置，而不是 padding。
- 问题前缀采用对应模型支持的 prefix-LM mask；回答 token 使用因果 mask，前缀不能读取回答 token。
- 新增回答 token 继续由 VLM 专家处理，不进入 Action Expert。
- tokenizer、BOS／EOS、换行、position IDs、右 padding 均正确；打分时 logits 与答案 token 进行正确的移位配对。
- Source 保持原动作 tokenizer；Target 使用独立文本输入入口，不能让动作模板污染问答格式。

OpenPI 的动作 tokenizer 会根据配置清洗文本并选择不同格式，不能直接假设它等于标准问答 tokenizer。[8]

无缓存完整重算时，每次都保持相同的 prefix／answer 因果边界。进入 patching 阶段后，每次重算都需施加同一注入；不能只在首个 token 生效。

### 5.2 冒烟数据与候选提示

先准备 20–40 张内容明确的图片，包含简单单物体图与 LIBERO 截图。只问类别和颜色，暂不问复杂规划。

候选格式：

```text
caption en\n
answer en What color is the cup?\n
answer en What object is on the table?\n
```

`\n` 在实际输入中是换行。上述格式来自 PaliGemma 官方提示说明；π0.5 是否仍支持，需要实测。[9]

每项尽量构造成对图片：问题不变，目标属性改变。记录原始输出、token IDs、EOS 情况、精确匹配结果和非法答案比例。可附加候选短答案的条件似然作诊断，但不能把封闭候选选择成功写成自由文本生成成功。

### 5.3 阳性对照与进入下一阶段的条件

使用已验证能完成这些问题的 PaliGemma 检查点作为任务阳性对照。`google/paligemma-3b-mix-224` 是可测试的候选，其官方模型卡提供图文输入、文本输出的用法。[10]

它答对，只能说明任务和输入可行，不能替 π0.5 的自定义解码实现背书。工程上更强的检查是让同一个对照检查点通过官方路径和新解码路径，比较 logits 与生成结果。若移植条件不允许，明确标记这项检查未完成。

探索性进入条件：在标签均衡的简单任务中，能正确随图片切换答案，整体优于问题-only／固定控制图基线；正式统计判断留给较大的独立测试集。20 张图只用于排错，不支撑强结论。

若只能做可靠的候选答案打分，可以继续进行“封闭集合语义可解码性”实验，但必须与自然语言解释路线分开报告。

## 6. 阶段 C：验证注入实现，而不是直接扫描所有层

### 6.1 恒等自替换

同一模型、同一输入、同一层、同一位置，将提取到的 hidden state 原样写回。结果应与不注入相同。该测试用于排除 hook、dtype、索引等错误。

### 6.2 同模型 layer-0 全图像替换：强阳性控制

在同一个 Target 上运行：

```text
参考：真实图像 + 固定问题 → 答案／logits
干预：固定控制图 + 同一问题
      在 VLM layer 0，将全部有效视觉 token 替换为
      “同一个 Target 对真实图像计算出的视觉输入 token”
      → 答案／logits
```

只要预处理、图像顺序、文本、mask、位置编号和其他输入完全相同，干预后的 VLM 输入应与参考完全一致，因此后续输出也应一致。这是计算等价性的检查，不是“发现了高层语义”的结果。

**外部 Target 情况下也先用 Target 自己产生的视觉 token 完成这个控制。** 不能用 π0.5 的视觉 token 去要求另一个检查点获得完全一致的结果。

### 6.3 可选的文本表征控制

源文本使用已知颜色／类别描述，目标端先验证普通文本形式下能回答该属性，再把描述替换成占位符并注入源 hidden state。

例如候选 Target：`answer en What color is X?\n`。

该提示本身是否有效也需要校准。文本 token 补丁失败不必阻断已经通过工程检查的多视觉 token 路线；两种接口需要分别验证。

## 7. 阶段 D：探测任务与数据集

### 7.1 第一轮只做三类语义

| 任务 | 输出标签 | 成对／交叉变化 | 要排除的捷径 |
|---|---|---|---|
| 视觉属性 | 目标颜色；或简单物体类别 | 保持任务含义，改变视觉属性 | 指令直接写出答案，或物体类别固定绑定颜色 |
| 空间关系 | 左／右，先只做明确二分类 | 固定指令与相机，改变目标相对夹爪的位置 | 图像翻转、坐标系混乱、几乎重叠的边界样本 |
| 指令条件化视觉定位 | 被指令选中的目标位于左／右，或其颜色 | 同图换目标指令，同时同指令换目标位置／外观 | 只读指令、只看图像，或根据任务名称背答案 |

**任务条件化示例：** 场景有杯子和碗，分别在左右两侧。使用“拿杯子”和“拿碗”两条合法指令，再交换两者位置，形成 2×2 组合。探测问题只问“任务目标在左还是右”。要四种组合都正确，不能只依靠指令中的名词。

颜色变化优先来自真实不同样本或仿真渲染。若使用重着色等合成变换，要保持几何／光照尽量一致，并与原生 LIBERO 数据分开报告。颜色可选类别以实际数据为准，不强行制造四分类。

### 7.2 坐标和标签

空间关系第一轮定义在模型实际看到的 agent-view 图像坐标系，而不是含混地使用“机器人左边”。根据预处理后的目标／夹爪投影位置生成标签，并设置不确定区间，剔除遮挡或近重合样本。

标签来自仿真元数据、可验证的几何规则或人工标注。不给同一个待评估语言模型自己生成标签。

目标物体名出现在任务指令里是允许的，但该实验的被测属性不能由这个名字直接泄露。若类别答案已经在指令里，类别探测只能算指令信息读出，不能算视觉识别。

### 7.3 规模和划分

第一轮建议收集约 300–600 个“观测＋指令”样本，均衡覆盖上述三类任务。先以约 30% 为校准集，70% 为独立测试集；扩展实验再增加轨迹和场景数量。

按 episode／场景及其派生反事实分组划分，不将相邻帧随机打散。成对样本及 2×2 样本必须在同一分区。Source 模型预训练是否见过某 benchmark，不由这个划分保证；这里保证的是探测配置的调参与测试分离。

校准集可选择 Target 提示、目标注入层、token 选择规则。测试集不再调整配置，不按单个测试样本挑选能答对的层。

### 7.4 每条记录的必要字段

```text
sample_id, episode_id, scene_id, counterfactual_group_id, split,
checkpoint_id, camera_paths, image_masks, original_task_instruction,
actual_state_input_if_any, task_type, label, label_source,
coordinate_frame, target_object_metadata, is_synthetic
```

探测问题、标签和元数据只由 Target／评估器读取，不能意外写入 Source prompt。输出目录名、文件名、样本序号也不要进入提示，防止泄露标签。

## 8. 两条互补的表征读出路径

### 8.1 路径 T：单个文本 token → Target 占位符

从 Source 的正常推理表示中，按预先定义的规则提取：

- 最后一个有效的任务文本 token；
- 任务中目标物体名的最后一个子词；
- 文本终止／分隔 token，作为独立位置类型。

“任务文本最后位置”与“整个 prefix 的最后有效位置”可能不是同一个位置；state 和分隔符也可能占据尾部，必须按 token map 区分。

将该向量替换到 Target 提示中 `X` 对应的完整 token span；若 `X` 不是单 token，应重新选用单 token 占位符或明确多 token 规则，不靠字符索引猜 token 索引。

$$
H^{(q)}_{T,j} \leftarrow H^{(l)}_{S,i}.
$$

第一轮使用恒等映射，不额外归一化或缩放。任何范数匹配／映射都作为单独的消融变体。

Target 提示示例只是待校准候选，不是保证有效的提示：

```text
answer en What color is X?\n
answer en Is X to the left or right of the gripper?\n
```

Target 不接收 Source 的真实图片、任务答案或能透露目标属性的描述；需要图像输入时使用所有样本共用的控制图。

### 8.2 路径 V：视觉 token 块 → Target 对应视觉位置

Source 中分别取 agent-view、wrist-view 或全部有效图像的 token，保持原顺序。Target 使用相同相机槽位、数量、mask 和 token 布局的固定控制图，保留 Target 自己的问题文本。

$$
H^{(q)}_{T,J_{\mathrm{image}}}
\leftarrow H^{(l)}_{S,I_{\mathrm{image}}}.
$$

先使用完整的一路图像 token，再测试全部有效相机。不要一开始就将数百个 token 平均池化为一个向量，并把池化失败解释为没有视觉信息。

ROI 选取仅作为可选诊断：若使用仿真真值框选择 token，结果需要标注为 oracle ROI，不代表系统自己找到了目标。不要通过目标槽位数／占位符位置编码标签。

**解释限制：** VLM prefix 在 OpenPI 中允许视觉语言 token 之间的全注意力；因此深层 agent-view 位置的表示可能已经包含腕部图像和指令信息。“从某相机位置读出”不等于“信息只来自该相机”。[7]

两条路径传递的 token 数不同，分别报告结果，不把全视觉块优于单 token 简单解释为模型或方法更优秀。

## 9. 注入层、续算与缓存规则

### 9.1 层配置

初始校准可采用源层 `{0, 3, 6, 9, 12, 15, 18}`，目标层 `{3, 6, 9, 12}`；非 18 层模型按实际深度调整。校准成功后，源层扩展为 `0…L`。

主结果中，每个任务／读出路径固定一个 Target、一个提示和一个目标层 q，再扫描全部源层 l。q 通过校准集的预设规则选择，例如跨候选源层的平均 balanced accuracy，而不是每层／每样本重新择优。

附加报告完整的源层×目标层结果。可以报告同层注入 `q=l`，但只能作为另一种配置：此时目标端剩余计算量随源层改变，不宜把曲线简单解释为源信息量变化。

### 9.2 注入之后必须真正影响后续计算

在约定的 block 边界替换 residual stream，然后从下一个 block 继续。不要把被替换位置之前的 KV 当成后续层的正确 KV。

最简单的正确实现是无缓存重算，并在每个生成步／每次候选评分中重新施加注入。优化后再使用 patched prefill 构建的缓存；下游受影响层的 KV 必须来自干预后的路径。

不能只修改展示出来的 `hidden_states` 返回值，却继续用旧缓存生成答案。多 token 回答中，注入之前各层的占位符缓存仍可能构成干扰，因此第一轮优先短标签回答，并保留这个方法局限。

## 10. 必须执行的对照

| 条件 | 操作 | 用途 |
|---|---|---|
| 真实源表示 | 注入当前样本正常推理产生的表示 | 主实验 |
| 无注入 | 控制图和问题正常推理 | 测试问题先验／Target 自带偏好 |
| 恒等自替换 | 写回 Target 自己同位置的表示 | 排查注入实现错误 |
| 匹配条件内打乱源表示 | 从不同标签、同提示类型／布局样本取得 donor | 检查答案是否依赖正确的源信息 |
| 范数匹配随机向量 | 同维度、相近范数的随机方向替换 | 检查一般扰动；不能单独作为主要对照 |
| layer-0 同模型视觉替换 | Target 自己的真实图像输入 token 替换控制图 token | 工程阳性检查 |
| Target 直接看真实图像问答 | 非 patching 的正常问答 | 任务能力参考，不是数学上的上界 |

对于打乱源表示，同时报告：对原样本标签的准确率，以及答案是否转向 donor 的属性。仅仅“正确率下降”不够；如果能受控地转向 donor 的答案，更能支持注入在传递被测属性。

采用至少两种固定控制图做稳定性检查。控制图可能改变分布，不能把一种控制图造成的失败当作源表示没有信息。

## 11. 指标、统计与可视化

### 11.1 主指标

**Balanced accuracy：** 在类别不完全均衡时对每类准确率取平均。非法／无法解析的生成按错误计入，不从分母中删除；解析规则和同义词表在看测试集之前固定。

**成对双正确率：**

$$
\mathrm{PairAcc} = \frac{1}{P}\sum_{p=1}^{P}
\mathbf{1}[\hat y_{p,A}=y_{p,A}\ \land\ \hat y_{p,B}=y_{p,B}].
$$

不要只统计“换图后答案有没有变化”，因为两个答案都错也可能变化。

**对照增益：**

$$
\Delta_{\mathrm{nopatch}} = \mathrm{BAcc}_{\mathrm{real}}-\mathrm{BAcc}_{\mathrm{nopatch}},\qquad
\Delta_{\mathrm{shuffle}} = \mathrm{BAcc}_{\mathrm{real}}-\mathrm{BAcc}_{\mathrm{shuffle}}.
$$

另报告非法答案率、donor 属性跟随率。任务条件化 2×2 组同时报告四种条件均正确的比例。

### 11.2 候选答案似然：辅助指标

针对有限标签集合，可计算完整短答案的条件似然：

$$
s(y)=\sum_{t=1}^{|y|}\log p(y_t\mid Q,\mathrm{patch},y_{<t}).
$$

尽量采用 token 长度一致的标签；否则明确序列长度偏差与预先固定的归一化／校准规则，并与自由生成结果分开报告。Teacher forcing 只能提供当前 token 之前的答案，不能把整个答案作为双向前缀。

### 11.3 统计

按 episode／场景反事实组进行配对 bootstrap，建议 1,000 次，给出 95% 置信区间。不要把同轨迹相邻帧视为大量独立样本。

主配置先在校准集冻结，再在测试集比较真实源表示与对照。大量逐层检验如要宣称“首次显著层”，需处理多重比较；探索阶段优先报告完整曲线及不确定性，不制造一个确定的“语义诞生层”。

### 11.4 应交付的图表

- 各语义任务的逐层 balanced accuracy 曲线，含无注入／打乱对照和置信区间。
- 源层×目标层准确率热图，明确主结果固定的目标层。
- 文本 token／视觉 token 块的结果对比，标注 token 数。
- 成对／2×2 反事实的一致性表和实际生成文本案例。

不得填入示意性的假数据，也不要只展示成功样本。预先随机抽取案例，并补充主要失败类型。

### 11.5 可选的开放式自然语言解释

定量属性测试通过后，再在同一批预先选定的样本上尝试开放式描述，例如询问“被注入的表示对应什么对象／场景”。提示与最大输出长度在校准集固定，保留无注入和错误 donor 对照。

把生成描述拆为可核对的原子事实，例如物体类别、颜色、相对位置、任务目标。由不知道层号与实验条件的评估者根据原图和原任务标注“支持／矛盾／无法判断”。报告正确事实覆盖率及矛盾或无依据事实比例，并展示预先随机抽取的案例。

开放式描述用于发现后续可测试的属性假设，不替代闭合集合定量实验，也不能视为 hidden state 全部信息的清单。尤其不能把流畅但不受源表示控制的描述当成解释成功。

## 12. 分支处理与停止条件

### 12.1 π0.5 不适合作为 Target

先完成工程排错和任务阳性对照，再尝试同源且已验证能完成任务的 PaliGemma 检查点作为外部 Target。Source 仍是原 π0.5。

先验证外部 Target 的自身 patching，再测试 π0.5→外部 Target。相同 hidden size 不代表空间兼容，原论文也将跨模型表示差异作为需要处理的问题。[1]

恒等映射失败，只说明该配置未能解码，不说明源模型缺少语义。

如尝试学习仿射映射，应单独立项、单独报告，不再称为完全 training-free。映射训练数据与探测测试数据严格分离；可优先用成对表示对齐而不是直接用语义答案训练，以降低把能力学进解码器的混淆。任何学习到的映射也都只能说明“在此读出器下可解码”。

### 12.2 文本 token 路径失败，视觉块路径成功

可以继续研究视觉块的逐层可解码性；不要求单 token 必须成功。可能是信息分布在多个位置，或文本占位符读出方式不匹配，但不能仅凭这一结果确定具体原因。

### 12.3 只能封闭候选分类，不能自由生成

保留封闭集合实验，明确报告能力边界；不能用“回答了几个备选标签”包装成开放式 hidden state 自解释。

### 12.4 两条路径都没有超过对照

结束当前配置的可行性测试，提交完整的排错记录与负结果。可选后续使用冻结源特征的线性分类探针，验证某些属性是否线性可解码；这属于需要训练读出器的新实验，不是主方案的替代完成。

## 13. 建议代码组织与产物

以下是待实现的文件职责，不代表文件已经存在或脚本已经验证可运行。

```text
experiments/pi05_hidden_probe/
  audit_model.py           # 记录真实结构、token map、mask、权重和环境
  test_text_decode.py      # 普通文本输出与同权重参考实现对照
  test_patch_integrity.py  # 恒等替换、layer-0 全视觉替换
  build_probe_manifest.py # 数据清单、标签、反事实分组和拆分
  extract_source.py       # 沿原动作输入提取各层表示
  run_patch_probe.py      # 两条读出路径、层扫描、所有对照
  analyze_results.py      # 指标、bootstrap、实际数据图表
  configs/
    smoke.yaml
    calibration.yaml
    evaluation.yaml
```

Source 特征只保存所需层和位置，分批写入 CPU／磁盘，避免长期在 GPU 保留全量 hidden state。缓存键包含 checkpoint、预处理配置、原始输入哈希、层号和位置规则，防止误复用。

结果应至少包含：

```text
model_audit.json
sample_manifest.jsonl
split_manifest.json
text_decode_smoke.jsonl
patch_integrity.json
frozen_probe_config.yaml
probe_results.jsonl
layer_metrics.csv
summary.md
figures/
```

每条 `probe_results.jsonl` 记录源／目标检查点、源层、目标层、token 位置或块范围、提示 ID、干预条件、donor ID、原始生成、解析答案、标签、候选分数（如有）以及随机种子。

## 14. 第一轮最小交付标准

第一轮完成以下范围即可，不要求一开始做所有扩展：

1. 正确审计当前检查点，完成 20–40 张图的文本解码冒烟测试。
2. 恒等替换与同模型 layer-0 图像替换通过工程一致性检查。
3. 完成视觉属性、左右关系、任务条件化定位三类小规模测试。
4. 跑通单文本 token 与完整视觉 token 块两种接口，或明确记录某接口失败。
5. 固定 Target 配置，在独立样本上扫描源层，包含无注入、打乱和成对反事实对照。
6. 提交真实结果与失败案例，结论只限于所测试信息和读出配置下的可解码性。

**优先级：先证明能读，再分析读出了什么，最后比较不同层；不要直接画一张逐层曲线，再倒推解码器一定有效。**

---

## 参考来源

以下为核对方案前提所使用的一手资料。代码链接指向 main，执行时应另行记录实际 commit。

[1] Ghandeharioun et al. *Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models*. ICML 2024. https://arxiv.org/html/2401.06102v4

[2] Physical Intelligence, OpenPI README. https://github.com/Physical-Intelligence/openpi/blob/main/README.md

[3] OpenPI JAX Gemma implementation. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py

[4] OpenPI PyTorch PaliGemma／Gemma implementation. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/gemma_pytorch.py

[5] OpenPI training configuration (`pi05_libero`). https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py

[6] OpenPI LIBERO input adapter. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py

[7] OpenPI prefix／action inference implementation. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py and https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py

[8] OpenPI tokenizer implementation. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py

[9] Google, PaliGemma prompt and system instructions. https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions

[10] Google, PaliGemma-3B-mix-224 model card. https://huggingface.co/google/paligemma-3b-mix-224
