# π0.5 视觉编码器 A/B 压缩实验设计与执行方案

版本：2026-10-20。用途：交给 coding agent，在现有 π0.5 项目内实现、计时和评测。

## 0. 任务与最终交付

本实验只回答两个主要问题：

1. 在相同 checkpoint、后端、精度、编译和输入条件下，A/B 能使 **vision encoder 的实际推理时延降低多少**？
2. 将压缩视觉特征接回原 π0.5，**闭环任务成功率下降多少**？

本方案中的“准确率”统一指 LIBERO 的 episode task success rate（SR）。视觉特征相似度、动作 MSE、离线动作预测误差都不能替代 SR。

执行顺序固定为：环境与基线审计 → 实现与正确性检查 → 视觉计时 → 少量闭环筛选 → 独立确认集 → 若已有 self-speculative 系统，则仅替换 draft 视觉路径做系统复测。

**实验本身不训练任何模型，不改变 checkpoint 参数。本文没有实验测量结果。最终报告中的数字必须由运行产生，失败和不加速的配置也应保留。**

### 给 coding agent 的任务指令

> 请在当前已能运行的 π0.5 项目中实现本文的 C0、A128、A64、B128、B64，保持 VLM、action expert、token 布局、动作采样步数、动作执行规则和权重不变。首先读取项目 AGENTS.md 和已有运行说明，优先复用现有环境、checkpoint 与评测器。主实验采用同一种确定性 ToMeSD 式匹配，A 仅压缩视觉 attention 残差分支，B 额外压缩 MLP 残差分支。每个分支计算后恢复原 token 布局，再加完整残差。完成正确性检查后，测量包含匹配与展开开销的视觉时延，并以相同任务、初始状态和动作噪声评测 LIBERO 成功率。按本文的预算分阶段执行，保留逐次计时与逐 episode 结果，生成 CSV、图和报告。若缺少 checkpoint 或可运行环境，完成不受阻的代码与检查，并明确记录缺失项；不得编造运行结果。不要创建训练任务或重做一套 verifier。

## 1. 实验边界与基线定义

### 1.1 两层评测分开报告

**第一层：完整 π0.5 的视觉压缩消融，必做。**

- C0：完整原始视觉塔 + 原始 VLM + 原始 AE。
- A/B：只改视觉塔；VLM/AE 与 C0 完全相同。
- 这一层隔离视觉近似的因果影响，回答 encoder 加速与完整策略 SR。
- 不同时加入已有的 VLM 跳层、AE 跳层、去噪减步或动态验证策略。

**第二层：现有 self-speculative 系统复测，有可运行实现时执行。**

- S0：现有 self-speculative 系统，draft 使用原视觉塔。
- SA/SB：只替换 draft 的视觉路径，其他 draft 配置、verifier、接受阈值、刷新周期和 fallback 原样保留。
- 单独报告系统 SR、完整 draft–verify 周期时延、接受动作数与回退比例。
- 第一层 SR 不能写成第二层系统 SR；视觉塔加速不能直接写成 speculative 系统加速。

### 1.2 实验中保持不变的内容

- 所有已有权重、normalization statistics、tokenizer、state 编码、图像预处理、相机顺序。
- 输入图像分辨率、patch embedding、原视觉位置嵌入、最终 encoder norm、原图像投影/head 及缩放。
- VLM 与 AE 的层数、计算路径、KV cache 格式、mask、position IDs、动作位置起点。
- flow-matching 的时间网格、步数、动作维度、预测 horizon 与每次实际执行的动作数。
- dtype、attention backend、编译选项、相机调用/批处理方式。

本轮不加入低分辨率、视觉跳层、跨帧缓存、KV 压缩、量化或训练后的 scorer，以免混淆收益来源。

## 2. 实验前审计与可复现清单

### 2.1 优先沿用当前能运行的后端

已有 JAX 就先用 JAX；已有 PyTorch 就先用 PyTorch。不要为了这次实验临时迁移后端。所有主表结果来自同一后端与同一硬件。

记录以下内容到 `manifest.json`：

- 项目 git commit、工作区 diff、模型代码与实际加载的 SigLIP 源码路径。
- checkpoint 路径或 URI、可获得的版本/校验信息、model config、normalization assets。
- Python、JAX/jaxlib 或 PyTorch、Transformers、CUDA、驱动、LIBERO、robosuite、MuJoCo 版本。
- GPU 型号、显存、功率限制、运行时是否有其他 GPU 任务、可获得的时钟/温度信息。
- dtype、attention 实现、JIT/torch.compile/CUDA Graph 配置。
- 图像尺寸、每图 patch 数、有效相机数、实际编码的相机槽位数、视觉宽度/层数/MLP 宽度。
- action horizon、action dim、去噪步数、replan_steps、state 编码设置、环境最大步数。
- 全部随机数种子、episode 划分、观察样本清单、算法静态配置。

### 2.2 OpenPI 参考配置

若项目尚无指定 LIBERO checkpoint，可使用官方 `pi05_libero` 配置及 `gs://openpi-assets/checkpoints/pi05_libero/`，以本地可访问性为准。[3]

当前官方配置中，`pi05_libero` **将 action_horizon 覆盖为 10**；不要套用其他配置常见的 50。官方 LIBERO evaluator 默认 `replan_steps=5`，每个任务默认 50 个 episode，并默认只跑 `libero_spatial`。本实验需要显式覆盖四个 suite。[4][5]

第一层实验默认使用原 `num_steps=10`。若现有 checkpoint 或项目基线有其他已确认配置，则所有组共同沿用并记录；不要为了改善某组表现单独调整。[7]

官方 README 的四套件平均 SR 为 96.85%，只能作为该 checkpoint 的参考值，不能替代本机 C0，也不能用作小样本验收硬阈值。[3]

LIBERO 输入适配通常包含两个真实视角与一个 padding 相机槽。代码中的 image mask 不等于该视角一定没有运行视觉塔。以真实执行路径为准，C0 与 A/B 使用相同槽位和调用方式；本轮不额外删除 padding 视角。[9]

## 3. 主实验矩阵

| config_id | attention 内部 token 数 | MLP 内部 token 数 | 每层输出 token 数 | 匹配算法                              | 启用层          |
| --------- | ----------------------: | ----------------: | ----------------: | ------------------------------------- | --------------- |
| C0        |                     256 |               256 |               256 | 无，原模型                            | 无              |
| A128      |                     128 |               256 |               256 | tome_deterministic                    | 全部视觉 blocks |
| A64       |                      64 |               256 |               256 | tome_deterministic                    | 全部视觉 blocks |
| B128      |                     128 |               128 |               256 | tome_deterministic，同 block 复用映射 | 全部视觉 blocks |
| B64       |                      64 |                64 |               256 | tome_deterministic，同 block 复用映射 | 全部视觉 blocks |

这里 K 表示**每个视觉子模块内部实际参与计算的 token 数**，不是最终输出给 VLM 的 token 数。每个子模块结束都恢复到 N；不累计跨层缩短序列。

主实验同时改变的唯一因素是 `mode ∈ {A,B}` 和 `K ∈ {128,64}`。不要把“只改中间层”和“全层压缩”混在同一组结果中。

若运行时不是 16×16、N=256 的视觉网格，先停止使用本表的硬编码，报告实际配置并按相同机制建立显式新配置；不要静默裁剪、补零或改变输入。

## 4. A/B 的精确算法

### 4.1 残差分支接法

令 X 为某个视觉 block 的入口 dense token；M 为合并，U 为按映射展开。

**A：只压缩 attention 分支。**

\[
Y=X+U\bigl(\operatorname{Attn}(M(\operatorname{LN}_1(X)))\bigr),
\qquad
X_{\mathrm{next}}=Y+\operatorname{MLP}(\operatorname{LN}_2(Y)).
\]

**B：attention 与 MLP 分支都压缩。**

\[
Y=X+U\bigl(\operatorname{Attn}(M(\operatorname{LN}_1(X)))\bigr),
\qquad
X_{\mathrm{next}}=Y+U\bigl(\operatorname{MLP}(M(\operatorname{LN}_2(Y)))\bigr).
\]

M/U 的映射由 block 入口 X 计算一次。B 在 MLP 分支复用该映射；下一 block 再根据其新入口 X 重新计算。公式省略了原实现的 dropout，实际代码应保留原位置，推理时关闭。[1][2]

```python
# 结构伪代码；接入时复用原模块，不能据此重写整个 SigLIP。
mapping = make_map(x, keep_tokens=K)

residual = x
z = original_norm1(x)                 # 先在 dense token 上做 LN
z = merge(z, mapping)                 # N -> K
z = original_self_attention(z)        # 包括原 Q/K/V 与 output projection
z = unmerge(z, mapping)               # K -> N，按原 patch 索引恢复
z = original_branch_dropout1(z)
x = residual + z                     # 完整 residual 不经过 merge

residual = x
z = original_norm2(x)
if mode == "B":
    z = merge(z, mapping)
    z = original_mlp(z)
    z = unmerge(z, mapping)
else:
    z = original_mlp(z)
z = original_branch_dropout2(z)
x = residual + z
```

不要将 `merge(norm(x))` 改成 `norm(merge(x))`；两者一般不同。也不要先计算全部256个 token 的 QKV 再合并，那会丢掉投影层的省算收益。

### 4.2 确定性匹配规范

主实验采用 ToMeSD 机制的确定性适配，命名为 `tome_deterministic`。这是 π0.5 上的新适配，不是已有 π0.5 复现结果，也不声称与 ToMeSD 的随机默认配置逐行相同。

每个视角、每个 batch 样本独立处理，不跨相机、跨帧或跨层合并。

1. 将256个 token 按原顺序视作16×16网格。每个2×2区域固定选左上角 token 为 destination，得到64个 dst；其余192个为 src。
2. 使用 block 入口、尚未 LayerNorm 的 X 做匹配特征。在 FP32 中沿通道做 L2 归一化，分母下限固定为 `1e-6`。
3. 计算每个 src 与全部64个 dst 的 cosine similarity，矩阵形状 `[B,192,64]`。这是全局 src–dst 匹配，不是只在同一2×2格内匹配。
4. 每个 src 选择相似度最高的 dst；并列时取原索引较小者。
5. 按各 src 的最高相似度降序排序，并列时按原 src 索引升序。合并前 `r=256-K` 个 src：K128 合并128个，K64 合并192个。
6. 建立原 token 到 compact token 的 group index `g[B,256]`。未合并 src 各占一组，按原索引升序排列；随后为64个 dst 组，按 dst 原索引升序排列。组数严格等于 K。
7. `merge(Z,g)` 对每组求算术平均，**包含 dst 本身**。统一使用 FP32 累加后 cast 回输入 dtype；所有配置使用同一实现。
8. `unmerge(Z_short,g)` 使用 `Z_short[b,g[b,i]]` 恢复原256个位置及顺序。

固定索引可以预先建立为静态常量或非持久化 buffer；输入相关的相似度与分组必须在线计算，并计入时延。GPU 上完成匹配、排序、合并与展开，禁止每层把分数搬到 CPU。

不继承 ToMeSD 针对 Stable Diffusion CFG 的 batch 奇偶随机开关。不使用额外随机选点，不污染动作采样 RNG。不添加 proportional attention、log(group_size) 或其他 attention 补偿；这些可以另立实验，不能混入本轮 A/B。[2]

默认2×2 dst 网格的最小 K 是64。不得只把删除比例设为90%，却声称得到了 K=26；本轮不测 K<64。

### 4.3 为什么不能直接用原版 ToMe patch

本实验要求完整 residual 保留，每个子模块结束恢复 N。原版 ToMe 的常见 ViT patch 会逐层合并整个 hidden state，并将短序列传到下一层；它不等于本文的 A/B。[1][2]

若增加 `local_mean` 对照：K128 固定为水平1×2 pooling，K64 固定为2×2 pooling，再按区域重复展开；必须单列配置名。不能 A 用 pooling、B 用相似度匹配，也不能用 pooling 的失败结果否定主实验的匹配方案。

## 5. 修改位置与后端注意事项

### 5.1 JAX 路线

已核对的官方入口：[6][7]

- 修改 `src/openpi/models/siglip.py::Encoder1DBlock.__call__` 的两个残差分支。
- 复用原 `MlpBlock`，保留 `Encoder` 的 `nn.scan`、最终 `encoder_norm`。
- 保留 `_Module` 的 patch stem、位置嵌入、pool_type、最终 Dense head。
- `Pi0.embed_prefix` 仍从原 `self.PaliGemma.img(..., train=False)` 路径获得完整视觉 token。

实施要求：

- mode、K、merge_kind 为静态配置，每组单独 JIT/编译和预热。
- scan carry 继续是 `[B,256,D]`，仅 block 内临时出现 `[B,K,D]`。
- 保持参数名称、创建顺序、层级路径与参数 tree；不要为了方便把 scan=True 改成 False。
- 不改原 `__call__` 参数顺序，避免影响 remat/static_argnums 的含义。
- 使用固定形状 sort/top-k/gather/scatter；不引入内容相关的变长输出、Python `.item()`、动态 nonzero。
- 若保留原辅助 out 字典，在展开后记录 dense 输出，并保持 scan return tree 一致。
- 时延函数只返回生产路径实际使用的图像特征；不要把所有层辅助输出也返回，阻止编译器消除无用结果。

### 5.2 PyTorch 路线

OpenPI 当前带有 Transformers replacement。首先检查运行时实际 import 的源码位置，不要只修改一份未被加载的仓库副本。[8]

- 视觉层入口：`src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py::SiglipEncoderLayer.forward`。
- 实例范围：仅 `vision_model.encoder.layers`，不要全局替换同名类；该类还可能被文本编码器使用。
- 保留 `SiglipVisionTransformer` 的 post_layernorm。
- 保留 `PaliGemmaWithExpertModel.embed_image -> paligemma.model.get_image_features` 原路径及其 projector/缩放。

实施要求：

- `.eval()`、`torch.inference_mode()`，所有参数冻结。
- 复用原 self_attn、MLP、LN、激活函数、epsilon 和 dtype。
- `output_attentions=False`；不为了获得评分让某一组从 SDPA/Flash 路径回退 eager，评分来自已有 X。
- 若运行时有非空视觉 padding mask，明确适配或报错；不能把256×256 mask直接传给K×K attention。
- 固定配置后再 compile，禁止运行期间反复全局 monkey patch。
- 固定坐标 buffer 设为 non-persistent，避免污染 checkpoint。
- C0 与 A/B 使用相同编译预算。若 A/B 产生 graph break，保留其真实延迟并记录原因；不能只关闭 C0 编译来获得漂亮加速比。

### 5.3 self-spec 系统中的开关

若 draft 与 verifier 共享同一对象，应使用隔离的静态执行入口或已有 draft 配置机制。不要用一个全局可变 flag 在并发调用之间切换压缩，否则可能误改 verifier。验证路径的权重和视觉执行模式须保持原样。

## 6. 正确性检查

这些检查只验证实现是否符合定义，不要求压缩输出与完整模型等价。

| 检查             | 验收内容                                                     |
| ---------------- | ------------------------------------------------------------ |
| K=N identity     | g_i=i，merge/unmerge保持数值、顺序、dtype；不做匹配          |
| 小张量手算       | 合并均值包含 dst；同组结果展开到正确原索引；未合并 token 保持独立 |
| dense residual   | 同组 token 得到同一分支更新，加回后组内原差分在浮点容差内保留 |
| encoder identity | K256 改造路径与原 encoder 在相同输入/权重/后端下数值对齐     |
| policy identity  | 少量真实 observation + 相同显式动作噪声，K256与原 sample_actions 对齐 |
| shape 与接口     | 各组最终每视角 `[B,256,vlm_width]`；prefix mask/positions/各层KV长度一致 |
| 参数不变         | 原参数名、形状和值不变；禁用压缩恢复原输出                   |
| 编译稳定         | 预热后同配置不持续触发编译；A/B存在实际K长度的投影/attention/MLP计算 |

关闭压缩时应直接保留原执行路径；K256 调试模式另外覆盖改造后的分支接法。两项不能相互替代。

数值对齐默认参考：FP32 `atol=1e-5, rtol=1e-4`；BF16先以 `atol=1e-2, rtol=1e-2` 检查，并同时记录 max_abs、mean_abs、relative L2。它们是初始排错阈值，不是任意放宽的许可。原模型重复运行的数值波动应一起测量；若误差明显超过该波动，先查实现、后端和 dtype。对浮点比对不应因个别接近零的值使用不稳定的纯相对误差。

压缩配置检查无 NaN/Inf、输出形状正确即可；其特征与动作变化是待评测对象。

## 7. 视觉时延怎么测

### 7.1 主指标与计时边界

主表必须使用 **T_vision_all_ms**：本次策略调用实际执行的全部视觉编码路径，从预处理完成、已在 GPU 的图像开始，到原图像投影/head 后、将要拼接给 VLM 的视觉 embeddings 完成为止。

该指标包含：stem、位置嵌入、全部视觉 blocks、matching、merge/unmerge、原最终 norm、原投影/head。它不包含 CPU 图像解码、MuJoCo render、磁盘/网络、语言 embedding、VLM prefill 或 AE。

辅助报告：

- `T_vision_single_view_ms`：每个真实相机的单视角编码，同样包含原投影。
- `T_backbone_only_ms`：encoder 输出 norm 后、投影前的时延，可选。
- `T_model_ms`：GPU 输入就绪到动作张量计算完成；不包含环境步进。
- `T_policy_wall_ms`：真实 policy 调用的 CPU 预处理、H2D、模型、D2H、输出处理的同步 wall time；如包含网络必须另外注明。
- compile 时间单列；不计入稳定推理时延。

不能把单图时延乘相机数冒充实际多视角时延。保持 C0/A/B 原调用方式：顺序视角仍顺序、已有相机批处理仍使用同样批处理。记录有效相机数与实际处理槽位数。

### 7.2 观察样本与测量参数

- 主测量 robot batch size=1。有效相机数不等同于 robot batch size。
- 至少64份真实 LIBERO observation，四个 suite 各至少16份，尽量包含运动前、接近、接触附近与后续阶段。
- 优先复用已有观察日志；否则在 C0 的短 rollout 中采集。随机图像仅用于 smoke，不能作为唯一最终计时输入。
- 全部配置读取相同 observation manifest，图像和必要输入提前准备到 GPU。
- 每个静态配置先编译，再预热至少30次；确认无新图编译、延迟稳定。
- 正式计时5轮，每轮200次。输入顺序预先固定，循环或打乱观察库，所有配置使用相同顺序。
- 轮次内配置顺序轮换，避免 C0 永远冷机、B64 永远热机。每轮开头可短预热已编译函数。
- 不在计时期间运行其他 GPU 训练/评测、MuJoCo GPU render、视频编码或 profiler。
- 硬件上优先使用用户现有的一张4090做主计时；若选择其他GPU，所有主结果共同使用它并记录。不要把PRO6000与4090结果相除。

### 7.3 同步是必需条件

JAX 是异步派发，调用返回不表示 GPU 已完成。官方 policy 的 `policy_timing.infer_ms` 当前在把输出转为 NumPy 前结束计时，因此不可直接当成可信的同步模型耗时。[10][11]

JAX 参考结构：

```python
# jit 函数只返回实际需要的视觉输出；输入提前 device_put 并就绪。
for _ in range(warmup):
    jax.block_until_ready(vision_fn(images))

t0 = time.perf_counter_ns()
features = vision_fn(images)
jax.block_until_ready(features)
elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
```

PyTorch 的主 wall-time 结构：

```python
with torch.inference_mode():
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    features = vision_fn(images)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
```

可另用 CUDA events 测 GPU 时间，start/end 必须在实际执行 stream 上，结束 event 完成后再读取 elapsed_time；多 stream 必须有完整依赖。两种时间分开命名，不混用分母。[12]

matching、排序、FP32临时计算、gather/scatter、展开和必要 host dispatch 都属于压缩实现成本，不得移出计时窗口。

### 7.4 编译和 profiler 的公平性

- 主表采用项目实际部署使用的优化配置，C0/A/B相同。
- eager可用于排错或单独对照，不能拿 eager C0 与 compiled B 比较。
- 分别编译五种静态配置；编译时间、图数量和 graph break 另存。
- 选少量样本用 profiler 确认实际 token 维度与 matching 开销；正式计时关闭 profiler。
- 不在完整模型的每层或每阶段插同步后，把碎片时间相加冒充原完整调用延迟。
- vision独立计时与完整policy计时都保留，因为融合边界、编译和调度可能使两者无法直接相加。

## 8. LIBERO 成功率评测

### 8.1 固定任务集与环境规则

四个 suite：

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`，报告中可注明其为长任务套件

预期每套10个任务，共40个；以运行时元数据核对，禁止默默漏掉 suite。保持官方 evaluator 的图像翻转/resize、state构造、动作反归一化、gripper约定和成功判断。[3][5]

参考 evaluator 默认：`num_steps_wait=10`，`replan_steps=5`；max_steps 分别为220、280、300、520。对象稳定等待步骤及最大步数的计数方式也沿用原实现，避免 off-by-one。[5]

使用仿真提供的 task-success 判据。模型输出 NaN、非法动作或压缩引起推理失败应记为该条件失败并保留错误原因；基础设施故障可以按预定规则对整组配对 episode 重试，不能只删除压缩组失败样本。

标准 LIBERO runner 等待推理完成后才推进环境。因此本轮 SR 主要衡量视觉近似如何改变动作与任务结果，不会自动体现真实机器人在等待推理时环境继续变化的影响。实时 staleness 收益应另设协议，不能从本轮静态等待式评测中推断。

### 8.2 任务、初始状态和动作噪声配对

固定 episode key：

```text
(suite, task_id, initial_state_index, repeat_seed)
```

五个配置使用同一 episode 清单、同一个 `get_task_init_states` 中的初始状态。初始状态 index 不等于环境 seed；必须两者都记录。

默认 `repeat_seed=0`、`environment_seed=7`。每个episode在 `env.reset()` **之前**显式设置 `env.seed(environment_seed)`，随后reset、set_init_state，再执行原来的settling步骤；若当前环境API不同，使用等效的显式重置并验证。环境RNG不能依赖前一个episode长度、worker顺序或断点恢复位置。manifest与逐episode结果均保存environment_seed，同一episode key的各条件一致。

仅设置 `np.random.seed(7)` 不足以配对动作噪声。不同策略的 episode 长度不同，会改变后续全局 RNG 消耗。按下面的独立 key 生成显式高斯噪声：

```text
(root_seed, suite, task_id, initial_state_index, repeat_seed, replan_index, stream_tag)
```

主消融的 `stream_tag` 固定为 `policy`，不包含 config_id，从而各条件同 episode、同 replanning 次数使用相同噪声。使用稳定哈希或 SeedSequence，不能使用受进程随机化影响的 Python hash()。噪声形状从实际 `action_horizon × action_dim` 读取，必要时加batch维。

官方 `Policy.infer(obs, noise=...)` 支持显式噪声，但原 WebSocket `infer(obs)` 不会自动把 observation 内的 noise 字段转成这个 keyword。应实现评测 adapter 或使用 in-process 调用，并做一次配对噪声校验；不要只把 noise 塞进 obs 后认为已经生效。[10]

不同条件闭环观测不同是预期现象；配对的是初始条件和随机数流，不能强行回放 C0 的所有观测来冒充闭环评测。

### 8.3 三阶段预算

**阶段0：smoke。**

每个 suite 固定1个任务、1个初始状态。C0/A128/A64/B128/B64各跑4个episode，共20个。用于确认环境与策略运行，结果不作为最终SR结论。

**阶段1：筛选集。**

40个任务，每任务使用初始状态索引0–4，5个episode。每配置200个，五配置合计1000个。可复用前述smoke中完全相同的episode，不重复计数。

先完成五组的视觉计时和筛选SR，再从A家族选最多一个、B家族选最多一个进入确认。选择规则固定为：

1. 必须先通过实现正确性检查。
2. 优先选择视觉确有加速、筛选相对SR至少0.70的配置；家族内选视觉加速更高者。
3. 若该家族没有0.70候选，则可选择相对SR在0.60–0.70且有加速的最快配置，标为激进候选。
4. 在满足同一SR门槛的候选中，若两个加速比的相对差 `abs(S128-S64)/max(S128,S64) <= 0.02`，选K128；否则选更快者。这是确定性的工程取舍，不是统计显著性判断。记录选择依据，不事后改规则。
5. 如果全部无加速或出现严重退化，报告负结果；按第12节做至多一次针对性排错/补救，再决定是否值得投入确认预算。

以上阈值只是本次工程筛选规则，来源于用户愿意接受60–70%相对SR的目标，不是论文定理。

**阶段2：独立确认集。**

默认每任务使用初始状态索引5–49，共45个。C0和选中的A/B各1800个episode，三配置合计5400个。这样与筛选集不重叠，避免反复调参后再在同一批episode上报告最终效果。

默认主矩阵预算上限为1000次筛选 + 5400次确认 = **6400个episode**，smoke若来自筛选集则不另加。只有一个家族入围时确认是3600；都不入围则不启动长评测。第12节的可选补救每个新增配置另加200个筛选episode，最多另加400；含两项补救时总计最多6800。

如果所有任务都有至少55个可用初始状态，可在开始筛选前统一将确认集预登记为5–54，即每任务50个、每配置2000个、三配置6000个，主矩阵总计7000个episode；若另有两项补救，则最多7400。不得跑完后临时扩展到有利结果。若想严格复现官方前50初始状态结果，另跑0–49并明确其含筛选样本，不能称为独立确认集。

如果资源只能完成筛选，应交付“筛选结果”，标明每配置200个episode与不确定性，不把它包装成正式确认。

筛选完成后，用实际每episode的policy查询数、模拟器耗时和推理耗时估算确认集预算。一个episode包含多次策略推理，不能用“episode数×一次推理时延”估算总运行时间。先输出预计GPU占用与wall time，再使用现有可用GPU按既定预算执行。

### 8.4 多卡分工与恢复

- 4×4090可以按任务/episode分片提高rollout吞吐；每个episode独立seed，不依赖worker执行顺序。
- 同一组配对episode尽量分配到同一块GPU，以减少跨卡数值/版本差异。
- 正式模型时延在独占的一张卡上另测，不能把多进程并行rollout时延混入主表。
- 每完成一个episode立即追加JSONL并flush；唯一键包含config_id和episode key。
- 重启时按完整唯一键跳过已完成项，检查清单无重复、无缺失。每个episode重置动作队列与相关缓存。
- 视频保存统一的小样本及失败案例，异步或在计时外写盘；不因开启录制改变某一条件的计时。

## 9. 指标、区间与判定

### 9.1 时延指标

各组报告 p50、p95、均值、样本数、五轮各自统计。主加速比：

\[
S_{\mathrm{vision}}=\frac{\operatorname{p50}(T_{\mathrm{vision},C0})}{\operatorname{p50}(T_{\mathrm{vision},c})},
\qquad
\text{latency reduction}=1-\frac{T_c}{T_{C0}}.
\]

不要把“2×加速”写成“时延减少200%”；2×对应时延减少50%。

保留相同观察顺序和轮次，通过配对的round/observation分块bootstrap估计加速比区间，重采样时保留同条件间配对。不能把GPU连续重复调用当作大量独立环境样本来夸大置信度；同时报告五轮最小/最大值和跨轮稳定性。

### 9.2 成功率与影响

先计算每个任务SR，再按任务等权求suite与四suite平均。每任务episode数相同时与直接汇总一致；出现缺失时不得无声改变权重。

\[
\Delta SR_{\mathrm{pp}}=100(SR_c-SR_{C0}),
\qquad
R_{\mathrm{SR}}=SR_c/SR_{C0}.
\]

同时报告绝对百分点变化和相对保留率。例：0.96→0.67是下降29个百分点、保留69.8%，不是“下降29%”。这只是算式示例，不能写入实测表。

SR和ΔSR默认使用10,000次任务内分层、episode配对bootstrap：固定40个任务，在每任务内部重采样episode key，同一个key对应C0/A/B的结果一起采样，再按任务等权聚合。报告95%区间。这里的区间针对当前固定任务集合，不代表对所有机器人任务的泛化保证。若C0成功数为0，相对保留率记为NA。

确认集至少输出：每suite SR、总SR、ΔSR百分点及区间、相对保留率、总成功数/总episode数。不能只选容易的任务报告平均值。

### 9.3 实用判定

- **确有视觉加速**：正确性检查通过、实际T_vision下降，区间与多轮数据支持，且没有通过改变backend/精度偷换基线。
- **工程上值得继续**：优先关注 `S_vision ≥ 1.20` 的配置；1.20是本次可调整的工程目标，不是成功定义的自然常数。
- 分别列出满足相对SR≥0.90、≥0.70、≥0.60的最快已测候选；不满足的门槛写“无候选”。
- 必须额外报告完整policy时延。视觉加速明显但policy收益有限，是可接受的真实结果，不应隐藏。

若视觉原占完整推理时间比例为f，视觉加速为s、其他部分不变，估算总体加速为1/((1-f)+f/s)。这只能作为检查趋势的估算，实测完整policy仍是依据。

## 10. 接入现有 self-speculative 系统

只有已有可运行系统时做本节，不为本实验重新设计 draft/verify。

固定已有VLM/AE裁剪、FM步数、周期、接受准则与回退策略，分别运行S0、SA、SB。优先用相同筛选集200个episode/条件做初测；需要正式系统结论时再使用同一独立确认清单。该预算单列，不包含在第8节的6400中。

额外字段：`draft_ms`、`verify_ms`、`fallback_ms`、`cycle_wall_ms`、`drafted_actions`、`committed_actions`、`fallback_count`、`vision_refresh_count`。接受动作数按算法真实commit定义，不能直接用action horizon代替。

如果同一观测的完整视觉原本由draft/verifier共享，只给draft另跑廉价视觉可能增加总计算。保留当前共享方案作为S0，记录新方案是否破坏共享、额外计算了多少次视觉或prefix，不只看draft函数自身快了多少。

不得把近似视觉特征冒充原完整verifier特征来复用。若verifier采用的是算法原本允许的旧条件，也保持其原刷新规则，不为A/B单独更改。

若当前系统尚不可运行，完成第一层并把系统SR写为“未评测”，不得用独立策略SR填充系统SR列。

## 11. 执行入口与建议文件

### 11.1 原仓库基线入口

以下是官方OpenPI参考启动方式；实际执行先检查当前版本的 `--help` 和项目文档。[3]

```bash
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir <EXISTING_CHECKPOINT_PATH_OR_URI>
```

官方客户端为 `examples/libero/main.py`。它本身不包含本方案的配置sweep、独立初始状态清单、显式噪声传递或统计汇总；coding agent需要补充评测适配，不能只改一个CLI参数就宣称实现完整协议。

### 11.2 需要新增或适配的入口

下面是**要求coding agent实现的建议接口，不是声称仓库已经存在的命令**。若项目已有等效脚本，复用它们并在报告写出真实命令。

```text
experiments/pi05_vision_ab/
  config.yaml
  run.py                 # audit / check / collect_obs / bench / screen / confirm / system / report
  compression.py         # 视后端可放回模型目录，保持清晰的开关与映射实现
  evaluation_adapter.py  # episode清单、噪声注入、结果追加和恢复
  summarize.py
```

建议运行顺序：

```bash
python experiments/pi05_vision_ab/run.py --config experiments/pi05_vision_ab/config.yaml --stage audit
python experiments/pi05_vision_ab/run.py --config experiments/pi05_vision_ab/config.yaml --stage check
python experiments/pi05_vision_ab/run.py --config experiments/pi05_vision_ab/config.yaml --stage collect_obs
python experiments/pi05_vision_ab/run.py --config experiments/pi05_vision_ab/config.yaml --stage bench
python experiments/pi05_vision_ab/run.py --config experiments/pi05_vision_ab/config.yaml --stage screen
python experiments/pi05_vision_ab/run.py --config experiments/pi05_vision_ab/config.yaml --stage confirm
python experiments/pi05_vision_ab/run.py --config experiments/pi05_vision_ab/config.yaml --stage report
```

每阶段要检查前置产物；例如没有正确性检查通过的记录，不得开始长rollout。不得把占位符当成实际checkpoint路径。

### 11.3 配置模板

```yaml
schema_version: 1
experiment_name: pi05_vision_ab

model:
  config_name: pi05_libero
  checkpoint: null                 # audit阶段解析现有checkpoint；运行前必须填实
  backend: existing               # 解析为jax或pytorch并写入manifest
  dtype: existing
  compile_mode: existing
  frozen_weights: true

policy:
  batch_size: 1
  num_steps: 10
  action_horizon: from_model_config
  action_dim: from_model_config
  replan_steps: 5
  camera_call_pattern: unchanged

compression:
  merge_kind: tome_deterministic
  layer_scope: all
  expected_grid: [16, 16]
  destination_cell: [2, 2]
  anchor: top_left
  score_dtype: float32
  accumulation_dtype: float32
  norm_epsilon: 0.000001
  tie_break: original_index
  mapping_per_block: true
  reuse_map_for_mlp: true
  restore_after_each_branch: true
  preserve_dense_residual: true

conditions:
  - {id: C0, mode: "off", keep_tokens: 256}
  - {id: A128, mode: A, keep_tokens: 128}
  - {id: A64, mode: A, keep_tokens: 64}
  - {id: B128, mode: B, keep_tokens: 128}
  - {id: B64, mode: B, keep_tokens: 64}

benchmark:
  real_observations_min: 64
  warmup_min: 30
  rounds: 5
  measured_calls_per_round: 200
  input_residency: gpu
  synchronized_wall_time: true
  cuda_event_time_optional: true
  rotate_condition_order: true

evaluation:
  suites: [libero_spatial, libero_object, libero_goal, libero_10]
  root_seed: 7
  environment_seed: 7
  repeat_seed: 0
  reset_environment_rng_each_episode: true
  smoke_tasks_per_suite: 1
  smoke_trials_per_task: 1
  screen_init_indices: {start: 0, stop_exclusive: 5}
  confirm_init_indices: {start: 5, stop_exclusive: 50}
  paired_action_noise: true
  noise_key_uses_config_id: false
  max_candidates_per_family: 1
  task_macro_average: true
  paired_stratified_bootstrap_samples: 10000

selection:
  preferred_relative_sr: 0.70
  aggressive_relative_sr_floor: 0.60
  engineering_vision_speedup_target: 1.20

system_evaluation:
  run_if_existing_harness_available: true
  change_draft_vision_only: true
  preserve_verifier_and_acceptance: true
```

`existing`/`from_model_config`是配置解析要求，不是可直接交给模型的值。audit阶段必须解析、验证并输出完整resolved config。若用户现有已确认基线使用不同num_steps/replan_steps，统一覆盖所有条件并记录原因。

## 12. 失败定位和有限补救

按顺序诊断，每次只改变一个因素，并保留原始结果。

1. **C0自身异常**：先核checkpoint、归一化assets、图像翻转、动作后处理、state编码、horizon和环境版本。C0无法正常工作时不能把问题归因于A/B。
2. **K256不等价**：查LN顺序、残差是否被合并、实际源码是否加载、是否漏head或scaling、dtype/attention backend是否改变。
3. **tokens减少却不加速**：检查排序/gather/scatter耗时、host同步、graph break、显式attention matrix、GPU占用。只进行一次有明确证据的实现修正，再重测C0与相应配置。
4. **attention-only收益小**：这可能是正常结果。按N=256、d=1152、MLP宽4304估算，MLP约占主要矩阵计算量63%；A不压MLP，不能期待其与B一样的省算。[6]
5. **B质量下降很大**：先排除实现错误。ToMeSD默认未启用MLP压缩，不能假设该近似低风险。[1][2]

可选补救最多增加每家族一个配置：在同一K、同一匹配算法下保留最前2层和最后2层完整计算，27层时仅压缩索引2–24，即 `[2,25)` 的23层。必须以新config_id记录，重新计时并在筛选集上选择；不能覆盖原all-layer结果。若确认集已经用于看结果，再修改算法后需新的确认清单，否则将结论标为探索性。

本轮不无限搜索层组合、不新增学习模块、不训练恢复精度。

## 13. 必须交付的原始数据与报告

建议输出目录：

```text
results/pi05_vision_ab/<run_id>/
  manifest.json
  resolved_config.yaml
  environment.txt
  implementation_diff.patch
  correctness.json
  observations_manifest.jsonl
  episode_manifest_screen.jsonl
  episode_manifest_confirm.jsonl
  latency_raw.csv
  latency_summary.csv
  rollouts_screen.jsonl
  rollouts_confirm.jsonl
  success_by_task.csv
  success_by_suite.csv
  summary.csv
  selection.json
  commands.sh
  report.md
  figures/
  videos/
```

`latency_raw.csv`至少包含：run_id、config_id、backend、compile_mode、GPU、round、iteration、observation_id、measurement_scope、wall_ms、optional_device_ms、actual_camera_slots、valid_camera_count。不同计时scope分行，不能把single_view与all_views混为同一列。

逐episode JSONL至少包含：config_id、stage、suite、task_id、initial_state_index、environment_seed、repeat_seed、action_noise_seed_rule、success、environment_steps、policy_calls、timeout、error_type、video_path、代码与checkpoint指纹。system结果另加draft/verify/commit字段。

### 主结果表

| 配置 | 编译/后端 | 视觉p50 ms | 视觉p95 ms | 视觉加速比 | 完整policy p50 ms | SR及95%区间 | ΔSR百分点及区间 | 相对SR | 成功/总episode |
| ---- | --------- | ---------: | ---------: | ---------: | ----------------: | ----------- | --------------- | -----: | -------------- |
| C0   | 待测      |          — |          — |       1.00 |                 — | —           | 0               |   1.00 | —              |
| A128 | 待测      |          — |          — |          — |                 — | —           | —               |      — | —              |
| A64  | 待测      |          — |          — |          — |                 — | —           | —               |      — | —              |
| B128 | 待测      |          — |          — |          — |                 — | —           | —               |      — | —              |
| B64  | 待测      |          — |          — |          — |                 — | —           | —               |      — | —              |

筛选与确认必须分表，未进入确认的配置不能填入筛选SR却不标阶段。

必须生成三类图：

1. 各配置T_vision_all的p50及跨轮变化，明确单位ms。
2. 视觉加速比—闭环SR散点图，标明A/B与K；筛选点与确认点使用不同标记。
3. 四suite的SR变化图，显示是否某套件或长任务特别敏感。

报告结论必须直接回答：

- A和B分别使视觉encoder快多少，节省多少ms，测量边界是什么？
- 最终SR下降多少个百分点，保留原SR多少，区间和样本数是多少？
- 哪个配置在≥90%、≥70%、≥60%相对SR门槛下最快？
- 完整policy是否同步加速？若没有，剩余瓶颈在哪里？
- 若做了self-spec系统评测，净收益与fallback/acceptance如何变化？
- 是否存在未解决的graph break、计时污染、数据缺失或评测范围限制？

## 14. 完成标准

- [ ] C0及A/B四组定义明确，只有视觉残差分支计算发生变化。
- [ ] 没有训练、没有修改权重，没有改变VLM/AE输入布局与位置处理。
- [ ] identity、映射与参数检查通过。
- [ ] 至少64份真实观察、5轮同步计时，包含所有压缩开销。
- [ ] 四个LIBERO suite全部进入预先固定的评测清单。
- [ ] 初始状态与动作噪声配对，逐episode结果可追溯。
- [ ] 筛选与确认分离，报告每配置episode数量，确认集不用于反复调参。
- [ ] 提供视觉加速、完整policy时延、最终SR、百分点损失与相对保留率。
- [ ] 不加速/低成功率配置保留，所有未测项明确标记。
- [ ] 交付真实命令、resolved config、代码diff、CSV/JSONL和图表。

## 15. 参考文献与源码

以下是机制或实现依据；超参数、预算、筛选阈值和统计流程是本实验的预设设计，并非这些来源已经验证了π0.5 A/B的结果。

1. Bolya & Hoffman. Token Merging for Fast Stable Diffusion. 2023. 重点§3.1–3.2、§4.2及Table 3。
   https://arxiv.org/html/2303.17604v1
2. ToMeSD 官方代码，残差接法、默认MLP开关、匹配与展开。
   https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py
   https://github.com/dbolya/tomesd/blob/main/tomesd/merge.py
3. OpenPI LIBERO 官方说明与checkpoint。
   https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md
4. OpenPI `pi05_libero` 配置；核对实际horizon及state设置。
   https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
5. OpenPI LIBERO evaluator；套件、初始状态、replan与终止规则。
   https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py
6. OpenPI JAX SigLIP；视觉block与最终head。
   https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/siglip.py
7. OpenPI π0/π0.5；视觉接入、prefix、动作采样。
   https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
8. OpenPI PyTorch SigLIP replacement与图像embedding路径。
   https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py
   https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/gemma_pytorch.py
9. OpenPI LIBERO输入适配与相机mask。
   https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py
10. OpenPI Policy.infer，显式noise支持与计时位置。
    https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/policy.py
11. JAX官方异步派发与同步计时说明。
    https://docs.jax.dev/en/latest/async_dispatch.html
12. PyTorch官方CUDA Event说明。
    https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html

源码链接指向main，执行时以本地固定commit为准并记录。不要把本文记录的当前默认配置覆盖到用户已有且验证过的不同checkpoint设置上。