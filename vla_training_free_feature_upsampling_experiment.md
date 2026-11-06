# VLA 视觉特征无训练上采样：技术路线与实验计划

> 版本：2026-10-17 · 可直接交给 Coding Agent 执行
>
> 目标模型：优先在 **openpi 的 π0 / π0.5 所用 SigLIP So400m/14** 上验证。严格限定 **Training-Free**：不训练新 Adapter、不蒸馏、不更新预训练权重。允许固定超参数、插值、经典滤波、使用已有 SigLIP Patch Embedding 权重；将“逐图测试时优化”单列为扩展，不纳入实时主线。
>
> **优先级：先验证特征重建是否有意义，再接完整 VLA；先测真实 GPU latency，再讨论理论 FLOPs。**

## 0. 核心问题与预期产物

### 研究问题

是否可以把 224×224 RGB 降至 112×112，让 SigLIP 只对 **64 而不是 256 个 Patch Tokens** 执行 27 层 Transformer，再用**不需训练的特征重建算法**产生 256 个视觉 Tokens，让下游 VLM / Action Expert 的输入接口保持不变？

```text
教师路径（原始）：RGB224 -> SigLIP(16x16=256) --------------------> F_teacher[256,D]
学生路径（目标）：RGB224 -> Resize112 -> SigLIP(8x8=64) -> U_x4 -> F_pred[256,D]
                         \-------------------------------- RGB224 / PatchEmbed224 作引导
```

> 这里的“×4”指 Token 个数 64→256，空间网格是 8×8→16×16。**不承诺无损重建**；SigLIP 的全局注意力和非线性会让高低分辨率特征显著不同。

**最终交付**：可运行脚本、参数化配置、各路线的特征误差和耗时 CSV、可视化、实验日志、一页结论报告；如果环境具备 π0 与 LIBERO，再给同一观测下的 Action Difference / 成功率对照。

## 1. openpi 实现约束（务必先核对本地 checkout）

官方参考：[openpi `siglip.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/siglip.py)、[`pi0.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)、[`models_pytorch/pi0_pytorch.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)、[`models/model.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/model.py)。

- JAX 的 `pi0.py` 通过 `_siglip.Module(num_classes=paligemma_config.width, variant="So400m/14", pool_type="none")` 创建图像模型；`embed_prefix()` 直接取 `self.PaliGemma.img(image)[0]` 作为视觉 Token。
- 对标准 224×224 图像，Patch 14×14，因此输出 16×16=256；So400m 的隐藏宽度为 1152、27 个 Transformer blocks，但**最终喂给 PaliGemma 的图像特征维度是 `paligemma_config.width`，不一定为 1152**。先读实际 tensor shape，绝对不要硬编码 `D=1152` 到最终特征。
- 当前 JAX `siglip.py` 使用 **learned `pos_embedding`**：默认参数 shape 对应 16×16。未经修改直接跑 112×112 可能因为位置嵌入 shape/参数树不匹配而报错。需从已加载 checkpoint 的 `16×16` 位置参数取出，reshape 为 `[1,D,16,16]`，双三次插值到 `[1,D,8,8]`，再以 `[1,64,D]` **仅在该分支前向时**使用。不可让框架重新初始化一个随机 8×8 位置参数。建议额外测试 `nearest/area` 位置下采样作为稳健性消融。
- 若使用 PyTorch 版本，确认 `get_image_features` / HF SigLIP 的 image_size 断言、位置嵌入及多模态 projector；禁止把 JAX 的参数接口直接套到 PyTorch。保持加载原 π0 权重后修改**前向路径**，尽量不修改 checkpoint 文件。
- `openpi` 对图像存在预处理与 normalize，务必确保：传给 SigLIP 的输入真的为 112×112，**不会被另一处 preprocess 静默 resize 回 224×224**。用于 RGB 引导的原图则单独保存为 `[0,1]` 浮点 RGB，避免与 SigLIP 的标准化图像混淆。
- π0 会循环处理多个摄像头，实验先采用单相机 / 单张图像；全链路测试要对每个相机相同处理，并检查 image masks、flatten 顺序、Token 位置和 prefix KV 长度。
- **必须验证 baseline consistency**：未改动的 224 分支，通过实验 hook 得到的特征与原模型直接输出 `allclose`（FP32）或误差在 BF16 数值容差内；否则后续比较无效。

## 2. 候选路线（全部不训练新参数）

### A. 插值基线：Nearest / Bilinear / Bicubic / Lanczos（P0）

**机制**：SigLIP 低分辨率输出 `F_low[1,D,8,8]` 直接插到 `[1,D,16,16]`。PyTorch `F.interpolate` 支持最近邻、双线性、双三次；Lanczos 可用图像处理实现，但跨后端 / 高通道性能先验证，不强求首日完成。

- **优点**：没有新参数，通常很便宜，必要对照组。
- **局限**：只重排 / 平滑 64 个 Token，不生成新的独立语义方向；不同位置的特征可能过于平滑。
- **实施**：`align_corners=False`（适用时）；`bicubic` 可能过冲，严查 NaN/数值范围。Feature 本身不是 RGB，**不要先裁剪到 [0,1]**。
- **对照**：Nearest、Bilinear、Bicubic 必跑；可选 Lanczos。

### B. RGB 引导的 Joint Bilateral Upsampling（JBU，P0）

经典论文：Johannes Kopf 等，[**Joint Bilateral Upsampling**](https://www.microsoft.com/en-us/research/publication/joint-bilateral-upsampling/)，SIGGRAPH/ACM TOG 2007。

**机制**：对每个高分辨率位置 `p`，读取原始 RGB 图像在该位置的引导颜色，遍历相邻 3×3 或 5×5 个低分辨率 Feature 位置 `q`，用空间距离与 RGB 相似度计算无参数权重：

\[
\hat F(p)=\frac{\sum_{q\in\mathcal N(p)}w(p,q)F_{\mathrm{low}}(q)}{\sum_{q\in\mathcal N(p)}w(p,q)},\qquad
w(p,q)=\exp\left(-\frac{\lVert x_p-x_q\rVert^2}{2\sigma_s^2}-\frac{\lVert C_p-C_q\rVert^2}{2\sigma_r^2}\right).
\]

- `C_p`：原始 224 RGB **区域池化或对齐采样到 16×16**，而非用 224 个原像素逐点对应；`C_q`：相同原图区域池化到 8×8 后，在低分辨率位置取值。先确定图像坐标与 Patch 中心一致（half-pixel convention）。
- `σ_s` 单位：8×8 低分辨率网格像素；建议试 `[0.75,1.5]`。若 `C ∈ [0,1]`，`σ_r` 建议试 `[0.1,0.25]`。这些仅为**扫描起点**，不是已验证最优参数。第一天先固定 `(1.0,0.15)`。
- 提供 `spatial-only` 对照（即关掉颜色项），确保 JBU 改善确实来自 RGB 引导；另对比 3×3 与 5×5。
- 实现建议：先 PyTorch `unfold / gather + softmax`，验证正确；测延迟时单独注明是否是未优化版本，避免因为不必要的 `[B,D,9,16,16]` materialization 误判方法不可行。可以后续写 fused CUDA / Triton kernel。
- **风险**：RGB 的颜色边界不等同于深层 SigLIP 的语义边界，语义特征可能被不恰当地切断。

### C. Guided Filter / Fast Guided Filter（P1）

参考：Kaiming He 等，[**Guided Image Filtering**](https://mlanthology.org/eccv/2010/he2010eccv-guided/)，ECCV 2010；[**Fast Guided Filter**](https://arxiv.org/abs/1505.00996)，2015。

**机制**：假设在局部窗口内特征通道与引导图（例如 RGB 或亮度）存在线性关系，估算局部系数，并上采样系数得到高分辨率输出。它是经典的确定性边缘感知滤波，不需要神经网络。

- 先用灰度引导或 RGB 三通道引导；对每个 `D` 的 Feature 通道应用**共享 RGB 统计量**，减少重复计算。
- 推荐 PyTorch/GPU 实现；不能直接调用某些 OpenCV API 对 1152/2048 通道做原样处理（部分 API 仅允许少量通道且常引入 CPU 拷贝）。
- 核心消融：窗口半径 `[1,2]`、`eps` 从 `1e-3` 到 `1e-2`（需随 RGB 归一化范围调整）。
- 相较 JBU，算法假设更强、实现也可能更复杂；首日可以跳过，不要影响 P0 交付。

### D. 使用原始高分辨率 Patch Embedding 引导插值（P0，探索性创新假设）

**重点**：不让昂贵的 27 层 Vision Transformer 处理 256 Tokens，但允许原始 **Patch Embedding Conv** 在 224×224 上执行一次（通常远便宜于整条 Transformer）。

```text
RGB224 ---SigLIP PatchEmbed---> Z_high [16,16,1152]
                        \--2x2 Pool--> Z_coarse [8,8,1152]
RGB112 ---SigLIP 27-layer---> F_low [8,8,D]
                           \---- Local weighted upsampling, weights guided by Z_high/Z_coarse
                                     -> F_pred [16,16,D]
```

局部权重不使用 RGB 距离，而用预训练 Patch Embedding 的余弦相似度：

\[
w_{pq}=\operatorname{softmax}_{q\in\mathcal N(p)}\left(\frac{\cos(Z^{hi}_p,Z^{coarse}_q)}{\tau}-\lambda\lVert p-q\rVert^2\right),\quad
\hat F_p=\sum_q w_{pq}F_{low,q}.
\]

- 对 `Z_high` 与 `Z_coarse` **不要使用额外随机投影或需要学习的 Q/K**；两者通道维度同为 SigLIP patch width，先 L2 normalize 后 cosine。
- `τ ∈ {0.05,0.2}`，`λ ∈ {0.5,2}` 先少量扫描。也可对 `Z_high` 先去掉均值，观察对 embedding 相似度的影响。
- **必须让 `Z_coarse` 来自 `Z_high` 的 pool**，不要直接假设来自低分辨率 RGB 的 Patch Embedding 与高分辨率 Patch Embed 分布一致。
- 权重只从 `Z` 中计算，数值只从 `F_low` 中取，**不会出现 1152→D 的通道不匹配**。
- 这是一个**未被证实、需要消融的原创候选**，不能把它写成已发表方法或保证重建结果更好。

### E. 高分辨率 Patch Residual / 高频细节旁路（P2，较高风险）

思想：

\[
\Delta Z=Z_{hi}-U\bigl(\operatorname{Pool}_{2\times2}(Z_{hi})\bigr),\qquad
\hat F=U(F_{low})+\alpha\,T(\Delta Z).
\]

- 可以利用 **已有模型权重** 做 `T` 的维度映射（例如当前视觉分支使用的多模态 projector），**但必须先确认输入语义与正则化兼容**；不得新建随机 linear 层。
- Patch Embedding 的早期纹理残差与 SigLIP 的深层语义特征不在同一分布，直接相加风险很高。要先输出 `alpha=0` baseline，再试非常小的系数和 RMS 对齐；`alpha` 的选择不应根据 LIBERO 测试集逐个任务调参。
- 仅当 D 方法带来可观改善，且还有时间时尝试；不建议首日优先。

### F. 不属于严格纯算法主线的文献对照（仅调研，不默认跑）

| 工作 | 论文/代码 | 方法性质 | 与本实验的关系 |
|---|---|---|---|
| **FeatUp**（ICLR 2024） | [Paper](https://arxiv.org/abs/2403.10516) · [Code](https://github.com/mhamilton723/FeatUp) | 包含**训练得到的** feedforward upsampler，以及单图拟合的隐式版本；虽然其内部借鉴 JBU，**整套 FeatUp 不能算纯算法无训练** | 理解 feature upsampling 与 RGB 引导 |
| **AnyUp**（ICLR 2026 Oral） | [Project](https://wimmerth.github.io/anyup/) · [Code](https://github.com/wimmerth/anyup) | 已经经过跨 backbone 训练，推理时无需针对新编码器训练；**不满足“完全无学习模块”** | 作为放宽约束后的质量上界候选 |
| **NAF**（CVPR 2026） | [Paper](https://arxiv.org/abs/2511.18452) · [Code](https://github.com/valeoai/NAF) | 邻域注意力引导，但使用**预训练的上采样网络** | 同上，不要误写为纯数学算法 |
| **Upsample Anything**（2025 arXiv；作者页面标注 CVPR 2026） | [Paper](https://arxiv.org/abs/2511.16301) · [Code](https://github.com/seominseok0429/Upsample-Anything_Pytorch) | **不依赖数据集训练权重**，但**逐图优化**各向异性高斯核；论文报告约 0.419 秒/224 图像（设备/实现须另行确认） | 符合不训练 Adapter，但不适于高频实时主线；离线质量对照可选 |

> **重要区别**：`无需为 SigLIP 重新训练` ≠ `不含预训练 upsampler` ≠ `无需逐图优化`。本实验的 A/B/C/D 是严格的固定算法和已有编码器权重；E 仅复用已有权重；F 单列。

## 3. 核心消融：区分“编码损失”与“上采样损失”

设原始高分辨率教师特征为 `T = SigLIP224(I)`（16×16，最终 projector 后），低分辨率学生为 `L = SigLIP112(Resize(I))`（8×8，同一 projector 后）。

**两种数据源必须同时测量：**

- **Oracle-downsample**：`L_oracle = AvgPool2x2(T)`；先从理想教师特征压成 64 个，然后用每种 U 恢复到 256。这用于测量“纯空间压缩/上采样本身的损失”。它在真实推理时**不可用**，只用于分析；不要把其耗时算进最终提速。
- **Real-lowres**：`L_real = SigLIP112(Resize(I))`；真正加速的路径。这里的误差既包含下采样信息损失，也包含 SigLIP 全局 attention、不同 token 数、位置编码变化导致的 **representation shift**。

若 Oracle-downsample 都重建得很差，说明现有 U 过弱；若 Oracle 很好而 Real 很差，重点瓶颈可能不是 U，而是**改变输入分辨率使 SigLIP 特征分布发生偏移**。

再加一个独立对照：**`Low64-no-U`**，直接把 64 tokens 交给 VLM（需要改下游序列长度及 attention masks）。它的下游计算更低，不能只因为“Token 数没恢复”就排除。

## 4. 实验设计与执行顺序

### 4.1 Phase 0：准备、基线与形状检查（P0）

- 固定 checkpoint、运行后端（JAX 或 PyTorch）、dtype（BF16/FP32）、硬件、软件版本、随机种子，记录 git commit。
- `eval()` + `no_grad()`；图像以同一预处理方式生成 `RGB224`；保存 `RGB112` 和两种 RGB 引导图。
- 实际打印并保存：`teacher.shape`、`student.shape`、`patch_high.shape`、`pos_embedding.shape`、视觉输出是否含 CLS、最终与语言嵌入拼接的维度。
- 让 `224x224` 特征 hook 与原始模型 `embed_image` 输出一致；确认 `112x112` 的前向**确实只运行 64 tokens**，模型无 shape error、无随机新位置权重、没有静默 resize。
- 检查 `pos_embedding` 插值方案并存档：默认 bicubic + `align_corners=False`。
- 拿 10 张图像肉眼检查 resize 是否发生边界裁切、颜色空间错误和 camera 顺序交换。

### 4.2 Phase 1：离线 Feature 重建（P0）

**数据建议**：使用可访问的 LIBERO 观测帧或已有的机器人手眼相机图片。优先 200～500 张、覆盖不同任务和摄像头。避免大量连续近重复帧；同一 episode 抽帧，按 episode 汇总，保证不被单条轨迹主导。没有 LIBERO 的话，先用 20 张本地图像 smoke test，再接正确分布的数据。

**默认实验矩阵：**

| ID | 特征源 | 恢复策略 | 是否利用高分辨率 RGB/Patch 信息 | 优先级 |
|---|---|---|---|---|
| T0 | Teacher 224 | 无 | 原始 Teacher | 必须 |
| L0 | Real Low112 | 不上采样（64 Tokens） | 否 | 必须（主要用于下游） |
| I1 | Real Low112 | Nearest ×2 | 否 | 必须 |
| I2 | Real Low112 | Bilinear ×2 | 否 | 必须 |
| I3 | Real Low112 | Bicubic ×2 | 否 | 必须 |
| J1 | Real Low112 | JBU 3×3 | RGB224 引导 | 必须 |
| P1 | Real Low112 | Patch-guided 3×3 | 高分辨率 Patch Embedding | 必须 |
| O1 | Oracle Pool(T) | Bilinear | 不额外引导 | 必须（诊断用） |
| O2 | Oracle Pool(T) | JBU | RGB224 引导 | 必须（诊断用） |
| O3 | Oracle Pool(T) | Patch-guided | Patch 引导 | 可选 |
| G1 | Real Low112 | Guided Filter | RGB224 引导 | 可选 |
| R1 | Real Low112 | Patch detail residual | 高分辨率 Patch Embedding | 延后 |

**特征评价**（所有特征先按相同 raster scan 顺序变为 `[B,256,D]`；不同方法不得重新排序）：

1. `cos_mean`: 全部对应位置 Token 的平均 cosine similarity：`mean_i cos(F_pred[i], T[i])`，另报 median / p10。
2. `rel_l2`: `||F_pred - T||_F / (||T||_F + 1e-8)`。
3. `norm_ratio`: `mean(||F_pred_i||_2) / mean(||T_i||_2)`；观察输出缩放 / distribution shift。
4. `spatial_delta`: 邻接位置差分与 Teacher 的差别（检查插值是否过平滑）；有条件可画 feature PCA/RGB 可视化。
5. **不要只看 cosine**：向量方向相似不说明幅值、下游注意力响应或动作不变。

**额外建议：分层输出**：有精力时在 SigLIP 中间层（例如 blocks 0 / 8 / 17 / 26）各采一份，分别插值后与对应教师层比较；但最终要以 **projector 后、真正送入 VLM 的特征** 为主要指标。若 checkpoint hook 很麻烦，先只测最终输出。

### 4.3 Phase 2：真实延迟与吞吐（P0）

**至少报告四部分 wall-clock / CUDA event 延迟（ms）：**

1. `T_encoder_224`：原始 SigLIP224 视觉路径。
2. `T_encoder_112`：小图 SigLIP112（包含必要的 resize/位置处理；把 resize 单独计时）。
3. `T_upsampler`：各 U 的额外计算（JBU RGB 引导、Patch-guided 的高分辨率 PatchEmbed 都要计入）。
4. `T_end2end`：可以接 VLA 时，完整一次 `sample_actions`；拆出视觉、VLM prefix、AE denoise 的开销，不能把 feature cache 命中导致的加速误记为上采样收益。

实验准则：

- 固定 batch size=1 为主，额外 B=4 可选；固定模型、分辨率、dtype、设备；JAX 与 PyTorch 不交叉比速度。
- 预热至少 30 次，测至少 100 次或直到稳定；GPU 用 CUDA Events / JAX `block_until_ready()`；记录 median/p90、peak VRAM、是否开启 compile / CUDA Graph，以及**是否包含初次编译**（首次编译单独报告、不计 steady-state）。
- 对每条路径 **包含** 必要 GPU 数据转换、引导特征构建、重排、Upsampler；再给一个分模块 breakdown。拒绝仅以 FLOPs 或 encoder-only 结论宣称端到端 4×。
- 资源允许时对两种硬件做同口径复现：本地 RTX 4090 24GB、远端 RTX PRO 6000 96GB。**不能直接把两卡不同软件栈的毫秒数并排解释为算法性能。**

**判读标准（研究用探索阈值，不是文献结果）：**

- 先看 `encoder112+upsampler` 是否比 `encoder224` 真实更快（例如 ≥1.5× 才有明显工程价值）；
- 再看 `end2end` 是否改善（例如 ≥10%）；
- 同时检查动作质量，不能为了一点 latency 牺牲大量精细操作成功率。

### 4.4 Phase 3：动作级验证（P1，如已有 π0 checkpoint）

在**同一组离线 observation、state、language instruction、固定 initial noise** 下分别执行教师路径与重建路径，冻结其余模块，比较最终 action chunks：

- `action_mae` / `action_rmse`（按维度与时间步汇总，明确数据归一化单位）；
- `action_cos`（对每步 action vector；注意相反方向的影响）；
- `first-k action error`（尤其前 1、5 步，因为这些动作会立即执行）；
- 对机器人末端位姿 / gripper 分组报告，避免以归一化均值遮盖关键细节；
- 固定去噪步数、同一噪声/seed，避免随机采样差异被误认为特征差异。

**重要**：`L0` 的 64 tokens 直接下传会改变 prefix 长度，必须相应正确重建 mask、position ids 与 KV cache 逻辑；恢复成 256 tokens 的其它方案原则上保持下游接口不变。

### 4.5 Phase 4：LIBERO 成功率（P2，资源和环境就绪时）

- 每条路线使用相同 checkpoint、同一任务集、同一组 evaluation seeds、相同相机图像裁切 / action horizon / control frequency；最少先选一批简单移动任务和精细抓取/接触任务。
- 首轮可以对前两种最有前途的方法每任务跑 20 个 episode 做 smoke test；用于最终对比时再增加到 50 甚至更多，报告成功率及置信区间。
- 必须区分任务阶段：自由空间平移 vs 接近物体 vs 抓取/接触，检查是否某些阶段系统性恶化。
- 成功率不足时进一步检查 **低分辨率语义分布漂移、位置编码、Patch 引导失配、动作误差累积**，不要直接归因于某个插值核。

## 5. Coding Agent 具体实施规范

### 建议工程结构

> 实际实现已整理在 `docs/experiments/feature_upsampling/`（代码统一放在 `code/` 子目录，
> 数据、结果、图表、日志分别在各子目录），以下为原始建议结构。

```text
docs/experiments/feature_upsampling/
  README.md
  configs/default.yaml
  extract_features.py       # 教师/学生/Oracle 输出；缓存样本、相机与特征元信息
  upsamplers.py             # nearest, bilinear, bicubic, JBU, patch_guided
  hooks.py                  # SigLIP 输出 / projector / patch embed hooks
  eval_features.py          # cosine, rel_l2, norm_ratio, spatial_delta
  bench_latency.py          # GPU events, warmup, median/p90
  eval_actions.py           # 有 π0 checkpoint 时启用
  tests/test_shapes.py
  tests/test_upsamplers.py
  results/*.csv
  plots/*.png
  reports/summary.md
```

### 最低 API 契约

```python
# 用真实仓库的后端对象实现：这里是接口约定，不是直接可运行的 openpi 代码。
# image_hr: [B,3,224,224], rgb01: [B,3,224,224] in [0,1]
# output:    [B,256,D], D == 实际 PaliGemma image embedding width

teacher = encode_siglip(image_hr, resolution=224, output="post_projector")
student64 = encode_siglip(resize(image_hr, 112), resolution=112,
                          posemb="interpolate_pretrained", output="post_projector")

f_low = student64.transpose(1, 2).reshape(B, D, 8, 8)
pred = upsample(f_low, rgb01=rgb01, patch_hi=patch_hi,
                method="jbu", target_hw=(16, 16))
pred = pred.flatten(2).transpose(1, 2)
assert pred.shape == teacher.shape
```

### 不可省略的单元测试

1. **恒定场保真**：输入所有低分辨率 Feature 相等时，最近邻/双线性/JBU/Patch-guided 输出（数值容差内）仍应为该常数。
2. **权重归一化**：JBU/Patch-guided 每个高分辨率像素的邻域权重之和约为 1，权重非负、没有 NaN。
3. **对齐测试**：人工造一个 8×8 网格与 16×16 规则图案，保证行列次序、坐标对应、边缘 padding 正确；特别验证没有 H/W 互换与 `align_corners` 冲突。
4. **原模型不回归**：切换回原始 224 分支时结果应复现未修改模型。
5. **没有意外训练**：所有参数 `requires_grad=False`；实验过程中不创建/优化适配器权重；只有推理计算。
6. **项目可重复**：每条路线用相同数据、checkpoint、seed、mask、视觉后处理；配置与实际运行设备写入 results meta。

### 测试与参数搜索纪律

- 默认算法配置在主测试集运行；小规模参数搜索只能在独立开发集/固定少量样本进行，避免按最终 LIBERO 测试分数挑参数。
- 若 JBU/Patch-guided **不如 Bicubic**，直接报告负结果；不得隐藏最差任务或只给最优图片。
- 任务执行失败必须把 stack trace / 环境问题写入 `reports/summary.md`；不可自动改用随机特征或其它模型后仍标记为 π0/SigLIP 结果。

## 6. 首日任务（建议严格按此顺序）

- [ ] **0. Baseline**：找到并 hook 正确 checkpoint 的 image embedding，验证 `[B,256,D]`；打印 SigLIP/pos embeddings 的真实形状。
- [ ] **1. Low-res branch**：适配 learned 8×8 pos emb，使 `[B,64,D]` 能运行；核对不加载随机位置参数。
- [ ] **2. Feature extraction**：对 20 张代表图片存 Teacher224、Real112、Oracle64，保留 raw RGB224 / 112 与相机名。
- [ ] **3. U algorithms**：Nearest、Bilinear、Bicubic、JBU 3×3、Patch-guided 3×3，过形状/常量/无 NaN 测试。
- [ ] **4. Offline metrics**：20 张 smoke test 后扩到 200～500 张，输出 `results/feature_metrics.csv` 与 `plots/` 可视化。
- [ ] **5. Latency**：batch=1、实际 GPU 编码与上采样耗时（含预热 / 同步），输出 `results/latency.csv`。
- [ ] **6. Decision**：把两个最好的纯算法与 `Bicubic` 对比，说明是否值得接 π0 Action/ LIBERO。

**首日最小验收**：不需要跑通 LIBERO，但必须能**客观回答**：(a) 112 的 SigLIP 是否运行且真的更快；(b) 插值/JBU/Patch-guided 是否更接近 Teacher；(c) 总视觉路径加速多少；(d) 差距主要来自低分辨率编码还是恢复算法。

## 7. CSV 列定义与报告格式

`results/feature_metrics.csv`（一张图×一路线×一种特征源）：

```csv
image_id,camera,method,source,pos_emb_method,cos_mean,cos_p10,rel_l2,norm_ratio,spatial_delta,valid
```

`results/latency.csv`：

```csv
method,device,framework,dtype,batch_size,warmup,iters,resize_ms,patch_guide_ms,encoder_ms,upsampler_ms,vision_total_ms,end2end_ms,median_ms,p90_ms,peak_vram_mb
```

其中 `vision_total_ms` 为实际一体化链路总计，独立测得；不要简单相加不同计时实验的 median。 `end2end_ms` 未测则留空，不得填 0。

`reports/summary.md` 最少给：

- 使用的 checkpoint、commit、硬件、dtype、相机、图像数量；
- 一个包含 `method / cos / rel_L2 / vision_total_ms / vision_speedup / end2end_ms(若有)` 的表；
- Oracle-downsample 与 Real-lowres 对照解释；
- 三组好/差案例可视化（不要仅 cherry-pick）；
- 失败案例、是否发生 learned pos emb 的尺度分布漂移；
- 明确结论：**值得继续 / 仅适合低频或简单阶段 / 暂不值得**，并解释依据。

## 8. 重要科学边界与预期风险

1. **特征数量恢复不等于信息恢复**：若 U 只是对 64 个低分辨率特征线性加权，单张图像形成的 `[256,D]` 特征矩阵秩不超过 64；即使引入 RGB/Patch 引导决定权重，也不会凭空增加独立的**深层特征方向**。用额外 Patch 高频值注入时才可能摆脱这个线性张成空间限制，但存在跨层特征分布失配风险。
2. **重建教师特征可能不是最终目标**：即使教师余弦相似度一般，只要动作质量保持且更快仍有价值；反过来，cosine 很高也不保证控制成功率。
3. **视角与精细动作**：缩小图像可能让 gripper 接触、小按钮、细物体等结构消失；没有学到的先验时，纯插值很难恢复。
4. **Position embedding 是关键变量**：Real-lowres 的性能恶化可能来自位置编码重新采样，而不一定是图像空间信息缺失；建议额外做 `pool224-patch-emb -> transformer64` 的消融，隔离图像 Resize 与 token pooling 差异。
5. **端到端开销不会缩小 4 倍**：视觉编码下降不意味着 VLM Prefix / AE 计算下降；因为 U 后仍是 256 Tokens。报告 encoder-only 和 end-to-end 两个数字。
6. **不要误解文献任务**：FeatUp/AnyUp/NAF 主要关注高分辨率稠密视觉预测，不能从语义分割效果直接推出 VLA 动作成功率。

## 9. 参考文献与官方实现（核心链接）

1. Kopf, Cohen, Lischinski, Uyttendaele. **Joint Bilateral Upsampling**, SIGGRAPH 2007. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/joint-bilateral-upsampling/) · [论文信息](https://graphics.uni-konstanz.de/publikationen/Kopf2007JointBilateralUpsampling/index.html)
2. He, Sun, Tang. **Guided Image Filtering**, ECCV 2010. [论文](https://mlanthology.org/eccv/2010/he2010eccv-guided/)
3. He, Sun. **Fast Guided Filter**, 2015. [arXiv](https://arxiv.org/abs/1505.00996)
4. Fu et al. **FeatUp: A Model-Agnostic Framework for Features at Any Resolution**, ICLR 2024. [arXiv](https://arxiv.org/abs/2403.10516) · [GitHub](https://github.com/mhamilton723/FeatUp)
5. Wimmer et al. **AnyUp: Universal Feature Upsampling**, ICLR 2026 Oral. [主页](https://wimmerth.github.io/anyup/) · [GitHub](https://github.com/wimmerth/anyup)
6. Chambon et al. **NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering**, CVPR 2026. [arXiv](https://arxiv.org/abs/2511.18452) · [GitHub](https://github.com/valeoai/NAF)
7. Seo, Hamilton, Kim. **Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling**, 2025 arXiv / CVPR 2026（作者主页信息）。[arXiv](https://arxiv.org/abs/2511.16301) · [GitHub](https://github.com/seominseok0429/Upsample-Anything_Pytorch)
8. Physical Intelligence. **openpi** 官方仓库与模型实现。 [Repo](https://github.com/Physical-Intelligence/openpi) · [SigLIP](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/siglip.py) · [π0](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)

---

### 可直接复制给 Coding Agent 的执行指令

> 阅读本文件，按 **第 6 节首日任务** 严格执行，优先在现有项目/已有 π0 checkpoint 上运行。不要训练任何适配器，不要修改原始 checkpoint，不要为了通过 shape 检查偷偷把 112 输入改回 224，也不要忽略 learnable position embedding 的插值问题。先交付 20 张图的 smoke test，再扩大到 200～500 张。必须保存可复现的代码、CSV、可视化和 `reports/summary.md`；如果缺少 checkpoint、数据集或 GPU，明确写下阻塞项，并完成所有不依赖该资源的单元测试。对失败路线如实报告。第一天不要求跑 LIBERO 成功率。
