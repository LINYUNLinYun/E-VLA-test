# Training-Free Self-Speculative VLA：Idea 梳理、风险分析与 π0 低成本验证计划

> 当前基准模型：**π0（Physical Intelligence / OpenPI）**  
> 目标：优先验证“能否无需训练，从完整 π0 中直接抽取一个低成本 fast path，并在简单/低风险状态下替代 full inference”。

---

## 1. 核心 Idea

现有不少 speculative VLA 系统采用：

```text
Small / Draft Model
        ↓
快速生成候选动作
        ↓
Full Model
        ↓
批量验证 / 拒绝 / 重规划
```

这种范式的问题是通常需要：

- 单独训练一个 draft model / verifier；
- 额外保存一套小模型参数；
- small model 与 full model 是两个相对独立的模型；
- fast path 可能仍然基于已经陈旧的 observation；
- verification 往往使用简单的 action distance 或固定阈值。

本项目希望探索一种 **training-free self-speculative VLA**：

> 不训练额外 drafter，而是直接从已经训练好的 full VLA 中抽取一条低成本 fast path。  
> fast path 使用更少的 Transformer 层、更少的 flow-matching steps，并尽量基于最新 observation 快速产生动作；仅在风险增大、环境发生变化或 fast path 失去可信度时，才启动完整 slow path。

概念上：

```text
                          Latest Observation
                                  │
                         Vision Encoder
                                  │
                           Fresh Vision Buffer
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                Fast Path                 Slow Path
          partial Transformer           Full π0
          few flow steps                18 layers
                     │                  10 flow steps
                     │                         │
                  Draft                      Anchor
                     │                         │
                     └─────────┬───────────────┘
                               │
                    Training-free Risk Gate
                               │
                    ┌──────────┴──────────┐
                    │                     │
                  SAFE                  EVENT
                    │                     │
              execute fast          invoke / commit
                 action               full π0
```

核心目标不是简单“small model + big model”，而是：

**让同一个 π0 根据当前状态动态切换 fast / slow computation path。**

---

# 2. 为什么选择 π0 作为第一阶段基准

当前先以 **π0** 而不是 π0.5 为主基准。

原因：

1. π0 是较基础、清晰的 **VLM + flow-matching action expert** 架构，适合验证核心机制。
2. π0.5 在 π0 基础上又引入了新的 state 输入方式与 AdaRMSNorm timestep conditioning，会增加变量。
3. 若 training-free fast path 在 π0 上成立，再迁移到 π0.5，可以作为后续 generalization experiment。
4. 官方 OpenPI 当前同时提供 π0 模型和 PyTorch 实现，方便做中间层 hook、profile 和快速 ablation。

当前 OpenPI 配置中：

- PaliGemma / Gemma-2B backbone：18 层；
- Action Expert / Gemma-300M：18 层；
- Vision Encoder：SigLIP So400m/14；
- 默认 action horizon：50；
- 默认 flow inference steps：10；
- inference 先对 image + language prefix 计算 KV cache，再迭代 Action Expert。

因此第一阶段天然有两个 computation budget：

\[
k=\text{Transformer depth}
\]

以及：

\[
m=\text{flow-matching steps}
\]

我们最终希望得到：

\[
\text{FastPath}(k,m),\qquad k<18,\;m<10
\]

---

# 3. 当前假设

## H1：完整 π0 的中间层已经包含足够的 action information

完整模型使用：

```text
Layer 1 → Layer 2 → ... → Layer 18 → action_out_proj
```

假设在简单状态下：

```text
Layer 1 → ... → Layer k → norm → action_out_proj
```

得到的 vector field 已经与最终 Layer 18 足够接近。

也就是说：

\[
v_t^{(k)} \approx v_t^{(18)}
\]

并且随着层数增加：

\[
D(v_t^{(k)},v_t^{(18)})\downarrow
\]

如果 H1 成立，则可以从 big model 中直接“切”出一个 small path，而无需重新训练 drafter。

---

## H2：部分 flow-matching steps 可以产生有意义的快速 draft

π0 默认从随机噪声出发，用多个 Euler steps 积分：

\[
x_{t+\Delta t}=x_t+\Delta t\,v_\theta(x_t,t)
\]

默认约 10 steps。

假设在简单状态下，较少步骤：

\[
m\in\{1,2,4,6\}
\]

也能得到和 10-step full result 足够接近的 coarse action。

但这个假设风险较高：

> 少跑几个 denoising / flow steps 不一定等于“较差但可执行的 action”；它也可能只是未收敛的中间状态。

因此必须先实验验证。

---

## H3：历史动作能够为 fast path 提供更好的 warm start

机器人连续控制具有强时间相关性，因此：

\[
a_t \approx a_{t-1}
\]

尤其是在：

- free-space motion；
- 直线运动；
- inactive / low-risk phase；
- 末端速度和方向变化较小的阶段。

因此相比：

```text
random noise → few flow steps
```

更可能有效的是：

```text
previous action chunk
        ↓
shift / extrapolate
        ↓
1~2 flow refinement steps
        ↓
draft action
```

即：

\[
a_t^{draft}
=
\mathrm{FewStepFlow}
(
\mathrm{WarmStart}(a_{t-1})
)
\]

---

## H4：是否需要 full π0 应该由“风险事件”决定，而不是固定 action L2 阈值

目标不是判断：

\[
a_\text{draft}\approx a_\text{full}?
\]

而是判断：

\[
\boxed{\text{当前 fast-path action 是否仍处于可信执行域？}}
\]

可能的信号包括：

- action consistency；
- kinematic consistency；
- end-effector pose residual；
- Kalman residual；
- visual feature change；
- wrist camera change；
- gripper/contact state；
- observation staleness；
- 当前执行 phase。

---

# 4. 当前 Idea 的主要风险 / 挑刺

## 4.1 最大风险：中间层不一定能够直接作为“小 π0”

这是整个项目最先需要证伪的问题。

虽然完整模型的中间层可能已经包含 affordance / spatial / action 信息，但：

```text
intermediate hidden state
        ≠
final hidden state distribution
```

原来的：

```text
final norm + action_out_proj
```

是在最终层输出分布上训练的。

因此直接把 Layer 6 / Layer 9 / Layer 12 输出接到现有 output projection，可能出现：

- representation mismatch；
- scale / norm mismatch；
- direction correct but magnitude unstable；
- early layer action completely unusable。

如果这一假设失败，则“training-free partial model”这一主线需要重构。

因此 **Experiment 1 是整个项目的 go / no-go 实验。**

---

## 4.2 π0 的“VLM backbone”和 Action Expert 并不是完全独立的两条网络

π0 更准确地说是双 expert / mixture-of-transformer 风格：

```text
PaliGemma token stream
        ↘
         joint attention
        ↗
Action Expert token stream
```

两者都为 18 层，并在各层进行联合 attention。

因此真正的：

```text
partial Transformer
```

不能简单理解成：

> 只跑前 6 层 VLM，然后完整 Action Expert。

更自然的第一版是：

> **同一深度预算下截断 paired VLM + Action Expert blocks。**

例如：

```text
k = 6 / 9 / 12 / 15 / 18
```

---

## 4.3 少量 flow steps 不一定产生可执行动作

从 random noise 开始只跑 1–2 个 Euler steps，输出可能仍然处于 flow trajectory 的早期状态。

因此：

```text
random → 2 steps
```

只能作为最基础的 sanity check。

更值得尝试的是：

```text
history warm start → 1~2 refinement steps
```

---

## 4.4 Vision freshness 的思路成立，但不能随意混合新旧 hidden state

目标是降低 observation staleness：

```text
t0：full model 开始处理 O_t0
t0+150ms：模型仍在运行
t0+200ms：输出动作
```

此时环境已经可能发生变化。

因此 fast path 可以持续读取：

```text
O_t0+20
O_t0+40
O_t0+60
...
```

但不能简单：

```text
old VLM hidden
+
new vision embedding
```

因为新的视觉 token 会改变后续每一层的 hidden representation。

需要明确 consistency boundary，例如：

```text
Latest Image
    ↓
Vision Encoder
    ↓
fresh prefix embedding
    ↓
first k joint layers
    ↓
Fast Action Path
```

而不是任意拼接新旧中间状态。

---

## 4.5 Full model 本身也会 stale

如果 slow path 在时刻 \(t_0\) 开始：

```text
Full π0(O_t0)
```

200 ms 后返回：

```text
A_full(O_t0)
```

不能直接认为它一定比当前 fast action 更可靠。

应该同时记录：

\[
\Delta t_\text{full}
\]

和：

\[
D(\phi(O_\text{now}), \phi(O_{t_0}))
\]

必要时丢弃已经过期的 full result，而不是无条件 commit。

---

## 4.6 “fast response”不等于“compute acceleration”

如果 full π0 永远在后台跑：

```text
Fast Path
+
Full Path continuously
```

虽然机器人 response latency 降低，但总 FLOPs 和 energy 未必降低，甚至可能更高。

真正理想的系统是：

```text
Fast
Fast
Fast
Fast
  │
EVENT
  ↓
Full π0
  ↓
new anchor
  ↓
Fast
Fast
...
```

这样才同时降低：

- average action latency；
- full-model invocation frequency；
- total FLOPs；
- energy / memory bandwidth。

---

## 4.7 Action L2 distance 是一个较弱的 verifier

只判断：

\[
\|a_d-a_f\|_2 < \epsilon
\]

存在几个问题。

### 1. Model agreement 不等于 action correctness

如果 small path 和 full path 都来自同一 π0：

```text
same backbone
    ↓
correlated representation
    ↓
correlated failure
```

二者可能非常接近，但同时错误。

因此：

\[
D(a_d,a_f)\approx 0
\]

不能证明动作正确。

### 2. 相同 action distance 在不同 phase 下风险不同

例如：

```text
free-space：2 cm 偏差可能无所谓
contact / grasp / insertion：2 cm 可能直接失败
```

### 3. Joint-space distance 不等于 Cartesian effect

近似：

\[
\Delta x \approx J(q)\Delta q
\]

所以同样的：

\[
\|\Delta q\|_2
\]

在不同 robot pose 下可能造成完全不同的末端偏差。

### 4. Action vector 的各维物理单位不同

translation、rotation、gripper 不应该简单使用统一 L2 权重。

---

## 4.8 Kalman Filter 有价值，但不能单独作为 verifier

Kalman / kinematic prediction 能检查：

- trajectory smoothness；
- velocity / acceleration consistency；
- end-effector motion prediction；
- dynamic consistency。

但是：

```text
kinematically plausible
        ≠
task semantically correct
```

例如机械臂平滑地向错误目标移动，Kalman 很可能仍认为正常。

因此 Kalman 更适合作为：

> **Risk Gate 的一个独立 physical signal**

而不是最终唯一判据。

---

# 5. 更合理的系统版本

建议将最终系统逐渐收敛为：

## Training-Free, Freshness-Aware, Event-Triggered Self-Speculative VLA

```text
                       Latest Observation O_t
                                │
                         Vision Encoder
                                │
                         Fresh Prefix Buffer
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
          Partial π0 Fast Path          Full π0 Slow Path
             depth = k                  depth = 18
             flow = m                   flow = 10
                  │                           │
                  ▼                           ▼
              draft action                anchor action
                  │                           │
                  └────────────┬──────────────┘
                               │
                    Training-Free Risk Gate
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
      action signal       physical signal      observation
                                              freshness
          │                    │                    │
 historical /         FK / Jacobian / KF      vision delta
 self-consistency       gripper/contact       wrist motion
          │                    │              elapsed time
          └────────────────────┼────────────────────┘
                               ▼
                     SAFE            EVENT
                      │                │
                      ▼                ▼
                execute fast      invoke Full π0
```

---

# 6. 第一阶段：四个低成本实验

四个实验按照下面顺序进行。

原则：

> **先验证 fast path 是否存在，再研究 verifier。**

不要反过来。

---

# Experiment 1 — Intermediate-Layer Action Probe

## 目的

回答整个项目最关键的问题：

> π0 未经重新训练的中间层表示，是否已经能够预测与最终层方向一致的 action vector field？

如果答案是否，则当前 self-derived small model 思路需要重新设计。

---

## 方法

暂时不真正跳层加速。

仍然完整跑 18 layers，只是在以下层保存 Action Expert hidden state：

```text
k ∈ {3, 6, 9, 12, 15, 18}
```

对每个中间状态：

```text
h_k
 ↓
原模型 expert final norm
 ↓
原模型 action_out_proj
 ↓
v_t^(k)
```

比较：

\[
v_t^{(k)}
\]

与：

\[
v_t^{(18)}
\]

---

## 为什么先做 probe 而不真的截断模型

这样：

- 不需要重写 KV cache；
- 不需要改控制逻辑；
- 不需要重新训练；
- 一次 full forward 就得到多个 layer 的结果；
- 可以非常快判断 H1 是否值得继续。

---

## 指标

### 1. Vector-field MSE

\[
\mathrm{MSE}_k
=
\|v_t^{(k)}-v_t^{(18)}\|_2^2
\]

### 2. Cosine Similarity

\[
\cos(v_t^{(k)},v_t^{(18)})
\]

重点看 **方向是否先收敛**。

### 3. Action dimension-wise error

分别看：

- translation；
- rotation；
- gripper。

### 4. Across-flow-time consistency

不要只测一个 flow timestep。

建议：

```text
t ∈ {1.0, 0.75, 0.5, 0.25}
```

因为 early flow state 和 late flow state 的 representation 可能不同。

---

## 第一轮采样量

无需闭环。

先取：

```text
100–500 个 LIBERO observation
```

即可。

---

## Go / No-Go 判断

如果例如：

```text
k = 9 / 12
```

已经出现：

```text
cosine > 0.9
```

且大部分 action dimensions error 明显下降，则说明：

> partial π0 作为 training-free draft path 值得继续。

如果到：

```text
k = 15
```

仍然高度随机，则不要立即做后面的复杂 verifier。

---

# Experiment 2 — Flow-Step Budget Sweep

## 目的

验证：

> 单独减少 flow steps 到底能不能形成一个有意义的 fast action。

这个实验官方 OpenPI 几乎已经支持，无需复杂修改。

---

## 方法

固定同一个 observation 和 **同一份 initial noise**。

分别运行：

```text
num_steps ∈ {1, 2, 4, 6, 8, 10}
```

其中：

```text
10-step = reference
```

必须共享相同 noise，否则比较会混入 sampling randomness。

---

## 指标

### Action deviation

\[
D_m
=
\|a^{(m)}-a^{(10)}\|
\]

### Cosine similarity

尤其关注 first executed actions：

```text
a[0]
a[0:4]
a[0:8]
```

因为机器人往往不会完整执行整个 horizon 后才重新观测。

### Endpoint / trajectory deviation

如果 action 是 delta pose，可累计得到：

\[
\hat{x}_{t+h}
\]

比较不同 step 数得到的末端 trajectory。

### Latency

记录：

- prefix latency；
- single denoise step latency；
- total action latency。

---

## 关键观察

需要确认误差是否呈：

```text
1 → 2：明显改善
2 → 4：明显改善
4 → 6：改善减弱
6 → 10：收益很小
```

如果出现这种曲线，则 low-step fast path 很有希望。

---

# Experiment 3 — Warm-Start Few-Step Flow

## 目的

验证一个可能比“随机噪声 + 少步 flow”更重要的假设：

> 机器人历史动作本身能否成为 training-free draft prior？

---

## Baselines

比较四种 initialization：

### A. Random

```text
Gaussian noise
```

### B. Previous Chunk Shift

假设上一轮 action chunk：

```text
[a0, a1, a2, ..., a49]
```

下一轮初始化：

```text
[a1, a2, ..., a49, a49]
```

### C. Constant-Velocity Extrapolation

根据最近动作：

\[
\Delta a = a_{t-1}-a_{t-2}
\]

进行短时 extrapolation。

### D. Kalman Initialization

利用：

- robot proprioception；
- end-effector pose；
- velocity；

得到 predicted trajectory 作为 initialization。

---

## Few-step refinement

每种 initialization 只运行：

```text
1 / 2 / 4 flow refinement steps
```

比较 full 10-step result。

---

## 一个重要实现注意

不要简单把 warm-start action 当成：

```text
t = 1 的 Gaussian noise
```

更合理的是增加：

```text
sample_actions_from_init(
    x_init,
    start_time=t_start,
    num_steps=m
)
```

例如：

```text
start_time ∈ {0.5, 0.25}
```

表示：

> 当前初始化已经是一个相对接近最终动作的 trajectory，不需要重新从 t=1 开始。

---

## 关键问题

如果出现：

```text
previous-shift + 2 steps
≈
random + 8~10 steps
```

这是非常强的结果。

因为这意味着：

> fast path 可以同时利用模型内部 representation redundancy + robot temporal redundancy。

---

# Experiment 4 — Training-Free Gate / False-Accept Test

## 目的

不要一开始做复杂闭环。

先回答：

> 在 draft 已经明显失效时，不同 training-free signal 谁最容易及时发现？

重点指标不是普通 accuracy，而是：

\[
\boxed{
P(\text{gate accepts}\mid\text{draft is bad})
}
\]

即：

**False Acceptance Rate (FAR)**。

---

## 构造 cheap failure cases

从 LIBERO rollout 中抽取 observation，制造环境变化：

### Case 1：stale observation

```text
draft 使用 O_t
reference 使用 O_t+Δ
```

逐渐增加：

```text
Δ = 1 / 2 / 4 / 8 simulator steps
```

### Case 2：object displacement

轻微移动目标物体。

### Case 3：wrist-view change

保持 global scene 接近，但改变末端 pose，使 wrist camera 显著变化。

### Case 4：near-contact phase

单独取：

- approach；
- pre-grasp；
- grasp/contact；
- lift；

比较不同 gate 的可靠性。

---

## Gate candidates

### G1. Action L2

\[
R_a = \|a_\text{fast}-a_\text{anchor}\|
\]

### G2. Kinematic / Kalman residual

\[
R_k
\]

### G3. Vision feature delta

直接复用 π0 Vision Encoder：

\[
R_v
=
1-
\cos(
\phi(I_t),
\phi(I_{anchor})
)
\]

分别记录：

```text
global camera
wrist camera
```

而不是混成一个指标。

### G4. Combined risk

第一版不要训练 classifier。

直接标准化后：

\[
R
=
w_aR_a+w_kR_k+w_gR_{global}+w_wR_{wrist}
\]

先用简单 rule / threshold。

重点仍然保持：

> training-free。

---

## 指标

- False Acceptance Rate；
- False Rejection Rate；
- Full-model trigger ratio；
- average fast-run length；
- latency saved；
- 按 phase 分组后的 FAR；
- global camera 与 wrist camera 的触发差异。

---

# 7. 四个实验的执行顺序

严格建议：

```text
Experiment 1
Intermediate-Layer Probe
        │
        ├─ 失败 → 暂停当前 partial-model 主线
        │
        ▼
Experiment 2
Flow-Step Sweep
        │
        ▼
Experiment 3
Warm Start
        │
        ▼
Experiment 4
Training-Free Gate
```

原因：

如果：

```text
partial layer representation
```

本身就不能生成合理 vector field，

那么：

- vision buffer；
- Kalman；
- event gate；
- async fast/slow runtime；

暂时都不值得先做。

---

# 8. 推荐的第一轮最小实验矩阵

为了控制成本，第一轮只测：

## Layer

```text
{6, 9, 12, 15, 18}
```

## Flow steps

```text
{2, 4, 6, 10}
```

组合为：

| Layers | 2 steps | 4 steps | 6 steps | 10 steps |
|---|---:|---:|---:|---:|
| 6 | ✓ | ✓ |  |  |
| 9 | ✓ | ✓ |  | ✓ |
| 12 | ✓ | ✓ | ✓ | ✓ |
| 15 |  | ✓ | ✓ | ✓ |
| 18 | ✓ | ✓ | ✓ | ✓ |

不要第一轮就完整 sweep。

目标只是找到：

\[
\text{possible Pareto region}
\]

---

# 9. 第一阶段推荐输出图

如果要判断是否值得继续，最有价值的不是 success-rate bar chart，而是下面几张。

## Figure 1 — Layer Depth vs Vector Similarity

```text
x: Transformer depth
y: cosine(v_k, v_full)
```

## Figure 2 — Flow Steps vs Action Error

```text
x: flow steps
y: trajectory deviation from 10-step reference
```

## Figure 3 — Latency vs Action Agreement Pareto Frontier

```text
x: latency
y: agreement / trajectory error
```

每个点：

```text
(k layers, m flow steps)
```

## Figure 4 — Warm Start Comparison

```text
Random
Previous Shift
Constant Velocity
Kalman
```

在 1 / 2 / 4 flow steps 下比较。

## Figure 5 — Gate ROC / FAR

重点比较：

```text
L2 only
KF only
Vision only
L2 + KF + Vision
```

---

# 10. 当前最可能的论文贡献形式

如果实验结果支持假设，论文主线可以收敛成：

## Contribution 1 — Self-Derived Training-Free Fast Path

直接从 pretrained π0 中抽取：

```text
partial transformer
+
truncated flow
```

无需：

- retraining；
- distillation；
- extra drafter；
- auxiliary verifier model。

---

## Contribution 2 — Fresh Observation Fast Path

fast path 可以比 slow path 更频繁刷新 observation，减少：

\[
\text{observation-to-action staleness}
\]

---

## Contribution 3 — Event-Triggered Slow Inference

完整 VLA 不再每一个 control cycle 都运行。

而是：

```text
fast → fast → fast → event → full → fast ...
```

---

## Contribution 4 — Physical + Perceptual Training-Free Verification

不只依赖：

\[
\|a_d-a_f\|_2
\]

而结合：

- action consistency；
- robot kinematics；
- observation change；
- wrist/global camera difference；
- staleness；
- gripper/contact phase。

---

# 11. 当前创新边界

需要避免把以下点单独当主要 contribution：

### Kalman Filter 本身

KERV 已经使用 kinematics-based Kalman Filter 做 speculative VLA correction / threshold adjustment。

因此：

> “使用 Kalman Filter 替代 action distance”

本身不够。

### Adaptive early exit 本身

A1 已经做 intermediate-layer action consistency + truncated flow matching。

因此贡献必须强调：

> **无需为 early layer / fast path 做额外训练或专门 supervision。**

### Temporal token pruning

SpecPrune-VLA 已经探索 training-free spatial-temporal pruning。

因此本项目更应聚焦：

> **whole inference path speculation / fast-slow execution**

而不是只做 token pruning。

---

# 12. 当前 Go / No-Go 标准

## Strong Go

如果出现：

- 中间层 9/12 层 action vector 与 final direction 高度一致；
- 2–4 flow steps 已经接近 10-step；
- history warm start 显著减少所需 flow steps；
- fast path 在 simple/inactive states 明显可靠。

则继续做：

```text
fresh observation buffer
+
event-triggered slow path
+
physical/perceptual verifier
```

---

## Weak Go

如果只有：

```text
15/18 layers
+
6/10 flow steps
```

才稳定，

则 fast path 节省有限。

此时更应该转向：

- temporal reuse；
- cache；
- dynamic horizon；
- event-triggered full inference。

---

## No-Go

如果：

- 中间层 action direction 长期不稳定；
- few-step flow 与 full trajectory 无明显相关；
- history warm start 也无法改善；

则说明：

> pretrained π0 没有天然形成一个可直接切出的 training-free sub-model。

此时应停止在“partial π0 drafter”上继续投入，避免后续 verifier / runtime 工程把一个错误前提包装得更复杂。

---

# 13. 与现有工作最关键的关系

- **Draft & Verify**：LLM 中证明可从同一模型通过减少层数形成 self-speculative drafter；本项目希望验证这种现象在 flow-based VLA 中是否成立。
- **A1**：说明 VLA 中间层 action consistency 和 truncated flow 有价值，但其完整设计不是“原模型零训练直接切 fast path”；因此它既是强相关工作，也是必须区分的边界。
- **SpecPrune-VLA**：training-free 地利用历史/局部信息做 VLA pruning，证明 temporal redundancy 可以在不训练的情况下利用。
- **KERV**：已经引入 kinematics + Kalman，因此 Kalman 更适合作为本项目 risk gate 的一个组成信号，而不是独立创新点。
- **Realtime-VLA FLASH**：证明 flow/diffusion VLA speculative fast path 能显著降低 latency，但需要额外 draft path 训练；这是本项目很自然的直接对照。

---

# 14. 下一步代码修改优先级

## P0：只做 probe

先不要真正修改 runtime。

增加：

```text
record_intermediate_expert_hidden = {6, 9, 12, 15, 18}
```

在 paired transformer loop 中保存 expert stream hidden state。

然后：

```text
expert_norm(hidden_k)
→ action_out_proj
→ v_k
```

离线导出 CSV。

---

## P1：直接利用现有 num_steps

OpenPI 当前：

```python
sample_actions(..., num_steps=10)
```

已经支持 step sweep。

第一轮只需要：

```text
same observation
same initial noise
different num_steps
```

---

## P2：增加 warm-start sampler

增加：

```python
sample_actions_from_init(
    x_init,
    start_time,
    num_steps,
)
```

不改权重。

---

## P3：最后再实现真正 max_layers fast path

只有 Experiment 1 证明中间层可用以后，再真正修改：

```text
PaliGemmaWithExpertModel.forward(max_layers=k)
```

以及对应的 partial prefix KV cache。

这样避免过早陷入复杂 KV-cache 工程。

---

# 15. 第一阶段结论

当前最重要的问题不是：

> Kalman verifier 怎么设计？

也不是：

> vision buffer 怎么异步？

而是首先回答：

\[
\boxed{
\text{未经训练的 π0 内部是否天然存在一个可用的 fast sub-path？}
}
\]

如果答案为 Yes，后面的：

- latest observation；
- event trigger；
- Kalman；
- visual change；
- wrist-camera-aware gate；
- full-model staleness handling；

才有必要继续。

因此实验优先级应该始终是：

> **Representation → Generation → Warm Start → Verification → Runtime**

---

## 参考资料 / 当前代码依据

- Physical Intelligence OpenPI: https://github.com/Physical-Intelligence/openpi
- π0 model implementation: `src/openpi/models/pi0.py`
- PyTorch implementation: `src/openpi/models_pytorch/pi0_pytorch.py`
- Gemma / Action Expert config: `src/openpi/models/gemma.py`
- Draft & Verify, ACL 2024
- A1: A Fully Transparent Open-Source, Adaptive and Efficient Truncated VLA, 2026
- SpecPrune-VLA, 2025/2026
- KERV: Kinematic-Rectified Speculative Decoding for Embodied VLA Models, 2026
- Realtime-VLA FLASH, 2026

---

# Experiment 1 实测结果（2026-09-27）

已严格按本计划完成 Intermediate-Layer Action Probe：

- 原始 `pi0_libero` Triton 部署权重，不训练、不 early-exit、不修改 KV cache 或控制 loop；
- 4 个 LIBERO suite × 10 个 task × 每 task 1 个 episode × 5 个均匀轨迹位置，共 200 observations；这是 40 个 episode 内的进度覆盖，不应视为 200 个独立 episode 或人工 phase 标签；
- probe Layer 6/9/12/15/18，实际 flow time 为 1.0/0.8/0.5/0.3；
- 每个中间 hidden 均通过原 final RMSNorm + `action_out_proj`；
- 200/200 个固定 observation/noise 的最终输出与 production CUDA graph **bitwise equal**，最大差值 0；
- checkpoint SHA256 实验前后相同，Layer 18 sanity 为 cosine=1、relative L2=0。

跨四个 flow time 的平均 cosine：

| Layer | 6 | 9 | 12 | 15 | 18 |
|---:|---:|---:|---:|---:|---:|
| Mean cosine | 0.4890 | 0.5574 | 0.7500 | 0.8278 | 1.0000 |

Layer 15 在 t=1.0 可达 0.9285，但随 flow 推进降为 0.8853/0.7914/0.7061；Layer 12/15 的 norm ratio 多为 1.87–2.45，说明即使方向部分形成，幅值也未收敛。rotation 比 translation/gripper 更早形成，但不足以支持完整 action fast path。

结论：**No-Go / Reconsider**。这 200 个跨 suite/task/轨迹阶段的 observation 没有支持“原 pretrained π0 存在一个可直接切出、跨 flow time 稳定的 training-free Layer 9/12/15 fast path”。因此按本计划的门控条件，当前不继续 Experiment 2/3，不用后续复杂实验包装一个在本轮数据中未成立的 partial-layer 前提。

完整产物位于：

- `policies/realtime-vla-flash/results/pi0_intermediate_probe/intermediate_probe.csv`
- `policies/realtime-vla-flash/results/pi0_intermediate_probe/layer_vs_cosine.png`
- `policies/realtime-vla-flash/results/pi0_intermediate_probe/layer_vs_relative_l2.png`
- `policies/realtime-vla-flash/results/pi0_intermediate_probe/experiment1_report.md`
