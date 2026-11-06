# Diffusion Policy 性能扰动实验：compute-bound 还是 memory-bound

- **对象**：`DiffusionUnetHybridImagePolicy`，Push-T image CNN/UNet，官方 seed 42 checkpoint
  （`epoch=0500-test_mean_score=0.884.ckpt`）
- **硬件**：NVIDIA RTX 4090（24 GB，FP32 峰值 82.6 TFLOPS，显存带宽 1008 GB/s），驱动 580.95.05
- **日期**：2026-09-18
- **关联**：[analysis-forward-bottleneck.md](analysis-forward-bottleneck.md)（单次 forward 的 nsys 瓶颈分析）

---

## 0. 结论

**既不是 compute-bound，也不是 memory-bound。**

用扰动实验（batch 1→16，去噪步数 10→100）标定出来的结果是：

1. **batch 1→4，同样的 kernel、同样的 GPU 时间，算力需求翻了 4 倍，耗时一点没变。**
   685 个卷积 kernel，卷积 GPU 时间 237.0 ms（B=1）vs 236.5 ms（B=4）。
   任何吞吐受限的资源（SM 算力或 DRAM 带宽）都不可能表现成这样——它说明 GPU 根本没被吞吐喂饱，
   而是被**每个 kernel 的固有延迟/并行度不足**卡住。

2. **算力和带宽的利用率都低到可以排除"饱和"**：每步 achieved **0.29–2.53 TFLOPS**（FP32 峰值的
   **0.35 %–3.1 %**）；权重流量折算 **46–83 GB/s**（峰值的 **4.5 %–8.2 %**）。两者都离饱和差一个
   数量级以上。

3. **batch ≥ 6 之后延迟曲线上出现的"台阶"，不是标度律，是 cuDNN 换算法。**
   cuDNN 在三档算法之间跳：batch 1–4 用 implicit GEMM，batch 6 用 direct grouped，
   batch 8+ 用 FFT 卷积。打开 `torch.backends.cudnn.benchmark` 让 cuDNN 实测选算法后，
   batch 8 快 21.7%、batch 16 快 31.4%。

**归类**：属于"两个利用率都低、kernel 背靠背但并行度不足"（小 GEMM/Conv 形状不利），
并叠加了 kernel 间隙带来的下发开销。不是 A（算力饱和），也不是 B（带宽饱和）。

**注意**：这是**扰动标定**得到的推断，不是硬件计数器的直接测量。本机性能计数器被
`ERR_NVGPUCTRPERM` 挡住（见 [analysis-forward-bottleneck.md 第 6 节](analysis-forward-bottleneck.md)），
严格判定仍需 Nsight Compute。扰动法的推理链和局限见第 8 节。

---

## 1. 方法

判据来自两种受限形态在 batch 方向上的**不同签名**：

| 受限类型 | batch 增大时的表现 |
| --- | --- |
| compute-bound（算力饱和） | 每样本延迟**恒定**，吞吐随 batch 线性上升后饱和 |
| 权重带宽受限 | 权重每次 launch 只读一遍，与 batch 无关 → 每样本延迟按 ~1/B 下降 |
| 激活带宽受限 | 激活随 batch 线性增长 → 每样本延迟**恒定** |

三种形态里 compute-bound 和"激活带宽受限"在 batch 方向的签名相同，靠延迟单独分不开。
所以再用**第二个轴**（去噪步数）做线性拟合分离固定成本与每步成本，并用实测 FLOPs 算出
achieved TFLOPS、用参数量算出权重流量，与 roofline 比对。

FLOPs 用 forward hook 直接统计 Conv/Linear 的乘加次数（`MacCounter`，只统计不计时的独立
一遍），不依赖任何外部库。

---

## 2. 复现命令

```bash
# 主网格：batch x 去噪步数，不带 nsys
policies/diffusion_policy/.venv/bin/python profiling/perturb_diffusion_policy.py \
  --batches 1,2,4,8,16 --steps 10,25,50,100 --reps 3 --warmup 2 --burn-in 5 \
  --output-dir profile_results/diffusion-policy/perturbation

# kernel 级采集：一次一个 (batch, steps)，带 NVTX 供 nsys 抓 profile_window
for B in 1 4 6 8 16; do
  d=profile_results/diffusion-policy/perturbation/b$B-n25
  mkdir -p $d
  CUDA_VISIBLE_DEVICES=0 NSYS_NVTX_PROFILER_REGISTER_ONLY=0 nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=nvtx --nvtx-capture='profile_window@*' --capture-range-end=stop \
    --force-overwrite=true --output=$d/b$B-n25 \
    policies/diffusion_policy/.venv/bin/python profiling/perturb_diffusion_policy.py \
    --nvtx --batches $B --steps 25 --reps 5 --warmup 2 --burn-in 5 --output-dir $d
done

# cuDNN autotuning 对照
policies/diffusion_policy/.venv/bin/python profiling/perturb_diffusion_policy.py \
  --cudnn-benchmark --batches 1,4,8,16 --steps 25 --reps 3 --warmup 2 --burn-in 5 \
  --output-dir profile_results/diffusion-policy/perturbation/benchmark-on
```

`--burn-in 5` 是必要的：网格从低负载开始，不加预热的话 GPU 还在低频，前几个点会虚高。

**无 nsys 的计时更接近真实延迟**：同一配置（B=1, N=100）不带 nsys 是 1216.2 ms，
带 nsys 是 1444 ms（主报告数字），**nsys 本身带来约 19% 的观测开销**。做绝对性能结论时
引用不带 nsys 的数字。

---

## 3. 延迟网格（ms，无 nsys）

| B \ N | 10 步 | 25 步 | 50 步 | 100 步 |
| --- | --- | --- | --- | --- |
| **1** | 126.8 | 311.0 | 617.5 | 1216.2 |
| **2** | 125.2 | 304.8 | 608.7 | 1209.4 |
| **4** | 126.4 | 307.2 | 617.6 | 1217.6 |
| **8** | 206.7 | 509.5 | 1019.0 | 2037.4 |
| **16** | 223.6 | 553.2 | 1104.6 | 2199.3 |

模型参数：`horizon=16`、`n_obs_steps=2`、`action_dim=2`；UNet 251.5 M 参数（1.006 GB FP32），
含视觉编码器共 262.7 M（1.051 GB）。

---

## 4. batch 标度：平台的形状

固定 N=100：

| B | 延迟 ms | 每样本 ms | 吞吐 samp/s | 相对 B=1 吞吐 | 相对 B=1 FLOPs |
| --- | --- | --- | --- | --- | --- |
| 1 | 1216.2 | 1216.2 | 0.82 | 1.00× | 1× |
| 2 | 1209.4 | 604.7 | 1.65 | 2.01× | 2× |
| 4 | 1217.6 | 304.4 | 3.29 | **4.00×** | 4× |
| 8 | 2037.4 | 254.7 | 3.93 | 4.78× | 8× |
| 16 | 2199.3 | 137.5 | 7.27 | 8.85× | 16× |

- **B=1→4：延迟几乎完全不变（1216.2 / 1209.4 / 1217.6 ms），吞吐线性翻 4 倍。**
  同样的时间里干了 4 倍的活。
- B=4→8：延迟跳到 1.67 倍。
- B=8→16：延迟只涨 1.08 倍，吞吐从 4.78× 升到 8.85×。

`T(N) = 固定成本 + 每步成本 × N` 的线性拟合（r² 全部 = 1.0000）：

| B | 固定成本 ms | 每步 ms | 每步 FLOPs | achieved TFLOPS | 占 FP32 峰值 |
| --- | --- | --- | --- | --- | --- |
| 1 | 8.39 | 12.097 | 3.47 G | **0.29** | **0.35 %** |
| 2 | 4.60 | 12.053 | 6.95 G | 0.58 | 0.70 % |
| 4 | 5.98 | 12.134 | 13.89 G | 1.15 | 1.39 % |
| 8 | 1.94 | 20.351 | 27.79 G | 1.37 | 1.66 % |
| 16 | 4.85 | 21.953 | 55.58 G | **2.53** | **3.06 %** |

**每步 FLOPs 随 batch 完美线性**（3.47 G/样本/步），固定成本（obs encoder + 前后处理）
约 5–8 ms 且不随 batch 变化——视觉编码器在这个规模下可以忽略。

而**每步成本在 B=1/2/4 三档完全相同（12.097 / 12.053 / 12.134 ms），算力需求却是 1:2:4。**

---

## 5. Roofline 判据

单步（去噪循环里的一次 UNet 前向）要读全部 UNet 权重 1.006 GB：

| B | 每步耗时 | achieved 算力 | 占 82.6 TFLOPS | 权重流量折算带宽 | 占 1008 GB/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 12.097 ms | 0.29 TFLOPS | 0.35 % | 83 GB/s | 8.2 % |
| 4 | 12.134 ms | 1.15 TFLOPS | 1.39 % | 83 GB/s | 8.2 % |
| 16 | 21.953 ms | 2.53 TFLOPS | 3.06 % | 46 GB/s | 4.5 % |

**两个资源都远未饱和。** 算力用到 0.35 %–3.1 %，权重流量折算带宽 4.5 %–8.2 %。

一个 kernel 若是 DRAM 带宽受限，它应当跑在峰值带宽的**很大一个比例**上；8 % 不构成"带宽受限"。
同理 0.35 % 的算力利用率不构成"算力饱和"。

---

## 6. kernel 级证据：cuDNN 换算法

在 N=25 上逐 batch 采集（nsys）：

| B | 主导卷积算法 | 卷积 kernel 数 / forward | GPU ms / forward | 卷积 ms / forward | wall ms / forward |
| --- | --- | --- | --- | --- | --- |
| 1 | `implicit_convolve_sgemm`（implicit GEMM） | 685 | 252.9 | **237.0** | 341.7 |
| 4 | `implicit_convolve_sgemm`（implicit GEMM） | **685** | 253.9 | **236.5** | 347.4 |
| 6 | `conv2d_grouped_direct_kernel`（direct grouped） | 200 direct + 530 implicit | 371.4 | 257.1 | 464.7 |
| 8 | FFT 卷积 `fft1d_r2c_32` + `ampere_gcgemm_64x64_nt` + `fft1d_c2r_32` | 800 FFT + 530 implicit | 466.1 | 351.7 | 549.4 |
| 16 | FFT 卷积 + `precomputed_convolve_sgemm` | 900 FFT + 583 implicit | 569.8 | 376.4 | 583.8 |

B=8 的 FFT kernel 计数为 `fft1d_r2c_32` 400 / `ampere_gcgemm_64x64_nt` 200 / `fft1d_c2r_32` 200，
比例与"正变换 + 频域复数乘 + 逆变换"的 FFT 卷积流水线一致。

**关键对比 B=1 vs B=4**：

| | B=1 | B=4 | 倍数 |
| --- | --- | --- | --- |
| 卷积 kernel 个数 | 685 | 685 | 1.00× |
| 卷积 GPU 时间 | 237.0 ms | 236.5 ms | **1.00×** |
| 总 GPU 时间 | 252.9 ms | 253.9 ms | 1.00× |
| 每步 FLOPs | 3.47 G | 13.89 G | **4.00×** |

**同一批 kernel、同样的个数、同样的耗时，算了 4 倍的活。** 这是"kernel 本身没被活喂饱"的
直接证据：每个卷积 kernel 的耗时由固定的延迟/依赖/占用率决定，而不是由它要做多少计算决定。

按每个卷积 kernel 平均 346 µs 计，它在 346 µs 里只推进了约 38 MB 权重（≈110 GB/s，峰值的 11 %），
下发成本只有 7.65 µs（2 %）——所以也不是 launch 受限，就是 kernel 自身的并行度不足。

结构上的原因：动作视野 `horizon=16`，batch=1 时卷积展开成 GEMM 后 N 维只有 16，
即便 batch=4 也只有 64，远不足以填满 128 个 SM。这与主报告里观察到的
"97 % 的大间隙落在每个去噪 step 的前 40 %"是同一个现象的两面。

---

## 7. cuDNN autotuning 验证

如果台阶来自 cuDNN 的启发式算法选择，那么让 cuDNN 对每个形状实测选算法应该能绕开。
N=25 对照：

| B | 启发式（默认） | `cudnn.benchmark=True` | 变化 | 峰值显存 |
| --- | --- | --- | --- | --- |
| 1 | 311.0 ms | 290.0 ms | −6.8 % | 1045 → 1013 MiB |
| 4 | 307.2 ms | 292.4 ms | −4.8 % | 1048 → 1019 MiB |
| 8 | 509.5 ms | **398.9 ms** | **−21.7 %** | 1563 → 1565 MiB |
| 16 | 553.2 ms | **379.2 ms** | **−31.4 %** | 1574 → **1160 MiB** |

autotuning 后 **B=16（379.2 ms）比启发式下的 B=8（509.5 ms）还快**，显存也降了 26 %。
台阶基本被抹平，进一步坐实"台阶 = 算法选择"而不是硬件标度律。

**这是本次实验最直接的工程结论**：官方默认（`cudnn.benchmark=False`，走启发式）在
batch ≥ 8 时选到了明显更差的卷积算法。

---

## 8. 结论与局限

### 结论

**既不是 compute-bound 也不是 memory-bound，是并行度/延迟受限（低占用率 + 小 GEMM 形状不利）**，
外加主报告已量化的 kernel 间隙下发开销。

- compute-bound 被排除：achieved 0.35 %–3.1 % 峰值，且 4× 算力不加时间。
- memory-bound 被排除：权重流量折算带宽 4.5 %–8.2 % 峰值；且 batch 1→4 的
  激活流量增加 4 倍而时间不变，与"激活带宽受限"的签名不符。
- 顺带排除了 launch 受限：下发 7.65 µs vs kernel 执行 346 µs，占 2 %。
- 剩下唯一自洽的解释：**每个 kernel 的网格太小**（`horizon=16` 导致 GEMM N 维过窄），
  GPU 大部分 SM 空闲，耗时由延迟与依赖链决定而非吞吐。

**工程含义**：把这个模型做成 batch 推理，batch ≤ 4 是**免费**的——延迟不变、吞吐 4 倍。
这对多环境并行的机器人策略服务是直接可用的。

### 局限

1. **这是推断，不是计数器测量。** 严格的 compute-vs-memory 判定需要 Nsight Compute 的
   SM Active / SM Issue / Tensor Active / DRAM Throughput 计数器，本机因
   `ERR_NVGPUCTRPERM`（`RmProfilingAdminOnly: 1` + 容器缺 `CAP_SYS_ADMIN`）不可用。
   第 5 节的 roofline 用的是**datasheet 峰值**，不是实测可达值；GDDR6X 的实际可达带宽
   通常显著低于峰值，因此"利用率 8 %"这个数字的绝对含义要打折扣——但即便按 50 % 可达
   带宽折算，也仍只有 16 %，不足以支撑"带宽受限"。
2. **权重流量的估算是解析值**（251.5 M 参数 × 4 B），假设每步完整读一遍全部权重。
   若 cuDNN 有缓存或复用，真实流量会更低，结论方向不变（更低只会更不像带宽受限）。
3. **延迟数字受 GPU 争用影响**：本机是共享的 4 卡机器，跨批次运行的绝对值可能有几个
   百分点漂移。同一批内的相对比较（尤其是 B=1 vs B=4 的 1.00×）不受影响。
4. **未做 batch 5、7 等中间点**，所以"算法在 6 和 8 切换"是采样到的边界，实际切换点
   可能在 5–6 与 7–8 之间。

---

## 附：产物

| 文件 | 说明 |
| --- | --- |
| `analysis-perturbation.md` | 本文 |
| `perturbation/perturbation.json` | 主网格：逐配置延迟、FLOPs、线性拟合、环境与 checkpoint 信息 |
| `perturbation/b{1,4,6,8,16}-n25/` | 各 batch 的 nsys 采集、sqlite、kernel_breakdown.csv、summary.txt |
| `perturbation/benchmark-on/perturbation.json` | cuDNN autotuning 对照 |

脚本：[profiling/perturb_diffusion_policy.py](../../../profiling/perturb_diffusion_policy.py)
