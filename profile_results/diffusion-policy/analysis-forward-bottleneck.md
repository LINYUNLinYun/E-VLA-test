# Diffusion Policy 单次 forward 性能瓶颈分析

- **对象**：`profile_results/diffusion-policy/diffusion-policy.nsys-rep`
- **模型**：`DiffusionUnetHybridImagePolicy`，Push-T image CNN/UNet，官方 seed 42 checkpoint
- **硬件**：NVIDIA RTX 4090（24564 MiB），驱动 580.95.05
- **工具**：Nsight Systems 2026.2.1.210
- **分析范围**：NVTX `policy_infer` × 10 次真实推理。**不含**程序启动、checkpoint 加载、模型初始化、JIT 编译和 5 次 warmup
- **日期**：2026-09-18

---

## 0. 一句话结论

单次 forward **1444 ms**，其中 **91.6% 是 100 步去噪循环**；去噪循环里 **93.8% 的 GPU 时间花在 2710 次 `implicit_convolve_sgemm` 卷积**上（每次 forward，平均 347 µs/次）。
卷积本身把 GPU 喂得很饱，但每次 forward 要下发 **27461 个 kernel**，其中 86.4% 是小到 5 µs 以下的 elementwise/norm kernel，**GPU 有 30.6% 的时间在空转**，且这个空转已用对照实验证明不是插桩造成的。

---

## 1. 复现命令

以下命令全部在仓库根目录 `benchmark/` 下执行，得到本文所有数字。

```bash
# 1) 生成（或复用）sqlite 派生库。先删旧的，否则 nsys 会复用陈旧导出
rm -f profile_results/diffusion-policy/diffusion-policy.sqlite
nsys export --type=sqlite --force-overwrite=true \
  --output profile_results/diffusion-policy/diffusion-policy.sqlite \
  profile_results/diffusion-policy/diffusion-policy.nsys-rep

# 2) 主分析：产出 summary.txt / kernel_breakdown.csv / denoising_steps.csv
python3 profiling/analyze_diffusion_policy.py \
  profile_results/diffusion-policy/diffusion-policy.nsys-rep \
  --output-dir profile_results/diffusion-policy

# 3) 区间层面的 GPU 归因（核对各阶段 wall 与 gpu 的关系）
python3 profiling/attribute_gpu_time.py \
  profile_results/diffusion-policy/diffusion-policy.sqlite

# 4) NVTX 区间树
nsys stats --report nvtx_pushpop_sum \
  profile_results/diffusion-policy/diffusion-policy.nsys-rep

# 5) 对照实验：关闭阶段边界同步重采一次，用于判断间隙是否由插桩引入
mkdir -p profile_results/diffusion-policy/control-nosync
CUDA_VISIBLE_DEVICES=0 NSYS_NVTX_PROFILER_REGISTER_ONLY=0 nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=nvtx --nvtx-capture='profile_window@*' --capture-range-end=stop \
  --force-overwrite=true \
  --output=profile_results/diffusion-policy/control-nosync/control-nosync \
  policies/diffusion_policy/.venv/bin/python profiling/diffusion_policy_profile.py \
  --no-sync-stages --output-dir profile_results/diffusion-policy/control-nosync
python3 profiling/analyze_diffusion_policy.py \
  profile_results/diffusion-policy/control-nosync/control-nosync.nsys-rep \
  --output-dir profile_results/diffusion-policy/control-nosync
```

临时查询（间隙分布、CUDA API 构成）用 Python `sqlite3` 直查 `.sqlite`，见正文各节标注。

---

## 2. Forward overview

| 指标 | 数值 |
| --- | --- |
| forward wall time | **1444.363 ms**（10 次合计 14443.629 ms，范围 1413.3–1469.8 ms） |
| GPU kernel covered time | **10031.229 ms**（占 wall 69.45%） |
| GPU busy ratio | **69.45 %** |
| GPU kernel gap / idle time | **4412.400 ms**（30.55%） |
| CUDA API time（sum） | **3198.459 ms**（占 wall 22.15%） |
| CUDA API busy（并集） | 3198.459 ms（与 sum 相等，无重叠） |

### 显存操作时间

| 类型 | calls | bytes | time |
| --- | --- | --- | --- |
| HtoD | 1020 | 2,220,000 | 0.472 ms |
| DtoH | 2040 | 10,000 | 2.239 ms |
| DtoD | 0 | 0 | 0 |
| Memset | 4000 | 512,000 | 3.321 ms |

显存操作合计 **6.03 ms**，占 forward 的 0.04%。**搬运不是这个模型的瓶颈**。

### 阶段墙钟拆分（manifest `nvtx_reconciliation`，10 次合计，折算单次）

| 阶段 | 单次 wall | 占 forward |
| --- | --- | --- |
| `observation_preprocess` | 0.40 ms | 0.03 % |
| `vision_encoder` | 6.06 ms | 0.42 % |
| `diffusion_denoising` | **1436.64 ms** | **99.46 %** |
| `action_postprocess` | 0.33 ms | 0.02 % |
| `model_forward` | 1444.03 ms | 99.98 % |
| `policy_infer` | 1444.60 ms | 100 % |

区间对账（`unaccounted_fraction`）：`policy_infer` 0.04%、`model_forward` 0.04%、`diffusion_denoising` 1.81%。标记完整，没有靠摊派掩盖的残差。

### CUDA API 构成

```
总 API 时间 3198.5 ms / 360270 次调用
  cudaLaunchKernel_v7000          274610 次   2102.0 ms   avg  7.65 us
  cudaDeviceSynchronize_v3020       4120 次    861.8 ms   avg 209.18 us
  cudaEventRecord_v3020            35210 次    119.7 ms   avg  3.40 us
  cudaMemcpyAsync_v3020             3060 次     45.6 ms   avg 14.90 us
  cudaMemsetAsync_v3020             4000 次     31.7 ms   avg  7.92 us
  cudaStreamIsCapturing_v10000     36210 次     28.4 ms   avg  0.78 us
```

`cudaDeviceSynchronize` 的 4120 次来自入口的 `--sync-stages`（每个阶段边界一次，`denoising_step` + `scheduler_step` 各 100 次/forward）。**这部分是我们插桩引入的，第 4 节用对照实验单独量化。**

---

## 3. Kernel breakdown

按 `shortName` 聚合（同一底层 kernel 的模板实例合并为一行）。完整表见 `kernel_breakdown.csv`。

| kernel | total_ms | % GPU kernel time | calls | avg µs | p50 µs | p95 µs | max µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **implicit_convolve_sgemm** | **9411.749** | **93.82** | **27100** | **347.3** | **272.8** | **660.0** | **1107.6** |
| elementwise_kernel | 132.3 | 1.32 | 88320 | 1.5 | 1.4 | 2.2 | 2.5 |
| Kernel | 123.1 | 1.23 | 4010 | 30.7 | 27.6 | 53.4 | 53.9 |
| vectorized_elementwise_kernel | 73.3 | 0.73 | 61270 | 1.2 | 1.3 | 1.4 | 1.6 |
| dgrad2d_alg1_1 | 62.6 | 0.62 | 2000 | 31.3 | 39.5 | 40.5 | 40.8 |
| RowwiseMomentsCUDAKernel | 57.2 | 0.57 | 25200 | 2.3 | 2.2 | 2.5 | 10.6 |
| ampere_sgemm_128x32_tn | 46.4 | 0.46 | 4000 | 11.6 | 11.6 | 11.8 | 12.1 |
| ComputeFusedParamsCUDAKernel | 38.5 | 0.38 | 25200 | 1.5 | 1.5 | 1.6 | 1.8 |
| kernel | 30.0 | 0.30 | 9000 | 3.3 | 3.2 | 4.0 | 4.9 |
| precomputed_convolve_sgemm | 16.4 | 0.16 | 1010 | 16.2 | 16.3 | 16.4 | 16.6 |
| CatArrayBatchedCopy | 6.7 | 0.07 | 4020 | 1.7 | 1.7 | 1.9 | 2.1 |
| globalKernel | 5.1 | 0.05 | 4000 | 1.3 | 1.3 | 1.4 | 1.5 |
| splitKreduce_kernel | 4.8 | 0.05 | 4000 | 1.2 | 1.2 | 1.2 | 1.2 |
| ampere_sgemm_128x128_nn | 3.4 | 0.03 | 90 | 37.7 | 21.9 | 79.3 | 79.7 |
| DeviceSelectSweepKernel | 2.4 | 0.02 | 2020 | 1.2 | 1.2 | 1.3 | 1.3 |

### `implicit_convolve_sgemm`（重点）

| 项 | 值 |
| --- | --- |
| 总时间 | **9411.749 ms** = GPU kernel 时间的 **93.82 %** |
| 调用次数 | **27100**（avg 347.3 µs / p50 272.8 µs / p95 660.0 µs / max 1107.6 µs） |
| **每次 forward** | **2710 次调用，941.2 ms** |
| 每次去噪 step | 27.0 次 |

模板变体 2 个（与 `nsys stats --report cuda_gpu_kern_sum` 的分组口径一致）：

| calls | total_ms | 模板 |
| --- | --- | --- |
| 24020 | 9160.8 | `implicit_convolve_sgemm<float, float, (int)128, (int)5, (int)5, (int)3, (int)3, (int)3, (int)3, ...>` |
| 3080 | 250.9 | `implicit_convolve_sgemm<float, float, (int)1024, (int)5, (int)5, (int)3, (int)3, (int)3, (int)3, ...>` |

排名第 2 的 kernel 只占 1.32%。**这个模型在 kernel 层面是单点主导的**：优化 Diffusion Policy 的推理性能，几乎是优化 2710 次卷积这一件事。

---

## 4. Denoising step

逐 step 明细见 `denoising_steps.csv`（1000 行 = 100 步 × 10 次）。

| 指标 | avg | p50 | p95 | min | max |
| --- | --- | --- | --- | --- | --- |
| step latency | **13.226 ms** | 12.970 ms | 14.610 ms | 12.756 ms | 20.531 ms |
| kernels / step | 257.0 | 257 | 257 | — | — |
| conv calls / step | **27.0** | 27 | 27 | — | — |
| conv time / step | **9.408 ms** | 9.407 ms | 9.427 ms | — | — |
| conv % of step kernel time | **94.04 %** | 94.04 % | 94.05 % | — | — |

- **step 数**：1000（100 步 × 10 次），与 checkpoint 的 `num_inference_steps=100` 一致。
- 去噪占 forward 的 **91.57 %**。
- 每个 step 的 kernel 数、卷积数、卷积占比**完全恒定**（p50 = p95 = mean），符合 UNet 静态形状的预期。
- 单步 13.226 ms 里，卷积占 9.408 ms，剩 3.818 ms 是非卷积 kernel + 间隙。

`diffusion_denoising` 与子区间对账：`denoising_step` 13432.00 ms + `scheduler_step` 674.90 ms = 14106.90 ms，父区间 14366.39 ms，残差 259.50 ms（1.81 %）。

---

## 5. Kernel granularity / launch overhead

### 粒度分布

| 时长区间 | count | % | total_ms | % of kernel time |
| --- | --- | --- | --- | --- |
| **< 5 µs** | **237340** | **86.4 %** | 369.1 | **3.7 %** |
| 5–20 µs | 6110 | 2.2 % | 78.3 | 0.8 % |
| 20–100 µs | 6150 | 2.2 % | 256.3 | 2.6 % |
| 100–500 µs | 16010 | 5.8 % | 3173.3 | 31.6 % |
| **> 500 µs** | 9000 | 3.3 % | **6154.1** | **61.3 %** |

kernel 实例总数 **274610**，合 **每次 forward 27461 个**。
kernel 时长 avg 0.037 ms / p50 0.001 ms / p95 0.273 ms / max 1.108 ms。

**86.4% 的 kernel 只贡献 3.7% 的 GPU 时间**——这是典型的"小 kernel 发射风暴"。

### 相邻 kernel 间隙

| 指标 | 值 |
| --- | --- |
| gap 个数 | 274600 |
| avg | 0.016 ms（16 µs） |
| p50 | 0.001 ms（1 µs） |
| p95 | 0.077 ms（77 µs） |
| max | 5.087 ms |
| **total gap time** | **4412.400 ms**（区间头 6.862 + 区间尾 2.080 + 相邻间隙 4403.459） |
| kernel gap ratio | **30.55 %** |

间隙分布形状（直查 sqlite 计算）：

| 间隙区间 | 个数 | % | 合计 | 占全部间隙时间 |
| --- | --- | --- | --- | --- |
| < 2 µs | 202470 | 73.7 % | 129.8 ms | 2.9 % |
| 2–10 µs | 10950 | 4.0 % | 51.4 ms | 1.2 % |
| 10–50 µs | 40191 | 14.6 % | 1062.4 ms | 24.1 % |
| 50–200 µs | 17715 | 6.5 % | 2062.2 ms | 46.8 % |
| > 200 µs | 3274 | 1.2 % | 1097.7 ms | 24.9 % |

73.7% 的间隙 <2 µs（基本是背靠背），但**间隙时间集中在 20989 个 >50 µs 的间隙里**（71.7%）。
其中落在 `denoising_step` 区间内的有 16149 个，它们的归一化位置分布如下（区间尺度不同，仅统计 step 内的部分）：

| 位置（step 内归一化） | 个数 | % |
| --- | --- | --- |
| 0–10 % | 5745 | 35.6 % |
| 10–20 % | 4309 | 26.7 % |
| 20–30 % | 3982 | 24.7 % |
| 30–40 % | 1629 | 10.1 % |
| 40–100 % | 484 | 3.0 % |

**97% 的大间隙落在每个 step 的前 40%**——正是 UNet 下采样段小 kernel 最密集的位置。后 60% 被大卷积填满，几乎无间隙。

### 对照实验：间隙不是插桩造成的

入口的 `--sync-stages`（默认开启）在每个阶段边界插 `torch.cuda.synchronize()`，每 forward 约 412 次。为了确认 30.55% 的间隙不是它引入的，用 `--no-sync-stages` 重采了一次同配置对照：

| | sync=ON（本文主结果） | sync=OFF（对照） | 差异 |
| --- | --- | --- | --- |
| forward wall / 次 | 1444.363 ms | 1451.094 ms | **+0.5 %** |
| GPU kernel covered time | 10031.229 ms | 10049.498 ms | +0.2 % |
| GPU busy ratio | 69.45 % | 69.25 % | −0.20 pt |
| **kernel gap ratio** | **30.55 %** | **30.75 %** | **+0.20 pt** |
| kernel 实例数 | 274610 | 274610 | 0 |
| `cudaLaunchKernel` | 274610 次 / 2102.0 ms / avg 7.65 µs | 274610 次 / 2253.4 ms / avg 8.21 µs | 同量级 |
| `cudaDeviceSynchronize` | 4120 次 / 861.8 ms | 20 次 / 0.2 ms | 即插桩本身 |

**结论**：插桩确实带来了 861.8 ms 的 host 阻塞，但**没有拉长 forward，也没有改变间隙比例**。GPU 空闲 30.6% 是模型和运行时本身的性质，不是我们的观测方法造成的。因此主结果（sync=ON）可以放心引用。

### 下发成本核算

- `cudaLaunchKernel`：274610 次 × 7.65 µs = **2102.0 ms** host 侧开销（占 wall 14.6%）。
- 小 kernel（<5 µs）一侧：GPU 执行合计 369.1 ms，而 host 为它们付出的下发成本约 237340 × 7.65 µs ≈ **1815.6 ms**——**下发成本是执行时间的 4.9 倍**。
- 卷积一侧：27100 次 × 7.65 µs ≈ 207 ms host 成本 vs 9411.7 ms GPU 时间——**卷积被喂得很饱，不是下发受限**。

即：**GPU 时间被大卷积支配，wall time 被小 kernel 的下发开销拖累**，两者是不同的瓶颈。

---

## 6. GPU hardware metrics

**Current NSYS report is insufficient for strict compute-vs-memory classification; Nsight Compute hardware counters are required.**

该 `.nsys-rep` 不含 GPU Metrics 表（54 张表中无任何 `GPU_METRICS` 相关表，`META_DATA_CAPTURE` 的 capture type 仅 `OSRuntime, CuDNN, CuBLAS, NvtxEvents, Cuda`，`COLLECT_GPU_MEMORY_USAGE false`）。

### 为什么没有采集（不是漏加 flag）

本机**无法**采集 GPU 硬件计数器，`--gpu-metrics-device` 与 Nsight Compute 都会被驱动拒绝：

```console
$ nsys profile --gpu-metrics-devices=help
GPU Metrics: None of the installed GPUs are supported:
	Ada AD102 | NVIDIA GeForce RTX 4090 PCI[0000:e1:00.0] - Insufficient privilege,
	see https://developer.nvidia.com/ERR_NVGPUCTRPERM

$ ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed --launch-count 1 python ...
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

根因在驱动参数：

```console
$ cat /proc/driver/nvidia/params | grep Profiling
RmProfilingAdminOnly: 1
```

`RmProfilingAdminOnly: 1` 对应模块参数 `NVreg_RestrictProfilingToAdminUsers=1`（驱动 418 之后的默认值），只允许具备 `CAP_SYS_ADMIN` 的进程访问性能计数器。当前进程虽然是 uid 0，但在容器内：

```console
$ grep CapEff /proc/self/status
CapEff:	00000000a80425fb          # bit 21 (CAP_SYS_ADMIN) = 0
$ ls /sys/module/nvidia/parameters/
ls: cannot access '/sys/module/nvidia/parameters/': No such file or directory
```

即容器没有 `CAP_SYS_ADMIN`，`/sys/module/nvidia` 也未挂载，**容器内无法自行解开**。

注意 `nsys profile` 遇到该权限问题不是警告后继续，而是**直接拒绝启动**：

```console
$ nsys profile --gpu-metrics-devices=0 ... 
Illegal --gpu-metrics-devices argument: 0.
Insufficient privilege, see https://developer.nvidia.com/ERR_NVGPUCTRPERM.
```

所以"重跑一遍带上 metrics"在当前环境下会直接失败，而不是产出一份没有 metrics 的报告。

### 解除条件（需要宿主机侧操作，二选一）

**方案 A（推荐，根本解法）**——宿主机把参数置 0 并重载驱动：

```bash
# 宿主机
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiler.conf
sudo update-initramfs -u
sudo reboot        # 或卸载/重载 nvidia 模块
```

**方案 B**——容器以 `--cap-add=SYS_ADMIN` 启动（驱动参数保持 1 时，具备该 capability 即被视为 admin）。宿主机改完参数或容器重建后，重跑：

```bash
CUDA_VISIBLE_DEVICES=0 profiling/run_diffusion_policy_nsys.sh   # 需在 launcher 加 --gpu-metrics-device=0
```

`profiling/analyze_diffusion_policy.py` 已按 `has_gpu_metrics()` 自动探测；一旦报告里有 GPU Metrics 表，第 5 节内容会随之填充。

### 因此本节不作判定

**不能**根据 kernel 名称（例如看到 `implicit_convolve_sgemm` 就推断 compute-bound）或 memcpy 时间（第 2 节已显示显存操作只占 0.04%）来推断 compute-bound / memory-bound。要区分 SM 饱和、DRAM 带宽饱和与低占用率，必须采集 SM Active / SM Issue / Tensor Active / DRAM Throughput 等计数器。

---

## 7. Memory operations

| 类型 | calls | bytes | time | per forward |
| --- | --- | --- | --- | --- |
| HtoD | 1020 | 2,220,000 | 0.472 ms | 102 次 / 0.047 ms |
| DtoH | 2040 | 10,000 | 2.239 ms | 204 次 / 0.224 ms |
| DtoD | 0 | 0 | 0 | — |
| Memset | 4000 | 512,000 | 3.321 ms | 400 次 / 0.332 ms |
| **合计** | 7060 | — | **6.032 ms** | **0.60 ms（占 forward 0.04%）** |

传输量都很小：HtoD 平均每笔 2176 B（就是输入 observation 上卡），DtoH 平均每笔仅 4.9 B，Memset 平均每笔 128 B（`torch.zeros` 之类）。

**重要限定**：这些只是**显式 CUDA memcpy/memset**，**不代表 kernel 内部的 HBM 访存流量**。卷积、GEMM、elementwise 在 kernel 内部对显存的读写完全不体现在这张表里。因此本表**不能**用来判断 DRAM 带宽是否是瓶颈——那需要 GPU Metrics 的 DRAM Throughput 计数器（见第 6 节）。

---

## 8. Summary

| 字段 | 值 |
| --- | --- |
| Forward latency | **1444.363 ms** |
| GPU busy ratio | **69.45 %** |
| Conv/GEMM percentage | **93.82 %** |
| `implicit_convolve_sgemm` percentage | **93.82 %** |
| Conv calls | **27100**（2710 / forward） |
| Denoising steps | **1000**（100 × 10） |
| Conv calls / denoising step | **27.0** |
| Average denoising latency | **13.226 ms** |
| SM Active | n/a（报告不含 GPU Metrics） |
| SM Issue | n/a（报告不含 GPU Metrics） |
| Tensor Active | n/a（报告不含 GPU Metrics） |
| DRAM bandwidth | n/a（报告不含 GPU Metrics） |
| Kernel gap ratio | **30.55 %** |

### 倾向性判断

> **补充**：本节的判定后来由 [analysis-perturbation.md](analysis-perturbation.md) 的
> batch / 去噪步数扰动实验补全。扰动实验证明 batch 1→4 用同一批 kernel、同样耗时完成了
> 4 倍的 FLOPs（achieved 算力 0.35 %–3.1 % 峰值、权重流量折算带宽 4.5 %–8.2 % 峰值），
> 因此 **A（算力饱和）和 B（带宽饱和）都可以排除**，实际形态是 **C（小 GEMM 形状导致
> 并行度不足）+ D（kernel 间隙下发开销）**。下文"无法判定"的表述保留，作为**未经扰动
> 实验时**的保守立场。

**A / B / C 无法判定**，因为报告不含 GPU Metrics：

> Current NSYS report is insufficient for strict compute-vs-memory classification; Nsight Compute hardware counters are required.

**D（kernel gap 大 → launch/CPU/sync 开销）有证据成立**，且已排除插桩嫌疑：

- kernel gap ratio **30.55%**，用 `--no-sync-stages` 对照重采得到 30.75%，wall time 差异仅 0.5%——**插桩不是原因**。
- 间隙时间集中在 21k 个 >50 µs 的间隔上，其中 **97% 落在每个 `denoising_step` 的前 40%**，即小 kernel 最密集的下采样段。
- `cudaLaunchKernel` 274610 次、avg 7.65 µs，合计 2102.0 ms host 开销。
- 对 <5 µs 的 237340 个 kernel，host 下发成本（≈1815.6 ms）是其 GPU 执行时间（369.1 ms）的 **4.9 倍**。

**C（kernel 背靠背但利用率低）部分可能成立**，但仅凭时序无法确证：73.7% 的间隙 <2 µs 说明大部分 kernel 确实背靠背，这些 kernel 是否低占用率、卷积形状是否不利，需要 Nsight Compute。

**综合判断：既非纯 compute 饱和、也非纯带宽饱和，而是"大卷积吃饱 + 小 kernel 下发拖累"的混合形态。**

### 主要瓶颈（按影响排序）

1. **100 步去噪循环占 forward 的 91.6%，其中 941.2 ms 是每次 forward 2710 次 `implicit_convolve_sgemm`。** 这是唯一有数量级意义的优化目标。
2. **每次 forward 27461 个 kernel 的下发开销**：86.4% 的 kernel <5 µs 却只贡献 3.7% 的 GPU 时间，直接导致 30.55% 的 GPU 空转。方向是减少 kernel 数（融合 norm/elementwise、CUDA Graph 捕获去噪循环），而不是减少计算量。
3. 显存搬运（0.04%）和前后处理（vision_encoder 6.06 ms，0.42%）都可以忽略。

---

## 附：产物

| 文件 | 说明 |
| --- | --- |
| `analysis-forward-bottleneck.md` | 本文 |
| `summary.txt` | 命令行版汇总（同源） |
| `kernel_breakdown.csv` | 按 kernel name 聚合的完整明细 |
| `denoising_steps.csv` | 逐去噪 step 的 1000 行明细 |
| `control-nosync/` | `--no-sync-stages` 对照采集的 .nsys-rep / .sqlite / summary.txt |
| `gpu_metrics.csv` | **未生成**——报告不含 GPU Metrics，需先解除第 6 节的权限限制 |

分析脚本：`profiling/analyze_diffusion_policy.py`
