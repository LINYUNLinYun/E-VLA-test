# 推理 profile

本目录的入口脚本在运行时插入 NVTX 标记，**不改动三个官方 policy 源码树**，
使用 benchmark 真实 observation，默认 5 次 warmup + 10 次 batch=1 推理采集。

最外层固定两个区间：`profile_window` 控制 nsys capture 范围（排除模型加载、
编译和 warmup），`policy_infer` 表示一次完整的策略推理。`model_forward` 把纯模型
计算从输入/输出处理里分出来。

## 运行

从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 profiling/run_openvla_nsys.sh
CUDA_VISIBLE_DEVICES=0 profiling/run_diffusion_policy_nsys.sh
CUDA_VISIBLE_DEVICES=0 profiling/run_pi05_nsys.sh
```

Diffusion Policy 环境用 uv 按官方 Python 3.9/CUDA 11.6 依赖重建，推理期依赖记录在
`diffusion_policy_inference_requirements.txt`，重建命令：

```bash
profiling/setup_diffusion_policy_env.sh
```

pi0.5 的命令只采集 GPU server 进程；LIBERO client 跑在独立的 Python 3.8 环境里，
刻意放在 nsys 之外。

三个 launcher 都设置 `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`：Python/PyTorch 的
push-pop 区间用的是动态字符串而非 NVTX 注册字符串，不设它 `--capture-range=nvtx`
不会触发采集。

## GPU 硬件计数器本机不可用

`--gpu-metrics-device` 和 Nsight Compute 在本机都会被驱动拒绝，报
`ERR_NVGPUCTRPERM`。根因：

```console
$ cat /proc/driver/nvidia/params | grep Profiling
RmProfilingAdminOnly: 1          # 即 NVreg_RestrictProfilingToAdminUsers=1
$ grep CapEff /proc/self/status
CapEff:	00000000a80425fb        # bit 21 (CAP_SYS_ADMIN) = 0
```

只允许具备 `CAP_SYS_ADMIN` 的进程访问性能计数器；容器内即使 uid 0 也没有该
capability，且 `/sys/module/nvidia` 未挂载，**容器内解不开**。

`nsys profile` 遇到该问题不是警告后继续，而是直接拒绝启动：

```console
$ nsys profile --gpu-metrics-devices=0 ...
Illegal --gpu-metrics-devices argument: 0.
Insufficient privilege, see https://developer.nvidia.com/ERR_NVGPUCTRPERM.
```

所以不要往 launcher 里加这个 flag——会直接跑不起来，而不是产出没有 metrics 的报告。
解除需要宿主机侧操作：把 `options nvidia NVreg_RestrictProfilingToAdminUsers=0` 写进
`/etc/modprobe.d/` 后重载驱动/重启；或让容器以 `--cap-add=SYS_ADMIN` 启动。

**在没有 GPU Metrics 的报告上，不要根据 kernel 名称或 memcpy 时间推断
compute-bound / memory-bound。** 该结论必须由 SM Active / SM Issue / Tensor Active /
DRAM Throughput 计数器支撑。

## 阶段边界必须同步

PyTorch 和 JAX 都是异步下发。**区间在 Python 返回时关闭，而不是在设备算完时关闭**，
所以不加同步的话区间度量的是下发时刻，相邻阶段会互相串味。实测 OpenVLA-OFT 不同步时
`vision_encoder` 区间 59.6 ms 里只有 12.3 ms 是 GPU kernel，而约 16 ms 的大 BF16 GEMM
在 `action_head` 关闭之后才排空，被记到了相邻阶段头上。

因此三个入口默认都开启阶段边界同步（`--sync-stages`，可用 `--no-sync-stages` 关闭
做对照）：

- PyTorch：`torch.cuda.synchronize()`
- JAX：`jax.block_until_ready()`。**不能用 `jax.effects_barrier()`** —— 它只建立执行
  顺序、不阻塞 host（实测 0.13 ms，而 `block_until_ready` 3.51 ms），拿它当同步等于
  没同步。

同步会牺牲阶段间的流水重叠，因此每次运行的 manifest 里同时记录：

- `stage_accounting_ms`：进程内按同一组边界测的墙钟统计
- `nvtx_reconciliation`：每个父区间与声明的子区间求和的差额
  （`unaccounted_ms` / `unaccounted_fraction`），用来判断标记是否完整

## 各模型的 NVTX 层级

`profiling/attribute_gpu_time.py` 把 GPU 执行片段按时间归属到区间，用于核对每个阶段
真正占用了多少设备时间。`gpu/wall` 越低说明该阶段越是 CPU 下发 / 等待受限，而不是
真的在算——做结论时必须区分这两者。

```bash
python3 profiling/attribute_gpu_time.py profile_results/openvla-oft/openvla-oft.sqlite
```

**不要用 `nsys stats --report cuda_gpu_kern_sum` 判断 GPU 占用。** JAX/XLA 默认开启
command buffer（CUDA graph），这部分计算只记录在 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`，
`CUPTI_ACTIVITY_KIND_KERNEL` 里一条都没有。实测 pi0.5 只读后者得到 75.4 ms GPU 时间，
漏掉 480.6 ms，占比 86%；换算下来 gemma_2b 的 prefill 只有 0.99 ms，低于权重带宽下限，
一眼就是错的。归属脚本会同时读两张表，并按区间取并集避免重复计数。

OpenVLA-OFT 和 Diffusion Policy 不使用 CUDA graph，两张表的结论一致。

### Diffusion Policy（Push-T）

```
profile_window
└── policy_infer
    └── model_forward
        ├── observation_preprocess
        ├── vision_encoder
        ├── diffusion_denoising
        │   ├── denoising_step          ← 每个去噪 step 一次，默认开启
        │   └── scheduler_step
        └── action_postprocess
```

`denoising_step` 的步数由 checkpoint 的 `num_inference_steps` 决定（Push-T seed 42 为
100 步）。入口会断言论元数等于 `num_inference_steps × iterations`，对不上直接报错，
免得再出现"区间在但没标上"的情况。

### OpenVLA-OFT（LIBERO-Spatial）

```
profile_window
└── policy_infer
    ├── input_preprocess            ← 图像 resize/crop/normalize + processor，纯 CPU
    └── model_forward
        ├── vision_encoder
        ├── multimodal_projector
        ├── proprio_projector
        ├── vlm_backbone
        ├── action_head
        └── action_postprocess
```

### π0.5（LIBERO-Spatial，JAX）

官方 `Policy` 用 `nnx_utils.module_jit(model.sample_actions)` 把整次推理编成**一个**
XLA 模块。jit 内部的 Python 层 NVTX 只在 trace 期执行一次，运行时不会触发，所以原本
只能看到 `model_forward` 一整块。

`--component-breakdown`（默认开启）把同一份官方计算重组成四个各自 jit 的组件，
每个组件外面套 NVTX：

```
profile_window
└── policy_infer
    ├── observation_preprocess      ← 官方 Policy._input_transform
    ├── model_forward
    │   ├── model_preprocess        ← _model.preprocess_observation
    │   ├── vision_encoder          ← PaliGemma.img（SigLIP So400m/14）
    │   ├── vlm_backbone            ← 文本嵌入 + 前缀组装 + PaliGemma.llm 前缀前向 -> KV cache
    │   └── flow_matching
    │       └── denoising_step × num_steps
    │           └── action_expert   ← embed_suffix + gemma_300m 后缀前向
    │                               + action_out_proj + Euler 更新
    └── action_postprocess          ← 官方 Policy._output_transform
```

代价是失去跨组件 XLA 融合、每次组件调用多一层 dispatch。因此：

- `monolithic_reference` 记录官方单体路径（`--no-component-breakdown`）的延迟，用于
  量化这份开销；实测约 85.7 ms → 93.2 ms（+8.8%）。
- `decomposition_equivalence` 校验拆解路径与官方单体路径在同一 rng、同一 noise 下的
  数值一致性（`max_abs_diff` / `mean_abs_diff` / `allclose_atol_1e-2`），不通过就不该
  相信拆解出来的分布。

#### 请求生命周期

拆解路径需要让每个组件各自完成 XLA 编译，编译不能落进 capture window，因此请求分四段，
总数由 launcher 计算后同时传给 server 和 client：

| 阶段 | 默认次数 | 用途 |
| --- | --- | --- |
| `reference_warmup` | 2 | 官方单体路径预热 |
| `reference` | 5 | 官方单体路径计时 -> `monolithic_reference` |
| `breakdown_warmup` | 3 | 拆解路径预热（含各组件编译） |
| `capture` | 10 | 进入 `profile_window` 的正式采集 |

可用 `PI05_REFERENCE_WARMUP` / `PI05_REFERENCE_ITERATIONS` / `PI05_WARMUP` /
`PI05_ITERATIONS` 覆盖。

## 扰动实验（判断 compute / memory bound）

拿不到硬件计数器时，用 batch 和去噪步数两个旋钮做扰动标定，可以从标度律反推受限类型：

```bash
policies/diffusion_policy/.venv/bin/python profiling/perturb_diffusion_policy.py \
  --batches 1,2,4,8,16 --steps 10,25,50,100 --reps 3 --burn-in 5
```

脚本同时用 forward hook 统计 Conv/Linear 的 FLOPs，配合参数量算出 achieved TFLOPS 与
权重流量折算带宽，与 datasheet 峰值比对。`--nvtx` 模式给计时区间套 NVTX，供 nsys 采集
kernel 级数据（注意该模式下一次只能给一个 `(batch, steps)` 组合，因为 `--capture-range`
一次只认一个 `profile_window`）。`--cudnn-benchmark` 打开 autotuning 做对照。

两个必须注意的点：

- **`--burn-in` 不能省。** 网格从低负载配置开始，不预热的话 GPU 还在低频率，前几个点会虚高。
- **nsys 本身带来约 19% 的观测开销**（同配置 1216 ms → 1444 ms）。引用绝对延迟时用不带
  nsys 的数字，nsys 结果只用于 kernel 级归因。

Diffusion Policy 的实测结果见 `profile_results/diffusion-policy/analysis-perturbation.md`：
batch 1→4 延迟不变而 FLOPs 翻 4 倍，compute / memory 两侧利用率都低一个数量级，
且 batch ≥ 6 后延迟台阶来自 cuDNN 切换卷积算法（implicit GEMM → direct grouped → FFT）。

## 产物

每条命令在 `profile_results/<policy>/` 下写入 manifest、逐次墙钟样本、最后的 action 和
`.nsys-rep`。manifest 是唯一事实来源：精确命令、环境变量、仓库 commit、checkpoint 标识、
解析后的输入形状与配置、包版本、迭代次数和原始计时样本都在里面。`.sqlite` 和
`nsys-stats.csv` 是派生文件，可从 `.nsys-rep` 重新生成：

```bash
rm -f profile_results/<policy>/<name>.sqlite profile_results/<policy>/nsys-stats.csv
nsys stats --report cuda_gpu_kern_sum --format csv \
  profile_results/<policy>/<name>.nsys-rep > profile_results/<policy>/nsys-stats.csv
```

先删 `.sqlite` 再生成，输出头部才是 `Generating SQLite file ...`；否则 nsys 会复用旧
导出并打印 `NOTICE: Existing SQLite export found`，容易把陈旧结果当成新的。

查看 NVTX 区间树：

```bash
nsys stats --report nvtx_pushpop_sum profile_results/<policy>/<name>.nsys-rep
```

## 直接查 sqlite

`.sqlite` 就是普通 SQLite 库，比 `nsys stats` 灵活。需要 `sqlite3` CLI
（`apt-get install sqlite3`；没有的话用 Python 的 `sqlite3` 模块，等价）。

按名字汇总 NVTX 区间：

```bash
sqlite3 -header -column profile_results/diffusion-policy/diffusion-policy.sqlite "
SELECT COALESCE(e.text, s.value) AS range, COUNT(*) AS n,
       ROUND(SUM(e.end-e.start)/1e6,1) AS total_ms
FROM NVTX_EVENTS e LEFT JOIN StringIds s ON s.id = e.textId
WHERE e.end IS NOT NULL GROUP BY 1 ORDER BY 3 DESC LIMIT 10;"
```

**统计 GPU 时间必须同时查两张表**（原因见上一节）：

```bash
for t in CUPTI_ACTIVITY_KIND_KERNEL CUPTI_ACTIVITY_KIND_GRAPH_TRACE; do
  sqlite3 profile_results/pi05/pi05-server.sqlite \
    "SELECT '$t', COALESCE(COUNT(*),0), ROUND(COALESCE(SUM(end-start),0)/1e6,1) FROM $t;" 2>/dev/null
done
```

pi0.5 上这条会打出 `KERNEL 13640 rows / 75.4 ms` 和 `GRAPH_TRACE 6870 rows / 480.6 ms`；
只读前者就漏了 86%。`NVTX_EVENTS` 没有 `parentId` 列，嵌套关系靠区间包含隐含表达。

`sqlite3` CLI 版本需不低于生成该库的 nsys 所用版本；本机为 3.37.2，与
`/usr/bin/python3` 的 `sqlite3` 模块同版本。
