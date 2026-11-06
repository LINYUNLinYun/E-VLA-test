# 目标文档
## 1. 目标
对openvla oft、pi0.5、diffusion policy三种机器人策略进行profile

## 2. 需求
- 文档与代码注释请用中文

- 调通三种模型的benchmark
- openvla oft采用libero、diffusion policy 采用push T、pi0.5 采用libero
- 采取官方仓库
- 使用uv管理环境，如果官方使用conda则需改成uv的形式
- 虚拟环境文件夹名为.venv，prompt为名称，如openvla-oft则prompt为openvla-oft
- 暂时只做inference 的profiling，暂定使用nvidia nsys
- 我不是要直接横比，只是先测下这三类模型的性能负载特点
- 理论上测量性能负载特点无所跑完所有demo，跑一点代表性的demo能得到nsys文件即可
- 本阶段完成环境、推理入口、基础采集和原始结果保存；严格的跨模型比较协议与结果分析留到下一阶段。


## 目录结构
```
benchmark/
├── LIBERO/
├── calvin/
├── SimplerEnv/
│
├── policies/
│   ├── openvla-oft/
│   ├── openpi/
│   └── diffusion_policy/
│
├── shared/
│   ├── checkpoints/
│   ├── datasets/
│   └── caches/
│
└── profile_results/
    ├── openvla-oft/
    ├── pi05/
    └── diffusion-policy/
```


## 项目组织约定

- OpenPI 保留官方 `third_party/libero` submodule。
- OpenVLA-OFT 使用独立、干净、固定 commit 的 LIBERO checkout。
- 数据集和 checkpoint 可以集中共享，源码环境不共享。
- 已经下载的数据不移动、不重复下载，通过 `shared/datasets/` 下的软链接统一索引。
- OpenPI 允许使用两套环境：
  - `.venv`，prompt 为 `pi05-server`
  - `examples/libero/.venv`，prompt 为 `pi05-libero-client`

## 暂定模型变体

- OpenVLA-OFT：L1 action head、双图像和 proprio、LIBERO-Spatial。
- π0.5：官方 JAX `pi05_libero` checkpoint、LIBERO-Spatial。
- Diffusion Policy：image-based Push-T CNN/UNet checkpoint。

## 阶段验收

- 三个官方模型均能加载官方 checkpoint 并完成少量 inference。
- 对应 benchmark 能完成 reset、observation、policy inference 和 step。
- 每个模型至少生成一个可被 `nsys stats` 正常解析的 `.nsys-rep`。
- 保存仓库 commit、checkpoint、环境版本、运行命令、环境变量和日志。



## profile
- 暂时只做做inference profile。不测 backward、optimizer 和训练数据加载。插入两个：
  - `profile_window`：控制 nsys capture 范围，排除模型加载、编译和 warmup。
  - `policy_infer`：标记一次完整的策略推理。
- 使用官方预训练 checkpoint。
- checkpoint 只负责提供模型参数，NVTX 插入本地推理代码。
- checkpoint 加载、模型初始化和首次编译不计入稳定态推理结果。
- 暂不直接比较三个模型的绝对性能，只分析各自的计算和负载特点。
结构：
```
profiling/
├── common.py
├── diffusion_policy_profile.py
├── openvla_oft_profile.py
├── pi05_server_profile.py
└── manifests/
```
- 先生成可解析的 `.nsys-rep`，结果分析下一步再做。
- batch size 固定为 1。
- 使用 benchmark reset 后产生的真实 observation。
- simulator reset/step 不放入 profile_window。
- checkpoint 加载、模型初始化和 JIT/首次编译不进入 profile_window。
- 编译完成后执行 5 次 warmup，随后默认采集 10 次真实 policy 调用；实际次数写入 manifest。
- 每次 policy_infer 在范围结束前确保输出 ready。
- policy_infer 表示完整用户调用路径，model_forward 表示纯模型计算。
- PyTorch 内部阶段使用 NVTX 做 kernel 归因，**阶段边界必须同步 CUDA**（`--sync-stages`，默认开启）。
  - 原因：PyTorch 异步下发，不同步时 NVTX 区间关闭于 Python 返回时刻而非设备完成时刻，区间会互相串味。实测 OpenVLA-OFT 不同步时 `vision_encoder` 区间 59.6 ms 内只有 12.3 ms 是 GPU kernel，而 ~16 ms 的大 BF16 GEMM 在 `action_head` 关闭之后才排空，被算进了相邻阶段。
  - 同步的 JAX 实现是 `jax.block_until_ready()`，**不是 `jax.effects_barrier()`**：后者只建立执行顺序、不阻塞 host（实测 0.13 ms vs 3.51 ms），拿它当同步等于没同步。
  - 同步后每个区间的父子求和必须在 manifest 的 `nvtx_reconciliation` 里对得上。PyTorch 侧 `unaccounted_fraction` 超过 5% 视为标记不完整；π0.5 侧允许更高残差，因为官方 `Policy.infer` 里的输入变换、`jnp.asarray` 上卡、输出 `.cpu()` 回传写在同一函数体内，不改官方源码就标不出来，这部分必须如实披露而不是摊到相邻阶段。
- JAX 内部阶段拆成独立 jit 组件分别计时，端到端 `policy_infer` 仍为主要测量结果。
  - 整段推理被 `nnx_utils.module_jit` 冻结成单个 XLA 模块时，Python 层的 NVTX 在运行时不会触发，因此无法在 jit 内部打点；拆成 `model_preprocess` / `vision_encoder` / `vlm_backbone` / `action_expert` 四个各自 jit 的组件后才能分别标记。
  - 拆解路径必须与官方单体路径做数值等价校验，结果写入 manifest 的 `decomposition_equivalence`。
- nsys 使用 `--capture-range=nvtx --nvtx-capture=profile_window`。
- 同时用 `time.perf_counter_ns()` 保存逐次 wall-clock 时间，计时结束前完成框架同步。
- 输入图像分辨率和相机数量按官方默认值，所有解析后的实际值写入 manifest。
- profile 前从 benchmark 获取真实 observation；capture 内复用该 observation，不执行 simulator reset/render/step。
- 保留 `policy_infer` 和 `model_forward` 两个层级，分别表示完整调用和模型计算。


#### Diffusion Policy

- 官方仓库：`policies/diffusion_policy/`
- Benchmark：Push-T
- 模型：image-based CNN/UNet Diffusion Policy
- 使用官方 seed 42（`train_0`）Push-T image CNN checkpoint：
  `epoch=0500-test_mean_score=0.884.ckpt`
- checkpoint URL、文件 SHA256 和 checkpoint 内解析出的配置写入 manifest。
- checkpoint 必须解析为 `DiffusionUnetHybridImagePolicy`，不能替换为 lowdim 或 transformer 变体。
- Profile 入口：`policy.predict_action(obs)`

至少标记：
- observation_preprocess
- vision_encoder
- diffusion_denoising
  - denoising_step（**默认开启**，逐去噪 step 标记）
    - 步数由 checkpoint 的 `num_inference_steps` 决定（Push-T seed 42 为 100 步），
      采集到的样本数必须等于 `num_inference_steps x iterations`，否则入口直接报错。
    - 与 `scheduler_step` 一起必须能与 `diffusion_denoising` 对账。
  - scheduler_step
- action_postprocess

#### OpenVLA-OFT

- 官方仓库：`policies/openvla-oft/`
- Benchmark：LIBERO-Spatial
- Checkpoint：`moojink/openvla-7b-oft-finetuned-libero-spatial`
- 配置：
  - L1 action head
  - 双图像输入
  - proprio 输入
  - center crop
  - BF16
  - 不使用 4-bit/8-bit 量化
- Profile 入口：`get_vla_action(...)`
至少标记：
- `input_preprocess`（图像 resize/crop/normalize + processor，纯 CPU）
- `model_forward`
  - `vision_encoder`
  - `multimodal_projector`
  - `vlm_backbone`
  - `proprio_projector`
  - `action_head`
  - `action_postprocess`
- 每个阶段区间的 GPU 占比用 `profiling/attribute_gpu_time.py` 核对，
  `gpu/wall` 偏低的阶段说明是 CPU 下发受限而非 GPU 计算受限，结论里必须区分这两者。
- **GPU 占用不能看 `nsys stats --report cuda_gpu_kern_sum`**：JAX/XLA 默认开启 command
  buffer（CUDA graph），这部分只记录在 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`。实测 pi0.5
  只读 kernel 表会漏掉 86% 的 GPU 时间（75.4 ms vs 480.6 ms），据此会得出
  "gemma_2b prefill 只要 0.99 ms" 这种低于带宽下限的错误结论。

#### π0.5

π0.5 内部阶段由 `--component-breakdown`（默认开启）拆成四个各自 jit 的组件，
组件边界有真实设备同步，因此区间与 GPU 实际执行边界一致。
端到端 `policy_infer` 仍是主要可信测量边界，拆解开销由 `monolithic_reference` 量化。

- 官方仓库：`policies/openpi/`
- Benchmark：LIBERO-Spatial
- 配置：`pi05_libero`
- Checkpoint：`gs://openpi-assets/checkpoints/pi05_libero`
- 使用官方 JAX policy server
- `nsys` 必须采集 GPU server，而不是只采集 LIBERO client
- Profile 入口：`policy.infer(observation)`
- JAX 输出必须调用 `jax.block_until_ready()`
- 拆解路径必须记录官方单体参考延迟，并做数值等价校验（`decomposition_equivalence`）

标记：
- `observation_preprocess`
- `model_forward`
  - `model_preprocess`
  - `vision_encoder`
  - `vlm_backbone`
  - `flow_matching`
    - `denoising_step` × `num_steps`
      - `action_expert`
- `action_postprocess`
- 阶段同步用 `jax.block_until_ready()`，不能用 `jax.effects_barrier()`（不阻塞 host）


