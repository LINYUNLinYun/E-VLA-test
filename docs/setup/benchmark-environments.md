# 环境、共享资产与重建参考

本页从根 README 的安装记录整理而来（整理日期：2026-10-24）。用于查找已有资产、隔离运行环境与按需重建；**版本、耗时、成功次数和早期下载范围均为历史记录，不是当前实验状态。** 已有环境优先复用，仅在确需重建时执行安装命令。

项目路径均相对 `/root/nfs/embodied/benchmark`；代码块中的绝对路径保持原有含义。历史启动示例的 GPU、端口和参数必须按当前任务调整，不代表资源预约。实验规模、GPU 等待／分片、断点恢复、吞吐选型统一见 [评测加速 skill](../../.agents/skills/embodied-eval-acceleration/SKILL.md)；当前结果见 [idea1 实验索引](../experiments/idea1/README.md)。

[返回项目入口](../../readme.md)

## 1. 资产与环境复用

### 1.1 数据集、权重和缓存复用

**同一份原始资产尽量只保存一次，以路径配置或软链接复用，作为只读输入。** 当前可复用入口如下：

| 资产 | 统一入口 | 使用边界 |
| --- | --- | --- |
| LIBERO HDF5 | `shared/datasets/libero_hdf5` | 软链接到 `LIBERO/libero/datasets`；新实验检查需要的 suite 是否齐全 |
| CALVIN debug | `shared/datasets/calvin_debug` | 软链接到 `calvin/dataset/calvin_debug_dataset`；仅用于调试和环境验证 |
| ALOHA 数据 | `shared/datasets/aloha_sim_transfer_cube_human`、`aloha_sim_transfer_cube_human_v21` | 两种格式分别保留，按加载器要求选择，不能仅凭任务名互换 |
| OpenPI 权重 | `shared/checkpoints/openpi` | OpenPI 与 FLASH 可复用下载目录；具体模型、格式和适配代码仍需匹配 |
| 其他策略权重 | `shared/checkpoints/<项目>/` | 先查已有目录，再决定是否下载或转换 |
| 下载与构建缓存 | `shared/caches/{huggingface,uv,xdg,wheels}` | 缓存共享不代表 Python 环境或依赖可以混用 |

从工作区根目录为新终端设置缓存（需要时使用）：

```bash
cd /root/nfs/embodied/benchmark
export HF_HOME="$PWD/shared/caches/huggingface"
export UV_CACHE_DIR="$PWD/shared/caches/uv"
export XDG_CACHE_HOME="$PWD/shared/caches/xdg"
export OPENPI_DATA_HOME="$PWD/shared/checkpoints/openpi"
```

复用前检查真实路径、文件完整性、数据格式、split 和模型 revision。软链接的源目录不能随意移动或删除。
LIBERO 可以共用 HDF5，但 `assets`、`bddl_files`、`init_states` 和配置应对应各自的源码 checkout；
不能为省事直接把另一项目的整份 `.libero/config.yaml` 照搬过来。
数据格式转换、过滤、重采样、特征提取应写入新目录，并记录输入版本、采样 seed、预处理和生成脚本。
已有特征缓存只有在模型权重、图像预处理、层、dtype、样本清单都匹配时才可直接复用。

### 1.2 代码和运行环境复用

- 优先复用已跑通的模型加载、observation 预处理、action 后处理、评测与 profiling 入口，先读对应 README。
  可参考 [profiling](../../profiling/README.md)、[idea1 实验索引](../../docs/experiments/idea1/README.md)
  和 [feature upsampling](../../docs/experiments/feature_upsampling/README.md)。
- 每个项目使用自己的 `.venv/bin/python`；server 与 client 可以属于不同环境，通过协议通信。
  OpenPI 和 FLASH 都叫 `openpi`，不要安装到同一环境，也不要让 `PYTHONPATH` 同时包含两份实现。
  切换项目建议使用新终端或显式指定解释器，并按本次任务重新设置路径变量。
- 先检查代码是否有未提交改动、后台任务是否还依赖它，再修改公共实现。
  并行实验需要不同源码时，在实际 Git 仓库建立独立 worktree 或 checkout，并检查其环境与子模块路径。
- 记录主仓库与子模块 commit，同时保存本地 diff；只记 commit 无法恢复未提交的实验修改。
  安装或升级依赖应明确指向目标解释器，避免破坏仍在运行的环境。

## 2. Benchmark 初次环境验证记录

| Benchmark | 官方仓库 | 本地路径 | uv 环境 | 验证状态 |
| --- | --- | --- | --- | --- |
| CALVIN | `https://github.com/mees/calvin` | `calvin/` | `calvin/.venv` (Python 3.8, prompt `calvin`) | GPU EGL reset/step 通过 |
| SimplerEnv | `https://github.com/allenai/SimplerEnv` | `SimplerEnv/` | `SimplerEnv/.venv` (Python 3.10, prompt `SimplerEnv`) | SAPIEN reset/step 通过 |
| LIBERO | `https://github.com/Lifelong-Robot-Learning/LIBERO` | `LIBERO/` | `LIBERO/.venv` (Python 3.8, prompt `libero`) | GPU EGL reset/step 通过 |

CALVIN 与 SimplerEnv 都已使用 `--recurse-submodules` 拉取官方子模块（LIBERO 无子模块）。初次配置记录为 4 张 NVIDIA GeForce RTX 4090、驱动 580.95.05；这不表示当前有四张空闲卡。 

## 3. Profiling 策略环境

| 策略 | 官方仓库路径 | benchmark | uv 环境 |
| --- | --- | --- | --- |
| OpenVLA-OFT | `policies/openvla-oft/` | 独立 LIBERO checkout | `.venv`，Python 3.10，prompt `openvla-oft` |
| π0.5 server | `policies/openpi/` | OpenPI 官方 `third_party/libero` | `.venv`，Python 3.11，prompt `pi05-server` |
| π0.5 LIBERO client | `policies/openpi/examples/libero/` | 同上 | `.venv`，Python 3.8，prompt `pi05-libero-client` |
| Diffusion Policy | `policies/diffusion_policy/` | 仓库内置 image Push-T | `.venv`，Python 3.9，prompt `diffusion-policy` |
| Realtime-VLA FLASH server | `policies/realtime-vla-flash/` | 独立 LIBERO checkout | `.venv`，Python 3.11，prompt `realtime-vla-flash` |
| FLASH LIBERO client | `policies/realtime-vla-flash/examples/libero/` | 同上 | `.venv`，Python 3.8，prompt `realtime-vla-flash-libero-client` |

源码和运行环境彼此隔离。OpenVLA-OFT 的 LIBERO 固定为
`8f1084e3132a39270c3a13ebe37270a43ece2a01`；OpenPI 使用主仓库锁定的
`third_party/libero` commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`；
Realtime-VLA FLASH 的 `third_party/libero` 锁定**同一个** commit
`f78abd68…`（源码逐字节相同），但仍保留独立 checkout。

已下载的数据不复制：`shared/datasets/libero_hdf5` 指向原有的
`LIBERO/libero/datasets`，`shared/datasets/calvin_debug` 指向 CALVIN debug
数据。各套 LIBERO 各有独立配置，源码资产来自各自 checkout，只有数据路径共享：

- OpenVLA-OFT：`policies/openvla-oft/.libero/config.yaml`
- π0.5 client：`policies/openpi/examples/libero/.libero/config.yaml`
- FLASH client：`policies/realtime-vla-flash/.libero/config.yaml`

> **注意**：这些 checkout 同样缺少 `libero/__init__.py`，`pip install -e LIBERO` 会静默装出
> 空包。已建好的环境均可运行；启动脚本会通过 `PYTHONPATH` 显式加入对应 LIBERO
> checkout。根因和手工修法见 [LIBERO 段落](#5-libero)。

进入环境和设置运行时路径：

```bash
# OpenVLA-OFT
cd /root/nfs/embodied/benchmark/policies/openvla-oft
source .venv/bin/activate
export LIBERO_CONFIG_PATH="$PWD/.libero"
export HF_HOME=/root/nfs/embodied/benchmark/shared/caches/huggingface

# π0.5 server
cd /root/nfs/embodied/benchmark/policies/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi

# π0.5 LIBERO client
cd /root/nfs/embodied/benchmark/policies/openpi
source examples/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH="$PWD/examples/libero/.libero"
export PYTHONPATH="$PWD/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"

# Diffusion Policy
cd /root/nfs/embodied/benchmark/policies/diffusion_policy
source .venv/bin/activate

# FLASH server（与 π0.5 server 共用同一份 checkpoint 目录）
cd /root/nfs/embodied/benchmark/policies/realtime-vla-flash
source .venv/bin/activate
export OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi

# FLASH LIBERO client
cd /root/nfs/embodied/benchmark/policies/realtime-vla-flash
source examples/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH="$PWD/.libero"
export PYTHONPATH="$PWD/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
```

初次 profiling 的三个官方 checkpoint 均已下载并完成稳定态 inference profiling。统一入口、采集边界、重建方式和
输出说明见 [`profiling/README.md`](../../profiling/README.md)，结果位于 `profile_results/`：

| 策略 | checkpoint | 实测 benchmark 通路 | 结果 |
| --- | --- | --- | --- |
| OpenVLA-OFT | `moojink/openvla-7b-oft-finetuned-libero-spatial`，revision `6d0231a…` | LIBERO-Spatial reset → 双图像/proprio observation → 8×7 action → step | `profile_results/openvla-oft/` |
| π0.5 | `gs://openpi-assets/checkpoints/pi05_libero` | LIBERO-Spatial reset → server inference → 10×7 action → step | `profile_results/pi05/` |
| Diffusion Policy | Push-T image seed 42, epoch 500, score 0.884 | Push-T reset → 96×96 image observation → 8×2 action → step | `profile_results/diffusion-policy/` |

每个目录包含 manifest、运行日志、逐次 wall-clock 统计和可由 `nsys stats` 解析的
`.nsys-rep`。采集为 batch size 1、5 次 warmup、
10 次稳定态调用；模型加载、首次编译及 simulator reset/step 均在 capture 外。

### 3.1 Realtime-VLA FLASH 初次跑通记录

`policies/realtime-vla-flash/` 是 openpi 的 fork，多出 `scripts/spec/` 下的投机推理链路，
但包名仍是 `openpi`，因此**必须和 `policies/openpi/.venv` 分开**。已在 LIBERO-Goal 上
跑通 speculative server（Triton 后端）+ LIBERO client 的完整通路。

资产集中放 shared，不重复下载：

| 用途 | 位置 |
| --- | --- |
| 基座 JAX checkpoint `pi0_libero` | `shared/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero`（12G，GCS 匿名可读） |
| draft 权重 | `shared/checkpoints/realtime-vla-flash/draft_libero_goal.pt`（441M，HF `Dexmal/RealtimeVLA-Flash`） |
| Triton 转换产物 | `shared/checkpoints/realtime-vla-flash/converted/{base,draft}`（`base_weights.pkl` 6.5G） |

HF 上每个 suite 各有一份 draft（`draft_libero_{goal,spatial,object,10}.pt`），初次跑通时只下载了
`libero_goal`；后续已评测四套件对应 draft，见 [v3.2 报告](../experiments/idea1/idea1_v3_2_report.md)。转换产物不放进仓库树，因为 `.gitignore` 只忽略 `data/`。

服务端（`--tokenizer-source local` 走 openpi 自带的 PaliGemma tokenizer，绕开脚本默认的
`hf-mirror.com` 端点；GPU 用 `CUDA_VISIBLE_DEVICES` 钉住，避免和机器上其他任务抢卡）：

```bash
cd /root/nfs/embodied/benchmark/policies/realtime-vla-flash
CK=/root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/spec/spec_serve_policy.py \
  --config pi0_libero --backend triton \
  --base-triton-path $CK/converted/base --draft-triton-path $CK/converted/draft \
  --task-suite-name libero_goal --tokenizer-source local --pytorch-device cuda:0
```

客户端（`MUJOCO_GL=egl`，渲染单独占一张卡）：

```bash
cd /root/nfs/embodied/benchmark/policies/realtime-vla-flash
LIBERO_CONFIG_PATH="$PWD/.libero" PYTHONPATH="$PWD/third_party/libero" \
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \
examples/libero/.venv/bin/python scripts/spec/spec_client_libero.py \
  --task-suite-name libero_goal --task 0 --num-trials-per-task 1 --host 127.0.0.1
```

实测（libero_goal task 0，输出在 `data/test/<run>/`）：成功 1/1，119 env steps，
10 次推理调用，draft 路由占比 0.89，视频与 trace 均落盘。

> **前两次调用是 warmup，不是稳态**：第 0 次 41.6 s（加载 6.5 GB 权重 + VLM prefill），
> 第 1 次 13.3 s（Triton kernel 首次编译，draft 5.3 s + verify 8.0 s）；第 2 次起往返
> 12–14 ms，服务端 `total_ms` ≈ 7.8 ms（enc 4.6 / draft 0.9 / verify 2.2）。直接读
> manifest 里的 `infer_latency_mean_ms` 会得到 4.67 s 这种被 warmup 污染的数，
> 做 profiling 时必须按 `profile_window` 口径排除前两次。

## 4. 快速使用与验证

```bash
cd /root/nfs/embodied/benchmark

# CALVIN：会用已下载的 debug 数据集创建环境并执行一步动作
./scripts/verify_calvin.sh

# SimplerEnv：会创建 google_robot_pick_coke_can 并执行一步动作
./scripts/verify_simpler_env.sh
```

手动进入环境：

```bash
source /root/nfs/embodied/benchmark/calvin/.venv/bin/activate
# 或
source /root/nfs/embodied/benchmark/SimplerEnv/.venv/bin/activate
# 或
source /root/nfs/embodied/benchmark/LIBERO/.venv/bin/activate
```

## 5. LIBERO

### 5.1 环境配置

使用 Python 3.8.13（uv 托管的 CPython，prompt `libero`），全部 Python 包通过 `uv pip` 安装。
以下命令按当前 `.venv` 的实际版本反推，依赖版本与 `LIBERO/requirements.txt` 逐条一致：

```bash
cd /root/nfs/embodied/benchmark/LIBERO

uv venv --python 3.8 --prompt libero .venv
uv pip install --python .venv/bin/python -r requirements.txt

# requirements.txt 未包含 PyTorch，需单独安装
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu113 \
  'torch==1.11.0+cu113' 'torchvision==0.12.0+cu113' 'torchaudio==0.11.0+cu113'
```

### 5.2 必须手动补 `libero.pth`（重要）

官方 `setup.py` 用 `find_packages()` 收集包，但仓库里 `libero/` 这一层**没有 `__init__.py`**
（上游就没有）。setuptools 会把非包目录直接跳过、且不再向下递归，所以：

```
$ python -c "from setuptools import find_packages; print(find_packages())"
[]
```

于是 `pip install -e .` **返回成功、写出 `dist-info`，但注册的映射表是空的** —— 装了等于没装。
这是最容易踩的坑，因为它完全不报错：`libero-0.1.0.dist-info/` 该有的都有，`import libero`
却照旧 `ModuleNotFoundError`。

老版本 pip 走 `setup.py develop`，会把仓库根目录写进 `easy-install.pth`，靠 Python 的
namespace package 机制让 `import libero` 蒙混过关；新版 pip/uv 改用 PEP 660，这个副作用没了，
问题才暴露出来。修法是手动把根目录加回 `sys.path`：

```bash
echo /root/nfs/embodied/benchmark/LIBERO > \
  .venv/lib/python3.8/site-packages/libero.pth
```

`libero.pth` 位于 venv 内部，而 `.venv/` 被 git 忽略 —— **每次重建 venv 后都要重跑这一条。**

### 5.3 验证

```bash
cd /root/nfs/embodied/benchmark/policies/my_policy
MUJOCO_GL=egl /root/nfs/embodied/benchmark/LIBERO/.venv/bin/python eval_libero_test_policy.py
```

在 LIBERO-Spatial task 0 上执行 10 步动作并输出 MP4，可确认 offscreen 渲染与 OSC_POSE 控制通路正常。

### 5.4 配置

LIBERO 通过 `~/.libero/config.yaml` 定位 `bddl_files` / `init_states` / `assets`，当前已指向本
checkout。需要环境隔离时可在运行前设 `LIBERO_CONFIG_PATH`，用法见上面的策略环境表格。

## 6. CALVIN

### 6.1 环境配置

CALVIN 官方依赖较旧，需要 Python 3.8、NumPy 1.23.5，并用旧 setuptools/cmake 构建 `pyhash` 和 `MulticoreTSNE`。重建命令如下，全部 Python 包均通过 `uv pip` 安装：

```bash
cd /root/nfs/embodied/benchmark
git clone --recurse-submodules https://github.com/mees/calvin.git calvin
cd calvin

uv venv --python 3.8 --prompt calvin .venv
uv pip install --python .venv/bin/python \
  'setuptools==57.5.0' wheel 'cmake==3.18.4.post1' 'numpy==1.23.5'

PATH="$PWD/.venv/bin:$PATH" uv pip install --python .venv/bin/python \
  --no-build-isolation 'MulticoreTSNE==0.1' 'pyhash==0.9.3'

PATH="$PWD/.venv/bin:$PATH" uv pip install --python .venv/bin/python \
  -e calvin_env/tacto -e calvin_env -e calvin_models
uv pip install --python .venv/bin/python 'numpy==1.23.5'
```

CALVIN 仓库根目录下的 `calvin_env/` 目录与 Python 包同名，直接从根目录运行 Python 会触发 namespace 遮蔽。训练和评测应按官方方式从 `calvin/calvin_models/calvin_agent/` 或 `calvin/calvin_models/` 运行。

### 6.2 数据集

已下载官方 debug 数据集（约 1.3 GB）：

- 解压路径：`calvin/dataset/calvin_debug_dataset/`
- 原始压缩包：`calvin/dataset/calvin_debug_dataset.zip`
- 校验文件：`calvin/dataset/sha256sum.txt`
- 完整性验证结果：`calvin_debug_dataset.zip: OK`

重新下载 debug 集：

```bash
cd /root/nfs/embodied/benchmark/calvin/dataset
bash download_data.sh debug
wget http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt
sha256sum --check --ignore-missing sha256sum.txt
```

官方还提供 D→D（166 GB）、ABC→D（517 GB）和 ABCD→D（656 GB）完整数据。下载任意一个 split：

```bash
cd /root/nfs/embodied/benchmark/calvin/dataset
bash download_data.sh D       # 保存到 task_D_D/
bash download_data.sh ABC     # 保存到 task_ABC_D/
bash download_data.sh ABCD    # 保存到 task_ABCD_D/
```

训练时把数据根目录传给 `datamodule.root_data_dir`：

```bash
cd /root/nfs/embodied/benchmark/calvin/calvin_models/calvin_agent
../../.venv/bin/python training.py \
  datamodule.root_data_dir=/root/nfs/embodied/benchmark/calvin/dataset/calvin_debug_dataset \
  datamodule/datasets=vision_lang
```

debug 数据集用于环境验证和调试；正式训练应选择 D、ABC 或 ABCD split。服务器无桌面，`scripts/visualize_dataset.py` 的 OpenCV 交互窗口需要 X11 转发才能使用。

## 7. SimplerEnv

### 7.1 环境配置

当前安装的是官方“minimal requirements”，可运行全部预打包仿真环境。RT-1/Octo/MolmoAct 等策略推理属于官方的 full installation，需另外下载模型权重。

```bash
cd /root/nfs/embodied/benchmark
git clone --recurse-submodules https://github.com/allenai/SimplerEnv.git SimplerEnv
cd SimplerEnv

uv venv --python 3.10 --prompt SimplerEnv .venv
uv pip install --python .venv/bin/python 'numpy==1.24.4' 'setuptools==80.9.0'
uv pip install --python .venv/bin/python -e ManiSkill2_real2sim
uv pip install --python .venv/bin/python -e .
```

`setuptools<81` 是 SAPIEN 2.2.2 导入时使用 `pkg_resources` 所需的兼容约束。

### 7.2 资产与数据

SimplerEnv 的最小环境运行不需要额外数据集；场景、物体和机器人资产已包含在官方子模块中：

- `SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/assets/`
- `SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/`

运行 RT-1 等预训练策略需要相应 checkpoint；统一保存到 `shared/checkpoints/<项目>/`，按加载器配置路径。具体下载命令见 `SimplerEnv/README.md` 的 Full Installation 部分。

## 8. Docker / GPU 注意事项

SAPIEN 使用 Vulkan，CALVIN/PyBullet 使用 EGL。新建容器时应暴露 NVIDIA graphics capability，例如：

```bash
docker run --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  ...
```

当前容器已补齐 `libvulkan1`、`mesa-vulkan-drivers` 以及与宿主机驱动严格匹配的 NVIDIA 580.95.05 Vulkan/GL 用户态库。如果在其他机器重建容器，NVIDIA 库版本必须与宿主机驱动一致，不要直接照抄 580.95.05。

## 9. 初次验证的版本快照

- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`
- OpenPI: `215abfb217dbac7d5f1273282331b9b1866c0479`
- Diffusion Policy: `5ba07ac6661db573af695b419a7947ecb704690f`
- CALVIN: `fa03f01f19c65920e18cf37398a9ce859274af76`
- `calvin_env`: `1431a46bd36bde5903fb6345e68b5ccc30def666`
- `tacto`: `dd53360d9a8c186f0d6439372ec0be0fa5e21731`
- SimplerEnv: `c701ea271b16b80616d6fed9713c4a50a0adea3b`
- `ManiSkill2_real2sim`: `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`
- LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
