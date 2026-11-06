# 在新机器上复现

本仓库保留各版本设计与最终分析，并提供当前源码、图表和轻量汇总。v1/v2 的中间结果和机器审计不上传；轻量运行记录以 v3.0 及以后为主，重复源码副本不上传。数据集、模型、视频、激活张量、原始逐 episode/观测/特征目录、逐案例 JSON receipt、完整请求流、Nsight trace、Python 环境及下载缓存需在实验机器上准备。首次导出来自仍有实验运行的工作区，各文件有独立 hash，但跨文件不构成暂停所有任务后的事务快照。

## 1. 获取代码与选择实验

```bash
git clone https://github.com/LINYUNLinYun/E-VLA-test.git
cd E-VLA-test
python3 scripts/restore_upstream_assets.py --list
```

源码按原工作区目录展开保存，包含本地修改；无须 `git submodule update`。上游地址、commit、本地 tracked 改动文件和被排除的仿真资产路径见 [upstreams.json](../repository/upstreams.json)。上游许可证继续适用于相应目录。

先阅读根目录 [readme](../../readme.md)、[实验代码索引](../code/README.md)及所选实验方案。历史脚本、配置和报告中有 `/root/nfs/embodied/benchmark` 绝对路径。新机器可在该路径检出；使用其他路径时，应修改**新运行的参数或配置**，不要改历史报告、receipt 和被冻结的输入清单。部分脚本还在源码中定义工作区路径，运行前用 `rg '/root/nfs/embodied/benchmark' <入口目录>` 检查。

## 2. 按需恢复仿真资产

例如 OpenPI 使用自带的 LIBERO checkout：

```bash
python3 scripts/restore_upstream_assets.py --repo policies/openpi/third_party/libero
```

FLASH 对应：

```bash
python3 scripts/restore_upstream_assets.py --repo policies/realtime-vla-flash/third_party/libero
```

独立 LIBERO、CALVIN 与 SimplerEnv 可分别选择：

```bash
python3 scripts/restore_upstream_assets.py --repo LIBERO
python3 scripts/restore_upstream_assets.py --repo calvin/calvin_env --repo calvin/calvin_env/tacto
python3 scripts/restore_upstream_assets.py --repo SimplerEnv/ManiSkill2_real2sim
```

恢复器会临时 fetch 指定上游 commit，复制清单中的缺失资产，保留已有文件。首次网络下载可能包含该上游 commit 的其他 Git 对象；不会将这些对象加入本仓库。它只恢复清单中的上游仿真资产，不下载训练数据、策略权重、演示视频或本地生成张量。`--list` 没有网络操作。

LIBERO 的 `bddl_files`、`assets`、`init_states` 应来自同一份 checkout。重新生成各 client 的 `.libero/config.yaml`；不要沿用原机器的绝对路径。详见 [环境参考](benchmark-environments.md)。

## 3. 环境隔离

沿用每个项目的 `pyproject.toml`、requirements、锁文件和自己的 README，历史已验证组合见 [环境参考](benchmark-environments.md)。仓库并非一个可统一 `pip install` 的 Python 包。

- OpenPI 和 FLASH 都提供 `openpi` 包，分别建 `.venv`；server/client 可用不同解释器。
- 已有源码目录不再执行历史安装记录中的同路径 `git clone`。
- 重建 LIBERO 时检查 namespace 包导入修复和 `PYTHONPATH`。
- 本次上传仅检查仓库内容与恢复工具；没有重新运行所有仿真和 GPU 实验。

在仓库根目录设置本机缓存：

```bash
mkdir -p shared/checkpoints shared/datasets shared/caches
export HF_HOME="$PWD/shared/caches/huggingface"
export UV_CACHE_DIR="$PWD/shared/caches/uv"
export XDG_CACHE_HOME="$PWD/shared/caches/xdg"
export OPENPI_DATA_HOME="$PWD/shared/checkpoints/openpi"
```

## 4. 下载权重、数据与公开 SAE

| 用途 | 来源与入口 |
| --- | --- |
| OpenPI π0 / π0.5 | [OpenPI README](../../policies/openpi/README.md)、`gs://openpi-assets/checkpoints/pi0_libero` 和 `gs://openpi-assets/checkpoints/pi05_libero`；使用原 loader 的下载实现，或按 README 下载到 `shared/checkpoints/openpi/` |
| FLASH draft | Hugging Face `Dexmal/RealtimeVLA-Flash` 的 `draft_libero_{goal,spatial,object,10}.pt`；[FLASH 说明](../../policies/realtime-vla-flash/README.md)与[环境参考](benchmark-environments.md)记录转换和路径合同 |
| OpenVLA-OFT | Hugging Face `moojink/openvla-7b-oft-finetuned-libero-spatial`；精确 revision 见 `docs/repository/external-assets/model_sources.json` |
| 其他策略 | 对应 `policies/<项目>/README*` 与实验 manifest；格式、归一化、revision 和 action horizon 需匹配 |
| LIBERO HDF5 | [LIBERO 下载入口](../../LIBERO/benchmark_scripts/download_libero_datasets.py)；也可用其 `scripts/` 下下载工具，按 suite 下载 |
| CALVIN | `bash calvin/dataset/download_data.sh debug` 的脚本需要在 `calvin/dataset/` 执行；debug 仅作环境检查，正式 split 另按方案准备 |
| Action Atlas | 先读 [SAE 实验入口](../../experiments/pi05_sae_semantic/README.md)及[当前 run manifest](../../results/pi05_sae_semantic/run_20261105/run_manifest.json)中三个 `public_release_revisions`；两层 SAE 的下载入口为 `experiments/pi05_sae_semantic/prepare_assets.py` |

Action Atlas 的历史固定 revision 为：`bag100/action-atlas-pi05@a7a717259aaadceb6896b8d4885fa72a104ad4d6`、`bag100/action-atlas-pi05-activations@82e03132f9119ea51fd522bb9ad4d5593aaba503`、`bag100/action-atlas-viz@ea09a0881066bddd3456096e1d9ab9d9cf67981d`。历史 `prepare_assets.py` 会先查询发布仓库当前版本；精确复现时须用 Hugging Face `hf_hub_download(..., repo_type='dataset', revision='<上述 SHA>')` 获取所需文件，再核对 manifest，不能把当前最新版当作历史版。

某些 PyTorch 权重由 JAX 权重本地转换而来；重新执行对应项目的转换脚本，保留转换参数与版本。仅下载名称相近的权重无法替代实验使用的格式与归一化。公开资产仍受原发布方的访问条件约束。

## 5. 理解缺失的大产物

[导出摘要](../repository/snapshot_summary.json)记录筛选规则和数量，[排除清单](../repository/excluded_files.json)记录被检查但未纳入的文件及原因。整个 `shared/`、虚拟环境和依赖缓存不逐文件列举。按原始产物目录批量排除的候选在摘要中计数；二进制后缀、环境与缓存不逐文件扫描。排除清单列出其余经过大小或内容检查的文件，仿真资产路径单列在上游版本清单中。

报告中的历史绝对路径、PID、端口、GPU UUID、lease 和 `running` 状态只描述采集时环境。不要据此恢复旧调度器或假定 GPU 被本仓库占有。先为本次实验建立新的输出目录和资源安排。

原始观测、图像数据、激活、完整 trace 或大矩阵可能需要重新采集才能运行后续分析；被排除的本地产物未自动上传到其他存储，本仓库也没有为它们生成虚构下载地址。轻量结果的保留不等价于提供了所有原始数据。
