# E-VLA-test：具身模型实验与 Benchmark

本仓库保留开发源码、各版本实验设计与结果分析报告，用于共享研究进展与并行开发。**v1/v2 的中间结果和机器审计记录不上传；轻量运行汇总以 v3.0 及以后为主。** 重复源码快照不上传。**数据集、权重、视频、激活张量、原始逐观测产物、大型 trace、虚拟环境与缓存不随仓库上传。** 仿真资产和模型按需在实验机器上恢复或下载。

| 首次协作 | 入口 |
| --- | --- |
| 在新机器准备代码、环境与资产 | [远程复现说明](docs/setup/remote-reproduction.md) |
| 分支、并行分工、结果与提交规范 | [CONTRIBUTING.md](CONTRIBUTING.md) · [Agent 入口](AGENTS.md) |
| 首次导出范围、文件 hash、排除项 | [导出摘要](docs/repository/snapshot_summary.json) · [文件清单](docs/repository/snapshot_files.tsv) · [排除清单](docs/repository/excluded_files.json) |
| 各上游源码版本、本地改动与资产恢复清单 | [upstreams.json](docs/repository/upstreams.json) |

## 当前研究入口

以下链接指向本次快照中的记录。实验仍可能在原机器继续推进，文档中的 PID、GPU lease 和进行中状态不是远程任务的实时状态。

| 路线 | 方案与实现 | 结果与证据 |
| --- | --- | --- |
| idea1：层/模块跳过、去噪与多模型 | [方案与版本索引](docs/experiments/idea1/README.md) | 同一索引内各版本报告，包含阴性与中断记录 |
| EASE insights：draft 迁移与内部路径 | [准备和实现](preparations/ease_insights_20261103/) | [证据总览](reports/ease_insights_20261103/README_zh.md)；内部路径类别互补性尚未复现 |
| idea2：编码流水、KV 复用与批处理 | [实验文档](docs/experiments/idea2/) · [代码](preparations/idea2/) | [两阶段批处理报告](docs/experiments/idea2/idea2_batch_pipeline_two_stage_report_20261101.md) |
| π0.5 隐藏表征与读出 | [方案](idea1_pi05_hidden_representation_inspect_plan.md) · [实现](experiments/pi05_hidden_probe/README.md) | [发现与进度](docs/experiments/idea1/idea1_pi05_hidden_representation_report.md) · [首轮结果](results/pi05_hidden_probe/run_20261105/summary.md) |
| π0.5 表征对齐 | [实现](experiments/pi05_representation_alignment/) | [准备入口](experiments/pi05_representation_alignment/prepare.py) · [评估入口](experiments/pi05_representation_alignment/evaluate.py) |
| π0.5 SAE 语义与 FFN | [方案](pi05_sae_semantic_skipping_plan.md) · [实现](experiments/pi05_sae_semantic/README.md) | [首轮报告](results/pi05_sae_semantic/run_20261105/interim_results.md)；闭环质量与 GPU 净加速尚未验证 |

用于快速验证 VLA 研究 idea、复现开源策略，以及在 LIBERO、CALVIN、SimplerEnv 等环境中评测与分析性能。原工作区路径：`/root/nfs/embodied/benchmark`；新机器路径配置见远程复现说明。

## 从这里开始

| 需求 | 入口 |
| --- | --- |
| 规划实验、提高评测吞吐、多卡调度与断点恢复 | [评测加速 skill](.agents/skills/embodied-eval-acceleration/SKILL.md) |
| 查已有加速方法的实测收益、限制与实现 | [加速经验证据](.agents/skills/embodied-eval-acceleration/references/evidence.md) |
| 找模型加载、评测、分析和调度代码 | [实验代码用途索引](docs/code/README.md) |
| 查实验方案、进度与报告 | [实验文档目录](docs/experiments/) · [idea1 索引](docs/experiments/idea1/README.md) |
| 配置环境、查共享资产或重建依赖 | [环境与资产参考](docs/setup/benchmark-environments.md) |
| 采集 Nsight trace、核对计时边界 | [Profiling 说明](profiling/README.md) |

实验与评测的通用流程统一维护在 skill 中；可要求执行代理“使用 `$embodied-eval-acceleration` 规划本轮评测”。具体实验的协议、命令和结果维护在其文档中。

## 目录职责

| 目录 | 内容 |
| --- | --- |
| `docs/experiments/<实验名>/` | 方案、执行状态、分析报告与复现入口 |
| `docs/code/`、`docs/setup/` | 代码用途索引、环境与资产配置 |
| `.agents/skills/` | 可跨实验复用的操作流程与经验 |
| `policies/<项目>/` | 模型源码及各自的 `.venv`；小型独立实验也可在实验目录下放 `code/` |
| `LIBERO/`、`calvin/`、`SimplerEnv/` | Benchmark 源码及环境 |
| `preparations/<实验名>/` | 实验专用适配、准备与调度实现 |
| `scripts/`、`profiling/` | 通用入口、性能采集工具 |
| `shared/{datasets,checkpoints,caches}/` | 共享数据、权重与下载缓存 |
| `results/`、`reports/`、`logs/` | 按运行隔离的原始结果、分析产物与日志 |
| `nsys/`、`profile_results/` | Trace 与性能采集结果 |

新增实验先查现有代码和资产，方案与报告放在对应实验目录，避免继续堆放根目录。新增或迁移代码入口时更新用途索引；通用评测经验更新 skill，安装经验更新环境参考。

## 使用约定

- **环境隔离**：各项目使用自己的 `.venv/bin/python`，新建环境的 prompt 使用项目名称。OpenPI 和 FLASH 都提供 `openpi` 包，不能混装；server 与 client 可使用各自环境。
- **资产复用**：原始数据与权重优先通过共享路径复用；转换产物另存并记录来源。LIBERO 的源码资产、配置和 init states 必须与所属 checkout 匹配，详见环境参考。
- **版本管理**：本远程仓库统一管理展开后的源码快照。原工作区各项目独立 Git 的来源保存在 `docs/repository/upstreams.json`；修改共享实现前确认其他实验是否依赖它。
- **记录归属**：每轮实验保留独立 run ID，文档链接实际结果。GPU 调度、恢复与计时规则以 skill 和本轮方案为准；历史 PID、端口和验证状态不代表当前可用资源。

## 最小环境验证入口

以下脚本会实际创建仿真环境并执行动作，运行前按本轮资源安排选择设备：

| 环境 | 入口 | 原有环境位置 |
| --- | --- | --- |
| CALVIN | [verify_calvin.sh](scripts/verify_calvin.sh) | `calvin/.venv` |
| SimplerEnv | [verify_simpler_env.sh](scripts/verify_simpler_env.sh) | `SimplerEnv/.venv` |
| LIBERO | [eval_libero_test_policy.py](policies/my_policy/eval_libero_test_policy.py) | `LIBERO/.venv` |

安装命令、LIBERO 导入修复、各策略 server/client 环境、Docker EGL/Vulkan 配置和初次验证版本，均见 [环境与资产参考](docs/setup/benchmark-environments.md)。最新科学结论以对应实验报告为准。
