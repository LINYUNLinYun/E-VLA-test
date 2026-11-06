# 实验代码索引

本仓库保留各版本设计和结果分析报告；轻量运行汇总以 v3.0 及以后为主。各模型、benchmark 的当前源码及本地修改保留在原目录，旧 v1/v2 中间结果及机器审计、重复源码副本和大产物不上传。

| 需求 | 入口 |
| --- | --- |
| 各版本方案与分析（含 v1/v2） | [idea1 索引](../experiments/idea1/README.md) |
| v3/v4 主线实现 | `preparations/idea1_v3*/`、`preparations/idea1_v4*/` |
| EASE draft 迁移和内部路径 | `preparations/ease_insights_20261103/`、[证据总览](../../reports/ease_insights_20261103/README_zh.md) |
| π0.5 隐藏表征与读出 | [实验实现](../../experiments/pi05_hidden_probe/README.md)、`experiments/pi05_self_readout/` |
| π0.5 SAE 语义与 FFN | [实验实现](../../experiments/pi05_sae_semantic/README.md) |
| 视觉编码器 A/B | [实现与协议](../../experiments/pi05_vision_ab/README.md) |
| 编码流水、KV 和批处理 | `preparations/idea2/`、[实验文档](../experiments/idea2/) |
| KV 复用探测 | `preparations/kv_reuse_probe/`、[报告](../experiments/kv_reuse_probe/report.md) |
| OpenPI、FLASH、其他模型 | `policies/<项目>/` 中的源码、安装配置和 README |
| Benchmark 和最小验证 | `LIBERO/`、`calvin/`、`SimplerEnv/`、`policies/my_policy/`、`scripts/verify_*.sh` |
| 推理性能采集 | [Profiling 工具](../../profiling/README.md) |
| 新机器准备资产与环境 | [复现说明](../setup/remote-reproduction.md) |

## 旧目录中的必要依赖

`preparations/idea1_v2_1/` 的模型适配和 RPC、`preparations/idea1_v2_2/` 的原生模型与采样日程，仍由多个 v3+ 模块直接调用。FLASH `scripts/idea1_v2*/` 中的 Kalman 实现也被 v3.2 使用。部分 `scripts/experiment1/6/8/10/11` 提供 decoder、轨迹、后端和预检组件。这些目录作为依赖保留，旧版的设计与最终分析见实验文档索引，中间产物不随之上传。

OpenPI 与 FLASH 都提供 `openpi` 包，继续使用各自的独立 Python 环境。历史代码中的原工作区绝对路径应在新运行的参数和配置中调整；不要修改历史报告来伪装新测量。

## 静态清单

[逐文件清单](inventory.tsv)与[统计](inventory_summary.json)由[索引脚本](../../scripts/index_workspace_code.py)生成，包含当前发布的主要源码目录、符号、CLI 参数和文件 hash。它只做静态扫描，不导入项目、不使用 GPU，也不代表所有实验已经重新验证。运行目录中少量独立辅助脚本以对应实验 README 为入口。

历史视觉特征上采样的[设计与最终分析](../experiments/feature_upsampling/README.md)也保留，特征张量和运行日志不上传。
