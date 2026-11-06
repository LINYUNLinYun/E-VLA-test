# 首次 Full 调用的 target 与归一化身份审计

**完整 160 对中 159 对动作前缀精确相等，1 对未通过 `atol=rtol=1e-5`。** 最新固定截面是 2026-11-03 13:28:58 UTC；Full 40 条、FLASH 四来源各 40 条均已覆盖，无缺失、重复或请求读取错误。保留成功与失败结果，未根据 outcome 选样。

逐项结果见 [固定 ledger](audit_20261103T132858Z.json)，实际参与比较的完整首次请求行已复制到 [请求快照](first_requests_20261103T132858Z.json)。`ledger.json` 是最近一次运行结果；带时间戳的文件保留各截面。较早 13:26:47 UTC 截面覆盖 153 对，当时 Long 来源的 Long task3–9 尚未完成，已保留未覆盖清单。

| Draft 来源 | FLASH/Full 配对 | raw 前缀精确相等 | decoded 前缀精确相等 | 容差通过 |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 40 | 39 | 39 | 39 |
| Object | 40 | 40 | 40 | 40 |
| Goal | 40 | 40 | 40 | 40 |
| Long | 40 | 40 | 40 | 40 |

160 对的两端都满足首次请求 `replan_id=0`、`full=1/draft=0/verify=0`，且同 case 的 input hash、noise hash/seed、prompt、state8 完全一致。各 run 的 base checkpoint 路径、normalization 说明与科学协议一致，冻结的 `server.py`、`common.py` hash 与实际文件一致。纯 draft 路由明确排除，没有拿纯 draft 动作与 Full 动作做身份比较。

唯一不匹配是 **Spatial 来源 → Spatial task1/init20**，Full 与 FLASH 的最终回合均成功：

- Full 请求：`draft_discovery/full_libero_spatial/requests.jsonl` 第 9 行。
- FLASH 请求：`draft_discovery/flash_libero_spatial/requests.jsonl` 第 1 行。
- raw 前缀最大绝对差 `0.0078125`，decoded 前缀最大绝对差 `0.00390387135027348`；两者各 28/84 个元素超过容差。
- 其他三个来源在同一 case 的首次 Full 动作均与 Full baseline 精确相等。该差异恰好位于 Spatial FLASH 服务的第一条请求；现有证据未定位原因，不能据此归因于冷编译或忽略差异。没有重跑或覆盖科学回合。

另以标准库标量运算独立重建全部 **200 条**首次 Full 输出的解码：从固定 `converted/base/assets/physical-intelligence/libero/norm_stats.json` 读取 action mean/std，计算 `raw[j] * (std[j] + 1e-6) + mean[j]`，前 6 维再加 `state8[j]`。200/200 均通过，最大绝对误差仅 `5.551115123125783e-17`，包括上述 raw 不匹配的一条。该资产的 SHA 与逐条重建检查写入 ledger；因此已覆盖数据支持共享归一化，不能将 raw 数值差异解释为不同 draft 来源使用不同输出归一化。

这项检查只覆盖保存的 12×7 首次动作前缀，不代表 H50 全部输出、后续分叉观测、target 权重的重新字节审计、成功率或独立复验结论。现有 target 元数据一致，但不能声称所有首次 target 动作均在 1e-5 内相同。

复验脚本只使用 Python 标准库，读取完成结果与请求日志，不导入模型代码、不初始化 GPU、不修改实验输出。比较键是 source +（target suite、task）+ init，避免不同 suite 同 task 编号混配。启动时的完整 result 文件限定快照成员，追加中的 JSONL 仅读到读取时文件大小；未完成的尾行被记录并忽略。每个被用到的请求行与 result 均保存来源路径、行号/哈希。

```bash
python3 -B /root/nfs/embodied/benchmark/reports/ease_insights_20261103/target_identity/audit.py
```

当前脚本退出码 **1** 是上述真实数值不匹配的审计结果。退出码 0 只代表已覆盖比较通过；是否完整必须另看 `summary.full_matrix_covered`。比较函数已检查精确一致、容差内小差异、超容差、NaN 与形状错误五类边界。
