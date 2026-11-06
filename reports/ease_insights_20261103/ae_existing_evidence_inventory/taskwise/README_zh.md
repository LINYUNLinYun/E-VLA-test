# AE-L2 / AE-L8 逐 task 配对结果

本报告只分析历史 closed-loop pilot 的 800 条轻量 episode-log 记录：L2、L8 各 400。按 suite/task/trial/seed 配对，共 400 个 case pairs；没有读取视频、轨迹数组或模型输出。`taxonomy.json` 是预先冻结的 metadata-only 分类，仅按 `(suite, task_id)` 且核对 log instruction 与 taxonomy 的 benchmark/BDDL instruction 后连接，没有使用 outcomes 定类。

**总效应。** L2 成功 356/400，L8 354/400，配对差 +0.50 pp；两边各失败 44 / 46 条。成对 discordance：L2-only 19，L8-only 17；both-success 337，both-failure 27。以 40 个 `(suite, task)` 为 cluster 的探索性 bootstrap 95% CI 为 [-2.75, 3.75] pp。

| Suite | L2 | L8 | L2−L8 | n pairs | L2-only / L8-only | task-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| libero_10 | 80/100 | 80/100 | +0.0 pp | 100 | 7 / 7 | [-9.0, 9.0] pp |
| libero_goal | 81/100 | 84/100 | -3.0 pp | 100 | 5 / 8 | [-9.0, 2.0] pp |
| libero_object | 99/100 | 97/100 | +2.0 pp | 100 | 2 / 0 | [0.0, 5.0] pp |
| libero_spatial | 96/100 | 93/100 | +3.0 pp | 100 | 5 / 2 | [-2.0, 9.0] pp |

逐 task 成功数、paired outcome、L2−L8 差值与 discordance 见 `analysis.json` 的 `taskwise`；全部 400 个配对 case 行见 `paired_case_outcomes.jsonl`。总体 leave-one-task-out 均值范围为 -0.26 至 1.28 pp；完整端点及被省略 task 见 JSON。

## Metadata-only taxonomy 探索

按冻结 taxonomy 聚合得到点估计交叉：state-change-only 中 L8 高 5 pp（2 tasks）；state-change+placement 中 L2 高 2.5 pp（4 tasks）；multiple-placement 中 L8 高 5 pp（6 tasks）；single-placement 中 L2 高 1.79 pp（28 tasks）。这些是探索性 task-cluster bootstrap，95% CI 依次为 [-10,0]、[-10,20]、[-16.67,3.33]、[-1.07,4.64] pp，均未给出稳健方向证据。类别和 suite 几乎重合：state-only 只在 Goal；multiple-placement 只在 LIBERO-10；Spatial/Object 全是 single-placement。因此这些交叉不能区分 task semantics 与 suite/task 难度，也不能作为已验证偏好。

## Identity 与失败处理

八个 manifest 与 800 条 episode log 均覆盖每个 suite 的 task 0–9、episode index 0–9、run seed 7、每 task 10 条；L2/L8 的 key、task description、max_steps、num_steps_wait 一致，taxonomy instruction join 全部匹配。episode log 没有初始状态 digest，因此只能验证 case label/trial/seed 配对，不能证明物理初态图像逐字节相同。`failure_reason` 对 800 条全为 null，没有技术失败专用字段；所有 `success=false` 均保留在分母，未按原因删样本。

## 判读

该历史 screen 在 suite/task-taxonomy 点估计上有弱交叉，但总体差仅 +0.5 pp（L2−L8），40-task cluster CI 为 [-2.75,3.75] pp；leave-one-task-out 均值仍在 [-0.26,1.28] pp。它适合作为后续独立验证候选，不证明互补。主代理另行整理的 provenance 审计报告旧 pilot 名义配置为 H50/R12/D10、AE 全层路径中的 L2/L8 单层消融；旧归档缺 checkpoint/norm 内容哈希、完整 argv 和历史源码快照，噪声 SHA 也不是实际 GPU tensor 抓取。因此这里只分析结果标签，不把旧 pilot runtime 精确性写成已完全复核，更不能与当前 VLM local-search v1 的 R5/D2 结果合并。它不改变当前队列，也不支持在线适应收益。

详细来源与所有数值门见 `analysis.json`；源 episode logs 位于 `policies/realtime-vla-flash/results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/closed_loop/` 下各 run root。
