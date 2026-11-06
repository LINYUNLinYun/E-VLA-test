# 既有 Action Expert 裁剪证据盘点

**结论：**有一对可用于 VLM 搜索若为 null 后独立验证的 AE 候选：只跳过 Action Expert 第 2 层（L2）或第 8 层（L8）。两者各跳过一个同形状完整 AE block，名义计算预算接近；两者都通过固定观测敏感度筛选，也有真实闭环 pilot。四个 suite 的闭环总成功率只有 0–3 个百分点差异，且轻微方向变化；汇总没有逐 task 成功数、配对检验或计时，不能据此认定存在稳定任务偏好。它们是合适的**下一条独立搜索轴**，不是互补 fast path 的已有证据。

本盘点只读了现有产物，不运行模型、不接触当前 local-search v1 队列或科学源码。当前 VLM local-search v1 只变更 observation-prefix mask、AE 保持 full18；若后续测 L2/L8，必须另建科学 manifest，不能把 AE 候选塞进现有 VLM 搜索条件。

## Action Expert L2 与 L8

候选固定观测 probe 的主记录在 [probe summary](</root/nfs/embodied/benchmark/policies/realtime-vla-flash/results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/probe/summary.json>)，门槛在 [prespecified criteria](</root/nfs/embodied/benchmark/policies/realtime-vla-flash/results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/probe/prespecified_criteria.json>)。该 probe 有 40 个 task、110 个固定 observation、3 个模型噪声 seed，统计每候选 330 个 observation-seed，未运行环境闭环。AE L2 的 task-mean NMSE 为 0.00259（task bootstrap 95% CI 0.00033–0.00542），夹爪符号翻转率 0.00069；L8 为 0.00409（0.00104–0.00851），翻转率 0.00093。二者均通过预设低影响门槛：前 12 动作 task-mean NMSE ≤0.01、bootstrap 上界 ≤0.02、夹爪翻转 ≤0.01、最差阶段 task-mean NMSE ≤0.05，并且每 suite/phase 至少 3 个 task。该门槛仅筛固定 observation 的动作敏感度，不代表闭环非劣。

[AE 闭环 pilot 汇总](</root/nfs/embodied/benchmark/policies/realtime-vla-flash/results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/closed_loop/screening_summary.json>)标记 complete，共 800 episodes；L2、L8 各在四个 suite 跑 100 episodes。8 份 rollout manifest 均给出 task 0–9、每 task 10 trials、seed 7、episode index 0–9、`num_steps_wait=10`、`replan_steps=12`。这是实际 LIBERO 闭环运行，但 manifest 没有记下 H 或 D；`R=12` 是 `replan_steps` 字段，不能替代完整的策略配置。L2/L8 的逐 task 成功数没有出现在汇总文件中，因此下面只列可核验的 suite 总数：

| Suite | L2 成功 | L8 成功 | 差值 L2−L8 | 每候选有效 N |
|---|---:|---:|---:|---:|
| LIBERO-Goal | 81/100 | 84/100 | −3 pp | 10 task × 10 trials |
| LIBERO-10 | 80/100 | 80/100 | 0 pp | 10 task × 10 trials |
| LIBERO-Spatial | 96/100 | 93/100 | +3 pp | 10 task × 10 trials |
| LIBERO-Object | 99/100 | 97/100 | +2 pp | 10 task × 10 trials |

这些总数中，L8 在 Goal 高 3 pp，L2 在 Spatial 高 3 pp、Object 高 2 pp、LIBERO-10 持平，看起来有弱方向变化，但没有逐 task 配对统计或置信区间，不能把它写成稳定 preference。该 pilot 的 screening 汇总只包含 L2 与 L8，没有 Full 配对 baseline，也没有实际延迟/计算量测量。候选名与总览表标为单层 `SkipAE_L2`、`SkipAE_L8`，所以两者是**同一个 AE block 数量**的名义预算对照；实际部署计时仍缺。

数值门的审计也需保留边界：[registered pattern parity](</root/nfs/embodied/benchmark/policies/realtime-vla-flash/results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/closed_loop/registered_pattern_parity.json>)对 L2/L8 的 6 个注册样本报告 bitwise parity；同时目录保留了早期 [pattern parity failure](</root/nfs/embodied/benchmark/policies/realtime-vla-flash/results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/closed_loop/pattern_parity_failure.json>)，曾在 L2 的保存结果与 graph/eager 比较中记录最大绝对差 0.00963。后续若复用需以新的已冻结 parity receipt 作为启动门，不能只引用通过样本。

## Attention / MLP 记录不是 AE 的同成本候选

[PI0 Attention / FFN 诊断报告](</root/nfs/embodied/benchmark/policies/realtime-vla-flash/results/pi0_attention_ffn_diagnosis/final_report.md>)里的 `skip_attn_layers` / `skip_ffn_layers` 对应 [pi0_infer.py 的 `transformer_encoder`](</root/nfs/embodied/benchmark/policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py:922>)，函数注释明确它运行 VLM prefix encoder，不是 AE decoder。早期固定 12-task 单初态对照中，Full D10 为 7/12，Attention L16 为 9/12，FFN L2 为 10/12；这些结果来自不同子模块/层位，不能视为已匹配的 AE attention-versus-MLP 预算对。

较大的 Attention-only follow-up 也没有形成 AE attention/MLP pair。配置明确为 `candidate_skip_attn_layers=[16]` 或 `[17]`，`candidate_skip_ffn_layers=[]`。Attn-L16 在 LIBERO-10 为 19/20、LIBERO-90 为 21/40；对应 Full 为 17/20 与 22/40。Attn-L17 分别为 17/20 与 14/40；对应 Full 为 17/20 与 22/40。所有计时均标为 eager diagnostic、无 latency claim。它表明 VLM prefix 某些层位影响闭环结果，但不填补 AE 子模块的缺口，也不能外推成 AE 任务偏好。

## 对当前研究的用途

如果 VLM local-search v1 的独立确认没有发现互补，L2/L8 可作为下一阶段有限的 AE 轴：固定其他设置，按新初态逐 task 配对比较两种单层 AE mask，并同时加入 Full 对照；先冻结并核实 H、D、噪声、skip 的真实实现和 runtime parity，再检查实际延迟。已有数据足以把 L2/L8 放进候选清单，但不足以预先指定它们各自擅长的类别。当前 VLM 队列与 manifest 无需更改。

## 后续日志追溯

[有界配置追溯](provenance/README_zh.md)已从原请求与server日志支持H50/R12/D10、AE17/18（L2/L8为1-based）的历史配置；32个抽样请求契约通过。上文H/D缺失指rollout manifest本身，不能再理解为完全没有历史请求证据。闭环checkpoint/norm内容hash、完整argv及历史源码快照仍缺；噪声SHA是独立重建报告值，未抓取实际GPU输入tensor。与当前R5结果不能合并。

## 逐任务与固定切分复核

[400对逐case分析](taskwise/README_zh.md)已完成：L2成功356/400，L8成功354/400，差+0.5pp；40-task区间[-2.75,3.75]pp，四个既有语义类别区间均含0。[固定前后trial切分](retrospective_split/README_zh.md)用前5选路径、后5评估，两个选法均181/200，相对训练选出的固定L2多2例，条件区间[0,2.5]pp。均为历史探索，尚未补上可信可复现偏好的证据。原始800条episode包含全部success=false；缺少技术失败字段和物理初态hash。当前科学队列与选择规则不变。

## 独立重建

[独立审阅结果](taskwise/independent_audit/independent_audit.json)直接从8份manifest和8份episode log重建800条记录、400个配对case，核对来源SHA及配对artifact逐字段一致；overall/suite/task/固定taxonomy、LOO和记录seed/draws下的bootstrap数值均匹配。可复现实现见 [rebuild_audit.py](taskwise/independent_audit/rebuild_audit.py)。原日志的物理初态hash与历史模型/源码内容绑定缺口仍然保留；统计一致不使历史探索变成独立复验。
