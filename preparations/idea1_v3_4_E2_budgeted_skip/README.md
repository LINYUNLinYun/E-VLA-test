# 可调时间预算的 Attention / FFN skip

在每次 Full 评分后，根据硬件成本和当前模块分数，混合选择 Attention、FFN 或同层两者。预算可指定为希望省下的毫秒数，或者完整 VLM encoder 耗时的比例；跳过的模块数量由选择器决定。

## 预算口径

- `saved_fraction=1/3`：目标省下完整 VLM encoder 基准耗时的三分之一。分母不包含 Vision、action expert、Full 评分和请求传输，因此不代表整次请求加速三分之一。
- `saved_ms=6`：目标省下 6 ms VLM 时间。
- 默认容差为 **±0.5 ms**。选择器在这个窗口内最小化影响分数；目标是窗口中心，不承诺至少省下中心值。可以显式缩小容差；窗口中没有合法 mask 时会报错，不自动放宽。
- 预算使用已测的各层 A / F / AF 净节省时间。同层 AF 使用直接测量的联合成本。多个层的总节省使用加和估计，实际耗时仍须在 GPU 上验证。
- 零预算只允许空 mask；正预算排除空 mask。超过当前间隔和可用性约束的最大节省时间会报错。

默认成本表来自 `results/idea1_v3_4_E2_mixed_cost/hardware_probe/cost_table.json`。比例分母读取其同目录 `request_timings.jsonl` 中未插入 profiling 的 `phase=hot, route=keep, vlm_ms`：544 条记录的中位数为 **19.23583984375 ms**。于是 20% 对应 3.847168 ms，1/3 对应 6.411947 ms。当前合法 mask 最大估计节省 9.046509 ms（约 47.0%），所以 50% 预算不可行。

如使用新硬件测量，将 `--cost-table` 指向新表。没有同目录完整 VLM 测量时，比例预算需要显式传入 `--reference-vlm-ms`；该值会作为调用者提供的基准留档。

## 命令行

从仓库根目录执行。准备预算和离线选择只需要 NumPy，不加载模型或使用 GPU。

```bash
policies/realtime-vla-flash/.venv/bin/python -m preparations.idea1_v3_4_E2_budgeted_skip prepare --save-fraction 1/3
policies/realtime-vla-flash/.venv/bin/python -m preparations.idea1_v3_4_E2_budgeted_skip prepare --save-ms 6 --tolerance-ms 0.25
```

输出含 `artifact_path`。预算、容差、成本表和基准测量共同决定缓存目录：`results/idea1_v3_4_E2_budgeted_skip/budgets/vlm-skip-<hash>/`。其中 `budget.json` 和 `candidates.npz` 可供后续复用，不覆盖历史 B4/B6 实验。

离线复放一个已经保存的第三次 Full 评分：

```bash
policies/realtime-vla-flash/.venv/bin/python -m preparations.idea1_v3_4_E2_budgeted_skip select \
  --budget results/idea1_v3_4_E2_budgeted_skip/budgets/vlm-skip-c8211bfd8fb6d76caa2a615b/budget.json \
  --scores results/idea1_v3_4_E2_mixed_cost/B6_M-E-U+/episodes/libero_10/task09/init20/arrays.npz \
  --score-key module_score_10 --weighted
```

评分数组形状为 `(17, 2)`，两列依次是 Attention、FFN；也接受单数组 NPY 或 JSON。默认最小化模块分数之和，`--weighted` 最小化 `sum(score / standalone_saved_ms)`；两种目标使用相同的硬件预算窗口与合法候选集。

## 运行中修改

已有原生 `Backend` 的调用方可以直接使用包装策略：

```python
from preparations.idea1_v3_4_E2_budgeted_skip.budget import prepare_budget
from preparations.idea1_v3_4_E2_budgeted_skip.runtime import BudgetedPolicy

costs = 'results/idea1_v3_4_E2_mixed_cost/hardware_probe/cost_table.json'
cache = 'results/idea1_v3_4_E2_budgeted_skip/budgets'
initial = prepare_budget(costs, cache, saved_fraction=1/3)
smaller = prepare_budget(costs, cache, saved_fraction=.2)
policy = BudgetedPolicy(backend, initial, cost_weighted=True)

# predict / acknowledge 的参数和原 mixed policy 相同。
policy.set_budget(budget_artifact=smaller)
# 也可直接指定：policy.set_budget(saved_ms=6)
```

`set_budget` 将新预算排队，在下一个 Full 开始时生效。当前周期余下 Fast 使用原来的 mask；不会清空两代真实 Full 残差历史，也不会重新创建评分 selector。未确认执行的 proposal 会阻止下一次 predict。连续提交多个预算时，以最新提交的预算为准。

首次准备候选会进行 CPU 枚举，宜提前 `prepare_budget` 再传 artifact，避免这部分准备耗时进入预测调度。已有 calibration 的 SHA 会被校验；修改标定或切换硬件成本表，需要重新准备预算并建立新的 policy 实例。

`configure()` / `reset()` 表示开始新 episode，会清空历史并采用已排队预算。预算控制保留 Full-D10 / Fast-D2 × 4、H50/R5、间隔层约束、FFN/Attention 各自的外推残差补偿，以及 Full-only 评分。第三次 Full 起使用外推误差评分，前两次沿用原评分。0 预算仍保留此调度和 Full 评分，只关闭模块跳过。

## 原生 RPC 服务

```bash
policies/realtime-vla-flash/.venv/bin/python -m preparations.idea1_v3_4_E2_budgeted_skip serve \
  --budget results/idea1_v3_4_E2_budgeted_skip/budgets/vlm-skip-c8211bfd8fb6d76caa2a615b/budget.json \
  --gpu 0 --output results/idea1_v3_4_E2_budgeted_skip/service --weighted
```

选择空闲 GPU；服务会拒绝已有计算进程的卡。默认绑定 localhost 的动态端口，ready JSON 和 `protocol.json` 中记录实际端口。使用现有 `preparations.idea1_v2_1.runtime_client.rpc` 的 pickle 协议，支持 `health`、`configure(config=...)`、`predict`、`ack`、`reset_all`，以及新增的：

```python
rpc(port, dict(op='set_budget', client='default', saved_fraction=.2))
# 或传入预先准备好的 budget_artifact 路径。
```

默认单客户端串行 B1；`--actors 2` 允许两个同任务客户端，各自保留历史和预算。`route` 中记录实际预算 hash、目标毫秒/比例、所选模块、估计节省时间，以及是否在这次 Full 应用新预算。预算排队事件写入 `budget_events.jsonl`。

## 验证范围

```bash
policies/realtime-vla-flash/.venv/bin/python -m unittest discover -s preparations/idea1_v3_4_E2_budgeted_skip -t . -p 'test_*.py' -v
```

CPU 测试覆盖任意预算的选择最优性、零/不可行预算、测量分母和缓存校验，以及运行中切换、pending ack、历史保留和 CLI prepare/select。

`results/idea1_v3_4_E2_budgeted_skip/selection_examples.json` 保存真实第三次 Full 分数下 20%、1/3、6 ms 预算及 U/W 的离线选择。这是单输入选择复放，不是新成功率评测。当前 GPU 被其他任务占用，新包装服务及预算切换尚未做 GPU 集成验证；复用的原生混合 skip 内核来自已经完成校验的历史实验。
