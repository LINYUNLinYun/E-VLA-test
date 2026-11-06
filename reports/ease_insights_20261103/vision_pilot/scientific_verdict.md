# Vision 子路径独立 pilot：预登记的双向偏好没有复现

冻结的 A 为 π0 vision Keep24，跳层 `[8,9,10]`；B 为同预算 Keep24，跳层 `[5,14,23]`。发现集是历史 init0；本次使用未参与这组候选筛选的官方 init20，四 suite 各10个任务，三条件共120条有效回合。预登记方向保持为 **Goal A>B、Long B>A**，没有根据新结果更换类别或方向。

| Suite | Full | A | B | 新 init20 的 A−B |
|---|---:|---:|---:|---:|
| Spatial | 10/10 | 10/10 | 10/10 | 0 |
| Object | 10/10 | 10/10 | 10/10 | 0 |
| Goal | 10/10 | 9/10 | 10/10 | −10pp |
| Long | 9/10 | 7/10 | 8/10 | −10pp |
| 全部 | 39/40 | 36/40 | 38/40 | −5pp |

历史 Goal 的 A10/B8 变为 A9/B10，预登记方向没有复现且点估计反向。历史 Long 的 A7/B9 变为 A7/B8，方向相同，但仅一例差异。Goal 的 A−B 配对 bootstrap 95%区间为 [−30,0]pp，单侧精确 McNemar p=1；Long 的 B−A 区间为 [0,30]pp，p=0.5。每任务仅一个新初态，因此该 pilot 只提供初步独立证据，既不能支持可靠互补性，也不能证明其他初态不会出现互补。

全部 A/B 不一致案例仅两个，均为 B 成功、A 失败，Full 均成功：

| Suite | Task | Init | A | B | Full |
|---|---:|---:|---|---|---|
| Goal | 3 | 20 | 失败 | 成功 | 成功 |
| Long | 6 | 20 | 失败 | 成功 | 成功 |

120条回合的 receipt artifact SHA 均通过验证。独立审计检查了4152次请求（Full1286、A1488、B1378）：实际 NFE10、H50/R5、正确 vision mask、完整原生 AE、请求种子公式均通过；三条件40组首输入、噪声、官方初态、BDDL及控制器身份一致，Full与A/B使用相同checkpoint/backend与数值协议。

这组路径目前应称为“固定裁剪子路径”。本次各自闭环输入上的平均模型请求耗时为 Full108.15ms、A108.55ms、B110.22ms，不能作为相同输入加速比；已有历史材料同样没有证明 A/B 的端到端推理加速。本轮结果既不支持自适应选择的收益，也不支持“已找到两条真正更快的互补路径”。

完整统计与逐case数据见 [analysis.json](analysis.json)、[analysis.csv](analysis.csv)。独立检查及来源 receipt 哈希见 [independent_audit.json](independent_audit.json)，可复现脚本为 [independent_audit.py](independent_audit.py)。自动分析页面中的“执行完整确认”是原报告工具的固定流程文案，不是本审计新增的实验授权或科学结论。
