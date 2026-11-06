# π0.5 流水线：200案例闭环成功率扩样

用户要求：将先前8例/配置的小样本扩到一两百例，验证流水线是否影响闭环成功率。

## 固定设计

每配置200个case：`libero_object/spatial/goal/10`四套件，各task0–9、官方init0–4。环境seed7、settling10步、20Hz控制，原suite上限280/220/300/520步不变。完整π0.5 LIBERO、BF16主体、B1/H10/D10/R5、checkpoint与原model_seed规则均不变。

比较四臂，共800回合：

| 配置 | 语义 |
|---|---|
| fresh18 | 当前完整新KV、全卡，参考 |
| sm16_tail0 | lead3，AE16/Prefix112，逐轮混合新旧KV |
| sm32_tail0 | lead3，AE32/Prefix96，无末轮强制刷新 |
| sm32_tail2 | lead3，AE32/Prefix96，最后2轮强制当前18层KV-ready |

跨帧KV是近似推理，即使没有数据竞争也可能影响成功率。本轮求证差异，不预设无损结论；未扩样的tail1/其他lead不能借本轮结论宣称无损。

## 复用与执行

- 先前正确EGL1/GPU2的32回合（每臂8例，task0/5 init0）通过独立provenance检查，计入固定网格；新增768回合。复用只用于成功率/覆盖度，不混入新吞吐成本。
- `case_id=suite_index*50+task*5+init`；所有配置用完全相同的suite/task/init与每次replan噪声种子。任务完成顺序不影响seed。
- 一个task的四配置×五init在同一GPU集中执行；配置顺序按task轮换、第二GPU反序，减少配置与硬件/时间段绑定。
- 模型服务每次最多4个task块，之后原graceful shutdown释放Green Context，再加载下一块。全程保持两份活KV、原Pool/Lease/epoch/generation/Event协议，旧runtime不修改。
- 每配置/task使用一个client进程批量执行原episode逻辑，节省Python导入；环境复用关闭，每episode仍构建/关闭环境、完整reset/settle并fresh bootstrap，不引入跨episode状态。
- GPU2/EGL1/19102/CPU0–15先启动；GPU3/EGL0/19103/CPU96–111在其他会话明确交还后加入。GPU0/1排除，不干涉其占用者。单dispatcher分配唯一task块，避免重复执行。
- 实际任务失败是有效结果，绝不重试到成功。技术失败最多重试一次、新目录保留原记录；归属/source/非零进程退出错误停止并审查。

## 输出和统计

原生动作、输入/噪声seed与SHA、环境identity、成功/终止原因、步数、EEF轨迹、逐轮KV frontier、首帧bootstrap与slot/epoch、RPC与episode wall保留；不保存全量视频。

统计预先固定在[prospective_statistics.md](../../../results/idea2/closedloop200_20261025/prospective_statistics.md)：

1. 每臂总体与分suite成功率、Wilson95%区间。
2. 同case配对：candidate救回/损失案例数、成功率差、精确McNemar；主三比较对fresh作Holm校正。
3. 40task聚类bootstrap95%区间（10000次、seed20261025），另给episode重采样敏感性。不是把同task的5个init当完全独立任务。
4. 全200/臂主分析与排除已知pilot后的新增192/臂敏感性，明确候选曾由pilot筛选、并非全盲独立确认。
5. 不把`p>0.05`称为无损；用户未指定非劣门槛，报告区间能/不能排除的退化幅度。零discordance的退化bootstrap不能当绝对无影响证明。
6. 技术失败或缺失病例使完整性未通过；实际任务失败保留在分母。直到800cell完整、provenance/退出/渲染/状态隔离检查通过，才生成最终比较。

同时报告完成步数、准备/捕获成本、episode工作墙钟、有效回合/小时、动作/秒；自由轨迹RPC不是同输入模型速度。LIBERO在推理时暂停，真实下一观测尚未产生，本轮`next_encoder=False`，验证混合KV算法质量，不声明真实摄像头SLO。

## 交付

新代码/原始结果：`results/idea2/closedloop200_20261025/`；汇总：`reports/idea2_closedloop200_20261025/`；最终报告同目录`idea2_closedloop200_report_20261025.md`。旧结果不覆写，唯一task网格、复现命令、复用证明、失败清单和统计输入一并保存。
