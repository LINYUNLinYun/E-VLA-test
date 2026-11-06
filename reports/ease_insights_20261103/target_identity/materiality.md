# Spatial task1/init20 单例差异量化补充

主分析保留这条回合，不重跑、不剔除、不放宽 `atol=rtol=1e-5`。本补充使用已保存的 13:28:58 UTC 首次请求快照；唯一差异仍为 Spatial 来源 FLASH 与 Full 在 Spatial task1/init20 的首次 Full 动作。逐项数值、配置证据、时序和敏感性输入哈希见 [materiality.json](materiality.json)，复验脚本为 [materiality.py](materiality.py)。

前 12 步各动作维度的最大绝对差：

| 维度 | raw 前缀 | decoded 前缀 |
| --- | ---: | ---: |
| X | 0.00390625 | 0.001283313329 |
| Y | 0.00781250 | 0.003278428119 |
| Z | 0.00390625 | 0.002408238542 |
| rotation 0 | 0.001953125 | 0.000684756061 |
| rotation 1 | 0.001953125 | 0.001777616187 |
| rotation 2 | 0.00146484375 | 0.000503417172 |
| gripper | 0.00390625 | 0.003903871350 |

按组取最大值，decoded 的 XYZ/rotation/gripper 分别为 **0.00327843 / 0.00177762 / 0.00390387**。这些是 policy 命令分量差值，不能直接称为真实末端位移或角度偏差。双方 raw 与 decoded 的 **12 步夹爪符号均为负，0 步发生符号变化**；并不据此断言差异不可能影响后续轨迹或成功。两次已记录回合最终均成功，也不能识别该数值差异的因果影响。

同 case 的环境身份 JSON 完全一致，包括官方初态/初始 sim state hash、BDDL、环境 seed7、20Hz、控制器及标度；对应 episode 均为 task1/init20、max_steps220、settling10。科学协议 H50/R12/D10、固定 base checkpoint、动作/VLM18层与10步采样、首次路由 full1/draft0/verify0均一致。input/noise、prompt、state8一致的检查已在主审计记录。

时序上下文存在差别：

| 项目 | Full baseline | Spatial来源 FLASH |
| --- | ---: | ---: |
| run 启动时间（UTC） | 09:04:35.754236 | 09:32:32.343706 |
| 请求日志行号 | 9 | 1 |
| 本case replan_id | 0 | 0 |
| 进程首个计划case | Spatial task0/init20 | Spatial task1/init20 |
| run 准备时间 | 22.045751s | 22.053764s |
| 该请求 policy_wall_ms | 4472.250540 | 7715.268094 |
| 该请求 sample_actions_ms | 4470.611621 | 5858.207736 |

Spatial FLASH 的这条是新服务的**首次冷启动请求**；task0已在 smoke 完成，因此 discovery服务从task1开始。Full 对照是该case的首次请求，但不是其进程首次请求。两run还使用了不同GPU UUID，详见JSON。没有显式逐请求capture/compile标记来定位差异来源；这里的冷启动描述仅指进程首请求，**原因未知**，不将数值差异归因于冷编译、GPU差异或任何已确认机制。

轻量接受率敏感性固定 Spatial 目标，用现有中期固定汇总中已完整的 Spatial/Object/Goal 三来源。主结果保留全部10个case；敏感性从三个来源**同时移除同一个task1/init20**，留下同样9个case，避免不等任务集合比较。接受率仍为累计接受动作数 /（12×verify请求数）。

| 来源 | 主结果（10case） | 同case删除敏感性（9case） |
| --- | ---: | ---: |
| Spatial | 511/816 = 62.6225% | 460/696 = 66.0920% |
| Object | 0/960 = 0% | 0/852 = 0% |
| Goal | 201/1308 = 15.3670% | 192/1188 = 16.1616% |

主结果成功数依次为10/10、10/10、9/10；同case删除后为9/9、9/9、8/9。这仅展示该case对汇总的影响；不替代主分析，不构成数值差异对接受率或成功率的因果估计。Long来源在所复用的旧中期汇总中不完整，因此不纳入这个三来源敏感性。

```bash
python3 -B /root/nfs/embodied/benchmark/reports/ease_insights_20261103/target_identity/materiality.py
```
