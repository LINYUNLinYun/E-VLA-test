# E4 零样本跨环境结果入口

108/108 条预先固定的有效轨迹已完成；138 次总尝试包含 30 次保留的 PushT 第0步导入失败。技术失败未计入策略成功率。仅补齐父进程的本地源码搜索路径后，在同一不可变 manifest 下续跑，前78条有效轨迹全部跳过，旧结果未覆盖。

| 条件 | CALVIN debug | Simpler pick coke can | PushT二维投影压力测试 |
|---|---:|---:|---:|
| Full D10 | 0/8 | 0/5 | 0/5 |
| V14 D10 | 2/8 | 0/5 | 0/5 |
| draft_goal | 0/8 | 0/5 | 0/5 |
| draft_object | 1/8 | 0/5 | 0/5 |
| draft_spatial | 0/8 | 0/5 | 0/5 |
| draft_10 | 0/8 | 0/5 | 0/5 |

Full在这三套冻结部署中也未成功，因此不能据此把失败归因于draft容量或泛化能力。draft_object确有一个原生成功样例，结果也不支持预设“draft只能用于LIBERO”。PushT是显式异构二维投影，不与机械臂结果汇成总体能力分数。

零样本指没有目标环境权重训练、微调或按结果挑选head；运行前已明确设计并冻结了机器人坐标、动作单位、夹爪、缺失相机和PushT投影桥接。不是“没有接口适配”。

- [主结果、原生分母及固定案例实际XYZ/XY轨迹](report_20261022T061020Z/report_zh.md)
- [18个预先固定case0视频：相对链接与播放器](report_20261022T061020Z/media_index.html)
- [独立逐案例审计与明确标注的事后case6成功对照图](outcomes_audit_20261022T061020Z/report_zh.md)
- [108条逐案例终止、实际状态范围、动作范围和分通道裁剪](outcomes_audit_20261022T061020Z/episode_outcomes.csv)
- [全部138次尝试历史](outcomes_audit_20261022T061020Z/attempt_history.csv)
- [审计收据与完整来源hash](outcomes_audit_20261022T061020Z/audit.json)

审计通过：异常0；18个案例的六条件初始RGB/状态哈希一致；5,580次质量请求与两个服务合并路由计数一致；18个视频完整解码通过，播放器40个相对资源链接均可解析；冻结源码及快照哈希一致。两次服务的原输入门控单独记入preflight，不混入质量调用。

所有条件仅运行在GPU3。两个自有模型服务均已退出，最终该卡compute PID为空、显存1 MiB；未向其他进程发送信号。见[清理收据](final_cleanup.json)。E4从05:00 UTC准备到约06:10 UTC完成，位于自定90分钟探索边界内；该边界不是用户为E4另行指定的预算。

manifest_hash（规范化payload摘要）：`3eeeb50e43b1078a61d9015bbb0fe371ff5f4bd98809e0242699de700369222f`。
