# 用户指定的 25 秒启动阈值

用户要求将连续空闲启动条件从 60 秒改为 25 秒，检查间隔保持 2 秒。本次修改已实际完成。

- 五个未完成的 local-search 条件，队列中的资源环境变量和元数据均为 25/2。
- `await_assigned_gpu.py` 默认值也改为 25/2，后续自动生成的复验任务不会因缺少环境变量而恢复到 60/10。
- 原有“无计算进程且显存低于 500 MiB”判定、占用/查询失败后清零、启动前再次查询同一 UUID 的检查保持。
- 两个旧 CPU 等待进程分别经队列提交、原 scheduler 和 supervisor 确认后退役。未终止模型，未改动作参数、科学源、科学 manifest 或已有有效案例。

两个事务的完整记录：

- `results/ease_insights_20261103/gate_policy_refreshes_25s/user25s_full_retry1_defaults25/`
- `results/ease_insights_20261103/gate_policy_refreshes_25s/user25s_spatial_retry1_defaults25/`

首事务只改变 Full 的资源任务身份/收据路径、资源阈值，以及三个 pending 条件的阈值；第二事务只改变 Spatial 的对应资源字段。最终队列 SHA256：

`c5b2cbef42cceca177bc64510b58d9cc0646ed1a60257d172012cf491d7e5fd5`

37 项 CPU 测试由实现者、独立审阅者和主代理分别验证通过。资源 gate 最终源码 SHA256：

`07aa0356850fa6d5199d5f3ba0b35632c4b5f500b163f5da4bb106da506a6e3f`

冻结 pipeline 中历史 `launch_gate` 元数据的默认标签仍为 60/10；它不控制实际等待时长。当前五条队列已显式规范化为 25/2，未来 gate 的实际 fallback 为 25/2；后续入队时仍应规范化资源标签。

02:15 UTC 的自然释放窗口提供了实际验证：新 Full 等待进程的收据为 `idle_threshold_seconds=25`、`poll_seconds=2`。GPU 1、2 约 12 秒后重新被占用，计时清零，模型未启动。该事件没有增加有效案例数。

新增的“取得 GPU 后跨任务持续保留占用”要求另由 persistent resource session 实现；本次 25 秒修改本身不声称已经解决任务之间的 GPU 交接。
