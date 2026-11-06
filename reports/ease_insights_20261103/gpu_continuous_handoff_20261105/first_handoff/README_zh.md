# v3 首次跨任务持有核验

截至 2026-11-05 05:38:21.882 UTC，slot00 和 slot01 都已跨过各自首个 complete task 边界，并由同一 keeper 保持 CUDA context，随后启动了绑定到原 UUID 的新 server。两份原 task outcome 均为 `complete`（各完成 120 个分配案例）；原 server/worker PID 已退出。session PID/start、keeper PID/start/PPID/nonce 及 64 MiB Driver allocation/context receipt 均未变。

中间各有一次 `Address already in use` 技术 attention。经已验证的 resume receipt 后，会话使用新的 `__port_retry1` task ID 继续；旧 outcome 保留。新 server protocol 分别确认 slot00 的 PID 638221/UUID0/`local_goal`，slot01 的 PID 638228/UUID1/`local_spatial`，manifest hash 相同且 real-input parity gate 为 true。重试 task 在本快照时仍运行，未声称其完成或科学成功。

watcher 最新 event 62、心跳 05:38:21.882 UTC 且无错误。自初次 holder 事件 47/48 起，UUID0/UUID1 的 durable event 记录均未出现 idle 转换。watcher 以 2 秒轮询，结论不排除采样间隔内未观测到的瞬态；本核验也不证明整卡独占。keeper 单次申请与 session task loop 的源码摘要、PID/start 与收据见 [evidence.json](evidence.json)。
