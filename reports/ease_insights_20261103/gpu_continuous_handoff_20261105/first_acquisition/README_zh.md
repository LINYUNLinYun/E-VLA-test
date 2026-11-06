# v3 首次 keeper CUDA 持有核验

结论：证据确认 v3 的 slot00、slot01 已分别在分配 UUID 上创建真实 CUDA Driver context，并保持各 64 MiB keeper allocation。该结论针对本次 v3 keeper 持有，不表示整个项目第一次使用 CUDA，也不表示独占整卡。

依据是三组相互吻合的记录：05:00:31.860 UTC watcher events 43–46 记录四卡均为 1 MiB、0 compute；两份 launch gate 随后记录目标卡连续自然空闲 15.494s / 15.514s、2s 轮询并通过最终 idle check。`/proc` 现场核验显示 scheduler PID 600088/start 2622761021 创建两个 session（631907/start 2623408608、631915/start 2623408624）；各 session 的直接子进程 keeper（632159/start 2623410211、632160/start 2623410216）仍存活，父 PID、start ticks、UUID argv 与 `binding.json` 对上。keeper 自身 argv 和 ready receipt 匹配各自 UUID；receipt 报告 `context_created=true`、Driver 13000、64 MiB allocation。05:00:49.898 UTC watcher events 47–48 随即记录两张目标卡各 460 MiB、1 compute。watcher 在 05:04:29.926 UTC 仍新鲜且无错误；两张卡报告 busy，后续 session server 也已由 `/proc` 确认为对应 session 子进程。

watcher 的 compute PID 是主机查询结果，本报告没有把它映射成容器 `/proc` PID。当前 session 仍在各自首个任务中；跨任务边界的 keeper 连续存活尚未发生，科学案例成功也未由本审计判断。详细机器可读记录见 [evidence.json](evidence.json)。
