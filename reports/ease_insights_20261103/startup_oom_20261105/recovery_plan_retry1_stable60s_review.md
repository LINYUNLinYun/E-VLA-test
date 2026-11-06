新恢复包已准备，尚未应用。目录为 `results/ease_insights_20261103/process_recovery/local_search_startup_20261105_retry1_stable60s_review/`；旧 5s/2s 包保留未改。仅运行严格加载和只读 `check`，没有修改真实 queue/jobs_state/controller state，没有启动评测或控制器、发送信号、查询 GPU。

根代理最终选择：五条未完成 local-search discovery job 统一使用连续 idle **60 秒**、poll **2 秒**。保持已经部署的全卡 2 秒通知和 gate exec 前 fresh scan；不实现 CUDA claim，不修改任何运行源码或冻结科学条件。原因是本轮 5 秒 gate 已在约 32 秒 idle 窗口中成功 exec，但加载期间出现持续约 18 GB 外部占用，两服务在首个显式 CUDA 申请处 OOM，尚无有效 case；延长持续空闲条件用于避开该短窗口。

复核结果：

- frozen `selection.load_stage(discovery, 'discovery')` 再次通过源码、manifest、receipt/artifact、case identity、初始观测与噪声严格核验。有效 **120/720**，invalid=0，unexpected=0；科学阶段未完成。
- Jac 为 120：99 成功、21 有效策略失败全部保留。Full 与 spatial 各 0 有效、120 missing；object/goal/long 各 0 有效、120 missing，没有重测任何有效 case。
- 实际 `/proc` 确认 335208/start_ticks 2615895606 为唯一匹配 queue/output 的 live scheduler；338645、415450、432881、432890 均 absent/dead；两个失败 run 的 exact output 路径无 live server。未使用 GPU host PID 发送信号。
- 当前 queue SHA256 仍为 `7e3f98aa5aad9420e17baf08802d08a1cb013cb1642282b20e89f5203d1ed1b6`。两失败 job 仍 failed，另三条资源更新 job 仍 pending；准备过程不写历史 jobs_state。
- 22 份文件逐字节复制至新包 archive，涵盖 queue、scheduler status/jobs_state/process、Sup state/status/process/log、coor state/status/process/log/events、两条失败 scheduler log/gate receipt、两份 startup identity/server log 和 manifest。archive_manifest 记录来源、时间、大小、SHA256，原件保留。

唯一合并队列候选为 `queue.proposed.json`，SHA256：

`83cb279d6637f64f0d49173d91ddd6c13de50d8a259c1d23e27f388b067289e4`

完整递归差异 `queue.complete_diff.json` 恰有 **14 处标量变化**：两个 failed ID 各加 `__startup_retry1`，两个 wrapper 的 `--status-file` 路径随新 ID 更新，五个 `env.EASE_GPU_GATE_IDLE_SECONDS` 从 `'5'` 变为 `'60'`，五个 `launch_gate.continuous_idle_seconds` 从 5 变为 60。五个 poll env/metadata 本来已经是 2，保持 2。没有添加或修改 resource_revision；原键原值保留。完整五个 before/proposed job 对象和 inner argv 在 `queue.field_diff.json`。

Jac 已完成 job 整对象不变；全部科学 inner argv、科学 env、launch_gate.original_argv、complete_file、manifest_hash、scientific_phase、expected_episodes、ready_files、depends_on 及其语义不变；其他 job 和顶层 queue 字段、顺序不变。frozen Sup 的纯 `completion_gate(proposed, old jobs_state)` 得到 completed=1/expected=6、complete=false，无 technical failure 异常；未调用 tick/ensure。

运行源码和候选状态的审阅身份：

| 对象 | SHA256 |
| --- | --- |
| deployed await_assigned_gpu.py | `b4b5ca1bf95c8f43103e9a40b4de8b8cd8036bc6c870b12c2bf58dac6653e4cf` |
| scheduler.py | `0786dd02cb35dc7da40ee06dd515dc4481aa0d28e09d4d78baa5df1338345752` |
| frozen local supervisor.py | `b232b49ebffe280a39e3e6438adbd600281ac6bf25b2613e14c2e07209dcaaeb` |
| gate_opportunity_coordinator.py | `70ce6ef8a86153ad1f3cc6ffe0c4d05cc7e2d2dd334423ceb3af9598a7207432` |
| frozen runner.py | `8df52ecd433ce6fc8d73ad55bdf08e1c51c134cc448a0cf928f417137dd4810b` |
| Sup state candidate | `5a7def22b94223c8b1a12928973d253e8edc9d66a65f9ff9ae6b26085d213efc` |
| coor state candidate | `4a9f3f5c5b81bf25ca2d221567e3e2bc95c30191969c48fce2d0a061bcb85046` |
| reviewed_recovery.py | `559a46b3dc220a2c6582c548129f8fcfeeabbf388fecfa8e8c439691a4405a15` |

科学 manifest identity 仍为 `032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9`。`source_hashes.json` pin 55 个部署/冻结源码文件，`frozen_source_evidence.json` 保存 manifest 的 46 source_files、9 local_runtime_sources 和 6 analysis_sources（去重后计数不同）；每个执行阶段都复核 hashes。所有源码均未由本包修改。

Sup/coor 最小状态恢复：

- Sup candidate 仅 `phase: needs_attention → waiting_discovery` 并移除归档中的 error/attention_unix；actions={}、旧 scheduler identity、gate 和其他字段保留。直接重启 terminal state 会再次退出，所以需要这一明确状态恢复。status/process 由新实例首 tick 写入，不手工伪造。
- coor 旧 phase=supervisor_terminal、pending 明确 null、sequence=1；controllers 固定旧 Sup 338645。新 Sup 健康后才能解绑。coor candidate 仅移除 controllers，其余 schema/pending/cooldown/sequence/phase 保留；该 phase 在 step 中非 sticky，首个正常 step 会绑定新 Sup 与原 scheduler。
- 每阶段同时校验两个 state candidate 的 receipt-pinned SHA，并从 archive 重构完整期望对象进行相等比较，拒绝任何额外字段变化。coor pending 若非 null/缺失、state CAS 不符、旧进程 live/unknown，均保留现场拒绝，不清未知 pending。
- 解绑前 `Backend.controllers(None)` 仅验证 OS/file 身份，要求新 Sup、同一个 335208 且原 `_scheduler_proof` 完全相同；不调用 inventory/gpus/lease/helper.operate。唯一 allocator 始终不重启、不替换。

根代理审阅后可按下列顺序执行。mutation 子命令只接受新 flag **`root-reviewed-stable60s-poll2`**；旧 5s/2s flag 已实际验证被 argparse 以 exit=2 拒绝。`check` 和 `check-adoption` 为只读，不要求 flag。

```bash
startup_recovery_script=/root/nfs/embodied/benchmark/results/ease_insights_20261103/process_recovery/local_search_startup_20261105_retry1_stable60s_review/reviewed_recovery.py
startup_recovery_python=/root/nfs/embodied/benchmark/policies/realtime-vla-flash/.venv/bin/python

"$startup_recovery_python" -B "$startup_recovery_script" check
"$startup_recovery_python" -B "$startup_recovery_script" commit --resource-policy root-reviewed-stable60s-poll2
"$startup_recovery_python" -B "$startup_recovery_script" check-adoption
"$startup_recovery_python" -B "$startup_recovery_script" launch-supervisor --resource-policy root-reviewed-stable60s-poll2
"$startup_recovery_python" -B "$startup_recovery_script" rebind-coordinator --resource-policy root-reviewed-stable60s-poll2
"$startup_recovery_python" -B "$startup_recovery_script" launch-coordinator --resource-policy root-reviewed-stable60s-poll2
```

`commit` 获取 queue lock、Sup/coor singleton lock 后再次核真实 queue/state CAS、两失败状态、另三 job 仍 pending，以及 `/proc` 无在途 local discovery gate/runtime；先恢复 Sup state，再一次原子替换整个 merged queue，不写 jobs_state。Sup state 与 queue 不是跨文件原子事务，异常会留下 partial_commit 记录供审查，不自动重复动作。

原 scheduler 每 3 秒重读 queue，未参与 queue read lock。`check-adoption` 因此在发布后有界 12 秒只读等待 scheduler 的 status/jobs_state 确认新 ID、移除当前 status 中旧 ID，然后验证五条资源更新：pending job 只能确认候选已被接纳、未来资源 env/metadata 为 60/2；已分配进程则对照 `/proc` PID/start_ticks/argv。仍为 gate 时验证真实 `/proc/environ` 的 60/2；已 exec runner 时 gate 已主动移除 resource env，改核同 PID 的 exact inner argv 和该 job 的新 gate receipt（idle_threshold_seconds=60、poll_seconds=2、final_idle_check=idle）。不查询 GPU；如果观察到旧资源策略在途或身份不明，拒绝验收并保留进程，不发送信号或静默声称五条都已切换。

新 Sup 与 coor 均用 Popen(start_new_session=True, stdin=DEVNULL, stdout/stderr=各自 daemon.log, close_fds=True) 持久启动。Sup 命令沿用 model venv、`-B -u -m ...supervisor --root <discovery> --confirmation-root <confirmation> --scheduler-pid 335208 --poll-seconds 10 --launch`；coor 为 `/usr/bin/python3 -B -u -m preparations.ease_insights_20261103.gate_opportunity_coordinator --launch`。frozen helper parser 支持 -B/-u。

Sup 启动后有界 20 秒确认 fresh nonterminal waiting_discovery、new PID、同一 scheduler PID/start_ticks。coor 启动前要求 live state 与完整 reviewed unbound candidate 相等；启动后有界 45 秒读 OS/process/state/status，确认 fresh 非 attention/terminal phase、绑定新 Sup 与原 scheduler 后才记录 coordinator_healthy。超界或不健康仅保留进程供检查，不自动第二次启动。每个实际 mutation/launch/healthy 阶段会在本包 execution_receipts 下写记录；准备完成时该目录不存在。

最终验收读取实际 queue/jobs_state、新 Sup/coor process/state/status、五 job 的实际 gate receipt：335208 始终唯一 allocator，新 Sup fresh 健康，coor pin 新 Sup；新 retry ID 被接纳，历史 failed records 保留；Jac120 和其中 21 个策略失败不重测，科学字段无变化。连续 idle60 并不能保证外部后续不会抢占；若出现新的 startup OOM，作为新技术事件检查，不自动追加 retry ID 循环。
