本材料仅准备恢复，未应用队列、修改现有控制器状态、发送信号、启动进程或查询 GPU。准备目录是 `results/ease_insights_20261103/process_recovery/local_search_startup_20261105_retry1_review/`。本轮采用项目 `embodied-eval-acceleration` 约束：保留有效策略失败，只重试技术中断；唯一 scheduler 继续分配资源。

结论与证据：

- frozen `selection.load_stage(discovery, 'discovery')` 已验证 manifest、冻结科学/分析源码、逐 case receipt/artifact/identity、初始观测与噪声身份。当前有效 **120/720**，invalid=0，unexpected=0，科学阶段未完成。
- `jac_anchor` 有效 120，其中成功 99、有效策略失败 21，全部保留；`full_d10` 与 `local_spatial` 各有效 0、missing 120，不存在需要续跑或重测的有效 case。object/goal/long 各 missing 120，原 pending job 原样保留。
- `/proc` 显示 PID 335208/start_ticks 2615895606 是唯一匹配当前 queue/output 的 live scheduler；PID 338645、415450、432881、432890 均 absent/dead；两次失败 run 的确切 output 路径均无 live server 匹配。未用 GPU 报告的 host PID 推断本命名空间 PID，也未发送任何信号。
- scheduler 当前两条 job 为 failed，returncode=1，active_pids=[]；Full 是 GPU0/UUID `GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1`，spatial 是 GPU3/UUID `GPU-555cde36-5e41-bc44-0c69-ea47358fd703`，这些映射来自已有 job/run identity。
- 两次 startup run 分别是 `full_d10/runs/discovery_shard00of01_1793818662106423845` 与 `local_spatial/runs/discovery_shard00of01_1793818662086250240`。每个目录只有 identity.json/server.log；堆栈在 `pi0_infer.py:1633` 的显式 CUDA 张量申请处报 OOM，尚未产生 protocol/gate/cross_service/worker/result 成果。启动资源根因仍由独立诊断决定。

归档与差异：

- `archive/` 逐字节复制 queue、scheduler status/jobs_state/process、sup state/status/process/daemon log、coor state/status/process/events/daemon log、两条 scheduler log/gate receipt、两次 startup run 的 identity/server log、manifest；`archive_manifest.json` 记录原路径、复制路径、时间、大小和 SHA256。原件未删除。
- `strict_stage_audit.json` 保存完整严格加载结果、所有 missing case 和 120 条有效 episode receipt 证据；`process_observation.json` 保存本命名空间 OS 观察。
- `queue.before.json` SHA256：`7e3f98aa5aad9420e17baf08802d08a1cb013cb1642282b20e89f5203d1ed1b6`。
- `queue.proposed.json` SHA256：`65e31eba1291d52020c247803a94e9f4db02119cc568983669caaecde5e41956`。
- 候选只在现有位置替换两个 failed job：原 ID 后缀各加 `__startup_retry1`；各 wrapper argv 的 `--status-file` 改为新 ID 的 launch_gates 路径。完整原值/新值在 `queue.field_diff.json`。总计 4 处标量差异，其余 job、位置和顶层字段不变。
- 原 inner argv、`launch_gate.original_argv`、complete_file、expected_episodes=120、scientific_phase、manifest_hash、ready_files、depends_on 与 queue env 全部不变；历史 jobs_state 完整保留，不由恢复脚本写入。新 ID 由 live scheduler 的 `state.setdefault(id, pending)` 接纳。
- 候选目前保持 `EASE_GPU_GATE_IDLE_SECONDS=5` / `EASE_GPU_GATE_POLL_SECONDS=2`，且两条新 gate receipt 尚不存在。根代理先决定启动资源方案；若改变资源策略，应另出可审核候选和新 hash，不能直接套用本包的 `commit`。
- 用候选 queue 加原历史 jobs_state，仅调用 frozen supervisor 的纯 `completion_gate()`，已得到 waiting、completed=1、expected=6，没有 technical-failure 异常。未调用 tick/ensure，也未启动 scheduler。

supervisor 状态处理：

旧 state 的 `phase=needs_attention` 属于 terminal，直接重启只会再次退出。`supervisor.state.proposed.json` 仅把 phase 改为 `waiting_discovery`，移除已归档的 `error` 和 `attention_unix`，其余字段全部保留，详见 `supervisor.state.field_diff.json`。旧 actions={}，没有未完成 CPU action；不需要删除 actions、gate 或 scheduler identity。status/process 不手工伪造，新实例首 tick 自己更新。

`SharedSchedulerWatch.observe` 会在原 scheduler identity、initial PID、共享 controller receipt、continuation/restart receipt 中找匹配 live scheduler，并检查 max-gpus=2 / idle-seconds=0。当前唯一 allocator 335208 会被直接 adopt；不应重启或替换 live scheduler。恢复脚本在每个动作前复核原 PID/start_ticks 和 `/proc` 唯一性。

coordinator 状态处理：

旧 coor 为 `phase=supervisor_terminal`，`pending=null`，sequence=1；controllers 中仍固定旧 Sup 338645。`Backend.controllers(previous)` 强制 `same_process(new_sup, previous.sup)`，因此只重启 coor 会在新 Sup 上报 `Original supervisor identity changed`。

正确顺序是：归档旧 coor；证明旧 coor 和旧 Sup OS 已死；启动并确认新 Sup nonterminal/fresh 且 adopted 同一个 335208；再获取 coor singleton lock，重新读取并 CAS 验证 state 未变且 pending 明确为 null，验证新 controllers identity 和原 scheduler `_scheduler_proof` 完全相同，随后仅移除 `controllers` 字段。`coordinator.state.proposed.json` 和 field_diff 保存该最小变化，pending/cooldown/sequence/schema/phase 全部保留。`supervisor_terminal` 在 step 中不是 sticky terminal；新 coor 首个正常 step 会重新绑定新 Sup 和原 scheduler，并自己刷新 state/status/process。

若重新检查时 pending 非 null/字段缺失、state 已变、旧进程 live/unknown 或存在未明事务，必须保留现场并审 pending.audit 和 commit outcome，不能用本候选覆盖或清 pending。这里仅对已观察到的 null pending 提供恢复。

待根代理完成资源方案审查后，具体执行入口如下。`reviewed_recovery.py` 默认和 `check` 都只读；本轮只执行了 `check`，结果通过。下列 mutation 子命令均未执行。显式的 resource-policy 参数表示根代理已经审查并决定继续保持本包的 5s/2s 资源策略；如果决定改动策略，则先更新并重新审查方案。

```bash
startup_recovery_script=/root/nfs/embodied/benchmark/results/ease_insights_20261103/process_recovery/local_search_startup_20261105_retry1_review/reviewed_recovery.py
startup_recovery_python=/root/nfs/embodied/benchmark/policies/realtime-vla-flash/.venv/bin/python

"$startup_recovery_python" -B "$startup_recovery_script" check
"$startup_recovery_python" -B "$startup_recovery_script" commit --resource-policy root-reviewed-unchanged-5s2s
"$startup_recovery_python" -B "$startup_recovery_script" launch-supervisor --resource-policy root-reviewed-unchanged-5s2s
"$startup_recovery_python" -B "$startup_recovery_script" rebind-coordinator --resource-policy root-reviewed-unchanged-5s2s
"$startup_recovery_python" -B "$startup_recovery_script" launch-coordinator --resource-policy root-reviewed-unchanged-5s2s
```

`commit` 在 queue lock、Sup singleton lock、coor singleton lock 下做 CAS 和最小状态处理，先使 Sup state 非 terminal，再原子替换 queue；不写 jobs_state。两者并非跨文件原子事务，若异常会保存 partial_commit 记录而不会盲目重复。live scheduler 可能在发布 queue 后立即开始两条 retry，属于正常分配；后续已有 retry case 成果不阻止启动 Sup/coor，也不得重测。

新 Sup 的实际 argv 是 model venv python `-B -u -m preparations.ease_insights_20261103.vlm_local_search_v1.supervisor --root <discovery> --confirmation-root <confirmation> --scheduler-pid 335208 --poll-seconds 10 --launch`。新 coor 是 `/usr/bin/python3 -B -u -m preparations.ease_insights_20261103.gate_opportunity_coordinator --launch`。两者均由 Popen(start_new_session=True, stdin=DEVNULL, stdout/stderr=daemon.log, close_fds=True) 持久启动。helper 的 frozen parser 明确允许 `-B/-u`。

每个实际动作在本包 `execution_receipts/` 写恢复记录。Sup 启动会有界等待 20s 验证新 PID、fresh nonterminal status、waiting_discovery 以及原 scheduler PID/start_ticks；不健康则保留进程供检查，不做第二次自动启动。coor 解绑前只做 OS/file identity preflight，不调用 inventory/gpus/lease/helper.operate；真正启动 coor 后才恢复它原有的正常观察和事务流程。

执行后的验收应读取新 sup/coor process/state/status 和当前 queue/jobs_state：新 Sup healthy、coor controllers.sup 是新 PID、scheduler 始终 335208/start_ticks 2615895606、两个新 ID 被原 scheduler 接纳为 pending/running、旧两条 failed 历史仍在、Jac120 和 21 个有效策略失败均未重测、object/goal/long 的 job 定义与科学字段不变。2s watcher/agent 通知沿用当前部署。任何新的 startup OOM 都作为新技术事件检查，不循环增加 retry ID。
