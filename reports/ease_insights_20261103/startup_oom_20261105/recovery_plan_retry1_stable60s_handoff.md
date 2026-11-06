本文件是独立交接说明，不修改已经交接的 stable60s 恢复包或候选。根代理确认已执行 commit，发布后 `check-adoption` 已通过（五条 job 当时均 pending，真实候选资源参数 60s/2s），随后成功启动并健康验收新 Sup PID 437956；根代理正在执行 coor 重新绑定和启动，最终 coor 健康结果以根代理的 execution receipt 为准。

已经交接的包：`results/ease_insights_20261103/process_recovery/local_search_startup_20261105_retry1_stable60s_review/`。候选 queue SHA256=`83cb279d6637f64f0d49173d91ddd6c13de50d8a259c1d23e27f388b067289e4`；本代理在停止进一步检查前只读观察到真实 queue 与该 hash 一致。原 scheduler 继续为 335208/start_ticks 2615895606。

准备时和独立审阅时，完整 precommit `check` 均曾 exit 0。提交后本代理尝试保存同一 precommit check 的 stdout，该检查按设计因真实 queue 已不同于 before 而拒绝；随后根代理确认是自己已经 commit。此拒绝属于 CAS 前置条件生效，不是候选被错误应用。已停止所有 prepare、precommit check、候选修改和追加测试。

最终准备材料复核包括：递归 queue diff 恰 14 项；五个未完成 job 仅资源 idle5→60、poll2 保持，两个失败 job 仅另加 retry ID/receipt；Jac 已完成 job 整体不变。严格 stage 120/720，Jac99 成功和21有效策略失败全部保留，Full/spatial各0，invalid/unexpected为0；22份归档副本 hash 经本代理和独立审阅代理分别验证。55个部署/冻结 source hashes 完整保留，deployed gate 为 `b4b5ca1bf95c8f43103e9a40b4de8b8cd8036bc6c870b12c2bf58dac6653e4cf`。

reviewed_recovery.py SHA256=`559a46b3dc220a2c6582c548129f8fcfeeabbf388fecfa8e8c439691a4405a15`，mutation flag 仅接受 `root-reviewed-stable60s-poll2`，旧5s flag已验证被拒绝。两份 state 候选既 pin SHA 又从 archive 重构完整期望状态；coor 解绑要求 pending明确null、旧进程已dead、新Sup健康、同一scheduler proof，保留 sequence/cooldown；coor启动要求live state整对象等于候选，且有45秒有界OS/state/status绑定健康验收。

本代理的实际操作始终只限准备、归档、严格加载和只读检查，没有执行 commit、启动评测或控制器、发送 signal、查询GPU。live mutations、发布后验收和恢复启动由根代理执行。`commit` 是实际重新入队边界：原 live scheduler 可在 queue 发布后立即调度，这一点应作为恢复时间线的起点。
