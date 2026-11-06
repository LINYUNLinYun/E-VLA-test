25秒/2秒门槛后Full启动OOM：只读核验

本run属于原gate→runner路径，identity没有新persistent session/keeper的执行revision；此次失败不是新keeper路径的测试结果。

Full PID579554绑定GPU1完整UUID …b202。gate记录连续空闲26.422秒（门槛25秒），exec前fresh scan为1MiB/0 compute。server在real-input gate首次backend.prepare期间，于Pi0Inference1633报CUDA OOM；这不是首次CUDA申请，日志提示可能异步暴露错误。

北京时间：watch33–36于10:59:53.700记录四卡空闲；run目录名对应11:00:20.694；event39于11:00:25.700记录GPU1重新18424MiB/1compute，距释放约32秒；11:00:57.700回到1MiB/0compute；scheduler在11:00:59.181记录该job failed；event42在11:01:15.700再次18424MiB/1compute。

run只有identity.json和server.log，无protocol/gate/preparation/cross_service/worker/result等健康/科学产物。Full episodes中只有预检空lock，无result/complete/episode_receipt，0新case。Jac原120组receipt/result/complete均valid、同manifest，原4 suite×10 task×3 init case集合完整，stage marker仍在；未改动或重跑case。

失败后18GB进程重新出现且后续仍busy，支持存在其他持续占用者。但events未记录当时host PID，server没有健康PID/NSpid映射；不能把后续host PID映射到本地，也不能反推为OOM时进程。现有材料不足以严格证明双方同时显存重叠或精确OOM因果。

retry2_25s恢复包只读审阅PASS：queue仅Full ID和outer status-file变化；科学inner argv完全相同。supervisor state仅needs_attention→waiting_discovery并删除error/attention_unix，其他证据保留，旧failed记录保留。before/proposed四项SHA256见diagnosis.json。

原supervisor没有--resume。needs_attention属于terminal；必须先由root完成已审阅的queue.lock CAS及state归档/复位，并fresh确认旧supervisor已退出。随后运行以下原源码CLI（本代理未执行）：

/root/nfs/embodied/benchmark/policies/realtime-vla-flash/.venv/bin/python -u -m preparations.ease_insights_20261103.vlm_local_search_v1.supervisor --root /root/nfs/embodied/benchmark/results/ease_insights_20261103/vlm_local_search_v1/discovery --confirmation-root /root/nfs/embodied/benchmark/results/ease_insights_20261103/vlm_local_search_v1/confirmation --queue /root/nfs/embodied/benchmark/results/ease_insights_20261103/queue.json --scheduler-output /root/nfs/embodied/benchmark/results/ease_insights_20261103/scheduler --scheduler-pid 335208 --output /root/nfs/embodied/benchmark/results/ease_insights_20261103/vlm_local_search_v1/supervisor --poll-seconds 10 --launch

本代理只读现有文件并写attempt25s报告，无GPU/进程查询、信号、queue/state/source修改或重启。
