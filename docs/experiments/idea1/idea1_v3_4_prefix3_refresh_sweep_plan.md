# 前三层实时更新：刷新周期4/6快速测试

2026-10-30。用户要求把刷新周期增加至4和6。沿用上一轮仅L1–3实时hidden更新、L4–17重放最近Full残差的方案，视觉、所有18层当前K/V、完整AE保持当前计算；L18本来只有K/V。采用embodied-eval-acceleration技能，无子代理，GPU1排除。

两个新条件prefix3_p4、prefix3_p6各40条，四suite×10task×init19，共80条新闭环；复用前轮prefix3_reuse周期2的同40case（28/40）为参考。周期P严格为Full-D10→P次draft-D2，H50/R5，均匀Euler，无verifier/夹爪保护/外推。缓存仅保存本路径最近一次Full的L4–17输出减输入FP32残差；draft不覆盖，reset清空。每层先投影当前K/V，再加历史残差写回BF16。周期2/4/6的最大draft缓存年龄为10/20/30实际控制步，即20Hz下0.5/1/1.5秒；Full到Full间隔通常15/25/35步。实际年龄按ack执行步数记录，不按理想轮数伪造。

seed和noise沿用idea1-v22，环境构建前Python/NumPy seed7、每条件独立新环境、官方init、env.seed7、10步settling、上限220/280/300/520。每case新两条件在同一服务执行并轮换顺序，首动作前校验初观测一致，参考仅在初观测和checkpoint/路径一致时复用；不兼容则本轮补跑周期2参考。所有有效失败保留，技术错误归档至多一次恢复。不看结果调周期/参数，不自动扩样。

只改刷新路由参数，不改已验证Backend、残差公式、dtype或模型权重。CPU单测检查2/4/6日程、reset、pending/ack、实际年龄。四suite真实输入核对Full capture不改K/V或动作、过去Full残差重放的重复性及与上一轮相同输入输出对齐，atol=rtol=1e-5，不放宽。保存原始H50动作、EEF、success、步数、每请求路由/缓存年龄、noise身份及耗时，无视频和逐帧图像。

组件成本复用上一轮每卡560次、三卡1680次独占固定输入计时，逐文件hash验证；按(Full_capture+P×draft)/(P+1)估算稳态耗时，明确是复用组件估计。本轮另外实测每动作策略wall、RPC、回合墙钟、episode/h、动作/s、请求/s。缓存年龄分布按draft请求单列。刷新频率变化同时改变缓存新鲜度和Full-D10动作占比，因此质量变化不能单独归因于残差老化。

GPU0/2/3当前有外部任务，准备后仅在无外部compute且显存≤1500MiB时启动，一个调度者约每30秒检查候选卡，不占用他人进程；每卡一个持续B1服务、一个独立环境worker，完成后释放。所有case完成后不为尚未空闲的未用卡继续等待。科学结果按suite/task配对，bootstrap2000次探索95%区间，40条不能宣称等效/非劣/官方全量。

输出preparations/idea1_v3_4_prefix3_refresh_sweep、results/idea1_v3_4_prefix3_refresh_sweep及本目录中文报告。
