# V12历史残差复用：刷新周期4/6与前三层方案对照

2026-10-30。用户指定V12跳层组合[2,3,5,14,16,17]＋历史残差复用，周期改4和6，各40条，判断是否比前三层实时更新更稳。采用embodied-eval-acceleration技能，无子代理，GPU1排除。

沿用原π0 BF16 Triton。仅上述六层重放本路径最近Full的FP32输出减输入残差，其余L1–17实时hidden更新、全部18层当前K/V、视觉和完整AE实时计算。周期P严格为Full-D10→P次draft-D2，H50/R5，最大draft缓存年龄P×5实际控制步（周期2/4/6为10/20/30步，20Hz下0.5/1/1.5秒）。不外推、不训练、不换mask、不加verifier或夹爪保护。复用上一轮可变mask Backend和周期路由，只改mask与period。

新条件v12_p4、v12_p6各40条：四suite各10task、init19，共80条新增；同40case的V12周期2和前三层周期2/4/6全部已有兼容结果，逐文件冻结hash后复用。V12周期2首观测不兼容则补跑；前三层参考初观测不匹配则在首动作前停止排查，不强行合并。不能按success筛选参考。环境构建前Python/NumPy seed7、每条件独立新环境、官方init、env.seed7、10步settling、220/280/300/520步上限，沿idea1-v22 case/request噪声。

共享已验证Policy，CPU单测覆盖period2/4/6日程、reset、pending/ack和实际执行年龄。模型服务启动后四suite真实输入核对Full残差提取不改K/V和输出，历史残差重放输出重复且与前轮V12固定输入一致，atol=rtol=1e-5。每case两新条件同服务执行并轮换顺序，记录原始H50输出、动作、EEF、逐请求route/cache年龄/noise/时延及success，无视频和逐帧RGB。有效失败保留；技术错误归档，最多一次恢复。结束后核验三种周期公共调度前缀输出。

仅在空闲且无外部compute、显存≤1500MiB的GPU0/2/3启动，一卡一个持续B1模型和一个环境worker。资源不足时单一调度者约30秒检查候选卡，不占用他人任务；完成释放进程。此前1680次独占固定输入组件计时涵盖V12和前三层，hash核对后按(Full_capture+P×draft)/(P+1)估算周期成本，不伪称本轮新实测端到端加速。另保留本轮闭环策略/RPC/环境墙钟、动作/s、请求/s和episode/h；历史与新轮墙钟仅描述。

报告逐suite成功数、同case配对恢复/新增失败、2000次suite分层task bootstrap探索95%区间，分别比较V12的周期2→4/6、同周期V12对前三层，并列出两方案周期拉长后的成功率下降。频率变化同时改变残差年龄和Full-D10动作比例，不能独归因于残差老化；两mask并不嵌套，不能单凭此断言某组深层必不可少。各40条是探索结果，不是官方全量或等效性证明，不自动扩样。

输出preparations/idea1_v3_4_v12_refresh_sweep、results/idea1_v3_4_v12_refresh_sweep及本目录报告。
