# 周期2：全残差复用与仅前三层更新快速闭环

2026-10-30。用户要求快速测试两个候选：全复用历史残差；仅L1–3实时hidden更新，后续全部复用残差。使用embodied-eval-acceleration技能，不使用子代理，排除GPU1。

固定原π0 BF16 Triton checkpoint、视觉塔、18层K/V投影、18层AE、H50/R5、D2均匀Euler。周期2=Full-D10→两次draft-D2。每次Full仅提取本条件需要的层残差：all_reuse为L1–17；prefix3_reuse为L4–17。L18原生仅产生K/V，没有hidden更新，故无L18残差。draft每层先从当前hidden投影该层K/V，再用最近一次本路径Full的FP32残差替代该层attention/MLP更新，写回BF16。无外推、verifier、夹爪保护或额外Full参考。当前图像、state、noise始终新鲜，不复用旧K/V或旧动作。reset清空缓存，按实际执行步ack推进年龄。

四suite各10task、官方init19，40条/条件，共80条新闭环，不自动扩样。env seed7、构建前Python/NumPy seed7、每条件独立新环境、10步settling、20Hz、上限220/280/300/520；请求seed沿用idea1-v22。同case两个候选同服务执行，顺序轮换，首动作前检查两条件初观测一致。复用已校验的V12历史复用40条（29/40）作为明确标识的参考，须核对checkpoint、原生输出及首观测；若某case不兼容则保留旧数据，对该参考case新跑，不把不匹配结果合并。

先做四suite真实输入工程门禁：新Full与原生Full的K/V和raw/decoded动作一致；无history的skip路径与原生对应mask一致；相同输入的Full残差重放恢复Full输出；过去真实Full历史上的融合复用与FP32张量参考atol=rtol=1e-5；V12六层路径与上一轮同输入参考对齐。失败不放宽容差。保存源码、manifest、checkpoint身份、GPU UUID、进程状态、cache路由及原始动作/EEF，不存视频和逐帧图像。

成本不能只看draft：固定40个真实观测、每suite各10task，三种配置（两个候选和V12复用参考）分别测含残差提取的Full-D10、draft-D2，再测无捕获Full-D10，共七条路径。每输入每路径先预热3次再正/逆序各测一次，每卡560个测量请求、840次预热；按(Full_capture+2×draft)/3摊销，包含视觉、模型、残差捕获/修补、解码，准备和预热另列。逐输入采样compute进程，只把未见共卡的测量作为独占速度对照。保留cache字节、模型/请求延迟和闭环墙钟、episode/h、动作/s、请求/s。固定bank历史只用于测速，不供闭环使用。

GPU0/2/3启动前确认无外部compute且显存≤1500MiB；一调度者每30秒发现可用卡，已运行不重复扫描；每卡一个B1持续服务和一个环境worker。有效科学失败不重跑；技术错误归档，至多一次恢复。质量按suite及task配对，suite分层task聚类bootstrap2000次探索95%区间，40条不能宣称等效或总体非劣。所有结果以统一口径与同周期V12复用参考对照，避免把全层残差复用说成直接缓存旧Full KV。

输出results/idea1_v3_4_residual_replay_quick、preparations/idea1_v3_4_residual_replay_quick及本目录中文报告。
