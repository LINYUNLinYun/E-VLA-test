# E2 时间权重在线选层闭环

2026-10-30。用户要求线性随机候选、指数固定候选与五步等权比较实际闭环效果和开销。为匹配七维输出方向，设置四条件：uniform_random、linear_random、uniform_fixed、exponential_fixed。随机方向固定 seed340710 的第一组七维±1；fixed 七维全+1。时间权重依次为均匀[1,1,1,1,1]/5、线性[1,2,3,4,5]/15、指数[1,2,4,8,16]/31。全部一次VJP。

复用原闭环40个 case：四套件各 task1/3/5/7/9 × init19/20；每条件40条，共160条。沿用π0原生BF16 checkpoint，D2/H50/R5、20Hz、settling10、env seed7、原步数上限220/280/300/520。每条件每case重新构建环境，在构建前播种；同case所有条件在同GPU运行，轮换条件顺序。核初态向量、BDDL、controller、prompt和首输入hash一致，不复用历史静态mask成功率当在线基准。

每次重规划：当前真实观测原生变换和vision；同checkpoint共享冻结BF16权重的可微PyTorch Full门控模型，gate=1计算单次VJP；依据17层abs梯度选择skip4，禁止连续三层；用当前观测原生Triton pruned VLM+完整18层AE生成动作，使用同一显式噪声seed，执行前五步。选层每次即时完成，不跨观测复用、无周期更新、无异步流水线、无补偿。可微Full计算不用于执行，全部成本计入。

CUDA Graph仅捕获可微前向+VJP，权重buffer作为GPU输入，可跨条件更新；每条件开始释放旧图，首次请求准备成本单列并计入总墙钟。借用原生GPU权重避免第二份大模型。采样前重新原生编码，保留原skip/KV语义。确定性torch与CUBLAS_WORKSPACE_CONFIG=:4096:8；FP32 RMS，无额外compile融合或batched-VJP近似。

上线前四套件已存真实请求核验：原生Full与存档atol=rtol=1e-5；实时vision投影/state与原始导出输入一致；共享权重PyTorch与已验证BF16导出模型一致；Graph与同输入eager raw atol=rtol=1e-5、gate梯度atol1e-4/rtol1e-3、mask完全相同，覆盖四条件不同权重。失败不放宽阈值；技术中断最多恢复一次，真实失败不重试。

输出成功率/分套件、配对恢复与新增失败、步数、每请求RPC/策略总耗时、选层热态/冷捕获、原生pruned VLM/AE/vision、完整episode墙钟、episode/小时与动作/秒。完整保存每次mask、17维分数、动作raw与执行遥测；首观测保留，不额外视频。包括冷态的费用与热态并列，不把机器人100ms预算与固定视觉输入核心时间混为同一口径。仅本轮独占空闲GPU调度，等待新卡时每30秒扫描。
