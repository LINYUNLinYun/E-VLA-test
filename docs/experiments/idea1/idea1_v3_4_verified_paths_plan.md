# V12/V14外推D2＋验证：周期3/5对Full-D10

2026-11-01，用户要求在当前四次反推400例完成后，继续评测V12/V14＋外推＋D2作为Fast、周期3/5，接入既有验证机制、不使用Kalman draft，对比baseline成功率和加速比并分析请求占比。

选层沿用最近上下文的四次Jacobian＋只允许间隔跳层；已向用户发出可选澄清，并明确此默认解释。selection.json保存最终设置及来源，冻结后不得热改。若用户在启动前更改，另存新冻结revision。四次为共享一次Full-D10前向，原正权＋三个固定seed340610正负方向，RMS评分；Full的反推成本全部计入。V12跳6层，V14跳4层；若用户指定原固定组合，则使用预声明V12[2,3,5,14,16,17]、V14[3,4,9,11]，不含在线选层成本，单独标明其相邻层口径。

四候选各200例，baseline Full-D10同200例。case沿用4 suites×task0..9×init19..23；原checkpoint、BF16/Triton、H50/Rmax5、env seed7、settling10、20Hz、prompt、原步数上限、case/replan噪声均保持。已有init19的40条Full-D10仅在身份/参数及当前首Full raw/actions兼容时复用；不兼容保留拒用证据后新测。其余160条Full新测，总记录1000，预计960–1000新回合。旧基线时延不混入同轮主要速度比。

验证严格沿用v3.3 π0父系统：重建时间t=.10/.05，空间tau=.10，前六维两分支最大误差阈值，允许连续前缀部分接受；保持verify/post夹爪保护。优先级为启动Full、pending gripper Full、周期Full、Fast→verify；零接受后付Fast＋verify成本再Full-D10接管。周期3/5指最近一次实际Full后最多2/4轮Fast，拒绝/夹爪可提前Full并重置计数；不声称实际Full率固定1/3或1/5。

Fast使用当前观测的V12/V14 VLM和D2 AE，外推仅来自本策略自身最近两个真实Full。每次实际Full（含拒绝接管）采集17层FP32真实残差，并独立clone verifier Full KV。首次历史hold，之后ratio=实际环境步age/两Full实际gap；部分接受和提前Full会改变ratio，允许沿用原公式超过1，记录age/gap/ratio分布，不静默截断。Full本身不补偿。零Kalman实例、forecast和Kalman draft调用。

冻结代码/方案/设置/旧参考哈希，当前四次反推评测complete和audit通过后才调度GPU。最多3张空闲卡，持续B1服务＋两个隔离环境，同task交叠；无卡约30秒查询。候选与baseline在同case组同卡执行，候选四组按case轮换顺序，baseline最后以便核对旧结果兼容性。有效失败不重跑，技术中断最多一次额外尝试。

CPU覆盖周期3/5、固定/单/四投影、部分执行步号、拒绝与夹爪接管计数、独立Full KV、reset及实际外推ratio。GPU核对旧固定V14无补偿strict父策略与新兼容控制、同输入Full-D10、验证接入和双客户端隔离；复用当前四投影的真实模型梯度证明，不重复完整机制实验。保留所有有效描述结果，数值共同前缀异常从相关严格配对剔除并另报完整200敏感性；协议、跨轨迹缓存、路由或源hash错误直接失败。

成功率每项200和suite分项，对Full-D10做恢复/新增失败、suite分层task聚类bootstrap2000次seed281034，探索区间未校正多重比较。成本主口径为同轮新baseline共同case的累计policy wall/实际执行动作，包含反推、被拒Fast、verify、Full、输入输出及冷捕获；这是自由闭环实际成本比，轨迹/请求组成不同，不将其解释为单模块因果提速。另列每请求耗时和真实评测wall/吞吐、模型准备/冷捕获。

为提供同卡同输入Full-D10参考，预选四suite的task0/2/init19共8例，保留各候选前15个实际请求输入；每个task组结束后，用同一服务在这些输入上预热并顺/逆序两次测量Full-D10，不执行参考动作、不修改正式策略状态。保存输入hash、噪声、同卡身份、重复输出核验。参考耗时与原候选请求耗时逐键配对，报告请求成本比及实际执行长度归一化成本比；候选与参考测量顺序非同时反平衡，作为同输入服务参考，不声称完全受控因果测速。profiling调用与耗时单列，不混入科学请求占比分母。

请求分为互斥的启动Full、周期Full、pending夹爪Full、空间拒绝Full、夹爪拒绝Full、Fast完整接受、Fast部分接受，总和100%。另列Full/Draft/Verify实际调用/100请求（可相加超过100，因为拒绝同请求含三者）、verify接受长度0–5分布、Fast实际执行动作占比、Full原因占比及时间占比。统计实际Full年龄、残差gap/ratio（含ratio>1比例）、各段Full/Fast/verify/selector冷/热成本。
