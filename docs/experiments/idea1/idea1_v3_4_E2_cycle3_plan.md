# E2 周期3：四种选层 × V14/V12

2026-10-30。用户指定Full、Fast、Fast循环，比较余弦、每次随机、Jacobian、开始固定随机，分别测V14/V12。沿用主线v3.4 periodic：真实执行Full-D10→Fast-D2→Fast-D2，H50/R5、20Hz、batch1，无verifier、夹爪额外刷新、补偿、旧KV复用。周期3是重规划周期，每周期最多执行15个动作，不能把Full探测冒充执行Full。重置后从Full开始。

8条件各40个匹配case，共320条新闭环。复用上一轮case身份（四suite各奇数task1/3/5/7/9×官方init19/20），不复用上轮纯Fast成功率。V14跳4层，V12跳6层，候选L1–17，L18固定保留KV shell；每mask最长连续跳2层。动作专家全18层。各预算都在全部合法组合中优化/随机，不因预算改变偷偷保护某些层。

余弦：仅Full轮从原生Full VLM已有hidden_trace获取输入/输出，逐token FP32 cosine后平均，在合法组合中最大化相似度和，后两Fast轮复用。复用Full本轮计算，不另跑探测VLM。hidden_trace复制和统计开销都计入Full轮，不将其计为免费。

Jacobian：仅Full轮、相同当前观测上用已验证BF16/FP32 RMS、D2/H50的gate=1可微模型，五步指数权重[1,2,4,8,16]/31、七维全+1，一次VJP选层，后两Fast轮复用。实际执行的Full仍是原生D10。D2可微探测没有改变为D10梯度，也不能自动借用原生无梯度激活；必要的额外前向和反传都只在Full轮计入。CUDA Graph仅捕获该探测；不使用未过数值核验的融合或batching。

每次随机：每次Fast使用独立均匀合法mask，root seed341101，任务/请求seed和预算经SHA256导出。开始固定随机：第一个Full轮抽取一次合法mask，整个episode后续Fast固定不变。同预算两随机策略共享第一次Fast的mask，后续只有是否重抽不同：第一mask在开始Full轮预抽，之后逐次随机每Fast重新抽。两方法每次Full执行均为Full18，与mask无关。不事后挑随机种子；仅本轮预设单种子探索，不能声称随机种子总体最优。

每条件新建环境，构建前seed7，env seed7/settling10、原controller/相机/prompt/噪声流/步数上限220/280/300/520。每case八条件同GPU、轮换顺序，核初态、首输入、首个Full动作一致；后续各自自由闭环，不跨轨迹拿Full/KV冒充当前观测。

上线前：沿用四suite真实输入原生D2参考及Jacobian Graph/eager核验（raw atol=rtol1e-5；梯度atol1e-4/rtol1e-3；mask一致）；原生Full D10带trace/不带trace同输入同noise输出一致；余弦统计复核；V14/V12合法mask；真实路由序列Full/Fast/Fast/Full，Full D10、Fast D2，Fast无Full探测。失败不放宽阈值。有效失败不重试，技术中断最多恢复一次。

输出分suite成功率和配对恢复/损失/探索区间；执行动作数/回合墙钟/吞吐；Full、Fast各自策略和RPC耗时；仅Full的选层开销及按实际请求摊销值；首次捕获成本和实际Full占比。热态统计排除每回合第一个请求，周期稳态另用完整三轮块统计，不能把65ms直接除以3冒充实测加速。完整保存每次mask和路由，审核周期、mask年龄和探测次数。单一场景/种子的小批对照，不按成绩扩到显著。
