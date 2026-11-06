# 周期3四选层：补齐随机补偿与余弦/Jacobian扩至200

2026-10-31，用户授权余弦/Jacobian扩至200评测，两项缺失随机补偿各40。主条件沿用上一份四方法表：原始Jacobian一次正权标量VJP、正确Full-D10/语言输入、每个Full选层；不改误差方向评分、不用多投影/稀疏刷新。V14/V12、补偿无/残差外推均保留。

矩阵：cosine/jacobian × V14/V12 × none/extrapolate，8条件各200；复用各40，共320，新增1280。random_online/random_fixed × V14/V12 × extrapolate，4条件各40，共160新回合。总目标1760条记录，新跑1440。随机无补偿40条只作已完成参考，不重跑。先完成random40，再做expand200；未按成功率增删条件或扩样。

200case=4 suites×task0..9×init19..23；40case为原奇数task1/3/5/7/9×init19/20。与既有固定V12补偿200完全同case，可追加匹配参考，但它不替代回合随机基线。原40与新增160另列，200主表和新160验证表分开，不把旧多次探索当全部新样本。

周期固定Full-D10→Fast-D2→Fast-D2，H50/R5，候选L1–17，V14跳4、V12跳6，最多相邻2层，AE全18。余弦沿Full trace评分，Jacobian完整Full-D10对前5×7动作以[1,2,4,8,16]/31×七维+1做一次VJP。random_online首次Full抽mask，其后除首Fast外每个Fast重抽；Full不另抽。random_fixed每回合首次Full抽一次、整个回合保持，跨回合按case噪声seed抽不同mask。随机根seed341101，与原40相同；不按结果选随机种子。

外推每Full采集全部17层真残差，保留本条件自身最近两个Full；首周期hold，之后Fast ratio=1/3或2/3，age5/10、gap15；KV来自当前观测。随机动态mask变化时也有全部层历史，不能只缓存上一mask。Full自身不补偿。无补偿路径复用原执行逻辑，不静默加入残差。

数值：冻结源代码/计划/参考manifest、audit、summary及复用记录hash。新统一策略先与冻结原无补偿/外推策略在真实同输入、同噪声、同历史下比较raw/actions/mask（1e-5）及梯度（1e-4,1e-3）；随机补偿恒定观测须恢复Full VLM+D2 oracle，保留首历史回退和新mask取全部层证明。既有Full10输入/Graph梯度与补偿核证明复用，不重做35反传实验。

调度：最多3张未被占用GPU，持续B1串行服务，每卡同一suite/task的2个独立环境/策略客户端交叠。先对两个Jacobian客户端（无/有补偿、不同预算、相同prompt不同真实观测/seed）做独立运行vs交错运行的raw/actions/mask/梯度隔离门。内存不足或隔离不通过，记录工程门失败并退回1环境，不放宽数值容差；不把工程门当科学结果。组切换前清理所有客户端图/历史，防止prompt运行时重复占内存。环境仍Fresh、seed7、settle10、20Hz及原suite上限，不新增视频。技术中断最多重试一次，保留已完成有效失败；约30秒发现空闲卡，不使用其他会话占卡。

每case各条件旋转执行顺序，显式噪声只由case/replan决定，不由actor/完成顺序决定。初态向量、BDDL、controller/prompt/seed/settle必须一致。共同首Full raw/actions、首Fast输入需匹配；同方法有/无补偿初始mask一致，两个随机补偿条件首mask及首Fast输出一致。若出现仅观测hash/共同前缀异常而官方状态协议正确，保留全部有效回合、不重复到匹配；从相关严格配对排除并公开差异，all-case敏感性分列。结构/路由/源hash/跨客户端串历史错误直接失败，不用统计排除掩盖。

输出：补齐随机后先生成四方法40条表；最终8条件200与新160、四方法原40、各suite、同预算补偿前后/方法间配对恢复与新增失败、suite/task等权点和suite分层task聚类bootstrap2000次探索区间（未校正多重比较）。新160任务初态数不均，点估计与区间统一task等权；另列逐episode描述。随机结果仅一个预设根seed，不代表随机种子总体。

成本记录policy内耗时、RPC排队（双环境时单独）、完整回合墙钟/冷图、加载核验、实际总吞吐。稳态完整Full/Fast/Fast块从第二Full开始，与旧表同口径；较小自由轨迹时延差不是同输入因果加速。新160实际墙钟与含旧320描述分开。既有PyTorch D10导数近似原生BF16，不声称精确原生Jacobian。
