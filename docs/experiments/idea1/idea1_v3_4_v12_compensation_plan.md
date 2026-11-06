# 用户指定V12与五种补偿：闭环及成本

2026-10-30。用户指定1-based跳层[2,3,5,14,16,17]，保留原18层K/V shell，跳过六层attention/FFN hidden更新，完整18层AE。沿用embodied-eval-acceleration技能，不用子代理，不使用GPU1。

七条件：pure_v12、v12_channel、v12_hadamard、v12_linear、periodic_v12（刷新对照）、periodic_reuse、periodic_extrapolate。前四项全程D2，后三项周期为Full-D10→两次V12-D2，episode首请求Full；没有verifier、夹爪保护、选层或策略搜索。周期=2解释沿用此前已解释的“两次draft后刷新”，不另行偷换为每两请求一次Full。周期对照用于隔离Full刷新与补偿本身的收益。

原π0 released checkpoint、BF16原生Triton，H50/R5，官方四suite各10task、init19，40条/条件、总280条闭环。env seed7、10步settling、20Hz，上限220/280/300/520，请求noise沿用idea1-v22 seed规则。相同case七条件由同一服务执行，顺序按case轮换；真实动作、EEF、success、终止步数、各阶段耗时及噪声身份保存，不录视频、不存逐帧图像。保留既有V14/Full结果仅作为明确标识的历史参照，主配对比较均为本轮新数据。

静态补偿只用旧init14真实观测拟合：80个偶数task0/2/4/6/8请求为校准，32个task1/5请求为验证，旧留出48请求不拟合不选参；闭环init19与之分开，但任务身份部分重叠，不能称跨任务泛化。每请求32个均匀视觉token及最多16个均匀语言token，原native BF16 Full层输出为目标；按六个跳层从浅到深，使用已经拟合的本方法上游补偿生成当前层真实输入，再拟合该层。因此不是把原V14输入参数直接套用。

channel按通道均值/标准差配准，std下限1e-6、scale clip[0.1,10]；Hadamard在正交归一变换域相同配准；linear为ridge=1e-3*mean(diag(Xc^T Xc))的闭式残差回归、随机SVD q40 niter2，seed=341200+layer，最多rank32。linear校准时上游用rank32；全mask rank8/32仅在32验证请求按前5步平移差最低且夹爪差不高于纯V12选择，若均不满足先夹爪差最低再平移；冻结全局rank后不看闭环调参。所有补偿以FP32算并写回BF16，模型权重不更新。静态拟合及选rank成本单列。

历史复用/外推只使用该候选本条episode的最近两次实际Full残差，所有六个目标层均存输出减输入的FP32张量。先按当前输入生成本层K/V，再对hidden补偿。外推r_last+(age/gap)*(r_last-r_prev)，不clip；不足两次历史退回复用。draft不覆盖缓存，reset清空，age由ack的真实执行步推进。提取、存取、补偿及Full动作/刷新成本全部计入。

实现复用已验收原生编码器算子顺序，独立文件不改旧实验。四suite真实输入核对Full、纯V12的K/V及raw/decoded动作；capture不改变Full。静态补偿逐层公式与dtype/有限性检查；历史融合实现与张量公式按atol=rtol=1e-5比较，门禁失败不放宽。跨服务一致性检查通过后执行同case条件，保存adapter hash、校准请求身份、manifest及源码hash。有效失败不重试，技术错误归档至多一次恢复。

GPU0/2/3启动前检查空闲，无外部compute且显存≤1500MiB才使用，一个调度者每30秒发现空卡；不扫描GPU1。校准先在一张空闲卡执行并释放，随后三卡持续B1服务、每卡一个环境worker。闭环前在各自服务做固定输入测速，保持服务直到闭环完成，避免释放卡后被占用导致反复启动。40个固定真实输入，每输入所有九条计算路径均预热5次后正/逆序各测一次；首次编译不混入热态，准备成本另列。固定bank的Full历史仅用于测速，不供闭环候选使用。测速前后及逐输入采样compute进程，共卡测量保留并标识，不冒充独占。

主比较：三种静态方法相对pure_v12；历史两种同时相对pure_v12及periodic_v12。报告suite成绩、配对恢复/新增失败、suite分层task聚类bootstrap2000次探索95%区间、策略ms/实际执行动作、回合墙钟、episode/h、动作/s、请求/s。固定输入全路径及周期摊销成本另表，不把不同轨迹平均耗时当同输入加速。40条探索评测不宣称官方全量或总体非劣，不据结果自动扩样。

输出results/idea1_v3_4_v12_compensation、preparations/idea1_v3_4_v12_compensation及本目录中文报告。
