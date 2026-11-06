# E4：V14历史残差补偿的真实闭环与部署成本

2026-10-30。用户授权补做复用与外推的闭环收益、部署加速，先判断能否改善V14。使用embodied-eval-acceleration技能，不用子代理。

固定原π0 checkpoint、BF16原生Triton、V14原1-based mask [3,4,9,11]、完整18层AE、H50/R5、D2均匀Euler；周期刷新为Full-D10，此后两次V14-D2，无verifier、夹爪保护、搜索或训练。五条件：pure_v14、periodic_v14、periodic_reuse、periodic_extrapolate、full10_reference。三个周期条件刷新日程相同。

四suite各10 task，官方init19，共40条/条件、200条新闭环。官方prompt、env seed7、10步settling、20Hz、上限220/280/300/520；请求seed沿用idea1-v22。相同case五条件在同一服务执行，按case轮换条件顺序。探索性快速评测，不自动扩大样本或根据成绩调参数。保存成功率、动作/末端轨迹、终止步数、请求、噪声身份与耗时，不存视频及全量图像。

Full请求在四个目标层各保存真实hidden输出减输入的FP32残差（输入/输出本身BF16）。仅缓存各自路径、当前episode的最近两次Full；draft不覆盖。复用h'=BF16(FP32(h)+r_last)；外推r_hat=r_last+(age/gap)*(r_last-r_prev)，不裁剪，不足两次Full回退复用。每层先按当前输入生成该层K/V，再补偿hidden，供下游层使用。刷新、缓存提取/读写、补偿全部计入策略成本；不调用当前Full为draft提供免费参考。reset清空历史，实际执行长度ack后才推进控制步。

原生编码器的局部独立实现保留原算子顺序，跳层确实省掉attention/FFN。四suite固定真实输入核对Full/V14的K/V及动作，捕获残差不改变Full输出；融合补偿与FP32张量参考按atol=rtol=1e-5检查，失败不放宽。重复性与跨服务同输入结果检查。负对照不补偿的V14、无历史回退及episode隔离检查通过才跑科学条件。

固定输入测速复用原160 pair bank按每task一个固定pair取40输入，每输入使用其真实previous/previous2建立固定历史。每服务先四suite预热（每suite每路径5次），测量40输入×五条件×一次，并用逆序第二遍；所有卡相同输入，AB/BA轮换。额外分别计Full刷新含残差提取、无补偿V14、reuse及有两次历史extrapolate的成本；周期成本按(Full+draft1+draft2)/3及每5执行动作折算。固定输入bank不证明闭环质量，闭环缓存始终自生成。固定视觉前向、模型、解码全部计入，同步计时；RPC/仿真另列。

主比较periodic_reuse/extrapolate相对periodic_v14：配对成功率、恢复/新增失败、suite分层task聚类bootstrap2000次探索95%区间、每执行动作策略ms。pure_v14和Full-D10为质量/速度参照。同时报告真实闭环墙钟、episode/hour、动作/秒、请求/秒、服务加载/编译及固定输入延迟；不同轨迹的请求平均不冒充同输入加速。40条不能宣称非劣或等效。

启动前GPU0/2/3空闲，GPU1外部任务占用而排除。每卡持续B1服务、一个环境worker，case动态领取但随机数不随调度变化；只有一个调度者。有效失败不重跑，技术中断归档，最多一次恢复。无空闲资源且仍有工作时每30秒检查可用候选卡，不干扰外部进程。输出results/idea1_v3_4_E4_closedloop及本目录中文报告。
