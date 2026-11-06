# 一次AE的未来动作先验：固定tau0.10小批A/B

用户授权：固定tau0.10，四个LIBERO套件各task0/2、官方init19、seed7、settling10，每组8例、共16条新闭环。只读复用同8例的保持夹爪tau.1、纯夹爪标准噪声tau.1和GaussianD1（24条）；不自动扩样或改参。

A：六维Kalman（既定冻结参数及裁剪）＋上轮AE尚未执行的未来夹爪序列。
B：上轮完整七维decoded控制动作序列平移，无额外预测裁剪。
两者均在真实ack后按实际执行r（1至5）平移上轮H50预测，最后预测动作重复r次补尾。只有真正执行的前缀更新KF，未执行预测仅作先验。用当前真实output_transform的可逆仿射坐标重新编码，25维padding先验都为0；所有维统一x_tau=.9*encoded_prior+.1*epsilon，t=.1，一步Euler dt=-.1、NFE1。首轮没有历史时均保持原GaussianD1；reset清空未来缓存及滤波器。

固定V12跳层[2,3,5,14,16,17]；Full只在重规划0/3/6刷新完整VLM残差，始终执行子路径D1动作，自身最新两组残差外推。无Full AE、Full动作、验证、夹爪保护、回退或encoder/AE CUDA Graph。H50/R5、仿真20Hz、无墙钟sleep。

同case同持续B1服务交替A/B顺序，初始观测/噪声/初态一致，首轮raw与decoded动作原atol=rtol1e-5。CPU核验部分ack、尾补、当前坐标重新编码、bootstrap及reset；GPU四suite短预检对照PyTorch残差reference，覆盖refresh0/3。一个poller每30秒只发现自然空闲卡并要求稳定60秒，以GPU UUID锁定；最多2张卡。保留有效失败，技术异常最多单次重试，外部显存冲突归档后有界续跑。全部完成后释放自有服务及锁。

报告A/B每suite及总成功数、同case恢复/新增失败、实际执行夹爪切换/100动作及闭合持续性、AE与策略热态时延、准备与完整墙钟。历史性能仅描述，不当受控加速比。每组8例属于已观察任务上的探索结果，不推断40例准确率。低tau可能不足以修复错误运动/夹爪先验；A不能保证优于B或摆脱0。
