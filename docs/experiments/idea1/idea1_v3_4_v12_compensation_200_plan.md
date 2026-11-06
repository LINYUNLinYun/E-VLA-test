# V12补偿扩展至200条及静态补偿周期化

2026-10-30。用户要求将40条/条件扩至200，并检验三种静态补偿加周期是否有效。使用项目embodied-eval-acceleration技能，不使用子代理。候选GPU0/2/3，明确排除GPU1。

保持π0原BF16 Triton、mask=[2,3,5,14,16,17]（1-based）、全部18层K/V及AE、H50/R5、draft D2、Full D10、period2=Full→draft→draft。无verifier或夹爪保护。十条件：pure_v12，v12_channel，v12_hadamard，v12_linear，periodic_v12，periodic_reuse，periodic_extrapolate，以及新增periodic_channel、periodic_hadamard、periodic_linear。静态周期条件在Full轮执行无补偿Full-D10，两个draft轮用原静态adapter；不提取无用残差。历史条件继续提取该路径最近两次Full残差，不跨episode。

四suite×10task×官方init19–23=200条/条件，共2000条有效记录。已有七条件init19的280条经来源、科学参数、初态身份核验后复用；新增三条件在init19首请求执行前必须与旧数据初观测hash相同。若不同，不执行不匹配的旧数据合并，保留旧记录并对该case原七条件以新统一初态补跑；这种选择只取决于初态，不依据成功与否。其他init全部新跑，预期新增1720条，技术不兼容替换另记。

补偿参数完整复用已冻结adapters.pt及rank8，不重新校准、不选rank、不依据闭环调参。继承原80校准/32验证的init14来源。报告原40、新增160、全部200三种口径，新增三周期条件均按相同三种口径；所有方法始终保留有效失败。

每条件独立构建环境，在构建前设置Python/NumPy seed7，再按官方初态、env.seed7、reset、10步settling；20Hz及上限220/280/300/520不变。每case首观测在首个动作执行前校验，所有条件须一致。保存动作、EEF、原始输出、初观测、noise身份、success、步数及计时，不存视频或逐帧图像。原代码和结果只读，新目录单独保存manifest、reuse收据、初态校验、源码hash、进度、退出状态和中文报告。

每GPU一个持续B1服务，两个独立环境进程交叠仿真与模型请求，服务串行推理，不做神经网络batch。每个环境进程有独立Policy、pending/ack、历史缓存和计数；新增实际后端交错/独立序列门禁须在四suite上按atol=rtol=1e-5验证raw/decoded输出一致，禁止放宽；再用首批真实case监测吞吐和初态。case动态领取，但请求seed与完成顺序无关。有效科学失败不重试，技术错误归档后最多一次恢复。每case所有新条件由同一服务处理；复用旧结果显式标识服务差异。

若交错门禁失败，保留门禁结果并使用单环境，不放宽数值条件。每卡按同task的五个init成组调度，组内两个环境并发，组完成后切换task；服务持续运行，环境进程按组重建以确保初始化独立。

既有40固定输入×九路径×两遍×三GPU共2160次独占热态测量在同一checkpoint/adapter/后端hash未变时复用。新增静态周期成本按(Full-D10+2×对应静态draft)/3推导，历史周期按含残差提取的Full摊销。此为复用组件测量的稳态估计，不冒充新跑整体计时。实际闭环策略wall/动作、RPC含队列、回合墙钟、episode/h、动作/s、请求/s另外实测；双环境吞吐与单请求部署耗时区分。

主要质量比较：五种周期补偿分别对periodic_v12，以及三个新增周期静态条件对各自无周期条件；原三静态对pure_v12和周期对pure_v12为辅助比较。suite分层task聚类bootstrap2000次95%探索区间，五个init作为task内重复，列恢复/新增失败。未校正多重比较，200条仍不是官方50-init全量。全量结束后统一出报告，不按前几批成绩停掉方法。

输出preparations/idea1_v3_4_v12_compensation_200、results/idea1_v3_4_v12_compensation_200及本目录报告。
