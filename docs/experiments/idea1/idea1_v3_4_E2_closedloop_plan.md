# E2选层指标：快速留出闭环对照

2026-10-30。用户授权后台快速评测余弦选层是否影响闭环效果。当前4张4090均无compute进程，候选GPU0–3，启动前重查归属，不干扰其他任务。沿用项目embodied-eval-acceleration流程，四卡独立B1服务、每卡一个环境worker，无视频和新梯度计算。

固定π0原checkpoint、原BF16 Triton、D2均匀Euler、H50/R5、官方初态、原prompt、seed7、10步settling、原suite上限220/280/300/520。所有mask保留原层号KV shell，只跳VLM attention/FFN更新；AE保留18层。纯剪枝执行，无Full周期刷新、verifier、夹爪保护或补偿。

校准仅复用E2原40请求中的偶数task0/2/4/6/8（四suite20请求、init14）。每个请求对L1–17排序，层号打破并列；余弦按输入输出余弦降序；Jacobian按前5步raw动作的Jacobian列范数升序，实际动作影响按单层真实跳过相对Full的raw动作RMS升序。后两者平移/旋转/连续夹爪分别在17层内排序后等权平均。每种指标再对20个校准请求等权平均秩，冻结最低4层的全局静态mask，不使用闭环结果选层或调权。

五条件：cosine4、jacobian4、action_mse4、原V14[3,4,9,11]、Full-D2。前四项均固定跳4层；不称严格实测时延相等，实际每动作成本单列。两个选层方法若产生相同mask则只执行一份并显式标记别名，不重复跑同一科学条件。

闭环使用留出奇数task1/3/5/7/9、init19/20，四suite×5task×2init=40条/条件，最多200条新闭环。校准task及初态均与测试分开。相同case的所有条件由同一模型服务执行，条件顺序按case轮换；case/seed不由完成顺序决定。有效科学失败不重试；技术中断归档，最多一次恢复。

原始EEF、动作、噪声身份、success、终止步数、请求与策略wall保存。模型服务四suite固定真实输入检查重复性，并与原生Full-D2参考按atol=rtol=1e-5核对；失败不放宽容差，不合并失败服务。闭环结果全部新跑，旧成功率不直接拼接。

主要比较：cosine4与action_mse4的配对成功率；jacobian4、V14与Full-D2为辅助比较。按suite分层task聚类bootstrap95%探索区间，恢复/新增失败和逐suite成绩同时报告。40条小批没有差异也不能声称等效；不能将动作影响或单个mask输赢推广为余弦方法在所有预算/模型上失效。不据初筛结果自动扩样或改mask。

准备/校准和闭环部署成本分开。离线Jacobian成本不加入每次控制请求，但要明确这是离线静态选型，不是已实现在线梯度指标。后台自动保存manifest、GPU/PID/退出状态、进度和中文报告至results/idea1_v3_4_E2_closedloop与docs/experiments/idea1。
