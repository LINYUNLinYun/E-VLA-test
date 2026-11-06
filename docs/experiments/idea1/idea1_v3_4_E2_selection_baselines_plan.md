# E2 Jacobian 与随机/余弦的匹配闭环对照

2026-10-30。此前20/40仅证明四个Jacobian输出权重方案中最高，尚未证明优于随机或余弦。旧余弦10/40为离线固定mask，不能直接作本次在线选层基准。

复用已完成 online_time_weights 的指数时间权重+七维正等权 Jacobian 40条（exponential_fixed）；不重跑成功率。新条件：cosine_online、random_online_s1/s2/s3、random_episode_s1，每条件40条，共200条。场景、checkpoint、D2/H50/R5、skip4/max2、允许候选L1–17、执行AE全18、显式噪声流、原生BF16执行、控制和步数上限均继承Jacobian实验。每条件新建环境，构建前seed7，同case新基准全部在同GPU执行，并核对新基准与已完成Jacobian的初态、首观测、prompt、BDDL、控制器一致。

随机必须在所有合法四层组合中均匀采样，不能把随机分数的受约束最优解冒充均匀随机mask。独立root seed预设为341101/341102/341103，以namespace+root+请求噪声seed的SHA256确定局部随机流，保证任务调度无关。online每次重规划重新选；episode_s1与online_s1共享第一请求的mask，后续整回合固定，作为换层时序的辅助对照。三随机种子均公开，不挑最差随机种子。

余弦：当前观测Full VLM每层隐藏输入/输出，token逐个FP32 cosine后平均，score=1-mean_cos，约束组合中最小化score之和。使用与Jacobian共享的BF16原生冻结权重、FP32 RMS可微模型的无梯度encode路径，CUDA Graph捕获encode+余弦，无AE、无反传；实际选层和真实成本全部计入。原生pruned动作仍在相同当前观测和噪声上重新推理。

四套件固定真实输入上线核验：沿用原Jacobian Graph/eager及原生参考核验；余弦Graph/eager分数atol=rtol=1e-5、mask相同；切换观测和噪声时更新图输入。随机mask合法性及确定性检查。容差不因失败调整。首次图捕获计入整回合墙钟；热态成本排除每回合首请求，对随机也同样排除。

结果：每种子成功率和范围，Jacobian相对余弦、每个随机种子、随机种子均值的配对差和suite分层task聚类bootstrap区间。种子均值先在同case内平均，不能将120条随机轨迹当作120个独立case。报告成功率与策略/选层/RPC时延、episode墙钟、步数、mask变化率；不因结果扩样到显著。

同时复用40个FP32离线精确Jacobian和单层删除反事实，比较原始整体R5扰动下的候选四层平均扰动：指数固定、当前观测余弦、全部合法随机mask的精确期望。此指标是独立单层反事实的平均，不是四层同时删除或闭环成功率。它仅作为机制证据，不能替代闭环对照。技术失败最多恢复一次，真实失败不重试；新证据如不支持Jacobian优势则明确报告。
