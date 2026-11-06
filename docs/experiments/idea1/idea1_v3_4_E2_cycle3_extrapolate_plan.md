# 动态选层叠加历史残差外推

2026-10-30。用户询问另一对话的外推残差修补叠加当前两种选层如何。复用项目已完成的v12_compensation_200实现与200条证据，不将静态V12的134→169/200直接外推为动态选层结果。本轮余弦/Jacobian×V14/V12四条件各40条，共160条新闭环；无补偿参考复用cycle3的余弦和cycle3_full10_jacobian的修正Jacobian，旧错误输入的Jacobian行不使用。

保持Full-D10→Fast-D2→Fast-D2、H50/R5、max2、V14跳4/V12跳6、AE全18。Full上评分后两Fast复用mask；余弦从原生Full trace取分，Jacobian使用修正初始输入与D10可微重放、前5步指数时间权重/七维正等权的一次VJP。两者均未因叠加补偿而重新调选层公式。

每次实际Full采集全部L1–17的原生hidden残差R_l=H_l−H_(l−1)，FP32存储，仅保留本条件自身路径最近两次Full及其实际执行步号。采集全部层而非仅当前mask，避免动态mask换层后缺少该层旧历史。记录与修正初始语言输入的顺序分离：残差来自实际原生Full trace，可微重放恢复原始语言嵌入，不能把重放输入写回后再当作Full的hidden trace。

Fast中保留层实时计算，跳过层仍生成当前K/V shell，并在该层hidden更新处加上预测残差：Rhat=R_last+(age/gap)*(R_last−R_previous)。gap为两次Full之间实际执行动作数（正常15），age为距最新Full的实际动作数（5或10），比例为1/3或2/3。首Full无更早历史时回退R_last。实现复用已验证patch_hidden Triton核，FP32计算后写BF16，enable_fp_fusion=False。不是缓存旧K/V，不是修补输出动作。Full不加补偿，后续Full始终从当前观测完整重算。

前置核验：新Full与参考Full-D10输出一致；在同观测下将Full当步真实残差放回任意合法skip4/6，应重建Full VLM+D2动作（atol=rtol1e-5）；外推内核与原FP32公式reference路径同输入同噪声一致（相同比例1/3、2/3）；修正Jacobian输入身份/Graph梯度日程核验沿用并执行；真实路由Full/Fast/Fast，首次回退、双历史年龄、mask更新后的按层取缓存均审核。失败不放宽阈值。

场景继承相同40case：四suite奇数task1/3/5/7/9×init19/20。独立新环境，构建前seed7，原controller、settling10、噪声流和步数上限不变。四条件同case同GPU、轮换顺序；与各自无补偿参考核首观测、初态、首Full动作一致。每条件自由闭环，不能使用另一方法轨迹的Full残差。真实失败不重试，技术失败最多恢复一次。

报告成功率/分suite、补偿前后配对恢复损失与探索区间、完整三轮稳态/Full/Fast/选层/采集成本、回合墙钟及冷捕获、两份历史缓存显存。补偿开销计入实际策略，不能套用另一会话的静态mask组件时间。初筛固定40条，不按结果扩至显著。
