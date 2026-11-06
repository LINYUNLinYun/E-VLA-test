# E1：L12/L13/L14起仅首轮复用

用户指定三组：L12–18、L13–18、L14–18。每组都仅在Full后的第一轮draft复用最近Full的KV，第二轮用当前V14。固定Full-D10 → hybrid-D2 → V14-D2；不提前刷新Full，不用draft覆盖独立Full快照。

共同协议沿用L15已完成实验：π0原checkpoint、BF16/Triton、V14的1-based skip[3,4,9,11]、H50/R5、P=2、20Hz、官方init、seed7、10步settling、原prompt/请求noise与各suite步数上限；无verify或夹爪保护。三组分别保留当前L1–11/L1–12/L1–13的KV，复用旧L12–18/L13–18/L14–18，共7/6/5层；仅age5实际读取旧KV，age10不读取。

每组先四suite×10task×init14，共40条，合计120条。延续既定扩样门：首40条至少35成功，且同卡同输入周期模型耗时不高于无复用周期V14，合格组固定补init15–18至200条。三组独立决策，不因其他组结果改变门槛；最多600条。有效失败不重试，技术中断保留、有界恢复。首40/new160/all口径分开，非官方50-init全量基准。

冻结后按task交错三组，动态调度到空闲GPU；case、seed、输入与队列完成顺序无关。模型服务持续，configure重置每回合策略与Full缓存。GPU0/1/2/3均可，约30秒检查，没有其他compute进程且显存≤1500MiB才启动，不停止外部任务；其他卡释放可加入。

资源恢复修订：首个GPU1加载过程中被另一任务重新占用而OOM，未产生有效episode；原方案、manifest、代码与启动失败记录已归档。此后须连续约60秒（三次检查）空闲才启动；加载阶段OOM仅作技术启动失败保留，回到等待，每GPU最多2次此类失败（含已有1次）。数值核验失败不重试或放宽容差。科学配置、case/seed、扩样门均不变。

先通过同输入真实模型核验：截断结果须等价于完整计算V14后替换对应旧KV，覆盖三组四suite；保持既定atol=rtol=1e-5。服务输出与历史四suite参考、各case初始输入/Full raw和decoded输出相符后复用旧对照，失败不放宽容差。每个实际请求核验F-H-V、cutoff与旧层数、age5/age10、D10/D2日程、执行前缀、seed和EEF轨迹。

质量对照：无复用periodic_v14、旧L16两轮periodic_old16、L15仅首轮D15_first、L16仅首轮B16_first、L17仅首轮C17_first，并比较此次三组共享case的配对恢复/损失和suite分层task聚类探索区间。不以单请求动作误差代替闭环质量。

同输入成本沿用冻结80对，测Full10/V14/old12/old13/old14/old15，每输入各路径预热后AB/BA三次。每组周期=(Full10+oldN+V14)/3，包含Full刷新，不含RPC/仿真/加载；模型路径成本、实际策略ms/执行动作、回合吞吐分别报告，不把跨运行绝对wall当成加速。

单独目录保存manifest、源hash、历史输出核验、逐case/请求raw与decoded H50、实际动作/EEF、年龄/路由/seed/输入hash、周期测速、决策与中文报告。旧实验代码和结果保持原样。
