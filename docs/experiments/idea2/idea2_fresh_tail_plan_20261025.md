# π0.5 D10：末轮全新 KV 修正实验

用户授权：继续方向1，验证前几轮使用逐层新旧KV兜底、最后1–2轮强制全新KV。

## 条件与范围

- 完整π0.5 LIBERO，B1/H10/D10/R5，lead3，固定权重、BF16主体、原噪声/时间日程。
- 核心6臂：AE16/Prefix112或AE32/Prefix96 × fresh-tail 0/1/2。
- 全新指18层K/V-ready：在step9（tail1）或step8（tail2）的snapshot之前等待第18层Event，后续同stream自然全新。仍可与最后VLM FFN重叠。
- 不改变旧算法默认配置；在新results adapter中适配，旧runtime/历史结果保持原样。
- 两完整KV槽、唯一AE stream和单层scratch、Pool/Lease/epoch/generation/完整producer_done回收协议不变。尾部Event不得代替Pool的完整producer完成凭证。

## 先门检再计时

1. 同任务13条变化输入的短门检，覆盖tail0/1/2、两槽、reset，串行matched-frontier oracle容差max1e-5、MAE1e-6；尾K轮frontier必须全部18。force_old与非零tail互斥。
2. 从既有同任务真实轨迹预先收集100个固定current/actual-previous观测及seed，六臂和全fresh对照计算同输入；报告归一化与解码后H10/R5 MAE/max及7动作维度。每个新变体另核matched-frontier实现oracle，不把它当作全fresh相等。
3. GPU2测SM16三臂、GPU0测SM32三臂；每进程100组×8请求+100连续请求，预热20，含fresh18内部对照。反向顺序重复一次。资源配置之间是独立进程块，不能称同capture交替。每卡独占监控，不干涉GPU1外部进程。
4. GPU3做固定8case/配置闭环质量小样本，与前轮同suite/task/init/seed/settling。真实闭环禁未来观测lookahead；吞吐结论来自固定输入重放，不能套给闭环。
5. 无profiler计时和nsys分开；只采代表tail配置，验证最后轮前Event等待、真正Kernel尾部和AE/Encoder交集。不用trace受扰动的绝对耗时替代正式结果。

## 指标、选择和停止

主要指标为相对tail0的动作偏差变化、成功/失败和执行步数，以及完整8请求耗时/连续Hz；同时报告AE/VLM跨度、tail边界等待、AE结束−下一Encoder结束、CPU→CPU mean/P50/P95、显存和重捕获。

tail等待属于该轮iteration_ms的一部分，不称纯计算时间。最后几轮全新不能保证消除早轮旧KV的影响。优先判断是否存在质量/速度Pareto改进，不事后放宽实现容差、不以8/8宣称完整质量等价。

遇到实现oracle或尾K轮全18失败立即停止性能队列；外部GPU占用则排除受污染case并在空闲卡重做。无收益/质量退化仍保留有效结果，不无限扩参。新结果、命令、源码快照、CSV/nsys与分析写入`results/idea2/fresh_tail_20261025/`、`reports/idea2_fresh_tail_20261025/`及同目录最终报告。
