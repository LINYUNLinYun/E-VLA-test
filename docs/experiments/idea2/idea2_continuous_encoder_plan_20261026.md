# 完整 lead3/D10/下一 Encoder 流水线：连续闭环成功率

目标是直接测完整组合的闭环质量，不把已有同步200例成功率与离线1.12×加速拼接成一个结论。用户指定AE10轮，第1轮等前三层KV-ready，后续逐轮新KV就绪则使用、其余用旧KV；下一真实帧Encoder与AE尾部重叠。

## 固定矩阵

三臂各100场景：fresh18_no_encoder、sm16_no_encoder、sm16_encoder，共300回合。四LIBERO suite各task0–9/init0、1，偶数task0/2/4/6/8再加init2，各suite25。envseed7、settling10、原OSC20Hz、原任务上限、完整π0.5 BF16主体、D10/H10/R5、原权重不变。所有任务失败保留，不重试到成功。

先用object/task0/init0和spatial/task5/init0做6条接入门禁，符合协议的有效结果直接计入100网格。门禁看因果输入、R5/seed/mailbox、Event生命周期及实际可用重叠窗口，不按成功率筛场景。技术中断仅有界恢复一次、新目录留证。旧同步和旧250ms缓冲结果不并入本轮统计。

## 连续控制及噪声规则

主线程独占环境/EGL，按墙钟20Hz持续执行已选动作块前5步；后台网络线程持续接受主线程已经生成的较新观测。只有块边界允许选择“下一执行块”的最新完成预测，mailbox容量1；过期/被替换候选不执行。没结果则等待并记录控制gap，不造noop、补H10尾部或中途替换5步。观测/推理/执行解耦，不再固定滞后一整块。

每个实际消费块i沿用原model_seed(suite,task,init,i)。该块未消费候选可重复使用同一显式noise；prediction request_id与target_chunk_id分开，防止不同速度产生更多请求后消费不同随机种子。采样步数和schedule不变。

server唯一Host修改Pool；沿用原两槽/Lease/epoch/generation/Event，不新建第三完整KV。lookahead只接收真实step产生的新frame；下一请求必须消费同一已接纳Prepared对象。无新帧最多等75ms后返回not_yet，不reset策略；下次从实际较新frame继续。未消费的terminal ahead在drain后reset清理并记录。

## 对照与记录

三臂执行同一控制/接纳规则，只改fresh vs mixedKV及下一Encoder开关。这是新在线协议，和原同步benchmark不同；full fresh同协议是主要质量参考，原同步结果仅历史对照。

三臂都在lookahead执行相同backend.prepare（CPU变换/上传准备）；Encoder开关只控制下一Vision Graph是否提前执行，避免把CPU准备提前的收益混进GPU重叠消融。最初v2接入6条曾对no_encoder只保存raw，保留为初步接入记录，不并入修正后的主300格；v3六门禁按统一准备协议重做并计入。排除依据是事先检查发现的协议差异，不是成功/失败；原始有效失败不覆写。

记录所有候选的输入hash/step/时间、target/seed、动作/normalized输出、选中/过期/替换/终止状态；实际消费输入NPZ、逐步EEF和实际动作；每块capture→response→application年龄、deadline/gap、推理/控制频率、丢弃率、mailbox峰值。GPU Event在episode结束统一读取，正式质量评测不打开nsys。正的Event包围区间只表示可用窗口，真实Kernel机制已有同runtime的独立nsys证据；不把NVTX/Event交集当Kernel busy交集。

模型加载/捕获/首次bootstrap单列，不把连续GPU吞吐当任务完成提速。每task三臂同卡，顺序轮换、另一GPU反序；最多4task/模型服务，避免Green Context无限积累。GPU2/3在当前空闲和租约核验后使用，不干涉其他卡。

统计预先固定见[prospective_statistics.md](../../../results/idea2/continuous_encoder_20261026/prospective_statistics.md)：配对成功数、Wilson、40task不等样本量ratio-weighted bootstrap10000次、精确McNemar；主要对比只有sm16_encoder对fresh，开关消融另报。p>0.05不称无损，没有事后非劣门槛；100例不足以保证1pp以内无退化。

最终保存逐例CSV、300cell网格/进程退出/源码身份、质量与观测年龄/控制gap/真实窗口汇总、失败案例；报告路径 `docs/experiments/idea2/idea2_continuous_encoder_report_20261026.md`。上线默认不改。
