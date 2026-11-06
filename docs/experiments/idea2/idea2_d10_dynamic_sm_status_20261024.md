# idea2 D10 / 动态 SM 探索：执行记录

2026-10-24。本轮已完成，GPU0/1已释放；最终结论以[完整报告](idea2_d10_dynamic_sm_report_20261024.md)及[完成审计](../../../results/idea2/d10_dynamic_sm_20261024/requirements_audit.md)为准。下面保留按时间记录的历史状态，“下一步/尚未完成”指当时情况，不代表当前仍有待跑实验。目标和边界见[实验方案](idea2_d10_dynamic_sm_plan_20261024.md)。

## 设备交接

- 当前 GPU2/3 运行另一会话的 idea1 v3.3 Xiaomi；GPU0/1 原计划运行其 π0。已通过跨会话消息请求安全交出一张，未自行终止其他任务。
- 对方在完整 episode 边界停止 GPU1 的 dispatch/server/rollout，保留 8 条有效结果，并将 GPU1 登记为 idea2 租约。设备为 `GPU-599477fb-0507-40dc-14df-d491f7a7b202`。[交接收据](../../../results/idea1_v3_3/gpu1_release_receipt.json)。
- 18:16:31 的 `nvidia-smi -i 1 -q -d MEMORY,PIDS` 仍显示 Type G 的图形进程、宿主 PID 1978752、537 MiB。计算进程列表不足以证明设备独占；已将证据发给另一会话协调渲染归属。在该干扰清除前不启动正式 GPU 计时。
- 对方确认本机 EGL index 0/1/2/3 对应 CUDA GPU3/2/1/0，原 GPU2 的 rollout 将图形工作发到了 GPU1。已在安全边界迁移；18:19 后本方复核 GPU1 约 2–5 MiB、0% 利用率、无计算或图形进程，才启动门禁。新增 `gpu_guard.py` 检查 XML 中的 G/C+G 类型，禁止仅依赖 compute-apps 列表判断独占。

## 实现与验证进度

- Graph 原先只接受 D2，已开始支持 D10；按原 `generate()` 循环在 AE1 后分成两个 Graph，第二段为 AE2–10。
- 已增加预捕获的尾段 stream、GPU Event 交接和全 Prefix 完成门；最终 reader Event 位于真正的尾段 stream。只在提交异常路径排空相关 stream，正常执行没有逐轮 CPU 同步。
- Graph 仍只有原两个 KV Slot。冷热阶段分离：时间节点在冷预热核验，捕获阶段计数 NFE，正式图不加入诊断克隆算子。
- [76 项相关 CPU 测试](../../../results/idea2/d10_dynamic_sm_20261024/cpu_checks.log)通过，包括阶段交接、D10 日程/标签、双槽和图形进程隔离。
- [D2 回归](../../../results/idea2/d10_dynamic_sm_20261024/d2_regression/result.json)通过：13 个交替观测、双槽 smoke、原地输入和 reset 门禁；原始及 decoded 动作误差均为 0。小样本只作回归，不作性能证据。
- [D10 整段初筛](../../../results/idea2/d10_dynamic_sm_20261024/d10_whole_screen/result.json)通过：十轮原生 FP32 时间节点和捕获 NFE、两槽数值与生命周期均通过；已测原始/decoded 误差为 0。20 对 AB/BA 中，串行/逐层并发模型跨度 P50 为 66.735/65.888 ms，CPU→CPU 为 98.297/96.104 ms。**仅为未复测初筛**；AE Event 跨度包含等待，不能当成 AE 独立计算时间。
- [等分图全卡初筛](../../../results/idea2/d10_dynamic_sm_20261024/d10_split_full_screen/result.json)通过，20 对中串行/逐层模型跨度 P50 66.936/65.915 ms，原始动作误差 0。
- [16/112→全卡尾段门禁](../../../results/idea2/d10_dynamic_sm_20261024/d10_dynamic16_gate/result.json)通过；两槽、时间节点和生命周期检查通过，已测原始/decoded 误差为 0。仅 2 对加连续短序列，不能作为速度排名。
- [SM 独立曲线](../../../results/idea2/d10_dynamic_sm_20261024/sm_curves/result.json)五档均通过数值和逐档清理；[汇总 CSV](../../../results/idea2/d10_dynamic_sm_20261024/sm_curve_summary.csv)。同进程同输入、每阶段 20 预热/30 测量，以下是 Event 均值，不是端到端延迟：

| AE / Prefix SM | Vision ms | VLM ms | AE1 ms | AE10 ms |
|---|---:|---:|---:|---:|
| 全卡 / 全卡 | 10.111 | 31.608 | 2.569 | 25.232 |
| 16 / 112 | 10.477 | 37.538 | 4.377 | 39.963 |
| 32 / 96 | 11.836 | 41.793 | 3.053 | 28.877 |
| 64 / 64 | 15.015 | 55.695 | 2.728 | 25.549 |
| 全卡复测 | 10.105 | 31.748 | 2.561 | 25.242 |

这支持“固定小 AE 配额会拖慢后续九轮，限制 Prefix 也有明显代价”的诊断，不能直接预测并发净收益。由于 112 SM 的 VLM 已比全卡慢，下一步仅追加 8/120 一个小配额曲线点，再冻结至多两个动态候选，不扩大任意逐层搜索。

- 16/112→全卡的 [nsys](../../../nsys/idea2_experiment/d10_dynamic16_profile_20261024_v1.nsys-rep) 已完成独立采集；[资源审计](../../../results/idea2/d10_dynamic_sm_20261024/d10_dynamic16_trace_audit.json)通过：7072 个 VLM Kernel 位于 112 SM、8432 个 AE1 Kernel 位于 16 SM、75888 个后续 AE Kernel 位于不限额主 Context；8 帧的阶段依赖顺序均通过。实际 AE×VLM Kernel 交集共 49.486 ms（每帧均值约 6.186 ms），这是 profiler 诊断，不能当稳态收益。
- [追加 8/120 曲线](../../../results/idea2/d10_dynamic_sm_20261024/sm_curves8/result.json)通过：全卡/分区 VLM 为 31.516/35.963 ms，AE1 为 2.696/6.558 ms，AE10 为 25.190/64.644 ms。冻结候选为 8/120 和 16/112，不扩大配额搜索。
- 用户短暂停下审阅后已重新明确授权继续，19:25 确认 GPU1 空闲并通知另一会话继续保留租约。逐轮诊断进行中；正式配对复测、跨帧/旧 KV 对照与 SLO 尚未完成。

## 后续顺序

1. 交卡独占确认；完成 D2 回归和 D10 整段/分段数值门禁。
2. D10 普通双流、逐层和 Encoder 基线；独立模块 SM 敏感度。
3. 等分图/等 Event 的 full→full、partition→partition、partition→full 对照。
4. 按方案门槛决定是否组合跨帧；完成正式配对、独立复测、nsys 和系统指标后再形成结论。

本记录只表示当前事实，不是实验完成报告。

## 恢复执行后的补充证据

- [逐轮诊断](../../../results/idea2/d10_dynamic_sm_20261024/iteration_audit/result.json)：13 个真实观测、52 请求、520 个迭代检查，时间/velocity/Euler/最终动作均逐位一致；外部输入同地址复用、两槽和 reset 通过。诊断捕获含额外 time clone，禁止其时间参与性能排名。
- 16 SM 的 20 对短测：固定分区并发模型跨度 P50 85.087 ms，后九轮恢复主 Context 后 72.026 ms；CPU→CPU 分别 114.964/103.321 ms。资源回收有效，但当前全卡初筛约 65.9 ms，尚未获得净收益。跨进程配置比较尚无反向复测，不能据此声称正式加速比。
- 下一步先测普通 D10 Encoder 跨帧，再用不同槽的 AE10 × Vision 做 full/64/32/full 固定输入配对筛选。该微基准不计 VLM、Host 准备和请求延迟；只有配对显示机会才考虑拆 Vision/VLM 的资源归属。其余 SM/切点不扩展。

- G0/G1 首次正式运行（`d10_whole_formal`、`d10_split_full_formal`）在冷加载时发生了运行时源码修改。虽然数值检查和进程均通过，无法证明快照与实际 import 完全对应，状态已标为 `excluded_source_drift`，禁止进入排名；将以 `*_formal_stable` 补跑。其后 reverse repeats 的运行时代码快照已与冻结源码核对一致。新增队列前后 hash 检查，运行中源码变更将自动使结果失效。

## 最终收口

- 两轮单帧100对、两轮当前KV跨帧100×16帧×4臂、两轮旧KV三模式对照、两轮同提前返回API公平控制全部完成；正式数值误差为0，稳态零重捕获。
- 33/62/71 ms到达条件各反向重复，共2400请求；5%预算没有事后放宽。33 ms不通过，62 ms的F2/F4通过，71 ms输入限速。
- 六条当前KV、三条旧KV、一条提前返回串行控制trace均已分析实际Kernel和依赖。动态回收机制有效，但未胜普通全卡；O1纯重叠连续增量约8%，不是原接口比较的1.59×。
- 旧KV16例有效闭环，O0/O1均4/8成功，428对推理动作一致；新增GPU0新KV8例全部成功。额外GPU只用于质量，启动晚于GPU1正式计时结束，不跨卡比较延迟。
- SLO62反向初次启动遇空闲利用率残留，0条记录，保留失败并以`_resumed`完成；没有重复已有效组或隐去失败。
- 所有自有GPU服务器/子进程已退出，GPU1及GPU0依次归还协调会话。CSV、命令、trace、数值/生命周期与最终审计见完整报告链接。
