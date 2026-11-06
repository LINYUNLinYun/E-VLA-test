# lead3 混合 KV + 十轮 AE + 下一帧 Encoder：因果闭环小试

状态更新：首场景四臂门禁全部完成，224次在线请求；真实下一观测全部晚于AE完成，未形成目标窗口，且四臂均达到280步上限失败。按机制门禁停止其余28个计划回合，不扩样、不重试有效失败。另完成预准备输入的nsys Kernel机制见证，7/7个带下一Encoder的请求确有两段交集。见[报告](idea2_online_encoder_report_20261025.md)。

用户指定路线：VLM_t 前3层 KV-ready 后启动 AE_t；AE 共10轮，每轮固定当时新 KV frontier，其余层用旧 KV；VLM_t 完成后，Encoder_{t+1} 与 AE_t 剩余轮次重叠。保留原两槽/Lease/epoch/generation/Event，模型/权重/精度/D10/H10/R5不改。

此前固定观测重放已测过该完整组合：SM16/Prefix112 lead3 的8请求474.58/474.37ms，对完整新KV+Encoder的533.49/533.71ms约1.12×，尾部约3.8ms。此前200场景成功率实验则关闭next_encoder，两种证据不能拼成完整组合无损。

## 真实观测的因果接入

原同步LIBERO在AE返回后才执行动作，下一策略输入依赖该动作，不可能在原客户端里仅打开next_encoder。新实验独立命名为one-chunk-lag，不冒充原同步协议：

1. bootstrap以obs0/seed0生成action0，先不执行。
2. predict1以obs0/seed1提交GPU；客户端收到提交完成的advance消息后，实际执行已批准action0前5步，获得真实obs5，送给当前请求的lookahead callback。
3. 如果AE仍在运行，obs5的准备与Encoder可与它重叠；如果到达太晚，如实记录零窗口。下次请求必须消费这个同一Prepared及输入identity，不重复编码或偷换较新观测。
4. predict2使用obs5/seed2，与已返回action1的5步执行并行，以此类推。

消费动作仍每块最多5步，不使用H10后5步补队列、不每tick替换动作。该协议显式增加一块观测滞后（5个控制步，名义250ms）；这是用户此前允许探索的延迟/质量取舍，不是严格同步算法等价。20Hz只表示原仿真控制步长，实验不额外声称真实墙钟20Hz。

四臂使用完全相同新在线协议：fresh18_no_encoder、fresh18_encoder、sm16_no_encoder、sm16_encoder。先四suite各task0/5、init0，共8场景×4=32回合的有界小试，先跑一个场景接入门禁再补齐；真实失败不重试。旧200同步结果独立保留，不合并到新在线成功率。

## 记录与判断

- 服务端唯一Host管理Pool，原runtime不改；同一RPC双向交换advance→真实observation→actions，客户端只推进自己的环境，不读未来轨迹。
- 保留输入/动作SHA、原model_seed、逐步EEF、动作所属请求、实际执行步数、input/capture/application step和终态丢弃预测；episode独立创建/关闭/reset。
- AE提交、下一观测收到时及准备后AE Event.query()、Host等待/准备、CUDA Event模块跨度、AE尾巴与下一Encoder结束差；Event在episode结束后汇总，不为了计时阻塞动作返回。
- 比较四臂质量/完成步数/episode wall，单次延迟包含观测交换时不得叫纯模型速度。Encoder已完成但输出等待环境也不算GPU隐藏收益。
- 若真实下一观测比AE晚到，不改捕获时刻或刻意延长AE以制造重叠。再决定该闭环粒度是否值得扩样，不能直接沿用重放1.12×宣称在线收益。

仅新建 `results/idea2/online_encoder_20261025/`，GPU2重新经空闲/租约检查后使用；GPU0/1/3不占用。性能重放已有充分记录，不重复原100对测试；本轮优先补真正因果接入的缺口。
