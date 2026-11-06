# VLM领先分层KV兜底＋跨帧Encoder覆盖实验

用户于2026-10-25授权执行。完整π0.5 LIBERO、BF16、B1/H10/D10/R5，权重、时间递推、噪声构造和动作数量不变。不同层/不同AE轮可选择新旧KV是本次明确授权的近似算法条件，独立于此前exact same-frame与complete-old-KV路线。

目标是让本帧AE先与本帧VLM重叠，剩余AE与已经到达的下一帧Encoder重叠。最理想是AE和下一Encoder同时完成，但不得以等待/饿死Encoder、提前消费未来帧或放宽质量要求制造该对齐。

## 实现语义

- 同一Host管理两个完整KV Slot。旧槽提供完整只读KV，新槽VLM逐层生成；原K/V都完成后release发布ready frontier。
- VLM至少完成0/1/3/6层后放行AE。每轮AE在GPU acquire读取一次frontier并冻结：已完成层用新KV，其余层用旧KV。本轮内不切换。
- 复用原Attention、投影、归一化及FP32 Euler。只增加发布/快照与按位选层拷贝，全部AE共用一层大小scratch，不保存第三份完整KV。
- 两个Slot同时有Lease，AE尾部用两个不同Event分别释放。每generation先取得旧完整Lease，再提交新Prefix并获取新层Lease。事件不可跨活generation冒用。
- 下一帧Encoder只写旧槽中独立的Vision输入/embedding/metadata workspace；AE读取旧槽的K/V，不读取旧槽metadata。显式KV-only契约和ahead ticket绑定未来epoch/slot/generation；旧KV读者结束且新KV发布后才可复用旧槽写下一帧KV。
- 下一Encoder在本帧VLM完成且AE第一轮完成后启动，避免三路同时争抢。VLM完结到Encoder开始的空洞以及Encoder结束与AE结束差都报告。
- 第一次没有旧KV，先完整bootstrap。任务/episode reset排空本调度拥有的生产者、读者与ahead票据；异常必须显式reset，禁止继续复用半提交状态。

## 有界实验量与判定

1. CPU协议及bitwise selector GPU门禁；完整模型覆盖0/1/3/6/18层、两槽轮换、同地址内容更新、reset与ahead。按实际记录的每轮frontier串行重算，max≤1e-5/MAE≤1e-6。相对fresh的动作误差单独报告，不混为实现误差。
2. 初筛：相同Graph engine，0/1/3/6与完整18层等待对照，各host-only/Encoder两种提交，每条件20组×8请求，20预热；每条件独立bootstrap及显式噪声，按组交替顺序。
3. 正式：四个lead的Encoder路线，加18层host-only及Encoder对照，每条件100组×8请求、100连续请求；独立反向复测。初筛无收益仍保留四lead结果，不扩大任意层数搜索。性能与nsys分开。
4. 合成相机到达33/62 ms检查可接纳输入、丢帧、capture→action P95及新旧KV来源年龄；在有限时间内确认代表候选，不追加SLO控制器搜索。饱和bank重放不冒充真实未来观测。
5. 代表性nsys：lead0/3/6＋完整18层Encoder对照，各8请求，实际Kernel交集、模块busy/envelope、AE/Encoder结束差。不使用NVTX重叠替代真实Kernel证据。
6. 有限闭环：四套件task0/5 init0，seed7，每lead0/1/3/6与fresh18各8例，共40例。复用原环境/settling/噪声/动作执行。闭环拿不到下一观测，因此不提前使用next Encoder；只检验混合KV质量，不能声称验证了实时相机流水收益。没有全量视频，保留输入、动作、EEF、case成功/步数和每轮frontier。任何任务失败作为有效结果保留。

## 交付与资源

新实验目录 `results/idea2/progressive_kv_20261025/`，报告与CSV分别放docs和reports下。保存运行时source hash、真实命令、失败/排除记录、各模式样本量和边界。

GPU1 UUID `GPU-599477fb-0507-40dc-14df-d491f7a7b202` 已在启动时检查C/G空闲并登记租约；不将历史GPU编号作为永久授权。其他GPU如用于闭环，须再次确认归属、登记并分离Host CPU affinity；不跨GPU直接排名延迟。

## 执行中的必要修正与有界补充

- 早期gate_v2/screen/formal误用了包含8种instruction的原始100条bank，已停止性能队列并标记`excluded_cross_task_bank`，保留跨任务诊断而不排名。现使用`same_task_bank.pkl`：首instruction的13条观测，原索引0/3/4/7/8/11/12/15/16/19/20/23/24；来源与SHA见同名JSON。`gate_same_task`已重新通过。
- 初筛测CPU normalized输出，正式计时results-only包装增加原output_transform解码，记录CPU动作可用时间。冻结runtime未改。基础bench的`admission_to_action_ms`只能称loop-entry→return，正式后处理不把它当真实admission。
- 合成到达使用独立`slo_bench.py`，真正选择最新已到达观测、只允许一个已接纳lookahead，保存capture/admission/output绝对时钟与received/admitted/dropped/buffered恒等式。不得使用基础bench的历史source+1模式作为latest-wins SLO。
- 在全部四lead正常双流主矩阵之外，只增加两个资源配额16/112、32/96，固定lead3，检验移除KV强依赖后更多AE轮可并行是否改变SM隔离结论；再测下一Encoder在AE第4或第7轮完成后放行（仍须当前VLM结束）。不增加Host睡眠，不为对齐终点而忽略整轨迹耗时/观测年龄。该补充属于筛选，仅当整路径收益明显才补正式确认/必要质量验证，不扩展任意SM/切点搜索。
