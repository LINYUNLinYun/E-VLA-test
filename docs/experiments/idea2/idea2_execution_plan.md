# idea2：单卡 VLM／Action Expert 异步探索执行方案

日期：2026-10-20。依据根目录 `idea2_探索性实验.md` v0.2；不改用户 idea 文档。与 idea1 v3.1 分开执行，先完成本轮。

## 1. 固定边界

- 使用现有 OpenPI π0 LIBERO 完整 checkpoint，18 层 VLM、27 层 vision、完整 action expert；D=10、H=50、每轮执行至多5个动作。
- 首轮 PyTorch eager、batch=1、单进程、单 GPU（优先 GPU0）、两个 CUDA stream，共享只读权重。
- 不引入跳层、verify、draft、降噪步数变化、分辨率变化或新训练模块。
- 每次完整生成固定同一份完整 KV；刷新从空 cache 开始。最多两份完整 snapshot，ready event 完成后才发布，reader 完成后才回收。
- 原始 checkpoint 为 JAX，使用本地仓库官方转换脚本生成独立 PyTorch checkpoint；记录转换、配置、代码版本与数值检查。不得把旧 Triton 跳层路径混入此实验。
- 状态按 π0 原定义进入 suffix；动作转换使用本次请求 reference state。旧视觉与当前 state 的错配是需要测量的实验因素，不隐瞒。

## 2. 执行顺序与门槛

### E0：拆分及缓存正确性

真实 LIBERO 记录观测，四套件至少各一条；固定显式噪声。原始 `sample_actions` 与完整 prefix + 完整10步 AE 配对，比较 velocity、归一化动作及转换后7维动作。

先跑重复原始路径确定可重复性。预先使用相同 eager 算子、FP32动作输出的 `atol=1e-5, rtol=1e-5` 作为拆分门槛；若不能通过则定位原因，不直接放宽。检查所有 KV 层内容与长度未变，全部时间递推与原始 CUDA 张量一致。CPU 固定10次循环只替代 Python 的 CUDA 标量条件判断，不改变递推公式。

协议测试覆盖两槽背压、旧 epoch 完成事件、pending 覆盖、慢 reader、输出所有权和 reset 排空。未通过 E0 不开始策略比较。

### E1：同任务单卡串行／并发微基准

固定旧 cache、噪声、当前 suffix state 与另一条待 prefill 观测。比较 AE→prefill 与两个 stream 并发；相同任务集，均等待全部工作完成。

先20次预热、100次测量；若 warmup 尚未稳定再补充并记录。分别报告动作可用延迟、全部工作完成时间、各模块 CUDA event 时间、P50/P95/P99、显存。另采少量 profiler trace，不混入正常测速。

后续E1b登记：E1 trace已定位大量CPU细粒度提交且实际重叠不足，故按原设计允许的瓶颈触发条件补做CUDA Graph诊断。GPU2单独执行原始eager、静态元数据eager、同wrapper graph三种实现×串行／并发，共六臂；每臂20warmup＋100measure。固定同输入／KV／噪声，真实GPU原始／wrapper数值门槛不变；两graph独立pool、固定地址、输出独立所有权，最多两个生产KV快照。不把该固定输入微基准宣称为动态双槽graph生产策略，也不变更E3科学组。此处各比值仅用GPU2同卡数据；E3仍只用GPU0，不是跨GPU流水线。

### E2：固定版本回放及陈旧误差

使用已有真实观测连续记录；刷新资格间隔1/2/4。固定每请求噪声，先检查串行与并发对相同 KV 版本得到相同动作，再比较当前 KV 和旧 KV 的差异。

分别报告前1–5步及完整50步的平移、旋转、夹爪分量，不能将归一化动作 L2 称为毫米。记录 request 差、实际时间年龄与 source observation ID；离线回放的墙钟年龄不能冒充原始相机采集年龄。

### E3：小规模闭环

先四套件各 task0、init0 配对 smoke test；通过后使用 task0、5，init0、1，共16个场景。

对照 `fresh_serial`、`serial_reuse`、`async_reuse`；复用间隔先1，再2/4。主矩阵 fresh16 + 两种reuse×3间隔×16=112 episodes，不追加500/suite全量评测。每个条件固定同任务初始状态和请求噪声规则；失败或技术异常分开统计，不把异常记为策略失败。

记录真实末端XYZ、实际执行动作、所选 KV 版本、条件年龄、请求耗时、成功率和超时。代表性视频固定各套件 task0/init0；视频全部帧解码检查。轨迹图 baseline 蓝色、变体橙色，使用半透明线及不同标记避免重叠。

LIBERO 在推理时暂停：本轮结论限于推理性能与陈旧条件下策略表现，不能声称获得真实动态环境的反应速度提升。串行复用不在 AE 前等待它本轮不会消费的新 KV；异步记录动作就绪和后台 drain 的分离耗时。

## 3. 产物与完成标准

- 实现和测试：`preparations/idea2/`，不修改模型源码。
- 原始数据：`results/idea2/`，与 idea1 数据隔离。
- 本目录保存环境身份、运行命令、进度、最终报告和图索引。
- E0/E1/E2及上述小规模E3须有有效原始数据、完成审计和可重现命令；如硬件竞争抵消收益，也如实报告。
- 完成本轮后再进入 idea1 v3.1：XR0/CogACT 跳层、评测系统优化、π vision encoder 跳层，另立方案与目录。
