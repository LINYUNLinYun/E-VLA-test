# AE Skip-L2 / Skip-L8 历史 pilot 的协议追溯

结论：这八个 run 可以作为 **H50 / R12 / D10、VLM Full18、AE 17/18 单层跳过**的历史探索线索。H、D 和实际报告的层索引已有原始请求证据，不依赖当前默认值。但闭环 run 缺少 checkpoint、norm 和执行源码的历史内容 hash 绑定，不能将当前源码 hash 或离线 probe 的 checkpoint hash 冒充闭环运行时证明。它们与当前 H50/R5 的实验不能直接合并。

核验范围有界：八个 run 各取首个 episode 和最后一个 episode，再取文件内首、尾请求，共 **16 个 infer 文件、32 个请求**；另读各 run 的 manifest、episode_log 和 server 日志。没有重新计算成功率、时延或全量解码 800 条 rollout 的动作数组。32 个请求的下述契约核验均通过，详见 [audit.json](audit.json)；脚本为 [audit.py](audit.py)。对未抽查请求不作全量一致声明。

| 项目 | 直接历史记录 | 证据边界 |
| --- | --- | --- |
| H=50，输出维度7 | 抽样 infer 的 `chunk_actions` 全为 50×7；各 server 日志所有 `actions shape=` 行均报告 `(50, 7)` | 不是从当前 Pi0 配置默认值推断 |
| R=12 | 各 manifest `replan_steps=12`；32 个请求 `accepted_prefix_len=chunk_exec_len=12` | 最后一个执行 chunk 可以因任务终止只执行部分动作；本审计不重做轨迹计数 |
| D=10 | `server_experiment.denoise_steps=10`，times=1.0,0.9,…,0.1，dt=−0.1；server 启动日志同样记录 | 请求元数据不是各步执行的独立 kernel trace |
| VLM 全18层 | full block indices=1…18，KV shell=[]；server 启动日志一致 | 样本内没有 VLM 裁剪 |
| AE 层2/8 | pattern=`ae_skip_l2`/`ae_skip_l8`；保留 indices 正好为1…18去掉2或8；server 启动日志一致 | 是 1-based 编号；其底层语义还需结合实现和 parity 记录解释 |
| 噪声 | 全局 seed12345，mode=`episode_infer`，rule=`SeedSequence(global_noise_seed, episode_idx, replan_idx)`，trace episode/infer 身份和 SHA 均存在 | 用 CPU NumPy 重建 50×32 FP32 Gaussian 后，32个报告 SHA 全匹配。闭环未保存实际 GPU 噪声 tensor；当前源码的 SHA 字段本身也是独立重建值，不是实际输入字节的抓取 |
| 环境协议 | manifest 记 task0…9、init0…9、wait10、seed7、resize224 | 与当前不同协议不能只因 task 名相同就合并 |
| 路由和刷新 | 32 个请求全部 route=`full`、did_prefill=1、is_full_pipeline_round=1、used_full_fallback=1，draft/verify ms=0 | 支持每请求执行 full prefix，不支持跨请求近似 prefix 复用；不是对所有请求逐条验证 |

具体样例：`exp9_ae_layer_pilot_SkipAE_L2_libero_10_i00_10_a01` 的 server.log 第1–4行记录物理18+18层、AE保留层、D10日程和实际 norm 加载路径；第42–46行附近开始记录 `(50,7)`、full/prefill/fallback 和 `ae_skip_l2`。精确源文件路径、行号、内容及当前文件 SHA 在 audit.json 每个 run 的 `server_log` 中，32条请求的文件 SHA、抽样行号在 `infer_files` 中。

底层 skip 的解释由当前实现和历史辅助证据共同支持，强度分开列出：

- 当前 `serve_action_layer_screen.py:13–24` 将 `layer-1` 从18层索引中排除，并同时注册 runtime 元数据和 `pi0_infer._ACTION_LAYER_PATTERNS`。所以 L2/L8 指 AE decoder 第2/第8块，而不是 VLM 第2/第8块。
- 当前 `pi0_infer.py:1379–1477` 在外层 flow loop 内使用相同 `layer_indices`；被删块在 **每一个 denoise step** 都不执行。被删的是整块 attention QKV、attention 输出投影与 FFN，隐藏状态沿 identity 穿过；不是只删 attention 或只删 MLP。
- 普通保留块仍有模型自身的 attention/FFN residual addition。当前这条 decoder 实现没有额外 residual compensation、历史 residual 注入或随步更换 mask。此点属于当前实现解释；由于没有闭环源码快照，无法以字节级溯源证明当时的全部实现细节。
- 当前 `spec_serve_policy.py:525–579` 的 force-full 分支重新运行 full pipeline；`triton_pi0_runtime.py:2005–2059` 每次依次运行 vision、prefill、decoder。一个请求内的 VLM prefix K/V 供其 D10 AE 共同使用，这是正常同观测复用。它不等于跨请求/跨观测近似复用。请求末尾仍可能创建 cache snapshot，但 force-full 条件下没有进入 draft/verify 消费分支。
- 历史离线 probe 的源码 `probe_action_expert_layers.py:61–94` 明确逐 D10 step 跳过同一完整 block，并检查各保留层访问10次、被跳层0次。历史 `registered_pattern_parity.json` 保存了 L2/L8 × initial/contact/goal_alignment 三个观测共6项 bitwise equal、max_abs_error=0。这个有限 parity 记录支持注册后的图路径与离线单层 skip 对齐，不能提升为所有闭环请求的实现等价证明。
- 旧 `pattern_parity_failure.json` 仍然存在：此前 L2 的 graph 与 same-runtime eager 相同，但与旧保存 probe 的差约0.00963，连 fresh full 与 saved full 也有差约0.00513。后续 screen_followup.log 记录先 warmup prefix 编译，再启动这八个 run；成功 parity 记录不能抹去此前失败，也不能单凭这两份文件断定编译是唯一原因。

checkpoint / norm / 执行来源仍有明确缺口：

- 当前对应启动器 `screen_action_expert_candidates.py:129` 传 `--config pi0_libero --base-triton-path .../converted/base --backend triton --force-full-each-round --denoise-steps 10 --ablation-noise-seed 12345 --max-exec-steps 12`。这是当前命令构造代码，**没有找到八个 run 的完整历史 argv receipt**，不可升级成当时实际 argv 的直接证明。
- 原始八个 server 日志都记录从 `shared/checkpoints/realtime-vla-flash/converted/base/assets/physical-intelligence/libero` 加载 norm。能证明所报告的历史路径，不能证明其历史内容。audit.json 记录了现在 `norm_stats.json` 的 SHA，并明确 `historical_run_hash_binding=false`。
- 历史离线 `probe/summary.json` 保存 base weight SHA `c423a508742f8461fa17d8fb42951837b4419804842ed27a4890a9903042f891`，且声明 checkpoint_unchanged=true。这仅绑定离线 probe；八个闭环 run 没有独立内容 hash 将其实际模型绑定到这个 digest。本次没有加载或重新 hash 约6.5GB权重。
- `runner_events.jsonl` 有每个精确 run 的 start/complete、suite、layer、GPU 和时间，但不含完整 argv/source hash。当前 Git 中两个 adapter/启动脚本未跟踪，`pi0_infer.py` 有工作区修改，相关 Git 历史仅见初始提交。audit.json 保存的11个源码 SHA 是本次观察值，不能恢复2026-10-05的执行源码版本。

因此它们适合用于寻找下一轮 AE 路径候选和检查历史任务差异，不能替代新协议下独立复验，也不能单独证明 task-dependent routing 已有稳定收益。
