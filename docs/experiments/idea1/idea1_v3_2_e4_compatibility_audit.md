# v3.2 E4：跨环境资格检查（尚非成功率实验）

检查日期：2026-10-22。对应[本轮方案 §5](idea1_v3_2_experiment_plan.md)。这份检查不代表 E4 已完成。

**05:00 UTC进展：**后续CPU预检已解除新指令embedding这一技术限制；主代理依据原跨环境请求补齐[零样本执行口径](idea1_v3_2_e4_zero_shot_protocol.md)，进入适配准备。下文“等待确认”为早期判断，不能当作新的权限禁令；没有伪称收到用户确认，也没有因此忽略动作映射混杂。

## 已核实的事实

1. 本地 `shared/checkpoints/realtime-vla-flash/` 有四个 `draft_libero_*.pt`。在线读取 [官方 HF 文件 API](https://huggingface.co/api/models/Dexmal/RealtimeVLA-Flash/tree/main?recursive=true&expand=false) 的清单也仅列出四个 LIBERO draft、README 和 gitattributes；检查时未列出 CALVIN／SimplerEnv／PushT head。
2. 读取本地 `draft_libero_goal.pt` 的 CPU tensor 计数为 **110,288,903 参数，约 110.3M**，不是主线括号中猜测的 40M。它是 `vlm1_query_decoder`，输入宽 2048，单 Gemma block，FFN 宽 16384，输出 `50×7`；另依赖 Full 的视觉编码／语言 embedding，不是完整流水线只有这 110M。
3. 本地 meta 记录 `cache_run_dir` 指向 `star_cache_pi0_libero_*`，`target_source=teacher_zero_noise`，`use_last_actions=false`，`use_seed_actions=false`。这是训练来源线索，不等于已证明仅能适应 LIBERO。
4. 原生 V14_A03 是 `pi0_libero` 18 层 VLM 中跳 `[3,4,9,11]`，AE 保留；它并不是未经 LIBERO 适配的通用 base。单看参数更多不能推出其跨环境一定胜过 draft。

## 环境和接口的当前差异

| 环境 | 当前可用资产 | 原生接口 | 与当前权重相比尚需确认 |
| --- | --- | --- | --- |
| LIBERO | Full、V14、四个 suite draft、正式初态及评测 | 两 RGB；状态为 XYZ＋轴角＋双指 qpos，共8维；7维归一化控制命令；20Hz | 本轮只复用历史，不重复做已有成功率 |
| CALVIN | 本地可运行 checkout／`.venv`；debug validation 仅8条语言段、6种任务，不是正式1000条长序列评测 | static `200×200`、gripper `84×84`；15维 robot_obs；7维相对 XYZ／Euler／夹爪；官方说明30Hz | 状态坐标、Euler→轴角、双指状态构造、相机方向、目标位姿累加规则、位移／旋转尺度、夹爪符号、重规划节奏和新指令 embedding |
| SimplerEnv | 本地 checkout／`.venv`、Google Robot／WidowX 任务实现 | 机器人依任务不同；单主相机为常见接口；7维 delta XYZ／轴角／夹爪，具体控制器和夹爪含义依机器人 | 机器人和标定、状态来源、缺失腕相机处理、单位／频率、控制器映射、对应权重／归一化 |
| PushT | 本地 diffusion_policy 环境及其独立策略 | 二维平面，动作是 `[0,512]` 像素坐标绝对目标，10Hz；不是机械臂7维动作 | 没有自然对应的姿态／夹爪／8维状态；直接取七维动作前两维不构成已验证接口 |

CALVIN debug 中实际读到的任务计数：`lift_blue_block_slider:3`，其余 `lift_red_block_table / place_in_slider / push_pink_block_right / turn_off_lightbulb / turn_on_lightbulb` 各1。它可做接口／少量迁移验证，不可写成正式 CALVIN benchmark 分数。

本地 `calvin/calvin_env/calvin_env/robot/robot.py` 默认 `max_rel_pos=.02`、`max_rel_orn=.05`；debug 配置为 `use_target_pose=true`，相对动作累加到控制器目标而非始终直接加到观测位置。与 LIBERO OSC 的 `.05m/.5rad` 动作尺度不同；机械地转发数值会同时改变物理行为。

## 额外的实现阻碍

当前历史 Triton 适配器只接受冻结 manifest 中的 LIBERO 指令 embedding：`preparations/idea1_v2_2/pi0_adapter.py` 对未知 instruction 显式拒绝。CALVIN 新语言不能用“最近似的 LIBERO 指令”替代，否则改变了任务。需要从同一 teacher 的 tokenizer／embedding 构造新指令并对齐 Full／V14／draft，而不是修改已有 manifest 冒充原生支持。

以上是明确的适配工作，不是权限错误，也不是物理上不可迁移；当前缺少经确认的比较定义。已向用户询问：接受清晰标注的 CALVIN 零样本小实验，还是等待匹配 checkpoint。其余三个独立实验正常推进。

## 可解释结论的边界

- 若用户接受零样本试跑，必须加 **Full 正对照**。Full／V14／draft 同任务同初态、同映射、同执行节奏，记录完整动作与真实末端位置，不能先挑 draft 差的片段。
- Full 也失败时，只能说明该权重与部署条件尚不适用，不能证明“小模型能力不足”。
- 不同分布上的失败即便只有 draft 明显，也只能支持该迁移设置下的性能差异；模型容量、训练覆盖和蒸馏目标仍混杂。
- 不擅自新增训练或下载不匹配的大模型来做预设结论；没有真实策略 rollout 的环境资格检查不纳入成功率分母。

接口参考：[CALVIN 官方说明](https://github.com/mees/calvin#-sensory-observations)、[SimplerEnv 官方说明](https://github.com/simpler-env/SimplerEnv#examples)、[FLASH 官方训练／部署入口](https://github.com/dexmal/realtime-vla-flash)。具体数字以本地已检视源码和 checkpoint 为准。
