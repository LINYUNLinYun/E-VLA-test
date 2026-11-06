# idea1 v3 incremental results

Valid episodes included: 680. Technical invalid attempts are retained in worker journals.
Calibration, P0 and scientific init6–10 rows are separate in the stage column. Failed tasks remain in denominators.
Single-request mixed-policy latency and end-to-end throughput are separate columns. Startup is configuration_startup.csv; lazy first-use is included in episode policy wall.
E2 initial adapter reports native total latency only; VLM/AE partition is unavailable. CPU/GPU/disk-wait activity is not sampled.
E2 的累计 counts.full 表示当前 mask 下的原生策略调用次数，不表示未剪 Full 或 Full 回退。E2 表中单列 native_policy_calls 与 mask，full_fallback_calls / Full 回退率记为 N/A。
E2 性能边界：Full 的 AE 使用原生 nn.scan；非空 mask 的 AE 虽保留完整18层，但由 Python 展开后交给 JAX 编译。因编译结构不同，端到端延迟差同时包含 AE 实现影响，不能全部归于 VLM 跳层。独立 stage profiler 保留该实际调用结构，AE 分项用于诊断此混杂；它不改变已跑科学样本。
E4 released converted draft has no history input; native draft.py explicitly discards last_actions. AE verification and Full call counts must both stay zero.
Legacy contact/grasp helper returns False on exceptions; negative truth availability is not independently established. Conservative analysis maps unsupported phase claims to unknown; original telemetry is retained.
History controls are references only until checkpoint/preprocessing/seed/initial-state/timing matches are established; no historical speed ratio or paired interval is fabricated.

| stage | condition | suite | success / valid |
|---|---|---|---|
| E2 | E2_Full | libero_10 | 48 / 50 |
| E2 | E2_Full | libero_goal | 49 / 50 |
| E2 | E2_Full | libero_object | 49 / 50 |
| E2 | E2_Full | libero_spatial | 50 / 50 |
| E2 | E2_V13_Custom | libero_10 | 8 / 10 |
| E2 | E2_V13_Custom | libero_goal | 5 / 10 |
| E2 | E2_V13_Custom | libero_object | 9 / 10 |
| E2 | E2_V13_Custom | libero_spatial | 10 / 10 |
| E2 | E2_V14_A03 | libero_10 | 9 / 10 |
| E2 | E2_V14_A03 | libero_goal | 5 / 10 |
| E2 | E2_V14_A03 | libero_object | 10 / 10 |
| E2 | E2_V14_A03 | libero_spatial | 9 / 10 |
| E2 | E2_transfer15 | libero_10 | 49 / 50 |
| E2 | E2_transfer15 | libero_goal | 48 / 50 |
| E2 | E2_transfer15 | libero_object | 49 / 50 |
| E2 | E2_transfer15 | libero_spatial | 42 / 50 |
| E2 | E2_transfer16 | libero_10 | 49 / 50 |
| E2 | E2_transfer16 | libero_goal | 48 / 50 |
| E2 | E2_transfer16 | libero_object | 50 / 50 |
| E2 | E2_transfer16 | libero_spatial | 49 / 50 |

[Video index](media_index.md) · [Playable links](media_index.html) · [Trajectory index](trajectory_index.csv)
[Latency and throughput](latency_and_throughput.csv) · [Routing](routing.csv) · [Future-only reconstruction error](future_reconstruction_error.csv)
[按阶段/未来1–5步/K分支误差](future_error_by_phase_step_branch.csv) · [episode等权误差](future_error_by_phase_step_branch_episode_equal.csv) · [逐episode依据](future_error_by_phase_step_branch_per_episode.csv)
[分阶段夹爪检查事件](gripper_check_events_by_phase.csv) · [episode等权事件率](gripper_check_events_by_phase_episode_equal.csv)
误差仅含已有future窗口，不含E3历史前缀；单位是模型归一化坐标，不是毫米或真实末端误差。episode等权误差为各episode内有符号均值/RMS/距离均值再等权平均；无该组观测的episode不补零。夹爪仅为保存的布尔检查事件及明确Full路由，未保存的逐步夹爪不一致率不可得。
Phase table records furthest observed phase, not a proven cause of failure. Missing task telemetry remains unknown.
