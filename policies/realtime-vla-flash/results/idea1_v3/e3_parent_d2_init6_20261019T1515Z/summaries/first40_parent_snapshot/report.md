# idea1 v3 incremental results

Valid episodes included: 236. Technical invalid attempts are retained in worker journals.
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
| E1 | E1_D10_tau0.15_P2 | libero_object | 10 / 10 |
| E1 | E1_D10_tau0.15_P2 | libero_spatial | 10 / 10 |
| E1 | E1_D2_tau0.15_P2 | libero_10 | 10 / 10 |
| E1 | E1_D2_tau0.15_P2 | libero_goal | 8 / 10 |
| E1 | E1_D2_tau0.15_P2 | libero_object | 9 / 10 |
| E1 | E1_D2_tau0.15_P2 | libero_spatial | 10 / 10 |
| E1 | E1_D2_tau0.1_P2 | libero_10 | 9 / 10 |
| E1 | E1_D2_tau0.1_P2 | libero_goal | 8 / 10 |
| E1 | E1_D2_tau0.1_P2 | libero_object | 10 / 10 |
| E1 | E1_D2_tau0.1_P2 | libero_spatial | 10 / 10 |
| E3 | E3_D2_tau0.1_P2_previous_anchor | libero_10 | 10 / 10 |
| E3 | E3_D2_tau0.1_P2_previous_anchor | libero_goal | 9 / 10 |
| E3 | E3_D2_tau0.1_P2_previous_anchor | libero_object | 10 / 10 |
| E3 | E3_D2_tau0.1_P2_previous_anchor | libero_spatial | 10 / 10 |
| calibration | E1_D10_tau0.15_P2 | libero_10 | 2 / 2 |
| calibration | E1_D10_tau0.15_P2 | libero_goal | 1 / 2 |
| calibration | E1_D10_tau0.15_P2 | libero_object | 2 / 2 |
| calibration | E1_D10_tau0.15_P2 | libero_spatial | 2 / 2 |
| calibration | E1_D10_tau0.15_P4 | libero_10 | 2 / 2 |
| calibration | E1_D10_tau0.15_P4 | libero_goal | 1 / 2 |
| calibration | E1_D10_tau0.15_P4 | libero_object | 2 / 2 |
| calibration | E1_D10_tau0.15_P4 | libero_spatial | 2 / 2 |
| calibration | E1_D10_tau0.1_P2 | libero_10 | 1 / 2 |
| calibration | E1_D10_tau0.1_P2 | libero_goal | 1 / 2 |
| calibration | E1_D10_tau0.1_P2 | libero_object | 2 / 2 |
| calibration | E1_D10_tau0.1_P2 | libero_spatial | 2 / 2 |
| calibration | E1_D10_tau0.1_P4 | libero_10 | 2 / 2 |
| calibration | E1_D10_tau0.1_P4 | libero_goal | 1 / 2 |
| calibration | E1_D10_tau0.1_P4 | libero_object | 2 / 2 |
| calibration | E1_D10_tau0.1_P4 | libero_spatial | 2 / 2 |
| calibration | E1_D10_tau0.25_P2 | libero_10 | 2 / 2 |
| calibration | E1_D10_tau0.25_P2 | libero_goal | 1 / 2 |
| calibration | E1_D10_tau0.25_P2 | libero_object | 2 / 2 |
| calibration | E1_D10_tau0.25_P2 | libero_spatial | 2 / 2 |
| calibration | E1_D10_tau0.25_P4 | libero_10 | 2 / 2 |
| calibration | E1_D10_tau0.25_P4 | libero_goal | 1 / 2 |
| calibration | E1_D10_tau0.25_P4 | libero_object | 2 / 2 |
| calibration | E1_D10_tau0.25_P4 | libero_spatial | 2 / 2 |
| calibration | E1_D2_tau0.15_P2 | libero_10 | 2 / 2 |
| calibration | E1_D2_tau0.15_P2 | libero_goal | 1 / 2 |
| calibration | E1_D2_tau0.15_P2 | libero_object | 1 / 2 |
| calibration | E1_D2_tau0.15_P2 | libero_spatial | 2 / 2 |
| calibration | E1_D2_tau0.15_P4 | libero_10 | 2 / 2 |
| calibration | E1_D2_tau0.15_P4 | libero_goal | 0 / 2 |
| calibration | E1_D2_tau0.15_P4 | libero_object | 2 / 2 |
| calibration | E1_D2_tau0.15_P4 | libero_spatial | 2 / 2 |
| calibration | E1_D2_tau0.1_P2 | libero_10 | 2 / 2 |
| calibration | E1_D2_tau0.1_P2 | libero_goal | 1 / 2 |
| calibration | E1_D2_tau0.1_P2 | libero_object | 2 / 2 |
| calibration | E1_D2_tau0.1_P2 | libero_spatial | 2 / 2 |
| calibration | E1_D2_tau0.1_P4 | libero_10 | 2 / 2 |
| calibration | E1_D2_tau0.1_P4 | libero_goal | 1 / 2 |
| calibration | E1_D2_tau0.1_P4 | libero_object | 2 / 2 |
| calibration | E1_D2_tau0.1_P4 | libero_spatial | 2 / 2 |
| calibration | E1_D2_tau0.25_P2 | libero_10 | 2 / 2 |
| calibration | E1_D2_tau0.25_P2 | libero_goal | 1 / 2 |
| calibration | E1_D2_tau0.25_P2 | libero_object | 1 / 2 |
| calibration | E1_D2_tau0.25_P2 | libero_spatial | 2 / 2 |
| calibration | E1_D2_tau0.25_P4 | libero_10 | 2 / 2 |
| calibration | E1_D2_tau0.25_P4 | libero_goal | 0 / 2 |
| calibration | E1_D2_tau0.25_P4 | libero_object | 2 / 2 |
| calibration | E1_D2_tau0.25_P4 | libero_spatial | 2 / 2 |

[Video index](media_index.md) · [Playable links](media_index.html) · [Trajectory index](trajectory_index.csv)
[Latency and throughput](latency_and_throughput.csv) · [Routing](routing.csv) · [Future-only reconstruction error](future_reconstruction_error.csv)
Phase table records furthest observed phase, not a proven cause of failure. Missing task telemetry remains unknown.
