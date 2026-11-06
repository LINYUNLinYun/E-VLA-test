# idea1 v3 incremental results

Valid episodes included: 200. Technical invalid attempts are retained in worker journals.
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
| E4 | E4_pure_learned | libero_10 | 22 / 50 |
| E4 | E4_pure_learned | libero_goal | 29 / 50 |
| E4 | E4_pure_learned | libero_object | 40 / 50 |
| E4 | E4_pure_learned | libero_spatial | 46 / 50 |

[Video index](media_index.md) · [Playable links](media_index.html) · [Trajectory index](trajectory_index.csv)
[Latency and throughput](latency_and_throughput.csv) · [Routing](routing.csv) · [Future-only reconstruction error](future_reconstruction_error.csv)
Phase table records furthest observed phase, not a proven cause of failure. Missing task telemetry remains unknown.
