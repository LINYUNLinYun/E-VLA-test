# idea1 v3 incremental results

Valid episodes included: 40. Technical invalid attempts are retained in worker journals.
Calibration, P0 and scientific init6–10 rows are separate in the stage column. Failed tasks remain in denominators.
Single-request mixed-policy latency and end-to-end throughput are separate columns. Startup is configuration_startup.csv; lazy first-use is included in episode policy wall.
E2 initial adapter reports native total latency only; VLM/AE partition is unavailable. CPU/GPU/disk-wait activity is not sampled.
E4 released converted draft has no history input; native draft.py explicitly discards last_actions. AE verification and Full call counts must both stay zero.
History controls are references only until checkpoint/preprocessing/seed/initial-state/timing matches are established; no historical speed ratio or paired interval is fabricated.

| stage | condition | suite | success / valid |
|---|---|---|---|
| E4 | E4_pure_learned | libero_10 | 4 / 10 |
| E4 | E4_pure_learned | libero_goal | 6 / 10 |
| E4 | E4_pure_learned | libero_object | 8 / 10 |
| E4 | E4_pure_learned | libero_spatial | 9 / 10 |

[Video index](media_index.md) · [Playable links](media_index.html) · [Trajectory index](trajectory_index.csv)
[Latency and throughput](latency_and_throughput.csv) · [Routing](routing.csv) · [Future-only reconstruction error](future_reconstruction_error.csv)
Phase table records furthest observed phase, not a proven cause of failure. Missing task telemetry remains unknown.
