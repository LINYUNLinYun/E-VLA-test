# Fixed-batch preparation: ready for GPU smoke, not benchmarked

Client date: 2026-10-31. Status: `prepared_not_run`.

- CPU checks: **81 passed**, zero failures/errors, CUDA not initialized.
- CPU-only Green Context bridge compilation passed (`green_build/`).
- Model / algorithm: full pi05_libero, D10/H10, original precision and FP32 Euler recurrence.
- Conditions: B1/B2/B4, full-GPU serial versus pipeline, separate-process SM16 Green pipeline; prepared and online-prepare replay measured separately.
- Same 12 trusted recorded observations and per-record noise identities across batch sizes; fixed lane membership, whole-batch reset, two KV slots.
- No GPU query, lease claim, experiment, profiler capture or polling task was started for this preparation.

`cpu_validation.json` records source hashes and checks. `cpu_checks.log` contains the detailed output. `input_schedule_preview.json` records input grouping. `experiment_plan/plan.json` has 36 **not-started** jobs and their source snapshot hashes; `experiment_plan/commands.md` provides commands, with an unassigned GPU UUID placeholder.

Run smoke first after explicit GPU allocation. Full-model GPU numerical validation, capture compatibility, memory use, timing and actual kernel overlap remain untested. The recorded-input replay is not evidence of multi-robot closed-loop success or online observation freshness.

Design: [experiment plan](../../../docs/experiments/idea2/idea2_batch_pipeline_plan_20261031.md).
