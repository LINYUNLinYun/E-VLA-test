# Batch preparation, 2026-11-01

This directory records preparation only. No GPU query, lease, model launch, or background GPU waiter was started for this preparation.

- `cpu_validation.json`: 87 CPU checks passed; CUDA uninitialized; source hashes frozen.
- `cpu_checks.log`: complete test output.
- `preparation_check.json`: all 36 generated commands parse, match the verified sources, and have not started.
- `strict_b1/commands.md`: 12 strict B1 candidate commands.
- `batch_diagnostic/commands.md`: 24 explicitly qualified B2/B4 scheduling-diagnostic commands.
- Each plan includes a `plan.json`, a placeholder GPU UUID, and fresh future run directories. The command lists are not automatic queues.

Guide: `docs/experiments/idea2/idea2_batch_pipeline_preparation_20261101.md`.

Default strict numerical tolerances are unchanged. Historical cross-B1 numerical failures remain valid failures in `results/idea2/batch_study_20261031/`. Diagnostic-only measurements cannot establish B1 equivalence or closed-loop quality. Reuse existing valid results rather than rerunning the whole list.
