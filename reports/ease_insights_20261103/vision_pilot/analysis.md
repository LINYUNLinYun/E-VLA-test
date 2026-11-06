# EASE fast-path independent confirmation

Pilot complete; execute the frozen full confirmation regardless of observed direction.

Stage: `pilot`. Full matrix complete: **True**. Artifact hashes verified: **True**.

| Condition | Successes | Mean completed model request (ms) |
|---|---:|---:|
| pi0_full | 39/40 (expected 40) | 108.15 |
| pi0_keep24_2_front_middle | 36/40 (expected 40) | 108.55 |
| pi0_keep24_1_uniform | 38/40 (expected 40) | 110.22 |

| Suite and direction | N | Difference (pp) | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Wins/losses | One-sided McNemar p | Tasks +/−/= |
|---|---:|---:|---|---|---:|---:|---|
| libero_spatial: A > B (descriptive) | 10 | +0.0 | [0.0, 0.0] | [0.0, 0.0] | 0/0 | 1 | 0/0/10 |
| libero_object: A > B (descriptive) | 10 | +0.0 | [0.0, 0.0] | [0.0, 0.0] | 0/0 | 1 | 0/0/10 |
| libero_goal: A > B (predeclared) | 10 | -10.0 | [-30.0, 0.0] | [-30.0, 0.0] | 0/1 | 1 | 0/1/9 |
| libero_10: B > A (predeclared) | 10 | +10.0 | [0.0, 30.0] | [0.0, 30.0] | 1/0 | 0.5 | 1/0/9 |

A = `pi0_keep24_2_front_middle`; B = `pi0_keep24_1_uniform`. Positive differences follow each row’s stated direction.

Directions/candidates/cases were frozen before validation. One-sided exact McNemar and two-sided 95% paired percentile intervals report effect and uncertainty; equal-task cluster bootstrap and exact task sign-flip are sensitivity analyses. Strong wording additionally requires complete data and both positive cluster intervals. This wording rule is an analysis convention, not an additional preregistered statistical test.

Per-run wall time includes workers and serial model queue; sum of concurrent run walls is resource time, not end-to-end makespan. Failed/incomplete run wall time must be counted separately for total experiment cost.

Request latency compares each policy’s own closed-loop observations. Service preparation, cold compilation and 20 warmup requests are separate in per-run records. A paired same-input benchmark is needed to isolate compute speedup.

Per-task and per-init effect counts, paired case outcomes, Full retention, and complete per-run timing are included in the JSON.
