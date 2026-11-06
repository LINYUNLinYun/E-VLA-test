# EASE fast-path independent confirmation

Incomplete matrix: provisional descriptive evidence; no confirmation claim.

Stage: `pilot`. Full matrix complete: **False**. Artifact hashes verified: **True**.

| Condition | Successes | Mean completed model request (ms) |
|---|---:|---:|
| pi0_full | 0/0 (expected 40) | — |
| pi0_keep24_2_front_middle | 0/0 (expected 40) | — |
| pi0_keep24_1_uniform | 0/0 (expected 40) | — |

| Suite and direction | N | Difference (pp) | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Wins/losses | One-sided McNemar p | Tasks +/−/= |
|---|---:|---:|---|---|---:|---:|---|

A = `pi0_keep24_2_front_middle`; B = `pi0_keep24_1_uniform`. Positive differences follow each row’s stated direction.

Directions/candidates/cases were frozen before validation. One-sided exact McNemar and two-sided 95% paired percentile intervals report effect and uncertainty; equal-task cluster bootstrap and exact task sign-flip are sensitivity analyses. Strong wording additionally requires complete data and both positive cluster intervals. This wording rule is an analysis convention, not an additional preregistered statistical test.

Per-run wall time includes workers and serial model queue; sum of concurrent run walls is resource time, not end-to-end makespan. Failed/incomplete run wall time must be counted separately for total experiment cost.

Request latency compares each policy’s own closed-loop observations. Service preparation, cold compilation and 20 warmup requests are separate in per-run records. A paired same-input benchmark is needed to isolate compute speedup.

Per-task and per-init effect counts, paired case outcomes, Full retention, and complete per-run timing are included in the JSON.
