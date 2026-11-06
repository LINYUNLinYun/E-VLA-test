# Draft generalization analysis

Primary service metric: complete native policy-call milliseconds per actually executed action. Client RPC cost per executed action is reported alongside it. All inference calls, including rejection rounds that execute zero actions, contribute to the numerator. Mean request latency is retained as a diagnostic and cannot alone establish actual acceleration. GPU-stage timing is not end-to-end.

| Mode | Source | Target | Success / n | Policy ms / action | RPC ms / action | Episode wall mean (s) |
|---|---|---|---:|---:|---:|---:|
| draft | libero_object | libero_object | 40/40 | 4.132 | 4.407 | 6.284 |
| draft | libero_object | libero_spatial | 0/40 | 3.062 | 3.330 | 10.325 |
| draft | libero_spatial | libero_object | 0/40 | 1.978 | 2.237 | 9.541 |
| draft | libero_spatial | libero_spatial | 35/40 | 5.202 | 5.470 | 7.462 |
| flash | libero_object | libero_object | 40/40 | 7.185 | 7.512 | 6.626 |
| flash | libero_object | libero_spatial | 39/40 | 15.608 | 16.101 | 8.090 |
| flash | libero_spatial | libero_object | 39/40 | 9.486 | 9.954 | 7.476 |
| flash | libero_spatial | libero_spatial | 39/40 | 12.492 | 12.907 | 7.962 |
| full | libero_spatial | libero_object | 40/40 | 8.234 | 8.504 | 6.895 |
| full | libero_spatial | libero_spatial | 37/40 | 11.419 | 11.687 | 7.518 |

Same-case comparisons are in `paired_action_costs_vs_full.json`; aggregate ratios are in `speed_vs_full.json`. Service-cost speedup is Full cost / condition cost; below1 means higher condition cost. Real episode-wall ratios and both success counts are retained because trajectories and episode lengths may differ.

Preregistered held-out latency subsets are in `latency_subsets.json`: all init21–24, first-task init21, and fixed repeat-task init22–24. Same-case Full/matching-source cost ratios are in `latency_subset_comparisons.json`. Once-per-run readiness/preparation and run wall are separate in `run_preparation_costs.json`. Repeat-task is not guaranteed free of compilation/capture and is never labeled deployment warm-state acceleration. No success/latency-based filtering is used.
