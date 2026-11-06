# Draft generalization analysis

Primary service metric: complete native policy-call milliseconds per actually executed action. Client RPC cost per executed action is reported alongside it. All inference calls, including rejection rounds that execute zero actions, contribute to the numerator. Mean request latency is retained as a diagnostic and cannot alone establish actual acceleration. GPU-stage timing is not end-to-end.

| Mode | Source | Target | Success / n | Policy ms / action | RPC ms / action | Episode wall mean (s) |
|---|---|---|---:|---:|---:|---:|
| draft | libero_goal | libero_10 | 0/10 | 5.106 | 5.367 | 18.359 |
| draft | libero_goal | libero_goal | 8/10 | 17.457 | 17.713 | 8.081 |
| draft | libero_goal | libero_object | 0/10 | 4.696 | 5.038 | 10.405 |
| draft | libero_goal | libero_spatial | 0/10 | 11.233 | 11.498 | 13.092 |
| draft | libero_spatial | libero_10 | 0/1 | 6.523 | 6.744 | 22.483 |
| draft | libero_spatial | libero_goal | 0/1 | 12.429 | 12.672 | 14.667 |
| draft | libero_spatial | libero_object | 0/1 | 13.247 | 13.487 | 14.740 |
| draft | libero_spatial | libero_spatial | 1/1 | 90.703 | 90.992 | 14.756 |
| flash | libero_goal | libero_10 | 6/10 | 16.516 | 16.994 | 18.309 |
| flash | libero_goal | libero_goal | 10/10 | 41.201 | 41.531 | 9.731 |
| flash | libero_goal | libero_object | 10/10 | 21.021 | 21.476 | 9.093 |
| flash | libero_goal | libero_spatial | 9/10 | 39.874 | 40.363 | 12.484 |
| flash | libero_spatial | libero_10 | 9/10 | 22.105 | 22.609 | 17.244 |
| flash | libero_spatial | libero_goal | 9/10 | 46.629 | 47.155 | 12.979 |
| flash | libero_spatial | libero_object | 10/10 | 28.127 | 28.611 | 10.509 |
| flash | libero_spatial | libero_spatial | 10/10 | 57.571 | 58.663 | 13.444 |
| full | libero_spatial | libero_10 | 9/10 | 14.232 | 14.473 | 13.693 |
| full | libero_spatial | libero_goal | 8/10 | 26.248 | 26.501 | 10.214 |
| full | libero_spatial | libero_object | 10/10 | 18.214 | 18.497 | 8.696 |
| full | libero_spatial | libero_spatial | 10/10 | 41.239 | 41.499 | 11.534 |

Same-case comparisons are in `paired_action_costs_vs_full.json`; aggregate ratios are in `speed_vs_full.json`. Service-cost speedup is Full cost / condition cost; below1 means higher condition cost. Real episode-wall ratios and both success counts are retained because trajectories and episode lengths may differ.

The init21–24 latency subset protocol does not apply to these discovery-only inputs. No warm-state or repeat-task cost table is generated.
