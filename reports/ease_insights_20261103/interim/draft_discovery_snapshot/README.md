# Draft generalization analysis

Primary service metric: complete native policy-call milliseconds per actually executed action. Client RPC cost per executed action is reported alongside it. All inference calls, including rejection rounds that execute zero actions, contribute to the numerator. Mean request latency is retained as a diagnostic and cannot alone establish actual acceleration. GPU-stage timing is not end-to-end.

| Mode | Source | Target | Success / n | Policy ms / action | RPC ms / action | Episode wall mean (s) |
|---|---|---|---:|---:|---:|---:|
| draft | libero_goal | libero_10 | 0/10 | 5.106 | 5.367 | 18.359 |
| draft | libero_goal | libero_goal | 8/10 | 17.457 | 17.713 | 8.081 |
| draft | libero_goal | libero_object | 0/10 | 4.696 | 5.038 | 10.405 |
| draft | libero_goal | libero_spatial | 0/10 | 11.233 | 11.498 | 13.092 |
| draft | libero_object | libero_10 | 0/10 | 4.953 | 5.225 | 18.026 |
| draft | libero_object | libero_goal | 0/10 | 8.641 | 8.915 | 12.742 |
| draft | libero_object | libero_object | 10/10 | 10.957 | 11.248 | 7.246 |
| draft | libero_object | libero_spatial | 0/10 | 12.666 | 12.934 | 13.121 |
| draft | libero_spatial | libero_10 | 0/9 | 6.946 | 7.216 | 20.131 |
| draft | libero_spatial | libero_goal | 0/9 | 8.949 | 9.218 | 13.666 |
| draft | libero_spatial | libero_object | 0/9 | 6.937 | 7.198 | 11.765 |
| draft | libero_spatial | libero_spatial | 8/9 | 23.415 | 23.700 | 10.233 |
| flash | libero_10 | libero_spatial | 3/3 | 116.766 | 117.181 | 19.903 |
| flash | libero_goal | libero_10 | 6/10 | 16.516 | 16.994 | 18.309 |
| flash | libero_goal | libero_goal | 10/10 | 41.201 | 41.531 | 9.731 |
| flash | libero_goal | libero_object | 10/10 | 21.021 | 21.476 | 9.093 |
| flash | libero_goal | libero_spatial | 9/10 | 39.874 | 40.363 | 12.484 |
| flash | libero_object | libero_10 | 8/10 | 20.771 | 21.218 | 16.554 |
| flash | libero_object | libero_goal | 10/10 | 52.212 | 52.671 | 10.760 |
| flash | libero_object | libero_object | 10/10 | 23.120 | 23.459 | 9.029 |
| flash | libero_object | libero_spatial | 10/10 | 49.015 | 49.454 | 11.896 |
| flash | libero_spatial | libero_10 | 8/9 | 22.175 | 22.679 | 16.728 |
| flash | libero_spatial | libero_goal | 8/9 | 45.394 | 45.926 | 13.112 |
| flash | libero_spatial | libero_object | 9/9 | 25.944 | 26.425 | 9.996 |
| flash | libero_spatial | libero_spatial | 9/9 | 53.458 | 54.610 | 12.964 |
| full | libero_spatial | libero_10 | 9/10 | 14.232 | 14.473 | 13.693 |
| full | libero_spatial | libero_goal | 8/10 | 26.248 | 26.501 | 10.214 |
| full | libero_spatial | libero_object | 10/10 | 18.214 | 18.497 | 8.696 |
| full | libero_spatial | libero_spatial | 10/10 | 41.239 | 41.499 | 11.534 |

Same-case comparisons are in `paired_action_costs_vs_full.json`; aggregate ratios are in `speed_vs_full.json`. Service-cost speedup is Full cost / condition cost; below1 means higher condition cost. Real episode-wall ratios and both success counts are retained because trajectories and episode lengths may differ.

The init21–24 latency subset protocol does not apply to these discovery-only inputs. No warm-state or repeat-task cost table is generated.
