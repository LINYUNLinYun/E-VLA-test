# Draft generalization analysis

Primary service metric: complete native policy-call milliseconds per actually executed action. Client RPC cost per executed action is reported alongside it. All inference calls, including rejection rounds that execute zero actions, contribute to the numerator. Mean request latency is retained as a diagnostic and cannot alone establish actual acceleration. GPU-stage timing is not end-to-end.

| Mode | Source | Target | Success / n | Policy ms / action | RPC ms / action | Episode wall mean (s) |
|---|---|---|---:|---:|---:|---:|
| draft | libero_spatial | libero_10 | 0/1 | 6.523 | 6.744 | 22.483 |
| draft | libero_spatial | libero_goal | 0/1 | 12.429 | 12.672 | 14.667 |
| draft | libero_spatial | libero_object | 0/1 | 13.247 | 13.487 | 14.740 |
| draft | libero_spatial | libero_spatial | 1/1 | 90.703 | 90.992 | 14.756 |
| flash | libero_spatial | libero_10 | 1/1 | 21.563 | 22.062 | 21.881 |
| flash | libero_spatial | libero_goal | 1/1 | 61.466 | 61.935 | 11.776 |
| flash | libero_spatial | libero_object | 1/1 | 47.240 | 47.752 | 15.132 |
| flash | libero_spatial | libero_spatial | 1/1 | 107.188 | 107.560 | 17.768 |

Same-case comparisons are in `paired_action_costs_vs_full.json`; aggregate ratios are in `speed_vs_full.json`. Service-cost speedup is Full cost / condition cost; below1 means higher condition cost. Real episode-wall ratios and both success counts are retained because trajectories and episode lengths may differ.

The init21–24 latency subset protocol does not apply to these discovery-only inputs. No warm-state or repeat-task cost table is generated.
