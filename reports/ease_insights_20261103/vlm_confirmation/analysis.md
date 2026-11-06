Independent VLM confirmation: 600/600 valid episodes; status=complete.
Conclusion: both_prespecified_directions_not_reproduced.

| Frozen direction | n | Difference | A-only / B-only | One-sided exact p | Holm p | Case 95% CI | Task-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---|---|
| libero_spatial: fixed_v12_jacobian > fixed_v12_action_mse | 50 | 0.0200 | 4 / 3 | 0.5000 | 1.0000 | [-0.080, +0.120] | [-0.080, +0.140] |
| libero_10: fixed_v12_action_mse > fixed_v12_jacobian | 50 | 0.0000 | 5 / 5 | 0.6230 | 1.0000 | [-0.120, +0.120] | [-0.120, +0.100] |

Per-task/per-init consistency, leave-one-group-out checks, Full references, complete policy/RPC costs, fixed-input probes, and every known contention record are in analysis.json.
Quality outcomes are retained unchanged. Timings do not establish exclusive-GPU speed. Non-significance does not establish equivalence; no path or category is reselected.
