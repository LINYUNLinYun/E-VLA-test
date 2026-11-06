Supplementary VLM confirmation: 600/600 valid episodes; status=complete.
Original Spatial/Long conclusion: both_prespecified_directions_not_reproduced.
Supplementary Goal/Long conclusion: both_prespecified_directions_not_reproduced.
Prospective registration: prospective_supplementary.

| Fixed contrast | n | Difference | A-only / B-only | One-sided exact p | Holm-3 p | Case 95% CI | Task-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---|---|
| primary_spatial: fixed_v12_jacobian > fixed_v12_action_mse | 50 | 0.02 | 4 / 3 | 0.5 | 1 | [-0.080, +0.120] | [-0.080, +0.140] |
| primary_long: fixed_v12_action_mse > fixed_v12_jacobian | 50 | 0 | 5 / 5 | 0.623047 | 1 | [-0.120, +0.120] | [-0.120, +0.100] |
| supplementary_goal: fixed_v12_jacobian > fixed_v12_action_mse | 50 | -0.08 | 3 / 7 | 0.945312 | 1 | [-0.200, +0.040] | [-0.200, +0.020] |

supplementary_analysis.json includes all task/init/case effects, leave-one-task/init-out checks, the original Holm-2 results, registration evidence, costs and all known contention records.
Original primary failure is retained. Quality outcomes are unchanged. Timings do not establish exclusive-GPU speed; non-significance does not establish equivalence.
