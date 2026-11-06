VLM discovery: 160/160 valid episodes.
Selection status: selected. Independent validation has not run.

| Condition | Suite | n | Success | Full request fraction | Policy mean ms |
|---|---|---:|---:|---:|---:|
| full_d10 | all | 40 | 38 | 1.000 | 160.891 |
| fixed_v12_jacobian | all | 40 | 35 | 0.342 | 127.191 |
| fixed_v12_action_mse | all | 40 | 32 | 0.340 | 107.449 |
| fixed_v12_cosine | all | 40 | 15 | 0.339 | 88.157 |
| full_d10 | libero_spatial | 10 | 10 | 1.000 | 149.998 |
| fixed_v12_jacobian | libero_spatial | 10 | 9 | 0.348 | 110.171 |
| fixed_v12_action_mse | libero_spatial | 10 | 8 | 0.339 | 119.842 |
| fixed_v12_cosine | libero_spatial | 10 | 5 | 0.345 | 95.958 |
| full_d10 | libero_object | 10 | 10 | 1.000 | 149.785 |
| fixed_v12_jacobian | libero_object | 10 | 10 | 0.346 | 131.918 |
| fixed_v12_action_mse | libero_object | 10 | 10 | 0.346 | 131.078 |
| fixed_v12_cosine | libero_object | 10 | 5 | 0.341 | 89.765 |
| full_d10 | libero_goal | 10 | 10 | 1.000 | 219.198 |
| fixed_v12_jacobian | libero_goal | 10 | 10 | 0.344 | 143.378 |
| fixed_v12_action_mse | libero_goal | 10 | 7 | 0.340 | 113.299 |
| fixed_v12_cosine | libero_goal | 10 | 4 | 0.336 | 99.354 |
| full_d10 | libero_10 | 10 | 8 | 1.000 | 148.607 |
| fixed_v12_jacobian | libero_10 | 10 | 6 | 0.338 | 124.732 |
| fixed_v12_action_mse | libero_10 | 10 | 7 | 0.338 | 87.880 |
| fixed_v12_cosine | libero_10 | 10 | 1 | 0.337 | 79.997 |

All same-case comparisons, discordant counts, task diagnostics, timings, and cost probes are in analysis.json.
A task row is one initial-state outcome, not evidence for a task category. Selected suite differences require independent init31–35 validation.

Selected discovery pair: fixed_v12_jacobian, fixed_v12_action_mse.
- libero_spatial: fixed_v12_jacobian > fixed_v12_action_mse, discovery difference +0.100.
- libero_10: fixed_v12_action_mse > fixed_v12_jacobian, discovery difference +0.100.
