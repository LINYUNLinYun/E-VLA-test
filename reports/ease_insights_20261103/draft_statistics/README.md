# Independent held-out confirmation

Status: **complete**. No pair is chosen using these results.

## libero_object: matching versus libero_spatial

Primary status: **replicated**.

| Condition | Valid / planned | Missing |
|---|---:|---:|
| matching_pure | 40/40 | 0 |
| foreign_pure | 40/40 | 0 |
| matching_FLASH | 40/40 | 0 |
| foreign_FLASH | 40/40 | 0 |
| Full | 40/40 | 0 |

Pure success: matching 40/40, foreign 0/40; matching-only=40, foreign-only=0.
One-sided exact McNemar p=9.09495e-13; Holm-adjusted p=1.81899e-12.
Case mean gap=1.0000, paired-case95% CI=[1.0, 1.0]; task mean gap=1.0000, task-cluster95% CI=[1.0, 1.0].

| Secondary metric | Mode / subset | Paired n | Matching − foreign | Case95% CI | Task-cluster95% CI |
|---|---|---:|---:|---|---|
| success | flash / all4_init21_24 | 40 | 0.025 | [0.0, 0.075] | [0.0, 0.075] |
| acceptance_rate | flash / all4_init21_24 | 40 | 0.73871 | [0.6980950727513228, 0.7777410300925927] | [0.689745453042328, 0.783411044973545] |
| rejection_full_fraction | flash / all4_init21_24 | 40 | -0.35217 | [-0.3707308039617682, -0.3344695920436955] | [-0.3752011236356188, -0.32991613198183456] |
| policy_ms_per_executed_action | draft / all4_init21_24 | 40 | 2.0688 | [0.46137207126048857, 4.308236534551418] | [0.6949067550928462, 3.9215576183239693] |
| client_rpc_ms_per_executed_action | draft / all4_init21_24 | 40 | 2.086 | [0.4659126755237832, 4.366238684213068] | [0.7087998179939394, 3.900267142762352] |
| policy_ms_per_executed_action | draft / repeat_task_init22_24 | 30 | 0.019986 | [0.005528487326410697, 0.0349232578993366] | [0.006827655397234631, 0.033576867650744614] |
| client_rpc_ms_per_executed_action | draft / repeat_task_init22_24 | 30 | 0.037769 | [0.009108008370594462, 0.06576877969136644] | [0.01864665888637076, 0.05852314409279975] |
| policy_ms_per_executed_action | flash / all4_init21_24 | 40 | -2.3983 | [-3.1565638629331034, -1.833124533115502] | [-3.079776529121634, -1.8253744590470713] |
| client_rpc_ms_per_executed_action | flash / all4_init21_24 | 40 | -2.5429 | [-3.287770681228723, -1.9693778484545834] | [-3.2207796353708655, -1.9696118294805558] |
| policy_ms_per_executed_action | flash / repeat_task_init22_24 | 30 | -1.9132 | [-2.0185206574119365, -1.800687222948342] | [-2.042060825535577, -1.7838239809481835] |
| client_rpc_ms_per_executed_action | flash / repeat_task_init22_24 | 30 | -2.0611 | [-2.188374097279075, -1.930045736872175] | [-2.214149135784746, -1.9137111445162411] |

Positive differences favor matching for success/acceptance; negative differences favor matching for rejection fraction and cost. Full baseline, both success counts, episode walls, and request/action totals are in statistics.json.

## libero_spatial: matching versus libero_object

Primary status: **replicated**.

| Condition | Valid / planned | Missing |
|---|---:|---:|
| matching_pure | 40/40 | 0 |
| foreign_pure | 40/40 | 0 |
| matching_FLASH | 40/40 | 0 |
| foreign_FLASH | 40/40 | 0 |
| Full | 40/40 | 0 |

Pure success: matching 35/40, foreign 0/40; matching-only=35, foreign-only=0.
One-sided exact McNemar p=2.91038e-11; Holm-adjusted p=2.91038e-11.
Case mean gap=0.8750, paired-case95% CI=[0.775, 0.975]; task mean gap=0.8750, task-cluster95% CI=[0.8, 0.95].

| Secondary metric | Mode / subset | Paired n | Matching − foreign | Case95% CI | Task-cluster95% CI |
|---|---|---:|---:|---|---|
| success | flash / all4_init21_24 | 40 | 0 | [-0.075, 0.075] | [-0.075, 0.075] |
| acceptance_rate | flash / all4_init21_24 | 40 | 0.59855 | [0.5532537784842472, 0.6444785860840548] | [0.5555237644300144, 0.6414871708152958] |
| rejection_full_fraction | flash / all4_init21_24 | 40 | -0.34141 | [-0.3626837473354333, -0.31990009913477907] | [-0.365752439450645, -0.3158157211295757] |
| policy_ms_per_executed_action | draft / all4_init21_24 | 40 | 3.0641 | [1.014771338559628, 5.91247214848751] | [1.5409018302285684, 5.221097514611256] |
| client_rpc_ms_per_executed_action | draft / all4_init21_24 | 40 | 3.0636 | [1.0166729208691034, 5.838920809159473] | [1.544086228965448, 5.154350127808976] |
| policy_ms_per_executed_action | draft / repeat_task_init22_24 | 30 | 0.0091259 | [-0.005441963291592893, 0.023970087152045974] | [-0.006575329722192541, 0.025909501123924605] |
| client_rpc_ms_per_executed_action | draft / repeat_task_init22_24 | 30 | 0.010673 | [-0.01752337721584307, 0.03848677966151636] | [-0.02065630195509913, 0.04209373826947378] |
| policy_ms_per_executed_action | flash / all4_init21_24 | 40 | -3.2863 | [-5.879928300215531, -1.3729797586374428] | [-5.536943896911519, -1.4139589232832166] |
| client_rpc_ms_per_executed_action | flash / all4_init21_24 | 40 | -3.3602 | [-5.935441884153471, -1.482993560314977] | [-5.5777729709169765, -1.505843409559398] |
| policy_ms_per_executed_action | flash / repeat_task_init22_24 | 30 | -1.7201 | [-1.8435751429680078, -1.593828699597953] | [-1.8300812311992583, -1.599665811512544] |
| client_rpc_ms_per_executed_action | flash / repeat_task_init22_24 | 30 | -1.8266 | [-1.9803789174286506, -1.6733809372029744] | [-1.9639706768180123, -1.685639049873809] |

Positive differences favor matching for success/acceptance; negative differences favor matching for rejection fraction and cost. Full baseline, both success counts, episode walls, and request/action totals are in statistics.json.

McNemar is a case-level analysis on repeated initial states of fixed tasks. Task-cluster intervals use ten clusters and do not establish generalization to unseen tasks. Secondary metrics are descriptive; requests are never independent samples.

FLASH need not lose success for draft generalization to matter: Full can preserve success while verification or service cost deteriorates. Cost uses complete policy/RPC milliseconds per executed action, including zero-action rejection requests; per-request latency is not actual acceleration. Repeat-task init22–24 is not guaranteed compilation-free.

Additional exports: [confirmed figures and CSV](../draft_confirmation_figures/README_zh.md). A separately labeled [post hoc task-level sensitivity](../draft_task_sensitivity/README_zh.md) rereads the original160 pure-head results and groups all four initial states within each task; both frozen directions have ten positive task differences and task-signflip Holm2 p=0.001953125. It adds no independent cases and leaves the preregistered case-level analysis unchanged.
