# Continuous Encoder: prospective outcome statistics and artifact contract

Prepared for the new online protocol before its outcome matrix. This document owns only statistical design and recommended evidence fields; it does not alter the runtime or choose new gate scenes. The root plan can link this document. No prior synchronous,250ms-lag, or other quality episodes are reused.

## Fixed cases and gate inclusion

Three arms: `fresh18_no_encoder`, `sm16_no_encoder`, `sm16_encoder`. Each has100 cases: each of four suites (`libero_object`, `libero_spatial`, `libero_goal`, `libero_10`) contains tasks0–9 at init0/1, plus init2 for even tasks0/2/4/6/8. Thus each suite has25 cases, with five three-init tasks and five two-init tasks. The complete matrix has300 unique `(configuration,suite,task,init)` cells.

The initially fixed two-scene × three-arm gate contributes six of those300 cells if its action/provenance/safety gate is valid. Choose the two scene IDs before seeing their task outcomes. Passing the gate means the consumed-action protocol, real observation provenance, two-slot lifecycle, and measured scheduling window are valid; it does **not** require task success. A native valid task failure in a gate scene remains a failure in the final denominator. Do not replace a hard scene, alter the grid or rerun until success. A technically invalid attempt may be retried at most once in a new directory, retaining both attempts and the specific invalidity reason. Statistics receives only the explicitly selected valid attempt after provenance review; duplicates are rejected, not silently resolved.

All three arms use the new online wall-clock controller, real camera source, and identical chunk-consumption rules. Old synchronous or fixed-lag results are not a baseline for the primary outcome comparison. Model/checkpoint, BF16/D10/H10/R5, environment init/seed, noise construction and action transformation must be fixed. A different online observation stream is an intended consequence of treatment, not permission to use arbitrary candidate seeds.

## Online protocol evidence that must accompany the outcomes

The environment advances on its actual20Hz wall clock and consumes only the selected R5 actions. At an actual five-action boundary, select the latest completed eligible prediction for the next target chunk. Candidates for the same target chunk all use the original `model_seed(suite,task,init,target_chunk)`; discarding or superseding a candidate does not advance this seed index. Overdue/unselected outputs are recorded as discarded, never executed, and H10 entries5–9 are not an emergency extension of the chosen chunk. Missing eligible results cause a recorded wait/gap according to the fixed client protocol.

Evidence must show that every selected candidate used an already-realized per-step camera frame, and identify whether a next-Encoder preparation was actually reused. An overlapping Host/NVTX region is not sufficient evidence of a useful GPU window; this gate is checked separately from the task success statistic. All candidates, including unconsumed ones, retain identities so producer/reader/slot/Event histories can be reconstructed. No success-dependent gate selection is allowed.

Repeated unconsumed candidates are not extra independent observations for success-rate inference. The unit of the primary quality analysis is one whole environment episode. The unit of pairing is the same suite/task/init across arms, not a candidate number or divergent later frame.

## Estimands, comparisons and confidence intervals

Report per arm success count/100 and per-suite count/25, with nominal95% Wilson intervals using z=1.959963984540054. These are descriptive episode-binomial intervals, not cluster-robust population guarantees.

There is exactly one overall primary contrast: **sm16_encoder minus fresh18_no_encoder**. The matched Encoder ablation is **sm16_encoder minus sm16_no_encoder**. The third contrast, **sm16_no_encoder minus fresh18_no_encoder**, describes the combined mixed-KV/resource path and is exploratory. Suite-level contrasts are also exploratory. Do not promote the ablation or a favorable suite into an extra confirmatory test after observing outcomes. No multiplicity adjustment is necessary for the single prespecified primary test; this does not make all the other reported p-values confirmatory.

For each contrast, retain every casewise win/loss and report b = candidate-only success, c = reference-only success, both-success and both-failure counts. The paired success-rate difference is Δ=(b−c)/n; positive favors the candidate. Calculate two-sided exact conditional McNemar as min(1,2·P[Binomial(b+c,0.5)≤min(b,c)]), with p=1 for no discordance. No mid-p or asymptotic chi-square substitution. The conditional exact calculation assumes independent episode discordances; task dependence limits its interpretation.

Use10,000 bootstrap draws, seed20261026,95% percentile intervals with linear interpolation at index `(draws−1)·p`. Resample the40 `(suite,task)` clusters with replacement, preserving all paired initializations in a sampled task. Twenty tasks have three cases and twenty have two; each replicate must compute **sum(paired differences)/sum(sampled episode counts)**. An unweighted mean of task means would change the episode-average estimand. Per-suite sensitivity resamples10 tasks. As a separate sensitivity, resample100 paired episodes (25 within a suite), using seed20261027. Neither method resamples arms independently. Overall task resampling is global and may vary suite composition in a bootstrap replicate.

Task/episode bootstrap degeneracy is explicitly flagged. Zero discordances produce a [0,0] empirical interval, which does not prove true equality. Only40 task clusters, and10 per suite, are available. Sparse failures, ceiling success and selection of this grid can all limit precision and generalization.

## What100 cases do not establish

One episode is1 percentage point overall and4pp within a suite. Even100/100 has a nominal Wilson lower endpoint of about96.30%, not a guarantee of1pp safety. Five discordant pairs all favoring one arm give exact two-sided p=.0625; six give.03125. Thus failure to reject equality can coexist with important practical differences. With iid discordant fraction q=.10, the approximate95% half-width near Δ=0 is6.2pp; task correlation can make precision worse. This is an illustration, not a guaranteed power calculation.

Do not interpret p>0.05 as no effect, quality preservation, equivalence or noninferiority. No user-approved noninferiority margin or SLO threshold is introduced by this statistics module. Show the estimates, uncertainty and actual valid failures. A lower MAE, faster candidate generation or higher success in this selected grid does not by itself establish general robotic safety or an official full LIBERO score.

## Normalized case schema and CSV outputs

Minimum normalized row:

```json
{"configuration":"sm16_encoder","suite":"libero_object","task":0,"init":0,
 "status":"valid","success":false,"steps":280,"directory":"/absolute/episode/path"}
```

The module accepts `configuration`, `label`, or `arm`; if multiple are supplied they must agree. `success` must be an actual JSON boolean; a valid failure is false, while pending/technical failure has a nonvalid status and must not masquerade as a quality outcome. Input can be a row list or `{"cases":[...]}`. It verifies the exact300-cell grid, strict task/init coordinates, duplicates and invalid states before producing paired inference. Partial runs may produce clearly provisional descriptive counts only.

Recommended per-case provenance fields (the runner owns validation; statistics preserves them in `cases.csv`):

- Stable `case_id`/`case_key`, configuration, suite/task/init, environment/model seed rule, initial-state/input hash, protocol ID/hash, model/checkpoint identity, source hashes, physical GPU UUID, and gate membership.
- Selected attempt directory and attempt index; prior technical attempt directories/reasons; original terminal reason and task limit; valid task failures never retried.
- Consumed chunks/actions, candidate submissions/completions, unselected/expired candidates, wait/gap count and total/max duration, requested20Hz versus observed cadence, actual step limit and episode wall duration.
- Current and old observation age summaries with declared clock/denominator, consumed-candidate source step IDs and target chunk IDs; no synthetic previous-frame age. Separate first-result/fill timing from steady operation.
- Lifecycle/trace receipt paths, client/server exits, renderer attribution, epoch/generation/slot and two-slot checks, cleanup result, and per-candidate event/log paths.

Recommended candidate ledger fields: `candidate_id,target_chunk,seed,capture_step,capture_time,submit_time,complete_time,selected_time,selected,discard_reason,source_observation_id,old_kv_source_observation_id,slot,epoch,generation,frontiers,encoder_prepared_source_id`. Recommended action ledger fields: `executed_step,target_chunk,candidate_id,action_index_in_chunk,scheduled_time,actual_time,gap_before_action_ms`. Record the original chosen H10 and transformed actions for validation, while execution maps only indices0–4. These ledgers are provenance, not independent statistical samples.

`statistics.py --input normalized_cases.json --output statistics.json --csv-dir reports_dir` writes:

- `success_rates.csv`: overall/suite counts and Wilson intervals.
- `paired_comparisons.csv`: primary/ablation/exploratory labels, b/c, exact p, task-cluster and episode sensitivity CIs.
- `casewise_comparisons.csv`: all300 casewise pair outcomes for the three contrasts, with original directories and steps.
- `discordant_cases.csv`: the subset of those pairs with different success outcomes.
- `cases.csv`: the300 original selected outcomes and supplied provenance, retaining valid failures.

Only one primary comparison row has `confirmatory=true`; no SLO/noninferiority pass/fail field is created. JSON records input and source SHA. CSVs must be interpreted with JSON completion status. Closed-loop time or steps can be descriptive secondary outcomes, but divergent free trajectories are not matched-input GPU speed measurements.

## CPU checks and terminal handoff

`statistics.py --self-test --output statistics_self_test.json` uses no GPU/Torch import. Tests cover hand-computable exact McNemar values/symmetry, Wilson100/100, ratio weighting for unequal task sizes, percentiles, the100-case grid, exactly one primary contrast,300-cell completeness, failure sign/denominator, invalid odd-task init2, duplicate rejection and technical-vs-task failure distinction.

After the fixed matrix reaches terminal status, the root/runner artifact audit must establish protocol, seed-by-consumed-chunk, real observations, selected/discarded output semantics, two-slot safety and all process exits. This module then produces the prespecified statistics. It does not independently certify those runtime assertions merely because the normalized row says valid. No extra experiment or expanded gate is authorized by this statistical plan.
