# Prospective statistics: four configurations, 200 paired episodes each

Version2; prepared before completion of the newly authorized768 episodes. The32 proposed pilot episodes already have known outcomes: this is a prospective analysis plan for the expansion, not a blinded preregistration of all800 observations. No GPU is launched by this statistics module.

## Fixed population, case identity and provenance gate

Four configurations: `fresh18`, `sm16_tail0`, `sm32_tail0`, `sm32_tail2`. Each has200 episodes: suites `libero_spatial/libero_object/libero_goal/libero_10` × tasks0–9 × init0–4. Pair on `(suite,task,init)` across configurations. Use the original seed7 model-seed derivation, complete π0.5 checkpoint, D10/H10/R5, unchanged precision, noise construction and control/client protocol. No next-frame lookahead in closed-loop.

Only the correct-EGL prior run `fresh_tail_20261025/closedloop_gpu2_eglmap` is a proposed reuse source. For each of the32 reused cases, audit exact suite/task/init/configuration, environment/model seed, model/checkpoint identity, D/H/R/precision, requested and actual SM allocation, tail policy, original client/control protocol, valid result, bootstrap/reset evidence, source hash compatibility and rendering guard. A matching label or an old `success=true` alone is insufficient. Record immutable original receipt paths/hashes and reuse decisions; do not rewrite old artifacts. Checkpoint path/file metadata do not constitute a full weight-content hash; report the actual evidence level.

An ordinary task failure is a valid observation with success=false and remains in every denominator. Technical failures and missing cases are separately listed and prevent a final800-case analysis. Never retry a valid task failure to replace it with success. A technical retry requires a documented excluded attempt and identity-preserving replacement. Duplicates must not silently choose the most favorable result. All four arms use exactly the same case grid. All32 reuse decisions are based on compatibility, not outcomes.

## Estimands and primary comparisons

For each arm report overall success count/200 and each suite count/50, with nominal95% Wilson intervals (z=1.959963984540054). These are episode-binomial descriptive intervals; five inits from the same task can be correlated, so do not portray Wilson intervals as a cluster-robust population guarantee.

Three primary overall paired comparisons are each candidate versus fresh18. Report:

- b = candidate succeeds / reference fails; c = candidate fails / reference succeeds.
- Both-success and both-failure counts, and paired success-rate difference Δ=(b−c)/n. Positive means candidate is better.
- Two-sided conditional exact McNemar p = min(1,2·P[Binomial(b+c,0.5)≤min(b,c)]); no mid-p, no asymptotic chi-square. If b+c=0, p=1.
- Holm-adjusted p across the three primary overall tests, alongside raw p. Remaining direct comparisons (including sm32_tail2 versus sm32_tail0 or sm16_tail0) and suite-level comparisons are exploratory; they do not create extra confirmatory findings.

“Exact” describes the conditional binomial calculation under independent discordances. Episode dependence within a task weakens that model's interpretation; do not infer task-population certainty solely from the exact p-value. p>0.05 is not evidence of no impact, equivalence, or noninferiority.

## Paired uncertainty and resampling

Use10,000 draws and seed20261025, fixed before analysis. Primary uncertainty summary resamples40 `(suite,task)` clusters globally with replacement; a sampled task contributes all five paired initializations. Take the mean of the40 task differences. This preserves pairing and all within-task outcomes; never resample the two arms independently. Global cluster sampling allows the represented suite counts to vary. Per-suite summaries resample the10 tasks in that suite.

Report a95% percentile interval using linear interpolation at index `(draws−1)·p` for p=.025/.975. As a sensitivity analysis, resample200 paired episodes (50 within suite) using seed20261026 and report that interval separately. It is not a replacement for the task-cluster result. Fixed ordering of suite/task/init and fixed RNG make outputs reproducible. Confidence intervals are not multiplicity-adjusted simultaneous bounds.

Only40 task clusters (10 per suite) are available. Sparse discordances can make either bootstrap unstable or degenerate. In particular, zero observed discordances gives [0,0] from empirical bootstrap; **this does not prove zero true difference**. Report the degeneracy. As an explicitly iid-episode diagnostic only, zero discordances in200 gives a two-sided95% binomial upper discordance probability `1−0.025^(1/200)≈1.83%`; task clustering can invalidate that iid interpretation. Do not use the bootstrap's zero width to claim a1pp noninferiority result.

## Prespecified new192 sensitivity, separating the known pilot

The full200-per-arm table is the main requested summary, but candidate selection used the earlier pilot. Add a clearly labeled secondary analysis excluding exactly `(task in {0,5}, init=0)` in each of four suites: eight cases per arm,32 rows total. Identify exclusions by suite/task/init, not a potentially inconsistent `source_kind` field. Link and hash `reuse32.json` as the provenance evidence; verify its32-case mapping independently at final audit.

The strict800-cell completeness/validity gate is unchanged. Only after it passes, analyze the retained192 cases per arm (48 per suite), using the same paired tests, Wilson summaries, fixed seed/draws, task-cluster and episode sensitivity methods. This is768 new outcomes, not a second independent experiment or a blinded confirmation of the already-known pilot. Full and new-only results are correlated and must not be presented as two replications.

There are still40 task clusters: eight tasks retain four initializations and32 retain five. Resample40 tasks and carry every retained initialization of each sampled task; each bootstrap replicate is `sum(paired success differences) / sum(retained episode counts)`. Do not take an unweighted mean of unequal-size task means, which would change the episode-average estimand. Per-suite sampling uses10 tasks with two four-init and eight five-init clusters. Episode sensitivity resamples192 paired cases, or48 within a suite.

All32 pilot outcomes are currently recorded as success and baseline pairing is identical. If final provenance confirms this, b/c and exact McNemar p are unchanged when pilot pairs are removed; the full200 paired difference is0.96 times the new192 difference. Wilson denominators and bootstrap precision still change. Report actual counts and intervals, not simply a rescaled point estimate. No additional noninferiority margin or quality pass threshold is introduced.

## What200 episodes can resolve

One episode equals0.5 percentage points overall and2pp within a suite. If all200 succeed, the Wilson lower endpoint is about98.12%; even200/200 is not proof of perfect success. With only five discordant pairs all favoring one arm, exact two-sided McNemar p=.0625; with six it is.03125 before multiplicity adjustment. Thus a difference of1–2pp with very few discordances commonly remains inconclusive, and even a3pp one-sided observed discrepancy may not survive three-test correction.

For orientation only, an iid paired normal approximation has standard error approximately sqrt(q/200) near Δ=0, where q is the discordant fraction. Its95% half-width is approximately3.1pp at q=.05,4.4pp at q=.10,6.2pp at q=.20, and9.8pp at q=.50. These are illustrations, not prospective power guarantees; unknown discordance structure and task correlation determine actual precision. Forty task clusters can have substantially less information than200 independent episodes. Per-suite analyses have particularly low precision.

The report may describe whether interval lower bounds are above −1/−2/−5pp, but these are transparent descriptive thresholds only. The user has not approved any noninferiority margin or SLO, so no threshold becomes a pass/fail quality criterion. Always show the actual interval, discordances, and degeneracy caveats. The implementation leaves threshold flags null for a degenerate bootstrap interval rather than suggesting that its zero width excludes harm. Failure to exclude deterioration is not proof of deterioration; failure to reject equality is not proof of no deterioration.

## Data and executable interface

`statistics.py --input cases.json --output statistics.json` consumes a JSON list or an object with a `cases` list. Every row needs `configuration` (or `label`/`arm`), `suite`, `task`, `init`, `status="valid"`, and a genuine JSON boolean `success`. Extra provenance fields remain the responsibility of the runner and final artifact audit. The statistics function checks the exact800-cell grid, duplicate/unknown cases, and invalid outcomes. Incomplete data may show clearly marked provisional descriptive counts but produce no final paired comparisons.

`statistics.py --self-test --output statistics_self_test.json` runs CPU-only hand-checkable checks: exact McNemar values/symmetry, Wilson200/200, quantile interpolation, Holm monotonicity, degenerate bootstrap, complete-grid shape, failure sign/denominator, missing cases, duplicates and strict boolean outcomes. It imports no Torch or GPU library. Source and input hashes are stored in output.

After the runner reaches terminal state, perform one independent CPU audit of all800 case receipts, provenance/reuse compatibility, original seeds, bootstrap/reset/request identities, task failures, valid renderer mapping, source stability and case completeness. Then run the fixed analysis once and reconcile counts and paired identities with the final report. Free closed-loop action trajectories and work time are descriptive, not matched-input GPU speed benchmarks. No claim of complete LIBERO generalization follows from this selected grid alone.
