# Prespecified independent held-out statistics

The protocol is frozen in `heldout_statistics_protocol.json`. The script uses the one or two pairs already selected from discovery; it never chooses or replaces a pair using held-out results. No real held-out outcomes were read while implementing or testing this module.

```bash
python3 preparations/ease_insights_20261103/draft/heldout_statistics.py \
  --selection preparations/ease_insights_20261103/draft/replication/selection.json \
  --analysis <held-out-analyze-output> \
  --output <held-out-statistics-output>
```

Only `episodes.json` from the held-out analysis is consumed. It must contain exactly the prescribed conditions for init21–24; discovery cases, duplicate cases, and unprescribed conditions are rejected. Every pair has matching pure, foreign pure, matching FLASH, foreign FLASH and Full, each with ten tasks × four new initial states. Missing cases are listed for every prespecified pair and condition. Until the **entire** matrix across all selected pairs is complete, the report is `incomplete_not_tested`; no p-values, bootstrap intervals or replicated labels are emitted. Technical gaps remain missing rather than policy failures.

The primary outcome is pure-draft success, with the frozen direction matching source > foreign source on the same target/task/init. Let `b` be matching-only successes and `c` foreign-only successes. The exact one-sided conditional McNemar p-value is `P[Binomial(b+c, 0.5) >= b]`, with p=1 for zero discordance. Holm correction covers the whole frozen family of one or two pairs. A pair receives `replicated` only when the complete matrix is available, its success gap is positive, and its Holm-adjusted one-sided p≤0.05. A complete but non-significant result is `not_replicated`; it is not proof that the models are equivalent.

Both case-average and equal-weight task-average differences are reported. The paired-case bootstrap resamples complete paired outcomes; the task-cluster bootstrap resamples tasks while retaining all paired initial states within each sampled task. Both use20,000 deterministic resamples and percentile95% intervals. The task-level differences and denominator are retained for inspection.

The McNemar model treats cases as conditionally independent. Repeated initial states within a task can be correlated, so task-cluster intervals are mandatory robustness information. Only ten task clusters are available; neither the case-level test nor cluster bootstrap alone establishes generalization to unseen tasks. The confirmation tests new initial states of the same task set used during discovery.

Secondary results are descriptive paired effects and intervals, without extra uncorrected confirmatory p-values:

- FLASH success, per-case verified acceptance, and per-case rejection-triggered Full proportion.
- Complete policy and client RPC milliseconds per executed action for pure draft and FLASH, separately for all init21–24 and the fixed repeat-task init22–24 subset.
- Both success counts, actual episode-wall totals/means, action/request totals and the Full baseline.

Every statistical unit is a paired episode. Individual policy requests are never treated as independent samples. Acceptance may be undefined if a case has no verified rounds; those pairs are reported with their reduced denominator and are not imputed as zero. Differences always mean matching minus foreign: positive favors matching for success/acceptance; negative favors matching for rejection fraction and service cost.

A FLASH success decrease is not required: Full may preserve success while acceptance falls or service cost rises. Zero-action rejected requests still contribute to policy/RPC cost. Mean per-request latency is a diagnostic, not actual acceleration. GPU-stage cost is not end-to-end. The init22–24 subset is called repeat-task and is not guaranteed free of compilation/capture costs.

Outputs are `statistics.json` and a readable `README.md`, with hashes of the input episodes, frozen pair selection, statistics protocol and script. The module does not mutate selection, runtime, experiment files or the queue, and never starts GPU work.

Run `--self-test` for CPU-only fixtures covering direction, zero discordance, Holm correction, task clustering, incomplete matrices, both members of a two-pair family, and case-level secondary denominators. Synthetic fixtures are generated in memory and are not reported as experimental evidence.
