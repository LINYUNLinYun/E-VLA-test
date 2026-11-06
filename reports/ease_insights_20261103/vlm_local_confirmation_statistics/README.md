# Independent local-search confirmation statistics

`rebuild.py` is a standalone CPU reconstruction for the frozen 600-case heldout matrix: `full_d10`, `jac_anchor`, `local_long`; four suites × ten tasks × init39–43 per condition. It imports Python standard-library modules and NumPy only. It never imports or calls the scientific analysis helpers, loads models/environments, uses a GPU, or writes case data. All output is restricted to this directory.

**Prepared and tested with synthetic data only. The active confirmation cases have not been scanned or analyzed by this entrypoint.** `contract_result.json` records the synthetic checks; no real-data `audit.json` has been created during preparation.

The pinned manifest content hash is `80176d9f4b01e68abb022345bd6dc1a7bfe4e5aef5d69d13257801ca1eb6bdc0`, its file SHA256 is `3c4b8bf35ecf768f129c00721ce0fd0e49ef2c209f7b9995ed0a1a128796365a`, and the selection file SHA256 is `f6314a52f71ab0a127afb7bae0fd1adb201c934d1634634567217490a4d1e9ed`. The independent 600 acceptance script is pinned to SHA256 `a9ee976a07997a20cb56d6a0d77df315d7956292fc1d78e88fd6b0a7f1f06da9`; the frozen main analysis script is pinned to `525800ce944944a73d0c5d3fec52c8966c4a229dc801d15d3c2c73de38d5572b`.

## Run only after completion

First finish the independent 600 acceptance at `reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_confirmation/audit.json` with `status: pass`, and the supervisor's complete `results/ease_insights_20261103/vlm_local_search_v1/confirmation/analysis/analysis.json`. Record their final SHA256 values:

```bash
sha256sum reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_confirmation/audit.json results/ease_insights_20261103/vlm_local_search_v1/confirmation/analysis/analysis.json
```

Pass those exact final values, replacing the two uppercase placeholders:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  policies/realtime-vla-flash/.venv/bin/python \
  reports/ease_insights_20261103/vlm_local_confirmation_statistics/rebuild.py \
  --acceptance-sha256 ACCEPTANCE_REPORT_SHA256 \
  --supervisor-sha256 SUPERVISOR_ANALYSIS_SHA256 \
  --output reports/ease_insights_20261103/vlm_local_confirmation_statistics/audit.json
```

These explicit SHA arguments freeze the future report identities; no final report SHA is guessed from a partial run. The script refuses a different root or matrix and does not offer a partial-analysis mode. Exit code 0 requires complete accepted data and zero statistical differences. A scientific conclusion of `not_reproduced` can still produce an audit `status: pass` when the independently reconstructed result agrees with the supervisor.

Before reading any raw outcomes, the entrypoint verifies the acceptance status, complete 600/600 coverage, zero missing/invalid/unexpected/request issues, pairing and serving gates, pinned audit implementation, frozen manifest, all 46 original source files, six analysis sources, nine local runtime sources, checkpoint identity artifacts, protocol/design evidence, selection, and every acceptance-declared input-source hash. It also requires the complete supervisor report to name the same manifest, selection, analysis source, and exact 600 receipt paths/hashes.

Each original `episode_receipt.json` is read once as bytes, hashed against both reports, and parsed from those same bytes. Its `result.json` is read and hashed against `receipt.artifacts["result.json"]`. Both identities and success types must agree, and both successes must be actual JSON booleans. Every parsed receipt/result path and current SHA is included in `input_evidence.episodes`. There is no CSV input. Original 600 acceptance owns the broad episode artifact/request validation; this reconstruction rechecks the two raw files needed for success statistics and preserves their acceptance hash chain. Reports and manifest are rehashed after the calculation to detect changes during the run.

## Statistical contract

- Rebuild 135 condition summary rows: overall, four suites, and 40 suite/task combinations, each for three conditions; compare success counts, denominators and rates.
- Rebuild the frozen `libero_object: jac_anchor > local_long` and `libero_10: local_long > jac_anchor` directions, all 50 paired discordances, and ten five-case task differences for each.
- Enumerate all 1,024 task sign flips, including ties and zero-difference tasks. Apply Holm across exactly two one-sided task hypotheses. Compute one-sided exact conditional case McNemar and its separate Holm2 sensitivity adjustment, plus the descriptive two-sided paired values.
- Generate 20,000 paired-case and 20,000 whole-task bootstrap samples per direction. Two fresh independent `Generator(PCG64(20261104))` instances select replacement indices. Case order is task00 init39–43, then task01, … task09. Task bootstrap samples the ten float64 five-case means in task order. Intervals use `numpy.quantile([.025, .975], method="linear")` with no rounding. The synthetic checks separately compare index sampling with the frozen `default_rng(seed).choice(..., replace=True)` API formulation; no original helper is called.
- Rebuild all ten leave-one-task-out means. Rebuild candidate usability through exact integer comparisons implementing `success >= .75` and `success >= Full - .15` over 200 cases.
- Independently decide each direction using a positive difference, task Holm2 p < .05, task-cluster lower CI > 0, and every task LOO > 0. Both directions plus both usable paths determine `complementarity_reproduced`; otherwise `not_reproduced`. The paired-case interval is reported but is not a local-confirmation verdict gate.

`independent_statistics` contains this independent result; `differences` lists field paths and both values. Every independent statistical field must exist in the supervisor report. Integer/Boolean/string comparisons are exact; finite floating values use a prespecified absolute tolerance of `1e-12` for floating arithmetic. Timing, resource-contention records, preparation costs and prose are outside this success-statistics reconstruction. The Python/NumPy versions and explicit PRNG/quantile contract are recorded. If sources, identities or completeness fail, the output is non-PASS; no matching-by-parameter-adjustment is performed.

## Synthetic CPU check

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  policies/realtime-vla-flash/.venv/bin/python \
  reports/ease_insights_20261103/vlm_local_confirmation_statistics/test_contract.py
```

The checks exercise complete/incomplete acceptance gates, refusal before outcome reads, exact tests with ties, Holm ordering, nontrivial bootstrap API equivalence and fixed intervals, 135-row counts and both verdicts, exact eligibility boundaries, supervisor receipt-set/hash/seed mismatch, and 600 synthetic receipt/result hash bindings with tamper and non-Boolean rejection. Temporary synthetic files are created only under this directory and removed afterward. `contract_result.json` records the check count, status and source SHA values.
