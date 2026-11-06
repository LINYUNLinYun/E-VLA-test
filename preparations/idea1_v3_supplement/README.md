# Supplement implementation handoff

All commands are CPU preparation/dry-run unless `--launch` is explicit. GPU execution belongs to the execution owner. Existing v3 and safe-QK sources are unchanged. The frozen research grid is in `docs/experiments/idea1/idea1_v3_supplement_plan.md`.

## Science

E2 uses `python3 -m preparations.idea1_v3_supplement.cli prepare --attempt NEW --masks 'NAME_TO_SKIP_JSON' --shards 2 --launch`, then the same module `worker --attempt NEW --shard 0 --gpu GPU --launch`. Each explicit new mask has exactly four/five original 1-based layers, default retaining 1/2. The prefix keeps original-index KV shells; suffix dispatches to the actual original native scan. H10/D10/R5. This entrypoint is frozen while E2 runs.

E1-A uses `python3 -m preparations.idea1_v3_supplement.e1_cli prepare --attempt NEW --phase-a --shards 2 --launch`, then that module `worker --attempt NEW --shard 0 --gpu GPU --safety-gate PASSED_RECOVERY_GUARD --launch`. The recovery gate is checked in the launcher/evaluation child; the bounded kernel is installed in the actual fresh model server. Per-worker first-QK reference receipts remain mandatory. Startup/periodic are D10; only reject reasons can use D2. None disables periodic, not startup/rejection. Strict retains the old pending-gripper path; confirmed/off do not inherit it. No action clamping was added.

Both CLIs support `stop` (stop dispatch after current episode) and `resume`, each requiring explicit `--launch`. Valid scientific failures are completed data and skipped, not retried. E1 unresolved technical-invalid keys block automatic continuation.

## Selection and incremental preparation

Use `python3 -m preparations.idea1_v3_supplement.advance` with:

- `phase-b --source A --source OLD_E1_ROOT [--source ...] --output B --shards 2 --launch`: reads only init6 forty for A and old strict D2 .10/.15 P2. Outputs four new relaxed configs, frozen `B/selection.json`, source ranking/cost/filter receipts. Excludes calibration; old200 never ranks as 200. Waits for every relevant complete40.
- `phase-c --source B --parent-selection B/selection.json --output C --shards 2 --launch`: needs both relaxed40 per parent; strict plus best relaxed get rejection-only D2. Startup/periodic stay D10. B recursively references A; older references remain separate.
- `e1-select --source C --output E1_SELECTION.json --launch`: picks fastest observed >=90% and >=80% new candidates (dedup, no fill). Screening uses actual closed-loop ms/action, not causal ratios.
- `e2-select --source E2_SCREEN --output E2_SELECTION.json --launch`: within skip4/5, requires >=32/40, ranks success then ms/action then stable mask name. No eligible candidate means no expansion.
- `expand --source FIRST40_SOURCE --conditions CONDITION [CONDITION] --output NEW160 --shards 2 --launch`: at most two explicit selected conditions; requires actual complete40; only init7..10, references original40, task/inits adjacent and on the same static owner. Never reruns init6.

Run E1 expansion with `python3 -m preparations.idea1_v3_supplement.capture_worker worker --attempt NEW160 --shard 0 --gpu GPU --safety-gate PASSED_RECOVERY_GUARD --launch`, not the ordinary E1 worker. This wrapper saves up to25 existing real predict inputs/seed for init7/task0 per suite into that episode's `real_predict_inputs_first25.pkl` plus receipt, without additional episodes or predictions. Already valid episodes are not rerun to obtain a missing capture.

## Summaries and timing

`python3 -m preparations.idea1_v3_supplement.summary --attempt FINAL [--attempt ...] --reference-attempt OLD_ROOT --output SUMMARY` follows explicit new source attempts once. Only small result/infer records are read; no telemetry/video/checkpoint scan. It emits success/n, selection40 versus confirmation160, Wilson intervals, full route/reason/D/actual-R counts, complete policy-ms/actual-action costs, startup records and explicit config/source indices. Old init6 reference keys stay in a separate table. No pairing is guessed from condition names; this entrypoint does not draw XYZ or compare old latencies.

`python3 -m preparations.idea1_v3_supplement.bench --attempt NEW --condition CONDITION --inputs-root OLD_P0 --output BENCH --gpu GPU --launch` supports E2 with prompt outermost: five warm real observations then five measured five-point repeats (20 warm/100 measured total). Optional `--pi05-check` performs one bounded first-observation staged-native parity per suite, atol/rtol .02, separately labelled diagnostic. It does not replace or sum the fused total. Use `--reference pi05Full` for the same-revision native Full timing, no quality episode.

E1 timing **requires** four explicit `--captured-input FILE` (one continuous source sequence per suite), with `--safety-gate PASSED_RECOVERY_GUARD`. No old five-point reset benchmark is allowed for P8/none. Each suite warms at most5 observations, resets router only, then measures its actual <=25-request sequence once (<=20 warm/100 measured). Use the **same fixed four-file bank** for `--reference FullD10`, `--reference FullD2`, and at most two selected router configs. Do not concatenate short episodes or duplicate frames. Replay acknowledgements are virtual and its route distribution is not the real closed-loop distribution. Runtime setup is flagged; no blanket pure-hot claim.

`python3 -m preparations.idea1_v3_supplement.bench_compare --reference FULL_BENCH [--reference ...] --candidate CANDIDATE_BENCH [--candidate ...] --output RATIOS.csv` rejects differing input hashes/order, physical GPU UUID, thread environment, replay revision, model/backend or request counts. Primary ratio is reference ms/executed action divided by candidate ms/executed action; per-request ratio is separate. Both are explicitly same-input replay ratios, never actual closed-loop deployment speedups. Real closed-loop cost remains the science summary's sum(policy wall)/sum(actual actions).

## CPU checks

- `python3 -m preparations.idea1_v3_supplement.e1_checks`
- `python3 -m preparations.idea1_v3_supplement.post_checks`
- `python3 -m preparations.idea1_v3_supplement.checks --dispatch-only`
- `JAX_PLATFORM_NAME=cpu PYTHONPATH=$PWD:$PWD/policies/openpi/src policies/openpi/.venv/bin/python -m preparations.idea1_v3_supplement.checks`

The last uses a tiny18 native Gemma, not a checkpoint/GPU: exact suffix parity, scan trace and stable prefix/cache for four/five masks including consecutive skips. No new baseline quality evaluation is authorized by these commands.
