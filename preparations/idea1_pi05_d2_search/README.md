# π0.5 skip-n + actual D2 search

Isolated implementation; old v3/supplement sources/results remain frozen. The initial `cli.py`, `policy.py`, `server.py`, `checks.py` are frozen while their four workers run. Actual GPU execution belongs only to the execution owner.

`policy.py` independently constructs native `create_trained_policy(..., sample_kwargs={'num_steps':2})`, asserts the resulting sampler kwargs, and preserves H10/R5. The original sampler evaluates t=1,.5 and ends at0. Prefix hidden updates skip only explicit original layers3..18 (0..16 skips); all original-index KV shells remain. The actual original native scanned AE suffix executes all18 layers. No old D10 policy or hard-coded D10 stage profiler is relabelled as D2.

## Existing science attempts

Initial6: FullD2[], n2[3,9], n3[3,9,11], n4[3,9,11,15], n5[3,9,11,14,17], n6a[3,6,9,11,14,17]. Each condition has40 init6 keys, four static shards. FullD2 is a new quality reference, not a rerun of old FullD10.

The fixed followup13 was already prepared using the original `cli prepare`:

`policies/realtime-vla-flash/results/idea1_pi05_d2_search/pi05_d2_followup_init6_20261020T0544Z`

**Do not create a second attempt for these same13 masks or edit that frozen manifest.** It has no `source_attempts` link to Initial6. Consequently odd/select/expand and summary commands must explicitly receive both roots (`--source INITIAL6 --source FOLLOWUP13`, or `--attempt INITIAL6 --attempt FOLLOWUP13`). Do not rely on followup-only recursion. The collector deduplicates explicitly repeated/recursive source roots by resolved path.

The existing worker command is:

```sh
python3 -m preparations.idea1_pi05_d2_search.cli worker --attempt ATTEMPT --shard OWNER --gpu GPU --launch
```

`stop` stops new dispatch after the active episode, and `resume` clears STOP; both require explicit `--launch`. Valid scientific failures are valid data and skipped on continuation. Unresolved technical-invalid keys block automatic retry. No GPU reset, external process termination or baseline D10 quality rerun is included.

## Remaining preparation/selection

`advance` defaults to dry-run, and requires explicit `--launch` to write immutable new outputs.

```sh
python3 -m preparations.idea1_pi05_d2_search.advance screen --source INITIAL6 --source FOLLOWUP13 --masks 'FROZEN_NEW_NAME_TO_SKIP_JSON' --output NEW_ATTEMPT --shards 4 --launch
python3 -m preparations.idea1_pi05_d2_search.advance odd --source INITIAL6 --source FOLLOWUP13 --output ODD_ATTEMPT --shards 4 --launch
python3 -m preparations.idea1_pi05_d2_search.advance select --source INITIAL6 --source FOLLOWUP13 --source ODD_ATTEMPT --output SELECTION.json --launch
```

Only use `screen` for a genuinely new authorized mask, never the already prepared13. Across all supplied source manifests it rejects repeated semantic skip lists (even renamed), conflicting configs and >20 total conditions. Odd waits for actual complete40 for all fixed19. If highest passing n is even2..14, it adds the first still-retained layer in `[6,15,4,7,12,5,16,8,10,13,18,17,14,11,9,3]` to the best success/cost/name parent at that n. Otherwise it writes only `odd_decision.json`, no manifest/episodes; omit that no-manifest directory from later `--source`. Select automatically rejects missing required odd data.

Selection uses each condition's common init6 forty, never its later200. Among **n>0 candidates** with overall>=26/40 (not per-suite), select: highest passing n's highest success/cost/name; globally lowest complete policy-ms/actual-action with success/name tie-break; >=30/40 lowest-cost backup. Deduplicate, never fill a fourth. FullD2 is separate and unconditionally included for new200 quality, even if it is faster than all pruned candidates or itself below65%.

```sh
python3 -m preparations.idea1_pi05_d2_search.advance expand --source INITIAL6 --source FOLLOWUP13 --source ODD_ATTEMPT --conditions pi05D2_FullD2 SELECTED_ID ... --output NEW160 --shards 4 --launch
```

Use `SELECTION.json`'s explicit `expansion_conditions`. At most FullD2+3 candidates; actual complete40 required; pruned candidates must pass overall26/40. Only init7..10 keys are created, with same task's inits adjacent on one owner. Original init6 is referenced, never repeated. Existing expanded keys cannot be reproduced in another new attempt. Budget: at most20×40 screens plus4×160, <=1440 episodes.

## CPU summaries and bounded timing

```sh
python3 -m preparations.idea1_pi05_d2_search.summary --attempt INITIAL6 --attempt FOLLOWUP13 --attempt ODD_ATTEMPT --attempt NEW160 --old-full-d10 EXISTING_OLD_FULL_ROOT --output SUMMARY
```

Only explicit small result/infer/event records are read. Outputs include each suite and overall success/n, selection40 versus additional160, observed overall>=65% labels/Wilson intervals, individual scientific failures, actual native-policy calls (not fallback), full request costs/actual actions, R distributions, exact config/runtime D2 protocol and startup/technical-invalid records. New main denominators never include old FullD10. The old FullD10 reference has separate key/suite/overall tables, including common init6 and existing-all coverage. At200, reclassify overall>=65% from actual combined200; a suite may be below65%. Sample thresholds are not population lower-bound guarantees.

At most five serial same-card timing configurations: FullD10, FullD2, and <=3 selected pruned D2 candidates. No timing all20 masks. FullD10 is created only as `timing_reference=True`, with actual num_steps10; D2 references/candidates use actual num_steps2. No stage profiler or split AE timing is used.

```sh
python3 -m preparations.idea1_pi05_d2_search.bench --reference FullD10 --inputs-root EXISTING_P0 --output BENCH_D10 --gpu GPU --launch
python3 -m preparations.idea1_pi05_d2_search.bench --reference FullD2 --inputs-root EXISTING_P0 --output BENCH_D2 --gpu GPU --launch
python3 -m preparations.idea1_pi05_d2_search.bench --attempt SCREEN --condition SELECTED_ID --inputs-root EXISTING_P0 --output BENCH_CANDIDATE --gpu GPU --launch
python3 -m preparations.idea1_pi05_d2_search.bench_compare --reference BENCH_D10 --reference BENCH_D2 --candidate BENCH_CANDIDATE --output REPLAY_RATIOS.csv
```

All five use the same20 real saved observations/seeds, same GPU UUID, threads and native revision. Prompt is outermost: five exact-shape warm observations, then five measured repeats without reconstructing the model/prompt between repeats (20warm+100measure/config). Compilation may remain unobserved: not a blanket pure-hot claim. Comparison rejects different input hashes/order, card UUID, thread environment, revision/backend and request counts. Main replay ratio uses total request-ms/actually proposed virtual execution lengths; per-request ratio is separate. This is fixed-input replay, not new physical history or measured closed-loop deployment speedup. Actual closed-loop sum(policy wall)/sum(actual actions) is reported separately, with shared CPU/card-load limitations.

## CPU validation

```sh
python3 -m preparations.idea1_pi05_d2_search.checks --dispatch-only
JAX_PLATFORM_NAME=cpu PYTHONPATH=$PWD:$PWD/policies/openpi/src policies/openpi/.venv/bin/python -m preparations.idea1_pi05_d2_search.checks
python3 -m preparations.idea1_pi05_d2_search.post_checks
```

The CPU test actually invokes original `Pi0.sample_actions` with a tiny fake model and records runtime callbacks `[1,.5]` (two suffix evaluations, dt=-.5, endpoint0). Tiny18 Gemma checks skip0/6/16 original KV shells and exact full native scanned-suffix parity/scan trace. Postfixtures cover overall65-only gating, all three selection categories even when FullD2 is fastest, odd priority, semantic duplicate rejection, new-init-only keys and weighted action denominators. No checkpoint/GPU/episode is loaded by these tests.
