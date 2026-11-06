# Experiment 10 — phase-sensitive draft search

Authoritative design: `/root/nfs/embodied/benchmark/idea1_experiment_plan.md`.
Results: `results/experiment10_phase_draft_20261009/`.

## Running stages

The four-suite real-observation collection and VLM numerical/intervention probes
were launched on 2026-10-09. Collection reuses one environment across a task's
five inits, matching the original client. A first attempt using a fresh environment
per init failed strict replay checks; failed attempts remain in the event log and
are not merged into completed suite manifests. No replay tolerance was relaxed.

Each observation probe evaluates 18 interventions and a shared full reference
under three noise seeds. The prefix is noise-independent, so activation statistics
are collected once per observation, not counted as three independent observations.
The first pair of ordinary/recorded baselines and repeated post-intervention
baseline must be bitwise equal. Layer18 is explicitly KV-only, not a deletion prior.

Run initial stages (four GPUs maximum, dry-run without `--launch`):

    .venv/bin/python -m scripts.experiment10.run_pipeline
    .venv/bin/python -m scripts.experiment10.run_pipeline --launch

`run_followup` is a separately launched supervisor which checks actual upstream
PIDs and queues complete-interval D1/D2/D3/D4/D5 and V16/V13 shadow jobs when a
suite finishes. It then generates task-clustered phase contrasts and search-only
priors. GPU idle polling is 300 seconds; a new owned job-completion event permits
immediate dispatch. CPU checks of child completion are not GPU-state polling.

    .venv/bin/python -m scripts.experiment10.run_followup --upstream-pid PID

## Verification and boundaries

    .venv/bin/python -m unittest scripts.experiment10.test_protocol -v
    .venv/bin/python -m py_compile scripts/experiment10/*.py

The heuristic rules and beam/Pareto/local-exchange search now have a real-model
multi-GPU evaluation pool. `queue_search --upstream-pid PID` waits for the
four-suite search-only prior before launching the bounded search; being queued
does not mean candidate evaluation has started. The arbitrary-mask D1..D10 backend requires its D10 parity gate
before returning results, and each schedule owns separate graph-captured tensors.
Any fresh-instance D10 differences in shadow probes are explicitly recorded and
are not called pruning effects; shadow candidate/reference comparisons use the
same instance, inputs and noise.

Current status: the 800 initial validated closed-loop episodes and their D1/D2/D4
trajectory figures and 51 synchronized videos are complete. Search-only
Attention/FFN interventions and token/channel numerical summaries are complete
for all four suites. The bounded global and phase-specific candidate search,
refinement, validation and freezing are complete. Short-time branches are complete
for all four suites. Recover_D3_H8 and Recover_V16_D10_H8 each passed all four
50-episode/full-video validators. The new Full18-D3 and Full18-D5 standard-suite
runs also passed all eight validators (400 episodes total): 186/200 and 188/200
successes, respectively. A dedicated isolated timing queue was started at
2026-10-09 15:42 UTC and completed on all four GPUs. Each suite measured four
validation task-6/7 observations (one per available phase), three noise seeds and
five repeats for each condition; all eight conditions were run sequentially on the
same GPU within each suite. Per-suite phase-equal median prepare+inference times
were 34.1–34.9 ms for Full18-D10, 26.4–27.0 ms for D3, 28.6–29.2 ms for D5,
32.0–32.7 ms for V16-D10 and 28.8–29.5 ms for V13-D10. These are isolated policy
measurements, not environment/client end-to-end latency or task-level confidence
intervals. An online D3-H8 verify-then-execute screen started at 2026-10-09
15:50 UTC across all four suites (tasks 8-9, init0-4; 40 episodes total). Its
translation/rotation thresholds are frozen from validation tasks 6-7 at the
task-equal 95th percentile. Each request compares D3 and Full18-D10 on the same
observation/noise, accepts only the longest matching prefix up to eight actions,
requires gripper-sign agreement, and falls back to 12 Full18 actions if no draft
action passes. This is reference agreement, not a semantic safety guarantee, and
verification cost may eliminate any speedup. The phase-interaction table has 408
comparisons and none passes Holm-adjusted p<0.05. In addition, the frozen
search-only mask is identical across all four phases and equals the static L2
challenge mask `[2,4,6,8,10,12]`; a separate phase-routing rollout would not test a
different mask, so it is not duplicated as a claimed phase-specific candidate.

An executed-command Kalman-gate pilot started at 2026-10-09 16:06 UTC across all
four GPUs. It compares the Full18-D3 candidate with validation-calibrated Q/R and
50/75/90/95% command-residual thresholds, using tasks8-9 and init0-4 (40 episodes
per threshold). Q50 has passed all four 10-episode/full-video validators: 34/40
successes versus 35/40 for paired Full18-D10. Across its 705 requests, 188 (26.7%)
accepted a D3 prefix (mean accepted length 10.27) and 517 fell back to Full18; most
gate rejections were command-prediction disagreements. Mean per-request
`sample_actions_ms` was 78.8 ms for Q50 (30.9 ms on accepted drafts, 96.2 ms on
fallbacks), versus 65.4 ms for paired Full18-D10 requests (baseline median 38.3 ms).
Thus Q50 currently shows no net inference-speed benefit. This small exploratory
split is not a significance or safety claim. Q75 has also passed all validators:
34/40 successes, 410/760 (54.0%) accepted D3 prefixes, and mean request latency
67.3 ms versus the same Full18-D10 mean of 65.4 ms. Thus Q75 improves draft use but
still shows no net mean-latency benefit. Q90 has passed all four validators too:
31/40 successes and a 69.7% draft-prefix ratio, with a 62.0 ms per-request mean.
However, its 812 requests sum to 50.3 s of policy inference versus 44.6 s across
682 paired Full18-D10 requests, and success is four episodes lower. Q95 has also
passed all four validators: 32/40 successes, 72.6% accepted draft calls, and 49.9 s
of policy inference. The client sends only actually executed command
history and observation timestamps. Fewer than four executed commands, a
gate failure, or an observation older than the frozen 100 ms cutoff selects
Full18-D10; otherwise it executes the accepted contiguous D3 prefix, up to 12
actions. This is a synchronous closed-loop screen, not evidence of asynchronous
overlap or physical safety. Q50 started before the client route-metric flag fix;
its raw `used_full_fallback` metadata is authoritative. The fix is loaded by later
workers. `kalman_gate_status.json` is authoritative for queue progress, and the
raw-route summarizer is `summarize_kalman_gate.py`.

Completed paired gate screen (tasks8-9/init0-4; same 40 baseline episodes each):

| Threshold | Success | Accepted D3 calls | Total policy inference (ms) |
|---|---:|---:|---:|
| Full18-D10 | 35/40 | — | 44,609 |
| Q50 | 34/40 | 26.7% | 55,544 |
| Q75 | 34/40 | 53.9% | 51,146 |
| Q90 | 31/40 | 69.7% | 50,325 |
| Q95 | 32/40 | 72.6% | 49,904 |

All four gate settings were slower in summed policy inference time and lower in
successes on this small paired split. This is a negative screen for this particular
history predictor/gate, not proof that adaptive execution is generally ineffective.
Raw per-request, paired-outcome and latency data are in
`results/experiment10_phase_draft_20261009/kalman_gate/`.

The D3 recovery-horizon screen started at 2026-10-09 16:20 UTC. It compares prefix
lengths h=1/2/4/8/12 followed by Full18-D10 on a fresh observation, using validation
tasks6-7 and init0-4 (40 episodes per horizon; 200 total). The initial availability
check allocated GPUs0-3 immediately. `recovery_horizon_screen_status.json` tracks 20
suite-runs; owned-worker completion dispatches immediately, and if externally
occupied cards remain, the next availability check is after 300 seconds.
All five horizons have now passed all four suite validators. Paired results against
the same 40 Full18-D10 episodes are:

| D3 prefix h | Success | Full18-D10 baseline | Policy inference total | Change vs baseline |
|---:|---:|---:|---:|---:|
| 1 | 39/40 | 39/40 | 42.45 s | +2.6% |
| 2 | 38/40 | 39/40 | 44.18 s | +6.8% |
| 4 | 35/40 | 39/40 | 42.66 s | +3.1% |
| 8 | 40/40 | 39/40 | 40.83 s | -1.3% |
| 12 | 38/40 | 39/40 | 40.69 s | -1.6% |

H8 is the best observed joint outcome on this small split, but its 0.53 s aggregate
inference reduction and one-episode success difference are too small to establish a
reliable speed or quality gain. H12 is marginally faster in aggregate but has one
fewer success than baseline. Inference totals sum policy `sample_actions_ms`, not
simulator wall time. The screen uses only two validation tasks and five inits per
task; those tasks were also used in command-gate calibration. The recovery is
synchronous (a fresh observation followed by Full18-D10), so this does not measure
asynchronous overlap. Raw episode/request data and the paired summary are under
`results/experiment10_phase_draft_20261009/recovery_horizon/`.
For every horizon and suite, selected paired EEF XYZ plots (PNG/PDF), source CSVs,
and a synchronized H.264 video are under
`results/experiment10_phase_draft_20261009/closed_loop_analysis/xyz/RecoverD3_H{1,2,4,8,12}_T67/`;
the generated videos passed frame-count and full ffmpeg decode validation.

A two-GPU-per-suite parallel verifier has completed on paired tasks8-9/init0-4.
Each observation was sent concurrently to D3 and Full18-D10 servers on separate
GPUs; the client executes a D3 prefix only when the same-observation comparison
passes the frozen VerifyD3_H8 thresholds, otherwise it executes Full18. This is an
actual concurrent inference path, unlike the earlier synchronous screens. All four
suites passed episode/video validation. It achieved 35/40 successes, matching both
the paired Full18-D10 baseline and the previous synchronous verifier. All 1,052
parallel decisions exactly matched the synchronous verifier's route and accepted
prefix, and all 1,052 executed action prefixes matched exactly (maximum absolute
action difference 0); 679 (64.5%) used D3 and 373 fell back to Full18. Paired
outcomes were 32 both-success, three parallel-only, and three baseline-only.

| Suite | Parallel | Sync verify | Full18-D10 | Parallel request mean* |
|---|---:|---:|---:|---:|
| Object | 10/10 | 10/10 | 9/10 | 40.0 ms |
| Spatial | 10/10 | 10/10 | 9/10 | 40.6 ms |
| Goal | 10/10 | 10/10 | 10/10 | 53.6 ms |
| LIBERO-10 | 5/10 | 5/10 | 7/10 | 44.5 ms |

*Latency is client websocket roundtrip after excluding the first request per suite
to avoid cold-start/import effects. Overall parallel mean/median/p95 were 44.5 /
40.3 / 41.8 ms (p99 44.2 ms), versus 71.2 ms mean for the earlier synchronous
verifier: 37.5% lower mean request latency on this split. Two task-9 first-inference
re-capture outliers exceed 2 s in both methods (parallel max 2.45 s, synchronous
max 3.06 s); the steady-state p95 remains 41.8 ms. The summed CUDA time of both
parallel branches was 67.63 s versus 65.84 s in the synchronous run, so the latency
reduction comes from concurrent use of two GPUs, not less model computation. This
does not establish higher system throughput or an end-to-end wall-clock gain; each
rollout uses two GPUs. LIBERO-10 is still two successes below baseline, and the
40-episode sample is exploratory. Full paired data are under
`results/experiment10_phase_draft_20261009/parallel_verify/`; XYZ plots and four
decoded H.264 synchronized videos are under
`results/experiment10_phase_draft_20261009/closed_loop_analysis/xyz/ParallelVerifyD3_H8_T89/`.

The parallel-verifier horizon screen (H4 and H12) completed all eight suite-runs
on tasks8-9/init0-4, with all 80 episodes and videos validated. The aggregate
comparison is:

| D3 horizon | Success | Full18 success | Inference calls | D3-prefix ratio | Mean per-episode policy wait | vs Full18 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 34/40 | 35/40 | 1,423 | 69.4% | 1,553 ms | 1.84× |
| 8 | 35/40 | 35/40 | 1,052 | 64.5% | 1,166 ms | 1.38× |
| 12 | 35/40 | 35/40 | 918 | 61.5% | 1,030 ms | 1.22× |

H12 has the best observed rollout-level inference-wait total of the three horizons,
but still exceeds the single-branch Full18 baseline's 845 ms/episode by 22% because
every verification request computes both branches. H12's suite successes were
Object 10/10, Spatial 9/10, Goal 9/10 and LIBERO-10 7/10; the aggregate match to
baseline masks one lower success each in Spatial and Goal and two candidate-only
successes in LIBERO-10. H4 also averaged 43.8 ms per request, close to H8/H12, but
its shorter prefix led to more replans and the largest per-episode total. These are
40-episode exploratory screens, not non-inferiority claims.

The initial availability scan found one free pair (0,2); after an owned worker
finished, an event-triggered availability check found all four cards free and
launched two suite-runs at once. While all four cards were owned, the scheduler did
not query GPU state. Status is in `parallel_verify_horizon_screen_status.json`;
paired JSON/CSV summaries are under
`results/experiment10_phase_draft_20261009/parallel_verify_horizons/`. H4/H12
trajectory comparisons and synchronized videos are under
`results/experiment10_phase_draft_20261009/closed_loop_analysis/xyz/ParallelVerifyD3_H4_T89/`
and `.../ParallelVerifyD3_H12_T89/`.

The D5 candidate extension has also completed (H8/H12; 80 episodes, all suites
validated). D5 raised same-observation prefix acceptance above 90%, but did not
recover task success: H8 was 33/40, with 1,099 requests and 1,255 ms mean
per-episode policy wait (1.49x Full18); H12 was 33/40, with 779 requests and 900 ms
per-episode policy wait (1.06x Full18). H12's suite success was 10/10 Object, 9/10
Spatial, 8/10 Goal, and 6/10 LIBERO-10. This shows that high command agreement alone
does not protect task success, particularly with a long open-loop prefix. The D5
paired summaries are under `results/experiment10_phase_draft_20261009/parallel_verify_horizons/`;
paired XYZ plots and synchronized videos are under
`results/experiment10_phase_draft_20261009/closed_loop_analysis/xyz/ParallelVerifyD5_H8_T89/`
and `.../ParallelVerifyD5_H12_T89/`.

The D5-H4 extension has now passed all four suite validators as well: 33/40
successes, 1,824 requests, 90.6% accepted D5 prefixes, and 1,937 ms mean
per-episode policy wait (2.29x Full18). Its success matches D5-H8/H12, while the
shorter horizon sharply increases requests and total wait; this is evidence against
H4 for this verifier configuration on the current split.

A D7 H8/H12 screen completed on the same paired tasks8-9/init0-4 split, with all
eight suite-runs validated. D7's same-observation agreement with Full18-D10 was
very high (H8: 99.4%; H12: 98.9%), but this did not preserve baseline success:
H8 reached 31/40 and H12 32/40 versus 35/40 for Full18-D10. H12 used 702 D7
prefixes and eight fallbacks across 710 requests, with mean per-episode policy wait
843 ms, essentially break-even against Full18's 845 ms (0.998x). The 2 ms aggregate
difference is too small to support a speedup claim, while the three-episode success
gap remains. H8 made 1,103 requests and averaged 1,251 ms/episode (1.48x baseline).
Suite successes for H12 were Object 9/10, Spatial 9/10, Goal 9/10, and LIBERO-10
5/10; H8 was 9/10, 10/10, 8/10, and 4/10, respectively. Thus D7-H12 has the best
rollout wait among the tested D3/D5/D7 horizons, but not a quality-preserving gain.
These are small exploratory paired screens, not non-inferiority claims. Summaries
are under `results/experiment10_phase_draft_20261009/parallel_verify_horizons/`;
paired XYZ plots/videos are under
`results/experiment10_phase_draft_20261009/closed_loop_analysis/xyz/ParallelVerifyD7_H{8,12}_T89/`.
In the selected worst-p95 LIBERO-10 H12 trajectory pair (diagnostic, not
representative sampling), the same-step XYZ deviation stays small through roughly
step 220, then grows during `goal_alignment` around step 252 and exceeds 1 m by
step 384 in the subsequent transport segment. This is consistent with a late
trajectory divergence, but one selected episode cannot establish the general
failure mechanism.

The H4 extension is complete for both D5 and D7 (another 80 paired episodes; all
eight suite-runs validated). D7-H4 reached 32/40 successes, the same as D7-H12 but
three below Full18, while using 2,110 requests with 99.3% D7-prefix acceptance and
2,278 ms mean per-episode policy wait (2.70x Full18). Thus shortening D7's horizon
from 12 to 4 did not recover success and substantially increased wait. Together
with D5-H4's 33/40 at 2.29x wait, this closes the H4/H8/H12 exploratory matrix for
D3/D5/D7; among the tested parallel settings, D7-H12 is the fastest, but still has
three fewer successes than baseline and no reliable measured speed advantage over
Full18. H4 summaries are under
`results/experiment10_phase_draft_20261009/parallel_verify_horizons/`; trajectory
plots and synchronized H.264 videos are under
`results/experiment10_phase_draft_20261009/closed_loop_analysis/xyz/ParallelVerifyD{5,7}_H4_T89/`.

For queued experiments, GPU availability is checked at startup and when an owned
worker finishes. While owned workers are still running, the scheduler does not
poll GPU utilization continuously; if work remains queued and no owned completion
event frees resources, availability is rechecked only every 300 seconds. If all
four GPUs are owned by this run, no availability query is made until a worker
completion event.

`queue_frozen_static --search-pid PID` now waits for the completed frozen-candidate
manifest and then launches the existing leased closed-loop scheduler for up to two
static masks across all four suites. Its live state is recorded in
`frozen_static_queue_status.json`; a waiting state does not represent GPU work.
The frozen static and phase masks are both classified as single-rule L2-deletion
challenges, not heuristic-legal finalists. The two static challenge masks are in
closed-loop evaluation now; `queue_frozen_legal --challenge-queue-pid PID` waits
for that queue to finish, then selects up to two legal validation candidates and
queues their closed-loop evaluation. See `search/frozen_mask_classification.json`
and `frozen_legal_queue_status.json` for provenance and live status.

As of 2026-10-09 14:27 UTC, FrozenStatic01 and FrozenStatic02 each passed the
50-episode validator in all four suites (400 episodes per mask); the static
challenge queue exited successfully. The heuristic-legal closed-loop queue has
started FrozenLegal01 on Object/Spatial, with FrozenLegal02 and remaining suites
queued. Both schedule probes are complete (Spatial 392/392; LIBERO-10 486/486).
Object/Goal branch summaries are complete; actual-simulator Spatial and LIBERO-10
branch jobs are active. Live process/status files show jobs assigned to all four
GPUs (legal closed-loop on 1/3, branches on 2/0); this snapshot did not query GPU
utilization. The leased schedulers dispatch on owned-job completion and otherwise
poll for idle devices every 300s.

As of 2026-10-09 14:36 UTC, FrozenLegal01 Object and Spatial each passed validation
(50 episodes and 50 full video decodes). Their released workers were immediately
replaced by FrozenLegal01 Goal and LIBERO-10. FrozenLegal02 remains queued for all
four suites. The actual-simulator Spatial and LIBERO-10 branch jobs are still active;
status files assign all four GPUs to these four workers. This snapshot did not query
GPU utilization. The process handles and growing episode logs were checked, and no
worker failures are recorded.

As of 2026-10-09 14:46 UTC, all four branch-prefix suite summaries exist and the
branch queue is complete. FrozenLegal01 Object, Spatial and Goal each passed
50-episode/50-video validation; its LIBERO-10 run has 34 logged episodes and is
still active. FrozenLegal02 Object, Spatial and Goal are active, with LIBERO-10
pending; process status assigns the four active workers to GPUs 0–3. No fresh GPU
utilization query was made for this snapshot.

Operational incident: after a branch worker released GPU 2, I briefly started
FrozenLegal02/Object there before the main queue independently dispatched the same
condition on GPU 0. I stopped the one-off GPU 2 process as soon as the duplicate was
detected. The two workers briefly shared that run directory; the queue-managed GPU 0
worker is now the only active writer, and its full 50-episode/video validator is the
acceptance gate. The partial attempt is not counted as a result.

As of 2026-10-09 14:49 UTC, the queue-managed FrozenLegal02/Object run passed its
50-episode and 50-video validator, resolving the brief duplicate-writer concern.
GPU 0 was immediately assigned FrozenLegal02/LIBERO-10. FrozenLegal02/Spatial and
Goal remain active on GPUs 1 and 2, while FrozenLegal01/LIBERO-10 is at 43/50
episodes on GPU 3. The queue has no pending jobs at this point; all four workers
remain active. No fresh GPU utilization query was made for this snapshot.

As of 2026-10-09 14:57 UTC, the missing Spatial and LIBERO-10 paired-prefix
analyses are complete. Across 29 task-phase cells per variant/horizon, descriptive
task-phase-equal h=8 mean XYZ deviations were 1.80 mm for D3 and 2.69 mm for
V16-D10, versus 7.98 mm for V13-D10 and 8.39 mm for V13-D3. These are validation
tasks6-7/init0 representative branches, not recovery or safety evidence. The
recovery pilot therefore screens D3 and V16-D10 at an 8-action prefix, then returns
to Full18-D10; its baseline reuses the matching, fully validated Full18-D10 suite
runs. `recovery_h8_status.json` tracks this scheduler separately from the still
finishing legal-mask queue. Three recovery workers started on GPUs 1-3; FrozenLegal02
LIBERO-10 remains on GPU 0, with five recovery suite-runs queued. Per-queue status
paths and GPU leases avoid status-file collision and duplicate GPU assignment.

As of 2026-10-09 14:59 UTC, both FrozenLegal candidates completed all four suite
validators (50 episodes and 50 full video decodes per suite). Their aggregate
successes were 70/200 for FrozenLegal01, 184/200 for FrozenLegal02, versus 186/200
for the validated Full18-D10 baseline. These exploratory totals support screening
FrozenLegal02, but are not a non-inferiority claim. The recovery scheduler has now
started Recover_D3_H8 on all four suites (GPUs 0-3); the four Recover_V16_D10_H8
suite-runs are queued. The separate-status-path scheduler change passed all 29
experiment protocol unit tests.

As of 2026-10-09 15:04 UTC, Recover_D3_H8 Object and Goal passed their 50-episode
and full-video validators. Spatial has 50 logged episodes and is in validation;
LIBERO-10 has 25 logged episodes. Recover_V16_D10_H8 Object and Spatial started
immediately on the two released GPUs; its Goal and LIBERO-10 suites remain queued.
The four active workers occupy GPUs 0-3 and no failures are recorded. No new GPU
utilization query was needed while all four slots were assigned.

As of 2026-10-09 15:12 UTC, Recover_D3_H8 passed all four suite validators (200
episodes/200 full video decodes) with 191/200 successes; the matching Full18-D10
baseline has 186/200. Mean executed action counts were 163.66 vs 169.69 per episode;
this is not a latency measurement, and the exploratory difference is not a
significance or non-inferiority claim. Recover_V16_D10_H8 Object/Spatial/Goal also
passed (150/150 videos; 146/150 successes); LIBERO-10 remains active with 12 logged
episodes. A separately tracked Full18-D3/D5 static closed-loop queue has started:
Full18-D3 Object/Spatial/Goal occupy GPUs 1-3 while V16-H8 LIBERO-10 uses GPU 0;
remaining D3/D5 suites are queued. Both schedulers coordinate through GPU leases.

As of 2026-10-09 15:24 UTC, Recover_V16_D10_H8 passed all four validators (200
episodes and 200 full video decodes), with 189/200 successes. Full18-D3
Object/Spatial/Goal also passed their 50-episode/50-video validators. Its scheduler
PID exited while D3-LIBERO-10 and D5-Object workers were still live and three tasks
remained queued. The scheduler now resumes from its status file, adopts live worker
PIDs, and uses a per-status scheduler lock; it adopted both workers without
restarting them and immediately launched D5-Spatial and D5-Goal on GPUs 0 and 1.
At this snapshot, D3-LIBERO-10 and D5-Object occupy GPUs 3 and 2, respectively,
with D5-LIBERO-10 still queued. By 15:25 UTC D5-Object passed its 50-episode and
50-video validator; the scheduler immediately used GPU 2 for D5-LIBERO-10. All four
GPUs are now assigned (D3-LIBERO-10 and the remaining three D5 suites), with no
pending tasks or recorded failures. The change passed all 29 protocol unit tests.

No new independent draft-head checkpoint is used. VLM skipping preserves the
per-layer KV projections. Shadow timings are warm inference observations, not a
dedicated isolated benchmark. The suite probe includes all data roles for analysis;
mask priors and candidate selection must use search tasks only. Five inits/task
are exploratory, not a proof of non-inferiority.

`analyze_updates --suite SUITE` derives Attention/FFN cancellation and relative
update strengths from saved FP32 residual RMS. Zero denominators remain undefined.
This uses the equal-shape residual norm identity; it does not reconstruct token
locations or channel distributions from aggregate statistics.

`run_closed_loop --conditions scripts/experiment10/closed_loop_config.json --launch`
queues fresh Full18-D10 and D1/D2/D4 five-init rollouts. GPU leases coordinate
the shadow, search and closed-loop queues; external compute-process checks still
apply. All four suites for each of Full18-D10/D1/D2/D4 have passed their
50-episode and full video-decode validators (800 episodes total). Successful
D2-Goal and D4-Spatial retries preserve earlier GPU-OOM attempts under
`failed_attempts`; those attempts are not task failures. All four numerical
probes are complete. Search workers, bounded search, validation, and freezing
have completed; frozen-candidate closed-loop evaluation is underway as described
above.

`weight_reference` records 90 fixed VLM matrices on CPU, keeping fused QKV
explicit rather than guessing its internal mapping. `link_activity_intervention`
joins measured numerical features to h4 intervention errors and exports
task-equal cells. `activity_association_stability --suite SUITE` adds 10,000
task-bootstrap draws and leave-one-task-out ranges within each layer/phase/modality
cell, using only search tasks. All four suite tables are generated. Constant
resamples remain undefined and valid counts are reported; percentile intervals
condition on nonconstant resamples. Small task counts, multiplicity and missing
frozen validation tests prevent interpreting these associations as pruning safety
or confirmatory significance.

`select_submodule_layers` chooses at most four diagnostic layers from search-only
median update/effect quadrants. `queue_submodules` leases idle GPUs for attn-only,
ffn-only and both interventions, retaining all KV projections and full Expert/D10.
The initial screen uses search init0's first observation per phase and three noises;
it is not a full five-init module study. Module probes require ordinary/eager and
post-intervention baseline parity, and preserve actual candidate action arrays.

`queue_submodules --stage branches` queues actual-simulator prefixes from validation
tasks6-7, init0, the first observation per phase and noise12345. It waits for the
suite's complete shadow probe. `branch_prefixes` reconstructs independent environments
by baseline-history replay, checks camera/state equality and the original next
controller step, then measures h1/2/4/8/12 for Full18-D10 and four reference drafts.
Every emitted branch video is H.264/yuv420p, frame-count checked and fully decoded;
its short prefix is not a Full18 recovery test. Missing task phases are recorded
as unavailable rather than fabricated.

D2-Goal's first closed-loop attempt failed with a CUDA out-of-memory error and
was rejected by the validator, not counted as normal task failures. A terminal-only
`retry_closed_job --condition-index 2 --suite libero_goal` queue preserves failed
outputs under `failed_attempts` before a fresh same-configuration retry. The client
now aborts the batch on inference/transport exceptions; the server logs GPU memory
on preparation failures. The exact origin of the transient allocation failure is
not established merely by its OOM traceback.

`command_predictor` updates only from actually executed command history and gates
a contiguous prefix, separately checking translation, rotation and gripper sign.
`calibrate_command_gate` uses only Full18-D10 tasks6-7 (40 episodes, eight tasks)
for heuristic Q/R scales and task/episode-balanced residual quantiles. These are
not physical noise estimates or a task correctness verifier. Four-threshold online
gate screening is complete and did not meet the paired baseline on this pilot;
variance-floor sensitivity remains untested.

`request_ledger` tracks episode/observation/execution versions and observation age.
After draft execution it rejects the old Full18 result and requires fresh observation
before submitting a new request. It is tested infrastructure, not an integrated
asynchronous execution experiment. D1/D2/D4 paired outcome-stratified XYZ plots
and synchronized RGB/XYZ examples cover all four suites (12 validated indexes,
51 decoded composite videos).

The VerifyD3_H8 screen has now completed its 40-episode paired split (tasks 8-9,
init0-4), with all 10 episodes and videos validated in each suite. It achieved
35/40 successes, matching the paired Full18-D10 baseline's 35/40; 32 episode pairs
succeeded under both, with three candidate-only and three baseline-only successes.
Across 1,052 policy requests, 679 (64.5%) accepted a D3 draft prefix and 373
(35.5%) fell back to Full18; accepted draft prefixes averaged 5.49 actions. This
is an exploratory small split, and reference agreement is not semantic correctness.
The original scheduler status still records worker exit code 1 for all four suites:
the validator incorrectly expected 50 episodes although this config requested 10.
After correcting that count check, all four run directories passed revalidation,
including full video decoding. The original failure record is preserved; the
post-hoc analysis is in `verify_gate/verify_d3_h8_summary.json`.

GPU polling follows a queue-only rule: query for free devices only when pending jobs
cannot be dispatched to a known-free device. While our workers own GPUs, rely on
their process/status files and completion events rather than repeatedly querying
GPU state; if queued work is blocked because no device is known to be available,
retry the availability check after 300 seconds. A known owned-device completion
triggers immediate dispatch without waiting for that interval.

## Authoritative status

- `events.jsonl`: process starts/finishes and failed attempts;
- `pipeline_status.json`, `followup_status.json`: scheduling summaries (not by themselves proof of a live process);
- `observations/SUITE/manifest.json`: verified replay provenance and stage coverage;
- `probe/SUITE/progress.json`: complete numerical/intervention sample count;
- `probe/SUITE/summary.json`: full suite completion; requires all source observations;
- `shadow/SUITE/summary.json`: shadow completion, **not** closed-loop evidence;
- `phase_analysis/layer_prior.json`: search-only cosine protection and sensitivity prior.

Check the PIDs from status files and increasing progress before treating a job as
live. Do not restart on an observation timeout; resume only verified terminal jobs.
