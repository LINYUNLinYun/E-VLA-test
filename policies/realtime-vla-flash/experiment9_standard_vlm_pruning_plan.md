# Experiment 9: Standard LIBERO VLM pruning evaluation

Created 2026-10-04. This plan is also the configuration source for `scripts/experiment9/experiment9_config.json`.

## Conditions

Evaluate Full18 and ten VLM variants, keeping Action Expert Full18 and denoising at D10. Skip-L2, L3, L4, L5 are independent single-block interventions; Skip-L2-5 removes the four updates together. V16, V14, V12 and V10-Mid reproduce the older retained-layer patterns. V13-Custom retains L1, L2, L5, L6, L8, L9, L10, L11, L13, L14, L16, L17, L18. In Triton, skipped blocks preserve their per-layer prefix K/V projections for the Action Expert; they skip Attention/FFN hidden-state updates.

## Suites and schedule

Use all ten tasks in each of Spatial, Object, Goal and LIBERO-10. Initial states come from the official suite arrays, seed 7, with the same index paired across conditions. Use D10, warmup 10, replan 12 and suite-default action limits.

1. Pilot every condition on indices 0–9 for every task and suite: 4,400 episodes total after adding V13-Custom.
2. Complete Full18 first, then complete every remaining condition/suite to indices 0–49. Reuse verified pilot episodes; the final VLM matrix is 22,000 unique condition-task-state rollouts. With the three complete Action Expert denoising variants, the main matrix contains 28,000 rollouts.
3. Schedule V12/Object immediately after baseline completion for the specific object-suite hypothesis. Keep every other suite/condition in the completion queue regardless of its pilot score.
4. Run a condition only on a GPU with at most 2 GiB allocated and at most 10% utilization. When no card meets both criteria, check again every 300 seconds. Never preempt an existing process. Multiple free cards may run independent jobs; each gets a separate prompt-cache directory.
5. Evaluate in non-overwriting ten-state chunks. A fully validated chunk is reusable after interruption. Partial chunks remain in their attempt directory and are excluded from summaries; retries receive new directories.

Every rollout records actual server-side VLM pattern, layer indices, noise hash, runtime metadata, official task/state identity, episode success, action trace, synchronized initial-plus-post-action EEF/object/contact telemetry and full-rate H.264 environment video. Semantic phase labels are derived from the simulator telemetry using the established LIBERO phase classifier.

## Analysis and deliverables

Report success rates by variant, suite and task; paired new-failure/recovery counts; task-clustered bootstrap intervals; Wilson intervals; and every observed suite result at or above 90%. Keep the 90% threshold separate from the paired baseline comparison. Report the prespecified two-percentage-point non-inferiority reference descriptively, without claiming non-inferiority from an underpowered pilot.

For failure stages, report task/episode reach rates and conditional transitions through approach, contact, grasp/manipulate, transport, goal alignment, final release and success. Inspect whether each pruning condition reaches early motion stages but loses success during precise interactions. Mark inapplicable/unknown stages explicitly.

For trajectory assets, plot measured MuJoCo EEF positions in world XYZ. Use blue solid lines with circles for Full18 and orange dashed lines with triangles for each variant, reduced opacity, sparse markers, common-step deviations and task-phase labels. For each pilot variant/suite, produce one paired RGB/XYZ video for each outcome category present (new failure, recovery, both success, both failure), choosing the largest same-step XYZ deviation within that category. Preserve N+1 state/video-frame alignment.

Deliver the interim pilot and final Markdown reports, machine-readable episode and paired summaries, success/phase figures, XYZ plots and MP4 galleries under the Experiment 9 results directory. The root plan remains tracked; bulk experiment artifacts remain ignored.

## Action Expert follow-up

After VLM evaluation, compare D10/D7/D5/D3 with VLM Full18 on the same official states. Pilot D7/D5/D3 across all suites; expand combinations with at least 90% observed success or within 2 percentage points of D10 to 50 states per task. Report paired success, end-to-end latency and measured denoise latency by suite. Then analyze Action Expert hidden similarity and fixed-noise action sensitivity at initial, contact and goal-alignment observations from the baseline traces. Only advance the two least action-sensitive single-layer candidates to closed-loop screening; if probe criteria do not yield credible candidates, report that layer pruning lacks support. Do not combine VLM and Action Expert pruning in this round.

## Completion audit and supplement (2026-10-05)

The audit found 27,600 of 28,000 planned rollouts: V13-Custom indices 0–9 were absent in all four suites. Four GPU jobs began at 12:36 UTC to backfill these 400 rollouts. The custom runner now schedules pilot chunks as well as full chunks. Raw data and the completion audit are saved; the full analysis report remains deferred at the user's request.

The Action Expert structural probe replays Full18 official init0 for all 40 tasks, checks every replay EEF position and terminal success, and exports observations at exact initial/contact/goal-alignment telemetry indices. Unobserved phases are listed explicitly. Use all 18 decoder layers, D10, common noise seeds 12345/23456/34567, and action-token hidden states at flow steps 0, 2, 5, 9. Save layer-pair cosines, adjacent relative residuals, and unnormalized 7-dimensional action sensitivity for each complete single-layer skip, using first-12 and full-50 action windows. Instrumented and repeated baselines must be bitwise equal to ordinary inference, and checkpoint hashes must stay unchanged.

Screening gates are recorded before probing: task-mean first-12 NMSE ≤0.01, task-clustered bootstrap upper 95% bound ≤0.02, task-mean first-12 gripper-sign flip rate ≤0.01, worst-phase task-mean NMSE ≤0.05, and at least three distinct tasks for every suite/phase. Average observations and noise repeats within tasks before bootstrapping. Rank eligible layers by action sensitivity and screen at most two; these diagnostic gates do not establish closed-loop success. First verify that each registered CUDA-graph skip exactly matches saved offline skip actions. Then screen on all four suites, ten tasks each, official indices 0–9, D10, seed 7, replan 12 and matched noise, saving XYZ/video/inference metadata. Keep screening data separate from the 28,000-rollout main matrix.
