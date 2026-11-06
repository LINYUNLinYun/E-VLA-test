# EASE insight 1: fixed draft checkpoint generalization

This package runs a target-fixed, case-paired cross-suite experiment without editing the existing FLASH repository. It is scheduled by the main controller; these scripts never choose or acquire GPUs automatically.

## Frozen discovery matrix

- Four FLASH source heads × four target LIBERO suites × ten tasks × official init index 20 = 160 FLASH conditions.
- The same 160 conditions using the pure draft route.
- Full Pi0: four suites × ten tasks × init20 = 40 conditions. The Goal draft artifact is loaded solely because the native runtime constructor requires it; route counters must show no draft or verify call.
- Protocol: common Pi0 target, H50, R12, D10 for Full, standard 10 settling steps, environment seed7, native per-suite maximum control steps (Spatial220/Object280/Goal300/Long520). FLASH uses radius .15 and periodic Full after two draft rounds; native gripper verification remains enabled.
- Independent confirmation: after examining discovery, freeze source-target comparisons and predicted direction before running init21–24. These are exploratory cases, not official 50-init benchmark numbers.

Noise is generated from suite/task/init/replan identity, independent of scheduling, source head, and route mode. Pure draft does not consume diffusion noise. FLASH and Full use the same noise at the same replan index; their observations diverge in closed loop, so this is not a same-input action equivalence claim.

## Local model assets and normalization

`shared/checkpoints/realtime-vla-flash/converted/base/base_weights.pkl` is the shared target Pi0. Source heads are `converted/draft_spatial`, `draft_object`, `draft` (Goal), and `draft_10`; original released `.pt` files also exist in their parent directory.

The native `spec_serve_policy._build_policy_components` always reads state/action normalization from **base/assets** with `config=pi0_libero`. Swapping a draft path changes the source head only. The target suite controls task language and environment. A private mutable prompt manifest includes all existing cached prompts; checkpoint and prompt tensors are linked, so native prompt-cache writes cannot modify a shared checkpoint directory.

Pure draft calls the original `SpecTritonRuntimeSession.run_draft_with_timing`, including the common vision prefix and released head. It never runs a VLM action prefix, AE, verification, periodic Full, or initial Full. Unused Full graph construction is suppressed for this mode. Runtime route counters enforce zero Full/verify calls. Returned prefix length12 specifies execution length and **is not a verification acceptance rate**.

## Launch contract

```bash
python3 preparations/ease_insights_20261103/draft/run.py \
  --gpu 0 --gpu-uuid GPU-EXPLICIT-ALLOCATION --egl-index 3 \
  --mode flash --source-suite libero_goal --target-suite all \
  --output results/ease_insights_20261103/flash_goal
```

`--mode` is `flash`, `draft`, or `full`. `--target-suite` accepts `all`, a single suite, or a comma-separated subset. `--cases` (alias `--manifest`) accepts a JSON list or `{ "cases": [...] }`. Each case is `{ "suite": "libero_goal", "task_id": 0, "init_id": 20 }`; `target_suite`/`task_suite_name` and `episode_idx` aliases are accepted. Without a file, all ten tasks and `--init-ids 20` are generated. `--dry-run` performs filesystem preparation only.

One service persists through every selected case for one source and mode. The parent verifies allocation and controls the shared GPU lock. Physical CUDA and EGL numbering can differ; `--egl-index` pins the parent's freshly verified mapping. Without it, the existing `render_environment` helper enumerates the current mapping. Model CUDA is pinned by UUID; client rendering has its own mask.

An output with completed `result.json` is returned without rerunning. A failed/running output is refused: inspect authoritative process state and preserve completed cases before assigning a new technical retry directory. Policy failures are valid finished results and must never be retried because they failed.

## Evidence and accounting

Each case saves:

- `result.json`: official success oracle, valid failure versus technical error, executed action count, complete wall time and client RPC sum.
- `environment_identity.json`, `initial_state.npz`, `first_policy_observation.npz`: official init vector and simulator-state hashes, BDDL and actual controller hashes/configuration, instruction, action limits and control frequency.
- Native `trace.jsonl`, `infer.jsonl`, and `telemetry.jsonl`: executed actions, routes, measured end-effector and object states. Videos are prospectively omitted.
- Run-level `requests.jsonl`: input/noise hashes, raw and decoded first12×7 action values, explicit draft/Full/verify function counts, initial/periodic/rejection Full reasons and stage/request timings.

The stock client sets `failure_reason` only when an exception occurs. Ordinary timeout leaves it null and is recorded as a policy failure. Technical errors remain outside success-rate denominators.

```bash
python3 preparations/ease_insights_20261103/draft/analyze.py \
  --runs results/ease_insights_20261103/flash_goal \
         results/ease_insights_20261103/draft_goal \
         results/ease_insights_20261103/full \
  --output preparations/ease_insights_20261103/draft/analysis
```

The analyzer produces per-case and per-target summaries; paired foreign-source versus target-matching-source success counts; exact uncorrected McNemar p-values with multiple-comparison caution; acceptance among verified rounds only; overall Full fraction and rejection-specific fraction; latency and closed-loop wall comparisons against paired Full cases; first-input/noise identity checks. Pure draft acceptance is null. A high Full fraction can arise from initial or periodic refresh, so it is not synonymous with rejection fallback.

`total_ms` is native GPU stage timing. `sample_actions_ms` / server policy wall includes cold runtime construction; client RPC includes transport and server serialization. Per-episode wall includes environment/rendering and changed trajectory length. Relative closed-loop wall, inference work and mean request latency are separately labeled; none substitutes for a controlled same-observation microbenchmark. Startup health only guarantees model loading, while prompt-dependent graph preparation can occur in the first request.

CPU preparation checks: native server/client imports in their actual Python environments, all40 case/noise uniqueness, syntax, and synthetic accounting tests (technical failures excluded, pure-draft acceptance null, verified acceptance correctly denominator-weighted). GPU smoke validation is performed by the parent before dispatching the discovery matrix.

## Existing evidence boundary

The earlier `reports/idea1_libero_plus_half_final_20261022T2124Z/README.md` has 175 perturbed and20 original cases. It measures **pure** draft, not complete FLASH. Original-case pure-draft success was already40%, and the original/perturbed sets do not have fully identical mother tasks/initial states. Consequently its30.9% perturbed success is useful motivation but not an isolated perturbation effect, nor evidence that complete FLASH success or speed necessarily collapses.

## Spec-VLA feasibility, checked 2026-11-03

No Spec-VLA source repository or draft assets were found under local `policies` or `shared/checkpoints`. Existing OpenVLA-OFT is a different policy implementation and must not be silently substituted for Spec-VLA's autoregressive OpenVLA target.

The [official SpecVLA repository](https://github.com/PineTreeWss/SpecVLA) publishes four suite draft download links in its [README](https://github.com/PineTreeWss/SpecVLA/blob/main/README.md), with Python≥3.10, PyTorch2.2.0 and LIBERO dependencies. Its [Goal evaluation script](https://github.com/PineTreeWss/SpecVLA/blob/main/openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py) exposes independent target `pretrained_checkpoint` and `spec_checkpoint`; [model loading](https://github.com/PineTreeWss/SpecVLA/blob/main/openvla/experiments/robot/openvla_utils.py) obtains processor and action normalization statistics from the target checkpoint.

A future valid comparison can hold each target suite's OpenVLA checkpoint, processor, normalization key and relaxed threshold fixed while swapping source draft. However, each released source draft was associated with a separately fine-tuned target backbone, so hidden-feature compatibility must be checked; degradation would mix distribution change and cross-backbone feature transfer. This is a broader experimental condition than FLASH's four heads sharing one Pi0 base. The code and model assets still need installation/download and checkpoint compatibility checks; no Spec-VLA GPU experiment is claimed or used to delay FLASH.

## Prospective independent replication preparation

`prepare_replication.py` is a CPU-only gate and job-draft writer. It does not launch experiments or update the scheduler queue:

```bash
python3 preparations/ease_insights_20261103/draft/prepare_replication.py \
  --discovery-manifest results/ease_insights_20261103/draft_discovery_manifest.json \
  --runs results/ease_insights_20261103/draft_smoke results/ease_insights_20261103/draft_discovery \
  --output preparations/ease_insights_20261103/draft/replication \
  --run-output-root results/ease_insights_20261103/draft_replication
```

The default inputs and output match this command. `--audit-only` checks readiness without selecting. An incomplete discovery returns exit code2 and writes `discovery_audit.json`, with **no selection or jobs**. The gate requires exactly360 unique `(mode, source, target, task, init20)` identities, including the8 smoke cases; complete run receipts, case receipts, official episode identity, native client configuration, shared normalization/protocol and frozen runtime hashes must agree. The36-case remainder shards and40-case full-source shards must have identical scientific metadata.

Only after the complete matrix passes, the script compares all12 foreign-versus-matched pure-draft pairs. It ranks matching-source successes minus foreign-source successes, descending; ties use target SUITES order, then foreign-source SUITES order (Spatial, Object, Goal, Long). It selects the strongest positive pair and the strongest remaining positive pair with a different target. No positive gap produces `status: none` and zero jobs. All12 pure and12 FLASH discovery comparisons are saved regardless of selection; no discovery p-values are emitted.

For each selected target, the draft jobs contain40 new cases (ten tasks × init21–24) for matched pure, foreign pure, matched FLASH, foreign FLASH and Full. Thus two selected pairs produce ten jobs/400 validation episodes. The Full artifact source stays the same as discovery. The parent may append `jobs_draft.json.jobs` to the controller queue after inspection; this script never does so.

`selection.json` freezes the complete discovery evidence hashes, input manifest hash, all ranked pairs, selected direction, case indices and producer hash. Existing selection files cannot be overwritten. Before selection, the tool inventories held-out case files and metadata without reading their outcomes; existing init21–24 observations/results cause a refusal to claim a newly preregistered independent comparison. Planned cases without observations are listed but do not block. `--heldout-search-roots` defaults to the current experiment result root, so unrelated projects are not searched or used.

Final reports must show every discovery comparison. Confirmation success gaps, paired counts and any inferential p-values must use only held-out init21–24; the selected discovery gap is not an unbiased effect estimate. `--self-test` exercises complete/missing/duplicate identities, null selection, deterministic ties and ranking, and proves the held-out inventory does not parse outcome contents.

## Primary service-cost metric: charge every request to executed actions

The analysis now defaults to `policy_ms_per_executed_action` and `client_rpc_ms_per_executed_action`. Their numerators include **all** complete native policy calls or client RPCs, including FLASH rejection rounds that execute zero actions. Dividing by request count alone can reverse the conclusion: Full1×100ms for12 executed actions costs8.33ms/action; FLASH2×60ms for the same12 actions costs10ms/action. The service-cost speedup is100/120=0.833×, while the100/60=1.667 request-average ratio is only a diagnostic.

`paired_action_costs_vs_full.json` records each exact target/task/init comparison with both success outcomes, real episode walls and policy/RPC per-action costs. `speed_vs_full.json` pools only paired cases and reports Full cost divided by condition cost; below1 means the condition costs more. It also retains actual episode-wall ratios and both success counts. The old ambiguous request-latency speedup fields have been replaced with explicitly named `mean_policy_round_ratio_not_actual_speedup` and `mean_gpu_stage_round_ratio_not_actual_speedup` diagnostics; raw mean-round fields remain available.

The independent-confirmation figure uses three columns: success count/rate with Wilson intervals; complete policy cost per executed action (client RPC cost shown separately); real episode wall. Cold calls remain included. GPU-stage timing is never labeled end-to-end, and differing closed-loop trajectories/success outcomes must remain visible beside any service-cost comparison.
