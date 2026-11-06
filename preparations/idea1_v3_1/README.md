# idea1 v3.1 preparation

This README originally documented preparation-only interfaces. Actual execution is now in progress; authoritative progress and protocol clarifications are in [the execution status](../../docs/experiments/idea1/idea1_v3_1_execution_status.md) and [the plan](../../docs/experiments/idea1/idea1_v3_1_experiment_plan.md). Code presence or a successful CPU test never implies experiment completion.

## Active execution notes (2026-10-21)

- Use the existing immutable `manifest_v2.json`; do not regenerate it over an active run.
- Cross-process native/matched Full parity receipts below are retained **failed observations**, not the active vision gates. Use `results/idea1_v3_1/run/pi0_vision_same_process_parity.json` or `pi05_vision_same_process_parity.json` explicitly for the matching model; each passed original-tolerance same-loaded-model layer/raw/decoded checks on five real inputs covering all four suites. Details and original failures are preserved in plan §3.1.
- All four models now have independently audited positive same-service E2 evidence: π0.5 `e2_shared_service_validation/`, π0 `e2_shared_pi0/`, XR0 `e2_shared_xiaomi/`, CogACT `e2_shared_cogact/`, all under `results/idea1_v3_1/`. Quality there may explicitly use `--workers 2 --environment-reuse on`; the unchanged reuse gate still authenticates each model's own evidence. CogACT quality is waiting for an idle card; evidence registration does not mean its quality matrix has run.
- Quality data span the original `run/` and per-model shared-service roots. The active index is `results/idea1_v3_1/dataset_bundle_v3_all_models/index.json`; use `dataset_bundle_v3` to analyze/audit explicit episode, profile and benchmark origins. Older indices remain historical snapshots and can omit newer pending-condition roots. No index is a completion certificate.
- `final_audit_v2` corrects two old verifier metadata assumptions and requires affirmative same-process gate evidence; old failures remain observations. Missing screens, confirmations, benchmarks or conditional-frontier work keep the final audit false. Run the bundle audit for split-root data, not the old single-root command as a purported final certificate.
- Remaining examples below show historical interfaces and preparation commands. Do not launch duplicates of already valid scientific keys or use a failed old gate.

Run modules from the benchmark root. Model services use their own existing virtualenv; LIBERO workers use `policies/libero-eval-v21/.venv`. Every launch entrypoint defaults to a dry run and needs explicit `--launch` before loading a model or environment.

Freeze the manifest once after code preparation is complete:

```bash
policies/openpi/.venv/bin/python -m preparations.idea1_v3_1.manifest
policies/openpi/.venv/bin/python -m preparations.idea1_v3_1.manifest --write --output results/idea1_v3_1/manifest_v2.json
```

It contains 27 XR0 conditions (Full, historical reference, five masks at each skip12/15/18/21/24 budget), 22 CogACT conditions (Full, reference, five masks at skip9/12/15/18), and 21 vision conditions each for pi0/pi05. Screening is40 episodes per condition. Original one-based indices are recorded explicitly. XR0 protects1–3 and CogACT1–2 to satisfy the existing audited adapters. XR0 uses its effective processor horizon10, not configured action_length30, and D2/R10; CogACT uses native DDIM D10 and original suite-dependent R. Pi keeps native H50/H10 and D10/R5.

Inspect commands without starting anything:

```bash
policies/openpi/.venv/bin/python -m preparations.idea1_v3_1.controller run --manifest results/idea1_v3_1/manifest_v2.json --output results/idea1_v3_1/run --model pi05 --stage screen
```

Add `--gpu PHYSICAL_ID --bank REAL_BANK.pkl --launch` when the assigned GPU is free and execution is authorized. Launched conditions require a real-observation bank for adapter parity and20 requests of warmup. Services serialize model requests; two rollout workers overlap environment work and RPC waiting, without claiming model batching. Existing compute processes make the controller stop before loading a model.

For vision, only the SigLIP Encoder method is patched before tracing. A static gather selects retained original block parameters, then the original `nn.scan`/rematerialized block structure runs with the retained depth. Patch/position embeddings, final norm, projector and256 spatial tokens remain. Gemma VLM/AE methods and the original full sampler are untouched. Native Full uses the original encoder method. A separate all27 selected-scan Full is benchmark-only and must pass real-input parity before masked quality rollout:

```bash
# Collect100 unique actual records from completed idea2 original Full episodes.
policies/openpi/.venv/bin/python -m preparations.idea1_v3_1.collect_bank --source results/idea2/run_20261020/episodes/fresh_serial_r1 --output results/idea1_v3_1/input_bank.pkl --write
# Supply that bank, or other trusted banks totaling>=100 unique actual inputs.
policies/openpi/.venv/bin/python -m preparations.idea1_v3_1.controller benchmark --manifest results/idea1_v3_1/manifest_v2.json --output results/idea1_v3_1/run --condition pi05_full --gpu PHYSICAL_ID --bank BANK1.pkl BANK2.pkl BANK3.pkl BANK4.pkl --launch
policies/openpi/.venv/bin/python -m preparations.idea1_v3_1.controller benchmark --manifest results/idea1_v3_1/manifest_v2.json --output results/idea1_v3_1/run --condition pi05_full_vision_scan_reference --gpu PHYSICAL_ID --bank BANK1.pkl BANK2.pkl BANK3.pkl BANK4.pkl --launch
policies/openpi/.venv/bin/python -m preparations.idea1_v3_1.parity --native results/idea1_v3_1/run/pi05_full/benchmark.json --matched results/idea1_v3_1/run/pi05_full_vision_scan_reference/benchmark.json --output results/idea1_v3_1/run/pi05_vision_parity.json
```

Pass that artifact as `--vision-parity PATH` when launching masked vision rollouts. Report speedups against both native Full and the matched selected-scan reference; first compilation is recorded separately. The checkpoint-free CPU test checks this plumbing but does not replace real-model parity.

E2 profiles eight paired episodes: four suites, task0/init0–1, so environment reuse has a repeated task to exercise. Select one Full condition and run the four frozen variants independently with `--stage profile --variant full_io|lean_io|reuse_env|two_workers`. Policy cameras always render. Optional video/observation writes, environment reuse and workers are changed one factor at a time; real XYZ, commands, seeds, input/noise hashes and outcome logs remain. Model startup and warmup precede aggregate throughput timing. The two-worker arm uses one serialized RPC server; it is not batched inference.

The worker resumes only completed, matching-manifest episodes. Partial directories are preserved as `.invalid.UUID`. Quality screen and confirm share episode paths: confirmation reuses init0 and adds init1–4, totaling200. Freeze promotion with `controller promote ... --launch`; then use `--stage confirm --selection promotion.json`. A candidate needs complete40-episode screening. E1 best-per-budget confirms at relative retention≥0.60 (within the frozen0.10 margin of the0.70 target); vision best-per-budget confirms at≥4/40 successes. Full confirms regardless. These selection rules are exploratory, with shared screen data.

```bash
PYTHONPATH=policies/openpi/src:. JAX_PLATFORMS=cpu policies/openpi/.venv/bin/python -m unittest preparations.idea1_v3_1.test_protocol preparations.idea1_v3_1.test_vision_cpu -v
policies/openpi/.venv/bin/python -m preparations.idea1_v3_1.analyze --manifest results/idea1_v3_1/manifest_v2.json --root results/idea1_v3_1/run --output results/idea1_v3_1/summary.json
```

The lean runner reuses simulator object, predicate, contact and grasp telemetry to label conservative manipulation phases. Unavailable contact/grasp truth remains unknown. Its farthest phase is progress evidence, not a causal diagnosis of failure.

Limitations: no compile/CUDA Graph optimization is enabled for v3.1 and no RPC batching is implemented. These are explicit unmeasured follow-ups. Camera rendering remains part of simulation even when video is disabled. The same-GPU vision/request benchmark measures completed outputs separately; aggregate rollout throughput is a different metric.
# Prelaunch revision v2

Quality environment reuse defaults to **off**. `--environment-reuse on` on the
controller or rollout requires same-model eight-scene E2 `reuse_env` paired
input/noise/output/outcome parity and measured throughput benefit. E2 variants
retain their frozen reuse settings regardless of the quality override. Use
`manifest_v2.json`; the original manifest and exact source archive are preserved.

Read-only final evidence audit (partial work remains `passed: false`):

```bash
policies/openpi/.venv/bin/python -m preparations.idea1_v3_1.final_audit --manifest results/idea1_v3_1/manifest_v2.json --root results/idea1_v3_1/run --analysis-report results/idea1_v3_1/summary.json --allow-partial
```

Only explicit `--output PATH` writes an audit receipt. Stored videos are decoded
strictly and checked against telemetry frame counts at20Hz. Camera pixels omitted
by lean I/O cannot be reconstructed; failure-stage labels are not causal evidence.
The initial XR skip24/Cog skip18 caps do not close a frontier that remains usable:
near-floor candidates need confirmation and confirmed retention>=70% needs a
separately frozen higher-budget extension.

