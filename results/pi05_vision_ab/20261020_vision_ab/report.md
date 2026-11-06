# pi0.5 vision A/B compression — 20261020_vision_ab

Fixed protocol: C0/A128/A64/B128/B64; deterministic ToMeSD mapping; A compresses the attention residual branch, B additionally the MLP branch; each branch is unmerged before the full residual. Weights, VLM, action expert, token layout, sampling steps and execution rules are unchanged.

## Provenance

- checkpoint: `/root/nfs/embodied/benchmark/shared/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero`
- norm stats sha256: `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`
- openpi commit: `215abfb217dbac7d5f1273282331b9b1866c0479`
- backend: `jax` / dtype `bfloat16` / compile `jax_jit`
- model python: `/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python`; eval python: `/root/nfs/embodied/benchmark/policies/libero-eval-v21/.venv/bin/python`
- observation library: 90 real LIBERO observations

## Correctness

All required checks passed: **yes**.

| check | result |
|---|---|
| mapping_matches_independent_reference | True |
| hand_tensor | True |
| dense_residual | True |
| kN_identity | True |
| encoder_identity | True |
| policy_identity | True |
| params_unchanged | True |
| shapes | True |
| compressed_finite | True |
| compressed_actual_k_compute | True |

K=256 encoder max abs 0 (bitwise=True), policy action max abs 0.00293 (BF16 atol 0.01), repeat-run max abs 0. Compressed runs observed actual K-length branch cost and 256-token unmerge for every configuration.

## Vision latency

T_vision_all includes stem, position embedding, all 27 blocks, matching, merge/unmerge, final norm and projection; 3 sequential camera slots, 2 valid cameras, batch 1. Compile time is separate.

| config | p50 ms | p95 ms | mean ms | round p50 range | speedup | 95% CI | reduction |
|---|---|---|---|---|---|---|---|
| C0 | 13.998 | 14.493 | 14.063 | 13.714–14.095 | 1.000 | 1.000–1.000 | 0.0% |
| A128 | 15.108 | 15.619 | 15.198 | 15.037–15.145 | 0.927 | 0.925–0.928 | -7.9% |
| A64 | 14.656 | 15.232 | 14.737 | 14.386–14.792 | 0.955 | 0.954–0.957 | -4.7% |
| B128 | 14.184 | 14.808 | 14.315 | 14.144–14.213 | 0.987 | 0.985–0.988 | -1.3% |
| B64 | 12.766 | 13.363 | 12.840 | 12.663–12.848 | 1.097 | 1.095–1.098 | 8.8% |

Per-round p50 for T_vision_all (ms):

| config | r0 | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| C0 | 13.714 | 13.977 | 14.008 | 14.026 | 14.095 |
| A128 | 15.145 | 15.118 | 15.037 | 15.108 | 15.118 |
| A64 | 14.792 | 14.461 | 14.386 | 14.724 | 14.687 |
| B128 | 14.199 | 14.144 | 14.154 | 14.181 | 14.213 |
| B64 | 12.750 | 12.799 | 12.848 | 12.663 | 12.705 |

Full-policy timing is in `latency_summary.csv` (`model` = GPU-ready observation to action tensor; `policy_wall` = synchronized real policy call including CPU preprocessing/H2D/D2H).

Configurations reaching the 1.20x vision-speedup engineering target: none.

## Screen (init 0–4, 40 tasks)

SR intervals are task-stratified paired bootstrap over 10,000 resamples; ΔSR and relative SR are paired against C0 on the same task/init/noise keys.

| config | success/total | SR [95% CI] | libero_spatial | libero_object | libero_goal | libero_10 | ΔSR pp [95% CI] | rel. SR [95% CI] | speedup | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 194/200 | 0.970 [0.950, 0.990] | 1.000 | 0.960 | 0.980 | 0.940 | baseline | 1.000 | 1.000 | not selected |
| A128 | 191/200 | 0.955 [0.935, 0.970] | 1.000 | 0.960 | 0.980 | 0.880 | -1.48 [-4.00, +1.00] | 0.985 [0.959, 1.011] | 0.927 | not selected |
| A64 | 94/200 | 0.470 [0.420, 0.520] | 0.620 | 0.580 | 0.500 | 0.180 | -49.96 [-55.50, -44.50] | 0.485 [0.432, 0.537] | 0.955 | not selected |
| B128 | 179/200 | 0.895 [0.860, 0.930] | 1.000 | 0.860 | 0.860 | 0.860 | -7.46 [-11.50, -3.50] | 0.923 [0.883, 0.963] | 0.987 | not selected |
| B64 | 3/200 | 0.015 [0.005, 0.025] | 0.000 | 0.000 | 0.060 | 0.000 | -95.47 [-97.50, -93.00] | 0.016 [0.005, 0.026] | 1.097 | not selected |

- Family A: selected none — no candidate with a measured vision speedup > 1.0
- Family B: selected none — no speedup candidate reaches relative SR 0.60

Fastest measured candidate at each relative-SR threshold (a candidate must also have vision speedup > 1.0):

- >= 0.90: no candidate
- >= 0.70: no candidate
- >= 0.60: no candidate

Measured screen cost is 12.8 s/episode (mean wall time over 1000 episodes). For planning only: 3600 confirmation episodes would take about 12.8 GPU-hours (4.3 h over three GPUs) and 5400 about 19.2 GPU-hours (6.4 h over three GPUs). This budget was not executed because no candidate passed the fixed rule.

## Failure diagnosis (plan section 12)

Mode B at K=256 is an identity check on the real checkpoint: encoder features bitwise identical=True, relative L2 0; action max abs 0.00342 (BF16 noise), params unchanged=True. The MLP-branch integration therefore does not change the untriggered path.

| config | vision feat rel-L2 | vision feat max abs | action rel-L2 | action max abs | params unchanged |
|---|---|---|---|---|---|
| A128 | 0.173 | 56 | 0.129 | 0.307 | True |
| A64 | 0.282 | 71.5 | 0.280 | 0.806 | True |
| B128 | 0.206 | 56 | 0.298 | 0.789 | True |
| B64 | 0.331 | 89.5 | 0.642 | 2 | True |

Deviations are measured on the same 8 real LIBERO probe observations with fixed action noise, so they reflect the compression only. B64 is the only configuration with a measured vision speedup (1.097x), but its relative screen SR is 0.015 and its action deviation is the largest measured. The optional layer-scope rescue ([2, 25), keeping the first/last two layers dense) was not run: the implementation error is ruled out by the K=256 identity check, and leaving 4/27 layers dense can only reduce the already sub-target 1.097x vision speedup, so it cannot produce an engineering-relevant candidate. Per the fixed rule, no confirmation was started.

## Independent confirmation

No candidate entered confirmation under the fixed screening rule; this report contains screening results only.

## Notes and limitations

- Episodes are paired by task, initial state and explicit Gaussian action noise keyed without config id; environment seed is reset before every episode.
- `policy_wall` includes the policy's CPU transforms, host/device transfer and output conversion; it is not decomposed into per-stage sums.
- Timeouts are task failures; infrastructure errors block aggregation and must be rerun, never silently deleted.
- Screen episodes were sharded over three GPUs after a single-GPU start; the episode key includes stage/condition/task/init/noise, so resume did not duplicate or drop records. No confirmation episodes were run.
- Confirmation-budget estimate is not applicable: the fixed screening rule selected no A/B candidate, so the planned 3600/5400-episode confirmation was not started.
- Video capture was not enabled; failures are identified by episode keys and environment step counts in the JSONL rather than by video.
- Every episode JSONL record carries a `fingerprints` object (checkpoint params manifest/mtime, norm-stats sha256, openpi commit, code sha256) joined from the final audit manifest after the run; no measurement field was altered.
- Self-speculative system evaluation: not evaluated (no runnable pi0.5 draft/verifier harness in this workspace).
- Negative or non-speedup results are preserved; no configuration is excluded from the tables.

## Artifacts

`latency_raw.csv`, `latency_summary.csv`, `latency_rounds.csv`, `latency_compile_meta.json`, `correctness.json`, `rollouts_screen.jsonl`, `episode_manifest_*.jsonl`, `success_summary.csv`, `success_by_task.csv`, `success_by_suite.csv`, `summary.csv`, `episode_results.csv`, `selection.json`, `parity/diagnostics.json`, `fig_*.png`, `manifest.json`, `commands.sh`.
