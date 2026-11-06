# Fixed-mask VLM fallback

This isolated adapter implements three real V12 paths with fixed original one-based skip layers:

| Condition | Skip layers |
|---|---|
| `fixed_v12_jacobian` | 2, 3, 5, 14, 16, 17 |
| `fixed_v12_action_mse` | 2, 3, 5, 11, 16, 17 |
| `fixed_v12_cosine` | 2, 3, 5, 6, 8, 9 |

All use H50/R5, Full-D10 then two Fast-D2 requests, and their own mask's residual extrapolation. `full_d10` is the paired reference. The backend controls actual QKV versus KV-only kernels, hidden updates, and residual capture using the requested mask. It reuses only the historical difference/patch kernels and base model loader; it does not change the historical implementation. No selector or gradient computation runs.

Discovery is four suites × ten tasks × official init30 = 40 episodes per condition, 160 total. Init31–35 are reserved; `confirm` currently raises an error. Freeze a selected pair and category directions in a new manifest before validation. This is a periodic Full/fast-system test, not a claim about standalone submodels or adaptive selection.

Prepare (CPU only), with an output directory chosen by the main scheduler:

```bash
python3 -m preparations.ease_insights_20261103.vlm.prepare --output OUTPUT
policies/realtime-vla-flash/.venv/bin/python -m unittest preparations.ease_insights_20261103.vlm.test_protocol -v
```

The preparation checks official init30–35 availability and hashes those vectors. The final preparation receipt lives in `prepared_cost_final/manifest.json`. Preparation does not launch a GPU. Earlier preparation directories are superseded and cannot pass the current source verification.

Explicitly assigned device/port, bounded preflight only:

```bash
policies/libero-eval-v21/.venv/bin/python -m preparations.ease_insights_20261103.vlm.runner --output OUTPUT --condition full_d10 --gpu-uuid GPU_UUID --port PORT --preflight-only --launch
```

The same command without `--preflight-only` runs discovery after the gate; choose one of the four condition IDs. Omit `--launch` to print commands without GPU access. `--shard-index I --shards N` partitions `(suite,task)` deterministically and resumes completed valid outcomes. One server has exactly one worker because its residual history is stateful. The main scheduler owns the global GPU lease; this runner never discovers or claims another GPU and refuses an assigned device with a compute process.

The GPU gate uses four real suite inputs and every fixed mask. It checks native Full and pure-mask parity, exact captured residual identities, same-observation reconstruction of Full-D2 and KV, reference versus Triton extrapolation on two distinct observations, repeatability, actual kernel branch identities, and Full/Fast/Fast schedule. Numerical tolerances are fixed at atol=rtol=1e-5 and are not relaxed on failure. Cross-service outputs use the same tolerance. Gates and 20 warmup requests are recorded separately from rollout latency.

After the numerical gate and before service readiness, a bounded cost probe compares Full and all three masks on the same four suite observations. Each condition resets history, runs six warmup requests (two complete Full/Fast/Fast cycles), and measures twelve requests. Condition order rotates across suites. `fixed_input_cost.json` retains per-request completed policy latency, Full/Fast stage summaries, three-request-cycle mean/p50, and ratios to the pure Full policy. The probe uses 96 warmup and 192 measured requests in total, with identical per-round seeds across conditions. These are fixed-observation costs only; small differences remain uncertain, and no acceptance, quality, or closed-loop speed claim follows. Loading, numerical gates, probe warmup, and final service warmup remain separate from rollout cost.

The worker reuses the v3.4 episode implementation with fresh environments seeded before construction, official reset, ten settling actions, 20 Hz, and the original limits. The first observation and noise seed must match across conditions before the first action. Every completed episode checks round/step, condition-independent noise hash, actual layer path, actual AE NFE, official init hash, and service/manifest identity. Valid failures are retained; technical retries are bounded. Output includes success, action count, route counts, model/RPC time, raw actions, and end-effector telemetry.

CPU tests prove dispatch and protocol behavior with instrumented fake kernels; they do not prove GPU numerical parity. As prepared, GPU preflight and closed-loop results remain pending.
