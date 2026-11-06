# idea1 v2.2 execution

All entrypoints default to dry-run. GPU execution requires `--launch` and the
independent GPT-6.1-sol medium-effort implementation acceptance. Existing experiment sources
are read-only. Use the dedicated LIBERO evaluation environment for the runner;
the scheduler starts native services in each project's existing `.venv`.

```bash
policies/libero-eval-v21/.venv/bin/python -m unittest discover -s preparations/idea1_v2_2 -p test_pipeline.py -v
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py freeze --attempt policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/ATTEMPT --extension
```

Freeze configuration is JSON with `models` keyed by `pi0`,
`xiaomi-robotic0`, `db-cogact`. Each has `protocol_source` pointing to its real
historical protocol, and either `probe_source` pointing to v2.1 `probes.json`
or `observations` containing at least 32 distinct real Full inputs with
`input_path`, `suite`, `phase`, and optional fallback `instruction`. Frozen
selection balances suite/phase by deterministic round-robin. The freeze stage
checks every official task's init6–10 and freezes hashes of the initial vectors,
native action limits parsed from the official example, source protocols and code.
π0 accepts the old Full `agent_image/wrist_image/robot_state/prompt` NPZ schema.

Replace `ATTEMPT` and `CONFIG.json` below with the accepted attempt and real
source configuration. `--extension` retains all 54 conditions (10800 episodes);
omit it for the mandatory 38 conditions (7600 episodes). Xiaomi uniform2/D2
aliases are subject to successful native parity before any rollout.

```bash
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py freeze --attempt ATTEMPT --config CONFIG.json --extension --launch
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py parity --attempt ATTEMPT --launch
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py smoke --attempt ATTEMPT --launch
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py offline --attempt ATTEMPT --launch
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py run --attempt ATTEMPT --initial --launch
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py run --attempt ATTEMPT --launch
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py benchmark --attempt ATTEMPT --launch
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py analyze --attempt ATTEMPT --launch
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py audit --attempt ATTEMPT --launch
```

Each native inference stage can use `--model MODEL`. Scheduler priority is π0
two shards, Xiaomi two shards, then DB two shards. Up to four available GPUs are
used immediately; missing capacity is scanned every 30 seconds. Completion of
owned workers immediately releases their cards to the queue. No external
process is stopped. Place an `ATTEMPT/PAUSE` marker to stop new episodes; active
episodes close atomically. Remove the marker deliberately before resuming the
same command. Ledger reservations use PID plus process start ticks; a crash
after `validated.json` recovers its valid result, including valid task failure.
Technical invalid entries require explicit diagnosis before retry, and never
become task failures automatically.

The `smoke` stage is a real LIBERO diagnostic capped at two replans per mask,
with actual H/R and native action limit preserved separately from its diagnostic
cap. It records complete short videos, executed actions and XYZ in `readiness/`,
never the formal denominator. Prior model diagnostics (including the three
already logged π0 predictions), parity reservations and smoke predictions share
the 256-prediction budget. Resume skips valid formal episodes only after
rehashing their committed validation and nested raw artifacts. A diagnosed
technical invalid can be authorized for a bounded retry with:

```bash
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_2/runtime_runner.py retry --attempt ATTEMPT --key EPISODE_KEY --diagnosis 'concrete diagnosed cause and fix' --launch
```

Before formal rollout, offline analysis, or benchmarking, the independent medium-effort reviewer writes
`ATTEMPT/runtime_acceptance.json` with schema 1, reviewer_model
`gpt-6.1-sol`, reasoning_effort `medium`, the frozen manifest hash and exact runtime
code hashes, plus `models[MODEL]` containing accepted=true, the current parity
identity and evidence paths. Authorization is per model, allowing π0/Xiaomi to
proceed while DB remains blocked. The implementation and CPU tests do not write
this authorization.

Offline calls prepare one immutable VLM cache per input/mask and branch across
all schedules with the same initial seed. Benchmark calls independent complete
predictions using actual localhost TCP transport; client and server are
co-resident in one process and this placement is recorded. A cross-attempt GPU
file lock is held throughout, with GPU exclusivity checked at observation
boundaries. CPU exclusivity is not asserted: initial and final process snapshots
are retained. Client roundtrip includes native prediction/noise compression and
ledger persistence; server_total stops before that persistence. The exact
boundaries are retained with the benchmark evidence. Per-condition measurements are exactly 32 observations ×
5 warmup + 20 measurement, rotated by observation and repeat.

`audit` recomputes the full expected task-key universe, raw artifact hashes,
telemetry/executed-prefix counts, 32-point offline coverage, parity flags and
per-point/per-repeat benchmark coverage. It exits unsuccessfully for partial
results. `analyze` supports truthful partial reports, paired task-cluster
bootstrap, native-unit full-H action errors, independent action-generation
speedup, failure-stage denominators, XYZ plots and complete-video indices.

Current runtime source receipt: [readiness/runtime_source_receipt_medium_revision.json](readiness/runtime_source_receipt_medium_revision.json).
The current frozen runtime hash is
`b87b4b671bd8f16b9644e227e0a9f8068cc7b4e5dc59bee932de9aa17cdcc678`.
The [original receipt](readiness/runtime_source_receipt.json) and its
`b82905fbdb15ef03d0fda5ae0c8a8a54fe160f7add8b1572b3ba3a16b859f602`
runtime hash remain historical evidence for the earlier failed attempt.
The frozen boundary is the explicit `common.RUNTIME_FILES` plus upstream client,
mask/decoder, resize and telemetry sources and each model's scientific source
closure and processor assets. Analysis and test changes have separate source
receipts and do not invalidate already valid raw results. CPU verification:
27/27 passed (11 independent, 7 model, 9 pipeline); real official LIBERO imports
and 50 initial states in each suite were checked without a GPU. The pipeline
implementer has not launched formal GPU episodes.
