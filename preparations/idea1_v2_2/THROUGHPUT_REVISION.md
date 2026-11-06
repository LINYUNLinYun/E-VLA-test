# Two environment throughput execution revision

This execution mode uses one unchanged sequential `runtime_server.py` and two
independent LIBERO client processes. Scientific protocols, prediction code,
per-episode mask/seed rules, videos, XYZ telemetry and raw validation stay in the
frozen runtime. It does not supply deployment latency benchmark samples.

Only Luna schedules or switches GPUs. No command below attaches clients to an
existing server. The manager first acquires both original model shard locks and
the sole GPU lease, and refuses a busy GPU.

## Prepare and independent acceptance

Run with the evaluator Python environment from the repository root:

```text
throughput_revision.py prepare --attempt ATTEMPT --revision NEW_REVISION
```

`revision_manifest.json` binds the original attempt/manifest/frozen runtime and
the three new execution sources. Medium signs
`acceptance/MODEL.source.json` before handoff or a GPU gate. Every acceptance has
schema 1, accepted true, reviewer_model `gpt-6.1-sol`, reasoning_effort `medium`,
model, phase, revision_manifest_sha256 and nonempty evidence. Gate/pilot acceptance
also locks `phase_receipt_sha256` of `MODEL.gate_ready.json` or
`MODEL.pilot_ready.json`.

## Single card boundary handoff

`throughput_revision_handoff.py` inspects exact worker/server PID, start ticks,
commands, model/GPU/attempt, parent relationship and service lease. Scheduler
children are rejected; the first supported operational target is direct DB GPU2.
Supply explicit `--revision`, `--model`, `--gpu`, `--worker`, `--server`,
`--worker-start-ticks` and `--server-start-ticks`. Without `--execute` it only
inspects. With `--execute`, an active reservation returns
`busy_active_episode_no_signal`; retry later, never interrupt that episode.

With the exact original `episode_ledger.lock` held, no matching reserved entry
proves the last episode committed and prevents the next reservation. The helper
persists a boundary ledger snapshot, uses pidfds to terminate the precise idle
worker, confirms exit before releasing the ledger lock, then stops its precise
server and demonstrates both shard locks and the GPU lease are available.
`MODEL.handoff_complete.json` records the proof. It does not write PAUSE, alter
ledger rows or manufacture episode reservations.

## Gate, pilot and run

```text
throughput_revision.py gate --revision REVISION --model MODEL --gpu GPU
throughput_revision.py pilot --revision REVISION --model MODEL --gpu GPU
throughput_revision.py run --revision REVISION --model MODEL --gpu GPU
```

The gate reserves eight logical predictions in an independent cumulative
`readiness/MODEL/throughput_diagnostic_budget.json` under the original diagnostic
lock. Prior, parity, smoke and all throughput revisions together must stay at or
below 256. The original signed budget file is unchanged. Four reference calls use
order 0/2/1/3; two independent clients repeat order 0/1/2/3 with the same frozen
observation, Full/variant masks and two fixed seeds. Raw predictions, decoded
actions and noise must be exactly equal; hashes, NFE, schedules, cache identity,
code identity and eight-call count are checked. Startup forward ledgers are
separate evidence. Medium signs the gate receipt before pilot.

Pilot executes one fresh canonical formal key per environment. Actual artifacts
are written directly to their final parent `episodes/` locations. Until medium
signs the exact pilot-ready receipt, ledger directories point to escrow locations
without validated.json, while both owners stay alive. Dead-owner recovery cannot
misclassify an unapproved candidate. After signing, each client atomically commits
the final directory, validated hash, success and revision identity as valid. A
revision permits one pilot session only. Rejection, timeout or failure leaves the
candidate artifacts and an invalid entry requiring explicit diagnosis.

Run uses the already signed pilot and the original locked ledger, including
existing valid-artifact revalidation and invalid/retry rules. Clients drain their
current episode on a manager stop signal or peer failure. A server failure stops
the clients and requires investigation; there is no automatic technical retry.
All new valid entries retain execution revision/session metadata.

Session receipts report evaluation episodes/hour separately from startup and
human approval wall time, per-episode prediction timing sums, and maximum sampled
total GPU memory (five-second NVML sampling after server readiness, not an exact
allocator high-water mark). Independent single-request benchmark execution stays
separate and must never overlap this mode on the same GPU.
