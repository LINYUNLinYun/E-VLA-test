# E1 v2.1 execution

**Handoff guard (2026-10-16 12:59 UTC):** the four formal runners listed in
[`idea1_v2_1_luna_handoff.md`](../../idea1_v2_1_luna_handoff.md) are already active in the campaign root
below. These commands are execution recipes, not permission to duplicate them.
Before any launch, inspect the live PID plus process start time, server lease,
ledger reservations, and latest handoff snapshot. Do not change the shard count,
take over a live reservation, or terminate an existing runner without explicit
direction. The current user instruction is preparation/handoff only.

All commands run from `/root/nfs/embodied/benchmark`. The campaign root is
`policies/realtime-vla-flash/results/idea1_v2_1_multimodel_kalman/run_20261016T045100Z`.
Native model servers and parity belong to `runtime_models.py` / `runtime_server.py`;
the queue consumes their trusted localhost numpy RPC. Run parity and freeze the
measured native protocol before starting an environment episode. In particular,
configuration H and the actual action mask/output H must agree or be explicitly
documented before rollout.

```bash
policies/libero-eval-v21/.venv/bin/python -m unittest discover -s preparations/idea1_v2_1 -p test_runtime_runner.py -v
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_1/runtime_runner.py bootstrap --model xiaomi-robotic0 --port 9101 --gpu 1
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_1/runtime_runner.py probes --model xiaomi-robotic0 --port 9101 --gpu 1
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_1/runtime_runner.py masks --model xiaomi-robotic0 --port 9101 --gpu 1
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_1/runtime_runner.py formal --model xiaomi-robotic0 --port 9101 --gpu 1
```

For DB-CogACT substitute `--model db-cogact --port 9102 --gpu 2`.
`bootstrap` runs four separate official unadapted reference episodes, four adapted
Full task0/init0 smoke episodes, and eight Full task0/task5 init5 calibration
episodes. The four Full smoke episodes are durable formal keys and the formal
queue reuses them. Formal conditions always use the same model seed rule,
official init, H/R/D, preprocessing and decoded action convention. The server
owns camera rotation and DB gripper discretization. The client saves raw images
and proprioception without a second rotation.

`full-init0` can occupy an additional same-model server during bootstrap. It runs
only Full task1–9/init0 and cannot overlap the bootstrap task0 smoke or init5
calibration. For the complete formal queue, use fixed `--shard-count N` and one
`--shard-index 0..N-1` per dedicated server/GPU. All formal shards wait at the
global two-model 1040-key init0 barrier before continuing init1–4. Do not change
shard count while workers from the old allocation remain active.

Every environment attempt has a new output directory. `ledger.json` reservations
survive process restarts; valid task failures are final and cannot repeat. A
technical invalid may be retried only with `--retry-invalid --fix-note 'specific
diagnosis and fix'`, at most once per key and eight per model. A live reservation
cannot be silently taken over. Inspect a dead process and preserve its attempt
before explicitly settling its reservation; no automatic crash retry is assumed.
`reconcile --fix-note 'diagnosed dead process'` settles only dead reservations,
recovering durable validation or marking the attempt invalid. Live owners are
recognized using both PID and process start time. GPU discovery and server
leases belong to the campaign supervisor; this queue uses an already leased
server and never scans or acquires an external GPU itself.

Calibration uses no more than ten measured replans per episode, balanced across
the current measured semantic phases. Probe each allowed single layer using the
same observation/seed as Full. Xiaomi fusion layers 1–3 and DB front layers 1–2
are protected and their action impacts are explicitly missing. Twelve distinct
masks are generated directly from finite cosine/impact/random/adjacent/group
heuristics at the three predefined budgets; all twelve enter the formal queue.
The final readout layer is protected. No success based elimination occurs.
For `short_runs`, the complete mask is checked after every group insertion,
budget-filling insertion, and deduplication proposal, so separate groups cannot
join into more than three consecutive skipped layers. The CPU regression suite
now has eight passing tests, including both model depths with monotone/reversed/
shuffled layer rankings and a forced deduplication collision followed by an
illegal contiguous proposal. This fix does not launch models or evaluations.

Every episode saves complete H264/yuv420p video at 20 fps, N+1 measured simulator
states for N actions, all inference chunks, raw observations, input/prediction
hashes, contacts/grasp/object poses, phase labels and timing. Validation decodes
the whole video, checks 20 Hz simulation time and native executed prefixes, and
recomputes saved observation and prediction hashes. `reference` uses `native=True`
while the adapter slow reference is reserved for parity.

```bash
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_1/runtime_analysis.py policies/realtime-vla-flash/results/idea1_v2_1_multimodel_kalman/run_20261016T045100Z/e1/xiaomi-robotic0
```

Analysis uses an immutable source-hash subdirectory, includes all valid failures,
reports partial denominators, and samples tasks within each suite for paired
bootstrap intervals. Closed-loop timings and exclusive fixed-observation
microbenchmarks must be reported separately. Completion additionally requires
all 2600 formal identities, 8 calibration and 4 reference episodes, parity,
microbenchmark, complete trajectories/videos and provenance; partial rows or
mere file existence are not evidence of completion.
