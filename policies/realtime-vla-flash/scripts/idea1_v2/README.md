# idea1 v2

This isolated package implements CPU Kalman calibration/evaluation and budgeted
V14,V14,Full D3/50/5 rollout adapters. Importing it or using the default CLI never
initializes CUDA or LIBERO. All persistent commands require `--launch`.

From the repository root:

```bash
python3 -m unittest scripts.idea1_v2.test_cpu -v
python3 -m scripts.idea1_v2 --help
python3 -m scripts.idea1_v2 launch --output results/idea1_v2_pi_alternation_kalman_20261014/NEW_ATTEMPT
```

After explicit experiment authorization, use a unique output attempt, then
`freeze --launch`, `e4 --launch`, `smoke --gpu 0 --launch`, and `launch --launch`.
`resume --launch` continues the same attempt. E3 uses `--mode phase` and is gated
on all 200 E2 episodes, at least 160 successes and completed E4. `analyze`,
`render`, and `audit` also default to dry-run and require `--launch` to write.

The single-card smoke spends 64 switch-parity logical predictions, eight static
baseline episodes, and four E2 episodes which become part of the 200 E2 keys.
Weights, cached input digests, noise, source timelines, code hashes, full video
decoding, actual model/mask and executed action prefixes are checked.

Global discovery/leases/budget reservations live under the new campaign's
`status/`, never under historical results. Discovery is shared and limited to
one scan each 30 seconds (user update, 2026-10-15). A crashed reservation retains its conservative
spending charge. After diagnosing a dead owner, use `reconcile --fix-note TEXT
--launch`, then `resume --retry-invalid --fix-note TEXT --launch`. Valid success
and valid task failure are immutable; each invalid episode gets at most one
explicit retry, with at most eight retries campaign-wide.

Within the 30-second discovery cooldown, callers reuse the most recent shared
idle-card snapshot. A just-discovered idle card can therefore enter smoke/launch
immediately without another hardware query. GPU leases remain mandatory. Three CPU
regression tests cover cached discovery, discovery followed by smoke, and the
30-second boundary plus manifest/state timestamps; the readiness suite now has
24 passing tests. The shared interval comes from `GPU_DISCOVERY_INTERVAL_SECONDS`
in `core.py`; both the runner and execution manifest use that value.

CPU integration tests and source audits do not establish GPU correctness.
CUDA/graph parity and actual LIBERO baseline reproduction have not been run in
this implementation session. They are enforced before a formal matrix can
start. CPU validation attempts explicitly marked `gpu_launch_eligible=false`
cannot be used to launch a GPU experiment.

Offline `analyze` now writes per-task success, suite/overall paired bootstrap,
GPU/prepare/request timing summaries, furthest-phase counts and candidate events.
`render` writes the four-part Chinese report, measured XYZ figures and bounded
event clips for the preregistered selected cases. Repeating derived analysis/media
is supported; frozen Kalman calibration and raw episode outputs remain protected.
`audit` revalidates every episode (including full video decoding), exact 200-key
coverage, E4 arrays/splits/parameters, source and code hashes, smoke and ledger.
Task failures remain valid; diagnostics never alter success or gate the protocol.
Actual Markdown Preview Enhanced playback still requires an editor check and is
reported separately from automatic artifact validation.

The 2026-10-15 readiness revisions change code hashes. Do not resume an old frozen
attempt with the revised code or rewrite its manifest. Luna must `freeze` a new
unique attempt and run E4 in that attempt before smoke/E2. Previous E4 outputs are
preserved as historical CPU results; no automatic migration is implemented.
In particular, `luna_20261015T020649Z` was frozen with the old 300-second code and
its completed E4 is historical; use another unique attempt and rerun freeze/E4
for this 30-second revision. Do not rewrite that old manifest or its outputs.
