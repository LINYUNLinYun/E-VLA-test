# Fresh-tail quality experiments

The six conditions use D10, lead3, AE resources 16 or 32 SMs, and force the last
0, 1 or 2 Euler rounds to consume all current layer KV. The wait is for final
layer KV-ready, not final FFN completion. These are approximate algorithms.

`matched_quality.py` defaults to CPU planning. It builds 100 unique real
observations from one fixed task (`libero_object/task00/init00`), preserving each
observation's actual predecessor and original noise seed. The 12-input gate is
the first subset of that bank. Original eager full-current-KV actions are
generated before any progressive engine exists. Every variant reuses identical
explicit noise bytes and input pairs. Chain discontinuities reset to the saved
actual old observation; an additional midpoint reset is covered by an oracle.
Each variant has eight observed-schedule implementation oracle checks, distinct
from approximation deltas against fresh inference. The tolerance is fixed at
MAE <= 1e-6 and max absolute error <= 1e-5. Exactly two full KV slots are retained
per engine; closed-engine Python references must vanish before the next engine.

`quality.py` wraps the frozen original LIBERO episode client. Its default plan
contains six conditions plus unpartitioned fresh18, each with the original eight
suite/task/init cases, D10/H10/R5 and seeds. Conditions are grouped to require
seven captures rather than recapturing on each case. This grouping is for
quality evaluation, without a paired latency claim. No future observation is
provided (`next_encoder=False`). GPU, EGL, port and CPU affinity are explicit CLI
options. Plan-only execution imports no CUDA runtime.

GPU jobs must be coordinated by the root scheduler. For the assigned GPU3 gate:

```bash
CUDA_VISIBLE_DEVICES=GPU-555cde36-5e41-bc44-0c69-ea47358fd703 \
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_DEVICE_MAX_CONNECTIONS=32 \
policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.matched_quality \
  --execute --count 12 --output results/idea2/fresh_tail_20261025/matched_gate12_gpu3_cpujax \
  --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 --cpus 96-111
```

The formal `run_matched.py` supervisor requires the gate's passed result and
process receipt, checks GPU3 is idle, applies the same environment, and records
the formal command/process/exit/cleanup receipt. It publishes `MATCHED_READY.json`
only after process exit and all 600 quality rows pass structural/oracle checks.

```bash
policies/openpi/.venv/bin/python -u -m results.idea2.fresh_tail_20261025.run_matched \
  --gate results/idea2/fresh_tail_20261025/matched_gate12_gpu3_cpujax \
  --output results/idea2/fresh_tail_20261025/matched100_gpu3
```

`analyze_matched.py DIRECTORY` writes per-dimension R5/H10 MAE/max, all-input
paired MAE changes versus tail0, frontier statistics and implementation gates.
Incomplete runs remain `in_progress`. The fixed-input run deliberately includes
oracle and completion waits and does not establish a timing speedup.

The first `matched_gate12_gpu3` attempt omitted the established CPU-JAX launch
environment and failed during model loading, before any action output. Its log
and receipt are retained as `excluded_environment_setup_failure`, separate from
algorithm quality failures. The retry changes only the launch environment.

The scheduler-owned closed-loop output is `closedloop_gpu2`, using its frozen
56-case grouped plan after the matched run and GPU2 performance jobs exit.
`closedloop_gpu3` and `closedloop_gpu3_grouped` are unused alternative plans.
Do not start these alternatives. The CPU-only `watch_quality.py` updates
`reports/idea2_fresh_tail_20261025` until all 56 valid receipts and 48 bootstrap
checks are available. Task failures remain valid quality outcomes.
Source hashes are checked at execution; use a new output after source changes.
