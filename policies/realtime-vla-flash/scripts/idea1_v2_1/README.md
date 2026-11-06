# v2.1 E4 CPU preparation

This directory is independent of `scripts/idea1_v2`; it never edits historical
results. It imports only the old read-only schema validator and the frozen old
Kalman implementation. No model/environment packages are imported and no model
calls, GPU work, or environment steps are available here.

Execution status (2026-10-16): the authorized full CPU experiment is running in
`results/idea1_v2_1_multimodel_kalman/run_20261016T045100Z/e4_revision2/e4_kalman`.
Earlier frozen revision directories are retained with revision notes. The active
directory's receipts, not this README or unit tests, determine actual scientific
completion.

## Commands and safe defaults

Run from `policies/realtime-vla-flash`, using its existing Python environment:

```bash
.venv/bin/python -m scripts.idea1_v2_1
.venv/bin/python -m scripts.idea1_v2_1 tune --dry-run
.venv/bin/python -m unittest scripts.idea1_v2_1.test_cpu -v
```

No command, or `--help`, prints help. All stage commands preview by default;
`--launch` is necessary to do work. `--dry-run` overrides `--launch` and does not
read source data, create directories, freeze candidates, or search. The preview
reports five folds, 211 maximum candidate slots, 634 calibration and 6345
evaluation points. Shell command examples below are for a **later execution**,
not an instruction to launch during preparation.

Read-only integration smoke (first two actual predictions in each suite, no
results written; this is not the full preflight):

```bash
.venv/bin/python -m scripts.idea1_v2_1 sample-check \
  --source results/experiment11_layer_skip_prefix_20261010 --launch
```

Use a new attempt directory for the explicit stages, in this order:

```bash
e4_output=results/idea1_v2_1_multimodel_kalman/REPLACE_WITH_UNIQUE_ATTEMPT/e4_kalman
e4_source=results/experiment11_layer_skip_prefix_20261010
e4_v2=results/idea1_v2_pi_alternation_kalman_20261014/luna_20261015T030200Z
.venv/bin/python -m scripts.idea1_v2_1 freeze --output "$e4_output" --launch
.venv/bin/python -m scripts.idea1_v2_1 reproduce --output "$e4_output" --source "$e4_source" --v2 "$e4_v2" --launch
.venv/bin/python -m scripts.idea1_v2_1 tune --output "$e4_output" --source "$e4_source" --launch
.venv/bin/python -m scripts.idea1_v2_1 evaluate --output "$e4_output" --source "$e4_source" --launch
.venv/bin/python -m scripts.idea1_v2_1 decompose --output "$e4_output" --launch
```

Only `tune` supports `--resume`: it reuses complete candidate receipts after
matching manifest/parameters/source/code hashes. Interrupted candidate files are
not rewritten. Other stages create exclusive outputs; interrupted `evaluate` or
`decompose` must be inspected and a new attempt used, never overwritten silently.
Freezing includes all Python files in both the old and new analysis directories;
editing either afterwards requires a new attempt. This is analysis provenance,
not a substitute for E1's whole-model/backend provenance.

`reproduce` runs the old full read-only source validator (6979 unique replans,
200 episodes, 634/6345 split, action prefixes, official noise and input hashes,
20 Hz telemetry and baseline manifests) and checks original Full inference
chunks against captured Full chunks at absolute tolerance `1e-6`, zero relative
tolerance. V14's original capture is the hashed shadow `predictions.jsonl`
event; no independent V14 closed-loop inference log is asserted. Full current
phase must match telemetry. It reproduces every old raw mean and covariance at
`atol=1e-9, rtol=1e-8` before issuing the gate receipt. Source paths and original
v2 files are hashed and revalidated before tuning/evaluation. A missing or changed
source, mismatch, or numerical invalidity stops the relevant stage. The gate
does not yet independently audit every old derived CSV cell; that remains in
the final audit backlog.

## Filter and bounded selection

`filter.py` implements `x=[u,w]`, with per-action command changes (not measured
EEF velocity), Joseph update, trace-normalized R, fixed `P0=diag(R,R)`, and
positive definite covariance checks with condition limit `1e12`. Physical dt
is fixed to 0.05 s. Only already executed, clipped command history up to each
replan cutoff enters the filter; EEF, phases, and chunks do not enter the update.
Forecasts use a copy of state/covariance and a separate `g` transition, including
covariance propagation. Raw/clipped predictions, forecast covariance and latest
innovation per replan are saved; complete per-action innovation traces are not
yet exported.

The manifest freezes 49 isotropic `(q_u,q_w)` combinations and the exact rule
for expanding the best three into `3 g * 3 beta * 6 shape` hypotheses each.
There are 211 maximum slots; three stage2 hypotheses duplicate their stage1
parents and are not rescored, so at most 208 distinct hypotheses are evaluated.
Candidates do not expand based on evaluation results. Shapes use first command
differences for R and second differences for Q, never joining episodes. `half`
means `0.5 I + 0.5 normalized_empirical`; `full` means the normalized empirical
shape after the eigenvalue floor (metadata records both empirical weight and
shrinkage toward identity). Each
fold fits only task0 episodes from the other four official inits, shared across
all suites; final fitting uses all task0 episodes. Training episode IDs,
matrices/eigenvalues/condition/shrinkage/hashes, each fold/suite score, and
failures are saved. A zero-variation covariance explicitly uses identity.

Ranking first prefers candidates whose prefix L2 p95 **and** proxy lateral p95
are no more than 5% above the frozen old baseline in **every** held-out-init
fold, then prefix episode→task→suite equal mean, lateral p95, complexity and a
deterministic hash. Percentiles are sample weighted. Stage1 retains the first
three by that same preregistered rule. The final receipt records fold scores,
noise-grid boundary wins and whether any qualified prefix improvement exists.
No passing-tail candidate yields `parameters=null`, blocking evaluation. A
passing-tail candidate that fails to beat the old mean can be evaluated to
document a negative result and is marked `no_qualified_improvement`.

## Diagnostics and limits

`evaluate` saves all 6345 task1–9 prediction blocks once, with raw Full/V14,
HoldLast, source input/noise IDs, clip fractions, and post hoc current phase / EEF
geometry annotations. These are reused, previously inspected data, not a blind
test. Evaluation does not select or refit any candidate.

`decompose` implements all four pairs and saves pointwise signed/absolute axes,
axis signs, L2, amplitude difference/ratio, direction cosine/angle, cumulative
control-target path proxy delta, signed along-path and lateral errors, and
distance. Below-1-mm steps and >90°/reversing adjacent tangents have undefined
direction/proxy components (NaN with validity masks), never artificial zeros.
Vectors are normalized commands times 50 mm; neither the predicted blocks nor
their cumulative proxies are observed EEF trajectories. Along-path lead is
not proof of time lead. Time matching is explicitly an oracle diagnostic with
fixed Full offsets 4–9 for all shifts -3..+3 inside window1–12, equal overlap,
tie/ambiguity reporting, zero-shift and residual errors. Positive shift matches
a later predicted offset to the current reference offset.

`diagnostics.py` annotates merged semantic phases independently of past-only EEF
straightness (5 steps, 20-mm path threshold, 0.95 straightness), with 10-step and
0.90/0.98 sensitivities, speeds and turn angles. ±5-action transition windows
can inspect future *labels* only as explicitly post hoc annotations, never
filter inputs. Future offsets retain the current replan annotation, not a
claim about future true phases. Offset and 1–5/6–12/13–50 equal-mean and sample
p50/p95 summaries are implemented, but not final report inference.

The final decomposition stage now additionally generates:

- paired suite-stratified task-then-episode bootstrap confidence intervals;
- primary and sensitivity phase × straightness × transition × offset tables,
  plus descriptive task-fixed-effect amplitude/clip/history adjustment;
- offset plots, decomposition distributions, coverage-labeled heatmaps and
  deterministic deduplicated cases with actual Full EEF and separate proxies;
- complete per-action evaluation innovation/posterior/covariance traces;
- saved-array L2 mean/tail/denominator audits, report and artifact hashes.

Human visual verification of semantic phase labels remains an explicit final
review item; the program does not claim this has happened. Final computational
receipts initially use `computational_complete_human_review_pending`, preserving
this gap until actual reviewed images/videos and uncertainties are recorded.

Preparation verification uses tiny synthetic tests for no future leakage,
forecast-copy isolation, effective g, old dt unit invariance, PD and invalid
covariance rejection, held-out-init exclusion, candidate bounds, known
along/lateral geometry, shift recovery/ambiguity and EEF history cutoffs. It
does not benchmark model performance or launch tuning.
