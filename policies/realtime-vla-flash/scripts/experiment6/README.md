# Experiment 6: VLM pruning diagnosis

This experiment answers three separate questions before another pruning sweep:

1. Is Experiment 5 `Control` a trustworthy baseline?
2. Which VLM blocks are locally redundant **and** action-insensitive?
3. When a pruned policy creates a new failure, which semantic task stage is
   lost relative to the paired Control rollout?

The analyses deliberately keep three evidence levels separate. Action-trace
divergence is only a proxy; simulator replay supplies semantic phase labels;
success rate remains the final pruning criterion.

## Output layout

All generated artifacts live under
`results/pi0_vlm_pruning_diagnosis/`:

- `baseline_audit/`: protocol comparison and controlled smoke-run evidence.
- `telemetry_new_failures/`: deterministic action replay with simulator state.
- `failure_semantic/`: paired Control/candidate cohorts and phase attribution.
- `gallery/`: paired videos, contact sheets, aggregate plot, and HTML index.
- `probe_observations/`: phase-stratified fixed observations.
- `layer_probes/`: Full18 and 18 single-layer interventions with shared noise.
- `layer_analysis/`: hidden-state similarity, action sensitivity, bootstrap
  confidence intervals, and candidate ranking.

## Reproducible pipeline

Set `REPO` to this repository and run commands from it. LIBERO replay needs the
Python 3.8 environment; numerical analysis uses the repository environment.

```bash
LIBERO_CONFIG_PATH="$REPO/.libero" \
PYTHONPATH="$REPO/third_party/libero" \
examples/libero/.venv/bin/python scripts/experiment6/trajectory_replay.py \
  --episode-results results/pi0_vlm_backbone_fine_pruning/episode_results.csv \
  --video-index results/pi0_vlm_backbone_fine_pruning/video_index.csv \
  --testset-manifest scripts/experiment5/testset_manifest.json \
  --paired-analysis results/pi0_vlm_pruning_diagnosis/failure_proxy/paired_failure_analysis.csv \
  --output-dir results/pi0_vlm_pruning_diagnosis/telemetry_new_failures

# Use this after changing only the phase classifier; it does not rerun MuJoCo.
examples/libero/.venv/bin/python scripts/experiment6/reclassify_telemetry.py \
  results/pi0_vlm_pruning_diagnosis/telemetry_new_failures

.venv/bin/python scripts/experiment6/failure_analysis.py \
  --episode-results results/pi0_vlm_backbone_fine_pruning/episode_results.csv \
  --video-index results/pi0_vlm_backbone_fine_pruning/video_index.csv \
  --telemetry-dir results/pi0_vlm_pruning_diagnosis/telemetry_new_failures \
  --output-dir results/pi0_vlm_pruning_diagnosis/failure_semantic

.venv/bin/python scripts/experiment6/visualize_failures.py \
  --paired-analysis results/pi0_vlm_pruning_diagnosis/failure_semantic/paired_failure_analysis.csv \
  --telemetry-dir results/pi0_vlm_pruning_diagnosis/telemetry_new_failures \
  --max-cases 12 \
  --output-dir results/pi0_vlm_pruning_diagnosis/gallery
```

Export fixed observations from successful Control rollouts, then run the layer
collector with the prompt manifest matching each task suite. The collector uses
Full18/Full18/D10, identical observations, and identical diffusion noise for
Full18 and every single-block skip.

```bash
.venv/bin/python scripts/experiment6/layer_probe.py \
  results/pi0_vlm_pruning_diagnosis/layer_probes/*.layer_probe.npz \
  --output-dir results/pi0_vlm_pruning_diagnosis/layer_analysis
```

## Decision rule

A block is only a pruning candidate when all three signals agree across tasks
and task phases: high adjacent hidden similarity, a small residual update, and
low fixed-observation action sensitivity. Candidate sets must then pass a
paired closed-loop evaluation. Do not select layers from cosine similarity
alone, and do not infer semantic failure stage from action timing alone.

`Control` in Experiment 5 means the internal Full18 VLM / Full18 Action Expert
condition. It is not automatically the canonical baseline because Experiment 5
used D3, one custom HDF5-derived init state per task, and resumed artifacts.
The baseline audit therefore reports protocol provenance alongside accuracy.
