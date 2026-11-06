# Four simple NVTX names

Status: completed; all three files exported and verified. GPU3 released, other services untouched. See `completion.json`.

Requested names: `preprocess`, `encoder`, `vlm`, `AE`.

Delivered filenames under `nsys/idea2_experiment/simple_labels_20261026/`:

- `pi05_native_jax_d10.nsys-rep`
- `pi05_fresh_prepared_d10.nsys-rep`
- `pi05_pipeline_sm16_lead3_d10.nsys-rep`

Torch paths use the unchanged full pi0.5 D10 GPU computations and runtime. Per-layer and per-iteration hook labels are disabled; existing Graph-level labels are shortened. Client compensation is off for diagnostic captures to avoid profiler-induced prediction-horizon stopping. This does not change the GPU implementation. `preprocess` still includes input transforms, tensor conversion, blocking upload, and GPU observation preparation; it is not CPU-only arithmetic. In lookahead paths it prepares the next accepted observation.

The native JAX file is an explicitly **split-stage diagnostic**, with separately JIT-compiled original image embedding, full VLM Prefill, and original D10 AE equations. Each stage waits for its output solely to give readable complete NVTX ranges. Encoder includes vision projection and prefix embedding metadata; AE includes final CPU action output processing. Warmup compares decoded actions against the original full-JIT `Policy.infer` at max≤1e-5/MAE≤1e-6. Separate compilation/fences may change performance: do not interpret these stage sums as an exact partition of the original approximately103ms whole-JIT baseline. The unchanged whole-JIT trace remains `../pi05_threepaths_01_native_jax_d10_20261026.nsys-rep`.

The published native diagnostic uses `native_stages.py` and `capture_native_dynamic.py`: original GPU observation preprocessing stays inside the encoder JIT; `dt=-1.0/num_steps` keeps the original traced runtime argument and update path. Ten changing-input/explicit-noise warmup comparisons against the original whole JIT have max=0 and MAE=0. The earlier literal-dt/outside-JIT-preprocessing attempt failed the unchanged gate and captured zero requests; it remains in `capture_execution.json`, and was not published. These two preparation/compilation corrections were made together, so the failed attempt is not evidence for attributing the error to dt alone.

Eight profiled native diagnostic requests give mean stage ranges (including each diagnostic stage's completion fence): preprocess33.03ms, encoder12.43ms, vlm35.10ms, AE24.27ms. They sum to104.83ms for this diagnostic; they do not replace the original whole-JIT measurements.

Automatic XLA `TSL` NVTX ranges are excluded from the simple diagnostic. CUDA kernels remain visible. All native work is the complete BF16 B1 H10/D10 pi0.5 LIBERO, same checkpoint; no pruning or fewer denoising rounds. Profile data is not merged into closed-loop success-rate results.

The default application NVTX domain contains exactly the four requested names in each file. CUDA's own `CCCL` domain may remain as a separate library row in Torch captures; it is not an application stage. In Torch files, encoder/vlm/AE labels originate in CUDA Graph capture and appear in GPU NVTX projection; CPU preprocess scopes prepare the next accepted frame. NVTX precapture also retains warmup/capture metadata, while actual GPU activity contains only the8 requested steady requests. Inspect the time interval containing GPU kernels, not a cold metadata range.

Resource admission first waited for an external GPU3 job. User-authorized coordination requests were sent to the actual idea1 visualization thread and v3.3 coordinator; requests were cancelled once GPU3 freed. Visualization confirmed itsGPU1/2 workers were never paused. Only GPU3 was used for these files, and both model services and lease are now released.

Download from the local machine:

```bash
mkdir -p ~/Desktop/nsys
scp 'smc-gpu:/root/nfs/embodied/benchmark/nsys/idea2_experiment/simple_labels_20261026/*.nsys-rep' ~/Desktop/nsys/
```
