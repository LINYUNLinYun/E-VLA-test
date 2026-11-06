# Complete runtime NVTX ranges

Status: complete and verified. GPU3 released; no pending model jobs. `completion.json` contains the file hashes and72 verified core-stage envelopes (24 per path).

The previous simple-label files only renamed graph-construction NVTX scopes. Nsight projected those onto individual graph nodes, producing fragmented bars; actual replay had no module scope. This version removes capture-time scope aliases, adds one same-thread NVTX scope around each real `CUDAGraph.replay()`, and uses node tracing with `nvtx-live`, not `nvtx-precapture`. No extra owner-thread CUDA waits are added. Both ordinary stream execution and existing Pool/KV/Event protocols remain unchanged.

Names remain `preprocess`, `encoder`, `vlm`, `AE`. CPU encoder/vlm/AE ranges mean **submission**, not GPU completion. Use the GPU NVTX projection for execution envelopes. Each projected envelope starts at the first associated kernel and ends at the last, including gaps; it is not summed kernel busy time.

Before model capture, a synthetic format test retained81 actual graph nodes. The official `nvtx_gpu_proj_trace` report contains exactly9 projected ranges (3 per module), and every range's endpoints match its9 kernels' first/last timestamps. Thus it tests range grouping, not merely the set of label strings. It is not pi0.5 performance evidence. Initial env setup and a CSV-name parsing error were preserved; the already-correct GPU sample was reused after fixing CSV normalization.

Native JAX retains the previous validated split-stage diagnostic and original dynamic dt expression, with10 same-input/noise checks against the original whole JIT. Its CPU scopes include stage completion fences; they do not replace the original whole-JIT baseline. Disabling precapture also removes cold metadata ranges. Torch uses original GPU computations, without extra stage synchronization; compensation is off only in the profiling client.

Target files in `nsys/idea2_experiment/runtime_ranges_20261026/`:

- `pi05_native_jax_d10.nsys-rep`
- `pi05_fresh_prepared_d10.nsys-rep`
- `pi05_pipeline_sm16_lead3_d10.nsys-rep`

Completion requires model execution exit0 and original source hashes unchanged;8 runtime encoder/vlm/AE ranges for each Torch path; official GPU projections with one envelope per replay; exact envelope endpoint checks against actual retained kernels; native numerical gates unchanged. Do not publish merely because labels have the correct names.

## Verified results

All three files contain8 CPU scopes and8 GPU projected envelopes for each of the four requested names. Every encoder/vlm/AE projection is level0 with no child ranges; its first and last GPU operation timestamps exactly match the official projection, and its individual kernels remain present. Native retains52,520 kernels; each Torch file retains104,936 kernels. Ten native comparisons against the original whole JIT have decoded MAE/max0.

For example, the first pipeline vlm is **one38.179574ms GPU envelope containing902 kernels**, while its CPU submission scope is1.917128ms. Those are different quantities. The previous capture-node aliases are absent; cold metadata ranges are no longer imported into the display.

Initial native and fresh captures completed normally. After the fresh service exited, NVML briefly showed96% utilization with1MiB memory and zero processes/contexts; the strict idle guard stopped before starting the pipeline. The two valid files were retained. Only the pending pipeline was resumed after idle settled. `canonical_execution.json` combines passed stages; the original stop and `pipeline_resume_execution.json` remain separate. Admission now waits up to30s for a process-free utilization sample to settle without relaxing exclusivity.

Download the new files from a local terminal:

```bash
mkdir -p ~/Desktop/nsys
scp 'smc-gpu:/root/nfs/embodied/benchmark/nsys/idea2_experiment/runtime_ranges_20261026/*.nsys-rep' ~/Desktop/nsys/
```

Close the previously opened reports and reopen the new files. Read full module execution envelopes on the **GPU NVTX projection** row. The same-name CPU range remains a short submission range for asynchronous Torch replay. The native file remains the previously described validated split-stage diagnostic, not a replacement for original whole-JIT latency measurements.
