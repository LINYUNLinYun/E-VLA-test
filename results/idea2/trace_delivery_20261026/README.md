# Requested π0.5 three-path Nsight delivery

All three paths use complete π0.5 LIBERO, D10/H10, RTX4090 GPU3. These are short diagnostic captures, not new latency estimates or closed-loop success-rate samples. Each capture begins after ten warm requests and contains eight steady requests; loading, JIT, and Graph capture are outside the recorded range.

| File under `nsys/idea2_experiment/` | Actual path |
|---|---|
| `pi05_threepaths_01_native_jax_d10_20261026.nsys-rep` | Original OpenPI JAX `Policy.infer`, complete fresh KV on each recorded observation; no lookahead. Preprocessing and CPU action completion are inside each request. |
| `pi05_threepaths_02_fresh_prepared_d10_20261026.nsys-rep` | Actual `ContinuousPolicy`, `fresh18_no_encoder`: Graph/condition preparation enabled, next input prepared during the current request, current Encoder→full VLM→AE sequential. This is the configuration associated with the historical approximately71ms RPC figure, not a claim that a profiled request must take71ms. |
| `pi05_threepaths_03_pipeline_sm16_lead3_d10_20261026.nsys-rep` | Actual `ContinuousPolicy`, lead3/AE16/Prefix112, per-round mixed old/new KV, actual next Encoder overlap. Client prefix compensation is off for this diagnostic capture; the GPU computation is identical to the compensated implementation. |

The native capture replays the frozen recorded bank with its explicit period13 noise schedule. Paths02/03 run actual continuous LIBERO Object/task0/init0 with real camera exchange and the original target-chunk noise rule; their inputs can diverge after actions diverge. Inputs/noises are not matched across all three captures. They are suitable for inspecting the execution mechanism, not a matched-input end-to-end latency comparison. Native JAX and Torch are not assumed bitwise equivalent.

The first pipeline attempt used the control-step compensation client. Profiler overhead triggered its valid `prediction_horizon_exhausted` stopping rule before eight requests were captured. The incomplete attempt, its original failed execution receipt, and the truncated trace are retained, not counted as performance/quality evidence. The diagnostic replacement disables client compensation, which only changes action application indices; it preserves the model, D10, GPU runtime, lead3,16/112 resources, real camera exchange, and two-slot protocol. `recovery_execution.json` records this replacement. The original100-case quality results remain untouched.

In paths02/03, expand `Input/Prepare[next_observation_for_request_N+1]`, `Prefill/VisionEncoder+Projector+Metadata`, `Prefill/VLM[release_KV_frontier]`, and `AE/iteration_*`. Inputs prepared during requestN belong to requestN+1. The request label also records the actual input control step. A CPU range ending does not prove GPU completion: expand CUDA HW / Kernel rows.

`CaptureDrain/last_request_only` explicitly drains outstanding GPU work at the end of the eighth recorded request. It includes a real next-Encoder tail but perturbs that last RPC. No delay or drain is inserted into the other seven recorded requests. Nsight node tracing can also slow Host submission and alter overlap; use profiler-free historical measurements for latency and throughput conclusions.

The wrappers only add diagnostic NVTX / profiler controls and enable the runtime's existing capture labels. Frozen model, runtime, sampler, KV, and control source hashes are checked before/after. No data from these profile episodes is merged into the completed100-case quality study. `execution.json` records commands, actual configurations, completion, and GPU release.

## Verified capture contents

The delivered three files passed `completion.json`:8 request ranges intersecting the actual GPU capture in each file;52,544 /104,936 /104,936 CUDA kernels respectively. The first two completed stages remain in `execution.json`; the third valid stage is in the passed `recovery_execution.json`. Runtime source snapshots match, and the GPU lease is released.

Direct CUDA Graph launch correlations classify all104,824 model Graph kernels in each Torch trace, with no unclassified Graph kernels. The small remaining kernels include input preparation and GPU bookkeeping. Actual interval intersections over the eight captured requests:

| Trace | AE×VLM | AE×Vision |
|---|---:|---:|
|02 fresh, prepared|0ms|0ms|
|03 pipeline|269.007486ms|0ms|

Thus03 confirms AE×VLM but **does not witness AE×next Encoder in this particular capture**. All eight real next observations arrived before AE finished; however, the measured Host prepare calls took39.78/37.61/53.85/54.61/41.32/54.03/53.64/49.98ms, and all eight post-prepare Event queries found AE complete. Node tracing perturbs Host launch timing and the overlap opportunity. Do not substitute these profiled timings for profiler-free performance.

For a separate actual-kernel witness in which both intersections are present, the previously audited file remains available:

`nsys/idea2_experiment/online_sm16_prepared_witness_20261025_v1.nsys-rep`

It uses preprepared recorded current/future inputs, not real closed-loop camera arrivals. It demonstrates the two-stage GPU mechanism, not input freshness or end-to-end deployment latency. Its independent audit is `results/idea2/online_encoder_20261025/prepared_witness_analysis/witness_summary.json`.

Download from a local shell with SSH host `smc-gpu`:

```bash
mkdir -p ~/Desktop/nsys
scp 'smc-gpu:/root/nfs/embodied/benchmark/nsys/idea2_experiment/pi05_threepaths_0*.nsys-rep' ~/Desktop/nsys/
```

Optional prior two-stage mechanism witness:

```bash
scp smc-gpu:/root/nfs/embodied/benchmark/nsys/idea2_experiment/online_sm16_prepared_witness_20261025_v1.nsys-rep ~/Desktop/nsys/
```
