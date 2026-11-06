# Stage 2 amendment: input preparation stream placement

Written before implementing or testing the new stream-placement switch, 2026-11-01.

## Evidence that changes the next action

- `stage2_b1_sm16_online_full`: clean strict pass, 100 groups, 800 requests; 11.237826 observations/s, CPU→CPU mean88.974ms, lookahead prepare mean75.138ms. The same configuration with prepared inputs achieved18.074451/s; ordinary full-GPU online replay achieved15.545039/s. These are different timing boundaries, not a direct speedup claim.
- `stage2_b2_sm16_online_full`: clean same-B diagnostic pass;12.849887 observations/s. B>1 cross-B1 equivalence still fails.
- Actual `stage2_profile_b4_sm16_online_trace.nsys-rep` has8 full ranges each for preprocess/encoder/vlm/AE,114480 real kernels, and actual16/112 Green resource attribution. AE×VLM real-kernel overlap totals1064.452181ms; AE×nextEncoder is0.
- In profile request0, AE executes122.341302–296.066245ms. The next input's `preprocess` CPU range is98.516078–320.738303ms. Its first `cudaStreamSynchronize_v3020` starts111.773705ms and lasts184.305096ms, returning immediately after AE finishes. Its first GPU operation starts296.068901ms. This directly identifies a blocking input-preparation submission path; it is not evidence that the CPU transform itself takes222ms.
- Original code recursively invokes blocking `.to(device)` on the caller's current stream. `Observation.from_dict` then performs original GPU image float conversion/normalization. The caller has restored the primary default stream after submitting the Green graphs. General synchronization behavior is documented by NVIDIA's [CUDA12.6 stream API](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-driver-api/group__CUDA__STREAM.html). The exact internal driver dependency is not established by this trace; the observed submission/wait chain is the hypothesis to test.

## Minimal single-variable test

Add an explicit default-off `--prepare-stream prefix` benchmark option. Keep the default caller-stream path as the original control. For online replay, run the unchanged `prepare_batch` on the existing Prefix stream, record its ready Event there, and enqueue a wait on the caller stream before returning Prepared. This preserves the existing Graph input-ready Event chain; no host-wide synchronization is added or removed. Blocking copies may still wait for current VLM on Prefix; that wait is necessary in this variant and remains visible.

Use the existing Prefix stream so there is no new stream lifecycle, no shared static input, no new full KV allocation, and no extra Host worker. CPU transforms, copies, image arithmetic, noise, Euler steps, model, Pool/Lease and Graph captures are unchanged. Warmup and capture remain outside timing.

After all existing model/profile processes finish, implement the switch and run CPU checks. Then smoke B1 strict and B4 same-B diagnostic; retain all failures. If valid, formally measure B1/B2/B4 online with the baseline SM16 and tuned B2SM24/B4SM20, at least100 groups each. Only the preparation stream changes within each quota comparison. Repeat the key same-card result and inspect an independent nsys capture to see whether AE×nextEncoder overlap returns and the long prepare wait shrinks.

Keep batch-shaped B1 drift and the quota-dependent KV frontier mix explicit. Better timing alone cannot establish closed-loop quality or unchanged observation freshness. No automatic promotion or tolerance change is permitted.

## Follow-up evidence before the dedicated-stream test

The Prefix-placement B1/B2/B4 short tests all passed their appropriate numerical and lifetime gates, but did not recover prepared-input performance. They remain short tests, not formal speedup results. Prefix copies must wait for the currently executing VLM, splitting CPU preparation around that wait.

Three bounded synthetic probes (`stage2_input_stream_probe`, `stage2_input_graph_stream_probe`, `stage2_input_copy_tail_probe`) all completed and released resources. Eager Green work, sleeping Green graphs, and adding a tail D2D copy **did not** by themselves make caller-stream H2D block until AE completion. Thus the real-model wait cannot be attributed generically to “Green default streams always synchronize”, “all Graphs serialize”, or D2D alone. Preserve this negative evidence; the actual model trace remains the direct evidence for its observed wait chain.

Add a second explicit placement value, `--prepare-stream dedicated`: one ordinary nonblocking input stream, created and owned by this benchmark process. The same ready-Event join applies; explicitly drain this stream during teardown, including failure before encode consumes its Event. No new thread, model arithmetic, Graph scratch, or KV slot. Test the same quotas and inputs; if it does not recover real-request performance, retain the negative result rather than treating a small copy probe as success.
