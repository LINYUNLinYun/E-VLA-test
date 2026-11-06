# Resident runtime preparation

Status: CPU ownership contracts passed; GPU parity, peak-memory measurement and
scientific integration are pending. Nothing in this package is imported by the
active `vlm_within_suite_v1` experiment.

`PromptRuntimeOwner` keeps one initialized Pi0 runtime and all non-language
weight tensor references. At an acknowledged episode boundary it replaces only
prompt-shaped scratch and the language embedding, regenerates both RoPE buffers
with the pinned constructor's arithmetic, and rebuilds the native CUDA graph.
Vision and fixed-horizon decoder buffers retain their objects. Same-length
prompts update embedding contents without reallocating their graph. A failed
transition, invalid scratch shape, or replaced invariant weight makes the owner
unusable for further inference.

The future adapter must pass every cache that owns captured graphs, serialize
the global VLM hook, reset pending proposals and per-path residual history, and
update the wrapped backend's prompt-length field before calling its ordinary
input preparation. Existing schedule tensors do not depend on prompt length;
their runtime-identity keys can remain stable. Shape-graph initialization must
be recorded separately from full model initialization. The current fixed-
condition protocol must be separately versioned before integrating this helper.

`backend_adapter.py` now prepares that low-level attachment. Install it around
the pinned backend before the first preparation and provide a reviewed memory
retainer factory. Its episode boundaries reject pending actions; prompt changes
inside an episode are refused. On a new prompt shape it rebinds scratch, updates
the backend length first, then runs the original embedding and observation
transforms. This bypasses the original destructive length-change branch while
preserving its input preparation. Captured graph caches are invalidated and
schedule tensor caches stay live. A lost retention floor or externally replaced
runtime is a terminal adapter fault. This does not reset the outer policy's
residual history, supply a multi-condition RPC server, or validate mask changes.

`memory_floor.py` prepares the retention component. Its default plan
adds at least 2 GiB or 25% headroom, rounded up to 256 MiB, so an 8 GiB measured
working requirement gives a 10 GiB minimum retained floor. It deduplicates live
invariant CUDA storage, keeps an additional buffer alive across scratch changes,
and rejects a plan when working memory plus that buffer exceeds reviewed
capacity. For synthetic 8 GiB working / 6 GiB invariant storage, the buffer is
4 GiB: idle storage is at least 10 GiB and estimated active peak is 12 GiB.
The retained floor is a minimum, not an active-peak upper bound. Allocation
failure, another device, aliasing, undersized storage or changed addresses
cannot be credited as a retained floor. Final shutdown releases the buffer;
condition switches and analysis do not.

The components are not integrated into a resident server or the active experiment.
The actual CUDA allocation, NVML footprint and retention through graph changes
remain untested. Measure the real invariant, scratch, graph-pool and driver
footprint before choosing a plan; `available_bytes` must exclude the existing
keeper, other reviewed allocations and device safety margin. The working profile
must include all active pipeline allocations, including renderer/environment
contexts, graph pools and scratch, rather than counting weight tensors alone.
The assigned EASE GPU load was observed around 11.5 GiB during evaluation; that
is not a per-process measurement, measured maximum, or universal model size.
Neither a logical lease nor CPU fake-storage evidence
establishes physical retention on a GPU.

Before scientific use, compare frozen and resident numeric outputs at unchanged
`atol=rtol=1e-5` on the finite four-input gate and saved real inputs spanning
every available prompt. Include short/long/short prompt transitions, mask and
episode switches, Full/D2/D2 cadence, residual resets, seed/noise identity and
actual NFE. Separately verify physical invariant-weight storage addresses,
one live CUDA context/PID, no duplicate full weights, and retained memory across
idle periods. CPU references and shape checks do not establish these GPU facts
or path quality.

CPU command (no torch import or GPU work):

```bash
policies/libero-eval-v21/.venv/bin/python -B -m unittest -v \
  preparations.ease_insights_20261103.resident_runtime_v1.test_backend_adapter \
  preparations.ease_insights_20261103.resident_runtime_v1.test_memory_floor \
  preparations.ease_insights_20261103.resident_runtime_v1.test_prompt_runtime
```

The shape oracle reads the pinned original constructor's allocation expressions
independently of the new shape table. Ten contracts cover all changed scratch
fields, both attention dtypes, short/long/short ownership, same-length prompt
updates, graph-cache clearing, episode boundaries and device-operation failures.
Eight additional fake-storage contracts cover the user example, combined peak
budget, storage aliases, physical-address changes, devices and allocation errors.
Seven adapter contracts exercise repeated episodes, one model initialization,
retention across handoffs, original input preparation, prompt restrictions and
terminal faults. These integration fixtures use a CPU stub of the pinned
backend; only the future GPU replay can establish numerical equivalence with
the actual backend.
