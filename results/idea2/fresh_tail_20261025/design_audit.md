# Fresh-tail D10 CPU design audit

Status: conditional design approval; implementation/GPU gate not yet inspected. Auditor performed CPU source reads only. Existing runtime and previous experiment artifacts remain unchanged.

## Selected minimal implementation (parent decision)

Use the existing per-slot `layer_ready[-1]` Event; full-fresh means all 18 layers of current K/V are ready, **not** final VLM FFN completion. Override `_capture_ae` only and temporarily wrap runtime.snapshot. At step9 (tail1) or step8 (tail2), wait before calling original snapshot: external.wait inside capture, ordinary stream.wait_event during warmup. Restore runtime.snapshot in finally. tail0 directly delegates without installing any hook. Reject force_old with tail>0 before submitting GPU work. Existing Pool prefix_done and lifetime receipts stay unchanged. The anti-stale diagnostic must delay the new producer **before last-layer KV-ready**, not only before final FFN/completion.

The following full-producer Event designs were reviewed alternatives and are not the selected implementation.

## Reviewed alternatives and common requirements

Fix lead=3, D10, AE resources in {16,32}, fresh-tail rounds in {0,1,2}. Keep original timestep/dtype/noise/action path and two KV slots. At zero-based AE step `10-tail`, insert exactly one GPU Event wait **before snapshot(frontier, ...)**. All remaining rounds then snapshot frontier=18. Waiting only inside the wrapped original velocity is too late: frozen progressive_kv.py already snapshots before invoking that function. Do not overwrite frontiers to 18 without a real producer dependency.

A simpler alternative to adding a Prefix Graph node is a persistent per-slot tail-ready Event recorded in `prefill()` on prefix_stream immediately after the original prefix graph replay (after super.prefill is also safe), before AE graph launch is submitted. The AE Graph contains an external wait on that persistent Event. This adds no CPU synchronization, preserves the whole AE Graph, and ensures the current Event record has been submitted before the consumer launch. Keep the original fresh per-request prefix_done Event for Pool/lifecycle records; never substitute a persistent reused Event for that generation receipt.

If using the requested last-node `external.mark` in the Prefix Graph, place it after `language_prefix()` returns and **inside** capture. The existing `with labels, publishing_layers(...), torch.cuda.graph(...)` exits the graph context before publishing_layers exits, so putting mark in publishing_layers.__exit__ would silently place it outside capture. Persistent Events must be initialized before AE capture, retained per physical slot, and not replaced during reset or replay. Prefix warmup/capture/replay must establish completion before AE warmup waits. Restore any monkeypatched functions in finally, including failed capture.

For tail=0 preserve the original no-wait path. If all variants intentionally acquire an extra producer Event record for fair instrumentation, declare it and verify tail=0 remains original numerical semantics; frontiers can still change due to timing.

## Ownership, ordering, and failure hazards

- Existing prefill reset Event and lead3 layer-ready dependency remain mandatory. The host must enqueue the current producer before the consumer. A cached completed Event by itself is not a generation counter.
- No same-slot next generation may be submitted before both reader leases and the previous producer finish. Existing reserve/drain protocol enforces this; do not bypass it to drive graphs directly in the measured path.
- Never clear/reinitialize the persistent Event while its captured Graph survives. Reset invalidates Pool identities, not captured Event addresses. On the first post-reset request the producer must again be submitted before the AE consumer.
- Tail wait observes the **current/new** slot producer, not old producer and not ahead Vision. Waiting for full Prefix completion is conservative but satisfies the full-fresh tail requirement. layer_ready[17] only guarantees all KV writes, not final VLM FFN completion; do not silently conflate the two experiments.
- Current Prefill has no dependency on tail AE, so the proposed edge is acyclic. Next Encoder waits current Prefill plus AE1 and may overlap the fresh tail; do not change those edges or make current Prefill wait on next Encoder.
- Keep old and new leases until AE completes, with two different reader-tail Events. Even after the fresh transition, eagerly releasing old KV is unnecessary scope expansion.
- `force_old=True` overrides snapshots to zero in the frozen runtime. Reject its combination with tail>0 or document a diagnostic exception; otherwise the tail constraint is false despite the wait. The serial matched-frontier oracle need not use the asynchronous tail wait: it replays the already-observed schedule after Prefill completion.
- Existing iteration_mark precedes snapshot. A snapshot wrapper inserting wait will include that wait in the first fresh iteration's Event duration, which is correct if labeled iteration including synchronization. It does not isolate pure compute or wait duration. Additional before/after Events, if used, must be matched across measured configurations.
- Retain delayed Green Context cleanup: all caller tensors/Graphs/Events must be released before gc/empty_cache and pair.close after outer bench/serve returns. Never destroy Contexts while capture-exception traceback holds tensor locals.

## Minimal validation gate

1. CPU adapter checks: validate only D10/tail0,1,2 and lead3 for this matrix; every scoped monkeypatch restored on success/failure; per-slot Event objects distinct and captured addresses stable; no edits to frozen source hashes.
2. GPU gate (parent executes): SM16 and SM32, tails0/1/2; alternate changed image/state/noise contents at stable tensor addresses, rotate both physical slots repeatedly, and reset/bootstrap then replay again. Same instruction and mask/layout contract throughout each epoch.
3. For every tail>0 request, assert all recorded tail frontiers are exactly18. Do not require earlier frontiers to equal tail0: added waits or scheduling can change their observed values. All ten rounds, full KV builds, two slots, reader-release separation, and zero steady-state recapture still hold.
4. Matched-frontier serial oracle for each configuration, changing input and both slots, same max1e-5/MAE1e-6 tolerance as previous experiment. Report fresh-only output difference separately; tail freshness does not imply whole-trajectory fresh equivalence because noisy actions carry prior-round conditioning.
5. Anti-stale gate: after a previously completed replay, reuse the same slot/Event with changed KV contents and intentionally delay this generation's producer on its stream before completion. Consumer must not pass the fresh-tail boundary using the previous generation's signaled Event. Validate tail frontier18, oracle output, and Event/Kernel order. Include after-reset replay. This diagnostic delay is never included in performance runs.
6. Ensure AE tail waits for current producer and no deadlock, then final drain/reset and process exit0 with cleanup.passed. GPU numerical passed JSON is insufficient if process cleanup fails.
7. Compare all six configurations on the same physical GPU and identical Graph/prepare/decode settings. Warmup/capture outside timing; independent runs/order balance when resource contexts differ. Report 8-request all-work, continuous throughput, AE+VLM spans, first-fresh-round wait-containing duration, old-layer fractions, fresh tail invariants, CPU submission, output error, and selected quality/SLO only if authorized. Do not rank new-GPU measurements directly against prior GPU0/GPU2 absolute numbers.

No claim of implementation correctness or Event-generation safety is made until the above narrow gate passes. No GPU was launched by this auditor.

## Implemented adapter CPU review

The adapter at SHA256 `2763c283363869fdd1c358dfd6b1c81a1b8d9fd17b4a418276d9a7200059faf3` passed static review and six pure-Python fake-module checks (SM16/32 × tail0/1/2). `cpu_hook_audit.json` records the checks; `cpu_hook_audit.py` is reproducible without importing real Torch or initializing CUDA. Wait placement, warmup/capture APIs, untouched tail0 path, force_old rejection, and snapshot hook restoration under injected failure passed. This does not replace the GPU anti-stale and matched-frontier gates.

Two reporting/configuration limits: the adapter currently permits runtime step counts other than D10, so the matrix must enforce D10/lead3 through its fixed backend/conditions; `fresh_tail_wait_ms` is the Event span from tail-round start to completed wait, including small marker/scheduling overhead. Policy resource reconfiguration retains pending old contexts until outer serve returns; report their allocation/context lifetime and do not label such cases independent cold processes.
