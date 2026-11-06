# EASE fixed fast-path independent confirmation

The frozen candidates are native OpenPI π0 Full, A (`pi0_keep24_2_front_middle`, vision skip `[8,9,10]`), and B (`pi0_keep24_1_uniform`, skip `[5,14,23]`). Blocks are original one-based IDs. Both paths preserve 24/27 vision blocks and the original full VLM and action expert; D10, H50, R5. There is no residual compensation, Full refresh, learned draft, phase switch, or online path selection.

The discovery evidence is historical init0 (A 35/40, B 36/40; Goal A 10/10 versus B 8/10, Long A 7/10 versus B 9/10). The prospective directions are fixed **Goal A > B** and **Long B > A**. Other task/suite comparisons are descriptive. All three conditions run the same 40 official tasks with independent init20–24, 200 episodes each. The pilot uses init20 and is included in the 200. Pilot outcomes do not select candidates or change categories. This is exploratory confirmation, not the official 50-init benchmark.

## Invocation

Run from `/root/nfs/embodied/benchmark`. Preparation does not load a model or allocate a GPU:

```bash
policies/libero-eval-v21/.venv/bin/python -m preparations.ease_insights_20261103.fastpath.runner prepare --output RUN
policies/libero-eval-v21/.venv/bin/python -m preparations.ease_insights_20261103.fastpath.runner run --output RUN --condition pi0_full --stage pilot --port 9231
```

The second command prints exact commands. The single external scheduler assigns an idle GPU and owns `/tmp/embodied-benchmark-gpu-GPU-UUID.lock`, then launches one condition:

```bash
policies/libero-eval-v21/.venv/bin/python -m preparations.ease_insights_20261103.fastpath.runner run --output RUN --condition pi0_keep24_2_front_middle --stage pilot --gpu-uuid GPU-UUID --port 9231 --workers 2 --launch
```

Use `--stage confirm` to retain completed init20 and add init21–24; each condition reaches 200. Optional `--shard-index I --shards N` assigns whole tasks to an external partition; local worker sharding preserves whole tasks. `--gpu INDEX` resolves to a UUID, but explicit UUID is preferred. The UUID is passed to CUDA. EGL device selection comes from a clean subprocess enumerating EGL CUDA ordinals and CUDA driver UUIDs, so numeric NVIDIA/EGL indices are never assumed identical. No model context is created by enumeration. An existing compute process on the assigned UUID stops launch.

## Scientific and execution records

`manifest.json` freezes current implementation hashes, checkpoint metadata/assets, exact masks, case lists, hypotheses, original noise function, bank identity, and reusable same-process selected-scan Full parity evidence. The reused gate is explicitly historical evidence; its relevant implementation and checkpoint hashes must match. Unrelated historical source hashes do not restrict the new experiment. The model service performs its own real-observation explicit-noise repeatability check at each launch.

The original `idea1_v3_1.rollout.run_episode` remains unchanged. A thin wrapper seeds Python/NumPy **before** environment construction in addition to the original reset seeding. Fresh environments, official init vectors, seed7, 10 settling actions, 20Hz, controller scales, 256² cameras, and native suite step limits remain. A shared gate compares the full first observation digest and first noise seed across conditions **before the first inference**. Each episode retains original identity, init vector and BDDL hashes, actions, per-request model/RPC timing, telemetry, success, steps, and an additional receipt with artifact hashes and GPU UUID.

Valid failures are completed scientific results. Resume verifies the manifest, case identity and artifact hashes, then skips them. A crash after scientific completion repairs only receipt bookkeeping, without replaying the outcome. Technical failures retain `.invalid.UUID` directories and have at most three attempts across resumes. Protocol identity failures stop immediately. Two workers share a serialized B1 service and use independent environment state; environment reuse is disabled.

Per-launch `runs/*/preparation.json` separates service loading, parity/first compilation, and 20 warmup requests from closed-loop latency and wall time. `runs/*/result.json` reports closed-loop wall time, preparation-inclusive total wall time, newly completed episodes, and worker count. Cold or new shape compilation cannot be silently called steady-state latency; the per-request data remain available. No videos or observation images are saved; policy camera rendering and telemetry/actions remain enabled.

## Validation completed before launch

```bash
python3 -m unittest preparations.ease_insights_20261103.fastpath.test_protocol -v
```

The CPU checks cover independent case splits, task shards, pilot inclusion, paired observation/noise rejection, valid-failure resume, and artifact/manifest mismatch rejection. Passing these tests is preparation evidence only. Actual success, task preference, reproducibility and speed must come from the scheduled closed-loop runs.
