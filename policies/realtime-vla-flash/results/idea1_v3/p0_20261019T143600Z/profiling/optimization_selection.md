# P0 engineering selection



Common complete suite/task/init keys: 4 / 4.

A total episode seconds: 77.097; B: 49.550.

Shared service startup/load: 30.020 s. Standalone total cost A=107.117, B=79.570 s (episode includes lazy compile/warmup and final drain).

Success and executed action count agreement: False.

A = consolidated compressed NPZ; B = consolidated uncompressed NPZ. Scientific configuration unchanged.

Both use the same serial worker, persistent model and no global ledger lock.

Episode wall includes reset, simulation, RPC, telemetry, encoding, saving and drain. Server startup is in worker events.

environment_step_and_capture_s contains telemetry/video enqueue; these intervals must not be summed.

RPC includes model and tensor-copy time; CUDA sub-stages are nested and not additive to RPC.

GPU/CPU utilization and disk-wait signals are unavailable unless separately collected; no utilization-based speed claim.

No asynchronous writer: queue length is zero; synchronous save and drain remain on the main timeline.

Output array equality check below is required before selecting B.

Exact A/B consolidated array equality: False.

Selected execution: A.

B requires at least 10% common-workload improvement and output parity; otherwise retain A.

This first A/B ordering includes first-use prompt/kernel setup in A. No reversed-order confirmation is present: A is retained and B is not declared faster.

No warm-request latency conclusion until microbenchmark is complete. P0 episodes are excluded from scientific tables.
