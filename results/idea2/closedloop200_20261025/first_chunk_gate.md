# First chunk client-reuse gate

Status: **passed**. CPU-only audit; no model/environment execution.

Checked 80 episodes (64 new, 16 reused), 16 client batches, and 2960 request seed/source records. All 20 initial states match across four arms. Decoded and normalized initial actions have max error 0 and MAE 0 (limits 1e-5 / 1e-6).

All episodes bootstrap with fresh KV, request/source IDs restart at zero, and episode/epoch advance within each batch. Server episode numbers also advance across batches; Pool epochs can restart when configuration reconstructs the Pool. Frozen source/checkpoint metadata, original seeds, settle10, H10/R5/D10, limits, clean exits and render guards pass.

Valid failures are retained, each from its single attempt:

- `sm32_tail0/libero_object/task00/init02`: action limit at 280 steps.
- `sm32_tail0/libero_spatial/task00/init01`: action limit at 220 steps.
- `sm32_tail0/libero_spatial/task00/init03`: action limit at 220 steps.

No subsequent chunks were scanned. This gate supports safe Python/import reuse in the measured first chunk; it does not force later free trajectories to coincide.
