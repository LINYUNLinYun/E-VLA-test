# v2.2 native runtime API

`runtime_models.NativePolicy(model_id, output_dir=None)` supports `pi0`,
`xiaomi-robotic0`, and `db-cogact`. `protocol`, `layers`, `calls` are public.

`predict(obs, instruction, seed, skip=(), schedule=None, category='formal',
reference=False, probe=False, cache_path=None)` returns actions (H,7 numpy),
timing (`vision_ms`, `vlm_ms`, `action_expert_ms`, `action_generation_ms`,
`server_total_ms`, input processing/decode/diagnostic fields), input_hash,
noise_hash (initial noise only), noise_shapes, audit (actual timestep, updates,
NFE and sampler mapping), prediction_path and noise_path when persistence is
enabled. `schedule` is `{kind:'flow'|'native'|'ddim_custom', m:float|null,
num_steps:int}`. `None` selects the native control. `reset()` is stateless.

`prepare_condition(obs,instruction,skip=(),reference=False,probe=False)` computes
one model/mask-specific immutable conditioning object. `sample_condition(prepared,
seed,schedule,category='offline',cache_path=None)` samples from a fresh noise copy.
Prepared objects are local runtime handles, never JSON serializable. They cannot
be shared across policies/masks/inputs. These calls allow offline reuse only;
benchmark uses independent predict calls. Shared branches receive the same seed.

`schedules.normalize_schedule(model_id,schedule)` validates and canonicalizes the
schedule. `flow_schedule(model_id,m)` returns actual nodes/dt. DB actual training
timestep mapping is available only after native sampler construction and appears
in result audit. Native D2 and custom m=.5 are distinct DB conditions.

Raw LIBERO observations use the same keys as v2.1. No caller-side rotation,
gripper conversion, or state preprocessing is needed.

`category='warmup'|'measurement'|'benchmark'` selects minimal action generation:
no per-step trace tensor copies, trace events, or action-network hooks. Only
outer stage CUDA events remain. Other categories capture instrumented RMS traces;
their total action generation time includes capture, with no subtraction. DB
minimal NFE is inferred from the independently audited sampler, and explicitly
has `actual_NFE=null`; instrumented calls capture actual network timesteps/NFE.

`parity(obs,instruction,seed=7,skip=(),schedules=...)` checks all passed schedules
for cached/independent and minimal/instrumented agreement, compares native D2 and
Dnative with unmodified upstream references, compares flow u50/native D2, and
executes native→D2→all schedules→native. It reports `actual_predictions`, errors,
rtol=1e-4/atol=1e-5 and per-output actual NFE. Reserve `4*len(schedules)+8` calls
per point/mask; caller must cover two real points and both masks (<=192 calls
for <=10 schedules). `native_reference(..., return_details=True)` is a single
eager upstream prediction, including π0 (no hidden graph warmups), and persists
noise/prediction artifacts separately when output_dir is set.

Provenance uses shared `provenance.collect_checkpoint_assets` and
`collect_scientific_sources` in freeze and runtime. `scientific_code_hashes`
includes v2.1 preprocessing/adapters and the conservative local execution source
closure. `loaded_scientific_sources` additionally identifies actual loaded model,
processor/tokenizer and callable transform source files (including HF cache paths).
`checkpoint_assets` binds tokenizer/processing code/config/normalization and π0
suite-specific prompt manifests and each referenced embedding tensor.

`server_total_ms` stops before prediction/noise NPZ compression/writes and ledger
persistence. `request_wall_ms` includes those operations, excludes RPC transport
and caller logs. Action generation excludes VLM prefill and decode. These exact
boundaries are repeated in `protocol.timing_boundaries`; persistence is included
in caller-side roundtrip as well.
