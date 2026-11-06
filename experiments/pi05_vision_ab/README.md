# pi0.5 vision A/B compression experiment

Implements the protocol in
[`docs/experiments/vision_ab_experiment_plan/experiment_plan.md`](../../docs/experiments/vision_ab_experiment_plan/experiment_plan.md).

Conditions: `C0` (native), `A128`/`A64` (attention residual branch compressed),
`B128`/`B64` (attention + MLP residual branches compressed). The mapping is the
deterministic 2×2-cell ToMeSD variant from section 4.2; every branch is
unmerged to the full 256-token layout before the original residual is added.
Weights, VLM/action expert, token layout, flow-matching steps, action execution
and the camera call pattern are unchanged.

## Layout

| file | purpose |
|---|---|
| `config.yaml` | static protocol, conditions, budgets, paths |
| `compression.py` | thin experiment wrapper over `openpi.models.vision_compression` |
| `policy_adapter.py` | one native `pi05_libero` policy per process; vision/timing handles |
| `server.py` | localhost RPC model server (one static condition per process) |
| `evaluation_adapter.py` | episode manifests, paired action-noise keys, resume-safe JSONL |
| `checks.py` | CPU algorithm checks, K=N parity, compressed-config sanity |
| `bench.py` | 5 rounds × 200 synchronized calls for one condition |
| `summarize.py` | latency/SR aggregation, fixed selection rule, figures, report |
| `run.py` | stage driver: audit, check, collect_obs, bench, smoke, screen, select, confirm, system, report |

The model-side implementation is
`policies/openpi/src/openpi/models/vision_compression.py` plus a small edit in
`siglip.py::Encoder1DBlock.__call__`; mode `off` executes the original path.

## Commands

```bash
cd /root/nfs/embodied/benchmark
PY=experiments/pi05_vision_ab/run.py
# environment/checkpoint audit
policies/libero-eval-v21/.venv/bin/python $PY --stage audit --gpu 0
# CPU + real-checkpoint correctness (must pass before rollouts)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PWD/policies/openpi/src \
  policies/openpi/.venv/bin/python $PY --stage check --gpu 0
# >=64 real observations (>=16 per suite) for latency
policies/libero-eval-v21/.venv/bin/python $PY --stage collect_obs --gpu 0
# 5 conditions x 5 rounds x 200 calls (one GPU, no concurrent GPU work)
policies/openpi/.venv/bin/python $PY --stage bench --gpu 0
# closed-loop stages (3 free GPUs; each worker hosts its own model server)
policies/libero-eval-v21/.venv/bin/python $PY --stage smoke --gpus 0,2,3 --wait
policies/libero-eval-v21/.venv/bin/python $PY --stage screen --gpus 0,2,3
policies/libero-eval-v21/.venv/bin/python $PY --stage select
policies/libero-eval-v21/.venv/bin/python $PY --stage confirm --gpus 0,2,3
policies/libero-eval-v21/.venv/bin/python $PY --stage report
```

Run directories are `results/pi05_vision_ab/<run_id>/`; every stage is
resume-safe (episode JSONL is appended with `flock`+`fsync` and deduplicated by
`(stage, condition, suite, task, initial_state, repeat)`).

## Protocol notes

- Episodes pair task, official initial-state index and explicit Gaussian action
  noise. The noise key is a SHA-256 of `(root_seed, suite, task, init,
  repeat_seed, replan, "policy")` and deliberately excludes the condition id.
- `env.seed(7)` is called before `env.reset()` and `set_init_state`, so the
  environment RNG does not depend on worker order or previous episode length.
- Timing is synchronized (`jax.block_until_ready`) and scoped:
  `vision_all` includes stem/posemb/all 27 blocks/matching/merge/unmerge/final
  norm/projection over the three sequential camera slots (two valid cameras);
  compile time is recorded separately and never mixed into steady state.
- The selection rule is the fixed one from plan section 8.3: correctness first,
  relative SR ≥ 0.70 preferred, 0.60–0.70 aggressive, speedup tie within 2%
  chooses K=128. At most one A and one B enter confirmation.
- There is no runnable pi0.5 self-speculative draft/verifier in this workspace,
  so the system-level matrix is recorded as *not evaluated*; screening and
  confirmation SR are not substituted for it.
