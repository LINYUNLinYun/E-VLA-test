# Local OxyGen environment

- Official repository: https://github.com/air-embodied-brain/OxyGen
- Installed revision: `5c207f52bbd7f31b7ea4aa0fc0d647fa823b3a58`
- Environment: `.venv`, Python 3.11, shell prompt `(oxygen)`.
- Dependencies: upstream `uv.lock`, installed with `UV_LINK_MODE=copy` so the
  automatic Transformers overlays stay inside this environment.
- Cached assets: `../../shared/checkpoints/openpi` (existing LIBERO weights and tokenizer).

Activate from the repository root:

```bash
source activate_oxygen.sh
```

Validate shared-KV inference with the real local LIBERO checkpoint and a synthetic observation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/local_smoke.py
```

This checks execution, output shape, and finite action values. It does not check
robot task success, language quality, or acceleration. The upstream policy factory
currently disables action/state normalization and unnormalization, so this
benchmark path needs review before use for robot control.

Run a small continuous-batching experiment:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -m experiments.run_experiments \
  --settings continuous_batching --policies pi05_o2_libero \
  --num-denoise-steps 10 --max-decoding-steps 10 --steps-per-frame 5 \
  --total-frames 12 --gpu 0 --results-dir experiments/results_local
```

With `activate_oxygen.sh` sourced, the LIBERO experiment reuses the local cache.
PyTorch inference with real weights additionally requires checkpoint conversion;
the existing local checkpoint is JAX format. See `experiments/Experiments.md`.

Reinstall locked dependencies:

```bash
GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --locked
```

Local additions are `activate_oxygen.sh`, this document, and `scripts/local_smoke.py`.

## Validation (2026-10-09)

- `uv pip check`: all 231 installed packages compatible.
- Activation: `.venv` selected, `VIRTUAL_ENV_PROMPT=oxygen`, local asset cache selected.
- JAX 0.5.3 and PyTorch 2.6.0+cu124: GPU tensor operations passed on RTX 4090.
- OxyGen PyTorch model import passed; Transformers overlay check returned `True`.
- Real JAX LIBERO checkpoint, `pi05_o2_libero`, shared-KV inference with two
  denoising steps and two decoding steps: passed, actions `(10, 7)`, all finite,
  tokens `(2,)`. Language output from the synthetic image was not meaningful.
- No acceleration benchmark or robot rollout was performed.
