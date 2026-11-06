---
license: apache-2.0
library_name: transformers
tags:
- robotics
- vla
- pi0
- bridge-v2
- flow-matching
pipeline_tag: robotics
---

# INTACT-pi0 Bridge V2 (transformers ≥ 4.52 compatible)

A drop-in replacement for [`juexzz/INTACT-pi0-finetune-bridge`](https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge)
that works directly with current HuggingFace `transformers` (≥ 4.52, tested on
4.57). Same weights as the upstream INTACT-pi0 fine-tune; the layout has been
remapped to the new PaliGemma structure and the modeling code is vendored
self-contained (no `lerobot` dependency).

## What this is

- **Base model**: π0 (PaliGemma 3B + Gemma 1B "action expert" + flow-matching head)
- **Fine-tune**: INTACT-pi0 (AI4CE INT-ACT team) on Bridge V2, 15 epochs / ~22.7k
  steps / bs=1024 on 4×H100. Checkpoint chosen at epoch 5 (step 7565).
- **Action space**: Bridge V2 7-D per-step delta `[delta_xyz(3), delta_rpy_sxyz(3), gripper(1)]`
  after the post-processing pipeline (see "Action post-processing" below).
- **Chunk size**: 4 (`n_action_steps = 4`).

## What changed vs the upstream `juexzz` repo

1. **Modeling code is vendored** (`configuration_pi0.py`, `paligemma_with_expert.py`,
   `flex_attention.py`, `modeling_pi0.py`). No more `pip install lerobot==X` step.
2. **PR [huggingface/lerobot#1297](https://github.com/huggingface/lerobot/pull/1297) patches applied** to
   `paligemma_with_expert.py` for transformers ≥ 4.52 (`.language_model.model` →
   `.language_model`, `params_to_change_dtype` selector update).
3. **Stray `from pytest import Cache` removed** from `paligemma_with_expert.py`.
4. **State-dict keys re-mapped** to the new transformers 4.52+ PaliGemma layout
   (`paligemma.language_model.model.X` → `paligemma.model.language_model.X`, etc.).
   The conversion is documented in this repo's commit history.
5. **PI0Policy wrapper removed.** Original `lerobot.common.policies.pi0.PI0Policy`
   wrapped `PI0FlowMatching` with `Normalize` / `PreTrainedPolicy` / dataset-stats
   plumbing. INTACT was trained with `IDENTITY` normalization at the lerobot level
   and a custom Bridge V2 q01/q99 normalisation done *outside* the policy, so
   `Normalize` was effectively a no-op. We expose `PI0FlowMatching` through a
   thin HF `PreTrainedModel` (`PI0Model`) instead. Callers handle Bridge V2 input/
   output normalisation themselves.

## Bit-identity verified

Numerical parity against the upstream INTACT-pi0 sidecar (lerobot fork +
transformers 4.49) was tested on hala A6000 with **identical inputs + identical
injected noise**:

| Component         | Max-abs diff |
|-------------------|--------------|
| Raw `[delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]` (normalised) | < 1e-4 (effectively zero — 4 decimal places identical) |
| Post-processed `[xyz_m, axangle, gripper]` | 2.2e-3 (bfloat16 noise floor) |

## Usage

```python
import torch
from transformers import AutoConfig, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
config = AutoConfig.from_pretrained("petkopetkov/INTACT-pi0-finetune-bridge", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "petkopetkov/INTACT-pi0-finetune-bridge",
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to(device).eval()
```

Then preprocess your observation **as INTACT was trained** (Bridge V2 / SIMPLER
top-down frame for state, q01/q99 normalisation from `bridge_statistics.json`,
image in `[-1, 1]` float). The forward call:

```python
images = {"observation.images.top": img_bchw}  # [B, 3, 224, 224] float in [-1, 1]
state  = state_padded                          # [B, max_state_dim=32], with first 7 dims = Bridge V2 norm
tasks  = ["pick up the spoon"]                 # List[str]
chunk  = model.predict_action_chunk(images, state, tasks)  # [B, chunk_size=4, max_action_dim=32]
```

`chunk[..., :7]` is the 7-D normalised Bridge V2 action chunk. Apply the
denormalise + RPY → axangle pipeline (see `bridge_statistics.json` and INTACT's
`BridgeSimplerAdapter`) to get the runner-ready chunk.

## Action post-processing pipeline

The raw model output `[delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]` (in normalised space) is converted to runner-consumable form by:

1. **Clip first 6 dims to `[-1, 1]`** (the policy's tail extrapolations can push
   `xyz/rpy` past p99 by 8-10×).
2. **Bound-denormalise xyz_rpy** using `bridge_statistics.json` action `p01`/`p99`:
   `x = 0.5 (norm + 1) (p99 - p01) + p01`.
3. **Conjugate the RPY rotation back to the bridge frame** via the SIMPLER
   top-down rotation `R_topdown = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]`:
   `DR_bridge = R_topdown @ DR_topdown @ R_topdown^T`.
4. **Convert `DR_bridge` to axis-angle (rotvec)**.
5. **Gripper passes through unchanged.**

## Files

| File | Purpose |
|---|---|
| `config.json` | HF `PretrainedConfig` (architectures: `PI0Model`, auto_map → `modeling_pi0.PI0Model`). |
| `configuration_pi0.py` | `PI0Config` (slim, no lerobot). |
| `paligemma_with_expert.py` | PaliGemma + Gemma-expert dual-stack with shared attention. PR #1297 patches applied. |
| `flex_attention.py` | Flex-attention path (unused by default; `attention_implementation="eager"` in config). |
| `modeling_pi0.py` | `PI0FlowMatching` (verbatim NN) + `PI0Model` (HF wrapper). |
| `model.safetensors` | Re-keyed weights (6.5 GB). |
| `bridge_statistics.json` | Bridge V2 proprio + action q01/q99 quantiles for pre/post-processing. |

## License

Apache-2.0 (inherits from upstream `juexzz/INTACT-pi0-finetune-bridge`).

## Citation / credit

- **INTACT-pi0 paper** (AI4CE / INT-ACT team)
- **π0**: ["π0: A Vision-Language-Action Flow Model for General Robot Control"](https://www.physicalintelligence.company/download/pi0.pdf), Physical Intelligence, 2024.
- **Original lerobot port**: [huggingface/lerobot](https://github.com/huggingface/lerobot)
- **transformers ≥ 4.52 patches**: [PR #1297](https://github.com/huggingface/lerobot/pull/1297)
