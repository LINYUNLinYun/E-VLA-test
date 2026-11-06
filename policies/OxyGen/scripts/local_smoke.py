"""Validate the local JAX setup and shared-KV inference with cached LIBERO weights."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / "../../shared/checkpoints/openpi"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import numpy as np
import torch

from experiments.common.setup import create_policy
from experiments.common.workload import create_synthetic_observation

print(json.dumps({"python": sys.version.split()[0], "jax": jax.__version__,
                  "jax_devices": list(map(str, jax.devices())), "torch": torch.__version__,
                  "torch_cuda": torch.cuda.is_available()}, indent=2), flush=True)
checkpoint = Path(os.environ["OPENPI_DATA_HOME"]) / "openpi-assets/checkpoints/pi05_libero"
if not (checkpoint / "params/manifest.ocdbt").is_file():
    raise FileNotFoundError(f"Local checkpoint missing: {checkpoint}")
print(f"Loading real checkpoint: {checkpoint.resolve()}", flush=True)
policy = create_policy("pi05_o2_libero", checkpoint_dir=checkpoint)
observation = create_synthetic_observation("pick up the red cup", seed=0, policy_config="pi05_o2_libero")
result = policy.infer_text_actions_shared_kv(observation, num_steps=2, max_decoding_steps=2)
actions = np.asarray(result["actions"])
assert actions.shape == (10, 7), actions.shape
assert np.isfinite(actions).all(), "Non-finite actions"
print(json.dumps({"status": "PASS", "actions_shape": list(actions.shape),
                  "actions_finite": True, "tokens_shape": list(np.asarray(result["tokens"]).shape),
                  "text": result["text"]}, indent=2), flush=True)
