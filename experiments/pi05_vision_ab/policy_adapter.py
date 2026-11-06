"""Native JAX pi0.5 policy with a static vision A/B configuration.

One process hosts exactly one compression configuration: the switch is read by
``openpi.models.siglip.Encoder1DBlock`` at trace time, so it must be installed
before ``create_trained_policy``.  VLM, action expert, token layout, flow
matching steps and weights are untouched.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "shared/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero"
NATIVE_H = 10
NATIVE_D = 32
NUM_STEPS = 10
EXECUTE_ACTIONS = 5


def _native_observation(obs):
    from preparations.idea1_v2_1.runtime_models import native_observation

    return native_observation(obs)


class Pi05VisionPolicy:
    def __init__(self, mode: str, keep_tokens: int, condition_id: str):
        from openpi.policies import policy_config
        from openpi.training import config as training

        from experiments.pi05_vision_ab import compression

        self.mode = mode
        self.keep_tokens = int(keep_tokens)
        self.condition_id = condition_id
        compression.install(mode, self.keep_tokens)
        train = training.get_config("pi05_libero")
        if not train.model.pi05 or train.model.action_horizon != NATIVE_H:
            raise ValueError("pi05_libero with action_horizon=10 is required")
        tick = time.perf_counter()
        self.policy = policy_config.create_trained_policy(
            train, CHECKPOINT, sample_kwargs=dict(num_steps=NUM_STEPS)
        )
        self.load_seconds = time.perf_counter() - tick
        if self.policy._sample_kwargs.get("num_steps") != NUM_STEPS:
            raise RuntimeError("native sampler kwargs mismatch")
        self._vision_call = None
        self.protocol = {
            "model": "pi05_libero",
            "checkpoint": str(CHECKPOINT),
            "backend": "native OpenPI JAX",
            "H": NATIVE_H,
            "action_dim": NATIVE_D,
            "num_steps": NUM_STEPS,
            "execute_actions": EXECUTE_ACTIONS,
            "replan_steps": EXECUTE_ACTIONS,
            "vision_condition": condition_id,
            "compression_mode": mode,
            "keep_tokens": self.keep_tokens,
            "merge_kind": "tome_deterministic",
            "layer_scope": "all",
            "grid": [16, 16],
            "state_input": "discrete language tokens (pi0.5)",
            "camera_slots": 3,
            "valid_cameras": 2,
            "load_seconds": self.load_seconds,
        }

    # -- policy path -----------------------------------------------------
    def reset(self):
        return {"ok": True}

    def _data(self, obs, instruction):
        images, state = _native_observation(obs)
        return {
            "observation/image": np.asarray(images[0]),
            "observation/wrist_image": np.asarray(images[1]),
            "observation/state": state,
            "prompt": instruction,
        }

    def predict(self, obs, instruction, suite, noise):
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape == (NATIVE_H, NATIVE_D):
            pass
        elif noise.shape == (1, NATIVE_H, NATIVE_D):
            noise = noise[0]
        else:
            raise ValueError(f"noise shape {noise.shape} != {(NATIVE_H, NATIVE_D)}")
        data = self._data(obs, instruction)
        tick = time.perf_counter()
        response = self.policy.infer(data, noise=noise)
        wall_ms = (time.perf_counter() - tick) * 1000.0
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (NATIVE_H, 7) or not np.isfinite(actions).all():
            raise ValueError(f"invalid native pi0.5 actions {actions.shape}")
        return {
            "actions": actions,
            "execute_length": EXECUTE_ACTIONS,
            "suite": suite,
            "policy_wall_ms": wall_ms,
            "policy_timing": dict(response.get("policy_timing", {})),
        }

    # -- vision / policy timing -----------------------------------------
    def _prepared_inputs(self, raw):
        """Run the policy's own input transforms once (CPU, outside timing)."""
        data = {
            "observation/image": np.asarray(raw["agentview_image"]),
            "observation/wrist_image": np.asarray(raw["wrist_image"]),
            "observation/state": np.asarray(raw["state"], dtype=np.float32),
            "prompt": raw["instruction"],
        }
        transformed = self.policy._input_transform(data)
        image = transformed["image"]
        # Model path converts uint8 to [-1,1] float32 in Observation.from_dict.
        slots = []
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            value = np.asarray(image[key])
            if value.dtype == np.uint8:
                value = value.astype(np.float32) / 255.0 * 2.0 - 1.0
            slots.append(value.astype(np.float32))
        return transformed, slots

    def make_vision_fns(self):
        """JIT the exact production vision call for one/all camera slots."""
        from flax import nnx as _nnx  # noqa: F401
        from openpi.shared import nnx_utils

        img = self.policy._model.PaliGemma.img
        jitted = nnx_utils.module_jit(img.__call__, static_argnames=("train",))

        def single(slot):
            tokens, _ = jitted(slot[None], train=False)
            return tokens

        def all_slots(slots):
            outs = []
            for slot in slots:
                tokens, _ = jitted(slot[None], train=False)
                outs.append(tokens)
            return outs

        return single, all_slots

    def make_observation(self, raw):
        """Build the batched model Observation once, outside timed regions."""
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        transformed, _ = self._prepared_inputs(raw)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
        return _model.Observation.from_dict(batched)

    def sample_actions(self, observation, noise):
        """Native sample_actions call used for T_model_ms."""
        import jax.numpy as jnp

        noise = jnp.asarray(noise, dtype=jnp.float32)
        if noise.ndim == 2:
            noise = noise[None, ...]
        return self.policy._sample_actions(self.policy._rng, observation, noise=noise)


def block_until_ready(tree):
    import jax

    jax.block_until_ready(tree)
