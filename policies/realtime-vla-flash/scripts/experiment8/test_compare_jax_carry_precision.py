from __future__ import annotations

from typing import Any

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.experiment8.compare_jax_carry_precision import MODE_BF16_CARRY
from scripts.experiment8.compare_jax_carry_precision import MODE_FP32
from scripts.experiment8.compare_jax_carry_precision import NUM_STEPS
from scripts.experiment8.compare_jax_carry_precision import _run_trace_mode
from scripts.experiment8.compare_jax_carry_precision import _trace_numpy
from scripts.experiment8.compare_jax_carry_precision import trace_sample_actions_loop


class _FakeLLM(nnx.Module):
    def __call__(self, xs, **kwargs):
        del kwargs
        if xs[1] is None:
            return (None, jnp.zeros((1,), dtype=jnp.float32)), None
        return (None, xs[1]), None


class _FakePaligemma(nnx.Module):
    def __init__(self):
        self.llm = _FakeLLM()


class _FakeModel(nnx.Module):
    action_horizon = 1
    action_dim = 1

    def __init__(self):
        self.PaliGemma = _FakePaligemma()

    def embed_prefix(self, observation):
        del observation
        return (
            jnp.zeros((1, 1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.bool_),
            jnp.ones((1,), dtype=jnp.bool_),
        )

    def embed_suffix(self, observation, x_t, time):
        del observation, time
        return (
            x_t,
            jnp.ones((1, 1), dtype=jnp.bool_),
            jnp.ones((1,), dtype=jnp.bool_),
            None,
        )

    def action_out_proj(self, suffix):
        return suffix * jnp.asarray(0.317, dtype=jnp.float32) + jnp.asarray(0.12345, dtype=jnp.float32)


class _FakeModelLib:
    @staticmethod
    def preprocess_observation(rng, observation, *, train):
        del rng, train
        return observation


@struct.dataclass
class _FakeObservation:
    state: Any


def _fake_make_attn_mask(input_mask, mask_ar):
    del mask_ar
    batch, length = input_mask.shape
    return jnp.ones((batch, length, length), dtype=jnp.bool_)


def test_bf16_mode_rounds_only_after_euler_and_changes_following_velocity() -> None:
    model = _FakeModel()
    observation = _FakeObservation(state=jnp.zeros((1, 1), dtype=jnp.float32))
    noise = jnp.asarray([[[0.5]]], dtype=jnp.float32)

    fp32_device = trace_sample_actions_loop(
        model,
        observation,
        noise,
        jax=jax,
        jnp=jnp,
        model_lib=_FakeModelLib,
        make_attn_mask=_fake_make_attn_mask,
        mode=MODE_FP32,
    )
    bf16_device = trace_sample_actions_loop(
        model,
        observation,
        noise,
        jax=jax,
        jnp=jnp,
        model_lib=_FakeModelLib,
        make_attn_mask=_fake_make_attn_mask,
        mode=MODE_BF16_CARRY,
    )
    fp32 = _trace_numpy(fp32_device)
    bf16 = _trace_numpy(bf16_device)

    assert int(np.asarray(fp32_device["loop_count"])) == NUM_STEPS
    assert int(np.asarray(bf16_device["loop_count"])) == NUM_STEPS
    assert fp32["x_t"].shape == bf16["x_t"].shape == (NUM_STEPS, 1, 1)
    assert fp32["velocity"].shape == bf16["velocity"].shape == (NUM_STEPS, 1, 1)
    assert np.array_equal(fp32["x_t"][0], bf16["x_t"][0])
    assert np.array_equal(fp32["velocity"][0], bf16["velocity"][0])
    assert np.array_equal(fp32["euler_unrounded"][0], bf16["euler_unrounded"][0])
    assert not np.array_equal(fp32["next_carry"][0], bf16["next_carry"][0])
    assert not np.array_equal(fp32["x_t"][1], bf16["x_t"][1])
    assert not np.array_equal(fp32["velocity"][1], bf16["velocity"][1])
    assert np.array_equal(
        bf16["next_carry"],
        np.asarray(bf16["next_carry"], dtype=jnp.bfloat16).astype(np.float32),
    )
    assert float(fp32["time"][0]) == pytest.approx(1.0)
    assert fp32["time"].shape == (NUM_STEPS,)


def test_trace_numpy_squeezes_batch_and_preserves_float32_final_actions() -> None:
    trace = {
        "x_t": np.zeros((NUM_STEPS, 1, 1, 1), dtype=np.float32),
        "velocity": np.ones((NUM_STEPS, 1, 1, 1), dtype=np.float32),
        "euler_unrounded": np.ones((NUM_STEPS, 1, 1, 1), dtype=np.float32),
        "next_carry": np.ones((NUM_STEPS, 1, 1, 1), dtype=np.float32),
        "time": np.arange(NUM_STEPS, dtype=np.float32),
        "final": np.zeros((1, 1, 1), dtype=np.float32),
        "loop_count": np.asarray(NUM_STEPS, dtype=np.int32),
    }
    result = _trace_numpy(trace)
    assert result["x_t"].shape == (NUM_STEPS, 1, 1)
    assert result["final"].shape == (1, 1)
    assert result["final"].dtype == np.float32


def test_jitted_nnx_trace_returns_each_step_on_cpu() -> None:
    trace = _run_trace_mode(
        model=_FakeModel(),
        nnx=nnx,
        jax=jax,
        jnp=jnp,
        model_lib=_FakeModelLib,
        make_attn_mask=_fake_make_attn_mask,
        observation=_FakeObservation(state=jnp.zeros((1, 1), dtype=jnp.float32)),
        noise=jnp.asarray([[[0.5]]], dtype=jnp.float32),
        mode=MODE_FP32,
        device=jax.devices("cpu")[0],
    )
    assert int(np.asarray(trace["loop_count"])) == NUM_STEPS
    assert np.asarray(trace["x_t"]).shape == (NUM_STEPS, 1, 1, 1)


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported carry mode"):
        trace_sample_actions_loop(
            _FakeModel(),
            _FakeObservation(state=jnp.zeros((1, 1), dtype=jnp.float32)),
            jnp.zeros((1, 1, 1), dtype=jnp.float32),
            jax=jax,
            jnp=jnp,
            model_lib=_FakeModelLib,
            make_attn_mask=_fake_make_attn_mask,
            mode="other",
        )
