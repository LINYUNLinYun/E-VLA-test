#!/usr/bin/env python3
"""Compare JAX flow-matching carries with and without per-step BF16 rounding.

This diagnostic keeps the official pi0 model and every model operation intact.
The only intervention is to round ``x_t + dt * v_t`` to BF16 after each Euler
update, then decode those values back to FP32 for the next JAX computation. The
BF16 branch is a carry-precision ablation, not a complete emulation of Triton.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Reuse the reference probe's private seams for identical transforms/noise and metrics.
# ruff: noqa: SLF001
try:
    from scripts.experiment8 import reference_policy_probe as probe
except ModuleNotFoundError:  # ``python scripts/experiment8/compare_jax_carry_precision.py``
    import reference_policy_probe as probe  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JAX_CHECKPOINT = probe.DEFAULT_JAX_CHECKPOINT
DEFAULT_TRITON_BASE = probe.DEFAULT_TRITON_BASE
NUM_STEPS = probe.NUM_STEPS
MODE_FP32 = "fp32"
MODE_BF16_CARRY = "bf16_after_each_euler_step"
MODES = (MODE_FP32, MODE_BF16_CARRY)


def trace_sample_actions_loop(
    model: Any,
    observation: Any,
    noise: Any,
    *,
    jax: Any,
    jnp: Any,
    model_lib: Any,
    make_attn_mask: Any,
    mode: str,
    num_steps: int = NUM_STEPS,
) -> dict[str, Any]:
    """Copy the official ``Pi0.sample_actions`` loop while recording each step.

    ``noise`` is already BF16-rounded, but represented as FP32, as in the
    reference probe. ``bf16_after_each_euler_step`` rounds only the updated
    action state; converting the rounded BF16 value back to FP32 preserves all
    other JAX arithmetic and isolates carry-value precision.
    """
    if mode not in MODES:
        raise ValueError(f"Unsupported carry mode {mode!r}; expected one of {MODES}")
    if int(num_steps) != NUM_STEPS:
        raise ValueError(f"This diagnostic is fixed to official num_steps={NUM_STEPS}")

    observation = model_lib.preprocess_observation(None, observation, train=False)
    dt = -1.0 / int(num_steps)
    batch_size = observation.state.shape[0]
    action_horizon = int(model.action_horizon)
    action_dim = int(model.action_dim)
    x0 = jnp.asarray(noise, dtype=jnp.float32)
    if tuple(x0.shape) != (batch_size, action_horizon, action_dim):
        raise ValueError(f"Expected noise shape {(batch_size, action_horizon, action_dim)}, got {tuple(x0.shape)}")

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)

    history_shape = (int(num_steps), batch_size, action_horizon, action_dim)
    x_history = jnp.zeros(history_shape, dtype=jnp.float32)
    velocity_history = jnp.zeros(history_shape, dtype=jnp.float32)
    euler_unrounded_history = jnp.zeros(history_shape, dtype=jnp.float32)
    next_carry_history = jnp.zeros(history_shape, dtype=jnp.float32)
    time_history = jnp.zeros((int(num_steps),), dtype=jnp.float32)

    def cond(carry: tuple[Any, ...]) -> Any:
        _, _, time, *_ = carry
        # Match Pi0.sample_actions' robust floating-point stop condition.
        return time >= -dt / 2

    def step(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        index, x_t, time, xs, velocities, euler_raw, next_xs, times = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_suffix_mask = jnp.broadcast_to(
            prefix_mask[:, None, :],
            (batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1]),
        )
        full_attn_mask = jnp.concatenate([prefix_suffix_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        prefix_out, suffix_out = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )[0]
        # The official Pi0 loop asserts that only the suffix output is materialized.
        del prefix_out
        velocity = model.action_out_proj(suffix_out[:, -action_horizon:])
        unrounded_next = x_t + dt * velocity
        # Do not express this as ``astype(bfloat16).astype(float32)``. XLA may
        # legally fold that pair away when the carried array itself remains
        # float32. ``reduce_precision`` makes the value-rounding intervention
        # explicit while preserving a common float32 loop-carry dtype.
        next_x = (
            jax.lax.reduce_precision(unrounded_next, exponent_bits=8, mantissa_bits=7)
            if mode == MODE_BF16_CARRY
            else unrounded_next
        )

        xs = xs.at[index].set(x_t.astype(jnp.float32))
        velocities = velocities.at[index].set(velocity.astype(jnp.float32))
        euler_raw = euler_raw.at[index].set(unrounded_next.astype(jnp.float32))
        next_xs = next_xs.at[index].set(next_x.astype(jnp.float32))
        times = times.at[index].set(time.astype(jnp.float32))
        return index + 1, next_x, time + dt, xs, velocities, euler_raw, next_xs, times

    final_index, final_x, _, x_history, velocity_history, euler_unrounded_history, next_carry_history, time_history = (
        jax.lax.while_loop(
            cond,
            step,
            (
                jnp.asarray(0, dtype=jnp.int32),
                x0,
                # Preserve the official loop's weakly typed Python scalar.
                # Its weak type affects promotion inside the time embedding.
                1.0,
                x_history,
                velocity_history,
                euler_unrounded_history,
                next_carry_history,
                time_history,
            ),
        )
    )
    return {
        "x_t": x_history,
        "velocity": velocity_history,
        "euler_unrounded": euler_unrounded_history,
        "next_carry": next_carry_history,
        "time": time_history,
        "final": final_x,
        "loop_count": final_index,
    }


def _resolve_jax_device(jax: Any, device: Any) -> Any:
    if ":" not in str(device):
        raise ValueError("--device must include an explicit device index, e.g. cuda:0")
    platform, index_text = str(device).split(":", maxsplit=1)
    jax_platform = {"cuda": "gpu", "gpu": "gpu", "cpu": "cpu", "tpu": "tpu"}.get(platform)
    if jax_platform is None or not index_text.isdecimal():
        raise ValueError("--device must be one of cuda:N, gpu:N, cpu:N, or tpu:N")
    devices = jax.devices(jax_platform)
    index = int(index_text)
    if index >= len(devices):
        raise ValueError(f"Requested {device}, but only {len(devices)} {jax_platform} device(s) are visible")
    return devices[index]


def _load_components(
    *,
    jax_checkpoint_dir: Path,
    base_triton_dir: Path,
    prompt_manifest: Path,
    device: str,
) -> tuple[tuple[Any, ...], Any, Any, Any, Any]:
    """Load one official JAX model plus the paired full Triton runtime."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts/spec"))

    from flax import nnx
    import jax
    import jax.numpy as jnp
    import spec_serve_policy
    import torch

    import openpi.models.model as model_lib
    from openpi.models.pi0 import make_attn_mask
    from openpi.shared import nnx_utils
    from openpi.training import config as train_config_lib
    from scripts.spec.triton import triton_pi0_runtime as triton_runtime

    jax_checkpoint_dir = jax_checkpoint_dir.expanduser().resolve(strict=True)
    base_triton_dir = base_triton_dir.expanduser().resolve(strict=True)
    prompt_manifest = prompt_manifest.expanduser().resolve(strict=True)
    base_weights_path, artifact_dir = spec_serve_policy._resolve_triton_base_artifact(base_triton_dir)
    train_config, input_transform, output_transform = spec_serve_policy._build_policy_components(
        config_name="pi0_libero", checkpoint_dir=artifact_dir
    )
    official_config = train_config_lib.get_config("pi0_libero")
    if type(official_config.model).__name__ != "Pi0Config":
        raise TypeError(f"Expected official pi0_libero Pi0Config, got {type(official_config.model).__name__}")
    jax_device = _resolve_jax_device(jax, device)
    with jax.default_device(jax_device):
        params = model_lib.restore_params(jax_checkpoint_dir / "params", dtype=jnp.bfloat16)
        model = official_config.model.load(params)
    sampler = nnx_utils.module_jit(model.sample_actions)
    runtime_pool = triton_runtime.TritonRuntimePool(
        base_weights_path=base_weights_path,
        manifest_path=prompt_manifest,
        num_views=2,
        chunk_size=int(train_config.model.action_horizon),
        vlm_layer_pattern="vlm_full18",
        ablation_layer_pattern="full18",
        denoise_steps=NUM_STEPS,
    )
    components = (
        jax,
        jnp,
        torch,
        model_lib,
        sampler,
        input_transform,
        output_transform,
        runtime_pool,
        train_config,
        device,
    )
    return components, model, nnx, jax_device, make_attn_mask


def _run_trace_mode(
    *,
    model: Any,
    nnx: Any,
    jax: Any,
    jnp: Any,
    model_lib: Any,
    make_attn_mask: Any,
    observation: Any,
    noise: Any,
    mode: str,
    device: Any,
) -> dict[str, Any]:
    graphdef, state = nnx.split(model)

    def run(module_state: Any, model_observation: Any, explicit_noise: Any) -> dict[str, Any]:
        module = nnx.merge(graphdef, module_state)
        return trace_sample_actions_loop(
            module,
            model_observation,
            explicit_noise,
            jax=jax,
            jnp=jnp,
            model_lib=model_lib,
            make_attn_mask=make_attn_mask,
            mode=mode,
            num_steps=NUM_STEPS,
        )

    compiled = jax.jit(run)
    with jax.default_device(device):
        return compiled(state, observation, noise)


def _trace_numpy(trace: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result = {name: np.asarray(trace[name])[:, 0] for name in ("x_t", "velocity", "euler_unrounded", "next_carry")}
    result["time"] = np.asarray(trace["time"])
    final = np.asarray(trace["final"], dtype=np.float32)
    if final.ndim == 3 and final.shape[0] == 1:
        final = final[0]
    result["final"] = final
    return result


def _stepwise_metrics(fp32: Mapping[str, np.ndarray], bf16: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "step": index,
            "time": float(fp32["time"][index]),
            "x_t_fp32_vs_bf16_carry": probe._difference_metrics(fp32["x_t"][index], bf16["x_t"][index]),
            "velocity_fp32_vs_bf16_carry": probe._difference_metrics(fp32["velocity"][index], bf16["velocity"][index]),
            "euler_unrounded_fp32_vs_bf16_carry": probe._difference_metrics(
                fp32["euler_unrounded"][index], bf16["euler_unrounded"][index]
            ),
            "next_carry_fp32_vs_bf16_carry": probe._difference_metrics(
                fp32["next_carry"][index], bf16["next_carry"][index]
            ),
        }
        for index in range(NUM_STEPS)
    ]


def run_diagnostic(
    *,
    observation_npz: Path,
    noise_npz: Path,
    prompt_manifest: Path,
    jax_checkpoint_dir: Path,
    base_triton_dir: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Execute official FP32-carry JAX, BF16-carry JAX, and paired Triton."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    preflight = probe._check_inputs(
        observation_npz=observation_npz,
        noise_npz=noise_npz,
        prompt_manifest=prompt_manifest,
        jax_checkpoint_dir=jax_checkpoint_dir,
        base_triton_dir=base_triton_dir,
    )
    observation_raw, _ = probe._load_observation(observation_npz)
    source_noise, effective_noise, noise_evidence = probe._load_reference_noise(
        noise_npz, observation_sha256=preflight["observation"]["sha256"]
    )
    components, model, nnx, jax_device, make_attn_mask = _load_components(
        jax_checkpoint_dir=jax_checkpoint_dir,
        base_triton_dir=base_triton_dir,
        prompt_manifest=prompt_manifest,
        device=device,
    )
    jax, jnp, _, model_lib, official_sampler, input_transform, output_transform, _, train_config, _ = components
    transformed = input_transform(probe._clone_observation(observation_raw))
    transformed_fingerprints = probe._tree_fingerprints(transformed)
    batched = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], transformed)
    observation_for_model = model_lib.Observation.from_dict(batched)
    explicit_noise = jax.device_put(jnp.asarray(effective_noise, dtype=jnp.float32)[np.newaxis, ...], jax_device)

    _, sample_rng = jax.random.split(jax.random.key(0))
    with jax.default_device(jax_device):
        official_actions = official_sampler(
            sample_rng,
            observation_for_model,
            num_steps=NUM_STEPS,
            noise=explicit_noise,
        )
    official_raw = probe._to_numpy_actions(official_actions)

    traces_device: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    for mode in MODES:
        traces_device[mode] = _run_trace_mode(
            model=model,
            nnx=nnx,
            jax=jax,
            jnp=jnp,
            model_lib=model_lib,
            make_attn_mask=make_attn_mask,
            observation=observation_for_model,
            noise=explicit_noise,
            mode=mode,
            device=jax_device,
        )
        traces[mode] = _trace_numpy(traces_device[mode])

    triton_raw, triton_aux = probe._run_triton_full(
        components=components,
        observation=observation_raw,
        effective_noise=effective_noise,
    )
    triton_transformed_fingerprints = probe._tree_fingerprints(triton_aux["transformed"])
    if transformed_fingerprints != triton_transformed_fingerprints:
        raise RuntimeError("JAX and Triton transformed inputs differ; refusing a mismatched carry comparison")

    fp32 = traces[MODE_FP32]
    bf16 = traces[MODE_BF16_CARRY]
    if not np.array_equal(fp32["x_t"][0], bf16["x_t"][0]):
        raise RuntimeError("FP32 and BF16-carry branches must begin from the same effective noise")
    fp32_actions7 = probe._transform_policy_actions(
        raw_actions=fp32["final"], transformed=transformed, output_transform=output_transform
    )
    bf16_actions7 = probe._transform_policy_actions(
        raw_actions=bf16["final"], transformed=transformed, output_transform=output_transform
    )

    arrays = {
        "source_noise_float32": source_noise,
        "effective_noise_bfloat16_values": effective_noise,
        "official_sample_actions_final_32d": official_raw,
        "jax_fp32_carry_x_t_32d": fp32["x_t"],
        "jax_fp32_carry_velocity_32d": fp32["velocity"],
        "jax_fp32_carry_euler_unrounded_32d": fp32["euler_unrounded"],
        "jax_fp32_carry_next_x_t_32d": fp32["next_carry"],
        "jax_fp32_carry_final_32d": fp32["final"],
        "jax_bf16_carry_x_t_32d": bf16["x_t"],
        "jax_bf16_carry_velocity_32d": bf16["velocity"],
        "jax_bf16_carry_euler_unrounded_32d": bf16["euler_unrounded"],
        "jax_bf16_carry_next_x_t_32d": bf16["next_carry"],
        "jax_bf16_carry_final_32d": bf16["final"],
        "jax_fp32_carry_actions_7d": fp32_actions7,
        "jax_bf16_carry_actions_7d": bf16_actions7,
        "triton_full18_final_32d": triton_raw,
        "triton_full18_actions_7d": triton_aux["actions7"],
        "time_steps": fp32["time"],
    }
    loop_counts = {mode: int(np.asarray(traces_device[mode]["loop_count"])) for mode in MODES}
    if any(count != NUM_STEPS for count in loop_counts.values()):
        raise RuntimeError(f"Expected {NUM_STEPS} official Euler steps, got {loop_counts}")
    metrics = {
        "official_sample_actions_vs_manual_fp32_carry": probe._difference_metrics(official_raw, fp32["final"]),
        "fp32_carry_final_vs_triton": probe._difference_metrics(fp32["final"], triton_raw),
        "bf16_carry_final_vs_triton": probe._difference_metrics(bf16["final"], triton_raw),
        "bf16_carry_effect_vs_fp32_final": probe._difference_metrics(bf16["final"], fp32["final"]),
        "stepwise_fp32_vs_bf16_carry": _stepwise_metrics(fp32, bf16),
        "output_transformed_7d": {
            "fp32_carry_vs_triton": probe._difference_metrics(fp32_actions7, triton_aux["actions7"]),
            "bf16_carry_vs_triton": probe._difference_metrics(bf16_actions7, triton_aux["actions7"]),
        },
    }
    metadata = {
        **preflight,
        "inspection_only": False,
        "model_loaded": True,
        "device_initialized": True,
        "experiment": {
            "question": "Does rounding only the Euler carry to BF16 move official JAX outputs toward paired Triton?",
            "modes": list(MODES),
            "num_steps": NUM_STEPS,
            "dt_python_value": -1.0 / NUM_STEPS,
            "loop_condition": "time >= -dt / 2 (copied from official Pi0.sample_actions)",
            "loop_counts": loop_counts,
            "time_steps_float32": [float(x) for x in fp32["time"]],
            "mode_fp32": {
                "initial_noise_dtype": "float32 values exactly representing BF16-rounded source noise",
                "carry_value_dtype": "float32",
                "carry_compute_dtype": str(fp32["final"].dtype),
            },
            "mode_bf16_after_each_euler_step": {
                "rounding_point": "after x_t + dt * v_t, before the next loop iteration",
                "effective_carry_value_dtype": "BF16 values (round-trip decoded to float32 for JAX math)",
                "carry_compute_dtype": (f"{bf16['final'].dtype}; only carry values are rounded, not model math"),
            },
            "official_loop_source": "openpi.models.pi0.Pi0.sample_actions",
            "triton_comparison": "same observation, same input/output transforms and same explicit BF16-effective noise; full18/full10 Triton output",
            "not_claimed": "The BF16-carry JAX branch isolates state-value rounding only; it is not a full Triton emulation or causal conclusion.",
            "jax_device": str(jax_device),
            "jax_backend": str(jax.default_backend()),
            "input_fingerprints_identical": True,
            "transformed_input_fingerprints": transformed_fingerprints,
        },
        "noise": noise_evidence,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "compare_jax_carry_precision.npz",
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False)),
    )
    (output_dir / "compare_jax_carry_precision.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-npz", required=True, type=Path)
    parser.add_argument("--noise-npz", required=True, type=Path)
    parser.add_argument("--prompt-manifest", required=True, type=Path)
    parser.add_argument("--jax-checkpoint-dir", type=Path, default=DEFAULT_JAX_CHECKPOINT)
    parser.add_argument("--base-triton-dir", type=Path, default=DEFAULT_TRITON_BASE)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", help="Torch/Triton and corresponding visible JAX device")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metadata = run_diagnostic(
        observation_npz=args.observation_npz,
        noise_npz=args.noise_npz,
        prompt_manifest=args.prompt_manifest,
        jax_checkpoint_dir=args.jax_checkpoint_dir,
        base_triton_dir=args.base_triton_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "metrics": metadata["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
