#!/usr/bin/env python3
"""Compare official JAX pi0_libero sampling with the full Triton action path.

The command is intentionally explicit about the noise tensor: it requires an
observation NPZ and a submodule-probe NPZ containing the noise generated for
that exact observation.  ``--preflight-only`` validates provenance and local
dependencies without importing JAX, Torch, OpenPI, or initializing a device.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

# The serving stack exposes its shared transform/artifact helpers as private
# functions; using those exact implementations is intentional for comparability.
# ruff: noqa: SLF001
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JAX_CHECKPOINT = Path(
    "/root/nfs/embodied/benchmark/shared/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero"
)
DEFAULT_TRITON_BASE = Path("/root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted/base")
SCHEMA_VERSION = 1
NUM_STEPS = 10
ACTION_HORIZON = 50
ACTION_DIM = 32
OUTPUT_ACTION_DIM = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_scalar_text(value: Any, *, name: str) -> str:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{name} must be a scalar string value, got shape={value.shape}")
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _load_observation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Observation NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = ("agent_image", "wrist_image", "robot_state", "prompt")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Observation NPZ is missing keys: {missing}")
        observation = {
            "observation/image": np.array(archive["agent_image"], copy=True),
            "observation/wrist_image": np.array(archive["wrist_image"], copy=True),
            "observation/state": np.array(archive["robot_state"], copy=True),
            "prompt": _read_scalar_text(archive["prompt"], name="prompt"),
        }
        for key in ("observation/image", "observation/wrist_image", "observation/state"):
            if observation[key].dtype.kind not in "buif":
                raise ValueError(f"Observation {key} must be a numeric array")
            if not np.isfinite(observation[key]).all():
                raise ValueError(f"Observation {key} contains NaN or infinity")
        if "metadata_json" in archive:
            metadata = json.loads(_read_scalar_text(archive["metadata_json"], name="metadata_json"))
        else:
            metadata = {}
    evidence = {
        "path": str(path),
        "sha256": sha256_file(path),
        "prompt": observation["prompt"],
        "shapes": {
            key: list(observation[key].shape)
            for key in ("observation/image", "observation/wrist_image", "observation/state")
        },
        "dtypes": {
            key: str(observation[key].dtype)
            for key in ("observation/image", "observation/wrist_image", "observation/state")
        },
        "source_metadata": metadata,
    }
    return observation, evidence


def _float32_to_bfloat16_roundtrip(array: np.ndarray) -> np.ndarray:
    """Return float32 values represented exactly after BF16 round-to-nearest-even.

    Triton's input preparation casts the supplied noise to BF16. Feeding this
    round-tripped tensor to JAX ensures both implementations start from the
    same numeric noise values rather than comparing float32 JAX noise to BF16
    Triton noise.
    """
    values = np.ascontiguousarray(array, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    bf16_bits = (rounded >> 16).astype(np.uint16)
    return (bf16_bits.astype(np.uint32) << 16).view(np.float32)


def _load_reference_noise(path: Path, *, observation_sha256: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Noise NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "noise" not in archive:
            raise ValueError(f"Noise NPZ has no 'noise' tensor: {path}")
        source_noise = np.asarray(archive["noise"], dtype=np.float32)
        if "metadata_json" not in archive:
            raise ValueError("Noise NPZ has no metadata_json provenance")
        metadata = json.loads(_read_scalar_text(archive["metadata_json"], name="metadata_json"))
    if source_noise.shape == (1, ACTION_HORIZON, ACTION_DIM):
        source_noise = source_noise[0]
    if source_noise.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"Expected noise shape {(ACTION_HORIZON, ACTION_DIM)}, got {source_noise.shape}")
    if not np.isfinite(source_noise).all():
        raise ValueError("Noise contains NaN or infinity")
    source_hash = sha256_array(source_noise)
    recorded_noise_hash = metadata.get("noise_sha256")
    if recorded_noise_hash and recorded_noise_hash != source_hash:
        raise ValueError(
            f"Noise bytes do not match submodule-probe metadata: recorded={recorded_noise_hash}, computed={source_hash}"
        )
    recorded_observation_hash = metadata.get("observation_sha256")
    if recorded_observation_hash != observation_sha256:
        raise ValueError(
            "Noise provenance belongs to a different observation NPZ: "
            f"recorded={recorded_observation_hash!r}, current={observation_sha256}"
        )
    effective_noise = _float32_to_bfloat16_roundtrip(source_noise)
    evidence = {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "source_dtype": str(source_noise.dtype),
        "shape": list(source_noise.shape),
        "source_noise_sha256": source_hash,
        "effective_bfloat16_noise_sha256": sha256_array(effective_noise),
        "effective_noise_cast": "float32 -> BF16 round-to-nearest-even -> exactly representable float32",
        "probe_metadata": metadata,
    }
    return source_noise.copy(), effective_noise, evidence


def _check_inputs(
    *,
    observation_npz: Path,
    noise_npz: Path,
    prompt_manifest: Path | None,
    jax_checkpoint_dir: Path,
    base_triton_dir: Path,
) -> dict[str, Any]:
    observation, observation_evidence = _load_observation(observation_npz)
    source_noise, effective_noise, noise_evidence = _load_reference_noise(
        noise_npz, observation_sha256=observation_evidence["sha256"]
    )
    if source_noise.shape != effective_noise.shape:
        raise AssertionError("Noise cast unexpectedly changed tensor shape")

    jax_checkpoint_dir = jax_checkpoint_dir.expanduser().resolve()
    base_triton_dir = base_triton_dir.expanduser().resolve()
    params_dir = jax_checkpoint_dir / "params"
    jax_norm_stats = jax_checkpoint_dir / "assets/physical-intelligence/libero/norm_stats.json"
    triton_weights = base_triton_dir / "base_weights.pkl"
    triton_norm_stats = base_triton_dir / "assets/physical-intelligence/libero/norm_stats.json"
    for label, path in (
        ("JAX Orbax params", params_dir),
        ("JAX checkpoint norm stats", jax_norm_stats),
        ("converted Triton base weights", triton_weights),
        ("converted Triton norm stats", triton_norm_stats),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if prompt_manifest is not None:
        prompt_manifest = prompt_manifest.expanduser().resolve()
        if not prompt_manifest.is_file():
            raise FileNotFoundError(f"Prompt-cache manifest does not exist: {prompt_manifest}")
        manifest = json.loads(prompt_manifest.read_text(encoding="utf-8"))
        prompts = manifest.get("prompts", {})
        prompt_entry = prompts.get(observation["prompt"])
        if prompt_entry is None:
            raise ValueError(f"Observation prompt is absent from prompt manifest: {observation['prompt']!r}")
        embed_relative = prompt_entry.get("embed_path") if isinstance(prompt_entry, dict) else None
        if not isinstance(embed_relative, str) or not embed_relative:
            raise ValueError(f"Prompt manifest entry has no embed_path: {observation['prompt']!r}")
        embed_path = (prompt_manifest.parent / embed_relative).resolve()
        if not embed_path.is_file():
            raise FileNotFoundError(f"Prompt-cache tensor does not exist: {embed_path}")
    else:
        manifest = None
    jax_norm_hash = sha256_file(jax_norm_stats)
    triton_norm_hash = sha256_file(triton_norm_stats)
    if jax_norm_hash != triton_norm_hash:
        raise ValueError(
            "Raw JAX checkpoint and converted Triton base norm_stats.json differ; "
            "cannot guarantee identical normalization transforms: "
            f"jax={jax_norm_hash}, triton={triton_norm_hash}"
        )
    dependencies = {
        name: importlib.util.find_spec(name) is not None for name in ("jax", "flax", "torch", "openpi_client")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "inspection_only": True,
        "model_loaded": False,
        "device_initialized": False,
        "observation": observation_evidence,
        "noise": noise_evidence,
        "checkpoint": {
            "jax_checkpoint_dir": str(jax_checkpoint_dir),
            "jax_params_format": "Orbax; no model.safetensors required",
            "jax_params_metadata_sha256": sha256_file(params_dir / "_METADATA")
            if (params_dir / "_METADATA").is_file()
            else None,
            "jax_norm_stats_path": str(jax_norm_stats),
            "jax_norm_stats_sha256": jax_norm_hash,
            "triton_base_dir": str(base_triton_dir),
            "triton_base_weights_path": str(triton_weights),
            "triton_base_weights_size_bytes": triton_weights.stat().st_size,
            "triton_base_weights_mtime_ns": triton_weights.stat().st_mtime_ns,
            "triton_weights_conversion_provenance_verified": False,
            "triton_norm_stats_path": str(triton_norm_stats),
            "triton_norm_stats_sha256": triton_norm_hash,
            "norm_stats_identical": True,
        },
        "prompt_manifest": str(prompt_manifest) if prompt_manifest else None,
        "prompt_manifest_prompt_present": manifest is not None,
        "dependencies_available_without_import": dependencies,
        "explicit_noise_required": True,
        "num_steps": NUM_STEPS,
        "comparison": "independent official JAX reference vs Triton full18 output",
        "not_claimed": "This comparison is not Triton-vs-Triton parity and does not itself identify a root cause.",
    }


def _to_numpy_actions(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"Expected raw actions shape {(ACTION_HORIZON, ACTION_DIM)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Raw action output contains NaN or infinity")
    return result


def _transform_policy_actions(
    *,
    raw_actions: np.ndarray,
    transformed: Mapping[str, Any],
    output_transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> np.ndarray:
    result = output_transform(
        {
            "state": np.asarray(transformed["state"]).copy(),
            "actions": np.asarray(raw_actions, dtype=np.float32).copy(),
        }
    )["actions"]
    result = np.asarray(result, dtype=np.float32)
    if result.shape != (ACTION_HORIZON, OUTPUT_ACTION_DIM):
        raise ValueError(
            f"Expected output-transformed actions shape {(ACTION_HORIZON, OUTPUT_ACTION_DIM)}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("Output-transformed actions contain NaN or infinity")
    return result


def _clone_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }


def _tree_fingerprints(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        result: dict[str, dict[str, Any]] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            result.update(_tree_fingerprints(value[key], child_prefix))
        return result
    if isinstance(value, tuple | list):
        result = {}
        for index, child in enumerate(value):
            child_prefix = f"{prefix}/{index}" if prefix else str(index)
            result.update(_tree_fingerprints(child, child_prefix))
        return result
    if value is None:
        return {prefix: {"is_none": True}}
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise ValueError(f"Cannot fingerprint object-valued transformed input at {prefix}")
    return {prefix: {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": sha256_array(array)}}


def _difference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    difference = candidate - reference
    return {
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "exact_equal": bool(np.array_equal(reference, candidate)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "l2_error": float(np.linalg.norm(difference.ravel())),
    }


def _load_model_components(jax_checkpoint_dir: Path, base_triton_dir: Path, prompt_manifest: Path, device: str):
    """Load the JAX Orbax model and Triton runtime; called only after preflight."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts/spec"))

    import jax
    import jax.numpy as jnp
    import spec_serve_policy
    import torch

    import openpi.models.model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as train_config_lib
    from scripts.spec.triton import triton_pi0_runtime as triton_runtime

    base_weights_path, artifact_dir = spec_serve_policy._resolve_triton_base_artifact(base_triton_dir)
    train_config, input_transform, output_transform = spec_serve_policy._build_policy_components(
        config_name="pi0_libero", checkpoint_dir=artifact_dir
    )
    official_config = train_config_lib.get_config("pi0_libero")
    if type(official_config.model).__name__ != "Pi0Config":
        raise TypeError(f"Expected official pi0_libero Pi0Config, got {type(official_config.model).__name__}")
    import openpi.models.model as model_module  # explicit mirror of policy_config.create_trained_policy

    model = official_config.model.load(model_module.restore_params(jax_checkpoint_dir / "params", dtype=jnp.bfloat16))
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
    return jax, jnp, torch, model_lib, sampler, input_transform, output_transform, runtime_pool, train_config, device


def _run_jax_reference(
    *, components: tuple[Any, ...], observation: dict[str, Any], effective_noise: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    jax, jnp, _, model_lib, sampler, input_transform, output_transform, _, train_config, _ = components
    transformed = input_transform(_clone_observation(observation))
    batched = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], transformed)
    observation_for_model = model_lib.Observation.from_dict(batched)
    # Match Policy.infer's default key behavior: initialize with key(0), then split once.
    _, sample_rng = jax.random.split(jax.random.key(0))
    raw = sampler(
        sample_rng,
        observation_for_model,
        num_steps=NUM_STEPS,
        noise=jnp.asarray(effective_noise, dtype=jnp.float32)[np.newaxis, ...],
    )
    raw_actions = _to_numpy_actions(raw)
    actions7 = _transform_policy_actions(
        raw_actions=raw_actions, transformed=transformed, output_transform=output_transform
    )
    return raw_actions, {
        "transformed": transformed,
        "actions7": actions7,
        "action_horizon": int(train_config.model.action_horizon),
    }


def _run_triton_full(
    *, components: tuple[Any, ...], observation: dict[str, Any], effective_noise: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    _, _, torch, _, _, input_transform, output_transform, runtime_pool, train_config, device = components
    transformed = input_transform(_clone_observation(observation))
    from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed

    images, state, noise = prepare_triton_inputs_from_transformed(
        transformed=transformed,
        device=device,
        action_horizon=int(train_config.model.action_horizon),
        action_dim=int(train_config.model.action_dim),
        noise=effective_noise,
    )
    session = runtime_pool.start_session(str(observation["prompt"]))
    prepared = session.prepare_observation(images=images, state=state)
    raw = session.run_full(prepared=prepared, noise=noise)
    raw_actions = _to_numpy_actions(raw)
    actions7 = _transform_policy_actions(
        raw_actions=raw_actions, transformed=transformed, output_transform=output_transform
    )
    return raw_actions, {"transformed": transformed, "actions7": actions7}


def run_comparison(
    *,
    observation: dict[str, Any],
    effective_noise: np.ndarray,
    jax_runner: Callable[[dict[str, Any], np.ndarray], tuple[np.ndarray, np.ndarray]],
    triton_runner: Callable[[dict[str, Any], np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Small injected-interface seam so action handling can be CPU-tested."""
    jax_raw, jax_actions7 = jax_runner(observation, effective_noise)
    triton_raw, triton_actions7 = triton_runner(observation, effective_noise)
    return {
        "jax_raw_actions": _to_numpy_actions(jax_raw),
        "triton_raw_actions": _to_numpy_actions(triton_raw),
        "jax_actions7": np.asarray(jax_actions7, dtype=np.float32),
        "triton_actions7": np.asarray(triton_actions7, dtype=np.float32),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_results(output_dir: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    npz_path = output_dir / "reference_policy_probe.npz"
    json_path = output_dir / "reference_policy_probe.json"
    np.savez_compressed(
        npz_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False)),
    )
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observation-npz", required=True, type=Path, help="Original observation NPZ used by collect_submodules.py"
    )
    parser.add_argument(
        "--noise-npz", required=True, type=Path, help="Matching *.submodule_probe.npz with explicit noise and metadata"
    )
    parser.add_argument("--prompt-manifest", type=Path, help="Triton prompt-cache manifest containing this prompt")
    parser.add_argument(
        "--jax-checkpoint-dir",
        type=Path,
        default=DEFAULT_JAX_CHECKPOINT,
        help="Official pi0_libero Orbax checkpoint directory",
    )
    parser.add_argument(
        "--base-triton-dir", type=Path, default=DEFAULT_TRITON_BASE, help="Converted Triton base artifact directory"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda:0", help="Torch/Triton device; JAX uses the first visible accelerator"
    )
    parser.add_argument(
        "--num-steps", type=int, default=NUM_STEPS, help="Only the official ten-step sample_actions path is supported"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate input/checkpoint provenance and dependency availability; do not load model or device",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.num_steps != NUM_STEPS:
        raise SystemExit(f"This reference probe is fixed to official num_steps={NUM_STEPS}")
    if args.prompt_manifest is None and not args.preflight_only:
        raise SystemExit("--prompt-manifest is required for a full Triton run")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")

    preflight = _check_inputs(
        observation_npz=args.observation_npz,
        noise_npz=args.noise_npz,
        prompt_manifest=args.prompt_manifest,
        jax_checkpoint_dir=args.jax_checkpoint_dir,
        base_triton_dir=args.base_triton_dir,
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    observation, observation_evidence = _load_observation(args.observation_npz)
    source_noise, effective_noise, noise_evidence = _load_reference_noise(
        args.noise_npz, observation_sha256=observation_evidence["sha256"]
    )
    components = _load_model_components(
        args.jax_checkpoint_dir.expanduser().resolve(),
        args.base_triton_dir.expanduser().resolve(),
        args.prompt_manifest.expanduser().resolve(),
        args.device,
    )
    jax_raw, jax_aux = _run_jax_reference(
        components=components, observation=observation, effective_noise=effective_noise
    )
    # Retain JAX evidence in host memory, then execute the independent Triton full path.
    triton_raw, triton_aux = _run_triton_full(
        components=components, observation=observation, effective_noise=effective_noise
    )
    jax_input_fingerprints = _tree_fingerprints(jax_aux["transformed"])
    triton_input_fingerprints = _tree_fingerprints(triton_aux["transformed"])
    if jax_input_fingerprints != triton_input_fingerprints:
        raise RuntimeError("JAX and Triton transformed inputs differ despite sharing the same input transform")
    arrays = {
        "source_noise": source_noise,
        "effective_noise_bfloat16_values": effective_noise,
        "jax_raw_actions_32d": jax_raw,
        "triton_raw_actions_32d": triton_raw,
        "jax_policy_actions_7d": jax_aux["actions7"],
        "triton_policy_actions_7d": triton_aux["actions7"],
    }
    metadata = {
        **preflight,
        "inspection_only": False,
        "model_loaded": True,
        "device_initialized": True,
        "configuration": {
            "num_steps": NUM_STEPS,
            "model_config": "official pi0_libero / Pi0Config loaded directly from Orbax params",
            "jax_params_source": str(args.jax_checkpoint_dir.resolve() / "params"),
            "triton_path": "independent Triton full18 action output with full18 VLM and full10 flow",
            "input_output_transforms": "spec_serve_policy._build_policy_components(pi0_libero, converted/base); base norm_stats verified byte-identical to raw checkpoint stats",
            "output_action_dim": OUTPUT_ACTION_DIM,
            "device": args.device,
            "jax_backend": str(components[0].default_backend()),
            "jax_version": _package_version("jax"),
            "torch_version": _package_version("torch"),
            "transformed_input_fingerprints": jax_input_fingerprints,
        },
        "noise": noise_evidence,
        "metrics": {
            "raw_32d": _difference_metrics(jax_raw, triton_raw),
            "output_transformed_7d": _difference_metrics(jax_aux["actions7"], triton_aux["actions7"]),
        },
        "interpretation": "JAX is the independent reference implementation. Differences are evidence for investigation, not a standalone causal conclusion.",
    }
    _write_results(args.output_dir, arrays, metadata)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "metrics": metadata["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
