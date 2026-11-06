#!/usr/bin/env python3
"""Isolate FP32-vs-BF16 attention-logit storage in the Triton action decoder."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from scripts.experiment8 import reference_policy_probe as probe
except ModuleNotFoundError:
    import reference_policy_probe as probe  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_reference(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path.expanduser().resolve(strict=True), allow_pickle=False) as archive:
        required = {
            "effective_noise_bfloat16_values",
            "jax_raw_actions_32d",
            "triton_raw_actions_32d",
            "jax_policy_actions_7d",
            "triton_policy_actions_7d",
            "metadata_json",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Reference NPZ missing arrays: {missing}")
        arrays = {key: np.array(archive[key], copy=True) for key in required - {"metadata_json"}}
        metadata = json.loads(str(archive["metadata_json"].item()))
    return arrays, metadata


def _run_triton(
    *,
    observation: dict[str, Any],
    effective_noise: np.ndarray,
    base_triton_dir: Path,
    prompt_manifest: Path,
    device: str,
    attention_logits_fp32: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts/spec"))
    import spec_serve_policy
    import torch

    from scripts.spec.triton import triton_pi0_runtime

    base_weights_path, artifact_dir = spec_serve_policy._resolve_triton_base_artifact(base_triton_dir)
    train_config, input_transform, output_transform = spec_serve_policy._build_policy_components(
        config_name="pi0_libero", checkpoint_dir=artifact_dir
    )
    pool = triton_pi0_runtime.TritonRuntimePool(
        base_weights_path=base_weights_path,
        manifest_path=prompt_manifest,
        num_views=2,
        chunk_size=int(train_config.model.action_horizon),
        vlm_layer_pattern="vlm_full18",
        ablation_layer_pattern="full18",
        denoise_steps=probe.NUM_STEPS,
        attention_logits_fp32=attention_logits_fp32,
    )
    transformed = input_transform(probe._clone_observation(observation))
    images, state, noise = triton_pi0_runtime.prepare_triton_inputs_from_transformed(
        transformed=transformed,
        device=device,
        action_horizon=int(train_config.model.action_horizon),
        action_dim=int(train_config.model.action_dim),
        noise=effective_noise,
    )
    session = pool.start_session(str(observation["prompt"]))
    runtime = pool.get_runtime(str(observation["prompt"]))
    actual_logits_dtype = str(runtime.buffers.get("decoder_attn_logits_buf", runtime.buffers["decoder_attn_buf"]).dtype)
    probabilities_dtype = str(runtime.buffers["decoder_attn_buf"].dtype)
    prepared = session.prepare_observation(images=images, state=state)
    raw = probe._to_numpy_actions(session.run_full(prepared=prepared, noise=noise))
    actions7 = probe._transform_policy_actions(
        raw_actions=raw, transformed=transformed, output_transform=output_transform
    )
    evidence = {
        "attention_logits_fp32": attention_logits_fp32,
        "attention_logits_buffer_dtype": actual_logits_dtype,
        "attention_probabilities_buffer_dtype": probabilities_dtype,
        "transformed_input_fingerprints": probe._tree_fingerprints(transformed),
    }
    del session, runtime, pool, prepared, images, state, noise
    gc.collect()
    torch.cuda.empty_cache()
    return raw, actions7, evidence


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    observation, observation_evidence = probe._load_observation(args.observation_npz)
    reference, reference_metadata = _read_reference(args.reference_npz)
    reference_noise = np.asarray(reference["effective_noise_bfloat16_values"], dtype=np.float32)

    baseline_raw, baseline_7d, baseline_evidence = _run_triton(
        observation=observation,
        effective_noise=reference_noise,
        base_triton_dir=args.base_triton_dir,
        prompt_manifest=args.prompt_manifest,
        device=args.device,
        attention_logits_fp32=False,
    )
    if not np.array_equal(baseline_raw, reference["triton_raw_actions_32d"]):
        raise RuntimeError("Modified default Triton path is not bitwise equal to the saved baseline reference")
    fp32_raw, fp32_7d, fp32_evidence = _run_triton(
        observation=observation,
        effective_noise=reference_noise,
        base_triton_dir=args.base_triton_dir,
        prompt_manifest=args.prompt_manifest,
        device=args.device,
        attention_logits_fp32=True,
    )
    metrics = {
        "default_triton_reproduces_saved_reference": True,
        "raw32d": {
            "default_triton_vs_jax": probe._difference_metrics(reference["jax_raw_actions_32d"], baseline_raw),
            "fp32_logits_triton_vs_jax": probe._difference_metrics(reference["jax_raw_actions_32d"], fp32_raw),
            "fp32_logits_effect_vs_default_triton": probe._difference_metrics(baseline_raw, fp32_raw),
        },
        "actions7d": {
            "default_triton_vs_jax": probe._difference_metrics(reference["jax_policy_actions_7d"], baseline_7d),
            "fp32_logits_triton_vs_jax": probe._difference_metrics(reference["jax_policy_actions_7d"], fp32_7d),
            "fp32_logits_effect_vs_default_triton": probe._difference_metrics(baseline_7d, fp32_7d),
        },
    }
    payload = {
        "schema_version": 1,
        "experiment": "Triton decoder attention logits FP32 storage before softmax",
        "observation": observation_evidence,
        "reference_npz": str(args.reference_npz.expanduser().resolve()),
        "reference_metadata": reference_metadata,
        "same_effective_noise": True,
        "baseline": baseline_evidence,
        "intervention": fp32_evidence,
        "controlled_path": "Only QK logits storage/masked-softmax input is FP32; softmax probabilities and V path remain BF16.",
        "metrics": metrics,
        "not_claimed": "One fixed observation cannot establish closed-loop or population-level causality.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "attention_logits_precision.npz",
        effective_noise_bfloat16_values=reference_noise,
        jax_raw_actions_32d=reference["jax_raw_actions_32d"],
        default_triton_raw_actions_32d=baseline_raw,
        fp32_logits_triton_raw_actions_32d=fp32_raw,
        jax_policy_actions_7d=reference["jax_policy_actions_7d"],
        default_triton_policy_actions_7d=baseline_7d,
        fp32_logits_triton_policy_actions_7d=fp32_7d,
        metadata_json=np.asarray(json.dumps(payload, sort_keys=True, ensure_ascii=False)),
    )
    (output_dir / "attention_logits_precision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--base-triton-dir", type=Path, default=probe.DEFAULT_TRITON_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
