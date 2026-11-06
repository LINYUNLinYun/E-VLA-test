"""CPU-only tests for the independent JAX-reference/Triton probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiment8 import reference_policy_probe as probe

# These tests exercise pure helper functions that are intentionally internal to
# this one-shot CLI.
# ruff: noqa: SLF001


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    observation_path = tmp_path / "observation.npz"
    prompt = "put the black bowl on the plate"
    np.savez_compressed(
        observation_path,
        agent_image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        robot_state=np.zeros((8,), dtype=np.float32),
        prompt=np.asarray(prompt),
        metadata_json=np.asarray(json.dumps({"sample_id": "fixture"})),
    )
    noise = (
        np.arange(probe.ACTION_HORIZON * probe.ACTION_DIM, dtype=np.float32).reshape(
            probe.ACTION_HORIZON, probe.ACTION_DIM
        )
        / 1000
    )
    noise_path = tmp_path / "noise.npz"
    np.savez_compressed(
        noise_path,
        noise=noise,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "observation_sha256": probe.sha256_file(observation_path),
                    "noise_sha256": probe.sha256_array(noise),
                    "denoise_steps": 10,
                }
            )
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    (tmp_path / "fixture.pt").write_bytes(b"fixture embedding")
    manifest_path.write_text(json.dumps({"prompts": {prompt: {"embed_path": "fixture.pt"}}}), encoding="utf-8")
    jax_checkpoint = tmp_path / "jax_checkpoint"
    (jax_checkpoint / "params").mkdir(parents=True)
    (jax_checkpoint / "params" / "_METADATA").write_text("orbax metadata", encoding="utf-8")
    jax_stats = jax_checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    jax_stats.parent.mkdir(parents=True)
    jax_stats.write_text('{"fixture":true}\n', encoding="utf-8")
    triton_base = tmp_path / "triton_base"
    (triton_base / "assets/physical-intelligence/libero").mkdir(parents=True)
    (triton_base / "base_weights.pkl").write_bytes(b"fixture weights")
    (triton_base / "assets/physical-intelligence/libero/norm_stats.json").write_bytes(jax_stats.read_bytes())
    return observation_path, noise_path, manifest_path, jax_checkpoint, triton_base


def test_explicit_noise_preflight_verifies_observation_and_normstats(tmp_path: Path) -> None:
    observation, noise, manifest, jax_checkpoint, triton_base = _fixture(tmp_path)

    evidence = probe._check_inputs(
        observation_npz=observation,
        noise_npz=noise,
        prompt_manifest=manifest,
        jax_checkpoint_dir=jax_checkpoint,
        base_triton_dir=triton_base,
    )

    assert evidence["model_loaded"] is False
    assert evidence["device_initialized"] is False
    assert evidence["checkpoint"]["norm_stats_identical"] is True
    assert evidence["noise"]["probe_metadata"]["denoise_steps"] == 10
    assert set(evidence["dependencies_available_without_import"]) == {"jax", "flax", "torch", "openpi_client"}


def test_noise_from_a_different_observation_is_rejected(tmp_path: Path) -> None:
    observation, noise, _, jax_checkpoint, triton_base = _fixture(tmp_path)
    changed_observation = tmp_path / "changed-observation.npz"
    changed_observation.write_bytes(observation.read_bytes() + b"different")

    with pytest.raises(ValueError, match="different observation"):
        probe._check_inputs(
            observation_npz=changed_observation,
            noise_npz=noise,
            prompt_manifest=None,
            jax_checkpoint_dir=jax_checkpoint,
            base_triton_dir=triton_base,
        )


def test_bfloat16_roundtrip_is_round_to_nearest_even() -> None:
    # Midpoint between 1.0 and its next BF16 value rounds to the even endpoint.
    even_tie = np.asarray([0x3F808000], dtype=np.uint32).view(np.float32)
    # Midpoint with an odd lower BF16 significand rounds upward to an even one.
    odd_tie = np.asarray([0x3F818000], dtype=np.uint32).view(np.float32)

    rounded = probe._float32_to_bfloat16_roundtrip(np.concatenate([even_tie, odd_tie]))

    assert rounded.view(np.uint32).tolist() == [0x3F800000, 0x3F820000]


def test_fake_runners_receive_same_explicit_noise_and_keep_outputs_distinct() -> None:
    observation = {"prompt": "fixture"}
    effective_noise = np.ones((probe.ACTION_HORIZON, probe.ACTION_DIM), dtype=np.float32)
    raw_jax = np.zeros_like(effective_noise)
    raw_triton = np.ones_like(effective_noise)
    actions7_jax = np.zeros((probe.ACTION_HORIZON, probe.OUTPUT_ACTION_DIM), dtype=np.float32)
    actions7_triton = np.ones_like(actions7_jax)
    calls: list[tuple[str, np.ndarray]] = []

    def jax_runner(obs: dict, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert obs is observation
        calls.append(("jax", noise))
        return raw_jax, actions7_jax

    def triton_runner(obs: dict, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert obs is observation
        calls.append(("triton", noise))
        return raw_triton, actions7_triton

    results = probe.run_comparison(
        observation=observation,
        effective_noise=effective_noise,
        jax_runner=jax_runner,
        triton_runner=triton_runner,
    )

    assert [name for name, _ in calls] == ["jax", "triton"]
    assert calls[0][1] is calls[1][1] is effective_noise
    assert np.array_equal(results["jax_raw_actions"], raw_jax)
    assert np.array_equal(results["triton_raw_actions"], raw_triton)
    assert np.array_equal(results["jax_actions7"], actions7_jax)
    assert np.array_equal(results["triton_actions7"], actions7_triton)
    assert probe._difference_metrics(results["jax_raw_actions"], results["triton_raw_actions"])["max_abs_error"] == 1.0


def test_policy_output_transform_receives_copy_and_must_return_7d_actions() -> None:
    state = np.asarray([1.0, 2.0], dtype=np.float32)
    transformed = {"state": state}
    raw = np.zeros((probe.ACTION_HORIZON, probe.ACTION_DIM), dtype=np.float32)
    seen: list[dict] = []

    def output_transform(values: dict) -> dict:
        seen.append(values)
        return {"actions": np.zeros((probe.ACTION_HORIZON, probe.OUTPUT_ACTION_DIM), dtype=np.float32)}

    actual = probe._transform_policy_actions(
        raw_actions=raw, transformed=transformed, output_transform=output_transform
    )

    assert actual.shape == (probe.ACTION_HORIZON, probe.OUTPUT_ACTION_DIM)
    assert seen[0]["state"] is not state
    assert seen[0]["actions"] is not raw

    with pytest.raises(ValueError, match="Expected output-transformed actions shape"):
        probe._transform_policy_actions(
            raw_actions=raw,
            transformed=transformed,
            output_transform=lambda _values: {"actions": np.zeros((probe.ACTION_HORIZON, 6))},
        )


def test_observation_copy_and_transformed_fingerprints_detect_mutation() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    original = {"image": image, "nested": {"state": np.asarray([1.0, 2.0], dtype=np.float32)}}
    copy = probe._clone_observation(original)

    copy["image"][0, 0, 0] = 255

    assert original["image"][0, 0, 0] == 0
    assert probe._tree_fingerprints(original) != probe._tree_fingerprints(copy)


def test_noise_hash_is_based_on_exact_array_bytes(tmp_path: Path) -> None:
    observation, noise_path, _, _, _ = _fixture(tmp_path)
    obs_evidence = probe._load_observation(observation)[1]
    source, effective, evidence = probe._load_reference_noise(noise_path, observation_sha256=obs_evidence["sha256"])

    assert evidence["source_noise_sha256"] == hashlib.sha256(source.tobytes()).hexdigest()
    assert evidence["effective_bfloat16_noise_sha256"] == probe.sha256_array(effective)
