"""CPU-only interface tests for the official JAX reference websocket policy."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.experiment8 import reference_policy_probe as reference_probe
from scripts.experiment8 import serve_jax_reference_policy as jax_server
from scripts.experiment8 import serve_submodule_policy as paired_noise

# These tests intentionally exercise the pure helper seams shared across the
# paired rollout servers.
# ruff: noqa: SLF001


def _request(infer_id: int = 0) -> dict[str, Any]:
    return {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 127, dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "put the block in the drawer",
        "__semantic_task_key__": "libero_10/task_03",
        "__init_state_sha256__": "a" * 64,
        "__noise_seed__": 12345,
        "__infer_id__": infer_id,
    }


def _provenance_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    params = tmp_path / "params"
    (params / "ocdbt.process_0").mkdir(parents=True)
    (params / "_METADATA").write_bytes(b"orbax metadata")
    (params / "manifest.ocdbt").write_bytes(b"root manifest")
    (params / "ocdbt.process_0/manifest.ocdbt").write_bytes(b"process manifest")
    (params / "ocdbt.process_0/d/chunk-a").parent.mkdir(parents=True)
    (params / "ocdbt.process_0/d/chunk-a").write_bytes(b"small parameter chunk")
    raw_norm = tmp_path / "raw_norm_stats.json"
    transform_norm = tmp_path / "transform_norm_stats.json"
    norm_bytes = b'{"fixture":true}\n'
    raw_norm.write_bytes(norm_bytes)
    transform_norm.write_bytes(norm_bytes)
    source = tmp_path / "serve_jax_reference_policy.py"
    source.write_bytes(b"fixture script")
    return params, raw_norm, transform_norm, source


def test_policy_uses_same_request_noise_and_effective_bf16_values_as_triton() -> None:
    request = _request(infer_id=6)
    transformed_inputs: list[dict[str, Any]] = []
    action_runner_calls: list[tuple[dict[str, Any], np.ndarray]] = []

    def input_transform(inputs: dict[str, Any]) -> dict[str, Any]:
        transformed_inputs.append(inputs)
        return {
            "state": np.asarray(inputs["observation/state"], dtype=np.float32) / 2,
            "seen_prompt": inputs["prompt"],
        }

    def output_transform(values: dict[str, Any]) -> dict[str, Any]:
        return {"actions": np.asarray(values["actions"], dtype=np.float32)[:, :7] + 0.125}

    def action_runner(transformed: dict[str, Any], effective_noise: np.ndarray) -> np.ndarray:
        action_runner_calls.append((transformed, effective_noise.copy()))
        return effective_noise[None, ...]

    provenance = {"provenance_sha256": "0" * 64}
    provenance["provenance_sha256"] = paired_noise._provenance_digest(provenance)
    policy = jax_server.OfficialJaxReferencePolicy(
        input_transform=input_transform,
        output_transform=output_transform,
        action_runner=action_runner,
        run_provenance=provenance,
    )

    response = policy.infer(request)
    source_noise, source_kind, identity = paired_noise._extract_noise(request, action_horizon=50, action_dim=32)
    effective_noise = reference_probe._float32_to_bfloat16_roundtrip(source_noise)
    metadata = response["diagnostic_metadata"]

    assert source_kind == "request_key_deterministic"
    assert metadata["noise_source"] == source_kind
    assert metadata["noise_identity"] == identity
    assert metadata["noise_sha256"] == hashlib.sha256(source_noise.tobytes()).hexdigest()
    assert metadata["effective_noise_bfloat16_sha256"] == reference_probe.sha256_array(effective_noise)
    np.testing.assert_array_equal(action_runner_calls[0][1], effective_noise)
    assert response["actions"].shape == (50, 7)
    np.testing.assert_allclose(response["actions"], effective_noise[:, :7] + 0.125)
    assert metadata["runtime"] == "jax_reference"
    assert metadata["runtime_implementation"] == "official_jax_reference"
    assert metadata["eager_diagnostic"] is False
    assert metadata["skip_attn_layers"] == metadata["skip_ffn_layers"] == []
    assert metadata["denoise_steps"] == 10
    assert metadata["action_space"] == paired_noise.ACTION_SPACE
    assert policy.metadata["runtime"] == "jax_reference"


def test_policy_accepts_explicit_noise_but_preserves_the_same_hash_protocol() -> None:
    request = _request(infer_id=4)
    explicit_noise = np.full((50, 32), 0.125, dtype=np.float32)
    request[paired_noise.EXPLICIT_NOISE_KEY] = explicit_noise
    observed_noise: list[np.ndarray] = []

    policy = jax_server.OfficialJaxReferencePolicy(
        input_transform=lambda inputs: {"state": inputs["observation/state"]},
        output_transform=lambda values: {"actions": np.asarray(values["actions"])[:, :7]},
        action_runner=lambda _transformed, noise: observed_noise.append(noise.copy()) or noise[None, ...],
        run_provenance={
            "provenance_sha256": paired_noise._provenance_digest({"marker": "fake"}),
            "marker": "fake",
        },
    )

    response = policy.infer(request)

    effective = reference_probe._float32_to_bfloat16_roundtrip(explicit_noise)
    np.testing.assert_array_equal(observed_noise[0], effective)
    assert response["diagnostic_metadata"]["noise_source"] == "explicit_request_noise"
    assert response["diagnostic_metadata"]["noise_sha256"] == hashlib.sha256(explicit_noise.tobytes()).hexdigest()


class _FakeRandom:
    @staticmethod
    def key(seed: int) -> tuple[str, int]:
        return "key", seed

    @staticmethod
    def split(key: tuple[str, int]) -> tuple[tuple[str, int], tuple[str, int]]:
        return ("next", key[1]), ("sample", key[1])


class _FakeJax:
    random = _FakeRandom()
    tree = None

    @staticmethod
    def default_device(_device: str):
        return nullcontext()

    @staticmethod
    def device_put(value: Any, _device: str) -> Any:
        return value


class _FakeTree:
    @staticmethod
    def map(function, tree):
        if isinstance(tree, dict):
            return {key: _FakeTree.map(function, value) for key, value in tree.items()}
        return function(tree)


class _FakeJnp:
    float32 = np.float32

    @staticmethod
    def asarray(value: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(value, dtype=dtype)


class _FakeObservation:
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> dict[str, Any]:
        return value


def test_action_runner_invokes_explicit_noise_official_step_contract_without_jax() -> None:
    jax = _FakeJax()
    jax.tree = _FakeTree()
    calls: list[dict[str, Any]] = []

    def sampler(rng: Any, observation: Any, *, num_steps: int, noise: np.ndarray) -> np.ndarray:
        calls.append({"rng": rng, "observation": observation, "num_steps": num_steps, "noise": noise.copy()})
        return np.zeros((1, 50, 32), dtype=np.float32)

    runner = jax_server._make_jax_action_runner(
        jax=jax,
        jnp=_FakeJnp,
        model_lib=type("FakeModel", (), {"Observation": _FakeObservation}),
        sampler=sampler,
        device="cpu:0",
    )
    noise = np.full((50, 32), 0.25, dtype=np.float32)

    result = runner({"state": np.ones((8,), dtype=np.float32)}, noise)

    assert calls[0]["rng"] == ("sample", 0)
    assert calls[0]["num_steps"] == 10
    assert calls[0]["noise"].shape == (1, 50, 32)
    np.testing.assert_array_equal(calls[0]["noise"][0], noise)
    assert calls[0]["observation"]["state"].shape == (1, 8)
    assert result.shape == (1, 50, 32)


def test_orbax_tree_provenance_hashes_all_files_and_publishes_no_overwrite_record(tmp_path: Path) -> None:
    params, raw_norm, transform_norm, source = _provenance_fixture(tmp_path)
    provenance = jax_server._build_run_provenance(
        params_dir=params,
        checkpoint_norm_stats_path=raw_norm,
        transform_norm_stats_path=transform_norm,
        source_script_path=source,
        runtime_config={"runtime": "jax_reference", "denoise_steps": 10},
    )
    metadata_path = tmp_path / "startup" / "run.json"

    published = jax_server._publish_run_provenance(metadata_path, provenance)

    data = json.loads(published.read_text(encoding="utf-8"))
    assert data == provenance
    assert data["runtime"] == "jax_reference"
    assert data["params_file_count"] == 4
    assert data["params_metadata_sha256"] == hashlib.sha256(b"orbax metadata").hexdigest()
    assert data["raw_checkpoint_norm_stats_sha256"] == data["norm_stats_sha256"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        jax_server._publish_run_provenance(metadata_path, provenance)


def test_norm_stats_mismatch_is_rejected_before_policy_startup(tmp_path: Path) -> None:
    params, raw_norm, transform_norm, source = _provenance_fixture(tmp_path)
    transform_norm.write_text('{"different":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="norm_stats differ"):
        jax_server._build_run_provenance(
            params_dir=params,
            checkpoint_norm_stats_path=raw_norm,
            transform_norm_stats_path=transform_norm,
            source_script_path=source,
            runtime_config={"runtime": "jax_reference"},
        )


def test_cli_exposes_no_ablation_masks_and_requires_startup_record() -> None:
    parser = jax_server.build_argument_parser()

    args = parser.parse_args(["--run-metadata-path", "results/jax_reference/startup.json"])

    assert args.jax_checkpoint_dir == jax_server.DEFAULT_JAX_CHECKPOINT_DIR
    assert args.jax_device == "gpu:0"
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-metadata-path", "run.json", "--skip-attn-layers", "2"])
