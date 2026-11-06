"""CPU fake-pool tests for the eager Attention/FFN websocket policy."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest

from scripts.experiment8.serve_submodule_policy import (
    ACTION_SPACE,
    SubmoduleDiagnosticPolicy,
    _build_run_provenance,
    _publish_run_metadata,
)


class FakeSession:
    def __init__(self, owner: "FakePool", prompt: str) -> None:
        self.owner = owner
        self.prompt = prompt

    def prepare_observation(self, *, images: Any, state: Any) -> dict[str, Any]:
        prepared = {"images": images, "state": state}
        self.owner.prepared.append(prepared)
        return prepared

    def run_submodule_probe(
        self, *, prepared: Any, noise: np.ndarray,
        skip_attn_layers: tuple[int, ...], skip_ffn_layers: tuple[int, ...],
    ) -> np.ndarray:
        self.owner.calls.append(
            {
                "prompt": self.prompt,
                "prepared": prepared,
                "noise": np.array(noise, copy=True),
                "skip_attn_layers": skip_attn_layers,
                "skip_ffn_layers": skip_ffn_layers,
            }
        )
        return np.broadcast_to(noise[None, ...], (1, *noise.shape)).copy()


class FakePool:
    def __init__(self) -> None:
        self.prepared: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self.prompts: list[str] = []

    def start_session(self, prompt: str) -> FakeSession:
        self.prompts.append(prompt)
        return FakeSession(self, prompt)


def _request(infer_id: int) -> dict[str, Any]:
    return {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        "observation/state": np.arange(9, dtype=np.float32),
        "prompt": "pick up the block",
        "__semantic_task_key__": "libero_10/task_04",
        "__init_state_sha256__": "a" * 64,
        "__noise_seed__": 13,
        "__infer_id__": infer_id,
    }


def _components():
    transform_calls: list[dict[str, Any]] = []
    output_calls: list[dict[str, Any]] = []
    prepare_calls: list[dict[str, Any]] = []

    def input_transform(inputs: dict[str, Any]) -> dict[str, Any]:
        transform_calls.append(inputs)
        return {"state": np.asarray(inputs["observation/state"], dtype=np.float32) + 2.0, "transformed": True}

    def output_transform(output: dict[str, Any]) -> dict[str, Any]:
        output_calls.append(output)
        return {"actions": np.asarray(output["actions"], dtype=np.float32)[:, :7] + 0.25}

    def input_preparer(**kwargs: Any):
        prepare_calls.append(kwargs)
        return "prepared-images", np.asarray(kwargs["transformed"]["state"]), kwargs["noise"].copy()

    return input_transform, output_transform, input_preparer, transform_calls, output_calls, prepare_calls


def test_noise_is_request_order_independent_and_masks_map_from_1_to_0_based() -> None:
    input_transform, output_transform, input_preparer, transform_calls, output_calls, prepare_calls = _components()
    baseline_pool = FakePool()
    variant_pool = FakePool()
    baseline = SubmoduleDiagnosticPolicy(
        input_transform=input_transform,
        output_transform=output_transform,
        runtime_pool=baseline_pool,
        input_preparer=input_preparer,
        device="cpu",
    )
    variant = SubmoduleDiagnosticPolicy(
        input_transform=input_transform,
        output_transform=output_transform,
        runtime_pool=variant_pool,
        input_preparer=input_preparer,
        device="cpu",
        skip_attn_layers=(1, 17),
        skip_ffn_layers=(2,),
    )

    # Reverse the variant execution order to prove the noise depends on request
    # identity rather than a server-local call counter.
    variant_response = variant.infer(_request(8))
    baseline_response = baseline.infer(_request(8))

    baseline_call = baseline_pool.calls[0]
    variant_call = variant_pool.calls[0]
    np.testing.assert_array_equal(baseline_call["noise"], variant_call["noise"])
    assert baseline_call["skip_attn_layers"] == ()
    assert baseline_call["skip_ffn_layers"] == ()
    assert variant_call["skip_attn_layers"] == (0, 16)
    assert variant_call["skip_ffn_layers"] == (1,)
    for response in (baseline_response, variant_response):
        assert response["actions"].shape == (50, 7)
        assert np.isfinite(response["actions"]).all()
        assert response["diagnostic_metadata"]["eager_diagnostic"] is True
        assert response["diagnostic_metadata"]["denoise_steps"] == 10
        assert response["diagnostic_metadata"]["action_space"] == ACTION_SPACE
        assert response["diagnostic_metadata"]["noise_source"] == "request_key_deterministic"
        assert response["diagnostic_metadata"]["position_semantics"] == "not_measured_eef_xyz"
    assert baseline_response["diagnostic_metadata"]["noise_sha256"] == variant_response["diagnostic_metadata"]["noise_sha256"]
    assert variant_response["diagnostic_metadata"]["skip_attn_layers"] == [1, 17]
    assert variant_response["diagnostic_metadata"]["skip_ffn_layers"] == [2]
    assert variant_response["diagnostic_metadata"]["runtime_skip_attn_layers_zero_based"] == [0, 16]
    assert variant_response["diagnostic_metadata"]["runtime_skip_ffn_layers_zero_based"] == [1]
    assert baseline_pool.prompts == ["pick up the block"]
    assert len(transform_calls) == len(output_calls) == len(prepare_calls) == 2
    assert len(baseline_pool.prepared) == len(variant_pool.prepared) == 1
    np.testing.assert_array_equal(output_calls[0]["actions"], baseline_call["noise"])
    # Every request ran the eager intervention API, including the no-mask baseline.
    assert baseline_call["prepared"] is baseline_pool.prepared[0]


def test_startup_provenance_is_attached_to_response_and_server_metadata(tmp_path) -> None:
    inputs = {
        "checkpoint": b"fake checkpoint bytes",
        "norm_stats": b'{"mean": [0.0]}',
        "prompt_manifest": b'{"prompts": {}}',
        "source_script": b"diagnostic source",
    }
    paths = {}
    for key, content in inputs.items():
        paths[key] = tmp_path / key
        paths[key].write_bytes(content)
    provenance = _build_run_provenance(
        checkpoint_path=paths["checkpoint"],
        norm_stats_path=paths["norm_stats"],
        prompt_manifest_path=paths["prompt_manifest"],
        source_script_path=paths["source_script"],
        runtime_config={
            "skip_attn_layers_1based": [3],
            "skip_ffn_layers_1based": [4, 8],
        },
    )
    input_transform, output_transform, input_preparer, *_ = _components()
    policy = SubmoduleDiagnosticPolicy(
        input_transform=input_transform,
        output_transform=output_transform,
        runtime_pool=FakePool(),
        input_preparer=input_preparer,
        device="cpu",
        skip_attn_layers=(3,),
        skip_ffn_layers=(4, 8),
        run_provenance=provenance,
    )

    response = policy.infer(_request(0))

    assert policy.metadata["run_provenance"] == provenance
    assert response["diagnostic_metadata"]["run_provenance"] == provenance
    assert response["diagnostic_metadata"]["provenance_sha256"] == provenance["provenance_sha256"]
    assert provenance["runtime_config"]["skip_attn_layers_1based"] == [3]
    assert provenance["runtime_config"]["skip_ffn_layers_1based"] == [4, 8]


def test_provenance_file_hashes_correctly_and_refuses_overwrite(tmp_path) -> None:
    inputs = {
        "checkpoint": b"small test checkpoint payload",
        "norm_stats": b"norm stats payload",
        "prompt_manifest": b"prompt manifest payload",
        "source_script": b"source script payload",
    }
    paths = {}
    for key, content in inputs.items():
        paths[key] = tmp_path / f"{key}.bin"
        paths[key].write_bytes(content)
    provenance = _build_run_provenance(
        checkpoint_path=paths["checkpoint"],
        norm_stats_path=paths["norm_stats"],
        prompt_manifest_path=paths["prompt_manifest"],
        source_script_path=paths["source_script"],
        runtime_config={"skip_attn_layers_1based": [], "skip_ffn_layers_1based": [2]},
    )
    for key in inputs:
        assert provenance[f"{key}_sha256"] == hashlib.sha256(inputs[key]).hexdigest()
        assert provenance[f"{key}_path"] == str(paths[key].resolve())

    metadata_path = tmp_path / "nested" / "run.json"
    published_path = _publish_run_metadata(metadata_path, provenance)
    published = json.loads(published_path.read_text(encoding="utf-8"))
    assert published == provenance
    assert published["provenance_sha256"]
    assert published["runtime_config"]["skip_ffn_layers_1based"] == [2]

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _publish_run_metadata(metadata_path, provenance)
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == provenance
    assert list(metadata_path.parent.iterdir()) == [metadata_path]


def test_explicit_noise_is_used_and_hashed() -> None:
    input_transform, output_transform, input_preparer, *_ = _components()
    pool = FakePool()
    policy = SubmoduleDiagnosticPolicy(
        input_transform=input_transform,
        output_transform=output_transform,
        runtime_pool=pool,
        input_preparer=input_preparer,
        device="cpu",
    )
    request = _request(4)
    explicit = np.full((50, 32), 0.125, dtype=np.float32)
    request["__noise__"] = explicit

    response = policy.infer(request)

    np.testing.assert_array_equal(pool.calls[0]["noise"], explicit)
    assert response["diagnostic_metadata"]["noise_source"] == "explicit_request_noise"
    assert response["diagnostic_metadata"]["noise_sha256"]
    np.testing.assert_allclose(response["actions"], 0.375)


@pytest.mark.parametrize("layers", [(0,), (18,), (1, 1)])
def test_out_of_range_or_duplicate_layer_indices_are_rejected(layers: tuple[int, ...]) -> None:
    input_transform, output_transform, input_preparer, *_ = _components()
    with pytest.raises(ValueError, match="1..17|duplicate"):
        SubmoduleDiagnosticPolicy(
            input_transform=input_transform,
            output_transform=output_transform,
            runtime_pool=FakePool(),
            input_preparer=input_preparer,
            skip_attn_layers=layers,
        )


def test_seeded_noise_requires_all_pairing_keys() -> None:
    input_transform, output_transform, input_preparer, *_ = _components()
    policy = SubmoduleDiagnosticPolicy(
        input_transform=input_transform,
        output_transform=output_transform,
        runtime_pool=FakePool(),
        input_preparer=input_preparer,
    )
    request = _request(3)
    del request["__infer_id__"]
    with pytest.raises(ValueError, match="__infer_id__"):
        policy.infer(request)
