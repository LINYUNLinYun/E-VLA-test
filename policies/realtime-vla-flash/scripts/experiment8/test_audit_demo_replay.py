"""CPU-only tests for the Experiment 5 HDF5 demonstration audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from scripts.experiment8 import audit_demo_replay as audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, include_init_attr: bool = True) -> tuple[Path, Path, Path, dict[str, object]]:
    manifest_dir = tmp_path / "manifest"
    dataset_root = tmp_path / "datasets"
    manifest_dir.mkdir()
    hdf5_path = dataset_root / "libero_90" / "TASK_ALPHA_demo.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    init_state = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    states = np.stack((init_state, init_state + 0.5, init_state + 1.0))
    actions = np.zeros((3, 7), dtype=np.float64)
    with h5py.File(hdf5_path, "w") as dataset:
        data = dataset.create_group("data")
        data.attrs["env_args"] = json.dumps({"problem_name": "test_task", "env_kwargs": {"ignore_done": True}})
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("states", data=states)
        demo.attrs["model_file"] = "<mujoco model='test'/>"
        if include_init_attr:
            demo.attrs["init_state"] = init_state
    state_path = manifest_dir / "states" / "libero_90_task000.npy"
    state_path.parent.mkdir()
    np.save(state_path, init_state, allow_pickle=False)
    sample: dict[str, object] = {
        "sample_id": "libero_90_task000",
        "suite_name": "libero_90",
        "task_id": 0,
        "task_name": "TASK_ALPHA",
        "demo_id": "demo_0",
        "source_hdf5_path": "libero_90/TASK_ALPHA_demo.hdf5",
        "init_state_path": "states/libero_90_task000.npy",
        "init_state_sha256": _sha256(state_path),
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")
    return manifest_path, dataset_root, state_path, sample


def test_array_comparison_reports_exact_and_mismatched_shapes() -> None:
    values = np.asarray([1.0, 2.0])
    exact = audit._array_comparison(values, values.copy())
    changed = audit._array_comparison(values, np.asarray([1.0, 2.5]))
    different_shape = audit._array_comparison(values, np.asarray([[1.0, 2.0]]))

    assert exact["exact_equal"] is True
    assert exact["l2_error"] == 0.0
    assert changed["exact_equal"] is False
    assert changed["l2_error"] == 0.5
    assert different_shape["shape_match"] is False
    assert different_shape["l2_error"] is None


def test_state_component_metrics_isolate_time_from_physical_state_fields() -> None:
    expected = np.asarray([1.0, 0.1, 0.2, 0.3])
    actual = np.asarray([0.0, 0.1, 0.2, 0.3])
    metrics = audit._state_component_errors(expected, actual, {"nq": 2, "nv": 1, "na": 0})

    assert metrics["layout_valid"] is True
    assert metrics["time_l2_error"] == 1.0
    assert metrics["qpos_l2_error"] == 0.0
    assert metrics["qvel_l2_error"] == 0.0


def test_asset_mapping_uses_only_explicit_assets_suffix_and_existing_targets(tmp_path: Path) -> None:
    assets_root = tmp_path / "configured_assets"
    mesh_target = assets_root / "scenes/kitchen_background/visual/kitchen_background_vis.msh"
    texture_target = assets_root / "textures/tile.png"
    mesh_target.parent.mkdir(parents=True)
    texture_target.parent.mkdir(parents=True)
    mesh_target.write_bytes(b"mesh-placeholder")
    texture_target.write_bytes(b"texture-placeholder")
    xml = """<mujoco><asset>
      <mesh name="background" file="/Users/bo/Desktop/libero-dev/chiliocosm/assets/scenes/kitchen_background/visual/kitchen_background_vis.msh" />
      <texture name="tile" file="/old/chiliocosm/assets/scenes/../textures/tile.png" />
      <mesh name="unrelated" file="/tmp/custom/meshes/other.msh" />
      <mesh name="robosuite" file="/env/site-packages/robosuite/models/assets/robots/panda/link.stl" />
      <mesh name="missing" file="/old/chiliocosm/assets/not-present.msh" />
    </asset><include file="/tmp/other/assets/include.xml" /></mujoco>"""

    remapped_xml, mappings = audit._remap_explicit_asset_paths(xml, assets_root)
    root = ET.fromstring(remapped_xml)
    meshes = {element.get("name"): element.get("file") for element in root.findall("./asset/mesh")}
    texture = root.find("./asset/texture")

    assert len(mappings) == 3
    assert meshes["background"] == str(mesh_target.resolve())
    assert texture.get("file") == str(texture_target.resolve())
    assert meshes["unrelated"] == "/tmp/custom/meshes/other.msh"
    assert meshes["robosuite"] == "/env/site-packages/robosuite/models/assets/robots/panda/link.stl"
    assert meshes["missing"] == "/old/chiliocosm/assets/not-present.msh"
    assert mappings[0]["target_exists"] is True
    assert mappings[1]["assets_relative_suffix"] == "scenes/../textures/tile.png"
    assert mappings[2]["target_exists"] is False
    assert root.find("include").get("file") == "/tmp/other/assets/include.xml"


def test_asset_mapping_does_not_allow_suffix_escape_from_assets_root(tmp_path: Path) -> None:
    xml = '<mujoco><asset><mesh name="escape" file="/old/assets/../../outside.msh" /></asset></mujoco>'
    _, mappings = audit._remap_explicit_asset_paths(xml, tmp_path / "assets")

    assert mappings[0]["target_within_assets_root"] is False
    assert mappings[0]["target_exists"] is False
    assert mappings[0]["mapped_path"] is None


def test_inspect_only_writes_state_hashes_and_raw_arrays_without_sim_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, dataset_root, _, _ = _fixture(tmp_path)
    output_dir = tmp_path / "inspect"

    def simulator_import_forbidden() -> object:
        pytest.fail("inspect-only must not attempt to import simulator packages")

    monkeypatch.setattr(audit, "_load_replay_components", simulator_import_forbidden)
    report = audit.run_audit(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        sample_ids=["libero_90_task000"],
        output_dir=output_dir,
        inspect_only=True,
        warmup_steps=0,
    )

    record = report["records"][0]
    assert report["execution_mode"] == "inspect_only"
    assert report["simulator_replayed_count"] == 0
    assert record["replay_status"] == "not_run"
    assert record["exported_hash_matches_manifest"] is True
    assert record["comparisons"]["hdf5_states_0_vs_demo_init_state_attr"]["exact_equal"] is True
    assert record["comparisons"]["exported_npy_vs_demo_init_state_attr"]["exact_equal"] is True
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "audit_report.json").is_file()
    with np.load(record["raw_evidence_npz"], allow_pickle=False) as evidence:
        np.testing.assert_array_equal(evidence["hdf5_states_0"], evidence["demo_init_state_attr"])
        assert evidence["actions"].shape == (3, 7)
        assert "replay_post_action_states" not in evidence.files


def test_inspection_handles_missing_init_state_attribute_and_detects_export_mismatch(tmp_path: Path) -> None:
    manifest_path, dataset_root, state_path, _ = _fixture(tmp_path, include_init_attr=False)
    altered = np.asarray([1.0, 2.0, 3.0, 4.5], dtype=np.float64)
    np.save(state_path, altered, allow_pickle=False)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["samples"][0]["init_state_sha256"] = _sha256(state_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit.run_audit(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        sample_ids=["libero_90_task000"],
        output_dir=tmp_path / "missing_attr",
        inspect_only=True,
        warmup_steps=0,
    )

    record = report["records"][0]
    assert record["init_state_attr_present"] is False
    assert record["comparisons"]["exported_npy_vs_demo_init_state_attr"]["available"] is False
    assert record["comparisons"]["exported_npy_vs_hdf5_states_0"]["exact_equal"] is False
    assert record["comparisons"]["exported_npy_vs_hdf5_states_0"]["l2_error"] == 0.5


def test_manifest_and_dataset_paths_cannot_escape_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative path without"):
        audit._contained_path(tmp_path, "../outside.hdf5", field="source_hdf5_path")


def test_sample_ids_all_selects_every_manifest_entry(tmp_path: Path) -> None:
    manifest_path, _, _, sample = _fixture(tmp_path)
    manifest, selected = audit._load_manifest(manifest_path, ["all"])
    assert len(manifest["samples"]) == len(selected) == 1
    assert selected[0]["sample_id"] == sample["sample_id"]


def test_existing_audit_outputs_are_not_overwritten(tmp_path: Path) -> None:
    manifest_path, dataset_root, _, _ = _fixture(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "audit_report.json").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit.run_audit(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            sample_ids=["libero_90_task000"],
            output_dir=output_dir,
            inspect_only=True,
            warmup_steps=0,
        )
