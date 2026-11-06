#!/usr/bin/env python3
"""Compare Experiment 5 demo initial states and optionally replay HDF5 actions.

``--inspect-only`` reads the manifest, exported ``.npy`` files, and source
HDF5 files.  It does not import LIBERO, robosuite, MuJoCo, or initialize a
simulator.  Without that flag, it restores each demo's recorded XML and
``states[0]`` following LIBERO's ``scripts/create_dataset.py`` replay order,
then applies ``actions[j]`` and compares the resulting simulator state with
``states[j + 1]``.  ``--restore-mode evaluation_xml`` instead retains the
evaluation environment's BDDL-generated XML and loads the same recorded
``states[0]`` into it, mirroring the old cross-XML evaluation restore.  A
non-zero ``--warmup-steps`` deliberately adds the same dummy commands used by
the evaluation client before demo actions; transition errors are then a
warmup-perturbed comparison, not canonical demo playback.

This script collects evidence about state consistency and replay behavior; it
does not infer a cause for any model success-rate difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
DEFAULT_WARMUP_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
SAMPLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp_path.chmod(0o644)
    os.replace(temp_path, path)


def _contained_path(root: Path, relative: str, *, field: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must be a relative path without '..': {relative!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes its root: {relative!r}") from exc
    return resolved


def _load_manifest(path: Path, requested_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest must contain a 'samples' list")
    by_id: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict) or not sample.get("sample_id"):
            raise ValueError("each manifest sample must be an object with sample_id")
        sample_id = str(sample["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
        by_id[sample_id] = sample
    if requested_ids == ["all"]:
        return manifest, list(by_id.values())
    missing = sorted(set(requested_ids) - set(by_id))
    if missing:
        raise ValueError(f"requested sample IDs are absent from manifest: {missing}")
    return manifest, [by_id[sample_id] for sample_id in requested_ids]


def _array_comparison(reference: np.ndarray | None, candidate: np.ndarray | None) -> dict[str, Any]:
    if reference is None or candidate is None:
        return {"available": False, "shape_match": None, "exact_equal": None, "l2_error": None, "max_abs_error": None}
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    shape_match = reference.shape == candidate.shape
    result: dict[str, Any] = {
        "available": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_match": bool(shape_match),
        "exact_equal": None,
        "l2_error": None,
        "max_abs_error": None,
        "reference_finite": bool(np.isfinite(reference).all()) if reference.dtype.kind in "fci" else None,
        "candidate_finite": bool(np.isfinite(candidate).all()) if candidate.dtype.kind in "fci" else None,
    }
    if not shape_match:
        return result
    try:
        equal = np.array_equal(reference, candidate)
        difference = np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return result
    result["exact_equal"] = bool(equal)
    if difference.size and np.isfinite(difference).all():
        result["l2_error"] = float(np.linalg.norm(difference.ravel()))
        result["max_abs_error"] = float(np.max(np.abs(difference)))
    elif difference.size == 0:
        result["l2_error"] = 0.0
        result["max_abs_error"] = 0.0
    return result


def _decode_text(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return None


def _read_sample_evidence(
    *,
    sample: dict[str, Any],
    manifest_path: Path,
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sample_id = str(sample["sample_id"])
    if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
        raise ValueError(f"unsafe sample_id for output filename: {sample_id!r}")
    source_path = _contained_path(dataset_root, str(sample["source_hdf5_path"]), field="source_hdf5_path")
    if not source_path.is_file():
        raise FileNotFoundError(f"HDF5 demo does not exist for {sample_id}: {source_path}")
    demo_id = str(sample.get("demo_id", ""))
    if not re.fullmatch(r"demo_\d+", demo_id):
        raise ValueError(f"invalid demo_id in manifest for {sample_id}: {demo_id!r}")

    exported_state: np.ndarray | None = None
    exported_path: Path | None = None
    exported_hash: str | None = None
    exported_load_error: str | None = None
    init_state_path = sample.get("init_state_path")
    if init_state_path:
        try:
            exported_path = _contained_path(manifest_path.parent, str(init_state_path), field="init_state_path")
            if exported_path.is_file():
                exported_hash = _sha256_file(exported_path)
                exported_state = np.load(exported_path, allow_pickle=False)
            else:
                exported_load_error = "file_missing"
        except (OSError, ValueError) as exc:
            exported_load_error = f"{type(exc).__name__}: {exc}"

    with h5py.File(source_path, "r") as dataset:
        if "data" not in dataset or demo_id not in dataset["data"]:
            raise KeyError(f"missing /data/{demo_id} in {source_path}")
        root_data = dataset["data"]
        demo = root_data[demo_id]
        if "actions" not in demo or "states" not in demo:
            raise ValueError(f"{source_path}:/data/{demo_id} must contain actions and states")
        actions = np.asarray(demo["actions"][()])
        states = np.asarray(demo["states"][()])
        attr_state = None
        if "init_state" in demo.attrs:
            attr_state = np.asarray(demo.attrs["init_state"])
        model_xml = _decode_text(demo.attrs.get("model_file"))
        root_env_args = _decode_text(root_data.attrs.get("env_args"))
        root_problem_info = _decode_text(root_data.attrs.get("problem_info"))
        source_attributes = sorted(str(key) for key in demo.attrs.keys())
        root_attributes = sorted(str(key) for key in root_data.attrs.keys())

    first_state = states[0] if states.ndim >= 1 and states.shape[0] else None
    comparisons = {
        "hdf5_states_0_vs_demo_init_state_attr": _array_comparison(first_state, attr_state),
        "exported_npy_vs_demo_init_state_attr": _array_comparison(exported_state, attr_state),
        "exported_npy_vs_hdf5_states_0": _array_comparison(exported_state, first_state),
    }
    expected_manifest_hash = sample.get("init_state_sha256")
    evidence: dict[str, Any] = {
        "sample_id": sample_id,
        "suite_name": sample.get("suite_name"),
        "task_id": sample.get("task_id"),
        "task_name": sample.get("task_name"),
        "demo_id": demo_id,
        "source_hdf5_path": str(source_path),
        "source_hdf5_sha256": _sha256_file(source_path),
        "manifest_source_hdf5_path": sample.get("source_hdf5_path"),
        "demo_attrs": source_attributes,
        "root_data_attrs": root_attributes,
        "init_state_attr_present": attr_state is not None,
        "model_file_present": model_xml is not None,
        "model_file_sha256": None if model_xml is None else _sha256_bytes(model_xml.encode("utf-8")),
        "manifest_init_state_path": sample.get("init_state_path"),
        "exported_init_state_path": None if exported_path is None else str(exported_path),
        "exported_init_state_sha256": exported_hash,
        "manifest_expected_init_state_sha256": expected_manifest_hash,
        "exported_hash_matches_manifest": (
            None if expected_manifest_hash is None or exported_hash is None else exported_hash == expected_manifest_hash
        ),
        "exported_state_load_error": exported_load_error,
        "states_shape": list(states.shape),
        "actions_shape": list(actions.shape),
        "state_action_count_match": bool(states.ndim >= 1 and actions.ndim >= 1 and states.shape[0] == actions.shape[0]),
        "root_env_args_json": root_env_args,
        "root_problem_info_json": root_problem_info,
        "comparisons": comparisons,
        "replay_status": "not_run",
    }
    raw_arrays: dict[str, np.ndarray] = {"hdf5_states": states, "actions": actions}
    if attr_state is not None:
        raw_arrays["demo_init_state_attr"] = attr_state
    if first_state is not None:
        raw_arrays["hdf5_states_0"] = np.asarray(first_state)
    if exported_state is not None:
        raw_arrays["exported_init_state"] = np.asarray(exported_state)
    return evidence, raw_arrays


def _load_replay_components() -> tuple[tuple[Any, Any, Any], Any]:
    """Import simulator dependencies only for actual replay mode."""
    vendored_libero = str(REPO_ROOT / "third_party" / "libero")
    if vendored_libero not in sys.path:
        sys.path.insert(0, vendored_libero)
    spec_scripts = str(REPO_ROOT / "scripts" / "spec")
    if spec_scripts not in sys.path:
        sys.path.insert(0, spec_scripts)
    try:
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.utils.utils import postprocess_model_xml
        import spec_client_libero as libero_client
    except ImportError as exc:
        raise RuntimeError(
            "Replay mode requires this checkout's LIBERO package, robosuite, MuJoCo, and their runtime dependencies. "
            "Use the LIBERO evaluation environment; --inspect-only does not need these packages."
        ) from exc
    return (benchmark, libero_client, get_libero_path), postprocess_model_xml


def _xml_text_and_hash(xml_value: Any) -> tuple[str, str]:
    if isinstance(xml_value, bytes):
        xml_text = xml_value.decode("utf-8")
    else:
        xml_text = str(xml_value)
    return xml_text, _sha256_bytes(xml_text.encode("utf-8"))


def _remap_explicit_asset_paths(xml_text: str, assets_root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Map only mesh/texture paths carrying an explicit ``/assets/`` marker.

    LIBERO's vendored postprocessor handles robosuite paths, while this
    manifest's demo XML also has stale absolute paths under
    ``chiliocosm/assets``.  Other XML paths are preserved.  Missing or
    escaping targets remain unchanged and are recorded for the caller to
    reject before restoring the XML.
    """
    assets_root = assets_root.expanduser().resolve()
    root = ET.fromstring(xml_text)
    mappings: list[dict[str, Any]] = []
    for element in root.iter():
        if element.tag not in {"mesh", "texture"}:
            continue
        source_path = element.get("file")
        if not source_path:
            continue
        normalized_source = source_path.replace("\\", "/")
        marker = "/assets/"
        if marker not in normalized_source:
            continue
        prefix, relative_suffix = normalized_source.split(marker, 1)
        # The vendored postprocess_model_xml() has already normalized paths
        # rooted in robosuite's own package.  Their suffix is relative to
        # robosuite/models/assets, not LIBERO's configured assets directory.
        # Never remap those unrelated files into LIBERO's assets tree.
        if "robosuite" in {part.lower() for part in prefix.split("/") if part}:
            continue
        target = (assets_root / relative_suffix).resolve()
        try:
            target.relative_to(assets_root)
            target_within_root = True
        except ValueError:
            target_within_root = False
        target_exists = bool(target_within_root and target.is_file())
        mappings.append(
            {
                "element": element.tag,
                "name": element.get("name"),
                "original_path": source_path,
                "assets_relative_suffix": relative_suffix,
                "target_path": str(target),
                "target_within_assets_root": target_within_root,
                "target_exists": target_exists,
                "mapped_path": str(target) if target_exists else None,
            }
        )
        if target_exists:
            element.set("file", str(target))
    return ET.tostring(root, encoding="unicode"), mappings


def _sim_model_dimensions(env: Any) -> dict[str, Any]:
    model = env.sim.model
    state = np.asarray(env.sim.get_state().flatten())
    names = ("nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nsite", "ncam")
    dimensions = {name: int(getattr(model, name)) for name in names if hasattr(model, name)}
    dimensions["flattened_state_dim"] = int(state.size)
    dimensions["flattened_state_dtype"] = str(state.dtype)
    return dimensions


def _state_component_errors(
    expected: np.ndarray,
    actual: np.ndarray,
    dimensions: dict[str, Any],
) -> dict[str, Any]:
    """Split robosuite MjSimState.flatten() into time/qpos/qvel/act when sized.

    LIBERO's dataset replay compares the complete flattened state.  This
    supplemental view makes the leading simulation-time scalar explicit so
    it cannot be mistaken for a physical pose or velocity difference.
    """
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    if expected.shape != actual.shape:
        return {"layout_valid": False, "shape_match": False}
    nq = dimensions.get("nq")
    nv = dimensions.get("nv")
    na = dimensions.get("na", 0)
    if nq is None or nv is None:
        return {"layout_valid": False, "shape_match": True}
    sizes = {"time": 1, "qpos": int(nq), "qvel": int(nv), "act": int(na)}
    if sum(sizes.values()) != expected.size:
        return {
            "layout_valid": False,
            "shape_match": True,
            "expected_flattened_dim_from_model": int(sum(sizes.values())),
            "actual_flattened_dim": int(expected.size),
        }
    result: dict[str, Any] = {"layout_valid": True, "shape_match": True}
    offset = 0
    for name, size in sizes.items():
        delta = expected[offset : offset + size] - actual[offset : offset + size]
        result[f"{name}_l2_error"] = float(np.linalg.norm(delta))
        result[f"{name}_max_abs_error"] = float(np.max(np.abs(delta))) if delta.size else 0.0
        offset += size
    return result


def _replay_sample(
    *,
    sample: dict[str, Any],
    raw_arrays: dict[str, np.ndarray],
    evidence: dict[str, Any],
    warmup_steps: int,
    restore_mode: str,
    replay_components: tuple[Any, Any, Any],
    postprocess_model_xml: Any,
) -> dict[str, np.ndarray]:
    benchmark, libero_client, get_libero_path = replay_components
    if "model_file" not in evidence["demo_attrs"]:
        raise ValueError("demo attrs do not include model_file required for exact XML restoration")
    suite_name = str(sample["suite_name"])
    suite_class = benchmark.get_benchmark_dict()[suite_name]
    suite = suite_class()
    task = suite.get_task(int(sample["task_id"]))
    env_seed = int(sample.get("env_seed", 7))
    env, _ = libero_client._get_libero_env(task, libero_client.LIBERO_ENV_RESOLUTION, env_seed)
    actions = raw_arrays["actions"]
    states = raw_arrays["hdf5_states"]
    if actions.ndim != 2 or actions.shape[1] != 7:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise ValueError(f"expected HDF5 demo actions shaped [T,7], got {actions.shape}")
    if states.ndim != 2 or states.shape[0] != actions.shape[0]:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise ValueError(f"expected HDF5 states shaped [T,D] with T=len(actions), got {states.shape} vs {actions.shape}")

    post_states: list[np.ndarray] = []
    post_warmup_states: list[np.ndarray] = []
    state_errors: list[float] = []
    component_errors: list[dict[str, Any]] = []
    state_error_steps: list[int] = []
    done_values: list[bool] = []
    reward_values: list[float] = []
    first_done_step: int | None = None
    replay_error: str | None = None
    replayed_actions = 0
    final_success: bool | None = None
    warmup_action_count = 0
    processed_xml_hash: str | None = None
    postprocessed_xml_hash: str | None = None
    demo_xml_hash: str | None = None
    evaluation_xml_hash: str | None = None
    selected_xml_hash: str | None = None
    selected_compiled_xml_hash: str | None = None
    demo_model_dimensions: dict[str, Any] | None = None
    evaluation_model_dimensions: dict[str, Any] | None = None
    selected_model_dimensions: dict[str, Any] | None = None
    restored_initial_state: np.ndarray | None = None
    state_restoration_completed = False
    asset_path_mappings: list[dict[str, Any]] = []
    asset_mapping_error: str | None = None
    try:
        env.reset()
        with h5py.File(evidence["source_hdf5_path"], "r") as dataset:
            xml_value = dataset["data"][str(sample["demo_id"])].attrs["model_file"]
        raw_demo_xml, demo_xml_hash = _xml_text_and_hash(xml_value)
        model_xml = postprocess_model_xml(raw_demo_xml, {})
        model_xml = str(model_xml)
        postprocessed_xml_hash = _sha256_bytes(model_xml.encode("utf-8"))
        evaluation_xml, evaluation_xml_hash = _xml_text_and_hash(env.sim.model.get_xml())
        evaluation_model_dimensions = _sim_model_dimensions(env)
        # This order mirrors LIBERO/scripts/create_dataset.py: reset, restore
        # the demo XML, reset MuJoCo, set the flattened states[0], then apply
        # action[j] and compare against states[j+1].
        if restore_mode == "demo_xml":
            model_xml, asset_path_mappings = _remap_explicit_asset_paths(
                model_xml,
                Path(get_libero_path("assets")),
            )
            evidence["asset_path_mappings"] = asset_path_mappings
            unresolved = [mapping for mapping in asset_path_mappings if not mapping["target_exists"]]
            if unresolved:
                asset_mapping_error = json.dumps(unresolved, ensure_ascii=False, sort_keys=True)
                raise FileNotFoundError(
                    "demo XML asset targets are missing or outside the configured assets root: "
                    + asset_mapping_error
                )
            processed_xml_hash = _sha256_bytes(model_xml.encode("utf-8"))
            env.reset_from_xml_string(model_xml)
            selected_xml_hash = processed_xml_hash
            demo_model_dimensions = _sim_model_dimensions(env)
            selected_model_dimensions = demo_model_dimensions
        elif restore_mode == "evaluation_xml":
            selected_xml_hash = evaluation_xml_hash
            selected_model_dimensions = evaluation_model_dimensions
        else:
            raise ValueError(f"unsupported restore_mode: {restore_mode}")
        _, selected_compiled_xml_hash = _xml_text_and_hash(env.sim.model.get_xml())
        env.sim.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        restored_initial_state = np.asarray(env.sim.get_state().flatten()).copy()
        state_restoration_completed = True
        selected_model_dimensions = _sim_model_dimensions(env)
        for _ in range(warmup_steps):
            env.step(DEFAULT_WARMUP_ACTION)
            warmup_action_count += 1
            post_warmup_states.append(np.asarray(env.sim.get_state().flatten()).copy())
        for step, action in enumerate(actions):
            _, reward, done, _ = env.step(action)
            state = np.asarray(env.sim.get_state().flatten()).copy()
            post_states.append(state)
            replayed_actions += 1
            done_value = bool(done)
            done_values.append(done_value)
            reward_values.append(float(reward))
            if done_value and first_done_step is None:
                first_done_step = int(step)
            if step + 1 < len(states) and state.shape == states[step + 1].shape:
                state_errors.append(float(np.linalg.norm(state.astype(np.float64) - states[step + 1].astype(np.float64))))
                state_error_steps.append(int(step))
                component_errors.append(
                    _state_component_errors(states[step + 1], state, selected_model_dimensions or {})
                )
            elif step + 1 < len(states):
                state_errors.append(float("nan"))
                state_error_steps.append(int(step))
                component_errors.append({"layout_valid": False, "shape_match": False})
        check_success = getattr(env, "_check_success", None)
        final_success = bool(check_success()) if callable(check_success) else (done_values[-1] if done_values else False)
    except Exception as exc:  # Preserve inspection and partial replay evidence on simulator errors.
        replay_error = f"{type(exc).__name__}: {exc}"
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    error_array = np.asarray(state_errors, dtype=np.float64)
    finite_errors = error_array[np.isfinite(error_array)]
    evidence.update(
        {
            "replay_status": "error" if replay_error else "completed",
            "replay_error": replay_error,
            "restore_mode": restore_mode,
            "xml_restore": (
                "demo attrs model_file processed with LIBERO postprocess_model_xml; env.sim.reset(); states[0] restored"
                if restore_mode == "demo_xml"
                else "evaluation BDDL-generated XML retained; env.sim.reset(); demo states[0] restored across XMLs"
            ),
            "demo_xml_sha256": demo_xml_hash,
            "original_demo_xml_sha256": demo_xml_hash,
            "postprocessed_demo_xml_sha256": postprocessed_xml_hash,
            "libero_postprocess_demo_xml_sha256": postprocessed_xml_hash,
            "processed_model_file_sha256": processed_xml_hash,
            "asset_remapped_demo_xml_sha256": processed_xml_hash,
            "evaluation_xml_sha256": evaluation_xml_hash,
            "selected_xml_sha256": selected_xml_hash,
            "selected_compiled_xml_sha256": selected_compiled_xml_hash,
            "evaluation_model_dimensions": evaluation_model_dimensions,
            "demo_model_dimensions": demo_model_dimensions,
            "selected_model_dimensions": selected_model_dimensions,
            "state_restoration_completed": state_restoration_completed,
            "asset_path_mappings": asset_path_mappings,
            "asset_mapping_error": asset_mapping_error,
            "hdf5_recorded_state_dimensions": list(states.shape),
            "recorded_state_dim_matches_selected_model_flattened_dim": (
                None
                if selected_model_dimensions is None
                else int(states.shape[1]) == selected_model_dimensions["flattened_state_dim"]
            ),
            "warmup_steps": int(warmup_steps),
            "warmup_action": DEFAULT_WARMUP_ACTION.tolist(),
            "warmup_action_count": int(warmup_action_count),
            "replayed_action_count": int(replayed_actions),
            "first_done_step_zero_based": first_done_step,
            "success_any_step": bool(any(done_values)),
            "success_final_state": final_success,
            "done_values": done_values,
            "reward_values": [float(value) if np.isfinite(value) else None for value in reward_values],
            "state_error_reference": "norm(replayed_state_after_action_j - hdf5_states[j+1]); canonical only when warmup_steps=0",
            "state_error_action_steps_zero_based": state_error_steps,
            "state_error_l2": [None if not np.isfinite(error) else float(error) for error in error_array],
            "state_component_error_reference": (
                "MjSimState.flatten layout: time[1], qpos[nq], qvel[nv], act[na]; component norms are separate from full-state norm"
            ),
            "state_component_errors": component_errors,
            "state_error_count": int(error_array.size),
            "state_error_finite_count": int(finite_errors.size),
            "state_error_mean_l2": None if not finite_errors.size else float(np.mean(finite_errors)),
            "state_error_max_l2": None if not finite_errors.size else float(np.max(finite_errors)),
            "state_error_first_over_0_01": next(
                (int(step) for step, error in zip(state_error_steps, error_array) if np.isfinite(error) and error > 0.01),
                None,
            ),
            "state_error_alignment_note": (
                "canonical LIBERO demo action/state alignment"
                if warmup_steps == 0
                else "warmup commands perturb the recorded demo initial state; errors quantify divergence after warmup"
            ),
        }
    )
    raw_arrays["replay_post_action_states"] = (
        np.stack(post_states) if post_states else np.empty((0, states.shape[1]), dtype=states.dtype)
    )
    raw_arrays["replay_post_warmup_states"] = (
        np.stack(post_warmup_states)
        if post_warmup_states
        else np.empty((0, states.shape[1]), dtype=states.dtype)
    )
    raw_arrays["replay_state_error_l2"] = error_array
    if restored_initial_state is not None:
        raw_arrays["replay_restored_initial_state"] = restored_initial_state
    raw_arrays["replay_done"] = np.asarray(done_values, dtype=np.bool_)
    raw_arrays["replay_reward"] = np.asarray(reward_values, dtype=np.float64)
    return raw_arrays


def _parse_sample_ids(value: str) -> list[str]:
    sample_ids = [part.strip() for part in value.split(",") if part.strip()]
    if not sample_ids:
        raise argparse.ArgumentTypeError("provide at least one comma-separated sample ID")
    if len(sample_ids) != len(set(sample_ids)):
        raise argparse.ArgumentTypeError("sample IDs must not contain duplicates")
    return sample_ids


def run_audit(
    *,
    manifest_path: Path,
    dataset_root: Path,
    sample_ids: list[str],
    output_dir: Path,
    inspect_only: bool,
    warmup_steps: int,
    restore_mode: str = "demo_xml",
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if restore_mode not in {"demo_xml", "evaluation_xml"}:
        raise ValueError("restore_mode must be 'demo_xml' or 'evaluation_xml'")
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be a non-empty list without duplicates")
    all_samples_requested = sample_ids == ["all"]
    invalid_sample_ids = [
        sample_id for sample_id in sample_ids
        if not all_samples_requested and not SAMPLE_ID_PATTERN.fullmatch(sample_id)
    ]
    if invalid_sample_ids:
        raise ValueError(f"unsafe sample IDs for output filenames: {invalid_sample_ids}")
    manifest, selected = _load_manifest(manifest_path, sample_ids)
    sample_ids = [str(sample["sample_id"]) for sample in selected]
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    report_path = output_dir / "audit_report.json"
    evidence_paths = [output_dir / "raw_evidence" / f"{sample_id}.npz" for sample_id in sample_ids]
    if config_path.exists() or report_path.exists() or any(path.exists() for path in evidence_paths):
        raise FileExistsError(f"refusing to overwrite an existing audit in {output_dir}")
    config = {
        "schema_version": SCHEMA_VERSION,
        "execution_mode": "inspect_only" if inspect_only else "simulator_replay",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "dataset_root": str(dataset_root),
        "sample_ids": sample_ids,
        "output_dir": str(output_dir),
        "warmup_steps": int(warmup_steps),
        "restore_mode": restore_mode,
        "warmup_action": DEFAULT_WARMUP_ACTION.tolist(),
        "replay_alignment_reference": "third_party/libero/scripts/create_dataset.py (LIBERO replay order: states[0], action[j], states[j+1])",
        "state_error_definition": "L2(replayed simulator state after action[j] - recorded states[j+1]); omitted for the last action",
        "state_error_threshold_reference": 0.01,
        "inspect_only_imports_simulator": False,
        "causal_conclusion": None,
    }
    replay_components = postprocess_model_xml = None
    if not inspect_only:
        replay_components, postprocess_model_xml = _load_replay_components()
    _atomic_json(config_path, config)

    records: list[dict[str, Any]] = []
    artifact_dir = output_dir / "raw_evidence"
    artifact_dir.mkdir(exist_ok=True)
    for sample in selected:
        try:
            evidence, arrays = _read_sample_evidence(
                sample=sample,
                manifest_path=manifest_path,
                dataset_root=dataset_root,
            )
        except Exception as exc:
            records.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "inspection_status": "error",
                    "inspection_error": f"{type(exc).__name__}: {exc}",
                    "replay_status": "not_run",
                }
            )
            continue
        if not inspect_only:
            try:
                arrays = _replay_sample(
                    sample=sample,
                    raw_arrays=arrays,
                    evidence=evidence,
                    warmup_steps=warmup_steps,
                    restore_mode=restore_mode,
                    replay_components=replay_components,
                    postprocess_model_xml=postprocess_model_xml,
                )
            except Exception as exc:
                evidence["replay_status"] = "error"
                evidence["replay_error"] = f"{type(exc).__name__}: {exc}"
        evidence["inspection_status"] = "completed"
        npz_path = artifact_dir / f"{sample['sample_id']}.npz"
        np.savez_compressed(npz_path, **arrays)
        evidence["raw_evidence_npz"] = str(npz_path)
        evidence["raw_evidence_npz_sha256"] = _sha256_file(npz_path)
        records.append(evidence)

    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_mode": config["execution_mode"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": config["manifest_sha256"],
        "dataset_root": str(dataset_root),
        "sample_count": len(records),
        "inspected_count": len(records),
        "inspection_error_count": sum(record.get("inspection_status") == "error" for record in records),
        "simulator_replay_attempt_count": sum(record["replay_status"] in ("completed", "error") for record in records),
        "simulator_replayed_count": sum(record["replay_status"] == "completed" for record in records),
        "replay_error_count": sum(record["replay_status"] == "error" for record in records),
        "warmup_steps": int(warmup_steps),
        "restore_mode": restore_mode,
        "raw_evidence_directory": str(artifact_dir),
        "scope": (
            "State/file consistency and optional recorded-action replay only. These measurements are diagnostic evidence, "
            "not a causal attribution of Control's success rate."
        ),
        "records": records,
    }
    _atomic_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--sample-ids",
        type=_parse_sample_ids,
        required=True,
        help="comma-separated manifest sample IDs, or 'all' for every sample",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true", help="compare files/states without importing or launching LIBERO")
    parser.add_argument(
        "--restore-mode",
        choices=("demo_xml", "evaluation_xml"),
        default="demo_xml",
        help="restore the demo XML or keep the BDDL-generated evaluation XML before loading HDF5 states[0]",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="dummy actions between states[0] restoration and original demo actions (default: 0; use 10 to measure warmup divergence)",
    )
    args = parser.parse_args()
    report = run_audit(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        sample_ids=args.sample_ids,
        output_dir=args.output_dir,
        inspect_only=args.inspect_only,
        warmup_steps=args.warmup_steps,
        restore_mode=args.restore_mode,
    )
    print(
        json.dumps(
            {
                "execution_mode": report["execution_mode"],
                "sample_count": report["sample_count"],
                "simulator_replayed_count": report["simulator_replayed_count"],
                "simulator_replay_attempt_count": report["simulator_replay_attempt_count"],
                "replay_error_count": report["replay_error_count"],
                "inspection_error_count": report["inspection_error_count"],
                "restore_mode": report["restore_mode"],
                "warmup_steps": report["warmup_steps"],
                "report": str(args.output_dir.resolve() / "audit_report.json"),
            },
            indent=2,
        )
    )
    return 1 if report["replay_error_count"] or report["inspection_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
