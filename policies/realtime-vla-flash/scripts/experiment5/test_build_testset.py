"""Lightweight tests for LIBERO test-set manifest generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from build_testset import _task_map_from_source
from build_testset import build_testset
from build_testset import load_init_state
import h5py
import numpy as np
import pytest


def _write_task_dataset(path: Path, *, instruction: str, demo_lengths: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as dataset:
        data = dataset.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": instruction})
        for demo_index, length in demo_lengths.items():
            demo = data.create_group(f"demo_{demo_index}")
            demo.create_dataset("actions", shape=(length, 7), dtype=np.float64)
            demo.attrs["init_state"] = np.full(5, demo_index, dtype=np.float64)


def test_vendored_map_has_expected_90_plus_10_tasks() -> None:
    task_map = _task_map_from_source()
    assert {suite: len(tasks) for suite, tasks in task_map.items()} == {"libero_90": 90, "libero_10": 10}
    assert len({task for tasks in task_map.values() for task in tasks}) == 100


def test_shortest_demo_manifest_and_state_references_are_stable() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        dataset_root = root / "datasets"
        task_map = {
            "libero_90": ["TASK_ALPHA"],
            "libero_10": ["TASK_BETA"],
        }
        _write_task_dataset(
            dataset_root / "libero_90/TASK_ALPHA_demo.hdf5",
            instruction="move alpha",
            demo_lengths={8: 2, 2: 2, 0: 4},
        )
        _write_task_dataset(
            dataset_root / "libero_10/TASK_BETA_demo.hdf5",
            instruction="move beta",
            demo_lengths={0: 3, 1: 1},
        )
        manifest_path = root / "out/testset_manifest.json"

        first = build_testset(
            dataset_root=dataset_root,
            output_path=manifest_path,
            task_map=task_map,
            env_seed=7,
            noise_seed=12345,
            expected_task_counts={"libero_90": 1, "libero_10": 1},
        )
        first_bytes = manifest_path.read_bytes()
        second = build_testset(
            dataset_root=dataset_root,
            output_path=manifest_path,
            task_map=task_map,
            env_seed=7,
            noise_seed=12345,
            expected_task_counts={"libero_90": 1, "libero_10": 1},
        )

        assert first_bytes == manifest_path.read_bytes()
        assert first["manifest_sha256"] == second["manifest_sha256"]
        assert hashlib.sha256(
            json.dumps(
                {key: value for key, value in first.items() if key != "manifest_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest() == first["manifest_sha256"]

        alpha, beta = first["samples"]
        assert (alpha["suite_name"], alpha["task_id"], alpha["demo_id"]) == ("libero_90", 0, "demo_2")
        assert alpha["trajectory_length"] == 2
        assert beta["demo_id"] == "demo_1"
        assert beta["trajectory_length"] == 1
        assert alpha["env_seed"] == beta["env_seed"] == 7
        assert alpha["noise_seed"] == beta["noise_seed"] == 12345
        state = load_init_state(manifest_path, alpha)
        np.testing.assert_array_equal(state, np.full(5, 2.0))

        state_path = manifest_path.parent / alpha["init_state_path"]
        state_path.write_bytes(state_path.read_bytes() + b"tamper")
        with pytest.raises(ValueError, match="hash mismatch"):
            load_init_state(manifest_path, alpha)
