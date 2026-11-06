"""Build a reproducible one-demo-per-task LIBERO pruning test set."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
TASK_MAP_PATH = REPO_ROOT / "third_party/libero/libero/libero/benchmark/libero_suite_task_map.py"
DEFAULT_OUTPUT = Path(__file__).with_name("testset_manifest.json")
EXPECTED_TASK_COUNTS = {"libero_90": 90, "libero_10": 10}
DEMO_PATTERN = re.compile(r"demo_(\d+)")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return _sha256_bytes(payload.encode("utf-8"))


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


def _task_map_from_source(path: Path = TASK_MAP_PATH) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return {suite: value[suite] for suite in EXPECTED_TASK_COUNTS}
    raise ValueError(f"libero_task_map assignment not found in {path}")


def _config_dataset_root() -> Path | None:
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", REPO_ROOT / ".libero")).expanduser()
    config_path = config_dir / "config.yaml"
    if not config_path.is_file():
        return None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*datasets:\s*(.*?)\s*(?:#.*)?$", line)
        if match:
            value = match.group(1).strip().strip("'\"")
            return Path(value).expanduser()
    return None


def resolve_dataset_root(explicit_root: Path | None = None) -> Path:
    if explicit_root is not None:
        candidates = [explicit_root]
    else:
        candidates = [
            _config_dataset_root(),
            WORKSPACE_ROOT / "LIBERO/libero/datasets",
            REPO_ROOT / "third_party/libero/libero/datasets",
        ]
    for candidate in candidates:
        if candidate is not None and all((candidate / suite).is_dir() for suite in EXPECTED_TASK_COUNTS):
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"could not find libero_90 and libero_10 under dataset roots: {checked}")


def _demo_index(demo_id: str) -> int | None:
    match = DEMO_PATTERN.fullmatch(demo_id)
    return int(match.group(1)) if match else None


def _read_task_metadata(dataset_path: Path) -> tuple[int, int, str, np.ndarray, str]:
    """Return the shortest demo metadata and language task text."""
    with h5py.File(dataset_path, "r") as dataset:
        if "data" not in dataset:
            raise ValueError(f"missing /data group in {dataset_path}")
        data = dataset["data"]
        candidates: list[tuple[int, int, str]] = []
        for demo_id in data:
            index = _demo_index(demo_id)
            if index is None:
                continue
            demo = data[demo_id]
            if "actions" not in demo or demo["actions"].ndim < 1:
                raise ValueError(f"missing or invalid actions in {dataset_path}:/data/{demo_id}")
            length = int(demo["actions"].shape[0])
            if length < 1:
                raise ValueError(f"empty action trajectory in {dataset_path}:/data/{demo_id}")
            candidates.append((length, index, demo_id))
        if not candidates:
            raise ValueError(f"no demo_N groups found in {dataset_path}:/data")

        length, _, selected_id = min(candidates)
        selected_demo = data[selected_id]
        if "init_state" not in selected_demo.attrs:
            raise ValueError(f"missing init_state in {dataset_path}:/data/{selected_id}")
        init_state = np.asarray(selected_demo.attrs["init_state"], dtype="<f8")
        if init_state.ndim != 1 or init_state.size == 0 or not np.isfinite(init_state).all():
            raise ValueError(f"invalid init_state in {dataset_path}:/data/{selected_id}")

        instruction = ""
        problem_info = data.attrs.get("problem_info")
        if problem_info is not None:
            if isinstance(problem_info, bytes):
                problem_info = problem_info.decode("utf-8")
            try:
                instruction = str(json.loads(str(problem_info)).get("language_instruction", ""))
            except (TypeError, ValueError):
                instruction = ""
    demo_index = _demo_index(selected_id)
    assert demo_index is not None
    return length, demo_index, selected_id, np.ascontiguousarray(init_state), instruction


def _save_init_state(path: Path, init_state: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, init_state, allow_pickle=False)
    return _sha256_bytes(path.read_bytes())


def build_testset(
    *,
    dataset_root: Path,
    output_path: Path,
    task_map: dict[str, list[str]],
    env_seed: int = 7,
    noise_seed: int = 12345,
    expected_task_counts: dict[str, int] | None = EXPECTED_TASK_COUNTS,
) -> dict[str, Any]:
    if set(task_map) != set(EXPECTED_TASK_COUNTS):
        raise ValueError(f"task map must contain exactly {sorted(EXPECTED_TASK_COUNTS)}")
    if expected_task_counts is not None:
        actual_counts = {suite: len(tasks) for suite, tasks in task_map.items()}
        if actual_counts != expected_task_counts:
            raise ValueError(f"unexpected suite task counts: {actual_counts}; expected {expected_task_counts}")
    all_task_names = [task for tasks in task_map.values() for task in tasks]
    if len(all_task_names) != len(set(all_task_names)):
        raise ValueError("suite task lists must not contain duplicate tasks")

    output_path = output_path.resolve()
    state_dir = output_path.with_suffix("").with_name(f"{output_path.stem}.init_states")
    samples: list[dict[str, Any]] = []
    for suite_name in ("libero_90", "libero_10"):
        for task_id, task_name in enumerate(task_map[suite_name]):
            source_path = dataset_root / suite_name / f"{task_name}_demo.hdf5"
            if not source_path.is_file():
                raise FileNotFoundError(f"missing dataset for {suite_name} task {task_id}: {source_path}")
            trajectory_length, demo_index, demo_id, init_state, task_description = _read_task_metadata(source_path)
            sample_id = f"{suite_name}_task{task_id:03d}"
            init_state_path = state_dir / f"{sample_id}.npy"
            init_state_sha256 = _save_init_state(init_state_path, init_state)
            samples.append(
                {
                    "sample_id": sample_id,
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "task_name": task_name,
                    "task_description": task_description,
                    "demo_id": demo_id,
                    "demo_index": demo_index,
                    "source_hdf5_path": source_path.relative_to(dataset_root).as_posix(),
                    "init_state_path": init_state_path.relative_to(output_path.parent).as_posix(),
                    "init_state_sha256": init_state_sha256,
                    "trajectory_length": trajectory_length,
                    "env_seed": int(env_seed),
                    "noise_seed": int(noise_seed),
                }
            )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "vlm_backbone_fine_pruning",
        "selection": {
            "suites": {suite: len(task_map[suite]) for suite in ("libero_90", "libero_10")},
            "task_order_source": "vendored LIBERO libero_task_map order",
            "demo_rule": "minimum actions.shape[0]; ties resolved by lowest numeric demo index",
            "init_state_storage": "float64 NumPy .npy file, relative to this manifest",
        },
        "seed_policy": {"env_seed": int(env_seed), "noise_seed": int(noise_seed)},
        "samples": samples,
    }
    manifest["manifest_sha256"] = _canonical_json_hash(manifest)
    _atomic_json(output_path, manifest)
    return manifest


def load_init_state(manifest_path: Path, sample: dict[str, Any]) -> np.ndarray:
    """Load an init state referenced by a sample and verify its content hash."""
    relative_path = Path(sample["init_state_path"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("init_state_path must be a relative path contained by the manifest directory")
    path = manifest_path.resolve().parent / relative_path
    if _sha256_bytes(path.read_bytes()) != sample["init_state_sha256"]:
        raise ValueError(f"init state hash mismatch: {path}")
    state = np.load(path, allow_pickle=False)
    if state.ndim != 1 or state.dtype.kind != "f" or not np.isfinite(state).all():
        raise ValueError(f"invalid init state array: {path}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=12345)
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root)
    manifest = build_testset(
        dataset_root=dataset_root,
        output_path=args.output,
        task_map=_task_map_from_source(),
        env_seed=args.env_seed,
        noise_seed=args.noise_seed,
    )
    print(
        f"Wrote {len(manifest['samples'])} samples to {args.output.resolve()} "
        f"(sha256={manifest['manifest_sha256']})"
    )


if __name__ == "__main__":
    main()
