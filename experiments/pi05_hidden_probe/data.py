"""Validated observation manifests, leakage-safe splits, and matched donors.

The manifest contains independently supplied labels. This module never creates
semantic labels, flips images, invents robot states, or copies probe questions
into the policy's source prompt. Paths are resolved relative to the JSONL file.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import copy
import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping


CAMERA_KEYS = ("observation/image", "observation/wrist_image")
_CAMERA_ALIASES = {
    "observation/image": CAMERA_KEYS[0],
    "agent": CAMERA_KEYS[0],
    "base_0_rgb": CAMERA_KEYS[0],
    "agent/base_0_rgb": CAMERA_KEYS[0],
    "observation/wrist_image": CAMERA_KEYS[1],
    "wrist": CAMERA_KEYS[1],
    "left_wrist_0_rgb": CAMERA_KEYS[1],
    "wrist/left_wrist_0_rgb": CAMERA_KEYS[1],
}
_GROUP_FIELDS = ("episode_id", "scene_id", "counterfactual_group_id")
_TEXT_FIELDS = (
    "sample_id", *_GROUP_FIELDS, "original_task_instruction", "task_type",
    "label", "label_source", "question", "prompt_id",
)


class ManifestError(ValueError):
    """A manifest cannot support the promised experiment safely."""


def _fail(row: Mapping[str, Any], message: str) -> None:
    raise ManifestError(f"sample {row.get('sample_id', '<unknown>')!r}: {message}")


def _canonical_cameras(value: Any, row: Mapping[str, Any], field: str) -> dict:
    if not isinstance(value, dict) or not value:
        _fail(row, f"{field} must be a nonempty object")
    result = {}
    for name, entry in value.items():
        if name not in _CAMERA_ALIASES:
            _fail(row, f"unsupported camera {name!r} in {field}; expected {CAMERA_KEYS}")
        canonical = _CAMERA_ALIASES[name]
        if canonical in result:
            _fail(row, f"duplicate aliases for {canonical!r} in {field}")
        result[canonical] = entry
    return {key: result[key] for key in CAMERA_KEYS if key in result}


def _path(value: Any, base_dir: Path, row: Mapping[str, Any], field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(row, f"{field} must be a nonempty path string")
    path = Path(value).expanduser()
    return str((base_dir / path if not path.is_absolute() else path).resolve())


def _components(rows: list[dict]) -> list[list[int]]:
    """Union the three identity relations, including their transitive closure."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for field in _GROUP_FIELDS:
            key = (field, row[field])
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda indices: min(rows[i]["sample_id"] for i in indices))


def validate_manifest(
    rows: Iterable[Mapping[str, Any]], *, base_dir: str | Path | None = None,
    require_split: bool = False,
) -> list[dict]:
    """Return normalized copies, rejecting missing metadata and split leakage.

    Validation is metadata-only; :func:`load_observation` validates actual arrays.
    ``group_id`` is derived from the closure of episode/scene/counterfactual IDs
    and is suitable for cluster bootstrap. Counterfactual IDs remain unchanged.
    """
    base = Path(base_dir or Path.cwd()).resolve()
    normalized = []
    sample_ids = set()
    for original in rows:
        if not isinstance(original, Mapping):
            raise ManifestError("each manifest row must be an object")
        row = copy.deepcopy(dict(original))
        for field in _TEXT_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                _fail(row, f"{field} is required and must be a nonempty string")
        if row["sample_id"] in sample_ids:
            _fail(row, "duplicate sample_id")
        sample_ids.add(row["sample_id"])
        labels = row.get("labels")
        if (not isinstance(labels, list) or len(labels) < 2
                or any(not isinstance(x, str) or not x.strip() for x in labels)
                or len(set(labels)) != len(labels)):
            _fail(row, "labels must contain at least two unique nonempty answer strings")
        if row["label"] not in labels:
            _fail(row, "label is absent from labels")
        if type(row.get("is_synthetic")) is not bool:
            _fail(row, "is_synthetic must be an explicit boolean")
        if row.get("image_convention") != "policy_input":
            _fail(row, "image_convention must be 'policy_input'; convert dataset images explicitly before use")
        if row.get("coordinate_frame") is not None and not isinstance(row["coordinate_frame"], str):
            _fail(row, "coordinate_frame must be a string or null")
        if any(label.strip().lower() in ("left", "right") for label in labels):
            if not isinstance(row.get("coordinate_frame"), str) or not row["coordinate_frame"].strip():
                _fail(row, "left/right tasks require an explicit coordinate_frame")
        if "object_text" in row and (not isinstance(row["object_text"], str) or not row["object_text"].strip()):
            _fail(row, "object_text must be a nonempty string when supplied")
        has_npz = "observation_path" in row
        if has_npz == ("camera_paths" in row):
            _fail(row, "provide exactly one of observation_path or camera_paths")
        if has_npz:
            row["observation_path"] = _path(row["observation_path"], base, row, "observation_path")
            if Path(row["observation_path"]).suffix.lower() != ".npz":
                _fail(row, "observation_path must be an .npz file")
            if "state" in row or "state_path" in row:
                _fail(row, "NPZ observations supply their own observation/state; remove state/state_path")
        else:
            cameras = _canonical_cameras(row["camera_paths"], row, "camera_paths")
            if set(cameras) != set(CAMERA_KEYS):
                _fail(row, f"LIBERO policy observations require both cameras: {CAMERA_KEYS}")
            row["camera_paths"] = {key: _path(value, base, row, f"camera_paths[{key}]")
                                   for key, value in cameras.items()}
            if ("state" in row) == ("state_path" in row):
                _fail(row, "provide exactly one of state or state_path; robot state is never synthesized")
            if "state_path" in row:
                row["state_path"] = _path(row["state_path"], base, row, "state_path")
                if Path(row["state_path"]).suffix.lower() != ".npy":
                    _fail(row, "state_path must be an .npy file")
            else:
                state = row["state"]
                if (not isinstance(state, list) or not state
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in state)):
                    _fail(row, "state must be a nonempty finite numeric vector")
        if "image_masks" in row:
            masks = _canonical_cameras(row["image_masks"], row, "image_masks")
            if set(masks) != set(CAMERA_KEYS) or any(type(v) is not bool for v in masks.values()):
                _fail(row, "image_masks must explicitly give a boolean for both cameras")
            if not all(masks.values()):
                _fail(row, "the standard LIBERO adapter requires both physical cameras active; false image_masks are unsupported")
            row["image_masks"] = masks
        if "group_id" in row and (not isinstance(row["group_id"], str) or not row["group_id"]):
            _fail(row, "group_id must be a nonempty string when supplied")
        if "split" in row and row["split"] not in ("calibration", "test"):
            _fail(row, "split must be 'calibration' or 'test'")
        if require_split and "split" not in row:
            _fail(row, "split is required")
        normalized.append(row)
    if not normalized:
        raise ManifestError("manifest must contain at least one sample")
    has_splits = ["split" in row for row in normalized]
    if any(has_splits) and not all(has_splits):
        raise ManifestError("partially assigned splits are unsafe; provide all splits or remove all splits")
    for indices in _components(normalized):
        members = sorted(normalized[i]["sample_id"] for i in indices)
        digest = hashlib.sha256(json.dumps(members, ensure_ascii=False).encode()).hexdigest()[:16]
        splits = {normalized[i]["split"] for i in indices if "split" in normalized[i]}
        if len(splits) > 1:
            raise ManifestError(f"split leakage across episode/scene/counterfactual closure: {members}")
        existing_ids = {normalized[i]["group_id"] for i in indices if "group_id" in normalized[i]}
        if any(not isinstance(x, str) or not x for x in existing_ids) or len(existing_ids) > 1:
            raise ManifestError(f"inconsistent group_id within connected component: {members}")
        group_id = next(iter(existing_ids)) if existing_ids else f"group_{digest}"
        for i in indices:
            normalized[i]["group_id"] = group_id
    return normalized


def read_manifest(path: str | Path) -> list[dict]:
    """Read JSONL; normalize paths and aliases, and validate metadata and splits."""
    path = Path(path).expanduser().resolve()
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    return validate_manifest(rows, base_dir=path.parent)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def split_manifest(
    rows: Iterable[Mapping[str, Any]], calibration_fraction: float = 0.3, seed: int = 0,
) -> list[dict]:
    """Split whole independent components, preserving valid existing splits.

    A shuffled component boundary nearest the requested sample fraction is used.
    With unequal component sizes the achieved fraction is necessarily approximate.
    One connected component cannot be safely split, so that case fails explicitly.
    """
    if not 0 < calibration_fraction < 1:
        raise ManifestError("calibration_fraction must be strictly between 0 and 1")
    result = validate_manifest(rows)
    if all("split" in row for row in result):
        if {row["split"] for row in result} != {"calibration", "test"}:
            raise ManifestError("existing split must contain both calibration and test samples")
        return result
    groups = _components(result)
    if len(groups) < 2:
        raise ManifestError("all samples form one connected episode/scene/counterfactual group; add independent data instead of splitting frames")
    random.Random(seed).shuffle(groups)
    target = len(result) * calibration_fraction
    cumulative = 0
    boundaries = []
    for k, indices in enumerate(groups[:-1], 1):
        cumulative += len(indices)
        boundaries.append((abs(cumulative - target), k))
    boundary = min(boundaries)[1]
    for k, indices in enumerate(groups):
        for i in indices:
            result[i]["split"] = "calibration" if k < boundary else "test"
    return validate_manifest(result, require_split=True)


def split_summary(rows: Iterable[Mapping[str, Any]]) -> dict:
    rows = validate_manifest(rows)
    summary = {"total_samples": len(rows), "independent_groups": len({r["group_id"] for r in rows}), "splits": {}}
    for split in sorted({r.get("split", "unassigned") for r in rows}):
        selected = [r for r in rows if r.get("split", "unassigned") == split]
        summary["splits"][split] = {
            "samples": len(selected),
            "independent_groups": len({r["group_id"] for r in selected}),
            "episodes": len({r["episode_id"] for r in selected}),
            "scenes": len({r["scene_id"] for r in selected}),
            "counterfactual_groups": len({r["counterfactual_group_id"] for r in selected}),
            "task_counts": dict(sorted(Counter(r["task_type"] for r in selected).items())),
            "label_counts": dict(sorted(Counter(r["label"] for r in selected).items())),
        }
    return summary


def choose_donors(rows: Iterable[Mapping[str, Any]], seed: int = 0) -> dict[str, str]:
    """Choose different-label donors with identical question/layout within split.

    The matching key includes task, prompt, question, legal answer set, coordinate
    frame, camera layout, and masks. Calibration observations can never donate to test rows.
    Every recipient must have a suitable donor; no unmatched fallback is used.
    """
    rows = validate_manifest(rows, require_split=True)
    buckets = defaultdict(list)

    def key(row: dict) -> tuple:
        masks = row.get("image_masks", {k: True for k in CAMERA_KEYS})
        layout = tuple((camera, masks[camera]) for camera in CAMERA_KEYS)
        return (row["split"], row["task_type"], row["prompt_id"], row["question"],
                tuple(sorted(row["labels"])), row.get("coordinate_frame"), layout)

    for row in sorted(rows, key=lambda r: r["sample_id"]):
        buckets[key(row)].append(row)
    donors = {}
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        candidates = [r for r in buckets[key(row)] if r["label"] != row["label"]]
        if not candidates:
            _fail(row, "no different-label donor with matching task/prompt/question/labels/coordinate frame/camera layout in this split; add matched observations")
        donor_seed = hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).digest()
        donors[row["sample_id"]] = random.Random(donor_seed).choice(candidates)["sample_id"]
    return donors


def load_observation(row: Mapping[str, Any]) -> dict:
    """Load uint8 RGB HWC images and measured state into the original policy API.

    Input images must already have the orientation used at policy inference.
    Only ``original_task_instruction`` supplies ``prompt``. Extra NPZ members,
    including saved prompts, labels, and questions, are ignored.
    """
    import numpy as np

    row = validate_manifest([row])[0]
    try:
        if "observation_path" in row:
            with np.load(row["observation_path"], allow_pickle=False) as archive:
                required = (*CAMERA_KEYS, "observation/state")
                missing = set(required) - set(archive.files)
                if missing:
                    _fail(row, f"NPZ missing required arrays: {sorted(missing)}")
                observation = {name: archive[name].copy() for name in required}
        else:
            observation = {}
            for name, filename in row["camera_paths"].items():
                if Path(filename).suffix.lower() == ".npy":
                    observation[name] = np.load(filename, allow_pickle=False)
                else:
                    from PIL import Image
                    with Image.open(filename) as im:
                        if im.mode != "RGB":
                            _fail(row, f"{name} image mode must be RGB, got {im.mode!r}")
                        observation[name] = np.array(im)
            observation["observation/state"] = (
                np.load(row["state_path"], allow_pickle=False)
                if "state_path" in row else np.asarray(row["state"], dtype=np.float32)
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, ManifestError):
            raise
        _fail(row, f"cannot load safe observation arrays: {exc}")
    for camera in CAMERA_KEYS:
        pixels = observation[camera]
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3 or min(pixels.shape[:2]) < 1:
            _fail(row, f"{camera} must be uint8 HWC RGB, got shape={pixels.shape}, dtype={pixels.dtype}")
    state = observation["observation/state"]
    if (state.ndim != 1 or state.size == 0 or state.dtype.kind not in "iuf"
            or not np.isfinite(state).all()):
        _fail(row, "observation/state must be a nonempty finite numeric vector")
    observation["prompt"] = row["original_task_instruction"]
    return observation
