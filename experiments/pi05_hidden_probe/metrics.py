"""Conservative answer parsing and cluster-aware probe metrics.

An absent class makes balanced accuracy undefined (NaN), never a smaller-class
macro average. Bootstrap draws also have to contain every declared class; the
number of valid draws is reported. Calibration and test splits stay separate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math
import unicodedata
from typing import Any

import numpy as np


GROUP_FIELDS = (
    "split", "task_type", "probe_id", "prompt_id", "selector", "source_layer",
    "target_layer", "condition", "control_id", "config_hash",
    "source_checkpoint", "target_checkpoint", "seed",
)
_BLOCK_FIELDS = ("group_id", "episode_id", "scene_id", "counterfactual_group_id")
_EDGE_PUNCTUATION = " \t\r\n.,!?;:\"'`“”‘’()[]{}"


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().strip(_EDGE_PUNCTUATION).split())


def parse_answer(
    text: str | None,
    labels: Sequence[str],
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> str | None:
    """Accept a complete short label, allowing only case/edge punctuation changes.

    Aliases can be ``{alias: canonical}`` or ``{canonical: [alias, ...]}``.
    Explanations, answer prefixes and multiple competing labels are invalid.
    """
    lookup: dict[str, str] = {}

    def add(alias: str, label: str) -> None:
        if label not in labels:
            raise ValueError(f"Alias points to undeclared label: {label!r}")
        normalized = _normalize(alias)
        if not normalized or (normalized in lookup and lookup[normalized] != label):
            raise ValueError(f"Empty or ambiguous normalized alias: {alias!r}")
        lookup[normalized] = label

    for label in labels:
        add(label, label)
    for alias, canonical in (aliases or {}).items():
        if isinstance(canonical, str):
            add(alias, canonical)
        else:
            for name in canonical:
                add(name, alias)
    return lookup.get(_normalize(text)) if isinstance(text, str) else None


def _label_set(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if not rows:
        return ()
    declared = rows[0].get("labels")
    if not isinstance(declared, (list, tuple)) or not declared:
        raise ValueError("Each result must declare a nonempty labels list")
    if any(not isinstance(label, str) or not label for label in declared):
        raise ValueError("labels must contain nonempty strings")
    labels = tuple(sorted(declared))
    if len(set(labels)) != len(labels):
        raise ValueError("labels must not contain duplicates")
    for row in rows:
        row_labels = row.get("labels")
        if not isinstance(row_labels, (list, tuple)) or tuple(sorted(row_labels)) != labels:
            raise ValueError("Rows in one metric group declare inconsistent label sets")
        if row.get("label") not in labels:
            raise ValueError(f"Ground-truth label is not declared: {row.get('label')!r}")
    return labels


def balanced_accuracy(rows: Iterable[Mapping[str, Any]], prediction_key: str = "prediction") -> float:
    """Macro recall over all declared classes; invalid predictions count as wrong.

    Return NaN if the sample is empty or any declared class has no observations.
    ``summarize`` also reports the missing classes and a status field.
    """
    rows = list(rows)
    labels = _label_set(rows)
    recalls = []
    for label in labels:
        members = [row for row in rows if row["label"] == label]
        if not members:
            return math.nan
        recalls.append(sum(row.get(prediction_key) == label for row in members) / len(members))
    return float(np.mean(recalls)) if recalls else math.nan


def cluster_ids(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return connected components of sample/episode/scene/counterfactual links.

    Taking only one of those fields is insufficient: a counterfactual can bridge
    two episodes, and a shared scene can bridge multiple counterfactual groups.
    Missing metadata yields singleton sample blocks, counted in the report.
    """
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for field in ("sample_id", *_BLOCK_FIELDS):
            value = row.get(field)
            if value is None or value == "":
                continue
            key = (field, str(value))
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    identifiers: dict[int, int] = {}
    return [identifiers.setdefault(root(i), len(identifiers)) for i in range(len(rows))]


def _group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    key = tuple(row.get(field) for field in GROUP_FIELDS)
    try:
        hash(key)
    except TypeError as exc:
        raise ValueError("Metric grouping metadata must be scalar values") from exc
    return key


def _cf_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    identifier = row.get("counterfactual_group_id")
    if identifier is None or identifier == "":
        return None
    return row.get("split"), row.get("task_type"), identifier


def _bootstrap_bacc(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[str], prediction_key: str,
    draws: np.ndarray, block_indexes: np.ndarray,
) -> np.ndarray:
    """Compute bootstrap recalls from block sufficient statistics, not frame draws."""
    correct = np.array([row.get(prediction_key) == row["label"] for row in rows], dtype=float)
    recalls = []
    for label in labels:
        selected = np.array([row["label"] == label for row in rows], dtype=float)
        counts = np.bincount(block_indexes, weights=selected, minlength=draws.shape[1])
        successes = np.bincount(block_indexes, weights=selected * correct, minlength=draws.shape[1])
        denominators = draws @ counts
        numerators = draws @ successes
        recalls.append(np.divide(numerators, denominators, out=np.full(len(draws), np.nan), where=denominators > 0))
    return np.mean(recalls, axis=0) if recalls else np.full(len(draws), np.nan)


def _draws(
    rows: Sequence[Mapping[str, Any]], block_by_sample: Mapping[str, int],
    samples: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    local: dict[int, int] = {}
    indexes = np.array([local.setdefault(block_by_sample[str(row["sample_id"])], len(local)) for row in rows], dtype=int)
    n_blocks = len(local)
    # One cluster cannot support an uncertainty estimate.
    draws = (rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=samples)
             if n_blocks >= 2 and samples else np.empty((0, n_blocks), dtype=int))
    return draws, indexes, n_blocks


def _ci(values: np.ndarray) -> dict[str, Any]:
    valid = values[np.isfinite(values)]
    bounds = np.quantile(valid, [0.025, 0.975]) if len(valid) >= 2 else [math.nan, math.nan]
    return {"ci_low": float(bounds[0]), "ci_high": float(bounds[1]), "bootstrap_valid": len(valid)}


def _measure(
    rows: Sequence[Mapping[str, Any]], prediction_key: str,
    block_by_sample: Mapping[str, int], expected_cf: Mapping[tuple[Any, ...], set[str]],
    bootstrap_samples: int, rng: np.random.Generator,
) -> dict[str, Any]:
    labels = _label_set(rows)
    observed = {row["label"] for row in rows}
    missing = [label for label in labels if label not in observed]
    draws, indexes, n_blocks = _draws(rows, block_by_sample, bootstrap_samples, rng)
    result: dict[str, Any] = {
        "bacc": balanced_accuracy(rows, prediction_key), "n": len(rows),
        "labels": list(labels), "missing_labels": missing,
        "status": "missing_labels" if missing else "ok",
        "invalid_rate": sum(row.get(prediction_key) not in labels for row in rows) / len(rows),
        "n_blocks": n_blocks,
        "ungrouped_samples": sum(not any(row.get(field) not in (None, "") for field in _BLOCK_FIELDS) for row in rows),
    }
    result.update({f"bacc_{key}": value for key, value in _ci(_bootstrap_bacc(rows, labels, prediction_key, draws, indexes)).items()})
    cf_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _cf_key(row)
        if key is not None:
            cf_groups[key].append(row)
    result["incomplete_counterfactual_groups"] = sum(
        {str(row["sample_id"]) for row in group} != expected_cf[key]
        for key, group in cf_groups.items()
    )
    for size, name in ((2, "pair"), (4, "quad")):
        complete = [group for key, group in cf_groups.items()
                    if len(expected_cf[key]) == size and len(group) == size
                    and {str(row["sample_id"]) for row in group} == expected_cf[key]]
        result[f"{name}_n"] = len(complete)
        result[f"{name}_acc"] = (float(np.mean([all(row.get(prediction_key) == row["label"] for row in group) for group in complete]))
                                 if complete else math.nan)
    donors = [row for row in rows if row.get("donor_label") in labels and row["donor_label"] != row["label"]]
    result["donor_eligible_n"] = len(donors)
    result["donor_follow_rate"] = (sum(row.get(prediction_key) == row["donor_label"] for row in donors) / len(donors)
                                   if donors else math.nan)
    return result


def _find_control(
    metadata: Mapping[str, Any], condition: str,
    groups: Mapping[tuple[Any, ...], list[Mapping[str, Any]]],
) -> list[Mapping[str, Any]] | None:
    candidates = []
    for key, rows in groups.items():
        other = dict(zip(GROUP_FIELDS, key))
        if other["condition"] != condition:
            continue
        if any(other[field] != metadata[field] for field in GROUP_FIELDS if field not in ("condition", "source_layer", "target_layer")):
            continue
        if any(other[field] is not None and other[field] != metadata[field] for field in ("source_layer", "target_layer")):
            continue
        specificity = sum(other[field] == metadata[field] for field in ("source_layer", "target_layer"))
        candidates.append((specificity, rows))
    if not candidates:
        return None
    highest = max(item[0] for item in candidates)
    best = [rows for score, rows in candidates if score == highest]
    if len(best) != 1:
        raise ValueError(f"Ambiguous {condition} baselines for {metadata}")
    return best[0]


def _paired_delta(
    real: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]] | None,
    prediction_key: str, block_by_sample: Mapping[str, int],
    bootstrap_samples: int, rng: np.random.Generator,
) -> dict[str, Any]:
    empty = {"value": math.nan, "ci_low": math.nan, "ci_high": math.nan, "bootstrap_valid": 0,
             "paired_n": 0, "paired_n_blocks": 0, "missing_real_n": 0,
             "missing_control_n": len(real), "status": "missing_control"}
    if control is None:
        return empty
    real_map = {str(row["sample_id"]): row for row in real}
    control_map = {str(row["sample_id"]): row for row in control}
    common = sorted(real_map.keys() & control_map.keys())
    real_rows = [real_map[key] for key in common]
    control_rows = [control_map[key] for key in common]
    empty.update(paired_n=len(common), missing_real_n=len(control_map.keys() - real_map.keys()),
                 missing_control_n=len(real_map.keys() - control_map.keys()))
    if not common:
        empty["status"] = "no_paired_samples"
        return empty
    labels = _label_set(real_rows)
    if _label_set(control_rows) != labels or any(left["label"] != right["label"] for left, right in zip(real_rows, control_rows)):
        raise ValueError("Paired real/control rows have incompatible labels")
    point = balanced_accuracy(real_rows, prediction_key) - balanced_accuracy(control_rows, prediction_key)
    draws, indexes, n_blocks = _draws(real_rows, block_by_sample, bootstrap_samples, rng)
    deltas = (_bootstrap_bacc(real_rows, labels, prediction_key, draws, indexes)
              - _bootstrap_bacc(control_rows, labels, prediction_key, draws, indexes))
    empty.update(value=point, paired_n_blocks=n_blocks,
                 status="ok" if math.isfinite(point) else "missing_labels", **_ci(deltas))
    return empty


def summarize(
    rows: Iterable[Mapping[str, Any]], bootstrap_samples: int = 1000, seed: int = 0,
) -> list[dict[str, Any]]:
    """Summarize generated answers and optional candidate scoring independently.

    No-patch controls with null source/target layers broadcast over those axes.
    Every other grouping field must match. Deltas use the intersection of sample
    IDs with one shared block resample for the real and control predictions.
    """
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    rows = [dict(row) for row in rows]
    if not rows:
        return []
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    expected_cf: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    block_by_sample: dict[str, int] = {}
    for row, block in zip(rows, cluster_ids(rows)):
        if row.get("sample_id") in (None, ""):
            raise ValueError("Every result needs a nonempty sample_id")
        block_by_sample[str(row["sample_id"])] = block
        groups[_group_key(row)].append(row)
        cf_key = _cf_key(row)
        if cf_key is not None:
            expected_cf[cf_key].add(str(row["sample_id"]))
    for group in groups.values():
        ids = [str(row["sample_id"]) for row in group]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate sample_id in a metric group; separate repeated runs using seed/config_hash")
    rng = np.random.default_rng(seed)
    results = []
    for key in sorted(groups, key=repr):
        group = groups[key]
        metadata = dict(zip(GROUP_FIELDS, key))
        result = dict(metadata)
        result.update(_measure(group, "prediction", block_by_sample, expected_cf, bootstrap_samples, rng))
        result["bootstrap_samples"] = bootstrap_samples
        result["bootstrap_seed"] = seed
        for prediction_key, prefix in (("prediction", ""), ("score_prediction", "score_")):
            if prediction_key == "score_prediction":
                present = sum(row.get(prediction_key) is not None or bool(row.get("candidate_scores")) for row in group)
                result["score_n_available"] = present
                if not present:
                    continue
                # Missing scored rows count as invalid; report partial availability.
                scored = _measure(group, prediction_key, block_by_sample, expected_cf, bootstrap_samples, rng)
                result.update({prefix + field: value for field, value in scored.items()})
            if metadata["condition"] != "real":
                continue
            for condition in ("nopatch", "shuffle"):
                control = _find_control(metadata, condition, groups)
                delta = _paired_delta(group, control, prediction_key, block_by_sample, bootstrap_samples, rng)
                if prediction_key == "score_prediction" and control is not None and not any(
                    row.get(prediction_key) is not None or bool(row.get("candidate_scores")) for row in control
                ):
                    delta.update(value=math.nan, ci_low=math.nan, ci_high=math.nan,
                                 bootstrap_valid=0, status="control_scores_unavailable")
                for field, value in delta.items():
                    name = f"{prefix}delta_{condition}" + ("" if field == "value" else f"_{field}")
                    result[name] = value
        results.append(result)
    return results
