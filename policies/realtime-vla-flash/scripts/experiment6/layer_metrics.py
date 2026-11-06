"""Pure NumPy metrics for VLM hidden-state and action intervention probes."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import numpy as np


NUM_VLM_LAYERS = 18
LAST_HIDDEN_UPDATE_LAYER = 17  # one-based; layer 18 only produces prefix K/V


@dataclasses.dataclass(frozen=True)
class TokenSpan:
    name: str
    start: int
    stop: int

    def validate(self, token_count: int) -> None:
        if not self.name:
            raise ValueError("Token span name must be non-empty")
        if not (0 <= self.start < self.stop <= token_count):
            raise ValueError(
                f"Invalid token span {self.name!r}: [{self.start}, {self.stop}) "
                f"for {token_count} tokens"
            )


def make_modality_spans(
    *,
    token_count: int,
    num_views: int,
    tokens_per_view: int = 256,
    view_names: Sequence[str] | None = None,
) -> tuple[TokenSpan, ...]:
    """Build non-overlapping per-view and language spans for a Pi0 prefix."""
    if num_views <= 0 or tokens_per_view <= 0:
        raise ValueError("num_views and tokens_per_view must be positive")
    vision_tokens = num_views * tokens_per_view
    if vision_tokens >= token_count:
        raise ValueError(
            f"Expected at least one language token, got {token_count=} and {vision_tokens=}"
        )
    names = tuple(view_names or (f"vision_view_{index}" for index in range(num_views)))
    if len(names) != num_views:
        raise ValueError(f"Expected {num_views} view names, got {len(names)}")
    spans = [
        TokenSpan(name, index * tokens_per_view, (index + 1) * tokens_per_view)
        for index, name in enumerate(names)
    ]
    spans.append(TokenSpan("language", vision_tokens, token_count))
    return tuple(spans)


def tokenwise_cosine(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"Expected matching [tokens, hidden] arrays, got {x.shape} and {y.shape}")
    numerator = np.sum(x * y, axis=-1)
    denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return float(np.mean(numerator / np.maximum(denominator, eps)))


def relative_residual(x_before: np.ndarray, x_after: np.ndarray, eps: float = 1e-12) -> float:
    x_before = np.asarray(x_before, dtype=np.float64)
    x_after = np.asarray(x_after, dtype=np.float64)
    if x_before.shape != x_after.shape:
        raise ValueError(f"Hidden arrays differ in shape: {x_before.shape} and {x_after.shape}")
    return float(np.linalg.norm(x_after - x_before) / max(np.linalg.norm(x_before), eps))


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Centered linear CKA, using token Gram matrices to avoid hidden^2 storage."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"Expected matching [tokens, hidden] arrays, got {x.shape} and {y.shape}")
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    x_gram = x @ x.T
    y_gram = y @ y.T
    numerator = float(np.sum(x_gram * y_gram))
    denominator = math.sqrt(float(np.sum(x_gram * x_gram)) * float(np.sum(y_gram * y_gram)))
    if denominator <= eps:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def validate_hidden_trace(hidden_trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(hidden_trace)
    if trace.ndim != 3:
        raise ValueError(f"hidden_trace must be [19, tokens, hidden], got {trace.shape}")
    if trace.shape[0] != NUM_VLM_LAYERS + 1:
        raise ValueError(f"Expected H0..H18 (19 states), got {trace.shape[0]}")
    if not np.allclose(trace[-1], trace[-2], rtol=0.0, atol=0.0):
        raise ValueError("H18 must equal H17: VLM layer 18 is prefix-K/V-only and does not update hidden")
    return trace


def hidden_metrics(
    hidden_trace: np.ndarray,
    spans: Sequence[TokenSpan],
) -> tuple[list[dict[str, float | int | str | bool]], dict[str, np.ndarray]]:
    """Compute adjacent metrics and all-pairs cosine/CKA matrices by modality."""
    trace = validate_hidden_trace(hidden_trace)
    token_count = trace.shape[1]
    rows: list[dict[str, float | int | str | bool]] = []
    pairwise: dict[str, np.ndarray] = {}
    for span in spans:
        span.validate(token_count)
        states = trace[:, span.start : span.stop]
        cosine_matrix = np.eye(NUM_VLM_LAYERS + 1, dtype=np.float64)
        cka_matrix = np.eye(NUM_VLM_LAYERS + 1, dtype=np.float64)
        for left in range(NUM_VLM_LAYERS + 1):
            for right in range(left + 1, NUM_VLM_LAYERS + 1):
                cosine = tokenwise_cosine(states[left], states[right])
                cka = linear_cka(states[left], states[right])
                cosine_matrix[left, right] = cosine_matrix[right, left] = cosine
                cka_matrix[left, right] = cka_matrix[right, left] = cka
        pairwise[f"{span.name}/cosine"] = cosine_matrix
        pairwise[f"{span.name}/cka"] = cka_matrix
        for layer in range(1, NUM_VLM_LAYERS + 1):
            rows.append(
                {
                    "layer": layer,
                    "modality": span.name,
                    "token_count": span.stop - span.start,
                    "adjacent_cosine": cosine_matrix[layer - 1, layer],
                    "adjacent_cka": cka_matrix[layer - 1, layer],
                    "relative_residual": relative_residual(states[layer - 1], states[layer]),
                    "updates_hidden": layer <= LAST_HIDDEN_UPDATE_LAYER,
                }
            )
    return rows, pairwise


def _select_dims(actions: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    dims = tuple(int(dim) for dim in dims)
    if not dims:
        return actions[..., :0]
    invalid = [dim for dim in dims if dim < 0 or dim >= actions.shape[-1]]
    if invalid:
        raise ValueError(f"Action dimensions out of range for dim={actions.shape[-1]}: {invalid}")
    return actions[..., dims]


def action_intervention_metrics(
    baseline_actions: np.ndarray,
    skip_actions: np.ndarray,
    skip_layer_ids: Sequence[int],
    *,
    translation_dims: Sequence[int] = (0, 1, 2),
    rotation_dims: Sequence[int] = (3, 4, 5),
    gripper_dim: int = 6,
    gripper_threshold: float = 0.0,
    eps: float = 1e-12,
) -> list[dict[str, float | int | bool]]:
    """Compare fixed-noise single-layer-skip actions with the full model."""
    baseline = np.asarray(baseline_actions, dtype=np.float64)
    skips = np.asarray(skip_actions, dtype=np.float64)
    if baseline.ndim < 2:
        raise ValueError(f"baseline_actions must be [..., action_dim], got {baseline.shape}")
    if skips.ndim != baseline.ndim + 1 or skips.shape[1:] != baseline.shape:
        raise ValueError(
            f"skip_actions must be [layers, *baseline.shape], got {skips.shape} vs {baseline.shape}"
        )
    if len(skip_layer_ids) != skips.shape[0]:
        raise ValueError("skip_layer_ids length does not match skip_actions")
    if not 0 <= gripper_dim < baseline.shape[-1]:
        raise ValueError(f"gripper_dim={gripper_dim} outside action dim {baseline.shape[-1]}")

    rows = []
    base_flat = baseline.reshape(-1)
    base_energy = float(np.mean(base_flat * base_flat))
    for index, layer in enumerate(skip_layer_ids):
        candidate = skips[index]
        delta = candidate - baseline
        candidate_flat = candidate.reshape(-1)
        cosine_denom = np.linalg.norm(base_flat) * np.linalg.norm(candidate_flat)
        action_cosine = float(np.dot(base_flat, candidate_flat) / max(cosine_denom, eps))
        base_gripper = baseline[..., gripper_dim] >= gripper_threshold
        candidate_gripper = candidate[..., gripper_dim] >= gripper_threshold
        rows.append(
            {
                "layer": int(layer),
                "action_nmse": float(np.mean(delta * delta) / max(base_energy, eps)),
                "action_cosine": action_cosine,
                "translation_rmse": float(np.sqrt(np.mean(_select_dims(delta, translation_dims) ** 2))),
                "rotation_rmse": float(np.sqrt(np.mean(_select_dims(delta, rotation_dims) ** 2))),
                "gripper_rmse": float(np.sqrt(np.mean(delta[..., gripper_dim] ** 2))),
                "gripper_flip_rate": float(np.mean(base_gripper != candidate_gripper)),
                "updates_hidden": int(layer) <= LAST_HIDDEN_UPDATE_LAYER,
            }
        )
    return rows


def _minmax(values: Mapping[int, float]) -> dict[int, float]:
    finite = {key: value for key, value in values.items() if np.isfinite(value)}
    if not finite:
        return {key: 0.0 for key in values}
    low, high = min(finite.values()), max(finite.values())
    if high - low <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def rank_candidates(
    hidden_rows: Sequence[Mapping[str, object]],
    action_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, float | int | str]]:
    """Rank layers 1..17; layer 18 is reported separately as K/V-only by design."""
    hidden_by_layer: dict[int, list[Mapping[str, object]]] = {}
    for row in hidden_rows:
        layer = int(row["layer"])
        if layer <= LAST_HIDDEN_UPDATE_LAYER:
            hidden_by_layer.setdefault(layer, []).append(row)
    action_by_layer = {int(row["layer"]): row for row in action_rows}
    layers = sorted(set(hidden_by_layer) & set(action_by_layer))
    if not layers:
        return []

    raw: dict[str, dict[int, float]] = {
        "cosine_distance": {},
        "cka_distance": {},
        "relative_residual": {},
        "action_nmse": {},
        "gripper_flip_rate": {},
    }
    for layer in layers:
        hrows = hidden_by_layer[layer]
        raw["cosine_distance"][layer] = float(np.mean([1.0 - float(row["adjacent_cosine"]) for row in hrows]))
        raw["cka_distance"][layer] = float(np.mean([1.0 - float(row["adjacent_cka"]) for row in hrows]))
        raw["relative_residual"][layer] = float(np.mean([float(row["relative_residual"]) for row in hrows]))
        raw["action_nmse"][layer] = float(action_by_layer[layer]["action_nmse"])
        raw["gripper_flip_rate"][layer] = float(action_by_layer[layer]["gripper_flip_rate"])

    normalized = {name: _minmax(values) for name, values in raw.items()}
    weights = {
        "cosine_distance": 0.10,
        "cka_distance": 0.10,
        "relative_residual": 0.20,
        "action_nmse": 0.40,
        "gripper_flip_rate": 0.20,
    }
    ranked = []
    for layer in layers:
        score = sum(weights[name] * normalized[name][layer] for name in weights)
        ranked.append(
            {
                "layer": layer,
                "pruning_risk_score": float(score),
                **{name: raw[name][layer] for name in raw},
                "interpretation": "lower_is_better_candidate",
            }
        )
    ranked.sort(key=lambda row: (float(row["pruning_risk_score"]), int(row["layer"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked

