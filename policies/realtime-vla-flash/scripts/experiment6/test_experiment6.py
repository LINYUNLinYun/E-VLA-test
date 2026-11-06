from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from failure_analysis import cohort
from layer_metrics import (
    action_intervention_metrics,
    hidden_metrics,
    make_modality_spans,
    rank_candidates,
)
from trajectory_analysis import (
    first_action_divergence,
    first_unreached_transition,
    gripper_events,
    nonself_target_distances,
    telemetry_summary,
)
from visualize_failures import select_balanced_cases


def test_hidden_metrics_preserve_layer18_kv_only_semantics():
    rng = np.random.default_rng(7)
    trace = rng.normal(size=(19, 6, 8)).astype(np.float32)
    trace[-1] = trace[-2]
    spans = make_modality_spans(token_count=6, num_views=1, tokens_per_view=4, view_names=("camera",))
    rows, pairwise = hidden_metrics(trace, spans)
    final = [row for row in rows if row["layer"] == 18]
    assert final and all(row["adjacent_cosine"] == 1.0 for row in final)
    assert all(row["relative_residual"] == 0.0 for row in final)
    assert not any(row["updates_hidden"] for row in final)
    assert pairwise["camera/cosine"].shape == (19, 19)


def test_action_intervention_detects_gripper_flip_and_ranks_only_layers_1_to_17():
    baseline = np.zeros((2, 7), dtype=np.float32)
    baseline[:, 6] = -1
    skips = np.stack([baseline.copy(), baseline.copy()])
    skips[0, :, 6] = 1
    skips[1, :, 0] = 0.1
    actions = action_intervention_metrics(baseline, skips, (1, 18))
    assert actions[0]["gripper_flip_rate"] == 1.0
    hidden = [
        {"layer": layer, "modality": "vision", "adjacent_cosine": 0.99, "adjacent_cka": 0.98, "relative_residual": 0.01}
        for layer in (1, 18)
    ]
    ranked = rank_candidates(hidden, actions)
    assert [row["layer"] for row in ranked] == [1]


def test_action_and_semantic_failure_helpers():
    control = [[0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, -1]]
    candidate = [[0, 0, 0, 0, 0, 0, -1], [0.3, 0, 0, 0, 0, 0, 1]]
    divergence = first_action_divergence(control, candidate)
    assert divergence["first_divergence_step"] == 1
    assert divergence["gripper_sign_mismatch"]
    assert gripper_events(candidate)[0]["event"] == "close"
    summary = telemetry_summary(
        [
            {"step": 0, "semantic_phase": "initial", "goal_progress": 0, "goal_predicate_count": 1},
            {"step": 4, "semantic_phase": "grasp_or_manipulate", "goal_progress": 0, "goal_predicate_count": 1},
            {"step": 9, "semantic_phase": "success", "goal_progress": 1, "goal_predicate_count": 1, "success": True},
        ]
    )
    assert summary["furthest_phase"] == "success"
    assert summary["max_goal_progress"] == 1
    assert cohort(True, False) == "new_failure"
    assert cohort(False, True) == "candidate_rescue"


def test_self_target_distance_is_not_goal_progress():
    distances = nonself_target_distances(
        {
            "bowl->drawer_region": 0.19,
            "drawer_region->drawer_region": 0.0,
        }
    )
    assert distances == {"bowl->drawer_region": 0.19}


def test_failed_transition_never_points_backward():
    transition = first_unreached_transition(
        "release_or_final_action",
        ["initial", "approach", "contact", "success"],
        ["initial", "approach", "release_or_final_action"],
    )
    assert transition == "release_or_final_action->success"


def test_gallery_selection_balances_conditions():
    rows = [
        {"condition_id": condition, "sample_id": f"{condition}-{index}", "candidate_furthest_phase": "approach"}
        for condition in ("V10", "V12", "V14", "V16")
        for index in range(4)
    ]
    selected = select_balanced_cases(rows, 8)
    assert {row["condition_id"] for row in selected} == {"V10", "V12", "V14", "V16"}
    assert Counter(row["condition_id"] for row in selected) == Counter({"V10": 2, "V12": 2, "V14": 2, "V16": 2})
