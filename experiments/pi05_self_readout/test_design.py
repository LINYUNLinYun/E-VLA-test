"""Focused CPU checks of donor isolation and the finite diagnostic matrix."""
import numpy as np

from .run import PREVIOUS, color_pairs, expected_keys, matched_random, robot_prompt


def test_donors_are_opposite_colors_in_complete_calibration_pairs():
    pairs = color_pairs(PREVIOUS / "counterfactual_data_v2/manifest.jsonl", 12)
    assert len(pairs) == 12
    for pair in pairs:
        assert {r["label"] for r in pair} == {"red", "blue"}
        assert {r["split"] for r in pair} == {"calibration"}
        assert len({r["group_id"] for r in pair}) == 1
        assert len({r["original_task_instruction"] for r in pair}) == 1


def test_matrix_has_no_baseline_duplication_and_preserves_pair_members():
    pairs = color_pairs(PREVIOUS / "counterfactual_data_v2/manifest.jsonl", 2)
    rows = [r for pair in pairs for r in pair]
    keys = expected_keys(rows, [0, 3, 6, 9, 12, 15])
    assert len(keys) == 4 * (3 + 2 * 2 * 6 * 2 * 3)
    for row in rows:
        assert sum(k[0] == row["sample_id"] for k in keys) == 147


def test_random_control_is_deterministic_and_matches_each_token_norm():
    values = np.random.default_rng(4).normal(size=(5, 9)).astype(np.float32)
    a = matched_random(values, [42, "sample", "same_question", "images", 3])
    b = matched_random(values, [42, "sample", "same_question", "images", 3])
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=-1), np.linalg.norm(values, axis=-1), rtol=1e-6)
    assert not np.allclose(a, values)


def test_robot_context_keeps_original_instruction_and_newline_separator():
    assert robot_prompt({"original_task_instruction": " pick_up\nthe bowl "}) == "pick up the bowl\n"
