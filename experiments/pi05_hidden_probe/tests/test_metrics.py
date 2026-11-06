"""Metric contract tests independent of GPU/model dependencies."""

import json
import math

import pytest

from experiments.pi05_hidden_probe.metrics import (
    balanced_accuracy,
    cluster_ids,
    parse_answer,
    summarize,
)


def row(sample, label="red", prediction="red", **updates):
    record = dict(
        sample_id=str(sample), episode_id=f"episode-{sample}", scene_id=None,
        counterfactual_group_id=None, split="test", task_type="color",
        labels=["red", "blue"], label=label, prediction=prediction,
        probe_id="color_visual", prompt_id="question", selector="agent_image",
        question="What color?", source_layer=3, target_layer=1, condition="real",
        control_id="gray", config_hash="frozen", source_checkpoint="source",
        target_checkpoint="target", seed=7,
    )
    record.update(updates)
    return record


def test_parse_answer_requires_complete_answer_and_explicit_aliases():
    assert parse_answer('  "RED."\n', ["red", "blue"]) == "red"
    assert parse_answer("red or blue", ["red", "blue"]) is None
    assert parse_answer("The red object is on the left.", ["red", "blue"]) is None
    assert parse_answer("Answer: red", ["red", "blue"]) is None
    assert parse_answer("scarlet", ["red", "blue"]) is None
    assert parse_answer("scarlet", ["red", "blue"], {"scarlet": "red"}) == "red"
    assert parse_answer("crimson!", ["red", "blue"], {"red": ["scarlet", "crimson"]}) == "red"
    assert parse_answer(None, ["red", "blue"]) is None
    with pytest.raises(ValueError, match="ambiguous"):
        parse_answer("red", ["red", "blue"], {"RED": "blue"})


def test_bacc_counts_invalid_predictions_and_does_not_reduce_label_universe():
    rows = [row(0), row(1, prediction=None), row(2, label="blue", prediction="blue")]
    assert balanced_accuracy(rows) == 0.75  # red recall 1/2, blue recall 1
    rows[2]["prediction"] = "unknown"
    assert balanced_accuracy(rows) == 0.25
    assert math.isnan(balanced_accuracy(rows[:2]))
    metric = summarize(rows[:2], bootstrap_samples=0)[0]
    assert metric["missing_labels"] == ["blue"]
    assert metric["status"] == "missing_labels"
    assert math.isnan(metric["bacc"])
    with pytest.raises(ValueError, match="inconsistent"):
        balanced_accuracy([row(0), row(1, labels=["red", "green"])])


def test_pair_quad_and_donor_follow_are_separate_and_require_all_correct():
    rows = [
        row("p0", counterfactual_group_id="pair", donor_label="blue"),
        row("p1", "blue", "red", counterfactual_group_id="pair", donor_label="red"),
        row("q0", counterfactual_group_id="quad", donor_label="blue"),
        row("q1", "blue", "blue", counterfactual_group_id="quad", donor_label="red"),
        row("q2", counterfactual_group_id="quad", donor_label="blue"),
        row("q3", "blue", "blue", counterfactual_group_id="quad", donor_label="red"),
    ]
    metric = summarize(rows, bootstrap_samples=20)[0]
    assert metric["pair_n"] == 1
    assert metric["pair_acc"] == 0
    assert metric["quad_n"] == 1
    assert metric["quad_acc"] == 1
    assert metric["donor_eligible_n"] == 6
    assert metric["donor_follow_rate"] == 1 / 6
    assert metric["n_blocks"] == 2


def test_incomplete_quad_is_not_mislabeled_as_a_complete_pair():
    complete = [row(i, "red" if i % 2 == 0 else "blue", "red", counterfactual_group_id="quad") for i in range(4)]
    incomplete = [dict(item, condition="shuffle") for item in complete[:2]]
    metric = next(item for item in summarize(complete + incomplete, bootstrap_samples=0) if item["condition"] == "shuffle")
    assert metric["pair_n"] == 0
    assert metric["quad_n"] == 0
    assert metric["incomplete_counterfactual_groups"] == 1


def test_paired_controls_broadcast_null_layers_and_match_sample_ids():
    rows = []
    for layer in (0, 3):
        rows.extend([row("a", source_layer=layer), row("b", "blue", "blue", source_layer=layer)])
        rows.extend([
            row("a", prediction="blue", source_layer=layer, condition="shuffle", donor_label="blue"),
            row("b", "blue", "red", source_layer=layer, condition="shuffle", donor_label="red"),
        ])
    # Reverse order deliberately; pairing must use identity, not row order.
    rows.extend([row("b", "blue", "red", condition="nopatch", source_layer=None, target_layer=None),
                 row("a", condition="nopatch", source_layer=None, target_layer=None)])
    real = [item for item in summarize(rows, bootstrap_samples=100, seed=42) if item["condition"] == "real"]
    assert len(real) == 2
    for metric in real:
        assert metric["delta_nopatch"] == 0.5
        assert metric["delta_shuffle"] == 1
        assert metric["delta_nopatch_paired_n"] == 2
        assert metric["delta_nopatch_status"] == "ok"


def test_deltas_use_only_paired_samples_and_mark_missing_classes():
    real = [row("a"), row("b", "blue", "blue")]
    control = [row("a", condition="nopatch", source_layer=None)]
    metric = next(item for item in summarize(real + control, bootstrap_samples=0) if item["condition"] == "real")
    assert metric["delta_nopatch_paired_n"] == 1
    assert metric["delta_nopatch_missing_control_n"] == 1
    assert metric["delta_nopatch_status"] == "missing_labels"
    assert math.isnan(metric["delta_nopatch"])


def test_clustering_takes_transitive_closure_of_all_dependency_fields():
    rows = [
        row("a", episode_id="e1", scene_id="s1", counterfactual_group_id="c1"),
        row("b", episode_id="e2", scene_id="s1", counterfactual_group_id="c2"),
        row("c", episode_id="e2", scene_id="s3", counterfactual_group_id="c3"),
        row("d", episode_id="e4", scene_id="s4", counterfactual_group_id="c3"),
        row("separate"),
    ]
    blocks = cluster_ids(rows)
    assert len(set(blocks[:4])) == 1
    assert blocks[-1] != blocks[0]


def test_bootstrap_resamples_whole_episodes_not_individual_frames():
    rows = []
    # 200 frames but only two independent episodes. Every episode is balanced.
    for episode, correct in (("good", True), ("bad", False)):
        for i in range(100):
            label = "red" if i % 2 == 0 else "blue"
            prediction = label if correct else ("blue" if label == "red" else "red")
            rows.append(row(f"{episode}-{i}", label, prediction, episode_id=episode))
    metric = summarize(rows, bootstrap_samples=1000, seed=0)[0]
    assert metric["n_blocks"] == 2
    assert metric["bacc"] == 0.5
    assert metric["bacc_ci_low"] == 0
    assert metric["bacc_ci_high"] == 1
    assert metric["bacc_bootstrap_valid"] == 1000


def test_one_cluster_has_no_spurious_zero_width_interval():
    rows = [row(0, scene_id="one_scene"), row(1, "blue", "blue", scene_id="one_scene")]
    metric = summarize(rows, bootstrap_samples=100)[0]
    assert metric["n_blocks"] == 1
    assert math.isnan(metric["bacc_ci_low"])
    assert metric["bacc_bootstrap_valid"] == 0


def test_generation_and_candidate_scoring_stay_separate():
    rows = [row(0, prediction=None, score_prediction="red"),
            row(1, "blue", None, score_prediction="blue")]
    metric = summarize(rows, bootstrap_samples=0)[0]
    assert metric["bacc"] == 0
    assert metric["invalid_rate"] == 1
    assert metric["score_bacc"] == 1
    assert metric["score_invalid_rate"] == 0
    assert metric["score_n_available"] == 2


def test_disabled_candidate_scoring_is_not_reported_as_zero_accuracy():
    rows = [row(0, score_prediction=None, candidate_scores={}),
            row(1, "blue", "blue", score_prediction=None, candidate_scores={})]
    metric = summarize(rows, bootstrap_samples=0)[0]
    assert metric["score_n_available"] == 0
    assert "score_bacc" not in metric


def test_scored_delta_requires_control_scoring_to_exist():
    rows = [row(0, score_prediction="red"), row(1, "blue", "blue", score_prediction="blue"),
            row(0, condition="nopatch", source_layer=None),
            row(1, "blue", "blue", condition="nopatch", source_layer=None)]
    metric = next(item for item in summarize(rows, bootstrap_samples=0) if item["condition"] == "real")
    assert math.isnan(metric["score_delta_nopatch"])
    assert metric["score_delta_nopatch_status"] == "control_scores_unavailable"


def test_split_and_probe_ids_are_never_implicitly_pooled():
    rows = [row(0), row(1, split="calibration"), row(2, probe_id="other")]
    assert len(summarize(rows, bootstrap_samples=0)) == 3


def test_duplicates_are_rejected_instead_of_inflating_sample_size():
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        summarize([row(0), row(0)], bootstrap_samples=0)


def test_analysis_cli_writes_report_from_input_records(tmp_path):
    from experiments.pi05_hidden_probe.analyze_results import main

    records = [row(0, raw_generation="red"), row(1, "blue", "blue", raw_generation="blue")]
    input_path = tmp_path / "unit_test_records.jsonl"
    input_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    output_dir = tmp_path / "report"
    assert main(["--input", str(input_path), "--output-dir", str(output_dir), "--bootstrap-samples", "10"]) == 0
    assert (output_dir / "layer_metrics.csv").exists()
    assert "Observed rows: 2" in (output_dir / "summary.md").read_text()
    saved = [json.loads(line) for line in (output_dir / "sampled_generations.jsonl").read_text().splitlines()]
    assert saved == records
