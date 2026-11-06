import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_multitask_vlm_ablation as runner
import summarize_cross_task_ablation as summarizer


def test_config_has_exact_paired_layer_groups():
    config = json.loads((HERE / "experiment5_config.json").read_text(encoding="utf-8"))
    expected = {
        "V12": [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18],
        "V14": [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18],
        "V16": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18],
        "V10-Mid": [1, 2, 3, 4, 9, 10, 15, 16, 17, 18],
        "Control": list(range(1, 19)),
    }
    assert {item["condition_id"]: item["vlm_layer_indices_1based"] for item in config["conditions"]} == expected
    assert config["action_layer_pattern"] == "full18"
    assert config["denoise_steps"] == 3


def test_input_manifest_requires_90_plus_10_distinct_tasks(tmp_path):
    config_path = HERE / "experiment5_config.json"
    testset_path = tmp_path / "testset.json"
    samples = []
    for suite, count in (("libero_90", 90), ("libero_10", 10)):
        for task_id in range(count):
            samples.append({"sample_id": f"{suite}:{task_id}", "suite_name": suite, "task_id": task_id, "init_state": [0.0]})
    testset_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    _, _, normalized, _, _ = runner._read_inputs(config_path, testset_path)
    assert len(normalized) == 100
    assert normalized[97]["sample_index"] == 97


def test_rejects_missing_task_state(tmp_path):
    config_path = HERE / "experiment5_config.json"
    testset_path = tmp_path / "testset.json"
    samples = []
    for suite, count in (("libero_90", 90), ("libero_10", 10)):
        for task_id in range(count):
            item = {"sample_id": f"{suite}:{task_id}", "suite_name": suite, "task_id": task_id, "init_state": [0.0]}
            if task_id == 0 and suite == "libero_90":
                del item["init_state"]
            samples.append(item)
    testset_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks an initial state"):
        runner._read_inputs(config_path, testset_path)


def test_mcnemar_exact_p_is_bounded_and_symmetric():
    assert summarizer._exact_mcnemar_p(0, 0) == 1.0
    assert summarizer._exact_mcnemar_p(4, 0) == summarizer._exact_mcnemar_p(0, 4)
    assert 0.0 <= summarizer._exact_mcnemar_p(4, 0) <= 1.0
