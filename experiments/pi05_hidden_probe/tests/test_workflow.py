"""Integration checks for inference provenance and calibration/test separation."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from experiments.pi05_hidden_probe.backend import ForwardResult, Prefix
from experiments.pi05_hidden_probe.common import (
    FeatureCache, config_hash, control_observation, digest, load_config, observation_hash, read_jsonl,
)
from experiments.pi05_hidden_probe.data import load_observation, validate_manifest
from experiments.pi05_hidden_probe.metrics import summarize
from experiments.pi05_hidden_probe.workflow import freeze_config, run_probe


class TinyBackend:
    """A deterministic fixture, not a semantic model or experiment result."""
    def __init__(self):
        self.prepares = 0
        self.generations = 0

    def audit(self):
        return {"depth": 2}

    def prepare_source(self, obs):
        self.prepares += 1
        x = np.zeros((3, 4), np.float32)
        x[0] = obs["observation/image"].mean()
        x[1] = obs["observation/wrist_image"].mean()
        return Prefix(x, np.ones(3, bool), {"base_0_rgb": [0], "left_wrist_0_rgb": [1]}, [3], [2], task_text=obs["prompt"])

    def prepare_target(self, obs, prompt):
        return self.prepare_source(obs)

    def forward(self, prefix, capture_layers=(), capture_positions=None, patch=None):
        values = prefix.embedded.copy()
        if patch is not None:
            values[patch.positions] = patch.values
        selected = values if capture_positions is None else values[capture_positions]
        return ForwardResult({l: selected.copy() for l in capture_layers}, np.array([0., 1.]), values)

    def generate(self, prefix, patch=None, max_new_tokens=8):
        self.generations += 1
        value = self.forward(prefix, patch=patch).normalized[0, 0]
        return {"text": "red" if value > 150 else "blue", "token_ids": [3], "eos": True}

    def score_answers(self, prefix, answers, patch=None, length_normalize=False):
        value = self.forward(prefix, patch=patch).normalized[0, 0]
        answer = "red" if value > 150 else "blue"
        return {a: 0. if a == answer else -2. for a in answers}


def fixture(tmp_path):
    rows = []
    for index, (label, value) in enumerate([("red", 240), ("blue", 10)]):
        path = tmp_path / f"obs{index}.npz"
        np.savez(path, **{"observation/image": np.full((4, 4, 3), value, np.uint8),
                         "observation/wrist_image": np.zeros((4, 4, 3), np.uint8),
                         "observation/state": np.zeros(8, np.float32)})
        rows.append({"sample_id": str(index), "episode_id": "ep", "scene_id": "scene",
                     "counterfactual_group_id": "pair", "original_task_instruction": "pick cup",
                     "task_type": "visual_attribute", "label": label, "labels": ["red", "blue"],
                     "label_source": "unit_test_fixture", "question": "What color is cup?\n",
                     "prompt_id": "color", "is_synthetic": True, "image_convention": "policy_input",
                     "observation_path": str(path), "split": "calibration"})
    rows = validate_manifest(rows)
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    config = {"stage": "calibration", "seed": 42, "source_layers": [0, 1],
              "model": {"checkpoint": "fixture"}, "max_new_tokens": 8,
              "score_candidates": True, "length_normalize": False, "aliases": {},
              "output_dir": str(tmp_path / "calibration"), "manifest": str(tmp_path / "manifest.jsonl"),
              "controls": [{"id": "gray", "kind": "constant", "value": 127}],
              "conditions": ["real", "nopatch", "self", "shuffle", "random", "direct", "layer0"],
              "probes": [{"id": "visual", "task_type": "visual_attribute", "source_selector": "image:base_0_rgb",
                          "target_selector": "image:base_0_rgb", "target_layers": [1], "prompt": None}]}
    run = {"config_hash": config_hash(config), "source_identity": {"audit": {"depth": 2}},
           "source_input_digests": {r["sample_id"]: observation_hash(load_observation(r)) for r in rows}}
    return rows, config, run


def test_control_has_no_source_information():
    obs = {"observation/image": np.ones((2, 3, 3), np.uint8),
           "observation/wrist_image": np.ones((2, 3, 3), np.uint8),
           "observation/state": np.ones(8), "prompt": "the secret answer is red"}
    control = control_observation(obs, {"kind": "constant", "value": 127})
    assert control["prompt"] == ""
    assert not control["observation/state"].any()
    assert (control["observation/image"] == 127).all()
    assert obs["prompt"] == "the secret answer is red"


def test_cache_reuses_only_identical_source_input(tmp_path):
    rows, _, _ = fixture(tmp_path)
    backend = TinyBackend()
    obs = load_observation(rows[0])
    cache = FeatureCache(tmp_path / "cache", {"checkpoint": "fixture"})
    first, positions, key = cache.get(backend, obs, "images", [0, 1])
    again, _, same = cache.get(backend, obs, "images", [0, 1])
    assert backend.prepares == 1 and same == key
    np.testing.assert_array_equal(first[1], again[1])
    _, _, changed = cache.get(backend, {**obs, "prompt": "pick bowl"}, "images", [0, 1])
    assert changed != key and backend.prepares == 2


def test_run_and_resume_preserve_conditions_and_donor_labels(tmp_path):
    rows, config, run = fixture(tmp_path)
    backend = TinyBackend()
    out = tmp_path / "out"
    out.mkdir()
    path = run_probe(config, rows, backend, backend, out, run)
    records = read_jsonl(path)
    assert len(records) == 2 * (4 + 2 * 3)
    assert all(r["prediction"] == r["label"] for r in records if r["condition"] == "real")
    assert all(r["prediction"] == r["donor_label"] != r["label"] for r in records if r["condition"] == "shuffle")
    count = backend.generations
    run_probe(config, rows, backend, backend, out, run, resume=True)
    assert backend.generations == count
    assert read_jsonl(path) == records
    metrics = summarize(records, bootstrap_samples=5)
    main = [m for m in metrics if m["condition"] == "real"]
    assert all(m["bacc"] == 1 and m["delta_shuffle"] == 1 for m in main)


def test_resume_rejects_changed_input_content(tmp_path):
    rows, config, run = fixture(tmp_path)
    backend = TinyBackend()
    out = tmp_path / "out"
    out.mkdir()
    run_probe(config, rows, backend, backend, out, run)
    obs = load_observation(rows[0])
    obs["observation/image"][:] = 55
    np.savez(rows[0]["observation_path"], **{k: v for k, v in obs.items() if k != "prompt"})
    with pytest.raises(ValueError, match="Observation changed"):
        run_probe(config, rows, backend, backend, out, run, resume=True)


@pytest.mark.parametrize("source_layers", [[0, 1], "all"])
def test_freeze_uses_calibration_and_detects_later_edit(tmp_path, source_layers):
    rows, config, _ = fixture(tmp_path)
    config["probes"][0]["target_layers"] = [0, 1]
    config["source_layers"] = source_layers
    config["conditions"] = ["real", "nopatch", "shuffle"]
    config_path = tmp_path / "calibration.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config = load_config(config_path)
    records = [{"probe_id": "visual", "condition": condition, "split": "calibration",
                "sample_id": row["sample_id"], "label": row["label"], "labels": ["red", "blue"],
                "prediction": row["label"] if q == 1 else None, "target_layer": q,
                "source_layer": l, "control_id": "gray", "config_hash": config_hash(config)}
               for q in [0, 1] for condition in config["conditions"]
               for l in ([None] if condition == "nopatch" else (list(range(3)) if source_layers == "all" else source_layers)) for row in rows]
    identity = {"config_hash": config_hash(config), "sample_ids": [r["sample_id"] for r in rows],
                "manifest_digest": digest(rows), "source_identity": {"audit": {"depth": 2}}}
    (tmp_path / "run_identity.json").write_text(json.dumps(identity))
    inputs = tmp_path / "results.jsonl"
    inputs.write_text("\n".join(json.dumps(r) for r in records))
    output = tmp_path / "frozen.yaml"
    frozen = freeze_config(config, records, output, input_path=inputs)
    assert load_config(output)["probes"][0]["target_layers"] == [1]
    assert frozen["source_layers"] == "all"
    with pytest.raises(ValueError, match="incomplete"):
        freeze_config(config, records[:-1], output, input_path=inputs)
    with pytest.raises(ValueError, match="duplicate"):
        freeze_config(config, records + records[:1], output, input_path=inputs)
    frozen["probes"][0]["target_layers"] = [0]
    output.write_text(yaml.safe_dump(frozen))
    with pytest.raises(ValueError, match="intact frozen"):
        load_config(output)
    records[0]["split"] = "test"
    with pytest.raises(ValueError, match="non-calibration"):
        freeze_config(config, records, output, input_path=inputs)


def test_freeze_refuses_unmeasured_candidate_scores(tmp_path):
    rows, config, run = fixture(tmp_path)
    config["selection_prediction"] = "score_prediction"
    config["score_candidates"] = False
    config_path = tmp_path / "calibration.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config = load_config(config_path)
    run.update(config_hash=config_hash(config), manifest_digest=digest(rows),
               sample_ids=[r["sample_id"] for r in rows])
    out = tmp_path / "out"
    out.mkdir()
    backend = TinyBackend()
    results = run_probe(config, rows, backend, backend, out, run)
    with pytest.raises(ValueError, match="candidate scoring was not completed"):
        freeze_config(config, read_jsonl(results), tmp_path / "frozen.yaml", input_path=results)


def test_config_paths_stay_in_project():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    assert Path(config["model"]["checkpoint"]).is_dir()
    assert Path(config["model"]["openpi_root"]).is_dir()
    assert "/benchmark/results/" in config["output_dir"]
