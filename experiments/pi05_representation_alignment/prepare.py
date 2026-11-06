"""Freeze the corrected color experiment and verify its same-model gate."""
import json
from pathlib import Path

import numpy as np

from experiments.pi05_hidden_probe.common import digest, observation_hash, write_json
from experiments.pi05_hidden_probe.data import load_observation, read_manifest
from experiments.pi05_self_readout.run import aggregate, expected_keys, key, sha

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "results/pi05_hidden_probe/run_20261105"
SELF = ROOT / "results/pi05_hidden_probe/self_readout_20261105"
OUT = ROOT / "results/pi05_hidden_probe/alignment_20261106"
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1., 10.]


def verify_self():
    plan = json.loads((SELF / "plan.json").read_text())
    complete = json.loads((SELF / "complete.json").read_text())
    rows = [json.loads(line) for line in (SELF / "records.jsonl").read_text().splitlines() if line.strip()]
    expected = expected_keys(plan["samples"], plan["layers"])
    if (len(rows) != len(expected) or len({key(r) for r in rows}) != len(rows)
            or {key(r) for r in rows} != expected
            or any(r["identity"] != plan["identity"] for r in rows)
            or sha(SELF / "records.jsonl") != complete["records_sha256"]):
        raise ValueError("Same-model readout coverage/provenance verification failed")
    metrics = aggregate(rows)
    if metrics != json.loads((SELF / "metrics.json").read_text()):
        raise ValueError("Same-model metrics do not match raw records")
    gate = [r for r in metrics if r["layer"] == 3 and r["condition"] == "real"]
    if len(gate) != 8 or any(r["candidate_bacc"] != 1. or r["generation_bacc"] != 1. for r in gate):
        raise ValueError("The predeclared layer-3 same-model gate did not pass")
    integrity = json.loads((SELF / "integrity.json").read_text())
    if not integrity["passed"]:
        raise ValueError("Same-model numerical integrity did not pass")
    review = {"matrix_verified": True, "records": len(rows), "images": len(plan["samples"]),
              "scene_groups": len({r["group_id"] for r in plan["samples"]}),
              "source_target_layer": 3, "readout_gate_passed": True,
              "records_sha256": complete["records_sha256"], "precision": plan["dtype"],
              "scope": "previous calibration data only; this does not establish cross-model alignment"}
    write_json(SELF / "review.json", review)
    complete["interpretation_review_required"] = False
    write_json(SELF / "complete.json", complete)
    return review


def main():
    review = verify_self()
    manifest = PREVIOUS / "counterfactual_data_v2/manifest.jsonl"
    rows = [r for r in read_manifest(manifest) if r["split"] == "calibration"]
    groups = sorted({r["group_id"] for r in rows})
    if len(groups) != 12:
        raise ValueError("Expected the original 12 calibration scene groups")
    order = np.random.default_rng(42).permutation(len(groups))
    train = {groups[int(i)] for i in order[:8]}
    validation = set(groups) - train
    index = json.loads((PREVIOUS / "source_extraction_calibration/source_index.json").read_text())
    by_sample = {item["sample_id"]: item for item in index["features"] if item["probe_id"].endswith("_image")}
    paired, seen, probes = [], set(), []
    for row in rows:
        item = by_sample[row["sample_id"]]
        path = PREVIOUS / "external_source_cache" / (item["cache_key"] + ".npz")
        obs_hash = observation_hash(load_observation(row))
        partition = "train" if row["group_id"] in train else "validation"
        metadata = {"sample_id": row["sample_id"], "group_id": row["group_id"],
                    "partition": partition, "observation_path": row["observation_path"],
                    "observation_sha256": sha(row["observation_path"]),
                    "original_task_instruction": row["original_task_instruction"],
                    "observation_hash": obs_hash, "source_cache": str(path), "origin": row["origin"]}
        with np.load(path, allow_pickle=False) as data:
            cache_meta = json.loads(str(data["metadata"].item()))
            if cache_meta["spec"]["observation"] != obs_hash or "layer_3" not in data:
                raise ValueError(f"Source cache mismatch: {path}")
            metadata["source_offsets"] = [cache_meta["positions"].index(p) for p in range(256)]
        if obs_hash not in seen:
            seen.add(obs_hash)
            # Alignment inputs omit answer labels and readout questions.
            paired.append(metadata)
        if row["task_type"] == "visual_attribute":
            probes.append({**metadata, "label": row["label"], "labels": row["labels"],
                           "counterfactual_group_id": row["counterfactual_group_id"]})
    plan = {"experiment": "label-free pi05_libero -> PaliGemma alignment; color validation first",
            "source_checkpoint": str(ROOT / "shared/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero"),
            "target_checkpoint": str(ROOT / "shared/checkpoints/pi05_hidden_probe/paligemma-3b-mix-224"),
            "source_layer": 3, "target_layer": 3, "selector": "image:base_0_rgb", "tokens": 256,
            "model_precision": "bfloat16", "model_weights": "both frozen",
            "paired_prompt": "cleaned original robot instruction + newline; same in both models",
            "train_groups": sorted(train), "validation_groups": sorted(validation),
            "ridge_candidates": ALPHAS, "ridge_selection": "validation paired-state MSE only; no answer labels",
            "source_pairs": paired, "supervised_probe_samples": probes,
            "probe": "mean-pool 256 Source tokens; separate binary ridge classifier; validation chooses regularization",
            "fresh_test": {"suite": "libero_goal", "task": 8, "init_offset": 40, "scenes": 10,
                           "seed": 20261106, "forbidden_origins_manifest": str(manifest),
                           "requirement": "reject previously used suite/task/init; no fitting/selection on new test answers"},
            "evaluation_routes": ["identity", "mean_scale", "affine", "target_own"],
            "evaluation_controls": {"black": 0, "gray": 127},
            "evaluation_conditions": ["real", "shuffle", "random"], "max_new_tokens": 8,
            "report": "candidate BAcc and short-answer generation separately; pairs and donor following",
            "self_readout_gate": review, "old_test_policy": "previous test data is not used for fitting or validation",
            "limitations": ["color idea validation first", "a learned map is not training-free",
                            "a readout does not establish action causality or first semantic layer"]}
    plan["identity"] = digest(plan)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Preserve the frozen experiment; use a new campaign for changed settings")
    write_json(path, plan)
    print(json.dumps({"plan": str(path), "paired_inputs": len(paired),
                      "train_pairs": sum(r["partition"] == "train" for r in paired),
                      "validation_pairs": sum(r["partition"] == "validation" for r in paired),
                      "self_readout_records_verified": review["records"]}), flush=True)


if __name__ == "__main__":
    main()
