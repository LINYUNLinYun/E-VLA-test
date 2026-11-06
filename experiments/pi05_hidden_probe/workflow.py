"""Shared experiment steps, kept separate from the policy's inference API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import (FeatureCache, append_jsonl, check_resume, config_hash,
                     control_observation, digest, observation_hash, read_jsonl, source_layers,
                     target_prompt, write_json)
from .data import choose_donors, load_observation
from .metrics import balanced_accuracy, parse_answer


def difference(actual, expected, atol, rtol):
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    error = actual - expected
    return {"max_abs": float(np.max(np.abs(error))), "mse": float(np.mean(error**2)),
            "passed": bool(np.all(np.isfinite(actual)) and
                           np.allclose(actual, expected, atol=atol, rtol=rtol))}


def integrity_check(source, target, row, config):
    """Do not run semantic scans until these computation controls pass."""
    from .backend import Patch, select_positions
    obs = load_observation(row)
    probe = next(p for p in config["probes"] if p["task_type"] == row["task_type"])
    tol = config.get("integrity", {})
    atol, rtol = float(tol.get("atol", 1e-5)), float(tol.get("rtol", 1e-5))
    checks = {}
    for name, backend, prefix in (
        ("source", source, source.prepare_source(obs)),
        ("target", target, target.prepare_target(obs, target_prompt(row, probe))),
    ):
        depth = int(backend.audit()["depth"])
        layer = min(probe["target_layers"][0], depth - 1)
        reference = backend.reference_forward(prefix)
        repeat = backend.reference_forward(prefix)
        actual = backend.forward(prefix, capture_layers=[layer])
        checks[f"{name}_reference_repeat"] = difference(repeat.normalized, reference.normalized, atol, rtol)
        checks[f"{name}_extraction_vs_reference"] = difference(actual.normalized, reference.normalized, atol, rtol)
        checks[f"{name}_logits_vs_reference"] = difference(actual.logits, reference.logits, atol, rtol)
        positions = select_positions(prefix, "images")
        patched = backend.forward(prefix, patch=Patch(layer, positions, actual.hidden_states[layer][positions]))
        checks[f"{name}_self_replace"] = difference(patched.normalized, actual.normalized, atol, rtol)
    real = target.prepare_target(obs, target_prompt(row, probe))
    real_result = target.forward(real)
    for control in config["controls"]:
        blank = target.prepare_target(control_observation(obs, control), target_prompt(row, probe))
        positions = select_positions(real, "images")
        blank_positions = select_positions(blank, "images")
        if positions != blank_positions or not np.array_equal(real.input_mask, blank.input_mask):
            raise ValueError("layer-0 integrity requires identical masks and token positions")
        patch = Patch(0, blank_positions, np.asarray(real.embedded)[positions])
        result = target.forward(blank, patch=patch)
        checks[f"layer0_{control['id']}"] = difference(result.normalized, real_result.normalized, atol, rtol)
        checks[f"layer0_logits_{control['id']}"] = difference(result.logits, real_result.logits, atol, rtol)
    return {"passed": all(c["passed"] for c in checks.values()), "atol": atol, "rtol": rtol,
            "sample_id": row["sample_id"], "checks": checks,
            "scope": "native prefix vs explicit blocks; self replacement; target-only layer-0 image replacement"}


def decode(target, prefix, row, config, patch=None):
    generation = target.generate(prefix, patch=patch, max_new_tokens=config["max_new_tokens"])
    scores = (target.score_answers(prefix, row["labels"], patch=patch,
                                   length_normalize=config["length_normalize"])
              if config["score_candidates"] else {})
    if scores and not all(np.isfinite(v) for v in scores.values()):
        raise ValueError("Nonfinite candidate score")
    return {"raw_generation": generation["text"], "token_ids": generation["token_ids"],
            "eos": generation["eos"],
            "prediction": parse_answer(generation["text"], row["labels"], config["aliases"].get(row["task_type"], {})),
            "candidate_scores": scores,
            "score_prediction": max(scores, key=scores.get) if scores else None}


def result_base(row, probe, config, run, control, q):
    fields = ("sample_id", "episode_id", "scene_id", "counterfactual_group_id", "group_id",
              "split", "task_type", "label", "labels", "label_source", "is_synthetic")
    return {**{key: row[key] for key in fields}, "probe_id": probe["id"],
            "prompt_id": row["prompt_id"], "question": target_prompt(row, probe),
            "selector": probe["source_selector"], "target_selector": probe["target_selector"],
            "target_layer": q, "control_id": control["id"], "seed": config["seed"],
            "config_hash": run["config_hash"],
            "source_depth": run["source_identity"]["audit"]["depth"],
            "source_checkpoint": config["model"]["checkpoint"],
            "target_checkpoint": config.get("target_model", config["model"])["checkpoint"]}


def row_key(row):
    return digest([row.get(k) for k in ("sample_id", "probe_id", "source_layer", "target_layer",
                                       "condition", "control_id", "config_hash")])


def expected_result_keys(config, rows, layers):
    keys = set()
    for row in rows:
        for probe in config["probes"]:
            if probe["task_type"] != row["task_type"]:
                continue
            for q in probe["target_layers"]:
                for control in config["controls"]:
                    for condition in config["conditions"]:
                        for layer in (layers if condition in {"real", "shuffle", "random"} else [None]):
                            keys.add(row_key({"sample_id": row["sample_id"], "probe_id": probe["id"],
                                              "source_layer": layer, "target_layer": q, "condition": condition,
                                              "control_id": control["id"], "config_hash": config_hash(config)}))
    return keys


def run_probe(config, rows, source, target, out, run, *, resume=False):
    from .backend import Patch, select_positions
    out = Path(out)
    result_path = out / "probe_results.jsonl"
    if result_path.exists() and not resume:
        raise FileExistsError(f"Use --resume to continue this run, or a new output_dir: {result_path}")
    check_resume(out / "run_identity.json", run)
    existing = read_jsonl(result_path) if result_path.exists() else []
    seen = {row_key(r) for r in existing}
    if len(seen) != len(existing):
        raise ValueError("Duplicate result identities; repair the output before resuming")
    unknown = set(config["conditions"]) - {"real", "nopatch", "self", "shuffle", "random", "direct", "layer0"}
    if unknown or not {"real", "nopatch", "shuffle"}.issubset(config["conditions"]):
        raise ValueError("Conditions must include real, nopatch, shuffle; unknown conditions: " + str(unknown))
    donors = choose_donors(rows, seed=config["seed"])
    by_id = {r["sample_id"]: r for r in rows}
    layers = source_layers(config, source)
    expected = expected_result_keys(config, rows, layers)
    if seen - expected:
        raise ValueError("Results contain identities outside this run's planned combinations")
    cache = FeatureCache(config.get("cache_dir", out / "source_cache"), run["source_identity"])
    depth = int(target.audit()["depth"])

    def save(base, condition, layer, prediction, **extra):
        record = {**base, "condition": condition, "source_layer": layer,
                  "donor_id": None, "donor_label": None, **extra, **prediction}
        key = row_key(record)
        if key not in seen:
            append_jsonl(result_path, record)
            seen.add(key)

    for row in rows:
        obs = load_observation(row)
        if observation_hash(obs) != run["source_input_digests"][row["sample_id"]]:
            raise ValueError("Observation changed after the run identity was recorded")
        for probe in [p for p in config["probes"] if p["task_type"] == row["task_type"]]:
            source_features, source_positions, feature_key = cache.get(
                source, obs, probe["source_selector"], layers, row.get("object_text"))
            donor_row = by_id[donors[row["sample_id"]]]
            donor_obs = load_observation(donor_row)
            if observation_hash(donor_obs) != run["source_input_digests"][donor_row["sample_id"]]:
                raise ValueError("Donor observation changed after the run identity was recorded")
            donor_features, donor_positions, donor_key = cache.get(
                source, donor_obs, probe["source_selector"], layers, donor_row.get("object_text"))
            prompt = target_prompt(row, probe)
            for control in config["controls"]:
                target_prefix = target.prepare_target(control_observation(obs, control), prompt)
                positions = select_positions(target_prefix, probe["target_selector"], probe.get("placeholder", "X"))
                for q in probe["target_layers"]:
                    if not isinstance(q, int) or not 0 <= q < depth:
                        raise ValueError(f"Target patch must precede a token-mixing block: 0 <= q < {depth}")
                    base = result_base(row, probe, config, run, control, q)
                    shared = {"source_positions": source_positions, "target_positions": positions,
                              "feature_key": feature_key}
                    pending = lambda condition, layer=None: row_key({**base, "condition": condition, "source_layer": layer}) not in seen
                    for condition in ("nopatch", "self", "direct", "layer0"):
                        if condition not in config["conditions"] or not pending(condition):
                            continue
                        prefix, patch = target_prefix, None
                        extra = dict(shared)
                        if condition == "self":
                            own = target.forward(prefix, capture_layers=[q], capture_positions=positions)
                            patch = Patch(q, positions, own.hidden_states[q])
                        elif condition in {"direct", "layer0"}:
                            direct_prompt = prompt
                            if condition == "direct" and probe.get("direct_include_instruction", False):
                                direct_prompt = "Task: " + row["original_task_instruction"] + "\n" + prompt
                            real_prefix = target.prepare_target(obs, direct_prompt)
                            if condition == "direct":
                                prefix = real_prefix
                                extra["direct_includes_instruction"] = bool(probe.get("direct_include_instruction", False))
                            else:
                                image_positions = select_positions(prefix, "images")
                                real_positions = select_positions(real_prefix, "images")
                                if image_positions != real_positions:
                                    raise ValueError("Target layer-0 image slots do not match")
                                patch = Patch(0, image_positions, np.asarray(real_prefix.embedded)[real_positions])
                                extra["intervention_layer"] = 0
                                extra["target_positions"] = image_positions
                        save(base, condition, None, decode(target, prefix, row, config, patch), **extra)
                    for layer in layers:
                        for condition in ("real", "shuffle", "random"):
                            if condition not in config["conditions"] or not pending(condition, layer):
                                continue
                            values = np.asarray(source_features[layer], np.float32)
                            extra = dict(shared)
                            if condition == "shuffle":
                                values = np.asarray(donor_features[layer], np.float32)
                                extra.update(donor_id=donor_row["sample_id"], donor_label=donor_row["label"],
                                             donor_positions=donor_positions, donor_feature_key=donor_key)
                            elif condition == "random":
                                stable_seed = int(digest([config["seed"], row["sample_id"], probe["id"], layer, q, control["id"]])[:16], 16)
                                rng = np.random.default_rng(stable_seed)
                                noise = rng.standard_normal(values.shape).astype(np.float32)
                                values = noise * (np.linalg.norm(values, axis=-1, keepdims=True) /
                                                  np.maximum(np.linalg.norm(noise, axis=-1, keepdims=True), 1e-12))
                            if values.shape != (len(positions), target_prefix.embedded.shape[-1]):
                                raise ValueError("Source/target shapes differ; implicit pooling or learned mapping is forbidden")
                            patch = Patch(q, positions, values)
                            save(base, condition, layer, decode(target, target_prefix, row, config, patch), **extra)
            print(f"finished {row['sample_id']} / {probe['id']}", flush=True)
    if seen != expected:
        raise RuntimeError("Run ended without all planned sample/layer/control combinations")
    write_json(out / "completion.json", {"completed": True, "result_rows": len(seen),
        "run_identity_digest": digest(run), "results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()})
    return result_path


def freeze_config(config, records, output, *, input_path):
    """Select q only on calibration results; retain all source layers for evaluation."""
    import yaml
    if config["stage"] != "calibration":
        raise ValueError("Only a calibration config may be frozen")
    if any(r.get("split") != "calibration" for r in records):
        raise ValueError("Refuse target-layer selection using any non-calibration result")
    if any(r.get("config_hash") != config_hash(config) for r in records):
        raise ValueError("Calibration results do not match this config")
    identity_path = Path(input_path).parent / "run_identity.json"
    if not identity_path.exists():
        raise ValueError("Freezing requires the calibration run_identity.json beside its results")
    identity = json.loads(identity_path.read_text())
    if identity["config_hash"] != config_hash(config):
        raise ValueError("Calibration run identity does not match this config")
    from .common import select_rows
    expected_ids = set(identity["sample_ids"])
    planned_rows = [r for r in select_rows(config) if r["sample_id"] in expected_ids]
    if {r["sample_id"] for r in planned_rows} != expected_ids or digest(planned_rows) != identity["manifest_digest"]:
        raise ValueError("Calibration manifest differs from the recorded run")
    depth = int(identity["source_identity"]["audit"]["depth"])
    layers = list(range(depth + 1)) if config["source_layers"] == "all" else config["source_layers"]
    keys = [row_key(r) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected_result_keys(config, planned_rows, layers):
        raise ValueError("Calibration is incomplete or contains duplicate/unplanned result identities")
    metric_key = config.get("selection_prediction", "prediction")
    if metric_key not in {"prediction", "score_prediction"}:
        raise ValueError("selection_prediction must be prediction or score_prediction")
    if metric_key == "score_prediction" and (not config["score_candidates"] or any(
        r.get("score_prediction") not in r["labels"] or not r.get("candidate_scores")
        for r in records if r["condition"] == "real"
    )):
        raise ValueError("Cannot select by candidate scores when candidate scoring was not completed")
    result = {k: v for k, v in config.items() if k != "_config_path"}
    result["probes"] = []
    chosen = {}
    for probe in config["probes"]:
        candidates = {}
        identities = None
        expected_layers = layers
        for q in probe["target_layers"]:
            real = [r for r in records if r.get("probe_id") == probe["id"] and r["condition"] == "real" and r["target_layer"] == q]
            current = {(r["sample_id"], r["source_layer"], r["control_id"]) for r in real}
            if not real or (identities is not None and current != identities):
                raise ValueError(f"Incomplete/unmatched target-layer calibration for {probe['id']}")
            identities = current
            values = []
            for layer in expected_layers:
                for control in config["controls"]:
                    subset = [r for r in real if r["source_layer"] == layer and r["control_id"] == control["id"]]
                    value = balanced_accuracy(subset, prediction_key=metric_key)
                    if not np.isfinite(value):
                        raise ValueError("Calibration lacks complete label coverage; cannot select a target layer")
                    values.append(value)
            candidates[q] = float(np.mean(values))
        q = min(candidates, key=lambda q: (-candidates[q], q))
        result["probes"].append({**probe, "target_layers": [q]})
        chosen[probe["id"]] = {"target_layer": q, "mean_balanced_accuracy": candidates[q],
                                "candidates": candidates}
    result["stage"] = "evaluation"
    result["source_layers"] = "all"
    result["output_dir"] = str(Path(config["output_dir"]).parent / "evaluation")
    result["calibration_provenance"] = {"results": str(Path(input_path).resolve()),
        "sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "selection_prediction": metric_key, "selection": chosen}
    result["frozen_hash"] = config_hash(result)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
    return result
