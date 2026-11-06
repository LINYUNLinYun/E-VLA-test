"""Evaluate frozen aligned readouts on previously unused official initial states."""
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.pi05_hidden_probe.backend import Patch, select_positions
from experiments.pi05_hidden_probe.common import append_jsonl, control_observation, write_json
from experiments.pi05_hidden_probe.data import load_observation, read_manifest
from experiments.pi05_hidden_probe.hf_backend import HFBackend
from experiments.pi05_hidden_probe.metrics import balanced_accuracy, parse_answer
from experiments.pi05_self_readout.run import cached_backend_class, evaluate, matched_random, robot_prompt, sha
from .common import acquire_gpu, plan
from .linear import Affine
from .prepare import OUT

FIELDS = ("sample_id", "route", "control_id", "condition")


def key(record):
    return tuple(record.get(k) for k in FIELDS)


def expected_keys(rows, config):
    result = set()
    for row in rows:
        for control, condition in [("real_image", "direct"), ("black", "nopatch"), ("gray", "nopatch")]:
            result.add((row["sample_id"], None, control, condition))
        for route in config["evaluation_routes"]:
            for control in config["evaluation_controls"]:
                for condition in config["evaluation_conditions"]:
                    result.add((row["sample_id"], route, control, condition))
    return result


def load_map(path):
    with np.load(path, allow_pickle=False) as data:
        return Affine(data["weight"].copy(), data["bias"].copy())


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["route"], row["control_id"], row["condition"])].append(row)
    metrics = []
    for (route, control, condition), rows in groups.items():
        pairs = defaultdict(list)
        for row in rows:
            pairs[row["counterfactual_group_id"]].append(row)
        metrics.append({"route": route, "control_id": control, "condition": condition, "n": len(rows),
                        "scene_groups": len(pairs), "candidate_bacc": balanced_accuracy(rows, "score_prediction"),
                        "generation_bacc": balanced_accuracy(rows),
                        "candidate_pair_all_correct": float(np.mean([all(r["score_prediction"] == r["label"] for r in p) for p in pairs.values()])),
                        "generation_pair_all_correct": float(np.mean([all(r["prediction"] == r["label"] for r in p) for p in pairs.values()])),
                        "donor_follow_rate": (float(np.mean([r["score_prediction"] == r["donor_label"] for r in rows])) if condition == "shuffle" else None)})
    return metrics


def finish(out, records, config, probe_records, frozen):
    metrics = summarize(records)
    write_json(out / "metrics.json", metrics)
    write_json(out / "linear_probe_results.json", {"bacc": balanced_accuracy(probe_records), "n": len(probe_records), "records": probe_records})
    lookup = {(r["route"], r["control_id"], r["condition"]): r for r in metrics}
    lines = ["# π0.5 → PaliGemma 表示对齐：新初态颜色实验", "",
             "两个模型均冻结，Source/Target 均固定第 3 层，注入主相机 256 个视觉 tokens。模型前向使用 bfloat16。", "",
             "对齐仅拟合同观测、同机器人指令的成对表示；正则强度只按校准表示 MSE 选择。线性属性探针单独使用标签训练。", "",
             "测试采用此前未使用的官方初态 40–49，10 个场景组、20 幅合成颜色图。仍属于同一 LIBERO 场景模板。", "",
             "| 路线 | 黑图候选 | 灰图候选 | 黑图自由生成 | 灰图自由生成 | 黑图错误donor跟随 | 灰图错误donor跟随 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    pct = lambda value: f"{100 * value:.1f}%"
    for route in config["evaluation_routes"]:
        real = [lookup[(route, control, "real")] for control in config["evaluation_controls"]]
        shuffled = [lookup[(route, control, "shuffle")] for control in config["evaluation_controls"]]
        cells = [route] + [pct(r["candidate_bacc"]) for r in real] + [pct(r["generation_bacc"]) for r in real] + [pct(r["donor_follow_rate"]) for r in shuffled]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "identity=直接注入；mean_scale=训练集均值/整体尺度匹配；affine=无答案标签的正则仿射映射；target_own=同一PaliGemma自身特征。", "",
              f"独立的 Source 均值池化线性颜色探针测试 BAcc：{pct(balanced_accuracy(probe_records))}。", "",
              f"对齐验证选择：{frozen['alignment_selected']}；直接注入与均值/尺度匹配的验证表示 MSE：{frozen['validation_state_baselines']}。", "",
              "候选打分与自由生成分别报告；失败不表示 Source 缺少颜色信息。学习映射后不再属于免训练读出，结果也不证明机器人动作使用了该属性。", ""]
    (out / "summary.md").write_text("\n".join(lines))
    write_json(out / "complete.json", {"complete": True, "identity": config["identity"], "records": len(records),
              "records_sha256": sha(out / "records.jsonl"), "interpretation_review_required": True})


def main():
    config = plan()
    out = OUT / "evaluation"
    out.mkdir(exist_ok=True)
    for dependency in ("fit", "fresh_data", "fresh_source"):
        while not (OUT / dependency / "complete.json").exists():
            write_json(out / "status.json", {"phase": "waiting_for_" + dependency, "pid": os.getpid()})
            time.sleep(10)
    fit_complete = json.loads((OUT / "fit/complete.json").read_text())
    if sha(OUT / "fit/frozen.json") != fit_complete["frozen_sha256"]:
        raise ValueError("Fitted model selection changed")
    frozen = json.loads((OUT / "fit/frozen.json").read_text())
    if frozen["identity"] != config["identity"]:
        raise ValueError("Fitted mapping belongs to another plan")
    for dependency in ("fresh_data", "fresh_source"):
        if json.loads((OUT / dependency / "complete.json").read_text())["identity"] != config["identity"]:
            raise ValueError("Fresh test belongs to another plan")
    for name, expected in frozen["model_sha256"].items():
        if sha(OUT / "fit" / name) != expected:
            raise ValueError("Fitted mapping changed")
    rows = read_manifest(OUT / "fresh_data/manifest.jsonl")
    if len(rows) != config["fresh_test"]["scenes"] * 2 or len({r["group_id"] for r in rows}) != config["fresh_test"]["scenes"]:
        raise ValueError("Fresh test does not cover the predeclared scene groups")
    seen_origins = {tuple(r["origin"][k] for k in ("suite", "task", "init")) for r in read_manifest(config["fresh_test"]["forbidden_origins_manifest"])}
    if any(tuple(r["origin"][k] for k in ("suite", "task", "init")) in seen_origins for r in rows):
        raise ValueError("Test initial-state leakage")
    expected = expected_keys(rows, config)
    records = []
    if (out / "records.jsonl").exists():
        records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines() if line.strip()]
    done = {key(r) for r in records}
    if len(done) != len(records) or not done <= expected or any(r["identity"] != config["identity"] for r in records):
        raise ValueError("Invalid partial evaluation")
    source_index = json.loads((OUT / "fresh_source/index.json").read_text())
    if source_index["identity"] != config["identity"]:
        raise ValueError("Fresh Source index belongs to another plan")
    source_files = {item["sample_id"]: item for item in source_index["items"]}
    source = {}
    for row in rows:
        item = source_files[row["sample_id"]]
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Fresh Source states changed")
        with np.load(item["path"], allow_pickle=False) as data:
            source[row["sample_id"]] = data["x"].copy()
    affine, mean_scale, probe = [load_map(OUT / "fit" / name) for name in ["affine.npz", "mean_scale.npz", "linear_probe.npz"]]
    probe_records = [{"sample_id": r["sample_id"], "group_id": r["group_id"], "label": r["label"], "labels": r["labels"],
                      "prediction": "blue" if float(probe(source[r["sample_id"]].mean(axis=0))) >= 0 else "red"} for r in rows]
    if done == expected:
        finish(out, records, config, probe_records, frozen)
        return
    gpu, lease = acquire_gpu(out / "status.json")
    backend = cached_backend_class()(config["target_checkpoint"])
    write_json(out / "model_audit.json", backend.audit())
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["counterfactual_group_id"]].append(row)
    integrity_done, use_cache = False, True

    def record(row, route, control, condition, prefix, patch=None, donor=None):
        meta = {k: row[k] for k in ["sample_id", "group_id", "episode_id", "scene_id", "counterfactual_group_id", "split", "label", "labels"]}
        meta.update(route=route, control_id=control, condition=condition, identity=config["identity"],
                    donor_sample_id=donor["sample_id"] if donor else None, donor_label=donor["label"] if donor else None)
        if key(meta) in done:
            return
        meta.update(evaluate(backend, prefix, patch, row["labels"], config["max_new_tokens"], use_cache))
        append_jsonl(out / "records.jsonl", meta)
        records.append(meta)
        done.add(key(meta))
        if len(records) % 24 == 0:
            write_json(out / "status.json", {"phase": "running", "gpu": gpu, "records": len(records), "expected": len(expected)})
            print("evaluation", len(records), len(expected), flush=True)

    for pair_id in sorted(pairs):
        pair = sorted(pairs[pair_id], key=lambda r: r["sample_id"])
        if len(pair) != 2 or {r["label"] for r in pair} != {"red", "blue"}:
            raise ValueError("Incomplete fresh counterfactual pair")
        observations = [load_observation(row) for row in pair]
        real = [backend.prepare_target(obs, row["question"]) for obs, row in zip(observations, pair)]
        blanks = [{control: backend.prepare_target(control_observation(obs, {"kind": "constant", "value": value}), row["question"])
                   for control, value in config["evaluation_controls"].items()} for obs, row in zip(observations, pair)]
        values = []
        for obs, row in zip(observations, pair):
            native = backend.prepare_target(obs, robot_prompt(row))
            positions = select_positions(native, "image:base_0_rgb")
            own = backend.forward(native, capture_layers=[3], capture_positions=positions).hidden_states[3]
            x = source[row["sample_id"]]
            values.append({"identity": x, "mean_scale": mean_scale(x), "affine": affine(x), "target_own": own})
        if not integrity_done:
            cases = []
            for route, control in [("target_own", "gray"), ("identity", "black"), ("affine", "gray")]:
                prefix = blanks[0][control]
                patch = Patch(3, select_positions(prefix, "image:base_0_rgb"), values[0][route])
                full = backend.forward(prefix, patch=patch)
                cached_logits, cache = backend._cached_prefill(prefix, patch)
                cached = backend.generate(prefix, patch=patch, max_new_tokens=8)
                reference = HFBackend.generate(backend, prefix, patch=patch, max_new_tokens=8)
                cases.append({"route": route, "control_id": control,
                              "logits_exact": bool(np.array_equal(full.logits, backend._array(cached_logits[0]))),
                              "tokens_exact": cached["token_ids"] == reference["token_ids"]})
                del cache
            use_cache = all(r["logits_exact"] and r["tokens_exact"] for r in cases)
            write_json(out / "cache_integrity.json", {"cases": cases, "use_cache": use_cache, "fallback": "full reference recomputation if parity fails"})
            integrity_done = True
        for i, row in enumerate(pair):
            record(row, None, "real_image", "direct", real[i])
            for control in config["evaluation_controls"]:
                record(row, None, control, "nopatch", blanks[i][control])
            for route in config["evaluation_routes"]:
                random = matched_random(values[i][route], [42, row["sample_id"], route])
                for control in config["evaluation_controls"]:
                    prefix = blanks[i][control]
                    positions = select_positions(prefix, "image:base_0_rgb")
                    for condition, data in [("real", values[i][route]), ("shuffle", values[1-i][route]), ("random", random)]:
                        record(row, route, control, condition, prefix, Patch(3, positions, data), pair[1-i] if condition == "shuffle" else None)
        print("evaluation_pair_complete", pair_id, flush=True)
    if done != expected:
        raise ValueError("Incomplete evaluation matrix")
    finish(out, records, config, probe_records, frozen)
    write_json(out / "status.json", {"phase": "complete", "records": len(records), "gpu": gpu})
    lease.close()


if __name__ == "__main__":
    main()
