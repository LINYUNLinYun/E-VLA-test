"""Read real-image PaliGemma residuals in the same model's control-image pass.

This is a bounded diagnostic on the previous calibration color pairs. It does
not train a mapping, change either model, select a best test layer, or claim
that same-model success establishes pi0.5-to-PaliGemma compatibility.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from experiments.pi05_hidden_probe.backend import Patch, select_positions
from experiments.pi05_hidden_probe.common import append_jsonl, control_observation, digest, write_json
from experiments.pi05_hidden_probe.data import load_observation, read_manifest
from experiments.pi05_hidden_probe.hf_backend import HFBackend
from experiments.pi05_hidden_probe.metrics import balanced_accuracy, parse_answer
from experiments.pi05_hidden_probe.workflow import difference


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "results/pi05_hidden_probe/run_20261105"
CONTEXTS = ("same_question", "robot_instruction")
SELECTORS = ("image:base_0_rgb", "images")
CONTROLS = {"black": 0, "gray": 127}
CONDITIONS = ("real", "shuffle", "random")
KEY_FIELDS = ("sample_id", "source_context", "selector", "layer", "control_id", "condition")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def key(row):
    return tuple(row.get(field) for field in KEY_FIELDS)


def color_pairs(manifest, max_groups):
    groups = defaultdict(list)
    for row in read_manifest(manifest):
        if row["split"] == "calibration" and row["task_type"] == "visual_attribute":
            groups[row["counterfactual_group_id"]].append(row)
    pairs = []
    for identifier in sorted(groups)[:max_groups]:
        pair = sorted(groups[identifier], key=lambda row: row["sample_id"])
        if len(pair) != 2 or {r["label"] for r in pair} != {"red", "blue"}:
            raise ValueError(f"Expected one complete red/blue pair: {identifier}")
        if len({r["original_task_instruction"] for r in pair}) != 1:
            raise ValueError("Counterfactual instructions differ")
        if len({r["question"] for r in pair}) != 1 or len({r["group_id"] for r in pair}) != 1:
            raise ValueError("Counterfactual questions/groups differ")
        pairs.append(pair)
    if not pairs:
        raise ValueError("No calibration color pairs")
    return pairs


def expected_keys(rows, layers):
    expected = set()
    for row in rows:
        for control, condition in [("real_image", "direct"), ("black", "nopatch"), ("gray", "nopatch")]:
            expected.add((row["sample_id"], None, None, None, control, condition))
        for context in CONTEXTS:
            for selector in SELECTORS:
                for layer in layers:
                    for control in CONTROLS:
                        for condition in CONDITIONS:
                            expected.add((row["sample_id"], context, selector, layer, control, condition))
    return expected


def matched_random(values, identity):
    seed = int(digest(identity)[:16], 16)
    random = np.random.default_rng(seed).normal(size=values.shape).astype(np.float32)
    return random * (np.linalg.norm(values, axis=-1, keepdims=True)
                     / np.maximum(np.linalg.norm(random, axis=-1, keepdims=True), 1e-30))


def robot_prompt(row):
    # The audited pi05_libero config has discrete_state_input=False. Its source
    # tokenizer cleans the instruction and appends a newline answer separator.
    return row["original_task_instruction"].strip().replace("_", " ").replace("\n", " ") + "\n"


def cached_backend_class():
    path = PREVIOUS / "cached_target.py"
    spec = importlib.util.spec_from_file_location("self_readout_cached_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CachedHFBackend


def make_plan(args, rows):
    if len(set(args.layers)) != len(args.layers) or any(q < 0 or q >= 18 for q in args.layers):
        raise ValueError("Use distinct layer boundaries in 0..17; final boundary has no further token mixing")
    code_files = [Path(__file__), PREVIOUS / "cached_target.py",
                  ROOT / "experiments/pi05_hidden_probe/hf_backend.py",
                  ROOT / "experiments/pi05_hidden_probe/backend.py",
                  ROOT / "experiments/pi05_hidden_probe/common.py",
                  ROOT / "experiments/pi05_hidden_probe/data.py",
                  ROOT / "experiments/pi05_hidden_probe/metrics.py"]
    plan = {
        "experiment": "PaliGemma real-image residuals -> same checkpoint constant-image readout",
        "scope": "diagnostic; previous calibration color pairs only; no new independent-test claim",
        "checkpoint": str(args.checkpoint.resolve()), "dtype": args.dtype,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha(args.manifest),
        "samples": [{"sample_id": r["sample_id"], "group_id": r["group_id"],
                     "observation_sha256": sha(r["observation_path"])} for r in rows],
        "layers": args.layers, "layer_convention": "0=scaled input; k=after k full blocks, before final RMSNorm",
        "source_contexts": {"same_question": "real images + exact readout question",
                            "robot_instruction": "real images + cleaned original robot instruction + newline, matching audited pi05_libero text format"},
        "target_context": "constant images + exact readout question; fixed bidirectional prefix",
        "selectors": list(SELECTORS), "controls": CONTROLS, "conditions": list(CONDITIONS),
        "donor": "opposite-color member of the same counterfactual pair",
        "random": "per-token norm-matched; deterministic by sample/context/selector/layer, shared across controls",
        "max_new_tokens": args.max_new_tokens, "seed": 42,
        "candidate_scoring": "exact one-token label log probabilities; same prefill as generation",
        "primary_comparison": "robot_instruction / base camera / layer 3, matching previous color readout",
        "diagnostic_comparisons": "same_question context; both active cameras; all predeclared layers",
        "code_sha256": {str(p): sha(p) for p in code_files},
        "expected_records": len(expected_keys(rows, args.layers)),
        "previous_checkpoint_verification": str(PREVIOUS / "target_weight_verification.json"),
    }
    plan["identity"] = digest(plan)
    return plan


def integrity(backend, row, layers, max_new_tokens):
    obs = load_observation(row)
    real = backend.prepare_target(obs, row["question"])
    positions = select_positions(real, "images")
    captures = backend.forward(real, capture_layers=sorted(set(layers) | {0, 3}), capture_positions=positions)
    reference = backend.reference_forward(real)
    own = backend.forward(real, patch=Patch(3, positions, captures.hidden_states[3]))
    checks = {"native_logits": difference(captures.logits, reference.logits, 1e-5, 1e-5),
              "native_normalized": difference(captures.normalized, reference.normalized, 1e-5, 1e-5),
              "self_write": difference(own.logits, captures.logits, 1e-5, 1e-5)}
    cache_cases = []
    probe_layer = next((q for q in layers if q > 0), 3)
    for control, layer, selector in [("black", 0, "images"), ("gray", probe_layer, "image:base_0_rgb"),
                                     ("black", max(layers), "images")]:
        blank = backend.prepare_target(control_observation(obs, {"kind": "constant", "value": CONTROLS[control]}), row["question"])
        selected = select_positions(blank, selector)
        offsets = [positions.index(p) for p in selected]
        patch = Patch(layer, selected, captures.hidden_states[layer][offsets])
        full = backend.forward(blank, patch=patch)
        logits, cache = backend._cached_prefill(blank, patch)
        cached = backend.generate(blank, patch=patch, max_new_tokens=max_new_tokens)
        recomputed = HFBackend.generate(backend, blank, patch=patch, max_new_tokens=max_new_tokens)
        cache_cases.append({"control_id": control, "layer": layer, "selector": selector,
                            "logits_exact": bool(np.array_equal(full.logits, backend._array(logits[0]))),
                            "tokens_exact": cached["token_ids"] == recomputed["token_ids"],
                            "cached": cached, "reference": recomputed})
        del cache
    for control, value in CONTROLS.items():
        blank = backend.prepare_target(control_observation(obs, {"kind": "constant", "value": value}), row["question"])
        swapped = backend.forward(blank, patch=Patch(0, positions, captures.hidden_states[0]))
        checks[f"all_images_layer0_{control}"] = difference(swapped.logits, captures.logits, 1e-5, 1e-5)
    return {"checks": checks, "passed": all(c["passed"] for c in checks.values()),
            "cache_parity_cases": cache_cases,
            "use_cached_generation": all(c["logits_exact"] and c["tokens_exact"] for c in cache_cases)}


def evaluate(backend, prefix, patch, labels, max_new_tokens, use_cache):
    start = time.monotonic()
    ids = {label: backend._tokenizer.encode(label, add_special_tokens=False) for label in labels}
    if any(len(tokens) != 1 for tokens in ids.values()):
        raise ValueError("Shared-prefill scoring is restricted to exact one-token labels")
    if use_cache:
        logits_tensor, cache = backend._cached_prefill(prefix, patch)
        logits = backend._array(logits_tensor[0]).astype(np.float64)
        answer = []
        for _ in range(max_new_tokens):
            token = int(backend.torch.argmax(logits_tensor[0]).item())
            answer.append(token)
            if token == backend._tokenizer.eos_token_id or len(answer) == max_new_tokens:
                break
            logits_tensor, cache = backend._cached_advance(prefix, cache, answer)
        generation = {"text": backend._tokenizer.decode(answer, skip_special_tokens=True),
                      "token_ids": answer, "eos": answer[-1] == backend._tokenizer.eos_token_id}
        del cache
    else:
        logits = backend.forward(prefix, patch=patch).logits.astype(np.float64)
        generation = HFBackend.generate(backend, prefix, patch=patch, max_new_tokens=max_new_tokens)
    log_z = float(logits.max() + np.log(np.exp(logits - logits.max()).sum()))
    scores = {label: float(logits[tokens[0]] - log_z) for label, tokens in ids.items()}
    return {"raw_generation": generation["text"], "token_ids": generation["token_ids"], "eos": generation["eos"],
            "prediction": parse_answer(generation["text"], labels),
            "candidate_scores": scores, "score_prediction": max(scores, key=scores.get),
            "seconds": time.monotonic() - start}


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row.get(f) for f in KEY_FIELDS[1:])].append(row)
    result = []
    for fields, rows in groups.items():
        pairs = defaultdict(list)
        for row in rows:
            pairs[row["counterfactual_group_id"]].append(row)
        result.append({**dict(zip(KEY_FIELDS[1:], fields)), "n": len(rows), "scene_groups": len(pairs),
                       "candidate_bacc": balanced_accuracy(rows, "score_prediction"),
                       "generation_bacc": balanced_accuracy(rows),
                       "candidate_pair_all_correct": float(np.mean([all(r["score_prediction"] == r["label"] for r in pair) for pair in pairs.values()])),
                       "generation_pair_all_correct": float(np.mean([all(r["prediction"] == r["label"] for r in pair) for pair in pairs.values()])),
                       "donor_follow_rate": (float(np.mean([r["score_prediction"] == r["donor_label"] for r in rows]))
                                             if rows[0].get("donor_label") else None),
                       "raw_generation_counts": dict(Counter(r["raw_generation"] for r in rows))})
    return result


def finish(out, records, plan):
    metrics = aggregate(records)
    write_json(out / "metrics.json", metrics)
    lookup = {(r["source_context"], r["selector"], r["layer"], r["control_id"], r["condition"]): r for r in metrics}
    lines = ["# PaliGemma 同模型隐层读出检查", "",
             f"复用旧校准集的 {len(plan['samples'])} 个颜色样本、{len(set(r['group_id'] for r in plan['samples']))} 个场景组。只检查读出通道，不是新的独立测试，也没有训练映射。", "",
             f"Source 和 Target 是同一个冻结 PaliGemma-3B-mix-224，精度 {plan['dtype']}。提取层与注入层相同。base=agent 相机 256 tokens；all=两个有效相机 512 tokens。", "",
             "same_question：源端真实图片＋颜色问题；robot_instruction：源端真实图片＋原机器人指令，目标端仍是颜色问题。", "",
             "候选指标比较 red/blue 的首 token 对数概率；自由生成采用原有完整短答案解析，最长 8 tokens。", "",
             "| 源提示 | 视觉位置 | 层 | 黑图候选 | 灰图候选 | 黑图生成 | 灰图生成 | 黑图错误donor跟随 | 灰图错误donor跟随 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    pct = lambda value: f"{100 * value:.1f}%"
    for context in CONTEXTS:
        for selector in SELECTORS:
            for layer in plan["layers"]:
                real = [lookup[(context, selector, layer, control, "real")] for control in CONTROLS]
                shuffled = [lookup[(context, selector, layer, control, "shuffle")] for control in CONTROLS]
                cells = [context, "base" if selector != "images" else "all", str(layer)]
                cells += [pct(r["candidate_bacc"]) for r in real]
                cells += [pct(r["generation_bacc"]) for r in real]
                cells += [pct(r["donor_follow_rate"]) for r in shuffled]
                lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 未注入与正常看图", ""]
    for condition, control in [("direct", "real_image"), ("nopatch", "black"), ("nopatch", "gray")]:
        row = lookup[(None, None, None, control, condition)]
        lines.append(f"- {condition}/{control}：候选 {pct(row['candidate_bacc'])}，生成 {pct(row['generation_bacc'])}；实际输出 {row['raw_generation_counts']}。")
    lines += ["", "随机向量、成对全对率、原始生成及完整条件见 metrics.json 和 records.jsonl。",
              "", "本检查将同模型读出与跨模型表示兼容性分开；同模型成功本身不证明 π0.5→PaliGemma 的表示已经对齐。", ""]
    (out / "summary.md").write_text("\n".join(lines))
    write_json(out / "complete.json", {"complete": True, "identity": plan["identity"], "records": len(records),
                                     "records_sha256": sha(out / "records.jsonl"),
                                     "interpretation_review_required": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PREVIOUS / "counterfactual_data_v2/manifest.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "shared/checkpoints/pi05_hidden_probe/paligemma-3b-mix-224")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15])
    parser.add_argument("--max-groups", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_groups < 1 or args.max_new_tokens < 1:
        raise ValueError("Group/token limits must be positive")
    pairs = color_pairs(args.manifest, args.max_groups)
    rows = [row for pair in pairs for row in pair]
    plan = make_plan(args, rows)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        if json.loads((out / "plan.json").read_text())["identity"] != plan["identity"]:
            if args.prepare_only and not (out / "records.jsonl").exists() and not (out / "model_audit.json").exists():
                write_json(out / "plan.json", plan)
            else:
                raise ValueError("Plan/source/code changed; preserve the previous output and use a new directory")
    else:
        write_json(out / "plan.json", plan)
    if args.prepare_only:
        print(json.dumps({"plan": str(out / "plan.json"), "expected_records": plan["expected_records"]}), flush=True)
        return
    expected = expected_keys(rows, args.layers)
    records = []
    if (out / "records.jsonl").exists():
        if not args.resume:
            raise FileExistsError("Existing records require explicit --resume")
        records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines() if line.strip()]
    done = {key(r) for r in records}
    if len(done) != len(records) or not done <= expected or any(r["identity"] != plan["identity"] for r in records):
        raise ValueError("Duplicate, unexpected or mismatched records")
    if done == expected:
        finish(out, records, plan)
        return
    # Set thread count before loading/processing real images, especially on CPU.
    from experiments.pi05_hidden_probe.hf_backend import load_official_runtime
    torch, *_ = load_official_runtime()
    torch.set_num_threads(args.threads)
    backend = cached_backend_class()(args.checkpoint, device=args.device, dtype=args.dtype)
    write_json(out / "model_audit.json", backend.audit())
    print("model_loaded", flush=True)
    checks = integrity(backend, rows[0], args.layers, args.max_new_tokens)
    write_json(out / "integrity.json", checks)
    if not checks["passed"]:
        raise RuntimeError("Numerical integrity failed")
    use_cache = checks["use_cached_generation"]
    print("integrity_passed", "cached=" + str(use_cache), flush=True)
    start = time.monotonic()
    n_initial = len(records)

    def record(row, context, selector, layer, control, condition, prefix, patch=None, donor=None):
        metadata = {name: row[name] for name in ["sample_id", "group_id", "episode_id", "scene_id", "counterfactual_group_id", "split", "label", "labels"]}
        metadata.update(source_context=context, selector=selector, layer=layer, control_id=control,
                        condition=condition, identity=plan["identity"],
                        donor_sample_id=donor["sample_id"] if donor else None,
                        donor_label=donor["label"] if donor else None)
        if key(metadata) in done:
            return
        metadata.update(evaluate(backend, prefix, patch, row["labels"], args.max_new_tokens, use_cache))
        append_jsonl(out / "records.jsonl", metadata)
        done.add(key(metadata))
        records.append(metadata)
        if len(records) % 48 == 0:
            elapsed = time.monotonic() - start
            write_json(out / "status.json", {"phase": "running", "records": len(records), "expected": len(expected),
                                           "elapsed_this_attempt_seconds": elapsed,
                                           "records_per_second": (len(records) - n_initial) / max(elapsed, 1e-9)})
            print(f"records={len(records)}/{len(expected)} seconds={elapsed:.1f}", flush=True)

    for pair in pairs:
        observations = [load_observation(row) for row in pair]
        real_prefixes = [backend.prepare_target(obs, row["question"]) for obs, row in zip(observations, pair)]
        blanks = [{control: backend.prepare_target(control_observation(obs, {"kind": "constant", "value": value}), row["question"])
                   for control, value in CONTROLS.items()} for obs, row in zip(observations, pair)]
        for index, row in enumerate(pair):
            record(row, None, None, None, "real_image", "direct", real_prefixes[index])
            for control in CONTROLS:
                record(row, None, None, None, control, "nopatch", blanks[index][control])
        for context in CONTEXTS:
            source_prefixes = (real_prefixes if context == "same_question" else
                               [backend.prepare_target(obs, robot_prompt(row)) for obs, row in zip(observations, pair)])
            positions = select_positions(source_prefixes[0], "images")
            hidden = [backend.forward(prefix, capture_layers=args.layers, capture_positions=positions).hidden_states for prefix in source_prefixes]
            for index, row in enumerate(pair):
                for selector in SELECTORS:
                    selected = select_positions(source_prefixes[index], selector)
                    offsets = [positions.index(p) for p in selected]
                    for layer in args.layers:
                        values = hidden[index][layer][offsets]
                        donor_values = hidden[1 - index][layer][offsets]
                        random = matched_random(values, [42, row["sample_id"], context, selector, layer])
                        for control in CONTROLS:
                            target = blanks[index][control]
                            if select_positions(target, selector) != selected:
                                raise ValueError("Source/Target camera positions differ")
                            for condition, patch_values in [("real", values), ("shuffle", donor_values), ("random", random)]:
                                record(row, context, selector, layer, control, condition, target,
                                       Patch(layer, selected, patch_values), pair[1 - index] if condition == "shuffle" else None)
        print("pair_complete", pair[0]["counterfactual_group_id"], flush=True)
    if done != expected:
        raise RuntimeError(f"Incomplete matrix: {len(expected - done)} missing")
    finish(out, records, plan)
    write_json(out / "status.json", {"phase": "complete", "records": len(records), "expected": len(expected),
                                   "elapsed_this_attempt_seconds": time.monotonic() - start})
    print("complete", out, flush=True)


if __name__ == "__main__":
    main()
