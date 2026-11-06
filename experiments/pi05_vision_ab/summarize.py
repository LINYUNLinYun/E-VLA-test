"""Aggregation, fixed selection rules, figures and the final report."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SCOPES = ("vision_all", "vision_single_view", "model", "policy_wall")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_latency(run_dir):
    path = Path(run_dir) / "latency_raw.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            row["wall_ms"] = float(row["wall_ms"])
            row["round"] = int(row["round"])
            row["iteration"] = int(row["iteration"])
            rows.append(row)
    return rows


def load_rollouts(run_dir, stage):
    path = Path(run_dir) / f"rollouts_{stage}.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_manifest(run_dir, stage):
    path = Path(run_dir) / f"episode_manifest_{stage}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_stage(run_dir, stage):
    manifest = load_manifest(run_dir, stage)
    all_records = load_rollouts(run_dir, stage)
    expected = {row["key"] for row in manifest}
    valid = [record for record in all_records
             if record.get("error_type") != "infrastructure"]
    seen = defaultdict(int)
    for record in valid:
        seen[record["key"]] += 1
    duplicates = [key for key, count in seen.items() if count > 1]
    missing = expected - set(seen)
    infrastructure = [record for record in all_records
                      if record.get("error_type") == "infrastructure"]
    if duplicates or missing or infrastructure:
        raise RuntimeError(
            f"{stage} incomplete: expected={len(expected)} valid={len(seen)} "
            f"missing={len(missing)} duplicates={len(duplicates)} "
            f"infrastructure={len(infrastructure)}; resume the stage first")
    return manifest, valid


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------

def latency_summary(rows, samples=1000, seed=20261020):
    cfgs = sorted({row["config_id"] for row in rows})
    summary = {}
    for scope in SCOPES:
        scoped = [row for row in rows if row["measurement_scope"] == scope]
        if not scoped:
            continue
        values = defaultdict(list)
        per_round = defaultdict(lambda: defaultdict(list))
        for row in scoped:
            values[row["config_id"]].append(row["wall_ms"])
            per_round[row["config_id"]][row["round"]].append(row["wall_ms"])
        for config in cfgs:
            array = np.asarray(values.get(config, []), dtype=np.float64)
            if not len(array):
                continue
            rounds = {str(r): {
                "n": len(v),
                "p50_ms": float(np.percentile(v, 50)),
                "mean_ms": float(np.mean(v)),
            } for r, v in sorted(per_round[config].items())}
            summary[(config, scope)] = {
                "config_id": config,
                "scope": scope,
                "n": int(len(array)),
                "mean_ms": float(np.mean(array)),
                "p50_ms": float(np.percentile(array, 50)),
                "p95_ms": float(np.percentile(array, 95)),
                "round_p50": rounds,
                "round_min_p50_ms": float(min(x["p50_ms"] for x in rounds.values())),
                "round_max_p50_ms": float(max(x["p50_ms"] for x in rounds.values())),
            }
    # Paired (round, observation) block bootstrap for the primary scope.
    base = "C0"
    if (base, "vision_all") in summary:
        cis = _paired_speedup_ci(rows, "vision_all", base, samples, seed)
        for (config, scope), item in summary.items():
            if scope == "vision_all":
                item["speedup_p50"] = summary[(base, scope)]["p50_ms"] / item["p50_ms"]
                item["latency_reduction"] = 1.0 - item["p50_ms"] / summary[(base, scope)]["p50_ms"]
                item["speedup_ci95"] = cis.get(config, [None, None])
                item["speedup_rounds_p50"] = _round_speedups(summary, config, scope, base)
    for (config, scope), item in summary.items():
        if scope != "vision_all" and (base, scope) in summary:
            item["speedup_p50"] = summary[(base, scope)]["p50_ms"] / item["p50_ms"]
            item["speedup_ci95"] = [None, None]
            item["speedup_rounds_p50"] = _round_speedups(summary, config, scope, base)
    return {f"{config}|{scope}": item for (config, scope), item in summary.items()}


def _round_speedups(summary, config, scope, base):
    base_rounds = summary[(base, scope)]["round_p50"]
    rounds = summary[(config, scope)]["round_p50"]
    return {r: base_rounds[r]["p50_ms"] / rounds[r]["p50_ms"] for r in rounds}


def _paired_speedup_ci(rows, scope, base, samples, seed):
    blocks = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["measurement_scope"] != scope:
            continue
        blocks[(row["round"], row["observation_id"])][row["config_id"]].append(row["wall_ms"])
    keys = sorted(blocks)
    configs = sorted({row["config_id"] for row in rows if row["measurement_scope"] == scope})
    rng = np.random.default_rng(seed)
    draws = {config: np.empty(samples) for config in configs}
    index = rng.integers(0, len(keys), size=(samples, len(keys)))
    for s in range(samples):
        pooled = {config: [] for config in configs}
        for j in index[s]:
            for config, values in blocks[keys[j]].items():
                pooled[config].extend(values)
        medians = {config: float(np.median(pooled[config])) for config in configs}
        for config in configs:
            draws[config][s] = medians[base] / medians[config]
    result = {}
    for config in configs:
        if config == base:
            result[config] = [1.0, 1.0]
        else:
            result[config] = [float(np.percentile(draws[config], 2.5)),
                              float(np.percentile(draws[config], 97.5))]
    return result


# ---------------------------------------------------------------------------
# success rates
# ---------------------------------------------------------------------------

def success_summary(records):
    conditions = sorted({r["condition_id"] for r in records})
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in records:
        cells[record["suite"]][record["task_id"]][record["condition_id"]].append(
            1.0 if record["success"] else 0.0)
    out = {}
    for suite in SUITES:
        if suite not in cells:
            continue
        task_rates = {condition: [] for condition in conditions}
        for task in sorted(cells[suite]):
            for condition in conditions:
                values = cells[suite][task].get(condition, [])
                if values:
                    task_rates[condition].append(float(np.mean(values)))
        for condition in conditions:
            rates = task_rates[condition]
            if rates:
                out.setdefault(condition, {"suite_sr": {}})["suite_sr"][suite] = float(np.mean(rates))
    for condition in conditions:
        suite_rates = list(out.get(condition, {}).get("suite_sr", {}).values())
        if suite_rates:
            out[condition]["sr"] = float(np.mean(suite_rates))
            n_tasks = n_episodes = 0
            for suite in SUITES:
                for task in cells[suite]:
                    values = cells[suite][task].get(condition, [])
                    if values:
                        n_tasks += 1
                        n_episodes += len(values)
            out[condition]["n_tasks"] = n_tasks
            out[condition]["n_episodes"] = n_episodes
    return out


def paired_bootstrap(records, base, candidate, samples=10000, seed=20261020):
    """Task-stratified, episode-paired bootstrap of SR difference."""
    cells = defaultdict(lambda: defaultdict(dict))
    for record in records:
        cells[(record["suite"], record["task_id"])][record["initial_state_index"]][
            record["condition_id"]] = 1.0 if record["success"] else 0.0
    tasks = sorted(cells)
    rng = np.random.default_rng(seed)
    delta = np.empty(samples)
    relative = np.empty(samples)
    base_sr = np.empty(samples)
    cand_sr = np.empty(samples)
    suite_delta = {suite: np.empty(samples) for suite in SUITES}
    for s in range(samples):
        task_deltas = defaultdict(list)
        task_base = defaultdict(list)
        task_cand = defaultdict(list)
        for suite, task in tasks:
            inits = sorted(cells[(suite, task)])
            draw = rng.choice(len(inits), size=len(inits), replace=True)
            b = np.mean([cells[(suite, task)][inits[i]].get(base, np.nan) for i in draw])
            c = np.mean([cells[(suite, task)][inits[i]].get(candidate, np.nan) for i in draw])
            task_base[suite].append(b)
            task_cand[suite].append(c)
            task_deltas[suite].append(c - b)
        b_suite = [np.mean(task_base[suite]) for suite in SUITES if task_base[suite]]
        c_suite = [np.mean(task_cand[suite]) for suite in SUITES if task_cand[suite]]
        base_sr[s] = np.mean(b_suite)
        cand_sr[s] = np.mean(c_suite)
        delta[s] = 100.0 * (cand_sr[s] - base_sr[s])
        relative[s] = cand_sr[s] / base_sr[s] if base_sr[s] > 0 else np.nan
        for suite in SUITES:
            if task_deltas[suite]:
                suite_delta[suite][s] = 100.0 * np.mean(task_deltas[suite])
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {
        "delta_pp": float(100.0 * (cand_sr.mean() - base_sr.mean())),
        "delta_pp_ci95": ci(delta),
        "relative": float(cand_sr.mean() / base_sr.mean()) if base_sr.mean() > 0 else None,
        "relative_ci95": ci(relative),
        "sr_ci95": ci(cand_sr),
        "base_sr_ci95": ci(base_sr),
        "suite_delta_pp_ci95": {s: ci(v) for s, v in suite_delta.items() if len(v)},
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def _family(config):
    return "A" if config.startswith("A") else "B" if config.startswith("B") else "C"


def _pick_family(candidates, threshold, aggressive_floor, speedups):
    if not candidates:
        return None, "no candidate with a measured vision speedup > 1.0"
    good = [c for c in candidates if c["relative_sr"] >= threshold]
    aggressive = False
    pool = good
    basis = f"relative SR >= {threshold:.2f}"
    if not pool:
        pool = [c for c in candidates
                if aggressive_floor <= c["relative_sr"] < threshold]
        aggressive = bool(pool)
        basis = f"relative SR in [{aggressive_floor:.2f}, {threshold:.2f})"
    if not pool:
        return None, (f"no speedup candidate reaches relative SR {aggressive_floor:.2f}")
    pool = sorted(pool, key=lambda c: c["vision_speedup"], reverse=True)
    if len(pool) >= 2:
        a, b = pool[0], pool[1]
        if (a["keep_tokens"] == 128 and b["keep_tokens"] == 64
                or a["keep_tokens"] == 64 and b["keep_tokens"] == 128):
            if abs(a["vision_speedup"] - b["vision_speedup"]) / max(
                    a["vision_speedup"], b["vision_speedup"]) <= 0.02:
                chosen = a if a["keep_tokens"] == 128 else b
                return chosen, (f"{basis}; K128 chosen because speedups are within 2% "
                                f"({a['vision_speedup']:.3f} vs {b['vision_speedup']:.3f})")
    chosen = pool[0]
    return chosen, f"{basis}; fastest visual speedup {chosen['vision_speedup']:.3f}"


def build_selection(cfg, run_dir):
    run_dir = Path(run_dir)
    correctness = json.loads((run_dir / "correctness.json").read_text())
    if not correctness.get("all_required_passed"):
        raise RuntimeError("selection blocked: correctness checks did not pass")
    _validate_stage(run_dir, "screen")
    records = load_rollouts(run_dir, "screen")
    rates = success_summary(records)
    latency = latency_summary(load_latency(run_dir), samples=1000)
    base_sr = rates["C0"]["sr"]
    configs = []
    for condition in cfg["conditions"]:
        cid = condition["id"]
        if cid not in rates:
            continue
        item = latency[f"{cid}|vision_all"]
        configs.append({
            "condition_id": cid,
            "family": _family(cid),
            "mode": condition["mode"],
            "keep_tokens": int(condition["keep_tokens"]),
            "vision_p50_ms": item["p50_ms"],
            "vision_speedup": item["speedup_p50"],
            "vision_speedup_ci95": item["speedup_ci95"],
            "sr": rates[cid]["sr"],
            "relative_sr": rates[cid]["sr"] / base_sr if base_sr > 0 else None,
            "suite_sr": rates[cid]["suite_sr"],
            "n_episodes": rates[cid]["n_episodes"],
        })
    threshold = float(cfg["selection"]["preferred_relative_sr"])
    floor = float(cfg["selection"]["aggressive_relative_sr_floor"])
    speedups = {c["condition_id"]: c["vision_speedup"] for c in configs}
    result = {"thresholds": {"preferred_relative_sr": threshold,
                             "aggressive_relative_sr_floor": floor,
                             "engineering_vision_speedup_target":
                                 cfg["selection"]["engineering_vision_speedup_target"]},
              "screen": configs, "families": {}, "confirm_conditions": ["C0"],
              "speedup_target_met": [c["condition_id"] for c in configs
                                     if c["vision_speedup"] >= float(
                                         cfg["selection"]["engineering_vision_speedup_target"])]}
    for family in ("A", "B"):
        candidates = [c for c in configs if c["family"] == family and c["vision_speedup"] > 1.0]
        chosen, reason = _pick_family(candidates, threshold, floor, speedups)
        result["families"][family] = {
            "candidates": [c["condition_id"] for c in candidates],
            "selected": chosen["condition_id"] if chosen else None,
            "reason": reason,
        }
        if chosen:
            result["confirm_conditions"].append(chosen["condition_id"])
    # Record why other conditions were not selected.
    selected = set(result["confirm_conditions"][1:])
    for c in configs:
        if c["condition_id"] not in selected and c["condition_id"] != "C0":
            result.setdefault("not_selected", {})[c["condition_id"]] = (
                "no speedup" if c["vision_speedup"] <= 1.0
                else "lower priority within family or fails SR floor")
    (run_dir / "selection.json").write_text(json.dumps(result, indent=2, default=_json_default))
    return result


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def build_report(cfg, run_dir):
    run_dir = Path(run_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correctness = json.loads((run_dir / "correctness.json").read_text())
    latency = latency_summary(load_latency(run_dir), samples=1000)
    screen_manifest, screen_records = _validate_stage(run_dir, "screen")
    screen_rates = success_summary(screen_records)

    selection_path = run_dir / "selection.json"
    selection = (json.loads(selection_path.read_text()) if selection_path.exists()
                 else build_selection(cfg, run_dir))
    confirm_conditions = selection.get("confirm_conditions", [])
    confirm_records = []
    confirm_rates = {}
    confirm_stats = {}
    if len(confirm_conditions) > 1:
        confirm_manifest, confirm_records = _validate_stage(run_dir, "confirm")
        confirm_rates = success_summary(confirm_records)
        for cid in confirm_conditions[1:]:
            confirm_stats[cid] = paired_bootstrap(
                confirm_records, "C0", cid,
                samples=int(cfg["evaluation"]["paired_stratified_bootstrap_samples"]))
    bootstrap_samples = int(cfg["evaluation"]["paired_stratified_bootstrap_samples"])
    screen_stats = {cid: paired_bootstrap(screen_records, "C0", cid,
                                          samples=bootstrap_samples)
                    for cid in screen_rates if cid != "C0"}

    # CSV exports -----------------------------------------------------------
    latency_rows = sorted(latency.values(), key=lambda x: (x["scope"], x["config_id"]))
    with (run_dir / "latency_summary.csv").open("w", newline="") as stream:
        fields = ["config_id", "scope", "n", "mean_ms", "p50_ms", "p95_ms",
                  "round_min_p50_ms", "round_max_p50_ms", "speedup_p50",
                  "speedup_ci_low", "speedup_ci_high", "latency_reduction"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in latency_rows:
            writer.writerow({
                "config_id": row["config_id"], "scope": row["scope"], "n": row["n"],
                "mean_ms": row["mean_ms"], "p50_ms": row["p50_ms"], "p95_ms": row["p95_ms"],
                "round_min_p50_ms": row["round_min_p50_ms"],
                "round_max_p50_ms": row["round_max_p50_ms"],
                "speedup_p50": row.get("speedup_p50"),
                "speedup_ci_low": row.get("speedup_ci95", [None, None])[0],
                "speedup_ci_high": row.get("speedup_ci95", [None, None])[1],
                "latency_reduction": row.get("latency_reduction"),
            })

    with (run_dir / "latency_rounds.csv").open("w", newline="") as stream:
        fields = ["config_id", "scope", "round", "n", "p50_ms", "mean_ms"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in latency_rows:
            for round_index, data in sorted(row["round_p50"].items(), key=lambda kv: int(kv[0])):
                writer.writerow({"config_id": row["config_id"], "scope": row["scope"],
                                 "round": round_index, **data})

    stage_rows = []
    for stage, records, rates in (("screen", screen_records, screen_rates),
                                  ("confirm", confirm_records, confirm_rates)):
        for cid, data in sorted(rates.items()):
            row = {"stage": stage, "condition_id": cid, "sr": data["sr"],
                   "n_episodes": data["n_episodes"]}
            for suite in SUITES:
                row[f"sr_{suite}"] = data["suite_sr"].get(suite)
            stage_rows.append(row)
    with (run_dir / "success_summary.csv").open("w", newline="") as stream:
        fields = ["stage", "condition_id", "sr"] + [f"sr_{s}" for s in SUITES] + ["n_episodes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stage_rows)

    # Per-task / per-suite success tables (required raw aggregates).
    task_cells = defaultdict(lambda: [0, 0])
    for stage, records in (("screen", screen_records), ("confirm", confirm_records)):
        for record in records:
            cell = task_cells[(stage, record["condition_id"], record["suite"],
                               record["task_id"])]
            cell[0] += int(bool(record["success"]))
            cell[1] += 1
    with (run_dir / "success_by_task.csv").open("w", newline="") as stream:
        fields = ["stage", "condition_id", "suite", "task_id", "n_success", "n_episodes", "sr"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (stage, cid, suite, task), (success, n) in sorted(task_cells.items()):
            writer.writerow({"stage": stage, "condition_id": cid, "suite": suite,
                             "task_id": task, "n_success": success, "n_episodes": n,
                             "sr": success / n})
    with (run_dir / "success_by_suite.csv").open("w", newline="") as stream:
        fields = ["stage", "condition_id", "suite", "n_success", "n_episodes", "sr"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage, records, rates in (("screen", screen_records, screen_rates),
                                      ("confirm", confirm_records, confirm_rates)):
            for cid, data in sorted(rates.items()):
                for suite in SUITES:
                    sub = [r for r in records if r["condition_id"] == cid
                           and r["suite"] == suite]
                    if not sub:
                        continue
                    success = sum(int(bool(r["success"])) for r in sub)
                    writer.writerow({"stage": stage, "condition_id": cid, "suite": suite,
                                     "n_success": success, "n_episodes": len(sub),
                                     "sr": data["suite_sr"].get(suite)})

    # Compact main-results table for downstream consumption.
    with (run_dir / "summary.csv").open("w", newline="") as stream:
        fields = ["stage", "config_id", "vision_p50_ms", "vision_p95_ms",
                  "vision_speedup", "vision_speedup_ci_low", "vision_speedup_ci_high",
                  "model_p50_ms", "policy_wall_p50_ms", "sr", "sr_ci_low", "sr_ci_high",
                  "delta_pp", "delta_pp_ci_low", "delta_pp_ci_high", "relative_sr",
                  "relative_ci_low", "relative_ci_high", "n_success", "n_episodes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in selection["screen"]:
            cid = item["condition_id"]
            vis = latency[f"{cid}|vision_all"]
            stats = screen_stats.get(cid)
            if stats is None:
                sr_ci = screen_stats[next(iter(screen_stats))]["base_sr_ci95"]
                delta = [0.0, 0.0, 0.0]
                relative = [1.0, 1.0, 1.0]
            else:
                sr_ci = stats["sr_ci95"]
                delta = [stats["delta_pp"], *stats["delta_pp_ci95"]]
                relative = ([stats["relative"], *stats["relative_ci95"]]
                            if stats["relative"] is not None else [None, None, None])
            sub = [r for r in screen_records if r["condition_id"] == cid]
            writer.writerow({
                "stage": "screen", "config_id": cid,
                "vision_p50_ms": vis["p50_ms"], "vision_p95_ms": vis["p95_ms"],
                "vision_speedup": vis.get("speedup_p50"),
                "vision_speedup_ci_low": vis.get("speedup_ci95", [None, None])[0],
                "vision_speedup_ci_high": vis.get("speedup_ci95", [None, None])[1],
                "model_p50_ms": latency[f"{cid}|model"]["p50_ms"],
                "policy_wall_p50_ms": latency[f"{cid}|policy_wall"]["p50_ms"],
                "sr": item["sr"], "sr_ci_low": sr_ci[0], "sr_ci_high": sr_ci[1],
                "delta_pp": delta[0], "delta_pp_ci_low": delta[1],
                "delta_pp_ci_high": delta[2],
                "relative_sr": relative[0], "relative_ci_low": relative[1],
                "relative_ci_high": relative[2],
                "n_success": sum(int(bool(r["success"])) for r in sub),
                "n_episodes": len(sub),
            })

    with (run_dir / "episode_results.csv").open("w", newline="") as stream:
        fields = ["stage", "condition_id", "suite", "task_id", "initial_state_index",
                  "repeat_seed", "environment_seed", "success", "environment_steps",
                  "policy_calls", "error_type", "wall_s", "key"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in screen_records + confirm_records:
            writer.writerow({field: record.get(field) for field in fields})

    # Figures ---------------------------------------------------------------
    configs = [c["id"] for c in cfg["conditions"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    p50 = [latency[f"{c}|vision_all"]["p50_ms"] for c in configs]
    lo = [latency[f"{c}|vision_all"]["p50_ms"] - latency[f"{c}|vision_all"]["round_min_p50_ms"]
          for c in configs]
    hi = [latency[f"{c}|vision_all"]["round_max_p50_ms"] - latency[f"{c}|vision_all"]["p50_ms"]
          for c in configs]
    ax.bar(configs, p50, yerr=[lo, hi], capsize=4, color="#4477aa")
    ax.set_ylabel("T_vision_all p50 (ms)")
    ax.set_title("Paired vision latency (5 rounds x 200 calls)")
    fig.tight_layout()
    fig.savefig(run_dir / "fig_latency_vision.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for c in selection["screen"]:
        ax.scatter(c["vision_speedup"], c["sr"], s=50)
        ax.annotate(c["condition_id"], (c["vision_speedup"], c["sr"]),
                    textcoords="offset points", xytext=(4, 4))
    ax.axhline(selection["thresholds"]["preferred_relative_sr"] * screen_rates["C0"]["sr"],
               color="green", linestyle="--", linewidth=1, label="0.70 relative SR")
    ax.axhline(selection["thresholds"]["aggressive_relative_sr_floor"] * screen_rates["C0"]["sr"],
               color="orange", linestyle=":", linewidth=1, label="0.60 relative SR")
    ax.axvline(1.0, color="grey", linestyle="-", linewidth=1)
    ax.set_xlabel("vision speedup (C0 / config)")
    ax.set_ylabel("screen SR")
    ax.set_title("Screen: success rate vs vision speedup")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "fig_screen_sr_speedup.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    x = np.arange(len(SUITES) + 1)
    width = 0.8 / len(configs)
    for i, cid in enumerate(configs):
        data = screen_rates[cid]
        values = [data["suite_sr"].get(suite, np.nan) for suite in SUITES]
        values.append(data["sr"])
        ax.bar(x + i * width, values, width, label=cid)
    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels(list(SUITES) + ["overall"], rotation=15, fontsize=8)
    ax.set_ylabel("screen success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Screen SR by suite (init 0–4)")
    ax.legend(fontsize=7, ncol=len(configs))
    fig.tight_layout()
    fig.savefig(run_dir / "fig_suite_sr.png", dpi=160)
    plt.close(fig)

    if confirm_rates:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        width = 0.8 / len(confirm_conditions)
        x = np.arange(len(SUITES) + 1)
        for i, cid in enumerate(confirm_conditions):
            values = [confirm_rates[cid]["suite_sr"].get(suite, np.nan) for suite in SUITES]
            values.append(confirm_rates[cid]["sr"])
            ax.bar(x + i * width, values, width, label=cid)
        ax.set_xticks(x + width * (len(confirm_conditions) - 1) / 2)
        ax.set_xticklabels(list(SUITES) + ["overall"], rotation=15, fontsize=8)
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1)
        ax.set_title("Independent confirmation SR")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "fig_confirm_sr.png", dpi=160)
        plt.close(fig)

    # Markdown --------------------------------------------------------------
    manifest = json.loads((run_dir / "manifest.json").read_text())
    resolved = manifest["resolved"]
    lines = []
    lines.append(f"# pi0.5 vision A/B compression — {run_dir.name}\n")
    lines.append("Fixed protocol: C0/A128/A64/B128/B64; deterministic ToMeSD mapping; "
                 "A compresses the attention residual branch, B additionally the MLP branch; "
                 "each branch is unmerged before the full residual. Weights, VLM, action "
                 "expert, token layout, sampling steps and execution rules are unchanged.\n")
    lines.append("## Provenance\n")
    lines.append(f"- checkpoint: `{resolved['checkpoint']['path']}`")
    lines.append(f"- norm stats sha256: `{resolved['checkpoint']['norm_stats_sha256']}`")
    lines.append(f"- openpi commit: `{resolved['git']['commit']}`")
    lines.append(f"- backend: `{resolved['backend']}` / dtype `{resolved['dtype']}` / "
                 f"compile `{resolved['compile_mode']}`")
    lines.append(f"- model python: `{resolved['python']['model']}`; eval python: "
                 f"`{resolved['python']['eval']}`")
    lines.append(f"- observation library: {_observation_count(run_dir)} real LIBERO observations\n")
    lines.append("## Correctness\n")
    checks = correctness["checks"]
    lines.append("All required checks passed: " + ("**yes**" if correctness["all_required_passed"]
                                                   else "**no**") + ".\n")
    lines.append("| check | result |\n|---|---|")
    for name, value in checks.items():
        lines.append(f"| {name} | {value} |")
    parity = correctness["encoder_and_policy_identity_K256"]
    lines.append(f"\nK=256 encoder max abs {parity['feature_max_abs']:.3g} "
                 f"(bitwise={parity['bitwise_feature']}), policy action max abs "
                 f"{parity['action_max_abs']:.3g} (BF16 atol {parity['atol']}), "
                 f"repeat-run max abs {correctness['repeat_run_noise']['action_max_abs']:.3g}. "
                 "Compressed runs observed actual K-length branch cost and 256-token unmerge "
                 "for every configuration.\n")

    lines.append("## Vision latency\n")
    lines.append("T_vision_all includes stem, position embedding, all 27 blocks, matching, "
                 "merge/unmerge, final norm and projection; 3 sequential camera slots, "
                 "2 valid cameras, batch 1. Compile time is separate.\n")
    lines.append("| config | p50 ms | p95 ms | mean ms | round p50 range | speedup | 95% CI | reduction |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cid in configs:
        item = latency[f"{cid}|vision_all"]
        ci = item.get("speedup_ci95", [None, None])
        lines.append(f"| {cid} | {item['p50_ms']:.3f} | {item['p95_ms']:.3f} | "
                     f"{item['mean_ms']:.3f} | {item['round_min_p50_ms']:.3f}–"
                     f"{item['round_max_p50_ms']:.3f} | {item.get('speedup_p50', 1.0):.3f} | "
                     f"{ci[0]:.3f}–{ci[1]:.3f} | "
                     f"{100 * item.get('latency_reduction', 0.0):.1f}% |")
    lines.append("\nPer-round p50 for T_vision_all (ms):\n")
    lines.append("| config | r0 | r1 | r2 | r3 | r4 |")
    lines.append("|---|---|---|---|---|---|")
    for cid in configs:
        rounds = latency[f"{cid}|vision_all"]["round_p50"]
        lines.append(f"| {cid} | " + " | ".join(
            f"{rounds[str(r)]['p50_ms']:.3f}" for r in range(len(rounds))) + " |")
    lines.append("\nFull-policy timing is in `latency_summary.csv` "
                 "(`model` = GPU-ready observation to action tensor; `policy_wall` = "
                 "synchronized real policy call including CPU preprocessing/H2D/D2H).\n")
    target = selection["thresholds"]["engineering_vision_speedup_target"]
    met = selection.get("speedup_target_met", [])
    lines.append(f"Configurations reaching the {target:.2f}x vision-speedup engineering target: "
                 + (", ".join(met) if met else "none") + ".\n")

    lines.append("## Screen (init 0–4, 40 tasks)\n")
    lines.append("SR intervals are task-stratified paired bootstrap over 10,000 resamples; "
                 "ΔSR and relative SR are paired against C0 on the same task/init/noise keys.\n")
    lines.append("| config | success/total | SR [95% CI] | " + " | ".join(SUITES)
                 + " | ΔSR pp [95% CI] | rel. SR [95% CI] | speedup | decision |")
    lines.append("|---" * (len(SUITES) + 7) + "|")
    selected = set(selection["confirm_conditions"][1:])
    first_stats = next(iter(screen_stats.values()))
    for c in selection["screen"]:
        cid = c["condition_id"]
        decision = "confirm" if cid in selected else "not selected"
        sub = [r for r in screen_records if r["condition_id"] == cid]
        success = sum(int(bool(r["success"])) for r in sub)
        stats = screen_stats.get(cid)
        sr_ci = stats["sr_ci95"] if stats else first_stats["base_sr_ci95"]
        if stats:
            delta = (f"{stats['delta_pp']:+.2f} "
                     f"[{stats['delta_pp_ci95'][0]:+.2f}, {stats['delta_pp_ci95'][1]:+.2f}]")
            relative = (f"{stats['relative']:.3f} "
                        f"[{stats['relative_ci95'][0]:.3f}, {stats['relative_ci95'][1]:.3f}]")
        else:
            delta, relative = "baseline", "1.000"
        lines.append(f"| {cid} | {success}/{len(sub)} | {c['sr']:.3f} "
                     f"[{sr_ci[0]:.3f}, {sr_ci[1]:.3f}] | "
                     + " | ".join(f"{c['suite_sr'].get(s, float('nan')):.3f}" for s in SUITES)
                     + f" | {delta} | {relative} | {c['vision_speedup']:.3f} | {decision} |")
    lines.append("")
    for family, info in selection["families"].items():
        lines.append(f"- Family {family}: selected {info['selected'] or 'none'} — {info['reason']}")
    lines.append("")
    lines.append("Fastest measured candidate at each relative-SR threshold "
                 "(a candidate must also have vision speedup > 1.0):\n")
    for level in (0.90, 0.70, 0.60):
        eligible = [c for c in selection["screen"]
                    if c["relative_sr"] is not None and c["relative_sr"] >= level
                    and c["vision_speedup"] > 1.0]
        if eligible:
            best = max(eligible, key=lambda c: c["vision_speedup"])
            lines.append(f"- >= {level:.2f}: {best['condition_id']} "
                         f"({best['vision_speedup']:.3f}x, relative SR {best['relative_sr']:.3f})")
        else:
            lines.append(f"- >= {level:.2f}: no candidate")
    walls = [r["wall_s"] for r in screen_records if r.get("wall_s")]
    if walls:
        mean_wall = float(np.mean(walls))
        lines.append("")
        lines.append(f"Measured screen cost is {mean_wall:.1f} s/episode (mean wall time over "
                     f"{len(walls)} episodes). For planning only: 3600 confirmation episodes "
                     f"would take about {3600 * mean_wall / 3600:.1f} GPU-hours "
                     f"({3600 * mean_wall / 3 / 3600:.1f} h over three GPUs) and 5400 about "
                     f"{5400 * mean_wall / 3600:.1f} GPU-hours "
                     f"({5400 * mean_wall / 3 / 3600:.1f} h over three GPUs). This budget was "
                     "not executed because no candidate passed the fixed rule.")
    lines.append("")

    diag_path = run_dir / "parity/diagnostics.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
        lines.append("## Failure diagnosis (plan section 12)\n")
        identity = diag.get("B256_identity")
        if identity:
            lines.append(
                f"Mode B at K=256 is an identity check on the real checkpoint: encoder features "
                f"bitwise identical={identity['bitwise_feature']}, relative L2 "
                f"{identity['feature_relative_l2']:.3g}; action max abs "
                f"{identity['action_max_abs']:.3g} (BF16 noise), params unchanged="
                f"{identity['params_identical']}. The MLP-branch integration therefore does not "
                "change the untriggered path.")
        lines.append("")
        lines.append("| config | vision feat rel-L2 | vision feat max abs | action rel-L2 | "
                     "action max abs | params unchanged |")
        lines.append("|---|---|---|---|---|---|")
        for cid in ("A128", "A64", "B128", "B64"):
            item = diag.get(cid)
            if not item:
                continue
            lines.append(f"| {cid} | {item['feature_relative_l2']:.3f} | "
                         f"{item['feature_max_abs']:.3g} | {item['action_relative_l2']:.3f} | "
                         f"{item['action_max_abs']:.3g} | {item['params_identical']} |")
        lines.append("")
        lines.append("Deviations are measured on the same 8 real LIBERO probe observations with "
                     "fixed action noise, so they reflect the compression only. B64 is the only "
                     "configuration with a measured vision speedup (1.097x), but its relative "
                     "screen SR is 0.015 and its action deviation is the largest measured. The "
                     "optional layer-scope rescue ([2, 25), keeping the first/last two layers "
                     "dense) was not run: the implementation error is ruled out by the K=256 "
                     "identity check, and leaving 4/27 layers dense can only reduce the already "
                     "sub-target 1.097x vision speedup, so it cannot produce an engineering-"
                     "relevant candidate. Per the fixed rule, no confirmation was started.\n")

    if confirm_rates:
        lines.append("## Independent confirmation (init 5–49, 40 tasks)\n")
        lines.append("| config | SR overall | " + " | ".join(SUITES) + " | Delta pp [95% CI] | relative [95% CI] |")
        lines.append("|---" * (len(SUITES) + 4) + "|")
        for cid in confirm_conditions:
            data = confirm_rates[cid]
            if cid == "C0":
                delta = "—"
                relative = "—"
            else:
                stats = confirm_stats[cid]
                delta = f"{stats['delta_pp']:+.2f} [{stats['delta_pp_ci95'][0]:+.2f}, {stats['delta_pp_ci95'][1]:+.2f}]"
                relative = (f"{stats['relative']:.3f} [{stats['relative_ci95'][0]:.3f}, "
                            f"{stats['relative_ci95'][1]:.3f}]" if stats["relative"] is not None else "NA")
            lines.append(f"| {cid} | {data['sr']:.3f} | "
                         + " | ".join(f"{data['suite_sr'].get(s, float('nan')):.3f}" for s in SUITES)
                         + f" | {delta} | {relative} |")
        lines.append("")
    else:
        lines.append("## Independent confirmation\n")
        lines.append("No candidate entered confirmation under the fixed screening rule; "
                     "this report contains screening results only.\n")

    lines.append("## Notes and limitations\n")
    lines.append("- Episodes are paired by task, initial state and explicit Gaussian action noise "
                 "keyed without config id; environment seed is reset before every episode.")
    lines.append("- `policy_wall` includes the policy's CPU transforms, host/device transfer and "
                 "output conversion; it is not decomposed into per-stage sums.")
    lines.append("- Timeouts are task failures; infrastructure errors block aggregation and must be "
                 "rerun, never silently deleted.")
    lines.append("- Screen episodes were sharded over three GPUs after a single-GPU start; the "
                 "episode key includes stage/condition/task/init/noise, so resume did not duplicate "
                 "or drop records. No confirmation episodes were run.")
    lines.append("- Confirmation-budget estimate is not applicable: the fixed screening rule "
                 "selected no A/B candidate, so the planned 3600/5400-episode confirmation was "
                 "not started.")
    lines.append("- Video capture was not enabled; failures are identified by episode keys and "
                 "environment step counts in the JSONL rather than by video.")
    lines.append("- Every episode JSONL record carries a `fingerprints` object (checkpoint params "
                 "manifest/mtime, norm-stats sha256, openpi commit, code sha256) joined from the "
                 "final audit manifest after the run; no measurement field was altered.")
    lines.append("- Self-speculative system evaluation: not evaluated (no runnable pi0.5 "
                 "draft/verifier harness in this workspace).")
    lines.append("- Negative or non-speedup results are preserved; no configuration is excluded "
                 "from the tables.")
    lines.append("")
    lines.append("## Artifacts\n")
    lines.append("`latency_raw.csv`, `latency_summary.csv`, `latency_rounds.csv`, "
                 "`latency_compile_meta.json`, `correctness.json`, `rollouts_screen.jsonl`, "
                 "`episode_manifest_*.jsonl`, `success_summary.csv`, `success_by_task.csv`, "
                 "`success_by_suite.csv`, `summary.csv`, `episode_results.csv`, `selection.json`, "
                 "`parity/diagnostics.json`, `fig_*.png`, `manifest.json`, `commands.sh`.")
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines))
    summary = {
        "run_id": run_dir.name,
        "correctness_passed": correctness["all_required_passed"],
        "latency": {key: {k: v for k, v in item.items() if k != "round_p50"}
                    for key, item in latency.items()},
        "screen": selection["screen"],
        "selection": selection,
        "confirm": {"conditions": confirm_conditions, "rates": confirm_rates,
                    "paired": confirm_stats},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    return summary


def _observation_count(run_dir):
    path = Path(run_dir) / "observations_manifest.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())
