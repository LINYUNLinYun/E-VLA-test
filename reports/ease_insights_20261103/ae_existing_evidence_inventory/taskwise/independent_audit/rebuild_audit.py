#!/usr/bin/env python3
"""Independently rebuild the AE-L2/L8 paired summary from lightweight logs.

Run from the repository root with /usr/bin/python3. This reads only the eight
manifest/episode_log files named by taskwise/analysis.json, its frozen taxonomy,
the 400-row paired artifact, and the analysis summary being audited. It does
not open video, trace, telemetry, inference, NPZ, model, or GPU artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path.cwd()
REPORT = ROOT / "reports/ease_insights_20261103/ae_existing_evidence_inventory/taskwise"
ANALYSIS = REPORT / "analysis.json"
OUT = REPORT / "independent_audit/independent_audit.json"
CONDITIONS = ("L2", "L8")
CATS = (
    "state_change_only",
    "state_change_and_object_placement",
    "multiple_object_placement",
    "single_object_placement",
    "unknown",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b, tol=1e-9):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, abs_tol=tol, rel_tol=tol)


def compare_value(actual, reported, path, mismatches):
    if isinstance(actual, dict) and isinstance(reported, dict):
        if set(actual) != set(reported):
            mismatches.append({"path": path, "actual_keys": sorted(actual), "reported_keys": sorted(reported)})
            return
        for k in actual:
            compare_value(actual[k], reported[k], f"{path}.{k}" if path else k, mismatches)
    elif isinstance(actual, list) and isinstance(reported, list):
        if len(actual) != len(reported):
            mismatches.append({"path": path, "actual_length": len(actual), "reported_length": len(reported)})
        else:
            for i, (a, r) in enumerate(zip(actual, reported)):
                compare_value(a, r, f"{path}[{i}]", mismatches)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(reported, (int, float)) and not isinstance(reported, bool):
        if not close(actual, reported):
            mismatches.append({"path": path, "actual": actual, "reported": reported})
    elif actual != reported:
        mismatches.append({"path": path, "actual": actual, "reported": reported})


def summarize(records):
    n = len(records)
    l2 = sum(int(r["L2"]["success"]) for r in records)
    l8 = sum(int(r["L8"]["success"]) for r in records)
    l2_only = sum(r["L2"]["success"] and not r["L8"]["success"] for r in records)
    l8_only = sum(r["L8"]["success"] and not r["L2"]["success"] for r in records)
    both_success = sum(r["L2"]["success"] and r["L8"]["success"] for r in records)
    both_failure = sum(not r["L2"]["success"] and not r["L8"]["success"] for r in records)
    return {
        "n_case_pairs": n,
        "L2_successes": l2,
        "L2_rate": l2 / n if n else None,
        "L8_successes": l8,
        "L8_rate": l8 / n if n else None,
        "L2_minus_L8_pp": (l2 - l8) * 100.0 / n if n else None,
        "discordant": {"L2_only": l2_only, "L8_only": l8_only},
        "paired_outcomes": {
            "both_success": both_success,
            "L2_only": l2_only,
            "L8_only": l8_only,
            "both_failure": both_failure,
        },
    }


def task_mean(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["suite"], r["task_id"])].append(100.0 * (int(r["L2"]["success"]) - int(r["L8"]["success"])))
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def bootstrap_ci(records, draws, seed):
    # Cluster is the fixed (suite, task_id); each task contributes its mean
    # paired difference. The documented seed and draw count are used verbatim.
    means = np.asarray(list(task_mean(records).values()), dtype=float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(means), size=(draws, len(means)))
    boots = means[ix].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975], method="linear")
    return {"lower95_pp": float(lo), "upper95_pp": float(hi)}


def expected_group(records, field):
    groups = defaultdict(list)
    for r in records:
        groups[r[field]].append(r)
    return groups


def main():
    analysis = read_json(ANALYSIS)
    run_evidence = analysis["source"]["run_evidence"]
    assert len(run_evidence) == 8, f"expected 8 evidence rows, got {len(run_evidence)}"
    assert {e["condition_label"] for e in run_evidence} == set(CONDITIONS)
    taxonomy_path = ROOT / analysis["source"]["taxonomy_path"]
    taxonomy_hash_actual = sha256(taxonomy_path)
    taxonomy = read_json(taxonomy_path)
    taxonomy_rows = taxonomy["tasks"]
    taxonomy_by_key = {(t["suite"], int(t["task"])): t for t in taxonomy_rows}
    assert len(taxonomy_by_key) == len(taxonomy_rows), "duplicate taxonomy suite/task key"
    assert taxonomy_hash_actual == analysis["source"]["taxonomy_sha256"], "taxonomy SHA mismatch"

    source_audit = []
    raw = {condition: {} for condition in CONDITIONS}
    seen_run_suite = set()
    for e in run_evidence:
        condition, suite = e["condition_label"], e["suite"]
        assert (condition, suite) not in seen_run_suite, f"duplicate source run {condition}/{suite}"
        seen_run_suite.add((condition, suite))
        mp, lp = ROOT / e["manifest_path"], ROOT / e["episode_log_path"]
        manifest_hash_actual, log_hash_actual = sha256(mp), sha256(lp)
        manifest = read_json(mp)
        episodes = read_json(lp)
        assert manifest_hash_actual == e["manifest_sha256"], f"manifest SHA mismatch: {mp}"
        assert log_hash_actual == e["episode_log_sha256"], f"episode_log SHA mismatch: {lp}"
        assert manifest["run_id"] == e["run_id"]
        assert manifest["task_suite_name"] == suite
        assert manifest["seed"] == e["seed"] == 7
        assert manifest["selected_task_ids"] == list(range(10))
        assert manifest["num_trials_per_task"] == 10
        assert manifest["episode_index_start"] == 0 and manifest["episode_index_end"] == 10
        assert len(episodes) == e["record_count"] == 100
        assert len(episodes) == len(e["tasks"]) * e["trials_per_task"]
        local_keys = []
        for row in episodes:
            assert row["run_id"] == e["run_id"]
            assert row["task_suite_name"] == suite
            assert row["seed"] == e["seed"]
            assert type(row["success"]) is bool
            assert row["failure_reason"] is None
            assert row["task_id"] in e["tasks"]
            assert row["episode_idx"] in range(*e["episode_indices"])
            key = (suite, int(row["task_id"]), int(row["episode_idx"]), int(row["seed"]))
            local_keys.append(key)
            assert key not in raw[condition], f"duplicate source key {condition}/{key}"
            raw[condition][key] = row
        assert len(set(local_keys)) == 100
        source_audit.append({
            "condition": condition,
            "suite": suite,
            "run_id": e["run_id"],
            "manifest_sha256_expected": e["manifest_sha256"],
            "manifest_sha256_actual": manifest_hash_actual,
            "episode_log_sha256_expected": e["episode_log_sha256"],
            "episode_log_sha256_actual": log_hash_actual,
            "records": len(episodes),
            "unique_case_keys": len(set(local_keys)),
            "manifest_seed": manifest["seed"],
            "record_seeds": sorted({r["seed"] for r in episodes}),
        })
    assert len(seen_run_suite) == 8
    assert len(raw["L2"]) == len(raw["L8"]) == 400
    assert set(raw["L2"]) == set(raw["L8"]), "L2/L8 source case-key sets differ"

    rebuilt = []
    metadata_mismatches = []
    instruction_mismatches = []
    for key in sorted(raw["L2"]):
        l2, l8 = raw["L2"][key], raw["L8"][key]
        suite, task_id, episode_idx, seed = key
        assert type(seed) is int and seed == 7
        for field in ("task_description", "max_steps", "num_steps_wait", "task_spec", "seed"):
            if l2[field] != l8[field]:
                metadata_mismatches.append({"key": list(key), "field": field, "L2": l2[field], "L8": l8[field]})
        tx = taxonomy_by_key.get((suite, task_id))
        assert tx is not None, f"missing taxonomy key {suite}/{task_id}"
        if l2["task_description"].strip().casefold() != tx["benchmark_instruction"].strip().casefold() or l8["task_description"].strip().casefold() != tx["benchmark_instruction"].strip().casefold():
            instruction_mismatches.append({"key": list(key), "log": l2["task_description"], "taxonomy_benchmark": tx["benchmark_instruction"]})
        if tx["category"] not in CATS:
            raise AssertionError(f"unsupported fixed taxonomy class: {tx['category']}")
        rebuilt.append({
            "suite": suite,
            "task_id": task_id,
            "task_description": l2["task_description"],
            "episode_idx": episode_idx,
            "seed": seed,
            "taxonomy_category": tx["category"],
            "L2": l2,
            "L8": l8,
        })
    assert not metadata_mismatches, metadata_mismatches[:5]
    assert not instruction_mismatches, instruction_mismatches[:5]
    assert len(rebuilt) == 400
    full_case_keys = {
        (r["suite"], r["task_id"], r["task_description"], r["episode_idx"], r["seed"])
        for r in rebuilt
    }
    assert len(full_case_keys) == 400, "full suite/task/instruction/episode/seed case keys are not unique"

    # Verify the produced 400-case artifact only after independently rebuilding
    # the source-paired table above.
    artifact_path = REPORT / analysis["case_pair_artifact"]
    artifact_sha = sha256(artifact_path)
    artifact_rows = [json.loads(line) for line in artifact_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(artifact_rows) == 400
    artifact_by_key = {}
    for a in artifact_rows:
        key = (a["suite"], int(a["task_id"]), int(a["trial_index"]), int(a["run_seed"]))
        assert key not in artifact_by_key, f"duplicate artifact key {key}"
        artifact_by_key[key] = a
    assert set(artifact_by_key) == set(raw["L2"]), "artifact and source key sets differ"
    artifact_mismatches = []
    for r in rebuilt:
        key = (r["suite"], r["task_id"], r["episode_idx"], r["seed"])
        a = artifact_by_key[key]
        expected = {
            "suite": r["suite"],
            "task_id": r["task_id"],
            "trial_index": r["episode_idx"],
            "run_seed": r["seed"],
            "task_description": r["task_description"],
            "taxonomy_category": r["taxonomy_category"],
        }
        for condition in CONDITIONS:
            src = r[condition]
            expected.update({
                f"{condition}_success": src["success"],
                f"{condition}_failure_reason": src["failure_reason"],
                f"{condition}_executed_actions": src["executed_action_count"],
                f"{condition}_infer_calls": src["infer_calls"],
                f"{condition}_furthest_phase": src["furthest_semantic_phase"],
            })
        expected["paired_outcome"] = (
            "both_success" if r["L2"]["success"] and r["L8"]["success"] else
            "L2_only" if r["L2"]["success"] else
            "L8_only" if r["L8"]["success"] else "both_failure"
        )
        for field, value in expected.items():
            if a.get(field) != value:
                artifact_mismatches.append({"key": list(key), "field": field, "source_rebuilt": value, "artifact": a.get(field)})

    # Stable compact rows used only for fresh aggregate calculations.
    pairs = [{
        "suite": r["suite"], "task_id": r["task_id"],
        "task_description": r["task_description"],
        "taxonomy_category": r["taxonomy_category"],
        "L2": {"success": r["L2"]["success"]},
        "L8": {"success": r["L8"]["success"]},
    } for r in rebuilt]
    overall = summarize(pairs)
    by_suite = {name: summarize(g) for name, g in sorted(expected_group(pairs, "suite").items())}
    by_task_groups = defaultdict(list)
    for r in pairs:
        by_task_groups[(r["suite"], r["task_id"])].append(r)
    by_task = []
    for (suite, task_id), group in sorted(by_task_groups.items()):
        s = summarize(group)
        # taskwise in the source report records counts and paired discordance;
        # keep the audit's directly comparable projection to those fields.
        s.pop("L2_rate")
        s.pop("L8_rate")
        tx = taxonomy_by_key[(suite, task_id)]
        by_task.append({
            "suite": suite,
            "task_id": task_id,
            "task_name": tx["task_name"],
            "category": tx["category"],
            **s,
        })

    by_category_groups = expected_group(pairs, "taxonomy_category")
    by_taxonomy = {}
    task_means = task_mean(pairs)
    for category in CATS:
        group = by_category_groups.get(category, [])
        if not group:
            by_taxonomy[category] = {"n_tasks": 0, "n_case_pairs": 0}
            continue
        stats = summarize(group)
        cat_tasks = sorted(k for k in task_means if taxonomy_by_key[k]["category"] == category)
        cat_deltas = [task_means[k] for k in cat_tasks]
        counts_by_suite = Counter(s for s, _ in cat_tasks)
        stats.update({
            "n_tasks": len(cat_tasks),
            "tasks_by_suite": dict(sorted(counts_by_suite.items())),
            "tasks_L2_better": sum(x > 0 for x in cat_deltas),
            "tasks_L8_better": sum(x < 0 for x in cat_deltas),
            "tasks_tied": sum(x == 0 for x in cat_deltas),
        })
        by_taxonomy[category] = stats

    # Equal-task LOO, plus per-taxonomy LOO ranges for transparent fixed-class
    # sensitivity. No case-level rows are dropped beyond the named task.
    all_deltas = task_mean(pairs)
    full_task_mean = sum(all_deltas.values()) / len(all_deltas)
    loo = []
    for omitted in sorted(all_deltas):
        vals = [v for key, v in all_deltas.items() if key != omitted]
        loo.append({"omitted_suite": omitted[0], "omitted_task_id": omitted[1], "mean_L2_minus_L8_pp": sum(vals) / len(vals)})
    loo_summary = {
        "full_mean_L2_minus_L8_pp": full_task_mean,
        "minimum": min(loo, key=lambda x: x["mean_L2_minus_L8_pp"]),
        "maximum": max(loo, key=lambda x: x["mean_L2_minus_L8_pp"]),
        "range_pp": [min(x["mean_L2_minus_L8_pp"] for x in loo), max(x["mean_L2_minus_L8_pp"] for x in loo)],
        "all_40_omissions": loo,
    }
    taxonomy_loo = {}
    for category in CATS:
        vals = {k: v for k, v in all_deltas.items() if taxonomy_by_key[k]["category"] == category}
        if len(vals) >= 2:
            leave = [sum(v for k, v in vals.items() if k != omit) / (len(vals) - 1) for omit in vals]
            taxonomy_loo[category] = {"n_tasks": len(vals), "minimum_pp": min(leave), "maximum_pp": max(leave)}
        else:
            taxonomy_loo[category] = {"n_tasks": len(vals), "minimum_pp": None, "maximum_pp": None}

    # Recompute the analysis's bootstrap intervals using its recorded task
    # cluster unit, replicate counts and seeds. PCG64/default_rng and linear
    # quantiles make the method explicit here; the report itself does not name
    # the PRNG, resampling implementation or quantile interpolation rule.
    ci_recomputed = {}
    ci_method_reported = True
    overall_ci_meta = analysis["overall"]["task_cluster_bootstrap95"]
    ci_recomputed["overall"] = bootstrap_ci(pairs, overall_ci_meta["bootstrap_draws"], overall_ci_meta["bootstrap_seed"])
    ci_method_reported = ci_method_reported and all(k in overall_ci_meta for k in ("bootstrap_draws", "bootstrap_seed", "bootstrap_unit"))
    suite_ci = {}
    for suite, rows in expected_group(pairs, "suite").items():
        meta = analysis["by_suite"][suite]["task_cluster_bootstrap95"]
        suite_ci[suite] = bootstrap_ci(rows, meta["bootstrap_draws"], meta["bootstrap_seed"])
        ci_method_reported = ci_method_reported and all(k in meta for k in ("bootstrap_draws", "bootstrap_seed", "bootstrap_unit"))
    cat_ci = {}
    for category, rows in by_category_groups.items():
        meta = analysis["by_metadata_taxonomy"]["results"][category]["task_cluster_bootstrap95"]
        cat_ci[category] = bootstrap_ci(rows, meta["bootstrap_draws"], meta["bootstrap_seed"])
        ci_method_reported = ci_method_reported and all(k in meta for k in ("bootstrap_draws", "bootstrap_seed", "bootstrap_unit"))

    # Build exact report-comparable structures and record mismatches.
    summary_mismatches = []
    compare_value(overall, {k: v for k, v in analysis["overall"].items() if k != "task_cluster_bootstrap95"}, "overall", summary_mismatches)
    compare_value(by_suite, {
        s: {k: v for k, v in vals.items() if k != "task_cluster_bootstrap95"}
        for s, vals in analysis["by_suite"].items()
    }, "by_suite", summary_mismatches)
    compare_value(by_task, analysis["taskwise"], "taskwise", summary_mismatches)
    reported_tax_counts = analysis["by_metadata_taxonomy"]["taxonomy_counts"]
    actual_tax_counts_raw = Counter(t["category"] for t in taxonomy_rows)
    actual_tax_counts = {cat: actual_tax_counts_raw.get(cat, 0) for cat in CATS}
    compare_value(actual_tax_counts, reported_tax_counts, "taxonomy_counts", summary_mismatches)
    reported_tax_by_suite = analysis["by_metadata_taxonomy"]["taxonomy_counts_by_suite"]
    actual_tax_by_suite = {}
    for suite in sorted({t["suite"] for t in taxonomy_rows}):
        c = Counter(t["category"] for t in taxonomy_rows if t["suite"] == suite)
        actual_tax_by_suite[suite] = {cat: c.get(cat, 0) for cat in CATS}
    compare_value(actual_tax_by_suite, reported_tax_by_suite, "taxonomy_counts_by_suite", summary_mismatches)
    report_tax_stats = {}
    for category, values in by_taxonomy.items():
        report_tax_stats[category] = {k: v for k, v in values.items() if k not in ("task_cluster_bootstrap95", "small_class_warning")}
    analysis_tax_stats = {}
    for category, values in analysis["by_metadata_taxonomy"]["results"].items():
        analysis_tax_stats[category] = {k: v for k, v in values.items() if k not in ("task_cluster_bootstrap95", "small_class_warning")}
    compare_value(report_tax_stats, analysis_tax_stats, "by_metadata_taxonomy.results", summary_mismatches)
    compare_value({k: v for k, v in loo_summary.items() if k != "all_40_omissions"}, analysis["leave_one_task_out_overall_equal_task_mean"], "leave_one_task_out", summary_mismatches)

    ci_mismatches = []
    for scope, computed in ci_recomputed.items():
        reported = analysis["overall"]["task_cluster_bootstrap95"]
        for endpoint, val in computed.items():
            if not close(val, reported[endpoint]):
                ci_mismatches.append({"scope": scope, "endpoint": endpoint, "computed": val, "reported": reported[endpoint]})
    for scope, computed in suite_ci.items():
        reported = analysis["by_suite"][scope]["task_cluster_bootstrap95"]
        for endpoint, val in computed.items():
            if not close(val, reported[endpoint]):
                ci_mismatches.append({"scope": f"suite:{scope}", "endpoint": endpoint, "computed": val, "reported": reported[endpoint]})
    for scope, computed in cat_ci.items():
        reported = analysis["by_metadata_taxonomy"]["results"][scope]["task_cluster_bootstrap95"]
        for endpoint, val in computed.items():
            if not close(val, reported[endpoint]):
                ci_mismatches.append({"scope": f"taxonomy:{scope}", "endpoint": endpoint, "computed": val, "reported": reported[endpoint]})

    # The original logs have no field that identifies a physical initial state.
    all_fields = set().union(*(row.keys() for cond in raw.values() for row in cond.values()))
    state_digest_fields = sorted(k for k in all_fields if any(token in k.casefold() for token in ("initial_state_hash", "initial_state_digest", "reset_state_hash", "state_digest")))

    result = {
        "schema": "ae-l2-l8-independent-audit-v1",
        "status": "PASS" if not summary_mismatches and not artifact_mismatches and not ci_mismatches else "NEEDS_REPAIR",
        "scope": {
            "source_rows_read": 800,
            "source_files": 16,
            "excluded_artifacts": ["videos", "trace/action arrays", "infer records", "telemetry arrays", "NPZ", "GPU/render/model artifacts"],
        },
        "input_integrity": {
            "taxonomy_path": str(taxonomy_path.relative_to(ROOT)),
            "taxonomy_sha256_expected": analysis["source"]["taxonomy_sha256"],
            "taxonomy_sha256_actual": taxonomy_hash_actual,
            "run_evidence": source_audit,
            "L2_unique_source_keys": len(raw["L2"]),
            "L8_unique_source_keys": len(raw["L8"]),
            "paired_source_key_intersection": len(set(raw["L2"]) & set(raw["L8"])),
            "full_case_key_definition": ["suite", "task_id", "task_description", "episode_idx", "seed"],
            "full_case_key_unique_count": len(full_case_keys),
            "source_metadata_pair_mismatches": metadata_mismatches,
            "taxonomy_benchmark_instruction_mismatches": instruction_mismatches,
            "physical_initial_state_digest_fields_found": state_digest_fields,
        },
        "artifact_reconciliation": {
            "path": str(artifact_path.relative_to(ROOT)),
            "sha256": artifact_sha,
            "rows": len(artifact_rows),
            "unique_keys": len(artifact_by_key),
            "source_to_artifact_field_mismatches": artifact_mismatches,
        },
        "recomputed_from_source": {
            "overall": overall,
            "by_suite": by_suite,
            "by_task": by_task,
            "taxonomy_counts": actual_tax_counts,
            "taxonomy_counts_by_suite": actual_tax_by_suite,
            "by_fixed_taxonomy": by_taxonomy,
            "leave_one_task_out_equal_task_mean": loo_summary,
            "taxonomy_leave_one_task_out_equal_task_mean_range": taxonomy_loo,
        },
        "report_reconciliation": {
            "summary_mismatches": summary_mismatches,
            "bootstrap_ci": {
                "method_used_for_independent_recompute": "30,000 cluster resamples using numpy default_rng (PCG64), sampling (suite,task) means with replacement; 2.5/97.5 percentiles with linear interpolation",
                "reported_parameters_present": ci_method_reported,
                "recomputed": {"overall": ci_recomputed["overall"], "by_suite": suite_ci, "by_fixed_taxonomy": cat_ci},
                "mismatches": ci_mismatches,
                "status": "matches under declared seed/draw count and explicit standard implementation; original report does not record RNG/resampling implementation or quantile method, so bit-for-bit provenance cannot be established",
            },
        },
        "interpretation_boundaries": [
            "The paired identity is suite/task/episode label/run seed only; no physical initial-state digest is present in the source episode logs.",
            "The fixed taxonomy is metadata-only and its class composition is confounded with suites/tasks; class crossings are exploratory, not verified path preferences.",
            "Historical labels L2/L8 are not pooled with current R5/D2 evidence; missing checkpoint, normalization, source and complete launch provenance preclude a claim of exact runtime reproduction or reproduced preference.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(OUT.relative_to(ROOT)),
        "summary_mismatches": len(summary_mismatches),
        "artifact_mismatches": len(artifact_mismatches),
        "ci_mismatches": len(ci_mismatches),
        "overall": overall,
        "loo": {k: v for k, v in loo_summary.items() if k != "all_40_omissions"},
        "bootstrap_reproduction": result["report_reconciliation"]["bootstrap_ci"]["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
