#!/usr/bin/env python3
"""Independent CPU recomputation of completed draft confirmation metrics."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORTS = ROOT / "reports/ease_insights_20261103"


def load(path, evidence, jsonl=False):
    raw = path.read_bytes()
    evidence.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    return [json.loads(line) for line in raw.splitlines() if line] if jsonl else json.loads(raw)


def close(a, b):
    return a is b if a is None or b is None else math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def key(row):
    return row["mode"], row["source_suite"], row["target_suite"], row["task_id"], row["init_id"]


def aggregate(rows):
    names = ["requests", "verified_rounds", "full_rounds", "accepted_actions", "rejection_full_rounds",
             "periodic_full_rounds", "initial_full_rounds", "zero_action_draft_rounds",
             "env_steps", "policy_total_ms", "client_rpc_total_ms", "wall_seconds"]
    d = {name: sum(row[name] for row in rows) for name in names}
    d.update(n=len(rows), successes=sum(row["success"] for row in rows),
             acceptance_rate=d["accepted_actions"] / (12 * d["verified_rounds"]) if d["verified_rounds"] else None,
             full_round_fraction=d["full_rounds"] / d["requests"],
             rejection_full_fraction=d["rejection_full_rounds"] / d["requests"],
             policy_ms_per_executed_action=d["policy_total_ms"] / d["env_steps"],
             client_rpc_ms_per_executed_action=d["client_rpc_total_ms"] / d["env_steps"],
             mean_episode_wall_seconds=d["wall_seconds"] / len(rows))
    rates = [r["accepted_actions"] / (12 * r["verified_rounds"]) for r in rows if r["verified_rounds"]]
    d["case_mean_acceptance_rate"] = sum(rates) / len(rates) if rates else None
    return d


def main():
    inputs = []
    directory = REPORTS / "draft_confirmation_analysis"
    reported = load(directory / "episodes.json", inputs)
    summaries = load(directory / "summary.json", inputs)
    subsets = load(directory / "latency_subsets.json", inputs)
    analysis_audit = load(directory / "audit.json", inputs)
    statistics = load(REPORTS / "draft_statistics/statistics.json", inputs)
    discovery = load(REPORTS / "draft_discovery_analysis/audit.json", inputs)
    selection = load(Path(statistics["selection_path"]), inputs)
    protocol = load(Path(statistics["protocol_path"]), inputs)
    assert hashlib.sha256((directory / "episodes.json").read_bytes()).hexdigest() == statistics["input_sha256"]
    assert hashlib.sha256(Path(statistics["selection_path"]).read_bytes()).hexdigest() == statistics["selection_sha256"]
    assert hashlib.sha256(Path(statistics["protocol_path"]).read_bytes()).hexdigest() == statistics["protocol_sha256"]
    assert len(reported) == len({key(row) for row in reported}) == 400
    assert discovery["complete_conditions"] == 360 and analysis_audit["complete_conditions"] == 400
    assert not analysis_audit["uncompleted"] and not analysis_audit["initial_pairing_mismatches"]
    per_run = defaultdict(list)
    for row in reported:
        per_run[row["run"]].append(row)
    assert len(per_run) == 10
    recomputed, first_hashes = [], defaultdict(list)
    native_outer_policy_total = 0.0
    request_count = 0
    for run, expected_rows in sorted(per_run.items()):
        requests = load(Path(run) / "requests.jsonl", inputs, jsonl=True)
        request_count += len(requests)
        by_case = defaultdict(list)
        for request in requests:
            by_case[(request["suite"], request["task_id"], request["init_id"])].append(request)
        expected_keys = {(r["target_suite"], r["task_id"], r["init_id"]) for r in expected_rows}
        assert set(by_case) == expected_keys and len(expected_keys) == 40
        for old in expected_rows:
            result = load(Path(old["result"]), inputs)
            native = result["episode"]
            c = old["target_suite"], old["task_id"], old["init_id"]
            reqs = by_case[c]
            assert result["status"] == "complete" and result["success"] == bool(old["success"])
            assert (result["mode"], result["source_suite"], result["suite"], result["task_id"], result["init_id"]) == key(old)
            assert [r["replan_id"] for r in reqs] == list(range(len(reqs)))
            assert len(reqs) == native["infer_calls"]
            assert all(r["mode"] == old["mode"] and r["source_suite"] == old["source_suite"] for r in reqs)
            verified = [r for r in reqs if r["route_calls"]["verify"] == 1]
            full = [r for r in reqs if r["route_calls"]["full"] == 1]
            draft = [r for r in reqs if r["route_calls"]["draft"] == 1]
            if old["mode"] == "draft":
                assert not full and not verified and len(draft) == len(reqs)
                assert all(r["noise_used"] is False for r in reqs)
            if old["mode"] == "full":
                assert not verified and not draft and len(full) == len(reqs)
            if old["mode"] == "flash":
                assert len(verified) == len(draft) and len(full) + len(draft) == len(reqs)
            assert all(math.isfinite(r["policy_wall_ms"]) and r["policy_wall_ms"] > 0 for r in reqs)
            row = dict(mode=old["mode"], source_suite=old["source_suite"], target_suite=c[0], task_id=c[1], init_id=c[2],
                       success=int(result["success"]), requests=len(reqs), verified_rounds=len(verified), full_rounds=len(full),
                       accepted_actions=sum(r["timing"]["accepted_prefix_len"] for r in verified),
                       rejection_full_rounds=sum(r["timing"].get("ease_rejection_full", 0) for r in reqs),
                       periodic_full_rounds=sum(r["timing"].get("ease_periodic_full", 0) for r in reqs),
                       initial_full_rounds=sum(r["timing"].get("ease_initial_full", 0) for r in reqs),
                       zero_action_draft_rounds=sum(r["timing"]["accepted_prefix_len"] == 0 for r in verified),
                       env_steps=native["executed_action_count"], policy_total_ms=sum(r["policy_wall_ms"] for r in reqs),
                       client_rpc_total_ms=result["client_roundtrip_sum_ms"], wall_seconds=result["wall_seconds"])
            assert all(close(value, old[name]) for name, value in row.items() if isinstance(value, (float, int)))
            recomputed.append(row)
            native_outer_policy_total += native["policy_time_sum_ms"]
            first_hashes[c].append((reqs[0]["input_sha256"], reqs[0]["noise_sha256"], reqs[0]["noise_seed"]))
    assert len(first_hashes) == 80 and all(len(v) == 5 and len(set(v)) == 1 for v in first_hashes.values())
    cells = []
    for reference in summaries:
        cell_key = reference["mode"], reference["source_suite"], reference["target_suite"]
        rows = [r for r in recomputed if (r["mode"], r["source_suite"], r["target_suite"]) == cell_key]
        assert {(r["task_id"], r["init_id"]) for r in rows} == {(t, i) for t in range(10) for i in range(21, 25)}
        derived = aggregate(rows)
        assert all(close(value, reference[name]) for name, value in derived.items() if name != "case_mean_acceptance_rate")
        cells.append(dict(mode=cell_key[0], source_suite=cell_key[1], target_suite=cell_key[2], **derived))
    for reference in subsets:
        rows = [r for r in recomputed if all(r[k] == reference[k] for k in ["mode", "source_suite", "target_suite"])]
        if reference["subset"] == "first_task_init21":
            rows = [r for r in rows if r["init_id"] == 21]
        elif reference["subset"] == "repeat_task_init22_24":
            rows = [r for r in rows if r["init_id"] in (22, 23, 24)]
        else:
            assert reference["subset"] == "all4_init21_24"
        derived = aggregate(rows)
        for name in ["n", "successes", "policy_ms_per_executed_action", "client_rpc_ms_per_executed_action", "mean_episode_wall_seconds"]:
            assert close(derived[name], reference[name]), (reference["subset"], name)
    primary = []
    for pair in statistics["pairs"]:
        target, foreign = pair["target_suite"], pair["foreign_source_suite"]
        match = {(r["task_id"], r["init_id"]): r for r in recomputed if r["mode"] == "draft" and r["target_suite"] == target and r["source_suite"] == target}
        other = {(r["task_id"], r["init_id"]): r for r in recomputed if r["mode"] == "draft" and r["target_suite"] == target and r["source_suite"] == foreign}
        assert set(match) == set(other) and len(match) == 40
        wins = sum(match[k]["success"] and not other[k]["success"] for k in match)
        losses = sum(other[k]["success"] and not match[k]["success"] for k in match)
        n = wins + losses
        pvalue = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n
        expected = pair["primary"]
        assert wins == expected["matching_only_successes"] and losses == expected["foreign_only_successes"]
        assert math.isclose(pvalue, expected["exact_mcnemar_one_sided_p"], rel_tol=1e-12, abs_tol=0.0)
        primary.append(dict(target_suite=target, foreign_source=foreign, wins=wins, losses=losses,
                            one_sided_exact_p=pvalue, reported_holm_p=expected["holm_adjusted_one_sided_p"],
                            case_ci95=expected["paired_case_bootstrap_ci95"], task_cluster_ci95=expected["task_cluster_bootstrap_ci95"]))
    running_max = 0.0
    for i, row in enumerate(sorted(primary, key=lambda r: r["one_sided_exact_p"])):
        running_max = max(running_max, min(1.0, (len(primary) - i) * row["one_sided_exact_p"]))
        assert math.isclose(running_max, row["reported_holm_p"], rel_tol=1e-12)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="independent_CPU_metrics_verified",
                  discovery_completed=discovery["complete_conditions"], confirmation_completed=400,
                  request_rows_checked=request_count, conditions=10, same_initial_input_and_noise_groups=80,
                  primary=primary, condition_metrics=cells,
                  all_thirty_fixed_latency_subsets_recomputed=True,
                  source_scope=dict(policy="requests.jsonl policy_wall_ms: complete original_infer call in draft/server.py; includes zero-action rejection requests, excludes wrapper noise/hash/logging outside this timer",
                                    RPC="episode result client_roundtrip_sum_ms includes client roundtrip",
                                    denominator="native episode executed_action_count; no failure filtering",
                                    outer_policy_summary_ms=native_outer_policy_total,
                                    original_infer_sum_ms=sum(r["policy_total_ms"] for r in recomputed),
                                    note="Native episode policy_time_sum_ms also includes outer wrapper work; it is a different timing scope and must not be substituted silently."),
                  boundaries=["Frozen discovery-selected directions replicated on new init21-24 of the same ten tasks per suite; this is not unseen-task or all-trained-draft generalization.",
                              "Exact McNemar/Holm is the preregistered case-level family of two; repeated initial states share tasks. Task-cluster intervals over ten tasks are robustness information.",
                              "Secondary effects and their intervals are descriptive; case-average acceptance/cost contrasts differ from pooled numerator/denominator metrics.",
                              "Pure draft has no verify or Full fallback; acceptance is N/A. Its generated paired noise is unused by the pure head.",
                              "Repeat-task init22-24 is a fixed subset, not proven fully warm. Real trajectories, success rates, and external resource load differ; these are observed service costs, not a controlled deployment speed benchmark.",
                              "This audit recomputes route/count/timing statistics from 10 request logs and400 small result JSONs; it does not revalidate all array/controller hashes or numerical action parity."])
    inputs.append(dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), role="audit_source"))
    (HERE / "input_hashes.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "discovery_completed", "confirmation_completed", "request_rows_checked", "primary", "source_scope"]}))


if __name__ == "__main__":
    main()
