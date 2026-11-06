"""Strict discovery audit and deterministic, prospectively specified selection.

This module only reads scientific outcomes. A missing/invalid receipt or missing
completed cost probe blocks freezing; valid policy failures remain outcomes.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import statistics
import numpy as np

from preparations.ease_insights_20261103.pipeline import file_lock
from .common import (ROOT, SUITES, GATE_SUITES, CATEGORIES, TASKS, TASK_CATEGORIES, CONDITIONS, DISCOVERY_PHASE, CONFIRMATION_PHASE, SCIENCE_FILES, SHARED_ANALYSIS_FILES, atomic_json, digest, read_manifest, sha, cases, validate_manifest, verify_analysis_sources)
from preparations.ease_insights_20261103.vlm.analyze import (
    cost_records, key, load_episodes, paired, summary,
)
from preparations.ease_insights_20261103.vlm.analyze_confirmation import audit_identities, unexpected_cases

CONDITION_IDS = list(CONDITIONS)
ANALYSIS_DEPENDENCIES = ["preparations/ease_insights_20261103/vlm_within_suite_v1/" + name for name in SCIENCE_FILES] + list(SHARED_ANALYSIS_FILES)


def read_selection(path):
    value = json.loads(Path(path).read_text())
    payload = copy.deepcopy(value)
    expected = payload.pop("selection_hash", None)
    if expected is None or digest(payload) != expected:
        raise ValueError("Frozen selection hash is missing or invalid")
    if (value.get("revision") != "ease-vlm-within-suite-selection-v1" or
            value.get("complete_discovery_episodes") != 432 or
            value.get("expected_discovery_episodes") != 432 or
            value.get("independent_validation_launched") is not False):
        raise ValueError("Selection is not a complete within-suite discovery decision")
    return value


def check_scope(root, manifest, stage):
    from .common import validate_output, stage_for
    validate_output(root, manifest)
    verify_analysis_sources(manifest)
    validate_manifest(manifest)
    if stage_for(manifest) != stage:
        raise ValueError("Analysis stage differs from frozen within-suite manifest")
    return manifest["discovery_inits" if stage == "discovery" else "confirmation_inits"]


def load_stage(root, stage):
    """Reuse the original full artifact validator, then audit original identities."""
    root = Path(root).resolve()
    manifest = read_manifest(root / "manifest.json", verify_sources=True)
    inits = check_scope(root, manifest, stage)
    view = copy.deepcopy(manifest)
    view["conditions"] = [c for c in view["conditions"] if c["id"] in manifest["planned_conditions"]]
    view["discovery_inits"] = inits
    scenes = cases(manifest, stage)
    records, missing, invalid, evidence = load_episodes(root, view)
    invalid.extend(audit_identities(root, manifest, records))
    invalid.extend(audit_local_identities(root, manifest, records))
    unexpected = unexpected_cases(root, manifest, scenes)
    expected = len(scenes) * len(manifest["planned_conditions"])
    return dict(root=root, manifest=manifest, scenes=scenes, records=records,
        missing=missing, invalid=invalid, unexpected=unexpected, episode_evidence=evidence,
        expected=expected, complete=len(records) == expected and not missing and not invalid and not unexpected)


def audit_local_identities(root, manifest, records):
    """Read-only check of the stronger new-runtime receipt/initial-input contract."""
    from .common import runtime_revision
    from preparations.idea1_v2_1.runtime_client import input_digest
    from preparations.idea1_v2_2.common import seed
    configs = {c["id"]: c for c in manifest["conditions"]}
    official = {(c["suite"], c["task"]): c for c in manifest["official_init_availability"]["cases"]}
    revision = runtime_revision(manifest)
    invalid = []
    for (cid, scene_key), record in list(records.items()):
        scene = record["case"]
        directory = root / cid / "episodes" / scene["suite"] / ("task%02d" % scene["task"]) / ("init%02d" % scene["init"])
        try:
            receipt = json.loads((directory / "episode_receipt.json").read_text())
            identity = json.loads((directory / "identity.json").read_text())
            result = json.loads((directory / "result.json").read_text())
            protocol = identity["protocol"]
            expected = official[scene["suite"], scene["task"]]
            if (type(receipt["success"]) is not bool or type(result["success"]) is not bool or
                    receipt.get("technical_runtime_revision") != revision or
                    "input_0000.npz" not in receipt["artifacts"] or
                    protocol.get("qk_reference_passed") is not True or protocol.get("config") != configs[cid] or
                    protocol.get("checkpoint_sha256") != manifest["checkpoints"]["pi0"]["sha256"] or
                    identity.get("env_seed") != 7 or identity.get("settling") != 10 or
                    identity.get("policy_action_limit") != 400 or result.get("policy_action_limit") != 400 or
                    result.get("settling_actions") != 10 or result.get("actions", 401) > 400 or
                    identity.get("bddl_sha256") != expected["bddl_sha256"] or identity.get("prompt") != expected["prompt"]):
                raise ValueError("Local runtime revision, QK/config/checkpoint, official environment, or input coverage differs")
            first = json.loads((directory / "infer.jsonl").read_text().splitlines()[0])
            first_input = directory / "input_0000.npz"
            if first["input_file_sha256"] != sha(first_input):
                raise ValueError("First request input artifact differs")
            with np.load(first_input, allow_pickle=False) as saved:
                actual = input_digest({k: saved[k] for k in saved.files if k not in ("instruction", "seed")}, str(saved["instruction"].item()))
                saved_seed = int(saved["seed"])
            initial = record["initial_gate"]
            reference_path = root / "initial_observations" / scene["suite"] / ("task%02d" % scene["task"]) / ("init%02d.json" % scene["init"])
            reference = json.loads(reference_path.read_text())
            if (actual != first["input_hash"] or actual != initial["input_hash"] or
                    saved_seed != first["seed"] or saved_seed != initial["seed"] or
                    saved_seed != seed(scene["suite"], scene["task"], scene["init"], 0) or
                    Path(initial["reference"]).resolve() != reference_path.resolve() or
                    any(reference.get(k) != initial[k] for k in ("manifest_hash", "case", "input_hash", "seed"))):
                raise ValueError("Initial observation/noise differs from its paired reference")
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid.append(dict(condition=cid, case=scene, error=str(error)))
            del records[cid, scene_key]
    return invalid


def completed_service_costs(root, manifest):
    """Equal service weighting; only completed runner services enter eligibility."""
    cost_view = copy.deepcopy(manifest)
    cost_view["suites"] = list(GATE_SUITES)
    result = cost_records(Path(root), cost_view)
    accepted, excluded = [], list(result["excluded_or_pending"])
    for service in result["services"]:
        path = Path(service["path"])
        try:
            receipt = json.loads((path.parent / "result.json").read_text())
            protocol = json.loads((path.parent / "protocol.json").read_text())
            if (receipt.get("complete") is not True or receipt.get("manifest_hash") != manifest["manifest_hash"] or
                    receipt.get("condition") != protocol["condition_id"] or
                    protocol.get("qk_reference_passed") is not True or
                    protocol.get("config") != next(c for c in manifest["conditions"] if c["id"] == receipt["condition"]) or
                    protocol.get("checkpoint_sha256") != manifest["checkpoints"]["pi0"]["sha256"] or
                    path.parent.parent.parent.name != receipt["condition"]):
                raise ValueError("Cost service has no matching completed runner receipt")
            accepted.append(dict(service, artifact_evidence=[dict(path=str(path.parent / name), sha256=sha(path.parent / name))
                for name in ("fixed_input_cost.json", "protocol.json", "gate.json", "cross_service.json", "result.json")]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            excluded.append(dict(path=str(path), reason=str(error)))
    ratios = {cid: statistics.median(s["full_over_policy_cycle"][cid] for s in accepted)
        for cid in CONDITION_IDS} if accepted else {}
    return dict(valid_services=len(accepted), services=accepted, excluded_or_pending=excluded,
        median_full_over_policy_cycle=ratios,
        pooling="Median of per-completed-service same-input mean Full / candidate three-request native-policy cycle costs",
        scope="Native policy inference only; excludes RPC and environment. Existing finite old four-suite bank only; it does not cover every new task/prompt. Shared GPU load may affect ratios; no closed-loop acceleration claim.")


def usable(count, full_count, n):
    # Exact integer comparisons implement >=.75 and >=Full-.15 without rounding.
    return count * 4 >= n * 3 and (full_count - count) * 20 <= n * 3


def direction_summary(a, b, records, scenes, tasks):
    result = paired(a, b, records, scenes)
    rows = [dict(task=t, **paired(a, b, records, [c for c in scenes if c["task"] == t])) for t in tasks]
    total = result["success_difference_count"]
    loo = [dict(task=r["task"], difference_count=total - r["success_difference_count"],
        difference=(total - r["success_difference_count"]) / (len(scenes) - r["n_paired"])) for r in rows]
    result.update(by_task=rows, positive_tasks=sum(r["success_difference_count"] > 0 for r in rows),
        leave_one_task_out=loo, every_leave_one_task_out_positive=all(r["difference_count"] > 0 for r in loo),
        interpretation="Discovery-selected comparison; all p-values are descriptive only")
    return result


def category_tasks(manifest, category):
    return [t for t in manifest["tasks"] if manifest["task_categories"][str(t)] == category]


def category_cases(manifest, scenes, category):
    return [c for c in scenes if manifest["task_categories"][str(c["task"])] == category]


def select_pair(manifest, records, costs, complete):
    if not complete:
        return dict(status="incomplete_discovery", selected=None, eligible_crossings=[])
    if not costs.get("valid_services") or any(cid not in costs.get("median_full_over_policy_cycle", {}) for cid in CONDITION_IDS):
        return dict(status="missing_valid_cost_probe", selected=None, eligible_crossings=[])
    scenes = [dict(suite=s, task=t, init=i) for s in manifest["suites"] for t in manifest["tasks"] for i in manifest["discovery_inits"]]
    candidates = sorted(cid for cid in manifest["planned_conditions"] if cid != "full_d10")
    counts = {cid: sum(records[cid, key(c)]["success"] for c in scenes) for cid in manifest["planned_conditions"]}
    n = len(scenes)
    gates = {}
    for cid in candidates:
        ratio = costs["median_full_over_policy_cycle"][cid]
        passed = usable(counts[cid], counts["full_d10"], n)
        gates[cid] = dict(successes=counts[cid], n=n, success_rate=counts[cid] / n,
            full_success_rate=counts["full_d10"] / n, usability_pass=passed,
            median_fixed_input_speed_ratio=ratio, cost_pass=ratio >= 1.15,
            qualified=passed and ratio >= 1.15)
    comparisons, crossings, eligible = [], [], []
    for a, b in itertools.combinations(candidates, 2):
        directions = {}
        for category in manifest["categories"]:
            selected_cases = category_cases(manifest, scenes, category)
            for better, worse in ((a, b), (b, a)):
                d = direction_summary(better, worse, records, selected_cases, category_tasks(manifest, category))
                d.update(category=category, better=better, worse=worse)
                d["direction_eligible"] = (10 * d["success_difference_count"] >= d["n_paired"] and
                    d["positive_tasks"] >= 2 and d["every_leave_one_task_out_positive"])
                directions[better, category] = d
            comparisons.append(dict(pair=[a, b], category=category, a_over_b=directions[a, category], b_over_a=directions[b, category]))
        for x, y in itertools.permutations(manifest["categories"], 2):
            da, db = directions[a, x], directions[b, y]
            gap_a, gap_b = da["success_difference"], db["success_difference"]
            entry = dict(pair=[a, b], paths=[a, b], directions=[
                dict(category=x, better=a, worse=b, discovery_difference=gap_a),
                dict(category=y, better=b, worse=a, discovery_difference=gap_b)],
                minimum_category_gap=min(gap_a, gap_b), sum_category_gaps=gap_a + gap_b,
                minimum_overall_success=min(counts[a], counts[b]) / n,
                ranking_counts=[min(da["success_difference_count"], db["success_difference_count"]),
                    da["success_difference_count"] + db["success_difference_count"], min(counts[a], counts[b])],
                lexical_tie_key=[a, b, x, y],
                qualified_paths=gates[a]["qualified"] and gates[b]["qualified"],
                directions_eligible=da["direction_eligible"] and db["direction_eligible"])
            entry["eligible"] = entry["qualified_paths"] and entry["directions_eligible"]
            crossings.append(entry)
            if entry["eligible"]:
                eligible.append(entry)
    # Both categories have exactly36 cases and all paths72; integer counts
    # implement the rate ranking exactly without floating-point tie drift.
    eligible.sort(key=lambda e: (*(-v for v in e["ranking_counts"]), *e["lexical_tie_key"]))
    return dict(status="selected" if eligible else "no_candidate", selected=eligible[0] if eligible else None,
        condition_eligibility=gates, all_pair_category_comparisons=comparisons, all_ordered_category_crossings=crossings,
        eligible_crossings=eligible, candidate_order=candidates, category_order=manifest["categories"],
        rule="Fixed protocol selection; >=.75 and >=Full-.15 overall; median completed-service finite-bank cost ratio>=1.15; opposite category gaps>=.10, >=2 positive tasks and every task LOO positive; deterministic four-level ranking",
        interpretation="Exploratory selection only. Metadata classes do not assert preference or difficulty. No candidate ends this792episode branch.")


def condition_summaries(manifest, records, scenes):
    groups = [("all", None, None, scenes)]
    groups += [("category", cat, None, category_cases(manifest, scenes, cat)) for cat in manifest["categories"]]
    groups += [("task", manifest["task_categories"][str(t)], t, [c for c in scenes if c["task"] == t]) for t in manifest["tasks"]]
    rows = []
    for level, category, task, selected_cases in groups:
        for cid in manifest["planned_conditions"]:
            values = [records[cid, key(c)] for c in selected_cases if (cid, key(c)) in records]
            aggregate = summary(values)
            actions = aggregate["actions"]
            aggregate["native_policy_ms_per_executed_action"] = aggregate["policy"]["total_ms"] / actions if actions else None
            aggregate["full_client_rpc_ms_per_executed_action"] = aggregate["rpc"]["total_ms"] / actions if actions else None
            rows.append(dict(level=level, suite="libero_90", category=category, task=task, condition=cid, expected_n=len(selected_cases), **aggregate))
    return rows


def analyze_discovery(root):
    audit = load_stage(root, "discovery")
    manifest, records = audit["manifest"], audit["records"]
    costs = completed_service_costs(audit["root"], manifest)
    decision = select_pair(manifest, records, costs, audit["complete"])
    return dict(revision="ease-vlm-within-suite-discovery-analysis-v1", manifest_hash=manifest["manifest_hash"],
        manifest_path=str(audit["root"] / "manifest.json"), protocol_evidence=manifest["protocol_evidence"],
        design_evidence=manifest["design_evidence"], expected_episodes=audit["expected"], valid_episodes=len(records),
        complete=audit["complete"], missing=audit["missing"], invalid=audit["invalid"], unexpected=audit["unexpected"],
        selection=decision, condition_summaries=condition_summaries(manifest, records, audit["scenes"]), fixed_input_cost=costs,
        evidence=dict(episodes=audit["episode_evidence"],
            cost_services=[dict(path=s["path"], sha256=s["sha256"], artifact_evidence=s["artifact_evidence"]) for s in costs["services"]],
            manifest=dict(path=str(audit["root"] / "manifest.json"), sha256=sha(audit["root"] / "manifest.json")),
            analysis_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))),
        prior_negative_results_retained=True, no_confirmation_outcomes_used=True)


def freeze(output, report):
    if (not report["complete"] or report["valid_episodes"] != 432 or report["expected_episodes"] != 432 or
            report["selection"]["status"] not in ("selected", "no_candidate")):
        raise ValueError("Selection requires all 432 strictly valid episodes and completed gated cost probes")
    payload = dict(revision="ease-vlm-within-suite-selection-v1", discovery_manifest_hash=report["manifest_hash"],
        complete_discovery_episodes=432, expected_discovery_episodes=432,
        protocol_evidence=report["protocol_evidence"], design_evidence=report["design_evidence"],
        selection=report["selection"], evidence=report["evidence"], independent_validation_launched=False)
    payload["selection_hash"] = digest(payload)
    output = Path(output).resolve()
    path = output / "selection.json"
    with file_lock(output / "selection.lock"):
        if path.exists():
            if read_selection(path) != payload:
                raise ValueError("Refusing to replace a different frozen selection")
        else:
            atomic_json(path, payload)
    return path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--freeze-selection", action="store_true")
    args = parser.parse_args()
    report = analyze_discovery(args.root)
    output = Path(args.output).resolve()
    atomic_json(output / "analysis.json", report)
    selection = freeze(output, report) if args.freeze_selection else None
    print(json.dumps(dict(complete=report["complete"], valid_episodes=report["valid_episodes"],
        selection_status=report["selection"]["status"], selection=None if selection is None else str(selection))))


if __name__ == "__main__":
    main()
