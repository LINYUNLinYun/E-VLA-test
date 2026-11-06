"""Independent confirmation analysis of the two frozen local-search directions."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from preparations.ease_insights_20261103.vlm.common import atomic_json, sha
from preparations.ease_insights_20261103.vlm.analyze import exact_paired, key, paired
from preparations.ease_insights_20261103.vlm.analyze_confirmation import contention_records, holm
from reports.ease_insights_20261103.camera_paths_v6.confirmation_math import bootstrap,task_signflip as task_block_signflip
from reports.ease_insights_20261103.camera_paths_v6.discovery_analysis import condition_summaries,read_selection,usable,completed_service_costs
from reports.ease_insights_20261103.camera_paths_v6.confirmation_load import load_stage
from preparations.ease_insights_20261103.camera_paths_v6.common import CONDITIONS

BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20261121


def direction_analysis(direction, manifest, records, scenes, complete, replicates=BOOTSTRAP_REPLICATES):
    cases = [c for c in scenes if c["suite"] == direction["suite"]]
    a, b = direction["better"], direction["worse"]
    result = dict(frozen_direction=direction, **paired(a, b, records, cases))
    result["interpretation"] = "Prespecified independent confirmation direction" if complete else "Incomplete confirmation; no inferential conclusion"
    tasks = [dict(task=t, **paired(a, b, records, [c for c in cases if c["task"] == t])) for t in manifest["tasks"]]
    result.update(by_task=tasks, inference=None)
    if not complete:
        result["exact_mcnemar_two_sided"] = None
        for row in tasks:
            row["exact_mcnemar_two_sided"] = None
        return result
    values = [int(records[a, key(c)]["success"]) - int(records[b, key(c)]["success"]) for c in cases]
    task_ids = [c["task"] for c in cases]
    intervals = bootstrap(values, task_ids)
    loo = [dict(task=t, difference=float(np.mean([v for v, c in zip(values, cases) if c["task"] != t]))) for t in manifest["tasks"]]
    result["inference"] = dict(primary_task_block=task_block_signflip([t["success_difference_count"] for t in tasks]),
        primary_holm2_p=None,
        sensitivity_case_mcnemar_p=exact_paired(result["a_only"], result["b_only"], one_sided=True),
        sensitivity_holm2_p=None, confidence_intervals=intervals,
        leave_one_task_out=loo, positive_direction=result["success_difference_count"] > 0,
        every_leave_one_task_out_positive=all(x["difference"] > 0 for x in loo),
        task_cluster_lower_bound_positive=intervals["task_cluster"]["low"] > 0,
        direction_reproduced=None,
        sensitivity_note="Case McNemar is descriptive sensitivity; repeated initial states are clustered within task for primary inference")
    return result


def apply_verdict(directions, eligibility):
    """Fixed two-hypothesis family; no outcome-driven reselection or expansion."""
    if len(directions) != 2 or any(d.get("inference") is None for d in directions):
        return "incomplete"
    if len(eligibility) != 2:
        raise ValueError("A complete verdict requires usability results for exactly two selected paths")
    primary = holm([d["inference"]["primary_task_block"]["p_one_sided"] for d in directions])
    sensitivity = holm([d["inference"]["sensitivity_case_mcnemar_p"] for d in directions])
    for d, p, sensitivity_p in zip(directions, primary, sensitivity):
        inference = d["inference"]
        inference.update(primary_holm2_p=p, sensitivity_holm2_p=sensitivity_p)
        inference["direction_reproduced"] = (inference["positive_direction"] and p < .05 and
            inference["task_cluster_lower_bound_positive"] and inference["every_leave_one_task_out_positive"])
    reproduced = all(d["inference"]["direction_reproduced"] for d in directions) and all(
        gate["usable"] for gate in eligibility.values())
    return "complementarity_reproduced" if reproduced else "not_reproduced"


def check_selection(manifest):
    reference = manifest["selection_evidence"]
    selection = read_selection(reference["path"])
    if (sha(reference["path"]) != reference["sha256"] or
            selection["selection_hash"] != reference["selection_hash"] or
            selection["selection"] != reference["frozen_selection"]):
        raise ValueError("Confirmation selection evidence differs from its frozen file")
    chosen = selection["selection"].get("selected")
    if selection["selection"]["status"] != "selected" or not chosen:
        raise ValueError("Confirmation requires a selected discovery pair")
    pair = chosen["paths"]
    directions = chosen["directions"]
    if (len(pair) != 2 or pair != sorted(set(pair)) or "full_d10" in pair or
            manifest["planned_conditions"] != list(CONDITIONS) or
            manifest["preregistered_hypotheses"] != directions or len(directions) != 2 or
            len({d["suite"] for d in directions}) != 2 or {d["better"] for d in directions} != set(pair) or
            any({d["better"], d["worse"]} != set(pair) or d["suite"] not in manifest["suites"] for d in directions)):
        raise ValueError("Confirmation pair/directions differ from the frozen opposite contrasts")
    return selection


def preparation_costs(root, manifest):
    """Preserve every run receipt; failures never become repeat scientific cases."""
    records, invalid = [], []
    for path in sorted(Path(root).glob("*/runs/*/result.json")):
        try:
            value = json.loads(path.read_text())
            if value.get("manifest_hash") != manifest["manifest_hash"]:
                raise ValueError("Run receipt belongs to another manifest")
            total, loop = value.get("total_wall_s"), value.get("closed_loop_wall_s")
            records.append(dict(path=str(path), sha256=sha(path), receipt=value,
                preparation_and_service_overhead_s=total-loop if isinstance(total, (int, float)) and isinstance(loop, (int, float)) else None))
        except (OSError, ValueError, TypeError) as error:
            invalid.append(dict(path=str(path), error=str(error)))
    return dict(runs=records, unreadable_or_mismatched=invalid,
        scope="Per-run total wall minus closed-loop wall combines preparation, validation/probes, and service overhead; it is not isolated model initialization or experiment makespan. Technical attempt logs are retained.")


def analyze_confirmation(root, contention_root=None, replicates=BOOTSTRAP_REPLICATES):
    if replicates != BOOTSTRAP_REPLICATES:raise ValueError("Frozen confirmation requires20000bootstrapdraws")
    audit = load_stage(root, "confirm")
    manifest, records, scenes = audit["manifest"], audit["records"], audit["scenes"]
    selection = check_selection(manifest)
    primary = [direction_analysis(d, manifest, records, scenes, audit["complete"], replicates)
        for d in manifest["preregistered_hypotheses"]]
    costs = completed_service_costs(audit["root"], manifest)
    if audit['complete'] and costs['valid_services'] != 8:
        raise ValueError('Confirmation needs all eight completed and numerically gated cost services')
    ratios = costs["median_full_over_policy_cycle"]
    selected_paths = selection["selection"]["selected"]["paths"]
    pair_cost_pass = bool(ratios) and max(ratios[c] for c in selected_paths) <= 1.10 * min(ratios[c] for c in selected_paths)
    eligibility = {}
    if audit["complete"]:
        counts = {cid: sum(records[cid, key(c)]["success"] for c in scenes) for cid in manifest["planned_conditions"]}
        for cid in selection["selection"]["selected"]["paths"]:
            eligibility[cid] = dict(successes=counts[cid], n=len(scenes), success_rate=counts[cid]/len(scenes),
                full_success_rate=counts["full_d10"]/len(scenes), usable=usable(counts[cid], counts["full_d10"], len(scenes)),
                median_fixed_input_ratio=ratios.get(cid),cost_pass=ratios.get(cid,0)>=1.15,pair_cost_pass=pair_cost_pass)
    verdict_gates = {cid: dict(usable=g["usable"] and g["cost_pass"] and g["pair_cost_pass"]) for cid,g in eligibility.items()}
    conclusion = apply_verdict(primary, verdict_gates) if audit["complete"] else "incomplete"
    contention_root = Path(contention_root) if contention_root else audit["root"].parents[1] / "resource_contention"
    dense_dominates=all(counts["dense_d2"]>=counts[c] and ratios["dense_d2"]>=ratios[c] for c in selected_paths) if audit['complete'] else None
    return dict(DenseD2_dominates_both_at_no_higher_native_cost=dense_dominates,adaptive_routing_benefit_established=False,revision="ease-full-storage-image-mask-confirmation-analysis-v6", manifest_hash=manifest["manifest_hash"],
        selection_hash=selection["selection_hash"], protocol_evidence=manifest["protocol_evidence"],
        complete=audit["complete"], expected_episodes=800, valid_episodes=len(records),
        missing=audit["missing"], invalid=audit["invalid"], unexpected=audit["unexpected"],
        primary_directions=primary, overall_candidate_eligibility=eligibility, conclusion=conclusion,
        condition_summaries=condition_summaries(manifest, records, scenes),
        fixed_input_cost=costs,pair_cost_pass=pair_cost_pass,
        preparation_costs=preparation_costs(audit["root"], manifest),
        resource_contention=contention_records(contention_root, records),
        analysis_protocol=dict(primary="One-sided exact ten-task block sign-flip randomization including ties, Holm2",
            sensitivity="One-sided paired-case exact McNemar, Holm2, descriptive only",
            bootstrap_replicates=replicates, bootstrap_seed=BOOTSTRAP_SEED,
            intervals="Paired-case and whole-task percentile95 intervals", alpha=.05,
            verdict="Both positive directions, Holm2 task p<.05, task-cluster lower>0, every task LOO>0, both paths >=.75 and >=Full-.15 overall, each fixed-input ratio>=1.15 and pair costs within10%",
            limitation="Ten task clusters limit precision. Task sign-exchangeability is an assumption; no equivalence/noninferiority or exclusive-GPU speed claim.",
            no_reselection=True),
        evidence=dict(episodes=audit["episode_evidence"],
            manifest=dict(path=str(audit["root"] / "manifest.json"), sha256=sha(audit["root"] / "manifest.json")),
            selection=dict(path=manifest["selection_evidence"]["path"], sha256=manifest["selection_evidence"]["sha256"]),
            analysis_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))),
        original_negative_results_retained=True, scientific_outcomes_modified=False,
        followup="Preserve the fixed-pair verdict;negative results do not complete the overall research goal. No within-family reselection or significance retry.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contention-root")
    args = parser.parse_args()
    report = analyze_confirmation(args.root, args.contention_root)
    output = Path(args.output).resolve()
    atomic_json(output / "analysis.json", report)
    print(json.dumps(dict(complete=report["complete"], valid_episodes=report["valid_episodes"], conclusion=report["conclusion"])))


if __name__ == "__main__":
    main()
