"""Independent confirmation analysis of the two frozen within-suite directions."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .common import atomic_json, sha
from preparations.ease_insights_20261103.vlm.analyze import exact_paired, key, paired
from preparations.ease_insights_20261103.vlm.analyze_confirmation import contention_records, holm
from .selection import condition_summaries, load_stage, read_selection, usable, category_tasks, category_cases

BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20261105


def task_block_signflip(task_difference_counts):
    """One-sided exact12-task sign randomization; all4096 assignments, ties too."""
    values = list(task_difference_counts)
    if len(values) != 12 or any(not isinstance(v, (int, np.integer)) or abs(v) > 5 for v in values):
        raise ValueError("Task-block test requires12 integer five-case task differences")
    observed = sum(values)
    extreme = sum(sum(sign * value for sign, value in zip(signs, values)) >= observed
        for signs in itertools.product((-1, 1), repeat=12))
    return dict(p_one_sided=extreme / 4096, extreme_assignments=extreme, assignments=4096,
        observed_mean_difference=observed / 60, task_difference_counts=values,
        task_mean_differences=[v / 5 for v in values],
        null="Task-level paired differences sign-exchangeable under the null;12 tasks are inference units",
        ties_included=True)


def bootstrap(values, task_ids, seed_case=BOOTSTRAP_SEED, seed_task=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    values=np.asarray(values,dtype=np.float64)
    task_ids=list(task_ids);tasks=sorted(set(task_ids))
    if len(values)!=60 or len(tasks)!=12 or any(task_ids.count(t)!=5 for t in tasks):
        raise ValueError("Confirmation bootstrap requires12tasks x5paired initial states")
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("Positive integer bootstrap replicate count required")
    estimates=np.random.default_rng(seed_case).choice(values,size=(replicates,60),replace=True).mean(axis=1)
    means=np.asarray([values[[i for i,t in enumerate(task_ids) if t==task]].mean() for task in tasks])
    cluster=np.random.default_rng(seed_task).choice(means,size=(replicates,12),replace=True).mean(axis=1)
    def interval(samples,unit,seed):
        return dict(low=float(np.quantile(samples,.025,method="linear")),high=float(np.quantile(samples,.975,method="linear")),
            seed=seed,replicates=replicates,unit=unit,method="percentile",quantile_method="linear")
    return dict(case=interval(estimates,"paired case",seed_case),
        task_cluster=dict(interval(cluster,"whole task with its five initial states",seed_task),clusters=12))


def direction_analysis(direction, manifest, records, scenes, complete, replicates=BOOTSTRAP_REPLICATES):
    cases = category_cases(manifest, scenes, direction["category"])
    selected_tasks = category_tasks(manifest, direction["category"])
    a, b = direction["better"], direction["worse"]
    result = dict(frozen_direction=direction, **paired(a, b, records, cases))
    result["interpretation"] = "Prespecified independent confirmation direction" if complete else "Incomplete confirmation; no inferential conclusion"
    tasks = [dict(task=t, **paired(a, b, records, [c for c in cases if c["task"] == t])) for t in selected_tasks]
    result.update(by_task=tasks, inference=None)
    if not complete:
        result["exact_mcnemar_two_sided"] = None
        for row in tasks:
            row["exact_mcnemar_two_sided"] = None
        return result
    values = [int(records[a, key(c)]["success"]) - int(records[b, key(c)]["success"]) for c in cases]
    task_ids = [c["task"] for c in cases]
    intervals = bootstrap(values, task_ids, BOOTSTRAP_SEED, BOOTSTRAP_SEED, replicates)
    loo = [dict(task=t, difference=float(np.mean([v for v, c in zip(values, cases) if c["task"] != t]))) for t in selected_tasks]
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
    """Fixed two-hypothesis family-local alpha.05 only; no outcome-driven reselection or expansion."""
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
            manifest["planned_conditions"] != ["full_d10", *pair] or
            manifest["preregistered_hypotheses"] != directions or len(directions) != 2 or
            {d["category"] for d in directions} != set(manifest["categories"]) or {d["better"] for d in directions} != set(pair) or
            any({d["better"], d["worse"]} != set(pair) or d["category"] not in manifest["categories"] for d in directions)):
        raise ValueError("Confirmation pair/directions differ from the frozen opposite category contrasts")
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
    audit = load_stage(root, "confirm")
    manifest, records, scenes = audit["manifest"], audit["records"], audit["scenes"]
    selection = check_selection(manifest)
    primary = [direction_analysis(d, manifest, records, scenes, audit["complete"], replicates)
        for d in manifest["preregistered_hypotheses"]]
    eligibility = {}
    if audit["complete"]:
        counts = {cid: sum(records[cid, key(c)]["success"] for c in scenes) for cid in manifest["planned_conditions"]}
        for cid in selection["selection"]["selected"]["paths"]:
            eligibility[cid] = dict(successes=counts[cid], n=len(scenes), success_rate=counts[cid]/len(scenes),
                full_success_rate=counts["full_d10"]/len(scenes), usable=usable(counts[cid], counts["full_d10"], len(scenes)))
    conclusion = apply_verdict(primary, eligibility) if audit["complete"] else "incomplete"
    contention_root = Path(contention_root) if contention_root else audit["root"].parents[1] / "resource_contention"
    return dict(revision="ease-vlm-within-suite-confirmation-analysis-v1", manifest_hash=manifest["manifest_hash"],
        selection_hash=selection["selection_hash"], protocol_evidence=manifest["protocol_evidence"],
        complete=audit["complete"], expected_episodes=360, valid_episodes=len(records),
        missing=audit["missing"], invalid=audit["invalid"], unexpected=audit["unexpected"],
        primary_directions=primary, overall_candidate_eligibility=eligibility, conclusion=conclusion,
        condition_summaries=condition_summaries(manifest, records, scenes),
        preparation_costs=preparation_costs(audit["root"], manifest),
        resource_contention=contention_records(contention_root, records),
        analysis_protocol=dict(primary="One-sided exact 12-task block sign-flip randomization including ties, Holm2",
            sensitivity="One-sided paired-case exact McNemar, Holm2, descriptive only",
            bootstrap_replicates=replicates, bootstrap_seed=BOOTSTRAP_SEED,
            intervals="Paired-case and whole-task percentile95 intervals, explicit linear quantiles", alpha=.05,
            verdict="Both positive directions, Holm2 task p<.05, task-cluster lower>0, every task LOO>0, both paths >=.75 and >=Full-.15 overall",
            limitation="Twelve task clusters limit precision. Family-local Holm2 does not establish whole-research FWER. Shared scenes/prompts may induce dependence. Task sign-exchangeability is an assumption; no equivalence/noninferiority or exclusive-GPU speed claim.",
            no_reselection=True),
        evidence=dict(episodes=audit["episode_evidence"],
            manifest=dict(path=str(audit["root"] / "manifest.json"), sha256=sha(audit["root"] / "manifest.json")),
            selection=dict(path=manifest["selection_evidence"]["path"], sha256=manifest["selection_evidence"]["sha256"]),
            analysis_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))),
        original_negative_results_retained=True, scientific_outcomes_modified=False,
        followup="Bounded search ends here; no automatic further search or significance retry")


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
