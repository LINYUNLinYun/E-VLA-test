"""CPU-only analysis of the three frozen VLM contrasts; never changes the primary analysis.

Run as a module with --root, --output and optionally --registration. Scientific
inference requires all 600 independently audited confirmation episodes.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import re

from . import analyze_confirmation as primary
from .analyze import cost_records, key, load_episodes, summary
from .common import atomic_json, read_manifest, sha

# Byte identity of the registration written before the confirmation jobs began.
# The CLI deliberately offers no override of this pinned registration identity.
REGISTRATION_SHA256 = "8f2a6c3c1ee10008b5b7a2d48a247589b506308ecfd1095603d4df9f3723ba8f"
FAMILY = [
    dict(id="primary_spatial", suite="libero_spatial", better="fixed_v12_jacobian", worse="fixed_v12_action_mse"),
    dict(id="primary_long", suite="libero_10", better="fixed_v12_action_mse", worse="fixed_v12_jacobian"),
    dict(id="supplementary_goal", suite="libero_goal", better="fixed_v12_jacobian", worse="fixed_v12_action_mse"),
]


def analysis_protocol():
    protocol = copy.deepcopy(primary.analysis_protocol())
    protocol.update(
        primary_tests="Original Spatial/Long conclusions retain their original Holm family of two; separate three-contrast results use one-sided exact paired tests with Holm across exactly Spatial, Long, Goal",
        supplementary_tests="The registered Goal> and original Long> directions form the separate supplementary claim; no heldout selection or direction reversal",
        registration="Require pinned registration bytes, manifest/selection SHA bindings, and registration strictly before the earliest raw heldout identity mtime, including retained technical attempts",
        completeness="All 600 valid episodes and no missing, invalid, or unexpected completed cases; partial data receive counts only",
        inference_scope="Supplementary Goal/Long support is labelled prospective only when the registration audit passes; original primary failure remains visible",
    )
    return protocol


def audit_registration(root, manifest, registration_path):
    """Check immutable registration bindings and the earliest original evidence.

    The frozen rollout does not embed a Unix time in identity.json. File mtime is
    therefore the available original timestamp, not the later receipt timestamp.
    Every retained technical-attempt identity is included, not just valid cases.
    """
    root, registration_path = Path(root).resolve(), Path(registration_path).resolve()
    registration_sha = sha(registration_path)
    if registration_sha != REGISTRATION_SHA256:
        raise ValueError("Supplementary registration SHA changed; refuse prospective registration")
    obj = json.loads(registration_path.read_text())
    if (obj["version"] != "prospective-supplemental-goal-v1" or obj["family"] != FAMILY or
        obj["original_primary_directions"] != manifest["preregistered_hypotheses"] or
        {k: obj["supplementary_direction"][k] for k in ("suite", "better", "worse")} !=
        {k: FAMILY[2][k] for k in ("suite", "better", "worse")} or
        obj["confirmation_manifest_sha256"] != sha(root / "manifest.json") or
        obj["confirmation_manifest_hash"] != manifest["manifest_hash"] or
        obj["selection_sha256"] != sha(manifest["selection_evidence"]["path"]) or
        obj["selection_sha256"] != manifest["selection_evidence"]["sha256"] or
        obj["discovery_inits"] != [30] or obj["validation_inits"] != list(range(31, 36)) or
        obj["new_gpu_episodes"] != 0 or obj["planned_matrix_unchanged"] is not True or
        obj["model_or_runtime_changed"] is not False or obj["original_frozen_selection_changed"] is not False):
        raise ValueError("Supplementary registration differs from frozen protocol, manifest, or selection")
    registered = obj["registered_unix"]
    if isinstance(registered, bool) or not isinstance(registered, (int, float)) or not math.isfinite(registered) or registered <= 0:
        raise ValueError("Invalid supplementary registration time")
    roots = {root, Path(manifest["discovery_reference"]["root"]).resolve()}
    identities, errors = [], []
    for search_root in sorted(roots):
        for path in sorted(search_root.glob("*/episodes/*/task*/init*/identity.json")):
            if not re.fullmatch(r"init3[1-5](?:\..*)?", path.parent.name):
                continue
            try:
                identity = json.loads(path.read_text())
                if identity["case"]["init"] not in range(31, 36):
                    raise ValueError("Heldout identity directory and init disagree")
                identities.append(dict(path=str(path), sha256=sha(path), raw_identity_mtime_unix=path.stat().st_mtime))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
    identities.sort(key=lambda item: (item["raw_identity_mtime_unix"], item["path"]))
    earliest = identities[0] if identities else None
    reasons = []
    if obj["heldout_outcomes_read"] is not False or obj["heldout_evidence_files_at_registration"] != []:
        reasons.append("Registration declares preexisting heldout evidence or outcomes")
    if errors:
        reasons.append("Some original heldout identities cannot be audited")
    if earliest and registered >= earliest["raw_identity_mtime_unix"]:
        reasons.append("Registration is not strictly earlier than the first original heldout identity")
    return dict(accepted=not reasons, classification="prospective_supplementary" if not reasons else "prospective_label_refused",
        path=str(registration_path), sha256=registration_sha, registered_unix=registered,
        manifest_sha256=obj["confirmation_manifest_sha256"], selection_sha256=obj["selection_sha256"],
        first_original_heldout_identity=earliest, original_identity_count=len(identities),
        original_identity_evidence=identities, unreadable_identities=errors, refusal_reasons=reasons,
        searched_roots=[str(path) for path in sorted(roots)],
        timing_limit="Raw identity mtime is filesystem evidence, not an independently notarized clock; frozen identities contain no embedded Unix timestamp. Receipt creation times are not substituted.")


def pair_conclusion(rows, complete):
    """Use the unchanged main analysis stability rule, including its failure case."""
    if not complete:
        return "incomplete"
    if all(row["inference"]["stable_positive_direction"] for row in rows):
        return "supports_complementary_fixed_paths_on_the_two_prespecified_suites"
    if all(row["inference"]["positive_direction"] for row in rows):
        return "both_directions_positive_but_stability_uncertain"
    return "both_prespecified_directions_not_reproduced"


def analyze(root, registration_path=None, contention_root=None, replicates=primary.BOOTSTRAP_REPLICATES):
    root = Path(root).resolve()
    registration_path = Path(registration_path).resolve() if registration_path else root / "supplementary_goal_contrast.json"
    manifest = read_manifest(root / "manifest.json", verify_sources=True)
    selection = primary.check_manifest(root, manifest)
    registration = audit_registration(root, manifest, registration_path)
    view = copy.deepcopy(manifest)
    view["conditions"] = [next(c for c in manifest["conditions"] if c["id"] == cid) for cid in manifest["planned_conditions"]]
    view["discovery_inits"] = manifest["confirmation_inits"]
    scenes = [dict(suite=s, task=t, init=i) for s in manifest["suites"] for t in manifest["tasks"] for i in manifest["confirmation_inits"]]
    records, missing, invalid, evidence = load_episodes(root, view)
    invalid.extend(primary.audit_identities(root, manifest, records))
    unexpected = primary.unexpected_cases(root, manifest, scenes)
    complete = len(records) == 600 and not missing and not invalid and not unexpected
    directions = [*manifest["preregistered_hypotheses"], {k: FAMILY[2][k] for k in ("suite", "better", "worse")}]
    contrasts = []
    for spec, direction in zip(FAMILY, directions):
        row = primary.direction_analysis(direction, manifest, records, scenes, complete, replicates)
        row.update(contrast_id=spec["id"], role="original_primary" if spec["id"].startswith("primary_") else "supplementary_goal")
        row["by_case"] = [dict(task=c["task"], init=c["init"],
            a_success=None if (direction["better"], key(c)) not in records else records[direction["better"], key(c)]["success"],
            b_success=None if (direction["worse"], key(c)) not in records else records[direction["worse"], key(c)]["success"])
            for c in scenes if c["suite"] == direction["suite"]]
        if row["role"] == "supplementary_goal":
            row["interpretation"] = ("Prospective supplementary direction" if registration["accepted"] else
                "Descriptive supplementary direction; prospective label refused") if complete else "Incomplete confirmation; counts only, no conclusion"
        contrasts.append(row)
    original_primary = copy.deepcopy(contrasts[:2])
    if complete:
        for row, p in zip(original_primary, primary.holm([r["inference"]["exact_mcnemar_one_sided"] for r in original_primary])):
            row["inference"]["holm_adjusted_p"] = p
        for row, p in zip(contrasts, primary.holm([r["inference"]["exact_mcnemar_one_sided"] for r in contrasts])):
            row["inference"].update(holm_adjusted_p=p, holm_family_size=3)
    primary_conclusion = pair_conclusion(original_primary, complete)
    supplementary_conclusion = pair_conclusion([contrasts[2], contrasts[1]], complete)
    if complete and not registration["accepted"]:
        supplementary_conclusion = "prospective_label_refused; descriptive_only"
    # Check pinned registration bindings once more after loading the results.
    # This does not scan identities twice or rehash the episode arrays twice.
    if (sha(registration_path) != registration["sha256"] or
        sha(root / "manifest.json") != registration["manifest_sha256"] or
        sha(manifest["selection_evidence"]["path"]) != registration["selection_sha256"]):
        raise ValueError("Frozen registration evidence changed during analysis")
    costs = cost_records(root, manifest)
    costs["timing_limit"] = "Report separately from free-running rollout costs; known contention may affect fixed-input ratios too"
    contention_root = Path(contention_root).resolve() if contention_root else root.parent / "resource_contention"
    return dict(status="complete" if complete else "incomplete", complete=complete,
        conclusion="incomplete" if not complete else "see_separate_primary_and_supplementary_conclusions",
        valid_episodes=len(records), expected_episodes=600, missing=missing, invalid=invalid,
        unexpected_completed_cases=unexpected, manifest_hash=manifest["manifest_hash"],
        selection_hash=selection["selection_hash"], planned_conditions=manifest["planned_conditions"],
        prospective_registration=registration, analysis_protocol=analysis_protocol(),
        analysis_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__),
            reused_primary_source=dict(path=str(Path(primary.__file__).resolve()), sha256=sha(primary.__file__))),
        no_reselection=True, quality_results_modified=False, new_gpu_episodes=0,
        contrasts=contrasts, multiplicity=dict(method="Holm", family_size=3, contrast_ids=[r["id"] for r in FAMILY]),
        original_primary_analysis=dict(conclusion=primary_conclusion, contrast_ids=[r["id"] for r in FAMILY[:2]],
            holm_family_size=2, primary_directions=original_primary,
            note="Unchanged Spatial/Long claim and original two-test correction; inspect contrasts for the additional Holm-three family"),
        supplementary_goal_long_analysis=dict(conclusion=supplementary_conclusion,
            contrast_ids=["supplementary_goal", "primary_long"], prospective_label=registration["accepted"],
            holm_family_size=3, replaces_original_primary_claim=False),
        condition_summaries=[dict(condition=cid, expected_n=len(scenes),
            **summary([records[cid, key(c)] for c in scenes if (cid, key(c)) in records])) for cid in manifest["planned_conditions"]],
        fixed_input_cost=costs, resource_contention=primary.contention_records(contention_root, records),
        uncertainty="Non-significance is not equivalence. Both claims can fail. Ten task clusters and five initial states limit precision; no discovery-only positive claim. This tests fixed paths, not runtime selection or phase switching.",
        evidence=dict(episode_receipts=evidence, selection=manifest["selection_evidence"]))


def markdown(report):
    lines = [f"Supplementary VLM confirmation: {report['valid_episodes']}/{report['expected_episodes']} valid episodes; status={report['status']}.",
        "Original Spatial/Long conclusion: " + report["original_primary_analysis"]["conclusion"] + ".",
        "Supplementary Goal/Long conclusion: " + report["supplementary_goal_long_analysis"]["conclusion"] + ".",
        "Prospective registration: " + report["prospective_registration"]["classification"] + ".", "",
        "| Fixed contrast | n | Difference | A-only / B-only | One-sided exact p | Holm-3 p | Case 95% CI | Task-cluster 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|---|"]
    for row in report["contrasts"]:
        d, inference = row["frozen_direction"], row["inference"]
        fmt = lambda value: "pending" if value is None else f"{value:.6g}"
        interval = lambda name: "pending" if inference is None else f"[{inference['confidence_intervals'][name]['low']:+.3f}, {inference['confidence_intervals'][name]['high']:+.3f}]"
        lines.append(f"| {row['contrast_id']}: {d['better']} > {d['worse']} | {row['n_paired']} | {fmt(row['success_difference'])} | {row['a_only']} / {row['b_only']} | {fmt(None if inference is None else inference['exact_mcnemar_one_sided'])} | {fmt(None if inference is None else inference['holm_adjusted_p'])} | {interval('case')} | {interval('task_cluster')} |")
    lines += ["", "supplementary_analysis.json includes all task/init/case effects, leave-one-task/init-out checks, the original Holm-2 results, registration evidence, costs and all known contention records.",
        "Original primary failure is retained. Quality outcomes are unchanged. Timings do not establish exclusive-GPU speed; non-significance does not establish equivalence."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registration")
    parser.add_argument("--contention-root")
    args = parser.parse_args()
    report = analyze(args.root, args.registration, args.contention_root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "supplementary_analysis.json", report)
    (output / "supplementary_analysis.md").write_text(markdown(report))
    atomic_json(output / "supplementary_analysis_protocol.json", dict(source=report["analysis_source"], protocol=analysis_protocol()))
    print(json.dumps(dict(status=report["status"], valid_episodes=report["valid_episodes"], expected_episodes=600,
        original_primary_conclusion=report["original_primary_analysis"]["conclusion"],
        supplementary_conclusion=report["supplementary_goal_long_analysis"]["conclusion"], output=str(output))))


if __name__ == "__main__":
    main()
