"""Independent, complete-data-only CPU statistics for one frozen EASE matrix.

No scientific project module is imported. The only outcome inputs are the
acceptance- and supervisor-bound original episode receipt/result JSON files.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CASE_ROOT = ROOT / "results/ease_insights_20261103/vlm_local_search_v1/confirmation"
ACCEPTANCE = ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_confirmation/audit.json"
ACCEPTANCE_SCRIPT = ACCEPTANCE.with_suffix(".py")
SUPERVISOR = CASE_ROOT / "analysis/analysis.json"
MANIFEST_HASH = "80176d9f4b01e68abb022345bd6dc1a7bfe4e5aef5d69d13257801ca1eb6bdc0"
MANIFEST_SHA = "3c4b8bf35ecf768f129c00721ce0fd0e49ef2c209f7b9995ed0a1a128796365a"
SELECTION_SHA = "f6314a52f71ab0a127afb7bae0fd1adb201c934d1634634567217490a4d1e9ed"
ACCEPTANCE_SCRIPT_SHA = "a9ee976a07997a20cb56d6a0d77df315d7956292fc1d78e88fd6b0a7f1f06da9"
ANALYSIS_PATH = "preparations/ease_insights_20261103/vlm_local_search_v1/analyze.py"
ANALYSIS_SHA = "525800ce944944a73d0c5d3fec52c8966c4a229dc801d15d3c2c73de38d5572b"
CONDITIONS = ("full_d10", "jac_anchor", "local_long")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS = tuple(range(10))
INITS = tuple(range(39, 44))
DIRECTIONS = (("libero_object", "jac_anchor", "local_long"), ("libero_10", "local_long", "jac_anchor"))
SEED = 20261104
REPLICATES = 20000
FLOAT_ATOL = 1e-12


class GateError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha_bytes(Path(path).read_bytes())


def read_bound_json(path, expected_sha):
    """Hash the exact bytes parsed, without a hash/read race."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    actual = sha_bytes(raw)
    require(actual == expected_sha, "SHA256 mismatch: " + str(path))
    return json.loads(raw), dict(path=str(path), sha256=actual)


def project_path(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def cases():
    return [(suite, task, init) for suite in SUITES for task in TASKS for init in INITS]


def receipt_path(root, cid, scene):
    suite, task, init = scene
    return Path(root) / cid / "episodes" / suite / f"task{task:02d}" / f"init{init:02d}" / "episode_receipt.json"


def acceptance_gate(report):
    """This gate must finish before opening manifest, supervisor, or outcomes."""
    require(report.get("revision") == "full-local-search-confirmation-read-only-acceptance-v1", "Wrong acceptance revision")
    require(report.get("status") == "pass", "Independent 600 acceptance is not PASS; outcomes were not opened")
    require(report.get("root") == str(CASE_ROOT), "Acceptance root differs")
    require(report.get("manifest_hash") == MANIFEST_HASH and report.get("manifest_file_sha256") == MANIFEST_SHA,
            "Acceptance manifest identity differs")
    require(report.get("failed_checks") == [], "Acceptance has failed checks")
    stage = report.get("stage", {})
    require(stage.get("passed") is True and stage.get("complete") is True and
            stage.get("expected") == 600 and stage.get("valid") == 600 and stage.get("scenes") == 200,
            "Acceptance does not certify exactly 600 valid cases")
    require(stage.get("planned_conditions") == list(CONDITIONS) and stage.get("init_ids") == list(INITS) and
            stage.get("per_condition") == {cid: 200 for cid in CONDITIONS}, "Acceptance matrix differs")
    for name in ("missing", "invalid", "unexpected"):
        require(stage.get(name) == [] and stage.get(name + "_count") == 0, "Acceptance has " + name)
    require(stage.get("reasons") == [], "Acceptance stage has issues")
    checks = report.get("checks", [])
    require(len(checks) >= 4 and all(row.get("passed") is True for row in checks), "Acceptance checks incomplete")
    for name in ("source_identity", "frozen_selection", "requests", "initial_input_pairing", "discovery_reference", "serving_runs"):
        require(report.get(name, {}).get("passed") is True, "Acceptance gate missing: " + name)
    require(report["requests"].get("cases_checked") == 600 and report["requests"].get("failed_case_requests") == [],
            "Acceptance request coverage/issues differ")
    require(report["initial_input_pairing"].get("cases_checked") == 200 and
            report["initial_input_pairing"].get("failed_cases") == [], "Acceptance input pairing differs")
    for name, issue_key in (("frozen_selection", "errors"), ("serving_runs", "issues"), ("discovery_reference", "issues")):
        require(report[name].get(issue_key) == [], "Acceptance issues: " + name)
    scope = report.get("scope", {})
    require(scope.get("cases_checked") == 600 and scope.get("gpu_used") is False and
            scope.get("scientific_selection_or_statistics") is False, "Acceptance scope differs")


def evidence_map(rows, expected_paths, label):
    require(isinstance(rows, list), label + " evidence is missing")
    result = {}
    for row in rows:
        path = project_path(row["path"])
        require(path not in result, label + " has duplicate paths")
        require(re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) is not None, label + " has an invalid SHA")
        result[path] = row["sha256"]
    require(set(result) == set(expected_paths), label + " evidence is not the exact 600 receipt set")
    return result


def source_identity(manifest, report):
    """Match the acceptance's declared source bank to frozen manifest and disk."""
    verified = []
    identity = report["source_identity"]
    for group, count in (("source_files", 46), ("analysis_sources", 6), ("local_runtime_sources", 9)):
        bank = {project_path(p): expected for p, expected in manifest[group].items()}
        rows = identity.get("groups", {}).get(group, {})
        require(len(bank) == count and rows.get("count") == count and rows.get("expected_count") == count and
                rows.get("passed") is True, "Frozen source group differs: " + group)
        declared = rows.get("entries", [])
        require(len(declared) == count and {project_path(r["path"]) for r in declared} == set(bank),
                "Acceptance source coverage differs: " + group)
        for row in declared:
            path = project_path(row["path"])
            require(row.get("passed") is True and row.get("error") is None and
                    row.get("sha256") == row.get("expected_sha256") == bank[path], "Accepted source hash differs")
            require(file_sha(path) == bank[path], "Frozen source changed: " + str(path))
            verified.append(dict(path=str(path), sha256=bank[path], group=group))
    extra_bank = {}
    for checkpoint in manifest["checkpoints"].values():
        extra_bank.update({project_path(p): h for p, h in checkpoint["identity_artifacts"].items()})
    for name in ("protocol_evidence", "design_evidence"):
        item = manifest[name]
        extra_bank[project_path(item["path"])] = item["sha256"]
    declared = identity.get("checkpoint_identity_artifacts", []) + identity.get("protocol_and_design_evidence", [])
    require(len(declared) == len(extra_bank) and {project_path(r["path"]) for r in declared} == set(extra_bank),
            "Acceptance checkpoint/protocol/design coverage differs")
    for row in declared:
        path = project_path(row["path"])
        require(row.get("passed") is True and row.get("error") is None and
                row.get("sha256") == row.get("expected_sha256") == extra_bank[path] and
                file_sha(path) == extra_bank[path], "Frozen identity artifact changed: " + str(path))
        verified.append(dict(path=str(path), sha256=extra_bank[path], group="checkpoint_protocol_design"))
    selection = manifest["selection_evidence"]
    require(selection["sha256"] == SELECTION_SHA and file_sha(project_path(selection["path"])) == SELECTION_SHA,
            "Frozen selection file differs")
    require(report["frozen_selection"].get("sha256") == SELECTION_SHA and
            report["frozen_selection"].get("selection_hash") == selection["selection_hash"], "Accepted selection differs")
    # Bind all acceptance implementation dependencies as declared by that report.
    inputs = report.get("input_source_hashes", [])
    require(inputs and len({project_path(r["path"]) for r in inputs}) == len(inputs), "Acceptance input sources missing/duplicate")
    for row in inputs:
        path = project_path(row["path"])
        require(row.get("exists") is True and file_sha(path) == row.get("sha256"), "Acceptance input source changed: " + str(path))
        verified.append(dict(path=str(path), sha256=row["sha256"], group="acceptance_input_source"))
    audit = report.get("audit_script", {})
    require(project_path(audit.get("path", "")) == ACCEPTANCE_SCRIPT and audit.get("sha256") == ACCEPTANCE_SCRIPT_SHA and
            file_sha(ACCEPTANCE_SCRIPT) == ACCEPTANCE_SCRIPT_SHA, "Acceptance audit implementation differs")
    return verified


def manifest_gate(manifest):
    payload = dict(manifest)
    identity = payload.pop("manifest_hash", None)
    digest = sha_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    require(identity == digest == MANIFEST_HASH, "Manifest content digest differs")
    require(manifest.get("phase") == "local_search_confirmation_v1" and
            manifest.get("output_root") == str(CASE_ROOT) and manifest.get("planned_conditions") == list(CONDITIONS) and
            manifest.get("suites") == list(SUITES) and manifest.get("tasks") == list(TASKS) and
            manifest.get("confirmation_inits") == list(INITS) and manifest.get("cases_per_condition") == 200 and
            manifest.get("total_confirmation_episodes") == 600, "Frozen matrix differs")
    actual = tuple((d["suite"], d["better"], d["worse"]) for d in manifest["preregistered_hypotheses"])
    require(actual == DIRECTIONS and manifest["analysis_sources"].get(ANALYSIS_PATH) == ANALYSIS_SHA, "Frozen directions/source differ")


def supervisor_gate(report, manifest, accepted_receipts):
    require(report.get("revision") == "ease-vlm-local-search-confirmation-analysis-v1" and
            report.get("complete") is True and report.get("expected_episodes") == 600 and
            report.get("valid_episodes") == 600 and all(report.get(k) == [] for k in ("missing", "invalid", "unexpected")),
            "Supervisor analysis is incomplete or has issues")
    require(report.get("manifest_hash") == MANIFEST_HASH and report.get("selection_hash") == manifest["selection_evidence"]["selection_hash"],
            "Supervisor data identity differs")
    require(set(report.get("overall_candidate_eligibility", {})) == set(CONDITIONS[1:]),
            "Supervisor eligibility must contain exactly the two frozen candidates")
    evidence = report.get("evidence", {})
    for name, expected_path, expected_sha in (("manifest", CASE_ROOT / "manifest.json", MANIFEST_SHA),
            ("selection", project_path(manifest["selection_evidence"]["path"]), SELECTION_SHA),
            ("analysis_source", ROOT / ANALYSIS_PATH, ANALYSIS_SHA)):
        entry = evidence.get(name, {})
        require(project_path(entry.get("path", "")) == expected_path and entry.get("sha256") == expected_sha,
                "Supervisor evidence differs: " + name)
    supervisor_receipts = evidence_map(evidence.get("episodes"), accepted_receipts, "supervisor")
    require(supervisor_receipts == accepted_receipts, "Supervisor and acceptance receipt hashes differ")
    protocol = report.get("analysis_protocol", {})
    require(protocol.get("bootstrap_replicates") == REPLICATES and protocol.get("bootstrap_seed") == SEED and
            protocol.get("alpha") == .05 and protocol.get("no_reselection") is True, "Supervisor numerical protocol differs")


def load_outcomes(root, manifest, receipt_hashes):
    records, evidence = {}, []
    configs = {c["id"]: c for c in manifest["conditions"]}
    for cid in CONDITIONS:
        for scene in cases():
            path = receipt_path(root, cid, scene).resolve()
            receipt, receipt_source = read_bound_json(path, receipt_hashes[path])
            case = dict(zip(("suite", "task", "init"), scene))
            require(receipt.get("valid") is True and receipt.get("manifest_hash") == MANIFEST_HASH and
                    receipt.get("condition") == cid and receipt.get("case") == case and type(receipt.get("success")) is bool,
                    "Raw receipt identity/success differs: " + str(path))
            result_sha = receipt.get("artifacts", {}).get("result.json")
            require(isinstance(result_sha, str) and re.fullmatch(r"[0-9a-f]{64}", result_sha), "Receipt has no result SHA")
            result, result_source = read_bound_json(path.parent / "result.json", result_sha)
            require(result.get("valid") is True and result.get("manifest_hash") == MANIFEST_HASH and
                    result.get("condition") == cid and result.get("case") == case and result.get("config") == configs[cid] and
                    type(result.get("success")) is bool and result["success"] == receipt["success"],
                    "Raw result identity/success differs: " + str(path.parent))
            records[cid, scene] = result["success"]
            evidence.append(dict(condition=cid, case=case, receipt=receipt_source, result=result_source,
                                 receipt_binding="acceptance.receipt_evidence AND supervisor.evidence.episodes",
                                 result_binding="SHA256 in the bound original episode_receipt.artifacts"))
    require(len(records) == 600, "Raw success matrix is incomplete")
    return records, evidence


def binomial_tail(wins, losses, two_sided=False):
    discordant = wins + losses
    if not discordant:
        return 1.0
    threshold = min(wins, losses) if two_sided else wins
    indices = range(threshold + 1) if two_sided else range(threshold, discordant + 1)
    probability = sum(math.comb(discordant, k) for k in indices) / (1 << discordant)
    return min(1.0, 2 * probability) if two_sided else probability


def paired_counts(records, a, b, scenes):
    counts = Counter((records[a, scene], records[b, scene]) for scene in scenes)
    n = len(scenes)
    wins, losses = counts[True, False], counts[False, True]
    return dict(a=a, b=b, n_paired=n, expected_n=n, a_success=counts[True, True] + wins,
                b_success=counts[True, True] + losses, both_success=counts[True, True], both_failure=counts[False, False],
                a_only=wins, b_only=losses, success_difference_count=wins - losses, success_difference=(wins - losses) / n,
                exact_mcnemar_two_sided=binomial_tail(wins, losses, two_sided=True))


def sign_flip(differences):
    require(len(differences) == 10 and all(type(v) is int and abs(v) <= 5 for v in differences), "Invalid task differences")
    observed = sum(differences)
    # A subset replaces each selected task's observed sign; zero tasks still
    # contribute two assignments, so ties remain in the full 1024 denominator.
    extreme = 0
    for mask in range(1024):
        null_sum = observed - 2 * sum(v for index, v in enumerate(differences) if mask & (1 << index))
        extreme += null_sum >= observed
    return dict(p_one_sided=extreme / 1024, extreme_assignments=extreme, assignments=1024,
                observed_mean_difference=observed / 50, task_difference_counts=differences,
                task_mean_differences=[v / 5 for v in differences], ties_included=True)


def holm_two(pvalues):
    require(len(pvalues) == 2, "Exactly two hypotheses required")
    first, second = sorted(range(2), key=lambda i: (pvalues[i], i))
    adjusted = [None, None]
    adjusted[first] = min(1., 2 * pvalues[first])
    adjusted[second] = max(adjusted[first], pvalues[second])
    return adjusted


def percentile_interval(estimates):
    low, high = np.quantile(estimates, [.025, .975], method="linear")
    return dict(low=float(low), high=float(high), seed=SEED, replicates=REPLICATES, method="percentile")


def independent_bootstrap(deltas):
    """PCG64 index sampling matches frozen default_rng.choice(..., p=None).

    The input order is task00 init39..43, then task01, ... task09. Both
    generators start independently from SEED. Task means remain float64: they
    are not replaced by integer task totals, which could alter last bits.
    """
    values = np.asarray(deltas, dtype=np.float64)
    require(values.shape == (50,), "Bootstrap requires 50 ordered pairs")
    task_means = values.reshape(10, 5).mean(axis=1)
    case_rng = np.random.Generator(np.random.PCG64(SEED))
    case_indices = case_rng.integers(0, 50, size=(REPLICATES, 50))
    case = percentile_interval(values[case_indices].mean(axis=1))
    task_rng = np.random.Generator(np.random.PCG64(SEED))
    task_indices = task_rng.integers(0, 10, size=(REPLICATES, 10))
    cluster = percentile_interval(task_means[task_indices].mean(axis=1))
    case["unit"] = "paired case"
    cluster.update(clusters=10, unit="whole task with its five initial states")
    return dict(case=case, task_cluster=cluster)


def rebuild_statistics(records, directions):
    scenes = cases()
    summaries = []
    groups = [("all", None, None, scenes)]
    groups += [("suite", s, None, [c for c in scenes if c[0] == s]) for s in SUITES]
    groups += [("task", s, t, [c for c in scenes if c[:2] == (s, t)]) for s in SUITES for t in TASKS]
    for level, suite, task, group in groups:
        for cid in CONDITIONS:
            count = sum(records[cid, scene] for scene in group)
            summaries.append(dict(level=level, suite=suite, task=task, condition=cid,
                                  expected_n=len(group), n=len(group), successes=count, success_rate=count / len(group)))
    totals = {cid: sum(records[cid, scene] for scene in scenes) for cid in CONDITIONS}
    eligibility = {cid: dict(successes=totals[cid], n=200, success_rate=totals[cid] / 200,
                       full_success_rate=totals["full_d10"] / 200,
                       usable=totals[cid] * 4 >= 200 * 3 and (totals["full_d10"] - totals[cid]) * 20 <= 200 * 3)
                   for cid in CONDITIONS[1:]}
    primary = []
    for direction in directions:
        suite, a, b = direction["suite"], direction["better"], direction["worse"]
        group = [scene for scene in scenes if scene[0] == suite]
        counts = paired_counts(records, a, b, group)
        by_task = [dict(task=t, **paired_counts(records, a, b, [c for c in group if c[1] == t])) for t in TASKS]
        deltas = [int(records[a, c]) - int(records[b, c]) for c in group]
        intervals = independent_bootstrap(deltas)
        loo = [dict(task=t, difference=sum(delta for delta, c in zip(deltas, group) if c[1] != t) / 45) for t in TASKS]
        inference = dict(primary_task_block=sign_flip([row["success_difference_count"] for row in by_task]),
                         primary_holm2_p=None, sensitivity_case_mcnemar_p=binomial_tail(counts["a_only"], counts["b_only"]),
                         sensitivity_holm2_p=None, confidence_intervals=intervals, leave_one_task_out=loo,
                         positive_direction=counts["success_difference_count"] > 0,
                         every_leave_one_task_out_positive=all(row["difference"] > 0 for row in loo),
                         task_cluster_lower_bound_positive=intervals["task_cluster"]["low"] > 0, direction_reproduced=None)
        primary.append(dict(frozen_direction=direction, **counts, by_task=by_task, inference=inference))
    adjusted = holm_two([d["inference"]["primary_task_block"]["p_one_sided"] for d in primary])
    sensitivity = holm_two([d["inference"]["sensitivity_case_mcnemar_p"] for d in primary])
    for row, p, sp in zip(primary, adjusted, sensitivity):
        inf = row["inference"]
        inf.update(primary_holm2_p=p, sensitivity_holm2_p=sp)
        inf["direction_reproduced"] = (inf["positive_direction"] and p < .05 and
            inf["task_cluster_lower_bound_positive"] and inf["every_leave_one_task_out_positive"])
    reproduced = all(row["inference"]["direction_reproduced"] for row in primary) and all(g["usable"] for g in eligibility.values())
    return dict(complete=True, expected_episodes=600, valid_episodes=600, manifest_hash=MANIFEST_HASH,
                condition_summaries=summaries, overall_candidate_eligibility=eligibility, primary_directions=primary,
                conclusion="complementarity_reproduced" if reproduced else "not_reproduced")


def differences(expected, actual, path="", output=None):
    """Compare every independent field against the corresponding supervisor field.

    Extra supervisor timing/prose fields are outside this success-only rebuild.
    Integer and Boolean comparisons are exact. Float-only differences <=1e-12
    are tolerated, fixed before seeing any heldout outcomes.
    """
    output = [] if output is None else output
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            output.append(dict(path=path, independent=expected, supervisor=actual, issue="object missing/type differs"))
        else:
            for key, value in expected.items():
                if key not in actual:
                    output.append(dict(path=path + "/" + key, independent=value, issue="supervisor field missing"))
                else:
                    differences(value, actual[key], path + "/" + key, output)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            output.append(dict(path=path, independent_length=len(expected), supervisor=actual, issue="list length/type differs"))
        else:
            for index, value in enumerate(expected):
                differences(value, actual[index], path + "/" + str(index), output)
    else:
        same = type(expected) is type(actual) and expected == actual
        if type(expected) is float and type(actual) in (int, float) and not isinstance(actual, bool):
            same = math.isfinite(expected) and math.isfinite(actual) and abs(expected - actual) <= FLOAT_ATOL
        if not same:
            output.append(dict(path=path, independent=expected, supervisor=actual, issue="value/type differs"))
    return output


def run(acceptance_sha, supervisor_sha):
    report = dict(revision="independent-local-confirmation-statistics-v1", status="incomplete", started_unix=time.time(),
                  gpu_used=False, model_or_environment_loaded=False, scientific_files_modified=False,
                  raw_outcomes_opened=False, differences=[], independent_statistics=None, input_evidence={})
    try:
        acceptance, source = read_bound_json(ACCEPTANCE, acceptance_sha)
        report["input_evidence"]["acceptance"] = source
        acceptance_gate(acceptance)
        manifest, source = read_bound_json(CASE_ROOT / "manifest.json", MANIFEST_SHA)
        report["input_evidence"]["manifest"] = source
        manifest_gate(manifest)
        report["input_evidence"]["sources"] = source_identity(manifest, acceptance)
        expected_paths = [receipt_path(CASE_ROOT, cid, scene).resolve() for cid in CONDITIONS for scene in cases()]
        accepted = evidence_map(acceptance.get("receipt_evidence"), expected_paths, "acceptance")
        supervisor, source = read_bound_json(SUPERVISOR, supervisor_sha)
        report["input_evidence"]["supervisor"] = source
        supervisor_gate(supervisor, manifest, accepted)
        report["raw_outcomes_opened"] = True
        records, evidence = load_outcomes(CASE_ROOT, manifest, accepted)
        report["input_evidence"]["episodes"] = evidence
        independent = rebuild_statistics(records, manifest["preregistered_hypotheses"])
        independent["selection_hash"] = manifest["selection_evidence"]["selection_hash"]
        report["independent_statistics"] = independent
        report["differences"] = differences(independent, supervisor)
        require(file_sha(ACCEPTANCE) == acceptance_sha and file_sha(SUPERVISOR) == supervisor_sha and
                file_sha(CASE_ROOT / "manifest.json") == MANIFEST_SHA, "Input reports/manifest changed during rebuild")
        report["status"] = "pass" if not report["differences"] else "failed"
    except (GateError, OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        report["status"] = "incomplete" if not report["raw_outcomes_opened"] else "failed"
        report["blocking_error"] = str(exc)
    report.update(finished_unix=time.time(), python_version=sys.version, numpy_version=np.__version__, bootstrap_contract=dict(
        rng="Two separately initialized numpy.random.Generator(PCG64(20261104)) instances per direction",
        case="20000 x 50 uniform replacement indices into ordered paired float64 differences",
        task="20000 x 10 uniform replacement indices into sorted-task float64 five-case means",
        quantile="numpy.quantile(q=[.025,.975], method='linear'); no rounding",
        compatibility="Equivalent to frozen default_rng(seed).choice(values, size=..., replace=True), p=None; no project helper imported",
        float_comparison_absolute_tolerance=FLOAT_ATOL),
        comparison_scope="All success counts, paired discordances, task differences, exact tests/Holm2, bootstrap intervals, task LOO, eligibility and conclusion; timing/cost/prose excluded",
        script=dict(path=str(Path(__file__).resolve()), sha256=file_sha(__file__)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-sha256", required=True, help="Pin the final independent 600-case PASS report")
    parser.add_argument("--supervisor-sha256", required=True, help="Pin the final frozen supervisor analysis.json")
    parser.add_argument("--output", default=str(HERE / "audit.json"))
    args = parser.parse_args(argv)
    for value in (args.acceptance_sha256, args.supervisor_sha256):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            parser.error("Report SHA arguments must be lowercase SHA256")
    output = Path(args.output).resolve()
    require(output.parent == HERE and output.name not in ("rebuild.py", "test_contract.py", "README.md"), "Output must be a JSON file in the statistics directory")
    require(output.suffix == ".json", "Output must have .json suffix")
    result = run(args.acceptance_sha256, args.supervisor_sha256)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], differences=len(result["differences"]),
                         raw_outcomes_opened=result["raw_outcomes_opened"], report=str(output),
                         conclusion=(result["independent_statistics"] or {}).get("conclusion"))))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
