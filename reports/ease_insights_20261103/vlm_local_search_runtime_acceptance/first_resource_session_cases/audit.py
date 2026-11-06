"""Bounded read-only CPU audit of the earliest new-condition receipts."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np

from preparations.ease_insights_20261103.vlm.analyze import load_episodes
from preparations.ease_insights_20261103.vlm.analyze_confirmation import audit_identities
from preparations.ease_insights_20261103.vlm.recovery import scheduled_nfe
from preparations.ease_insights_20261103.vlm_local_search_v1 import common, selection
from preparations.ease_insights_20261103.vlm_local_search_v1.selection import audit_local_identities
from preparations.idea1_v2_2.common import seed


MANIFEST_HASH = "032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9"
STUDY = ROOT / "results/ease_insights_20261103/vlm_local_search_v1/discovery"
REPORT_DIR = Path(__file__).resolve().parent
REPORT_PATH = REPORT_DIR / "audit.json"
NEW_CONDITIONS = ("full_d10", "local_long")
BASELINE = "jac_anchor"
SAMPLED_RECEIPTS_PER_CONDITION = 3
SOURCE_COUNTS = {"source_files": 46, "analysis_sources": 6, "local_runtime_sources": 9}


def read(path):
    return json.loads(Path(path).read_text())


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_key(case):
    return case["suite"], case["task"], case["init"]


def episode_dir(root, condition, case):
    return (root / condition / "episodes" / case["suite"] /
            ("task%02d" % case["task"]) / ("init%02d" % case["init"]))


def layer_mask_matches(execution, mask):
    return (execution["kv_only_layers"] == mask and
            execution["patched_layers"] == mask and
            execution["qkv_layers"] == [layer for layer in range(1, 19) if layer not in mask] and
            execution["hidden_update_layers"] == [layer for layer in range(1, 18) if layer not in mask] and
            execution["grad_enabled"] is False and execution["selector_calls"] == 0)


def audit_requests(root, condition, case, config, record):
    """Check each sampled request against the frozen schedule and mask contract."""
    directory = episode_dir(root, condition, case)
    rows = [json.loads(line) for line in (directory / "infer.jsonl").read_text().splitlines()]
    history = []
    step = 0
    counts = Counter()
    issues = []
    mask = list(config["skip"])
    for index, row in enumerate(rows):
        full = config["mode"] == "full" or index % (config["period"] + 1) == 0
        expected_mask = [] if full else mask
        expected_capture = mask if full else []
        age = None if not history else step - history[-1]
        gap = None if len(history) < 2 else history[-1] - history[-2]
        ratio = 0.0 if gap is None else age / gap
        if full and mask:
            history = (history + [step])[-2:]
        route = row["route"]
        execution = route["execution_audit"]
        expected_seed = seed(case["suite"], case["task"], case["init"], index)
        expected_noise = hashlib.sha256(
            np.random.default_rng(expected_seed).standard_normal((50, 32), dtype=np.float32).tobytes()
        ).hexdigest()
        expected_d = config["refresh_D"] if full else config["D"]
        try:
            actual_d = scheduled_nfe(row["timing"]["nodes"])
        except (KeyError, TypeError, ValueError) as error:
            actual_d = None
            issues.append({"round": index, "issue": "Invalid native schedule nodes", "detail": str(error)})
        expected_path = "full" if full else "extrapolate"
        valid = (
            row["round"] == index and row["replan"] == index and row["step"] == step and
            row["seed"] == expected_seed and row["model_noise_sha256"] == expected_noise and
            row["execute_length"] == config["R"] == 5 and
            type(row["executed_length"]) is int and 0 < row["executed_length"] <= config["R"] and
            route["refresh"] is full and route["path"] == expected_path and
            route["skip_layers"] == expected_mask and route["capture_layers"] == expected_capture and
            route["compensation"] == config["compensation"] and
            route["has_second_history"] is (gap is not None) and
            abs(route["extrapolation_ratio"] - ratio) < 1e-12 and
            route["history_steps"] == history and row["full_cache_age"] == age and
            route["selector_calls"] == 0 and
            execution["kv_only_layers"] == expected_mask and
            execution["patched_layers"] == expected_mask and
            execution["captured_layers"] == expected_capture and
            execution["qkv_layers"] == [layer for layer in range(1, 19) if layer not in expected_mask] and
            execution["hidden_update_layers"] == [layer for layer in range(1, 18) if layer not in expected_mask] and
            execution["grad_enabled"] is False and execution["selector_calls"] == 0 and
            actual_d == expected_d
        )
        if not valid:
            issues.append({"round": index, "issue": "Action/schedule/mask/noise/history mismatch"})
        counts["full" if full else "draft"] += 1
        step += row["executed_length"]
    result = read(directory / "result.json")
    if (step != result["actions"] or len(rows) != result["predictions"] or
            counts["full"] != result["counts"]["full"] or counts["draft"] != result["counts"]["draft"]):
        issues.append({"issue": "Episode action/request accounting mismatch"})
    return dict(passed=not issues, requests=len(rows), actions=step, scheduled_requests=dict(counts),
                success=record["success"], success_used_as_acceptance_gate=False, issues=issues)


def audit_gate(root, manifest, condition, config, reference_path, baseline_service_path):
    run_gates = sorted((root / condition / "runs").glob("*/gate.json"))
    if len(run_gates) != 1:
        return dict(passed=False, issue="Expected one gate receipt for the active service", count=len(run_gates))
    gate_path = run_gates[0]
    run_dir = gate_path.parent
    gate = read(gate_path)
    protocol = read(run_dir / "protocol.json")
    identity = read(run_dir / "identity.json")
    cross = read(run_dir / "cross_service.json")
    checks = gate.get("checks", [])
    target_suite = "libero_spatial"
    mask = list(config["skip"])
    if condition == "full_d10":
        execution_name = target_suite + "/full_native"
        execution_rows = [check for check in checks if check.get("name") == execution_name]
        execution_passed = (len(execution_rows) == 1 and execution_rows[0].get("passed") is True and
                            execution_rows[0].get("decoded_max_abs") == 0.0 and
                            execution_rows[0].get("raw_max_abs") == 0.0)
    else:
        execution_name = target_suite + "/" + condition + "/actual_execution"
        execution_rows = [check for check in checks if check.get("name") == execution_name]
        execution_passed = (len(execution_rows) == 1 and
                            layer_mask_matches(execution_rows[0]["execution_audit"], mask) and
                            execution_rows[0]["execution_audit"]["captured_layers"] == [])
    actual_path = run_dir / "gate_actions.npz"
    cross_checks = cross.get("checks", [])
    numeric_comparison = []
    with np.load(reference_path, allow_pickle=False) as reference, np.load(actual_path, allow_pickle=False) as actual:
        keys_match = set(reference.files) == set(actual.files)
        if keys_match:
            for key in sorted(reference.files):
                delta = np.abs(actual[key] - reference[key])
                numeric_comparison.append(dict(key=key, max_abs=float(delta.max()),
                    passed=bool(np.allclose(actual[key], reference[key], atol=1e-5, rtol=1e-5))))
    valid = (
        gate.get("passed") is True and gate.get("manifest_hash") == manifest["manifest_hash"] and
        gate.get("condition_ids") == [item["id"] for item in manifest["conditions"]] and
        len(checks) == 332 and protocol.get("manifest_hash") == manifest["manifest_hash"] and
        protocol.get("condition_id") == condition and protocol.get("config") == config and
        protocol.get("fixed_skip_layers") == mask and protocol.get("refresh_D") == config["refresh_D"] and
        protocol.get("real_input_parity_passed") is True and protocol.get("qk_reference_passed") is True and
        protocol.get("checkpoint_sha256") == manifest["checkpoints"]["pi0"]["sha256"] and
        identity.get("condition") == condition and identity.get("manifest_hash") == manifest["manifest_hash"] and
        identity.get("stateful_workers_per_server") == 1 and execution_passed and
        cross.get("passed") is True and len(cross_checks) == 24 and
        all(item.get("passed") is True for item in cross_checks) and
        set(reference.files) == set(actual.files) and len(numeric_comparison) == len(reference.files) and
        all(item["passed"] for item in numeric_comparison)
    )
    return dict(passed=valid, path=str(run_dir), gate_sha256=digest_file(gate_path),
        gate_actions_sha256=digest_file(actual_path), cross_service_sha256=digest_file(run_dir / "cross_service.json"),
        gate_check_count=len(checks), selected_actual_execution_check=execution_name,
        selected_actual_execution_passed=execution_passed, cross_service_check_count=len(cross_checks),
        gate_actions_vs_jac_reference=numeric_comparison, jac_reference_service=str(baseline_service_path),
        issues=[] if valid else ["Gate/service identity or gate-action comparison did not match frozen expectations"])


def main():
    started = time.time()
    checks = []

    def require(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    manifest = common.read_manifest(STUDY / "manifest.json", verify_sources=True)
    require("frozen_manifest", manifest["manifest_hash"] == MANIFEST_HASH,
            actual_manifest_hash=manifest["manifest_hash"], expected_manifest_hash=MANIFEST_HASH)
    inits = selection.check_scope(STUDY, manifest, "discovery")
    require("bounded_discovery_manifest", inits == [36, 37, 38] and manifest["manifest_hash"] == MANIFEST_HASH,
            suites=manifest["suites"], tasks=len(manifest["tasks"]), inits=inits)

    source_identity = {}
    for group, expected_count in SOURCE_COUNTS.items():
        evidence = []
        for filename, expected_sha in manifest[group].items():
            path = ROOT / filename
            actual_sha = common.sha(path)
            evidence.append(dict(path=str(path), sha256=actual_sha,
                                 expected_sha256=expected_sha, passed=actual_sha == expected_sha))
        group_passed = len(evidence) == expected_count and all(item["passed"] for item in evidence)
        source_identity[group] = dict(count=len(evidence), expected_count=expected_count,
                                      passed=group_passed,
                                      mismatches=[item for item in evidence if not item["passed"]])
        require("source_identity:" + group, group_passed, count=len(evidence), expected_count=expected_count)

    for name in ("protocol_evidence", "design_evidence"):
        expected = manifest[name]
        actual_sha = common.sha(expected["path"])
        require("source_identity:" + name, actual_sha == expected["sha256"],
                path=expected["path"], sha256=actual_sha, expected_sha256=expected["sha256"])

    checkpoint_files = []
    for checkpoint in manifest["checkpoints"].values():
        for filename, expected_sha in checkpoint["identity_artifacts"].items():
            actual_sha = common.sha(filename)
            checkpoint_files.append(dict(path=filename, sha256=actual_sha,
                                         expected_sha256=expected_sha, passed=actual_sha == expected_sha))
    require("checkpoint_identity_sources", bool(checkpoint_files) and all(item["passed"] for item in checkpoint_files),
            count=len(checkpoint_files))

    baseline_report_path = ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/audit.json"
    baseline_report = read(baseline_report_path)
    require("previous_jac_anchor120_acceptance", baseline_report.get("status") == "accepted" and
            baseline_report.get("manifest_hash") == MANIFEST_HASH and not baseline_report.get("failed_checks"),
            report=str(baseline_report_path), failed_checks=baseline_report.get("failed_checks"))
    baseline_services = baseline_report.get("runtime_services", [])
    baseline_service_path = baseline_services[0]["path"] if len(baseline_services) == 1 else None
    reference_path = STUDY / "cross_service_reference.npz"
    reference_identity_path = STUDY / "cross_service_reference.json"
    reference_identity = read(reference_identity_path)
    require("jac_gate_action_reference_identity", reference_identity.get("manifest_hash") == MANIFEST_HASH and
            baseline_service_path is not None and
            Path(reference_identity["first_service"]).resolve() == Path(baseline_service_path).resolve(),
            reference=str(reference_identity_path), first_service=reference_identity.get("first_service"))

    ordered_cases = common.cases(manifest, "discovery")
    sampled = {}
    for condition in NEW_CONDITIONS:
        hits = []
        for case in ordered_cases:
            receipt_path = episode_dir(STUDY, condition, case) / "episode_receipt.json"
            if receipt_path.is_file():
                hit = dict(case=case, receipt_path=str(receipt_path), mtime_ns=receipt_path.stat().st_mtime_ns)
                hits.append(hit)
                if len(hits) == SAMPLED_RECEIPTS_PER_CONDITION:
                    break
        sampled[condition] = hits
        require("earliest_committed_receipt_sample:" + condition, len(hits) == SAMPLED_RECEIPTS_PER_CONDITION,
                sampled_count=len(hits), requested_count=SAMPLED_RECEIPTS_PER_CONDITION,
                cases=[item["case"] for item in hits],
                receipt_mtimes_ns=[item["mtime_ns"] for item in hits])
        require("serialized_receipt_order:" + condition,
                all(a["mtime_ns"] < b["mtime_ns"] for a, b in zip(hits, hits[1:])),
                receipt_mtimes_ns=[item["mtime_ns"] for item in hits])

    keys = {case_key(item["case"]) for condition in NEW_CONDITIONS for item in sampled[condition]}
    ordered_sample_keys = [case_key(case) for case in ordered_cases if case_key(case) in keys]
    selected_cases = [next(case for case in ordered_cases if case_key(case) == key) for key in ordered_sample_keys]
    shared_new_cases = (len(sampled["full_d10"]) == SAMPLED_RECEIPTS_PER_CONDITION and
                        len(sampled["local_long"]) == SAMPLED_RECEIPTS_PER_CONDITION and
                        {case_key(row["case"]) for row in sampled["full_d10"]} ==
                        {case_key(row["case"]) for row in sampled["local_long"]})
    require("new_condition_case_overlap", shared_new_cases,
            unique_new_cases=len(selected_cases), rows_per_condition={c: len(sampled[c]) for c in NEW_CONDITIONS})

    rows_to_validate = {}
    for case in selected_cases:
        view = copy.deepcopy(manifest)
        wanted = ("full_d10", "local_long", BASELINE)
        view["conditions"] = [cfg for cfg in manifest["conditions"] if cfg["id"] in wanted]
        view["suites"] = [case["suite"]]
        view["tasks"] = [case["task"]]
        view["discovery_inits"] = [case["init"]]
        loaded, missing, invalid, evidence = load_episodes(STUDY, view)
        rows_to_validate.update(loaded)
        require("receipt_artifacts:" + "/".join(map(str, case_key(case))),
                len(loaded) == 3 and not missing and not invalid,
                expected_condition_rows=3, valid_condition_rows=len(loaded),
                missing=missing, invalid=invalid,
                receipt_hashes=[row for row in evidence])

    identity_invalid = audit_identities(STUDY, manifest, rows_to_validate)
    local_identity_invalid = audit_local_identities(STUDY, manifest, rows_to_validate)
    require("official_case_and_service_identity", not identity_invalid, errors=identity_invalid)
    require("runtime_input_and_noise_identity", not local_identity_invalid, errors=local_identity_invalid)

    paired_inputs = []
    for case in selected_cases:
        key = case_key(case)
        matched = {condition: rows_to_validate[(condition, key)]["initial_gate"]
                   for condition in ("full_d10", "local_long", BASELINE)
                   if (condition, key) in rows_to_validate}
        identity_values = {(value["input_hash"], value["seed"]) for value in matched.values()}
        paired_inputs.append(dict(case=case, conditions=sorted(matched),
                                  input_hashes={condition: value["input_hash"] for condition, value in matched.items()},
                                  noise_seeds={condition: value["seed"] for condition, value in matched.items()},
                                  passed=len(matched) == 3 and len(identity_values) == 1))
    require("same_case_input_and_seed_pairing", len(paired_inputs) == len(selected_cases) and
            all(item["passed"] for item in paired_inputs), pairs=paired_inputs)

    request_checks = {}
    if not identity_invalid and not local_identity_invalid:
        for condition in (*NEW_CONDITIONS, BASELINE):
            request_checks[condition] = []
            config = next(cfg for cfg in manifest["conditions"] if cfg["id"] == condition)
            for case in selected_cases:
                record = rows_to_validate.get((condition, case_key(case)))
                if record is None:
                    item = dict(case=case, passed=False, issue="No receipt record survived strict identity checks")
                else:
                    item = dict(case=case, **audit_requests(STUDY, condition, case, config, record))
                request_checks[condition].append(item)
            require("action_schedule_mask:" + condition,
                    all(item.get("passed", False) for item in request_checks[condition]),
                    cases=request_checks[condition])
    else:
        for condition in (*NEW_CONDITIONS, BASELINE):
            request_checks[condition] = []
            require("action_schedule_mask:" + condition, False,
                    issue="Skipped because selected receipts failed strict identity validation")

    gates = {}
    for condition in NEW_CONDITIONS:
        config = next(cfg for cfg in manifest["conditions"] if cfg["id"] == condition)
        gates[condition] = audit_gate(STUDY, manifest, condition, config, reference_path, baseline_service_path)
        require("passed_service_gate_and_actual_actions:" + condition, gates[condition].get("passed", False),
                result=gates[condition])

    failed = [item for item in checks if not item["passed"]]
    results = {}
    for condition in (*NEW_CONDITIONS, BASELINE):
        results[condition] = []
        for case in selected_cases:
            record = rows_to_validate.get((condition, case_key(case)))
            results[condition].append(dict(case=case,
                receipt_sha256=(digest_file(episode_dir(STUDY, condition, case) / "episode_receipt.json")
                                if (episode_dir(STUDY, condition, case) / "episode_receipt.json").is_file() else None),
                success=None if record is None else record["success"],
                successful_outcome_required=False,
                initial_gate=None if record is None else record["initial_gate"],
                request_audit=next((item for item in request_checks.get(condition, [])
                                    if case_key(item["case"]) == case_key(case)), None)))

    result = dict(revision="first-resource-session-bounded-case-audit-v1",
        status="sample_pass" if not failed else "issues", started_unix=started, finished_unix=time.time(),
        manifest_hash=manifest["manifest_hash"], output_root=str(STUDY),
        scope=dict(conditions=list(NEW_CONDITIONS), baseline=BASELINE,
            new_condition_receipts_per_condition=SAMPLED_RECEIPTS_PER_CONDITION,
            unique_new_cases=len(selected_cases), baseline_receipts=len(selected_cases),
            sampled_case_order="Earliest committed receipts in the immutable single-worker cases() order; receipt mtimes are recorded and checked strictly increasing within each condition.",
            full_episode_scan=False, full_720_case_validation=False, full_332_gate_recomputation=False,
            video_read=False, gpu_or_model_started=False, success_as_acceptance_gate=False,
            permitted_writes=[str(REPORT_DIR)],
            reused_read_only_validators=["common.read_manifest", "selection.check_scope", "analyze.load_episodes",
                "analyze_confirmation.audit_identities", "selection.audit_local_identities", "recovery.scheduled_nfe"],
            avoided_mutating_or_full_scan_paths=["runtime_acceptance/audit.py:main", "selection.load_stage",
                "worker.finalize", "worker.completed", "recovery.recover_case", "runner.run"]),
        checks=checks, failed_checks=failed,
        source_identity=source_identity,
        checkpoint_identity_sources=checkpoint_files,
        sampled_receipts={condition: sampled[condition] for condition in NEW_CONDITIONS},
        case_results=results,
        request_audits=request_checks,
        new_service_gates=gates,
        baseline=dict(previous_acceptance_report=str(baseline_report_path),
            previous_acceptance_report_sha256=digest_file(baseline_report_path),
            prior_service_path=baseline_service_path,
            cross_service_reference_identity=reference_identity),
        audit_script=dict(path=str(Path(__file__).resolve()), sha256=digest_file(__file__)),
        limits=["Only the first three committed receipts per new condition and their same-case Jac receipts were validated.",
            "Receipt success/failure values are retained as outcomes and never used to accept or reject the implementation.",
            "Gate JSON passed status, sampled-condition actual-execution mask, and actual gate action arrays were checked; all other gate calculations were not recomputed.",
            "The discovery matrix, unsampled episodes, videos, full gate calculation, and model runtime were not examined."])
    REPORT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"], failed_checks=failed,
        sampled_cases=len(selected_cases), rows={condition: len(rows) for condition, rows in results.items()},
        report=str(REPORT_PATH)), indent=2))


if __name__ == "__main__":
    main()
