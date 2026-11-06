"""Strict read-only CPU acceptance of the frozen 720-case discovery matrix."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np

from preparations.ease_insights_20261103.vlm_local_search_v1 import common, selection
from preparations.ease_insights_20261103.vlm_local_search_v1.common import runtime_revision


MANIFEST_HASH = "032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9"
DEFAULT_ROOT = ROOT / "results/ease_insights_20261103/vlm_local_search_v1/discovery"
REPORT_DIR = Path(__file__).resolve().parent
REPORT_PATH = REPORT_DIR / "audit.json"
CONDITION_IDS = ("full_d10", "jac_anchor", "local_spatial", "local_object", "local_goal", "local_long")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_CASES_PER_CONDITION = 120
EXPECTED_TOTAL_CASES = 720
EXPECTED_SOURCE_COUNTS = {"source_files": 46, "analysis_sources": 6, "local_runtime_sources": 9}
GATE_SUFFIXES = (
    "actual_capture", "residual_identity", "same_obs_reconstruct_full_d2", "reconstructed_kv",
    "actual_execution", "pure_native", "extrapolation_reference", "explicit_noise_repeat",
)
FROZEN_GATE_ATOL = 1e-5
FROZEN_GATE_RTOL = 1e-5


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_path(filename):
    path = Path(filename)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def request_auditor():
    """Import only the reusable pure request validator; do not invoke its main."""
    path = REPORT_DIR.parent / "first_resource_session_cases" / "audit.py"
    spec = importlib.util.spec_from_file_location("first_resource_case_request_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the prepared per-request validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def stage_summary(audit):
    records = audit.get("records", {})
    counts = Counter(condition for condition, _ in records)
    missing = audit.get("missing", [])
    invalid = audit.get("invalid", [])
    unexpected = audit.get("unexpected", [])
    reasons = []
    if audit.get("expected") != EXPECTED_TOTAL_CASES:
        reasons.append("Expected discovery universe is not 720")
    if len(records) != EXPECTED_TOTAL_CASES:
        reasons.append("Valid receipt record count is not 720")
    if any(counts[condition] != EXPECTED_CASES_PER_CONDITION for condition in CONDITION_IDS):
        reasons.append("At least one condition does not have exactly 120 valid receipts")
    if missing:
        reasons.append("Missing receipts remain")
    if invalid:
        reasons.append("Invalid receipts or identities remain")
    if unexpected:
        reasons.append("Unexpected completed cases remain")
    if audit.get("complete") is not True:
        reasons.append("Frozen load_stage complete flag is false")
    return dict(passed=(not reasons), expected=audit.get("expected"), valid=len(records),
        complete=audit.get("complete"), per_condition={condition: counts[condition] for condition in CONDITION_IDS},
        missing_count=len(missing), invalid_count=len(invalid), unexpected_count=len(unexpected),
        missing=missing, invalid=invalid, unexpected=unexpected, reasons=reasons)


def final_status(stage_passed, failed_checks):
    if not stage_passed:
        return "incomplete"
    return "failed" if failed_checks else "pass"


def gate_names(condition_ids, suites):
    names = set()
    for suite in suites:
        names.add(suite + "/full_native")
        for condition in condition_ids:
            names.update(suite + "/" + condition + "/policy_round" + str(index) for index in range(6))
            names.add(suite + "/" + condition + "/schedule")
            if condition != "full_d10":
                names.update(suite + "/" + condition + "/" + suffix for suffix in GATE_SUFFIXES)
    return names


def layer_mask_matches(execution, mask):
    return (execution.get("kv_only_layers") == mask and execution.get("patched_layers") == mask and
            execution.get("qkv_layers") == [layer for layer in range(1, 19) if layer not in mask] and
            execution.get("hidden_update_layers") == [layer for layer in range(1, 18) if layer not in mask] and
            execution.get("captured_layers") == [] and execution.get("grad_enabled") is False and
            execution.get("selector_calls") == 0)


def verify_gate_record(gate, manifest, gate_path):
    checks = gate.get("checks", [])
    names = [row.get("name") for row in checks]
    expected_names = gate_names([item["id"] for item in manifest["conditions"]], manifest["suites"])
    issues = []
    if gate.get("passed") is not True:
        issues.append("gate.json passed is not true")
    if gate.get("manifest_hash") != manifest["manifest_hash"]:
        issues.append("Gate manifest hash differs")
    if gate.get("condition_ids") != [item["id"] for item in manifest["conditions"]]:
        issues.append("Gate condition order differs")
    if gate.get("atol") != FROZEN_GATE_ATOL or gate.get("rtol") != FROZEN_GATE_RTOL:
        issues.append("Gate numerical tolerance differs from frozen 1e-5")
    if len(checks) != 332 or len(set(names)) != 332 or set(names) != expected_names:
        issues.append("Gate check name/count coverage differs from the frozen 332 checks")
    if any(row.get("passed") is not True for row in checks):
        issues.append("At least one persisted gate check did not pass")
    checks_by_name = {row.get("name"): row for row in checks}
    for suite in manifest["suites"]:
        full_native = checks_by_name.get(suite + "/full_native")
        if full_native is None or full_native.get("passed") is not True:
            issues.append("Missing or failed full_native gate: " + suite)
        for config in manifest["conditions"]:
            condition = config["id"]
            if condition == "full_d10":
                continue
            name = suite + "/" + condition + "/actual_execution"
            check = checks_by_name.get(name)
            if (check is None or check.get("passed") is not True or
                    not layer_mask_matches(check.get("execution_audit", {}), list(config["skip"]))):
                issues.append("Actual layer execution mask mismatch: " + name)
    return dict(passed=not issues, path=str(Path(gate_path).resolve()),
        sha256=sha256(gate_path), check_count=len(checks), unique_name_count=len(set(names)),
        persisted_check_failures=sum(row.get("passed") is not True for row in checks),
        atol=gate.get("atol"), rtol=gate.get("rtol"), issues=issues)


def verify_manifest_sources(manifest):
    groups = {}
    for name, expected_count in EXPECTED_SOURCE_COUNTS.items():
        entries = []
        for filename, expected in manifest.get(name, {}).items():
            path = project_path(filename)
            try:
                actual = common.sha(path)
                error = None
            except OSError as exc:
                actual, error = None, repr(exc)
            entries.append(dict(path=str(path), sha256=actual, expected_sha256=expected,
                                passed=actual == expected, error=error))
        groups[name] = dict(count=len(entries), expected_count=expected_count,
            passed=len(entries) == expected_count and all(row["passed"] for row in entries), entries=entries)
    checkpoint_entries = []
    for name, checkpoint in manifest.get("checkpoints", {}).items():
        for filename, expected in checkpoint.get("identity_artifacts", {}).items():
            path = project_path(filename)
            try:
                actual = common.sha(path)
                error = None
            except OSError as exc:
                actual, error = None, repr(exc)
            checkpoint_entries.append(dict(checkpoint=name, path=str(path), sha256=actual,
                                           expected_sha256=expected, passed=actual == expected, error=error))
    protocol_entries = []
    for name in ("protocol_evidence", "design_evidence"):
        item = manifest[name]
        path = project_path(item["path"])
        try:
            actual = common.sha(path)
            error = None
        except OSError as exc:
            actual, error = None, repr(exc)
        protocol_entries.append(dict(name=name, path=str(path), sha256=actual,
                                     expected_sha256=item["sha256"], passed=actual == item["sha256"], error=error))
    all_passed = (all(group["passed"] for group in groups.values()) and
                  bool(checkpoint_entries) and all(row["passed"] for row in checkpoint_entries) and
                  all(row["passed"] for row in protocol_entries))
    return dict(passed=all_passed, groups=groups, checkpoint_identity_artifacts=checkpoint_entries,
                protocol_and_design_evidence=protocol_entries)


def explicit_input_hashes(request_audit_path, report_path):
    paths = [
        Path(__file__).resolve(), Path(request_audit_path).resolve(),
        (REPORT_DIR.parent / "audit.py").resolve(),
        ROOT / "preparations/ease_insights_20261103/vlm_local_search_v1/common.py",
        ROOT / "preparations/ease_insights_20261103/vlm_local_search_v1/selection.py",
        ROOT / "preparations/ease_insights_20261103/vlm/analyze.py",
        ROOT / "preparations/ease_insights_20261103/vlm/analyze_confirmation.py",
        ROOT / "preparations/ease_insights_20261103/vlm/recovery.py",
        ROOT / "preparations/idea1_v2_1/runtime_client.py",
        ROOT / "preparations/idea1_v2_2/common.py",
        ROOT / "preparations/idea1_v3_1/common.py",
        ROOT / "preparations/ease_insights_20261103/vlm_local_search_v1/runner.py",
        ROOT / "preparations/ease_insights_20261103/vlm_local_search_v1/worker.py",
        ROOT / "preparations/ease_insights_20261103/vlm_local_search_v1/server.py",
        ROOT / "preparations/ease_insights_20261103/vlm_local_search_v1/gate.py",
        DEFAULT_ROOT / "manifest.json",
        ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/audit.json",
    ]
    result = []
    seen = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            result.append(dict(path=str(path), sha256=sha256(path), exists=True))
        except OSError as exc:
            result.append(dict(path=str(path), sha256=None, exists=False, error=repr(exc)))
    return result


def same_run_path(value, expected_parent):
    path = Path(value).resolve()
    parent = Path(expected_parent).resolve()
    try:
        path.relative_to(parent)
        return path
    except ValueError:
        return None


def verify_source_bank(protocol, manifest):
    expected = {}
    for checkpoint in manifest["checkpoints"].values():
        for filename, value in checkpoint["identity_artifacts"].items():
            expected[Path(filename).resolve()] = value
    issues = []
    entries = []
    for path_key, hash_key in (("source_manifest", "source_manifest_sha256"),
                               ("provenance", "provenance_sha256")):
        path = Path(protocol.get(path_key, ""))
        if not path.is_absolute():
            issues.append("Service " + path_key + " path is not absolute")
            continue
        path = path.resolve()
        expected_hash = expected.get(path)
        try:
            actual_hash = common.sha(path)
            error = None
        except OSError as exc:
            actual_hash, error = None, repr(exc)
        passed = (expected_hash is not None and protocol.get(hash_key) == expected_hash and
                  actual_hash == expected_hash)
        entries.append(dict(name=path_key, path=str(path), sha256=actual_hash,
                            expected_sha256=expected_hash, protocol_sha256=protocol.get(hash_key),
                            passed=passed, error=error))
        if not passed:
            issues.append("Service source-bank evidence mismatch: " + path_key)
    return dict(passed=not issues, entries=entries, issues=issues)


def compare_gate_actions(actual_path, reference_path, cross_receipt):
    issues = []
    comparisons = []
    atol, rtol = cross_receipt.get("atol"), cross_receipt.get("rtol")
    if atol != FROZEN_GATE_ATOL or rtol != FROZEN_GATE_RTOL:
        issues.append("Cross-service tolerance differs from frozen 1e-5")
        atol, rtol = FROZEN_GATE_ATOL, FROZEN_GATE_RTOL
    try:
        with np.load(reference_path, allow_pickle=False) as reference, np.load(actual_path, allow_pickle=False) as actual:
            reference_keys, actual_keys = set(reference.files), set(actual.files)
            if reference_keys != actual_keys:
                issues.append("Gate-action array keys differ from Jac reference")
            else:
                for name in sorted(reference_keys):
                    expected, observed = reference[name], actual[name]
                    same_shape = expected.shape == observed.shape
                    if not same_shape:
                        passed, max_abs = False, None
                    else:
                        delta = np.abs(observed - expected)
                        max_abs = float(delta.max()) if delta.size else 0.0
                        passed = bool(np.allclose(observed, expected, atol=atol, rtol=rtol))
                    comparisons.append(dict(name=name, shape_match=same_shape, max_abs=max_abs, passed=passed))
                    if not passed:
                        issues.append("Gate-action mismatch: " + name)
    except (OSError, ValueError, KeyError) as exc:
        issues.append("Cannot read or compare gate-action arrays: " + repr(exc))
    cross_checks = cross_receipt.get("checks", [])
    if (cross_receipt.get("passed") is not True or len(cross_checks) != 24 or
            any(item.get("passed") is not True for item in cross_checks)):
        issues.append("Persisted cross_service receipt is absent, failed, or incomplete")
    return dict(passed=not issues, atol=atol, rtol=rtol, cross_check_count=len(cross_checks),
                array_count=len(comparisons), comparisons=comparisons, issues=issues)


def validate_service_run(root, condition, manifest, run_dir, marker_run_dir, reference_path):
    run_dir = Path(run_dir).resolve()
    run_identity_path = run_dir / "identity.json"
    protocol_path = run_dir / "protocol.json"
    gate_path = run_dir / "gate.json"
    cross_path = run_dir / "cross_service.json"
    errors = []
    try:
        identity = read_json(run_identity_path)
        protocol = read_json(protocol_path)
        gate = read_json(gate_path)
        cross = read_json(cross_path)
    except (OSError, ValueError, TypeError) as exc:
        return dict(passed=False, condition=condition, run_dir=str(run_dir),
                    issues=["Serving run lacks readable identity/protocol/gate/cross-service receipt: " + repr(exc)])
    config = next(item for item in manifest["conditions"] if item["id"] == condition)
    gpu_binding = identity.get("gpu_binding", {})
    protocol_ok = (
        identity.get("manifest_hash") == manifest["manifest_hash"] and identity.get("condition") == condition and
        identity.get("stateful_workers_per_server") == 1 and
        identity.get("technical_runtime_revision") == runtime_revision(manifest) and
        protocol.get("manifest_hash") == manifest["manifest_hash"] and protocol.get("condition_id") == condition and
        protocol.get("config") == config and protocol.get("fixed_skip_layers") == config["skip"] and
        protocol.get("refresh_D") == config["refresh_D"] and protocol.get("real_input_parity_passed") is True and
        protocol.get("qk_reference_passed") is True and
        protocol.get("checkpoint_sha256") == manifest["checkpoints"]["pi0"]["sha256"] and
        protocol.get("gpu_uuid") == gpu_binding.get("gpu_uuid") and
        run_dir.parent.name == "runs" and run_dir.parent.parent.name == condition)
    if not protocol_ok:
        errors.append("Run identity or service protocol differs from the frozen manifest/condition")
    source_bank = verify_source_bank(protocol, manifest)
    if not source_bank["passed"]:
        errors.extend(source_bank["issues"])
    gate_result = verify_gate_record(gate, manifest, gate_path)
    if not gate_result["passed"]:
        errors.extend(gate_result["issues"])
    action_result = compare_gate_actions(run_dir / "gate_actions.npz", reference_path, cross)
    if not action_result["passed"]:
        errors.extend(action_result["issues"])
    marker_result = None
    if marker_run_dir is not None and run_dir == Path(marker_run_dir).resolve():
        try:
            result_receipt = read_json(run_dir / "result.json")
            marker_path = root / condition / "discovery_shard00of01_complete.json"
            marker = read_json(marker_path)
            marker_result = dict(passed=(result_receipt == marker and result_receipt.get("complete") is True and
                result_receipt.get("manifest_hash") == manifest["manifest_hash"] and
                result_receipt.get("condition") == condition and
                result_receipt.get("completed_after") == EXPECTED_CASES_PER_CONDITION and
                Path(result_receipt.get("logs", "")).resolve() == run_dir),
                marker_path=str(marker_path), marker_sha256=sha256(marker_path),
                result_sha256=sha256(run_dir / "result.json"))
            if not marker_result["passed"]:
                errors.append("Completion marker does not exactly bind to the serving run result")
        except (OSError, ValueError, TypeError) as exc:
            marker_result = dict(passed=False, issues=[repr(exc)])
            errors.append("Cannot validate completion marker binding: " + repr(exc))
    return dict(passed=not errors, condition=condition, run_dir=str(run_dir),
        run_identity_sha256=sha256(run_identity_path), protocol_sha256=sha256(protocol_path),
        protocol=dict(server_pid=protocol.get("server_pid"), gpu_uuid=protocol.get("gpu_uuid"),
                      config=protocol.get("config"), source_manifest=protocol.get("source_manifest"),
                      source_manifest_sha256=protocol.get("source_manifest_sha256"),
                      provenance=protocol.get("provenance"), provenance_sha256=protocol.get("provenance_sha256")),
        protocol_identity_passed=protocol_ok, source_bank=source_bank, gate=gate_result,
        gate_actions_sha256=sha256(run_dir / "gate_actions.npz"),
        cross_service_sha256=sha256(cross_path), gate_actions_vs_jac_reference=action_result,
        completion_marker_binding=marker_result, issues=errors)


def audit_condition_services(root, manifest, stage_audit, reference_identity, reference_path):
    service_results = {}
    all_issues = []
    for condition in CONDITION_IDS:
        marker_path = root / condition / "discovery_shard00of01_complete.json"
        condition_issues = []
        try:
            marker = read_json(marker_path)
        except (OSError, ValueError, TypeError) as exc:
            service_results[condition] = dict(passed=False, marker_path=str(marker_path),
                                              issues=["Missing or invalid condition completion marker: " + repr(exc)])
            all_issues.append(dict(condition=condition, issue="Missing or invalid completion marker"))
            continue
        marker_ok = (marker.get("complete") is True and marker.get("manifest_hash") == manifest["manifest_hash"] and
                     marker.get("condition") == condition and
                     (marker.get("completed_after") == EXPECTED_CASES_PER_CONDITION or
                      marker.get("already_completed") == EXPECTED_CASES_PER_CONDITION))
        if not marker_ok:
            condition_issues.append("Condition completion marker does not certify all 120 cases")
        marker_run_dir = None
        if marker.get("logs"):
            marker_run_dir = same_run_path(marker["logs"], root / condition / "runs")
            if marker_run_dir is None:
                condition_issues.append("Completion marker points outside its condition runs directory")

        used_pids = set()
        used_gpus = set()
        pid_cases = Counter()
        for case in stage_audit["scenes"]:
            key = (case["suite"], case["task"], case["init"])
            identity_path = (root / condition / "episodes" / case["suite"] /
                             ("task%02d" % case["task"]) / ("init%02d" % case["init"]) / "identity.json")
            try:
                identity = read_json(identity_path)
                protocol = identity["protocol"]
                pid = protocol["server_pid"]
                gpu = protocol["gpu_uuid"]
                if type(pid) is not int or not gpu or protocol.get("condition_id") != condition:
                    raise ValueError("Episode protocol lacks server PID/GPU/condition identity")
                used_pids.add(pid)
                used_gpus.add(gpu)
                pid_cases[pid] += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                condition_issues.append("Episode service identity missing for " + repr(key) + ": " + repr(exc))

        run_protocol_paths = sorted((root / condition / "runs").glob("*/protocol.json"))
        pid_to_runs = {}
        for protocol_path in run_protocol_paths:
            try:
                protocol = read_json(protocol_path)
            except (OSError, ValueError, TypeError):
                continue
            pid = protocol.get("server_pid")
            if pid in used_pids and protocol.get("condition_id") == condition:
                pid_to_runs.setdefault(pid, []).append(protocol_path.parent.resolve())
        selected_runs = set()
        pid_bindings = {}
        for pid in sorted(used_pids):
            matches = sorted(set(pid_to_runs.get(pid, [])))
            pid_bindings[str(pid)] = [str(path) for path in matches]
            if len(matches) != 1:
                condition_issues.append("Episode server PID does not bind to exactly one run protocol: " + str(pid))
            selected_runs.update(matches)
        if marker_run_dir is not None:
            selected_runs.add(marker_run_dir)
            if marker.get("new_episodes", 0) > 0:
                try:
                    marker_pid = read_json(marker_run_dir / "protocol.json").get("server_pid")
                    if marker_pid not in used_pids:
                        condition_issues.append("Completion-marker run PID is not referenced by any episode identity")
                except (OSError, ValueError, TypeError):
                    condition_issues.append("Completion-marker run has no readable protocol PID")
        if condition == "jac_anchor":
            reference_service = same_run_path(reference_identity.get("first_service", ""), root / condition / "runs")
            if reference_service is None:
                condition_issues.append("Jac cross-service reference points outside Jac runs")
            else:
                selected_runs.add(reference_service)
        if not used_pids:
            condition_issues.append("No episode receipt is bound to a serving PID")
        run_results = []
        for run_dir in sorted(selected_runs, key=str):
            run_pid = None
            try:
                run_pid = read_json(run_dir / "protocol.json").get("server_pid")
            except (OSError, ValueError, TypeError):
                pass
            if run_pid is not None and run_pid not in used_pids and run_dir != marker_run_dir and condition != "jac_anchor":
                continue
            run_results.append(validate_service_run(root, condition, manifest, run_dir, marker_run_dir,
                                                    reference_path))
        if not run_results:
            condition_issues.append("No receipt-linked serving run was validated")
        if any(not row["passed"] for row in run_results):
            condition_issues.append("At least one receipt-linked serving run failed gate/source/action validation")
        condition_passed = marker_ok and not condition_issues and all(row["passed"] for row in run_results)
        service_results[condition] = dict(passed=condition_passed, marker_path=str(marker_path),
            marker_sha256=sha256(marker_path), marker_fields={key: marker.get(key) for key in
                ("complete", "condition", "manifest_hash", "completed_after", "already_completed", "logs", "new_episodes")},
            used_server_pids=sorted(used_pids), used_gpu_uuids=sorted(used_gpus),
            cases_by_server_pid=dict(pid_cases), pid_run_bindings=pid_bindings,
            validated_run_count=len(run_results), runs=run_results, issues=condition_issues)
        if not condition_passed:
            all_issues.append(dict(condition=condition, issues=condition_issues))
    return dict(passed=not all_issues, conditions=service_results, issues=all_issues)


def baseline_reference(root, manifest):
    identity_path = root / "cross_service_reference.json"
    array_path = root / "cross_service_reference.npz"
    prior_path = ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/audit.json"
    errors = []
    try:
        identity = read_json(identity_path)
        prior = read_json(prior_path)
    except (OSError, ValueError, TypeError) as exc:
        return dict(passed=False, issues=["Cannot read Jac reference or prior acceptance: " + repr(exc)])
    expected_array_hashes = {item["path"]: item["sha256"] for item in prior.get("input_evidence", [])}
    actual_array_hash = sha256(array_path)
    expected_array_hash = next((value for path, value in expected_array_hashes.items()
                                if Path(path).resolve() == array_path.resolve()), None)
    if prior.get("status") != "accepted" or prior.get("manifest_hash") != manifest["manifest_hash"] or prior.get("failed_checks"):
        errors.append("Prior Jac120 runtime acceptance is not accepted for this manifest")
    if identity.get("manifest_hash") != manifest["manifest_hash"]:
        errors.append("Jac reference manifest hash differs")
    if expected_array_hash is None or actual_array_hash != expected_array_hash:
        errors.append("Jac reference NPZ differs from the accepted source evidence")
    if not identity.get("first_service"):
        errors.append("Jac reference does not identify its creator service")
    return dict(passed=not errors, path=str(identity_path), sha256=sha256(identity_path),
        identity=identity, npz_path=str(array_path), npz_sha256=actual_array_hash,
        expected_npz_sha256=expected_array_hash, prior_acceptance_path=str(prior_path),
        prior_acceptance_sha256=sha256(prior_path), issues=errors)


def run_audit(root):
    started = time.time()
    root = Path(root).resolve()
    checks = []
    failures = []
    request_module, request_path = request_auditor()
    input_hashes = explicit_input_hashes(request_path, Path(__file__).resolve())
    result = dict(revision="full-local-search-discovery-read-only-acceptance-v1",
        status="incomplete", started_unix=started, finished_unix=None,
        manifest_hash=None, root=str(root), input_source_hashes=input_hashes,
        source_identity=None, stage=None, requests=None, episode_success_failures=None,
        initial_input_pairing=None, serving_runs=None,
        scope=dict(expected_episodes=EXPECTED_TOTAL_CASES, expected_conditions=list(CONDITION_IDS),
            expected_cases_per_condition=EXPECTED_CASES_PER_CONDITION, request_rows_checked=0,
            serving_run_gates_checked=0, gate_recomputations=0, gpu_used=False, model_started=False,
            scientific_selection_or_statistics=False, success_is_acceptance_gate=False,
            actual_internal_nfe_exported=False, actual_nfe_limitation="Only the frozen exported native time-node schedule is validated; the backend does not export its internal actual_NFE counter.",
            write_directory=str(REPORT_DIR), read_only_validators=["common.read_manifest", "selection.load_stage",
                "selection.audit_identities", "selection.audit_local_identities", "analyze.load_episodes",
                "analyze_confirmation.audit_identities", "recovery.scheduled_nfe", "first_resource_session_cases.audit_requests"],
            prohibited_mutators_not_called=["worker.completed", "worker.finalize", "recovery.recover_case", "runner.run"]))

    if root != DEFAULT_ROOT.resolve():
        failures.append(dict(name="frozen_root", issue="Refusing to audit a different discovery output root"))
        result["status"] = "failed"
        result["failed_checks"] = failures
        result["finished_unix"] = time.time()
        return result
    manifest_path = root / "manifest.json"
    try:
        manifest = common.read_manifest(manifest_path, verify_sources=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append(dict(name="frozen_manifest", issue=repr(exc)))
        result["status"] = "failed"
        result["failed_checks"] = failures
        result["finished_unix"] = time.time()
        return result
    result["manifest_hash"] = manifest["manifest_hash"]
    if manifest["manifest_hash"] != MANIFEST_HASH:
        failures.append(dict(name="frozen_manifest_hash", actual=manifest["manifest_hash"], expected=MANIFEST_HASH))
    source_identity = verify_manifest_sources(manifest)
    result["source_identity"] = source_identity
    if not source_identity["passed"]:
        failures.append(dict(name="frozen_source_identity", issue="At least one frozen source/evidence hash differs"))
    if failures:
        result["status"] = "failed"
        result["failed_checks"] = failures
        result["finished_unix"] = time.time()
        return result

    try:
        stage = selection.load_stage(root, "discovery")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append(dict(name="load_stage", issue=repr(exc)))
        result["status"] = "failed"
        result["failed_checks"] = failures
        result["finished_unix"] = time.time()
        return result
    stage_check = stage_summary(stage)
    result["stage"] = stage_check
    checks.append(dict(name="complete_720_discovery_receipts", **stage_check))
    if not stage_check["passed"]:
        result["status"] = "incomplete"
        result["checks"] = checks
        result["failed_checks"] = [check for check in checks if not check["passed"]]
        result["receipt_evidence"] = stage.get("episode_evidence", [])
        result["finished_unix"] = time.time()
        return result

    records = stage["records"]
    false_outcomes = {condition: [record["case"] for (cid, _), record in records.items()
                                  if cid == condition and record["success"] is False]
                      for condition in CONDITION_IDS}
    result["episode_success_failures"] = false_outcomes

    request_counts = {condition: dict(cases=0, passed_cases=0, requests=0, actions=0,
                                      scheduled_full=0, scheduled_fast=0) for condition in CONDITION_IDS}
    request_failures = []
    for (condition, _), record in sorted(records.items(), key=lambda item: (item[0][0], item[0][1])):
        config = next(item for item in manifest["conditions"] if item["id"] == condition)
        try:
            audited = request_module.audit_requests(root, condition, record["case"], config, record)
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            audited = dict(passed=False, requests=0, actions=0, scheduled_requests={},
                           success=record["success"], success_used_as_acceptance_gate=False,
                           issues=[{"issue": "Per-request validator error", "detail": repr(exc)}])
        summary = request_counts[condition]
        summary["cases"] += 1
        summary["passed_cases"] += int(audited.get("passed", False))
        summary["requests"] += audited.get("requests", 0)
        summary["actions"] += audited.get("actions", 0)
        summary["scheduled_full"] += audited.get("scheduled_requests", {}).get("full", 0)
        summary["scheduled_fast"] += audited.get("scheduled_requests", {}).get("draft", 0)
        if not audited.get("passed", False):
            request_failures.append(dict(condition=condition, case=record["case"], issues=audited.get("issues", [])))
    result["requests"] = dict(by_condition=request_counts,
        cases_checked=sum(item["cases"] for item in request_counts.values()),
        requests_checked=sum(item["requests"] for item in request_counts.values()),
        action_rows_checked=sum(item["actions"] for item in request_counts.values()),
        failed_case_requests=request_failures,
        passed=not request_failures)
    result["scope"]["request_rows_checked"] = result["requests"]["requests_checked"]
    if request_failures:
        failures.append(dict(name="all_720_request_schedule_mask_noise_action_checks", count=len(request_failures),
                             errors=request_failures))

    input_pair_errors = []
    paired_cases = 0
    for case in stage["scenes"]:
        key = (case["suite"], case["task"], case["init"])
        values = [records[(condition, key)]["initial_gate"] for condition in CONDITION_IDS]
        identities = {(value["input_hash"], value["seed"]) for value in values}
        paired_cases += 1
        if len(identities) != 1:
            input_pair_errors.append(dict(case=case, identities=sorted([list(value) for value in identities])))
    result["initial_input_pairing"] = dict(cases_checked=paired_cases, passed=not input_pair_errors,
                                           failed_cases=input_pair_errors)
    if input_pair_errors:
        failures.append(dict(name="all_120_initial_inputs_paired_across_conditions", errors=input_pair_errors))

    reference = baseline_reference(root, manifest)
    result["jac_reference"] = reference
    if not reference["passed"]:
        failures.append(dict(name="accepted_jac_gate_action_reference", errors=reference["issues"]))
    else:
        serving = audit_condition_services(root, manifest, stage, reference["identity"], Path(reference["npz_path"]))
        result["serving_runs"] = serving
        result["scope"]["serving_run_gates_checked"] = sum(
            row.get("validated_run_count", 0) for row in serving["conditions"].values())
        if not serving["passed"]:
            failures.append(dict(name="receipt_bound_service_gate_source_mask_cross_service", errors=serving["issues"]))

    checks.append(dict(name="all_720_request_schedule_mask_noise_action_checks",
                       passed=not request_failures, cases=result["requests"]["cases_checked"],
                       requests=result["requests"]["requests_checked"], errors=request_failures))
    checks.append(dict(name="all_120_initial_inputs_paired_across_conditions",
                       passed=not input_pair_errors, cases=paired_cases, errors=input_pair_errors))
    if result.get("serving_runs") is not None:
        checks.append(dict(name="receipt_bound_service_gate_source_mask_cross_service",
                           passed=result["serving_runs"]["passed"],
                           conditions_checked=len(result["serving_runs"]["conditions"]),
                           serving_run_gates_checked=result["scope"]["serving_run_gates_checked"],
                           errors=result["serving_runs"]["issues"]))
    if not reference["passed"]:
        checks.append(dict(name="accepted_jac_gate_action_reference", passed=False, errors=reference["issues"]))
    result["checks"] = checks
    result["failed_checks"] = failures
    result["receipt_evidence"] = stage.get("episode_evidence", [])
    result["status"] = final_status(stage_check["passed"], failures)
    result["audit_script"] = dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__))
    result["finished_unix"] = time.time()
    return result


def write_report(result, output):
    output = Path(output).resolve()
    if output.parent != REPORT_DIR.resolve():
        raise ValueError("Report output must remain inside the assigned full_discovery directory")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.root)
    except Exception as exc:
        result = dict(revision="full-local-search-discovery-read-only-acceptance-v1",
            status="failed", started_unix=None, finished_unix=time.time(),
            manifest_hash=None, root=str(Path(args.root).resolve()),
            failed_checks=[dict(name="unhandled_audit_error", issue=repr(exc))],
            limits=["Audit aborted before a complete report could be assembled; no scientific data were changed."])
    result["audit_script"] = dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__))
    write_report(result, args.output)
    print(json.dumps(dict(status=result["status"], manifest_hash=result.get("manifest_hash"),
        failed_checks=len(result.get("failed_checks", [])), report=str(Path(args.output).resolve())), indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
