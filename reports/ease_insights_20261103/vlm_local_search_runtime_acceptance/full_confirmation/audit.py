"""Strict read-only CPU acceptance of the frozen 600-case confirmation matrix."""
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


MANIFEST_HASH = "80176d9f4b01e68abb022345bd6dc1a7bfe4e5aef5d69d13257801ca1eb6bdc0"
MANIFEST_FILE_SHA256 = "3c4b8bf35ecf768f129c00721ce0fd0e49ef2c209f7b9995ed0a1a128796365a"
SELECTION_FILE_SHA256 = "f6314a52f71ab0a127afb7bae0fd1adb201c934d1634634567217490a4d1e9ed"
DISCOVERY_MANIFEST_HASH = "032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9"
DISCOVERY_REPORT_SHA256 = "c560c53b188229144ea8da0719fead97f40e6c2c07166c6f5231adfec3eab6d6"
DEFAULT_ROOT = ROOT / "results/ease_insights_20261103/vlm_local_search_v1/confirmation"
DISCOVERY_ROOT = ROOT / "results/ease_insights_20261103/vlm_local_search_v1/discovery"
DISCOVERY_REPORT = ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.json"
REPORT_DIR = Path(__file__).resolve().parent
REPORT_PATH = REPORT_DIR / "audit.json"
EXPECTED_PLANNED_CONDITIONS = ("full_d10", "jac_anchor", "local_long")
EXPECTED_DISCOVERY_CONDITIONS = ("full_d10", "jac_anchor", "local_spatial", "local_object", "local_goal", "local_long")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CONFIRMATION_INITS = (39, 40, 41, 42, 43)
EXPECTED_CASES_PER_CONDITION = 200
EXPECTED_TOTAL_CASES = 600
CONFIRMATION_MARKER_NAME = "confirm_shard00of01_complete.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_hash(path):
    try:
        return sha256(path)
    except OSError:
        return None


def project_path(filename):
    path = Path(filename)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def import_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import read-only audit helper: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def import_discovery_helpers():
    path = ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.py"
    module = import_module(path, "frozen_full_discovery_read_only_helpers")
    request_module, request_path = module.request_auditor()
    return module, request_module, path, request_path


def stage_summary(audit, manifest):
    planned = tuple(manifest.get("planned_conditions", ()))
    records = audit.get("records", {})
    counts = Counter(condition for condition, _ in records)
    missing = audit.get("missing", [])
    invalid = audit.get("invalid", [])
    unexpected = audit.get("unexpected", [])
    expected_scenes = {
        (suite, task, init)
        for suite in SUITES for task in range(10) for init in CONFIRMATION_INITS
    }
    actual_scenes = {(scene["suite"], scene["task"], scene["init"]) for scene in audit.get("scenes", [])}
    reasons = []
    if planned != EXPECTED_PLANNED_CONDITIONS:
        reasons.append("Confirmation planned condition set differs from the frozen Jac/local_long pair")
    if manifest.get("conditions") is None or tuple(c["id"] for c in manifest["conditions"]) != EXPECTED_DISCOVERY_CONDITIONS:
        reasons.append("Six-condition gate whitelist differs from the frozen condition order")
    if manifest.get("phase") != "local_search_confirmation_v1":
        reasons.append("Manifest phase is not confirmation")
    if manifest.get("confirmation_inits") != list(CONFIRMATION_INITS):
        reasons.append("Confirmation init universe differs from 39-43")
    if manifest.get("cases_per_condition") != EXPECTED_CASES_PER_CONDITION or manifest.get("total_confirmation_episodes") != EXPECTED_TOTAL_CASES:
        reasons.append("Confirmation manifest case totals differ from 200 per condition / 600 overall")
    if audit.get("expected") != EXPECTED_TOTAL_CASES:
        reasons.append("Expected confirmation universe is not 600")
    if len(records) != EXPECTED_TOTAL_CASES:
        reasons.append("Valid receipt record count is not 600")
    if any(counts[condition] != EXPECTED_CASES_PER_CONDITION for condition in EXPECTED_PLANNED_CONDITIONS):
        reasons.append("At least one planned condition does not have exactly 200 valid receipts")
    if actual_scenes != expected_scenes or len(audit.get("scenes", [])) != EXPECTED_CASES_PER_CONDITION:
        reasons.append("Confirmation case universe is not the 200 suite/task/init39-43 cases")
    if missing:
        reasons.append("Missing receipts remain")
    if invalid:
        reasons.append("Invalid receipts or identities remain")
    if unexpected:
        reasons.append("Unexpected completed cases remain")
    if audit.get("complete") is not True:
        reasons.append("Frozen load_stage complete flag is false")
    return dict(passed=not reasons, expected=audit.get("expected"), valid=len(records),
        complete=audit.get("complete"), planned_conditions=list(planned),
        per_condition={condition: counts[condition] for condition in EXPECTED_PLANNED_CONDITIONS},
        scenes=len(audit.get("scenes", [])), init_ids=list(manifest.get("confirmation_inits", [])),
        missing_count=len(missing), invalid_count=len(invalid), unexpected_count=len(unexpected),
        missing=missing, invalid=invalid, unexpected=unexpected, reasons=reasons)


def final_status(stage_passed, failed_checks):
    if not stage_passed:
        return "incomplete"
    return "failed" if failed_checks else "pass"


def complete_stage_check_record(stage_check):
    return dict(name="complete_600_confirmation_receipts", **stage_check)


def verify_selection_binding(manifest):
    evidence = manifest["selection_evidence"]
    path = project_path(evidence["path"])
    errors = []
    try:
        actual_sha = sha256(path)
        selected = selection.read_selection(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(passed=False, path=str(path), errors=["Cannot read frozen selection: " + repr(exc)])
    expected_sha = evidence["sha256"]
    if actual_sha != expected_sha or actual_sha != SELECTION_FILE_SHA256:
        errors.append("Selection file SHA differs from the confirmation manifest or pinned evidence")
    if selected.get("selection_hash") != evidence.get("selection_hash"):
        errors.append("Selection content hash differs from manifest evidence")
    decision = selected.get("selection", {})
    if decision.get("status") != "selected" or decision.get("selected", {}).get("pair") != ["jac_anchor", "local_long"]:
        errors.append("Frozen selected confirmation pair differs")
    if manifest.get("planned_conditions") != list(EXPECTED_PLANNED_CONDITIONS):
        errors.append("Manifest planned conditions differ from the frozen pair")
    if manifest.get("preregistered_hypotheses") != decision.get("selected", {}).get("directions"):
        errors.append("Manifest confirmation directions differ from frozen selection")
    return dict(passed=not errors, path=str(path), sha256=actual_sha,
        expected_sha256=expected_sha, pinned_sha256=SELECTION_FILE_SHA256,
        selection_hash=selected.get("selection_hash"), selected_pair=decision.get("selected", {}).get("pair"),
        directions=decision.get("selected", {}).get("directions"), errors=errors)


def input_source_hashes(discovery_helpers_path, request_audit_path, selection_path=None):
    paths = [
        Path(__file__).resolve(), Path(discovery_helpers_path).resolve(), Path(request_audit_path).resolve(),
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
        DISCOVERY_ROOT / "manifest.json", DISCOVERY_REPORT,
        DISCOVERY_ROOT / "cross_service_reference.json", DISCOVERY_ROOT / "cross_service_reference.npz",
    ]
    if selection_path is not None:
        paths.append(project_path(selection_path))
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


def validate_confirmation_run(root, condition, manifest, run_dir, marker_run_dir, reference_path,
                              discovery_helpers):
    run_dir = Path(run_dir).resolve()
    identity_path = run_dir / "identity.json"
    protocol_path = run_dir / "protocol.json"
    gate_path = run_dir / "gate.json"
    cross_path = run_dir / "cross_service.json"
    errors = []
    try:
        identity = read_json(identity_path)
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
        identity.get("technical_runtime_revision") == common.runtime_revision(manifest) and
        protocol.get("manifest_hash") == manifest["manifest_hash"] and protocol.get("condition_id") == condition and
        protocol.get("config") == config and protocol.get("fixed_skip_layers") == config["skip"] and
        protocol.get("refresh_D") == config["refresh_D"] and protocol.get("real_input_parity_passed") is True and
        protocol.get("qk_reference_passed") is True and
        protocol.get("checkpoint_sha256") == manifest["checkpoints"]["pi0"]["sha256"] and
        protocol.get("gpu_uuid") == gpu_binding.get("gpu_uuid") and
        run_dir.parent.name == "runs" and run_dir.parent.parent.name == condition)
    if not protocol_ok:
        errors.append("Run identity or service protocol differs from the frozen manifest/condition")
    source_bank = discovery_helpers.verify_source_bank(protocol, manifest)
    if not source_bank["passed"]:
        errors.extend(source_bank["issues"])
    gate_result = discovery_helpers.verify_gate_record(gate, manifest, gate_path)
    if not gate_result["passed"]:
        errors.extend(gate_result["issues"])
    action_result = discovery_helpers.compare_gate_actions(run_dir / "gate_actions.npz", reference_path, cross)
    if not action_result["passed"]:
        errors.extend(action_result["issues"])
    marker_result = None
    if marker_run_dir is not None and run_dir == Path(marker_run_dir).resolve():
        try:
            result_receipt = read_json(run_dir / "result.json")
            marker_path = root / condition / CONFIRMATION_MARKER_NAME
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
        run_identity_sha256=evidence_hash(identity_path), protocol_sha256=evidence_hash(protocol_path),
        protocol=dict(server_pid=protocol.get("server_pid"), gpu_uuid=protocol.get("gpu_uuid"),
                      config=protocol.get("config"), source_manifest=protocol.get("source_manifest"),
                      source_manifest_sha256=protocol.get("source_manifest_sha256"),
                      provenance=protocol.get("provenance"), provenance_sha256=protocol.get("provenance_sha256")),
        protocol_identity_passed=protocol_ok, source_bank=source_bank, gate=gate_result,
        gate_actions_sha256=evidence_hash(run_dir / "gate_actions.npz"), cross_service_sha256=evidence_hash(cross_path),
        gate_actions_vs_jac_reference=action_result, completion_marker_binding=marker_result, issues=errors)


def audit_confirmation_services(root, manifest, stage, reference_path, discovery_helpers):
    services = {}
    all_issues = []
    for condition in manifest["planned_conditions"]:
        marker_path = root / condition / CONFIRMATION_MARKER_NAME
        issues = []
        try:
            marker = read_json(marker_path)
        except (OSError, ValueError, TypeError) as exc:
            services[condition] = dict(passed=False, marker_path=str(marker_path),
                                       issues=["Missing or invalid condition completion marker: " + repr(exc)])
            all_issues.append(dict(condition=condition, issues=services[condition]["issues"]))
            continue
        marker_ok = (marker.get("complete") is True and marker.get("manifest_hash") == manifest["manifest_hash"] and
                     marker.get("condition") == condition and
                     (marker.get("completed_after") == EXPECTED_CASES_PER_CONDITION or
                      marker.get("already_completed") == EXPECTED_CASES_PER_CONDITION))
        if not marker_ok:
            issues.append("Condition completion marker does not certify all 200 cases")
        marker_run = same_run_path(marker["logs"], root / condition / "runs") if marker.get("logs") else None
        if marker.get("logs") and marker_run is None:
            issues.append("Completion marker points outside its condition runs directory")

        used_pids = set()
        used_gpus = set()
        pid_cases = Counter()
        for case in stage["scenes"]:
            identity_path = (root / condition / "episodes" / case["suite"] /
                             ("task%02d" % case["task"]) / ("init%02d" % case["init"]) / "identity.json")
            try:
                identity = read_json(identity_path)
                protocol = identity["protocol"]
                pid, gpu = protocol["server_pid"], protocol["gpu_uuid"]
                if type(pid) is not int or not gpu or protocol.get("condition_id") != condition:
                    raise ValueError("Episode protocol lacks server PID/GPU/condition identity")
                used_pids.add(pid)
                used_gpus.add(gpu)
                pid_cases[pid] += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append("Episode service identity missing for " + repr(case) + ": " + repr(exc))

        pid_to_runs = {}
        for protocol_path in sorted((root / condition / "runs").glob("*/protocol.json")):
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
            matches = sorted(set(pid_to_runs.get(pid, [])), key=str)
            pid_bindings[str(pid)] = [str(path) for path in matches]
            if len(matches) != 1:
                issues.append("Episode server PID does not bind to exactly one run protocol: " + str(pid))
            selected_runs.update(matches)
        if marker_run is not None:
            selected_runs.add(marker_run)
            if marker.get("new_episodes", 0) > 0:
                try:
                    marker_pid = read_json(marker_run / "protocol.json").get("server_pid")
                    if marker_pid not in used_pids:
                        issues.append("Completion-marker run PID is not referenced by any episode identity")
                except (OSError, ValueError, TypeError):
                    issues.append("Completion-marker run has no readable protocol PID")
        if not used_pids:
            issues.append("No confirmation episode is bound to a serving PID")
        run_results = []
        for run_dir in sorted(selected_runs, key=str):
            run_results.append(validate_confirmation_run(root, condition, manifest, run_dir, marker_run,
                                                         reference_path, discovery_helpers))
        if not run_results:
            issues.append("No receipt-linked confirmation serving run was validated")
        if any(not row["passed"] for row in run_results):
            issues.append("At least one receipt-linked confirmation run failed gate/source/Jac-action checks")
        passed = marker_ok and not issues and all(row["passed"] for row in run_results)
        services[condition] = dict(passed=passed, marker_path=str(marker_path), marker_sha256=sha256(marker_path),
            marker_fields={key: marker.get(key) for key in
                ("complete", "condition", "manifest_hash", "completed_after", "already_completed", "logs", "new_episodes")},
            used_server_pids=sorted(used_pids), used_gpu_uuids=sorted(used_gpus),
            cases_by_server_pid=dict(pid_cases), pid_run_bindings=pid_bindings,
            validated_run_count=len(run_results), runs=run_results, issues=issues)
        if not passed:
            all_issues.append(dict(condition=condition, issues=issues))
    return dict(passed=not all_issues, conditions=services, issues=all_issues)


def verify_discovery_reference(discovery_helpers):
    try:
        discovery_manifest = common.read_manifest(DISCOVERY_ROOT / "manifest.json", verify_sources=True)
        discovery_report = read_json(DISCOVERY_REPORT)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(passed=False, issues=["Cannot validate full-discovery Jac reference: " + repr(exc)])
    report_sha = sha256(DISCOVERY_REPORT)
    errors = []
    if discovery_manifest.get("manifest_hash") != DISCOVERY_MANIFEST_HASH:
        errors.append("Discovery manifest differs from its frozen hash")
    if discovery_report.get("status") != "pass" or discovery_report.get("manifest_hash") != DISCOVERY_MANIFEST_HASH:
        errors.append("Full discovery acceptance report is not PASS for the frozen manifest")
    if report_sha != DISCOVERY_REPORT_SHA256:
        errors.append("Full discovery acceptance report SHA differs from the accepted report")
    if (discovery_report.get("stage", {}).get("complete") is not True or
            discovery_report.get("stage", {}).get("expected") != 720 or
            discovery_report.get("stage", {}).get("valid") != 720 or
            not discovery_report.get("serving_runs", {}).get("passed")):
        errors.append("Full discovery report does not certify complete cases and serving gates")
    try:
        reference = discovery_helpers.baseline_reference(DISCOVERY_ROOT, discovery_manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reference = dict(passed=False, issues=[repr(exc)])
    if not reference.get("passed"):
        errors.extend(reference.get("issues", ["Jac reference did not pass its frozen check"]))
    return dict(passed=not errors, discovery_manifest_hash=discovery_manifest.get("manifest_hash"),
        discovery_report_path=str(DISCOVERY_REPORT), discovery_report_sha256=report_sha,
        discovery_report_script_sha256=discovery_report.get("audit_script", {}).get("sha256"),
        jac_reference=reference, issues=errors)


def run_audit(root):
    started = time.time()
    root = Path(root).resolve()
    discovery_helpers, request_module, helper_path, request_path = import_discovery_helpers()
    result = dict(revision="full-local-search-confirmation-read-only-acceptance-v1",
        status="incomplete", started_unix=started, finished_unix=None, manifest_hash=None, root=str(root),
        input_source_hashes=input_source_hashes(helper_path, request_path),
        frozen_selection=None, source_identity=None, stage=None, requests=None,
        episode_success_failures=None, initial_input_pairing=None, discovery_reference=None, serving_runs=None,
        scope=dict(expected_episodes=EXPECTED_TOTAL_CASES, expected_conditions=list(EXPECTED_PLANNED_CONDITIONS),
            expected_cases_per_condition=EXPECTED_CASES_PER_CONDITION, expected_inits=list(CONFIRMATION_INITS),
            request_rows_checked=0, cases_checked=0, serving_run_gates_checked=0,
            saved_gate_checks_recomputed_on_gpu=0, gpu_used=False, model_started=False,
            scientific_selection_or_statistics=False, success_is_acceptance_gate=False,
            actual_internal_nfe_exported=False,
            actual_nfe_limitation="Only the frozen exported native time-node schedule is validated; the backend does not export its internal actual_NFE counter.",
            write_directory=str(REPORT_DIR),
            read_only_validators=["common.read_manifest", "selection.load_stage", "selection.read_selection",
                "analyze.load_episodes", "analyze_confirmation.audit_identities", "selection.audit_local_identities",
                "recovery.scheduled_nfe", "first_resource_session_cases.audit_requests",
                "full_discovery.verify_gate_record", "full_discovery.verify_source_bank",
                "full_discovery.compare_gate_actions"],
            prohibited_mutators_not_called=["worker.completed", "worker.finalize", "recovery.recover_case", "runner.run"]))
    failures = []
    if root != DEFAULT_ROOT.resolve():
        failures.append(dict(name="frozen_root", issue="Refusing to audit a different confirmation output root"))
        result.update(status="failed", failed_checks=failures, finished_unix=time.time())
        return result
    manifest_path = root / "manifest.json"
    try:
        manifest_file_hash = sha256(manifest_path)
        manifest = common.read_manifest(manifest_path, verify_sources=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append(dict(name="frozen_manifest", issue=repr(exc)))
        result.update(status="failed", manifest_file_sha256=None, failed_checks=failures, finished_unix=time.time())
        return result
    result["manifest_hash"] = manifest["manifest_hash"]
    result["manifest_file_sha256"] = manifest_file_hash
    result["input_source_hashes"] = input_source_hashes(
        helper_path, request_path, manifest["selection_evidence"]["path"])
    if manifest["manifest_hash"] != MANIFEST_HASH or manifest_file_hash != MANIFEST_FILE_SHA256:
        failures.append(dict(name="frozen_confirmation_manifest", actual_hash=manifest["manifest_hash"],
            expected_hash=MANIFEST_HASH, actual_file_sha256=manifest_file_hash,
            expected_file_sha256=MANIFEST_FILE_SHA256))
    frozen_selection = verify_selection_binding(manifest)
    result["frozen_selection"] = frozen_selection
    if not frozen_selection["passed"]:
        failures.append(dict(name="frozen_selection_pair_and_directions", errors=frozen_selection["errors"]))
    source_identity = discovery_helpers.verify_manifest_sources(manifest)
    result["source_identity"] = source_identity
    if not source_identity["passed"]:
        failures.append(dict(name="frozen_source_protocol_design_checkpoint_hashes",
                             issue="At least one frozen source/evidence hash differs"))
    if failures:
        result.update(status="failed", failed_checks=failures, finished_unix=time.time())
        return result

    try:
        stage = selection.load_stage(root, "confirm")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append(dict(name="load_stage_confirm", issue=repr(exc)))
        result.update(status="failed", failed_checks=failures, finished_unix=time.time())
        return result
    stage_check = stage_summary(stage, manifest)
    result["stage"] = stage_check
    if not stage_check["passed"]:
        stage_record = complete_stage_check_record(stage_check)
        result.update(status="incomplete", checks=[stage_record],
            failed_checks=[stage_record],
            receipt_evidence=stage.get("episode_evidence", []), finished_unix=time.time())
        return result

    records = stage["records"]
    false_outcomes = {condition: [record["case"] for (cid, _), record in records.items()
                                  if cid == condition and record["success"] is False]
                      for condition in manifest["planned_conditions"]}
    result["episode_success_failures"] = false_outcomes
    request_counts = {condition: dict(cases=0, passed_cases=0, requests=0, actions=0,
                                      scheduled_full=0, scheduled_fast=0)
                      for condition in manifest["planned_conditions"]}
    request_failures = []
    for (condition, _), record in sorted(records.items(), key=lambda item: (item[0][0], item[0][1])):
        config = next(item for item in manifest["conditions"] if item["id"] == condition)
        try:
            audited = request_module.audit_requests(root, condition, record["case"], config, record)
        except Exception as exc:
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
    result["requests"] = dict(by_condition=request_counts, cases_checked=sum(row["cases"] for row in request_counts.values()),
        requests_checked=sum(row["requests"] for row in request_counts.values()),
        actions_checked=sum(row["actions"] for row in request_counts.values()),
        failed_case_requests=request_failures, passed=not request_failures)
    result["scope"]["cases_checked"] = result["requests"]["cases_checked"]
    result["scope"]["request_rows_checked"] = result["requests"]["requests_checked"]
    if request_failures:
        failures.append(dict(name="all_600_request_schedule_mask_noise_action_checks", count=len(request_failures),
                             errors=request_failures))

    input_pair_errors = []
    for case in stage["scenes"]:
        key = (case["suite"], case["task"], case["init"])
        values = [records[(condition, key)]["initial_gate"] for condition in manifest["planned_conditions"]]
        identities = {(value["input_hash"], value["seed"]) for value in values}
        if len(identities) != 1:
            input_pair_errors.append(dict(case=case, identities=sorted([list(value) for value in identities])))
    result["initial_input_pairing"] = dict(cases_checked=len(stage["scenes"]), passed=not input_pair_errors,
                                           failed_cases=input_pair_errors)
    if input_pair_errors:
        failures.append(dict(name="all_200_initial_inputs_paired_across_planned_conditions",
                             errors=input_pair_errors))

    discovery_reference = verify_discovery_reference(discovery_helpers)
    result["discovery_reference"] = discovery_reference
    if not discovery_reference["passed"]:
        failures.append(dict(name="accepted_discovery_jac_gate_action_reference", errors=discovery_reference["issues"]))
    else:
        reference_path = Path(discovery_reference["jac_reference"]["npz_path"])
        services = audit_confirmation_services(root, manifest, stage, reference_path, discovery_helpers)
        result["serving_runs"] = services
        result["scope"]["serving_run_gates_checked"] = sum(
            row.get("validated_run_count", 0) for row in services["conditions"].values())
        if not services["passed"]:
            failures.append(dict(name="receipt_bound_confirmation_service_gates_source_masks_and_jac_actions",
                                 errors=services["issues"]))

    result["checks"] = [
        complete_stage_check_record(stage_check),
        dict(name="all_600_request_schedule_mask_noise_action_checks", passed=not request_failures,
             cases=result["requests"]["cases_checked"], requests=result["requests"]["requests_checked"],
             errors=request_failures),
        dict(name="all_200_initial_inputs_paired_across_planned_conditions", passed=not input_pair_errors,
             cases=len(stage["scenes"]), errors=input_pair_errors),
    ]
    if result.get("serving_runs") is not None:
        result["checks"].append(dict(name="receipt_bound_confirmation_service_gates_source_masks_and_jac_actions",
            passed=result["serving_runs"]["passed"], conditions_checked=len(result["serving_runs"]["conditions"]),
            serving_run_gates_checked=result["scope"]["serving_run_gates_checked"],
            errors=result["serving_runs"]["issues"]))
    if not discovery_reference["passed"]:
        result["checks"].append(dict(name="accepted_discovery_jac_gate_action_reference",
                                     passed=False, errors=discovery_reference["issues"]))
    result["failed_checks"] = failures
    result["receipt_evidence"] = stage.get("episode_evidence", [])
    result["status"] = final_status(stage_check["passed"], failures)
    result["finished_unix"] = time.time()
    return result


def write_report(result, output):
    output = Path(output).resolve()
    if output.parent != REPORT_DIR.resolve():
        raise ValueError("Report output must remain inside the assigned full_confirmation directory")
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
        result = dict(revision="full-local-search-confirmation-read-only-acceptance-v1",
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
