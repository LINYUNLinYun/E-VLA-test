"""Read-only CPU acceptance for the frozen LIBERO90 432-case discovery."""
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
REPORT_DIR = Path(__file__).resolve().parent
REPORT_PATH = REPORT_DIR / "audit.json"
DEFAULT_ROOT = ROOT / "results/ease_insights_20261103/vlm_within_suite_v1/discovery"
MANIFEST_HASH = "6db11c5a8f1cdb5eab549f406ece3ba7b14661b2699da407c1ac87073e6c8a7d"
RAW_MANIFEST_SHA = "e9dc20f236a74da7e10493488ea644f0050501159f0c2e390e77c8f2ab653ddb"
OLD_MANIFEST_HASH = "032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9"
OLD_REFERENCE_SHA = "1ba90f16ffc16e2df9e74f3fc7a27004bfc182b8f47c1c177ab7e28a6e55621c"
CONDITIONS = ("full_d10", "jac_anchor", "local_spatial", "local_object", "local_goal", "local_long")
TASKS = (15,16,27,29,36,51,59,65,67,73,80,83,0,6,7,11,20,22,23,28,33,35,39,44)
CATEGORIES = ("single_object_placement", "state_change_only")
INITS = (36,37,38)
N_CASES, N_TOTAL = 72, 432
SOURCE_COUNTS = {"source_files": 70, "analysis_sources": 7, "local_runtime_sources": 14}
ARTIFACTS = ("identity.json", "result.json", "complete.json", "infer.jsonl", "telemetry.jsonl", "arrays.npz", "input_0000.npz")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def key(case):
    return case["suite"], case["task"], case["init"]


def scenes():
    return [dict(suite="libero_90", task=task, init=init) for task in TASKS for init in INITS]


def manifest_hash(manifest):
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def raw_manifest_pin(actual_sha):
    return dict(expected_sha256=RAW_MANIFEST_SHA, sha256=actual_sha, passed=actual_sha == RAW_MANIFEST_SHA)


def verify_matrix(m):
    issues = []
    task_counts = Counter(m.get("task_categories", {}).get(str(t)) for t in m.get("tasks", []))
    if (m.get("manifest_hash") != MANIFEST_HASH or manifest_hash(m) != MANIFEST_HASH or
            m.get("phase") != "within_suite_discovery_v1"):
        issues.append("Manifest payload/hash/phase differs from the frozen discovery")
    if (m.get("planned_conditions") != list(CONDITIONS) or
            [c.get("id") for c in m.get("conditions", [])] != list(CONDITIONS)):
        issues.append("Six condition IDs/order differ")
    if m.get("suites") != ["libero_90"] or m.get("tasks") != list(TASKS):
        issues.append("LIBERO90 task universe differs")
    if (m.get("categories") != list(CATEGORIES) or
            task_counts != Counter({c: 12 for c in CATEGORIES})):
        issues.append("Frozen task/category map is not 12 tasks per category")
    if (m.get("discovery_inits") != list(INITS) or m.get("cases_per_condition") != N_CASES or
            m.get("total_discovery_episodes") != N_TOTAL or
            m.get("case_table") != [dict(suite="libero_90", task=t) for t in TASKS]):
        issues.append("Discovery matrix is not 24 tasks × init36–38 × six conditions")
    return dict(passed=not issues, issues=issues, conditions=list(CONDITIONS), task_count=len(m.get("tasks", [])),
                tasks_per_category=dict(task_counts), cases_per_condition=m.get("cases_per_condition"),
                total=m.get("total_discovery_episodes"))


def file_pins(entries, count=None):
    rows = []
    for name, expected in sorted((entries or {}).items()):
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        try:
            actual, error = sha(path), None
        except OSError as exc:
            actual, error = None, repr(exc)
        rows.append(dict(path=str(path.resolve()), expected_sha256=expected, sha256=actual,
                         passed=actual == expected, error=error))
    return dict(passed=all(r["passed"] for r in rows) and (count is None or len(rows) == count),
                count=len(rows), expected_count=count, files=rows)


def evidence_pin(name, evidence):
    path, expected = evidence.get("path", ""), evidence.get("sha256")
    try:
        actual, error = sha(path), None
    except OSError as exc:
        actual, error = None, repr(exc)
    return dict(name=name, path=str(Path(path).resolve()), expected_sha256=expected,
                sha256=actual, passed=actual == expected, error=error)


def verify_sources(m):
    groups = {name: file_pins(m.get(name), count) for name, count in SOURCE_COUNTS.items()}
    evidence = [evidence_pin(name, m.get(name, {})) for name in ("protocol_evidence", "design_evidence")]
    checkpoint = {name: file_pins(value.get("identity_artifacts", {}))
                  for name, value in m.get("checkpoints", {}).items()}
    for name, value in checkpoint.items():
        value["weight_pin"] = dict(path=m["checkpoints"][name].get("path"),
                                   sha256=m["checkpoints"][name].get("sha256"),
                                   checked_via_service_protocol=True, large_file_rehashed=False)

    assets, conflicts = {}, []
    def pin(path, expected, kind):
        path = str(Path(path).resolve())
        row = assets.setdefault(path, {"sha256": expected, "kinds": []})
        row["kinds"].append(kind)
        if row["sha256"] != expected:
            conflicts.append(path)
    for item in m.get("task_assets", []):
        pin(item["bddl_path"], item["bddl_sha256"], "task_assets.bddl")
        pin(item["init_path"], item["init_sha256"], "task_assets.init")
    official = m.get("official_init_availability", {})
    for item in official.get("cases", []):
        pin(item["bddl_path"], item["bddl_sha256"], "official.bddl")
        pin(item["init_file"], item["init_file_sha256"], "official.init")
    asset_rows = []
    for path, pinrow in sorted(assets.items()):
        try:
            actual, error = sha(path), None
        except OSError as exc:
            actual, error = None, repr(exc)
        asset_rows.append(dict(path=path, kinds=pinrow["kinds"], expected_sha256=pinrow["sha256"],
                               sha256=actual, passed=actual == pinrow["sha256"], error=error))
    asset_ok = (not conflicts and len(asset_rows) == 72 and all(r["passed"] for r in asset_rows) and
                official.get("passed") is True and len(official.get("cases", [])) == 24 and
                official.get("discovery_inits") == list(INITS) and official.get("unique_prompts") == 20)
    assets_result = dict(passed=asset_ok, unique_files=len(asset_rows), expected_unique_files=72,
                         conflicts=conflicts, files=asset_rows)
    passed = (all(v["passed"] for v in groups.values()) and all(r["passed"] for r in evidence) and
              bool(checkpoint) and all(v["passed"] for v in checkpoint.values()) and asset_ok)
    return dict(passed=passed, source_groups=groups, protocol_design=evidence,
                checkpoint_identity_artifacts=checkpoint, official_assets=assets_result)


def verify_preflight(m):
    ref = m.get("preflight", {}).get("cross_service_reference", {})
    issues, pins = [], []
    if (m.get("preflight", {}).get("atol") != 1e-5 or m.get("preflight", {}).get("rtol") != 1e-5 or
            ref.get("manifest_hash") != OLD_MANIFEST_HASH or ref.get("sha256") != OLD_REFERENCE_SHA):
        issues.append("Frozen historical Jac NPZ or 1e-5 tolerance identity differs")
    for name, obj in (("reference_npz", ref), ("reference_manifest", ref.get("reference_manifest", {})),
                      ("source_protocol", ref.get("source_protocol", {}))):
        row = evidence_pin(name, obj)
        pins.append(row)
        if not row["passed"]:
            issues.append("Historical preflight artifact changed: " + name)
    try:
        old_manifest = read(ref["reference_manifest"]["path"])
        protocol = read(ref["source_protocol"]["path"])
        identity = (old_manifest.get("manifest_hash") == OLD_MANIFEST_HASH and
                    protocol.get("manifest_hash") == OLD_MANIFEST_HASH and
                    protocol.get("condition_id") == "jac_anchor" and
                    protocol.get("real_input_parity_passed") is True and protocol.get("qk_reference_passed") is True and
                    protocol.get("checkpoint_sha256") == m["checkpoints"]["pi0"]["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        identity = False
    if not identity:
        issues.append("Historical Jac manifest/protocol/checkpoint identity differs")
    try:
        copy_path = DEFAULT_ROOT / "cross_service_reference.npz"
        identity_path = DEFAULT_ROOT / "cross_service_reference.json"
        copy_identity = read(identity_path)
        copy_ok = (sha(copy_path) == OLD_REFERENCE_SHA and
                   copy_identity.get("scientific_manifest_hash") == m["manifest_hash"] and
                   copy_identity.get("source_manifest_hash") == OLD_MANIFEST_HASH and
                   copy_identity.get("source_npz", {}).get("path") == ref["path"] and
                   copy_identity.get("source_npz", {}).get("sha256") == OLD_REFERENCE_SHA and
                   copy_identity.get("reference_npz_sha256") == OLD_REFERENCE_SHA and
                   copy_identity.get("historical_reference") == ref)
        copied = dict(path=str(copy_path), sha256=sha(copy_path), identity_path=str(identity_path),
                      identity_sha256=sha(identity_path), identity=copy_identity, passed=copy_ok)
    except (OSError, ValueError, TypeError) as exc:
        copied = dict(passed=False, error=repr(exc))
        copy_ok = False
    if not copy_ok:
        issues.append("Discovery reference copy is not bound to the frozen historical NPZ")
    return dict(passed=not issues and identity and all(x["passed"] for x in pins),
                historical_identity_passed=identity, artifacts=pins, discovery_copy=copied, issues=issues)


def stage_summary(stage):
    records = stage.get("records", {})
    expected = {(cid, key(case)) for cid in CONDITIONS for case in scenes()}
    counts = Counter(cid for cid, _ in records)
    reasons = []
    if stage.get("expected") != N_TOTAL or len(records) != N_TOTAL or set(records) != expected:
        reasons.append("Strict loader does not contain the exact 432 receipt universe")
    if any(counts[c] != N_CASES for c in CONDITIONS):
        reasons.append("A condition does not have exactly 72 valid receipts")
    if len(stage.get("scenes", [])) != N_CASES or {key(x) for x in stage.get("scenes", [])} != {key(x) for x in scenes()}:
        reasons.append("Strict loader case list differs from the fixed task/init universe")
    for field in ("missing", "invalid", "unexpected"):
        if stage.get(field):
            reasons.append(field + " entries remain")
    if stage.get("complete") is not True:
        reasons.append("Strict load_stage complete flag is false")
    return dict(passed=not reasons, expected=stage.get("expected"), valid=len(records), complete=stage.get("complete"),
                per_condition={c: counts[c] for c in CONDITIONS}, missing=stage.get("missing", []),
                invalid=stage.get("invalid", []), unexpected=stage.get("unexpected", []), reasons=reasons)


def category_summary(m, case_list):
    task_counts = Counter(m.get("task_categories", {}).get(str(t)) for t in m.get("tasks", []))
    case_counts = Counter(m.get("task_categories", {}).get(str(x["task"])) for x in case_list)
    issues = []
    if m.get("categories") != list(CATEGORIES) or task_counts != Counter({c: 12 for c in CATEGORIES}):
        issues.append("Manifest does not map 12 fixed tasks to each category")
    if len(case_list) != N_CASES or case_counts != Counter({c: 36 for c in CATEGORIES}):
        issues.append("LIBERO90 discovery case map is not 36 cases per category")
    return dict(passed=not issues, tasks_per_category=dict(task_counts), cases_per_category=dict(case_counts), issues=issues)


def marker_check(marker, condition, manifest_hash):
    fields = [f for f in ("completed_after", "already_completed") if f in marker]
    count = marker.get(fields[0]) if len(fields) == 1 else None
    issues = []
    if marker.get("complete") is not True or marker.get("manifest_hash") != manifest_hash or marker.get("condition") != condition:
        issues.append("Completion marker manifest/condition/complete identity differs")
    if len(fields) != 1 or type(count) is not int or count != N_CASES:
        issues.append("Completion marker does not certify exactly 72 cases")
    return dict(passed=not issues, count=count, count_field=fields[0] if len(fields) == 1 else None, issues=issues)


def marker_run_path(marker, root, condition):
    value = marker.get("logs")
    if not isinstance(value, str) or not value.strip():
        return dict(passed=False, run_dir=None, issue="Completion marker logs path is required")
    run_dir = Path(value).resolve()
    expected_parent = (Path(root) / condition / "runs").resolve()
    try:
        run_dir.relative_to(expected_parent)
    except ValueError:
        return dict(passed=False, run_dir=str(run_dir), issue="Marker run is outside the condition runs directory")
    return dict(passed=True, run_dir=str(run_dir), issue=None)


def verify_markers(root, manifest):
    rows, issues = {}, []
    for condition in CONDITIONS:
        path = Path(root) / condition / "discovery_shard00of01_complete.json"
        try:
            marker = read(path)
            row = marker_check(marker, condition, manifest["manifest_hash"])
            row.update(path=str(path.resolve()), sha256=sha(path), marker=marker)
            run_binding = marker_run_path(marker, root, condition)
            if not run_binding["passed"]:
                row["issues"].append(run_binding["issue"])
            else:
                run_dir = Path(run_binding["run_dir"])
                try:
                    result_path = run_dir / "result.json"
                    result = read(result_path)
                    bound = (result == marker and result.get("complete") is True and
                             result.get("manifest_hash") == manifest["manifest_hash"] and
                             result.get("condition") == condition and result.get("logs") == str(run_dir))
                    row["run_dir"] = str(run_dir)
                    row["run_result"] = dict(path=str(result_path), sha256=sha(result_path), passed=bound)
                    if not bound:
                        row["issues"].append("Marker and serving result do not match")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["issues"].append("Marker run binding invalid: " + repr(exc))
            row["passed"] = row["passed"] and not row["issues"]
            rows[condition] = row
            if not row["passed"]:
                issues.append(dict(condition=condition, issues=row["issues"]))
        except (OSError, ValueError, TypeError) as exc:
            rows[condition] = dict(passed=False, path=str(path), issues=[repr(exc)])
            issues.append(dict(condition=condition, issues=[repr(exc)]))
    return dict(passed=not issues, conditions=rows, issues=issues)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load read-only helper: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from preparations.ease_insights_20261103.vlm_within_suite_v1 import common, selection, worker
    return common, selection, worker


def audit_source_hashes():
    paths = [Path(__file__), ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/first_resource_session_cases/audit.py",
             ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/common.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/selection.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/worker.py"]
    rows = []
    for path in paths:
        try:
            rows.append(dict(path=str(path.resolve()), sha256=sha(path), exists=True))
        except OSError as exc:
            rows.append(dict(path=str(path.resolve()), exists=False, error=repr(exc)))
    return rows


def verify_episodes(root, manifest, stage, request_audit, worker):
    records, errors = stage["records"], []
    official = {(x["suite"], x["task"]): x for x in manifest["official_init_availability"]["cases"]}
    configs = {x["id"]: x for x in manifest["conditions"]}
    receipts, identities, initial, pids = [], {}, {}, {c: {} for c in CONDITIONS}
    requests = {c: dict(cases=0, passed_cases=0, requests=0, actions=0, failures=[]) for c in CONDITIONS}
    false_outcomes = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        for case in stage["scenes"]:
            ck = key(case)
            directory = Path(root) / condition / "episodes" / case["suite"] / ("task%02d" % case["task"]) / ("init%02d" % case["init"])
            receipt_path = directory / "episode_receipt.json"
            try:
                receipt = read(receipt_path)
                identity, result = read(directory / "identity.json"), read(directory / "result.json")
                artifact_hashes = {}
                for filename, expected in receipt["artifacts"].items():
                    actual = sha(directory / filename)
                    artifact_hashes[filename] = dict(expected_sha256=expected, sha256=actual, passed=actual == expected)
                good = (receipt.get("valid") is True and receipt.get("manifest_hash") == MANIFEST_HASH and
                        receipt.get("condition") == condition and receipt.get("case") == case and
                        set(ARTIFACTS).issubset(receipt["artifacts"]) and
                        all(x["passed"] for x in artifact_hashes.values()) and
                        result.get("valid") is True and result.get("manifest_hash") == MANIFEST_HASH and
                        result.get("condition") == condition and result.get("case") == case and
                        result.get("success") == receipt.get("success") == records[condition, ck]["success"] and
                        identity.get("case") == case and identity.get("manifest_hash") == MANIFEST_HASH and
                        identity.get("config") == configs[condition])
                receipts.append(dict(condition=condition, case=case, receipt_sha256=sha(receipt_path),
                                     valid=receipt.get("valid"), success=receipt.get("success"),
                                     artifacts=artifact_hashes, passed=good))
                if not good:
                    errors.append(dict(condition=condition, case=case, issue="Receipt identity/artifact SHA mismatch"))
                identities[condition, ck] = identity
                if receipt.get("success") is False:
                    false_outcomes[condition].append(case)
                protocol = identity.get("protocol", {})
                pid = protocol.get("server_pid")
                if type(pid) is int:
                    pids[condition].setdefault(pid, []).append(case)
                else:
                    errors.append(dict(condition=condition, case=case, issue="Receipt protocol lacks server PID"))
                init_gate = records[condition, ck]["initial_gate"]
                initial[condition, ck] = dict(gate=init_gate, physical={k: identity.get(k) for k in
                    ("init_vector_sha256", "bddl_sha256", "prompt", "env_seed", "settling", "policy_action_limit", "control_hz")})

                rows = [json.loads(line) for line in (directory / "infer.jsonl").read_text().splitlines()]
                request = request_audit.audit_requests(root, condition, case, configs[condition], records[condition, ck])
                worker.validate_action_artifacts(directory, rows, result)
                hashes_ok = all(isinstance(x.get("input_hash"), str) and len(x["input_hash"]) == 64 and
                                all(c in "0123456789abcdef" for c in x["input_hash"]) for x in rows)
                first_ok = bool(rows) and rows[0].get("input_hash") == init_gate.get("input_hash")
                request_ok = request.get("passed") is True and hashes_ok and first_ok
                requests[condition]["cases"] += 1
                requests[condition]["passed_cases"] += int(request_ok)
                requests[condition]["requests"] += len(rows)
                requests[condition]["actions"] += sum(x["executed_length"] for x in rows)
                if not request_ok:
                    issue = dict(condition=condition, case=case, request_issues=request.get("issues", []),
                                 input_hashes_valid=hashes_ok, first_input_matches=first_ok)
                    errors.append(issue)
                    requests[condition]["failures"].append(issue)
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                errors.append(dict(condition=condition, case=case, issue=repr(exc)))
                requests[condition]["cases"] += 1
                requests[condition]["failures"].append(dict(case=case, issue=repr(exc)))

    paired_cases, pair_errors = [], []
    for case in stage["scenes"]:
        ck, rows = key(case), []
        asset = official[case["suite"], case["task"]]
        expected_physical = dict(init_vector_sha256=asset["init_sha256"][str(case["init"])],
            bddl_sha256=asset["bddl_sha256"], prompt=asset["prompt"], env_seed=7, settling=10,
            policy_action_limit=400, control_hz=20.0)
        for condition in CONDITIONS:
            item = initial.get((condition, ck))
            if item is None:
                continue
            first = item["gate"]
            ok = (item["physical"] == expected_physical and first.get("passed") is True and
                  first.get("manifest_hash") == MANIFEST_HASH and first.get("case") == case)
            rows.append(dict(condition=condition, physical=item["physical"], input_hash=first.get("input_hash"),
                             seed=first.get("seed"), passed=ok))
            if not ok:
                pair_errors.append(dict(condition=condition, case=case, issue="Official physical/initial identity differs"))
        refs_path = Path(root) / "initial_observations" / case["suite"] / ("task%02d" % case["task"]) / ("init%02d.json" % case["init"])
        try:
            ref = read(refs_path)
            refs_hash = sha(refs_path)
            values = {(x["input_hash"], x["seed"]) for x in rows}
            ref_ok = len(rows) == 6 and len({json.dumps(x["physical"], sort_keys=True) for x in rows}) == 1 and len(values) == 1 and (ref.get("input_hash"), ref.get("seed")) in values and ref.get("case") == case and ref.get("manifest_hash") == MANIFEST_HASH
        except (OSError, ValueError, TypeError):
            refs_hash, ref_ok = None, False
        if not ref_ok:
            pair_errors.append(dict(case=case, issue="Initial input/noise is not paired across all six conditions"))
        paired_cases.append(dict(case=case, passed=ref_ok and all(x["passed"] for x in rows),
                                 conditions=rows, shared_reference_sha256=refs_hash))
    request_pass = (not any(requests[c]["failures"] for c in CONDITIONS) and
                    all(requests[c]["cases"] == N_CASES and requests[c]["passed_cases"] == N_CASES for c in CONDITIONS))
    return dict(receipts=dict(passed=not errors and len(receipts) == N_TOTAL, count=len(receipts),
                              current_sha256=receipts, failures=errors,
                              valid_false_outcomes_by_condition=false_outcomes, success_is_acceptance_gate=False),
                pairing=dict(passed=not pair_errors and len(paired_cases) == N_CASES, cases_checked=len(paired_cases),
                             cases=paired_cases, failures=pair_errors),
                requests=dict(passed=request_pass, cases_checked=sum(requests[c]["cases"] for c in CONDITIONS),
                              request_rows_checked=sum(requests[c]["requests"] for c in CONDITIONS),
                              executed_actions_checked=sum(requests[c]["actions"] for c in CONDITIONS),
                              by_condition=requests, success_is_acceptance_gate=False),
                identities=identities, pid_cases=pids)


def service_audit(root, manifest, stage, identities, pid_cases, markers, reference, gate_helpers):
    common, _, _ = modules()
    ref = manifest["preflight"]["cross_service_reference"]
    selected_manifest = dict(manifest, suites=list(common.GATE_SUITES))
    expected_runtime = common.runtime_revision(manifest)
    results, issues = {}, []
    for condition in CONDITIONS:
        cond_root = Path(root) / condition
        pid_paths = {}
        for path in sorted((cond_root / "runs").glob("*/protocol.json")):
            try:
                protocol = read(path)
                if protocol.get("condition_id") == condition and type(protocol.get("server_pid")) is int:
                    pid_paths.setdefault(protocol["server_pid"], []).append(path.parent.resolve())
            except (OSError, ValueError, TypeError):
                pass
        binds, selected, local_issues = {}, set(), []
        for pid in pid_cases[condition]:
            found = sorted(set(pid_paths.get(pid, [])))
            binds[str(pid)] = [str(x) for x in found]
            if len(found) != 1:
                local_issues.append("Episode server PID must bind to exactly one run protocol: " + str(pid))
            selected.update(found)
        marker = markers["conditions"][condition].get("marker", {})
        marker_dir = markers["conditions"][condition].get("run_dir")
        if marker_dir:
            selected.add(Path(marker_dir).resolve())
        runs = []
        for run_dir in sorted(selected, key=str):
            files = {name: run_dir / name for name in ("identity.json", "protocol.json", "gate.json", "cross_service.json", "gate_actions.npz", "result.json")}
            errors = []
            try:
                ident, proto, gate = read(files["identity.json"]), read(files["protocol.json"]), read(files["gate.json"])
                cross, receipt = read(files["cross_service.json"]), read(files["result.json"])
                hashes = {name: sha(path) for name, path in files.items()}
            except (OSError, ValueError, TypeError) as exc:
                runs.append(dict(path=str(run_dir), passed=False, issues=[repr(exc)]))
                continue
            run_pids = {int(pid) for pid, paths in binds.items() if str(run_dir) in paths}
            identity_ok = (run_dir.parent.resolve() == (cond_root / "runs").resolve() and
                ident.get("manifest_hash") == MANIFEST_HASH and ident.get("condition") == condition and
                ident.get("stateful_workers_per_server") == 1 and ident.get("technical_runtime_revision") == expected_runtime and
                ident.get("gpu_binding", {}).get("gpu_uuid") == proto.get("gpu_uuid") and
                ident.get("resource_execution_revision", {}).get("scientific_manifest_hash") == MANIFEST_HASH and
                proto.get("manifest_hash") == MANIFEST_HASH and proto.get("condition_id") == condition and
                proto.get("server_pid") in run_pids and proto.get("gpu_uuid", "").startswith("GPU-") and
                proto.get("config") == next(x for x in manifest["conditions"] if x["id"] == condition) and
                proto.get("fixed_skip_layers") == next(x for x in manifest["conditions"] if x["id"] == condition)["skip"] and
                proto.get("refresh_D") == 10 and proto.get("technical_runtime_revision") == expected_runtime and
                proto.get("checkpoint_sha256") == manifest["checkpoints"]["pi0"]["sha256"] and
                proto.get("real_input_parity_passed") is True and proto.get("qk_reference_passed") is True and
                proto.get("policy_action_limit") == 400 and proto.get("settling_actions") == 10 and
                proto.get("evaluation_suites") == ["libero_90"] and proto.get("internal_actual_NFE_exported") is False)
            if not identity_ok:
                errors.append("Run PID/protocol/condition/GPU/manifest/checkpoint identity differs")
            if any(identities[condition, key(case)].get("protocol") != proto
                   for pid, cases in pid_cases[condition].items() if pid in run_pids for case in cases):
                errors.append("Receipt-embedded serving protocol differs from run protocol")
            source = gate_helpers.verify_source_bank(proto, manifest)
            if not source["passed"]:
                errors.extend(source["issues"])
            try:
                gate_result = gate_helpers.verify_gate_record(gate, selected_manifest, files["gate.json"])
                numeric = gate_helpers.compare_gate_actions(files["gate_actions.npz"], ref["path"], cross)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                gate_result, numeric = dict(passed=False, issues=[repr(exc)]), dict(passed=False, issues=[repr(exc)])
            cross_ok = (cross.get("scientific_manifest_hash") == MANIFEST_HASH and
                        cross.get("reference_identity") == reference["discovery_copy"].get("identity") and
                        cross.get("reference_identity", {}).get("historical_reference") == ref)
            receipt_ok = (receipt.get("complete") is True and receipt.get("manifest_hash") == MANIFEST_HASH and
                          receipt.get("condition") == condition and receipt.get("logs") == str(run_dir) and
                          (receipt.get("completed_after") == N_CASES or receipt.get("already_completed") == N_CASES))
            if not gate_result.get("passed"):
                errors.extend(gate_result.get("issues", ["Persisted old 332-check gate failed"]))
            if not numeric.get("passed"):
                errors.extend(numeric.get("issues", ["Saved gate_actions differ from the preflight Jac NPZ"]))
            if not cross_ok:
                errors.append("cross_service receipt is not bound to the frozen preflight reference")
            if not receipt_ok:
                errors.append("Serving run result is incomplete or misbound")
            if marker_dir and Path(marker_dir).resolve() == run_dir and receipt != marker:
                errors.append("Condition completion marker does not equal its serving result")
            runs.append(dict(path=str(run_dir), passed=not errors, server_pid=proto.get("server_pid"),
                gpu_uuid=proto.get("gpu_uuid"), manifest_hash=proto.get("manifest_hash"),
                artifact_sha256=hashes, source_bank=source, old_332_gate=gate_result,
                gate_actions_vs_preflight_reference=numeric, cross_service_reference_passed=cross_ok, issues=errors))
            if errors:
                local_issues.append(dict(path=str(run_dir), issues=errors))
        passed = bool(runs) and not local_issues and all(x["passed"] for x in runs)
        results[condition] = dict(passed=passed, receipt_linked_pids=sorted(pid_cases[condition]),
            cases_by_pid=pid_cases[condition], pid_run_bindings=binds, run_count=len(runs), runs=runs, issues=local_issues)
        if not passed:
            issues.append(dict(condition=condition, issues=local_issues))
    return dict(passed=not issues, conditions=results, issues=issues)


def source_hashes():
    paths = [Path(__file__), ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/first_resource_session_cases/audit.py",
             ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/common.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/selection.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/worker.py"]
    return [{"path": str(p.resolve()), "sha256": sha(p)} for p in paths]


def assemble_report(manifest_id, root, started, checks, failures, **sections):
    report = dict(revision="within-suite-discovery-432-read-only-acceptance-v1", status="incomplete",
        started_unix=started, finished_unix=None, manifest_hash=manifest_id, root=str(Path(root).resolve()),
        audit_input_sources=source_hashes(), checks=checks, failed_checks=failures,
        scope=dict(expected_episodes=N_TOTAL, expected_conditions=list(CONDITIONS), cases_per_condition=N_CASES,
            gpu_used=False, model_started=False, scientific_statistics_recomputed=False,
            success_is_acceptance_gate=False, actual_internal_nfe_exported=False,
            actual_nfe_limitation="Exported D10/D2 time-node schedules are checked; backend internal actual_NFE is not exported.",
            numeric_scope="Persisted old four-suite 332-check gate plus finite old-bank reference; not all new 20 prompts.",
            write_directory=str(REPORT_DIR.resolve()), prohibited_mutators_not_called=["worker.completed", "worker.finalize",
                "recovery.recover_case", "runner.run", "borrowed_controller.run", "GPU/model startup"],
            read_only_validators=["within_suite_v1.selection.load_stage", "within_suite_v1.selection.audit_local_identities",
                "within_suite_v1.worker.validate_action_artifacts", "first_resource_session_cases.audit_requests",
                "legacy full_discovery pure 332-gate/action comparators"]))
    report.update(sections)
    return report


def run_audit(root):
    started, root = time.time(), Path(root).resolve()
    checks, failures = [], []
    result = assemble_report(None, root, started, checks, failures, source_identity=None, preflight_reference=None,
                             stage=None, category_mapping=None, completion_markers=None, episode_evidence=None,
                             initial_input_pairing=None, saved_requests=None, serving_runs=None)
    def check(name, passed, details):
        row = dict(name=name, passed=bool(passed), details=details)
        checks.append(row)
        if not passed:
            failures.append(row)
    if root != DEFAULT_ROOT.resolve():
        check("frozen_root", False, dict(actual=str(root), expected=str(DEFAULT_ROOT.resolve())))
        result["status"], result["finished_unix"] = "failed", time.time()
        return result
    path = root / "manifest.json"
    try:
        m = read(path)
        result["manifest_file_sha256"] = sha(path)
    except (OSError, ValueError, TypeError) as exc:
        check("frozen_manifest", False, repr(exc))
        result["status"], result["finished_unix"] = "failed", time.time()
        return result
    result["manifest_hash"] = m.get("manifest_hash")
    raw_pin = dict(path=str(path.resolve()), **raw_manifest_pin(result["manifest_file_sha256"]))
    result["raw_manifest_file_identity"] = raw_pin
    matrix = verify_matrix(m)
    source = verify_sources(m)
    reference = verify_preflight(m)
    result["source_identity"], result["preflight_reference"] = source, reference
    check("frozen_manifest_identity_and_432_matrix", matrix["passed"], matrix)
    check("frozen_manifest_raw_file_SHA", raw_pin["passed"], raw_pin)
    check("source_protocol_design_checkpoint_norm_and_official_asset_SHA", source["passed"], source)
    check("old_Jac_reference_SHA_and_identity", reference["passed"], reference)
    if not (matrix["passed"] and raw_pin["passed"] and source["passed"] and reference["passed"]):
        result["status"], result["failed_checks"], result["finished_unix"] = "failed", failures, time.time()
        return result

    common, selection, worker = modules()
    try:
        # This is invoked only by the final post-completion command. Synthetic
        # preparation tests never call load_stage or walk real receipts.
        frozen_manifest = common.read_manifest(path, verify_sources=False)
        stage = selection.load_stage(root, "discovery")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        check("strict_load_stage", False, repr(exc))
        result["status"], result["failed_checks"], result["finished_unix"] = "failed", failures, time.time()
        return result
    summary = stage_summary(stage)
    result["stage"] = summary
    check("complete_strict_432_receipts", summary["passed"], summary)
    if not summary["passed"]:
        result["status"], result["failed_checks"], result["finished_unix"] = "incomplete", failures, time.time()
        return result

    category = category_summary(frozen_manifest, stage["scenes"])
    markers = verify_markers(root, frozen_manifest)
    result["category_mapping"], result["completion_markers"] = category, markers
    check("libero90_two_category_12_tasks_each", category["passed"], category)
    check("six_72_case_markers", markers["passed"], markers)
    request_audit = import_file("within_suite_saved_request_audit", ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/first_resource_session_cases/audit.py")
    artifacts = verify_episodes(root, frozen_manifest, stage, request_audit, worker)
    result["episode_evidence"], result["initial_input_pairing"], result["saved_requests"] = artifacts["receipts"], artifacts["pairing"], artifacts["requests"]
    result["scope_request_rows_checked"] = artifacts["requests"]["request_rows_checked"]
    check("all_432_receipt_artifact_SHA_and_valid_false_retention", artifacts["receipts"]["passed"],
          {k: v for k, v in artifacts["receipts"].items() if k != "current_sha256"})
    check("physical_initial_input_noise_pairing", artifacts["pairing"]["passed"],
          {k: v for k, v in artifacts["pairing"].items() if k != "cases"})
    check("saved_action_schedule_mask_history_noise_and_input_hash", artifacts["requests"]["passed"],
          {k: v for k, v in artifacts["requests"].items() if k != "by_condition"})
    old_gate = import_file("within_suite_old_gate_helpers", ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.py")
    services = service_audit(root, frozen_manifest, stage, artifacts["identities"], artifacts["pid_cases"], markers, reference, old_gate)
    result["serving_runs"] = services
    result["scope_serving_run_gates_checked"] = sum(x["run_count"] for x in services["conditions"].values())
    check("receipt_linked_runs_source_332_gate_cross_service_actions", services["passed"],
          {k: v for k, v in services.items() if k != "conditions"})
    result["failed_checks"], result["finished_unix"] = failures, time.time()
    result["status"] = "failed" if failures else "pass"
    return result


def write_report(result, output):
    output = Path(output).resolve()
    if output.parent != REPORT_DIR.resolve():
        raise ValueError("Report output must stay inside the assigned full_discovery_audit directory")
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
        result = assemble_report(None, args.root, time.time(), [],
            [dict(name="unhandled_audit_error", passed=False, details=repr(exc))], source_identity=None,
            preflight_reference=None, stage=None, category_mapping=None, completion_markers=None,
            episode_evidence=None, initial_input_pairing=None, saved_requests=None, serving_runs=None)
        result["status"], result["finished_unix"] = "failed", time.time()
    result["audit_script"] = dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))
    write_report(result, args.output)
    print(json.dumps(dict(status=result["status"], manifest_hash=result.get("manifest_hash"),
                          failed_checks=len(result.get("failed_checks", [])), report=str(Path(args.output).resolve())), indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
