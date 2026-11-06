"""Read-only CPU acceptance for a frozen 360-case within-suite confirmation."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = Path(__file__).resolve().parent
REPORT_PATH = REPORT_DIR / "audit.json"
DEFAULT_ROOT = ROOT / "results/ease_insights_20261103/vlm_within_suite_v1/confirmation"
DISCOVERY_ROOT = ROOT / "results/ease_insights_20261103/vlm_within_suite_v1/discovery"
DISCOVERY_HASH = "6db11c5a8f1cdb5eab549f406ece3ba7b14661b2699da407c1ac87073e6c8a7d"
DISCOVERY_RAW_SHA = "e9dc20f236a74da7e10493488ea644f0050501159f0c2e390e77c8f2ab653ddb"
OLD_MANIFEST_HASH = "032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9"
OLD_REFERENCE_SHA = "1ba90f16ffc16e2df9e74f3fc7a27004bfc182b8f47c1c177ab7e28a6e55621c"
CONDITIONS_ALL = ("full_d10", "jac_anchor", "local_spatial", "local_object", "local_goal", "local_long")
TASKS = (15,16,27,29,36,51,59,65,67,73,80,83,0,6,7,11,20,22,23,28,33,35,39,44)
TASK_CATEGORIES = {str(task): ("single_object_placement" if i < 12 else "state_change_only")
                   for i, task in enumerate(TASKS)}
CATEGORIES = ("single_object_placement", "state_change_only")
INITS = (39,40,41,42,43)
N_CASES, N_TOTAL = 120, 360
ARTIFACTS = ("identity.json", "result.json", "complete.json", "infer.jsonl", "telemetry.jsonl", "arrays.npz", "input_0000.npz")
MASKS = {
    "jac_anchor": (2,3,5,14,16,17), "local_spatial": (2,3,5,13,16,17),
    "local_object": (2,3,5,14,15,17), "local_goal": (2,3,11,14,16,17),
    "local_long": (2,3,5,12,16,17),
}
RESERVOIR_AMENDMENT = ROOT / 'results/ease_insights_20261103/reservoir_v1/resource_amendment.json'
RESERVOIR_AMENDMENT_SHA = '524357cd305ecde913362f023474465d1067711e9f214970a1729b219d8bca83'
RESERVOIR_SOURCES = tuple('preparations/ease_insights_20261103/reservoir_v1/' + name
                          for name in ('__init__.py', 'guardian.py', 'keeper.py'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def canonical_manifest_hash(manifest):
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return digest(payload)


def key(case):
    return case["suite"], case["task"], case["init"]


def cases(tasks=TASKS, inits=INITS):
    return [dict(suite="libero_90", task=task, init=init) for task in tasks for init in inits]


def raw_manifest_pin(actual, expected):
    return dict(expected_sha256=expected, sha256=actual, passed=actual == expected)


def resource_source_extension_contract(manifest, discovery, additions):
    """Preserve every original pin and permit exactly the three frozen resource files."""
    issues = []
    if set(additions) != set(RESERVOIR_SOURCES):
        issues.append('Resource source additions are not the exact three reservoir modules')
    projection = dict(manifest)
    for name in ('source_files', 'analysis_sources', 'local_runtime_sources'):
        original = discovery.get(name)
        if not isinstance(original, dict) or not original:
            issues.append('Missing original discovery source map: ' + name)
            continue
        expected = dict(original)
        if name != 'analysis_sources':
            expected.update(additions)
        if manifest.get(name) != expected:
            issues.append('Confirmation changed, omitted, or added an unexpected source pin: ' + name)
        # The unchanged discovery helper has fixed70/7/14 source counts. Its
        # scientific checks run on this exact original-map projection; the
        # three additional modules are separately hashed below. The actual
        # confirmation manifest and its canonical/raw pins are never changed.
        projection[name] = dict(original)
    return dict(passed=not issues, issues=issues, base_projection=projection)


def verify_confirmation_sources(manifest, source_helpers):
    try:
        discovery_path = DISCOVERY_ROOT / 'manifest.json'
        if sha(discovery_path) != DISCOVERY_RAW_SHA or sha(RESERVOIR_AMENDMENT) != RESERVOIR_AMENDMENT_SHA:
            raise ValueError('Frozen discovery or reservoir amendment SHA differs')
        discovery, amendment = read(discovery_path), read(RESERVOIR_AMENDMENT)
        additions = {name: amendment['source_hashes'][name] for name in RESERVOIR_SOURCES}
        binding = manifest.get('resource_execution_revision', {}).get('details', {}).get('reservoir_resource_amendment')
        expected = dict(path=str(RESERVOIR_AMENDMENT), sha256=RESERVOIR_AMENDMENT_SHA,
                        source_hashes=amendment['source_hashes'])
        if binding != expected:
            raise ValueError('Confirmation resource evidence does not bind the frozen reservoir amendment')
        contract = resource_source_extension_contract(manifest, discovery, additions)
        if not contract['passed']:
            return dict(passed=False, issues=contract['issues'])
        base = source_helpers.verify_sources(contract['base_projection'])
        extra = source_helpers.file_pins(additions, len(RESERVOIR_SOURCES))
        return dict(passed=base['passed'] and extra['passed'], base_scientific_checks=base,
            resource_source_additions=extra, exact_source_extension=True,
            actual_manifest_source_group_counts={name:len(manifest[name]) for name in
                ('source_files','analysis_sources','local_runtime_sources')},
            scope='All original70/7/14 pins preserved and verified, plus exactly three frozen reservoir modules; actual manifest canonical/raw SHA checked separately')
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, issues=[repr(error)])


def selection_contract(manifest, discovery_manifest, selection):
    """Pure binding check; this validates a frozen decision and never ranks candidates."""
    issues = []
    selection_payload = dict(selection)
    supplied_hash = selection_payload.pop("selection_hash", None)
    choice = selection.get("selection", {})
    selected = choice.get("selected") or {}
    pair, directions = selected.get("paths"), selected.get("directions")
    if supplied_hash is None or digest(selection_payload) != supplied_hash:
        issues.append("Frozen selection digest differs")
    if (selection.get("revision") != "ease-vlm-within-suite-selection-v1" or
            selection.get("complete_discovery_episodes") != 432 or
            selection.get("expected_discovery_episodes") != 432 or
            selection.get("independent_validation_launched") is not False or
            selection.get("discovery_manifest_hash") != discovery_manifest.get("manifest_hash")):
        issues.append("Selection does not bind the complete frozen 432-case discovery")
    if (selection.get("protocol_evidence") != manifest.get("protocol_evidence") or
            selection.get("design_evidence") != manifest.get("design_evidence") or
            manifest.get("selection_evidence", {}).get("selection_hash") != supplied_hash or
            manifest.get("selection_evidence", {}).get("frozen_selection") != choice or
            manifest.get("selection_evidence", {}).get("evidence") != selection.get("evidence")):
        issues.append("Selection evidence/protocol/design fields differ from the confirmation manifest")
    if not isinstance(pair, list) or len(pair) != 2 or pair != sorted(set(pair)) or any(x not in MASKS for x in pair):
        issues.append("Frozen pair must contain two sorted distinct non-Full conditions")
    if selected.get("pair") != pair:
        issues.append("Frozen selection pair alias differs from its selected paths")
    if choice.get("status") != "selected" or not selected:
        issues.append("Confirmation requires the already frozen selected pair")
    if isinstance(pair, list) and manifest.get("planned_conditions") != ["full_d10", *pair]:
        issues.append("Confirmation conditions differ from the frozen pair")
    if not isinstance(directions, list) or len(directions) != 2:
        issues.append("Frozen pair must have two preregistered directions")
    else:
        if manifest.get("preregistered_hypotheses") != directions:
            issues.append("Preregistered directions differ from the frozen discovery selection")
        if ({d.get("category") for d in directions} != set(CATEGORIES) or
                {d.get("better") for d in directions} != set(pair or []) or
                any({d.get("better"), d.get("worse")} != set(pair or []) for d in directions)):
            issues.append("Directions are not the frozen opposite-category pair contrasts")
    return dict(passed=not issues, selection_hash=supplied_hash, pair=pair, directions=directions, issues=issues)


def confirmation_matrix(manifest, expected_hash, expected_raw_sha):
    planned = manifest.get("planned_conditions", [])
    ids = [row.get("id") for row in manifest.get("conditions", [])]
    issues = []
    if (manifest.get("manifest_hash") != expected_hash or expected_hash is None or
            canonical_manifest_hash(manifest) != expected_hash):
        issues.append("Confirmation manifest identity differs from required explicit CLI pin")
    if manifest.get("phase") != "within_suite_confirmation_v1":
        issues.append("Manifest phase is not within-suite confirmation")
    if manifest.get("suites") != ["libero_90"] or manifest.get("tasks") != list(TASKS):
        issues.append("Frozen LIBERO90 24-task universe differs")
    if (manifest.get("categories") != list(CATEGORIES) or manifest.get("task_categories") != TASK_CATEGORIES):
        issues.append("Frozen task/category mapping differs")
    if (manifest.get("confirmation_inits") != list(INITS) or len(planned) != 3 or len(set(planned)) != 3 or
            not planned or planned[0] != "full_d10" or planned[1:] != sorted(planned[1:]) or
            any(c not in MASKS for c in planned[1:]) or ids != list(CONDITIONS_ALL)):
        issues.append("Confirmation must use Full plus the two frozen selected conditions")
    if (manifest.get("cases_per_condition") != N_CASES or
            manifest.get("total_confirmation_episodes") != N_TOTAL or
            manifest.get("case_table") != [dict(suite="libero_90", task=t) for t in TASKS]):
        issues.append("Confirmation matrix must be 24 tasks × init39–43 × three conditions")
    if manifest.get("science", {}).get("policy_action_limit") != 400 or manifest.get("science", {}).get("settling_actions") != 10:
        issues.append("Uniform 400-action plus 10-settling identity differs")
    return dict(passed=not issues, conditions=planned, task_count=len(manifest.get("tasks", [])),
                cases_per_condition=manifest.get("cases_per_condition"), total=manifest.get("total_confirmation_episodes"),
                issues=issues)


def stage_summary(stage, condition_ids):
    expected = {(cid, key(case)) for cid in condition_ids for case in cases()}
    records = stage.get("records", {})
    counts = Counter(cid for cid, _ in records)
    issues = []
    if stage.get("expected") != N_TOTAL or len(records) != N_TOTAL or set(records) != expected:
        issues.append("Strict loader does not contain the exact 360 receipt universe")
    if any(counts[cid] != N_CASES for cid in condition_ids):
        issues.append("A selected condition does not have exactly 120 valid receipts")
    if len(stage.get("scenes", [])) != N_CASES or {key(x) for x in stage.get("scenes", [])} != {key(x) for x in cases()}:
        issues.append("Strict loader case list differs from tasks × init39–43")
    for field in ("missing", "invalid", "unexpected"):
        if stage.get(field):
            issues.append(field + " entries remain")
    if stage.get("complete") is not True:
        issues.append("Strict load_stage complete flag is false")
    return dict(passed=not issues, expected=stage.get("expected"), valid=len(records), complete=stage.get("complete"),
                per_condition={cid: counts[cid] for cid in condition_ids}, missing=stage.get("missing", []),
                invalid=stage.get("invalid", []), unexpected=stage.get("unexpected", []), issues=issues)


def category_summary(manifest, scene_list):
    tasks = Counter(manifest.get("task_categories", {}).get(str(t)) for t in manifest.get("tasks", []))
    examples = Counter(manifest.get("task_categories", {}).get(str(x["task"])) for x in scene_list)
    issues = []
    if manifest.get("categories") != list(CATEGORIES) or tasks != Counter({c: 12 for c in CATEGORIES}):
        issues.append("Confirmation mapping must preserve 12 tasks per category")
    if len(scene_list) != N_CASES or examples != Counter({c: 60 for c in CATEGORIES}):
        issues.append("Each confirmation category must contain 60 cases per condition")
    return dict(passed=not issues, tasks_per_category=dict(tasks), cases_per_category=dict(examples), issues=issues)


def marker_check(marker, condition, manifest_hash):
    fields = [f for f in ("completed_after", "already_completed") if f in marker]
    count = marker.get(fields[0]) if len(fields) == 1 else None
    issues = []
    if marker.get("complete") is not True or marker.get("manifest_hash") != manifest_hash or marker.get("condition") != condition:
        issues.append("Completion marker manifest/condition/complete identity differs")
    if len(fields) != 1 or type(count) is not int or count != N_CASES:
        issues.append("Completion marker does not certify exactly 120 cases")
    return dict(passed=not issues, count=count, issues=issues)


def marker_run_path(marker, root, condition):
    logs = marker.get("logs")
    if not isinstance(logs, str) or not logs.strip():
        return dict(passed=False, run_dir=None, issue="Nonempty completion marker logs path is required")
    run_dir = Path(logs).resolve()
    try:
        run_dir.relative_to((Path(root) / condition / "runs").resolve())
    except ValueError:
        return dict(passed=False, run_dir=str(run_dir), issue="Marker run is outside the condition runs directory")
    return dict(passed=True, run_dir=str(run_dir), issue=None)


def verify_markers(root, manifest, condition_ids):
    rows, issues = {}, []
    for condition in condition_ids:
        path = Path(root) / condition / "confirm_shard00of01_complete.json"
        try:
            marker = read(path)
            row = marker_check(marker, condition, manifest["manifest_hash"])
            row.update(path=str(path.resolve()), sha256=sha(path), marker=marker)
            binding = marker_run_path(marker, root, condition)
            if not binding["passed"]:
                row["issues"].append(binding["issue"])
            else:
                run_dir = Path(binding["run_dir"])
                result_path = run_dir / "result.json"
                try:
                    result = read(result_path)
                    bound = (result == marker and result.get("complete") is True and
                             result.get("manifest_hash") == manifest["manifest_hash"] and
                             result.get("condition") == condition and result.get("logs") == str(run_dir))
                    row["run_dir"] = str(run_dir)
                    row["run_result"] = dict(path=str(result_path), sha256=sha(result_path), passed=bound)
                    if not bound:
                        row["issues"].append("Marker does not equal its serving-run result")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["issues"].append("Cannot validate marker result binding: " + repr(exc))
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
        raise RuntimeError("Cannot import read-only audit helper: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from preparations.ease_insights_20261103.vlm_within_suite_v1 import common, selection, worker
    return common, selection, worker


def verify_frozen_binding(root, manifest, expected_discovery_hash):
    common, _, _ = modules()
    issues = []
    ref = manifest.get("discovery_reference", {})
    selection_ref = manifest.get("selection_evidence", {})
    try:
        discovery_path = Path(ref["root"]).resolve() / "manifest.json"
        discovery = common.read_manifest(discovery_path, verify_sources=False)
        discovery_sha = sha(discovery_path)
        if (str(Path(ref["root"]).resolve()) != str(DISCOVERY_ROOT.resolve()) or
                ref.get("manifest_hash") != expected_discovery_hash or expected_discovery_hash != DISCOVERY_HASH or
                discovery.get("manifest_hash") != expected_discovery_hash or ref.get("manifest_sha256") != discovery_sha or
                discovery_sha != DISCOVERY_RAW_SHA or
                ref.get("valid_episodes_at_freeze") != 432 or discovery.get("total_discovery_episodes") != 432):
            issues.append("Confirmation discovery reference does not bind frozen complete432 manifest")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        discovery, discovery_sha = None, None
        issues.append("Cannot validate frozen discovery manifest reference: " + repr(exc))
    try:
        selection_path = Path(selection_ref["path"]).resolve()
        selection = read(selection_path)
        raw_sha = sha(selection_path)
        contract = selection_contract(manifest, discovery or {}, selection)
        selection_payload = dict(selection)
        selected_hash = selection_payload.pop("selection_hash", None)
        selection_hash_ok = (selected_hash is not None and digest(selection_payload) == selected_hash and
            raw_sha == selection_ref.get("sha256") and selected_hash == selection_ref.get("selection_hash") and
            selection_ref.get("frozen_selection") == selection.get("selection") and
            selection_ref.get("evidence") == selection.get("evidence"))
        if not selection_hash_ok:
            issues.append("Selection file SHA, content hash, or frozen evidence binding differs")
        if not contract["passed"]:
            issues.extend(contract["issues"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        selection, raw_sha, contract = None, None, dict(passed=False, issues=[repr(exc)])
        selection_hash_ok = False
        issues.append("Cannot validate frozen selection file: " + repr(exc))
    return dict(passed=not issues, discovery=dict(path=str(ref.get("root", "")), manifest_hash=None if discovery is None else discovery["manifest_hash"],
                manifest_sha256=discovery_sha, valid_episodes_at_freeze=ref.get("valid_episodes_at_freeze")),
                selection=dict(path=selection_ref.get("path"), sha256=raw_sha, selection_hash=selection_ref.get("selection_hash"),
                    binding=contract, file_and_manifest_binding_passed=selection_hash_ok), issues=issues)


def verify_preflight(manifest):
    ref = manifest.get("preflight", {}).get("cross_service_reference", {})
    issues, artifacts = [], []
    if (manifest.get("preflight", {}).get("atol") != 1e-5 or manifest.get("preflight", {}).get("rtol") != 1e-5 or
            ref.get("manifest_hash") != OLD_MANIFEST_HASH or ref.get("sha256") != OLD_REFERENCE_SHA):
        issues.append("Historical Jac reference or 1e-5 tolerance differs")
    for name, item in (("reference_npz", ref), ("reference_manifest", ref.get("reference_manifest", {})),
                       ("source_protocol", ref.get("source_protocol", {}))):
        try:
            actual = sha(item["path"])
            passed = actual == item.get("sha256")
        except (OSError, KeyError, TypeError):
            actual, passed = None, False
        artifacts.append(dict(name=name, path=item.get("path"), expected_sha256=item.get("sha256"), sha256=actual, passed=passed))
        if not passed:
            issues.append("Historical preflight artifact changed: " + name)
    try:
        old = read(ref["reference_manifest"]["path"])
        proto = read(ref["source_protocol"]["path"])
        identity = (old.get("manifest_hash") == OLD_MANIFEST_HASH and proto.get("manifest_hash") == OLD_MANIFEST_HASH and
                    proto.get("condition_id") == "jac_anchor" and proto.get("real_input_parity_passed") is True and
                    proto.get("qk_reference_passed") is True and
                    proto.get("checkpoint_sha256") == manifest["checkpoints"]["pi0"]["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        identity = False
    if not identity:
        issues.append("Historical Jac manifest/protocol/checkpoint identity differs")
    return dict(passed=not issues and identity and all(x["passed"] for x in artifacts),
                historical_identity_passed=identity, artifacts=artifacts, issues=issues)


def output_reference(root, manifest, historical):
    ref = manifest["preflight"]["cross_service_reference"]
    try:
        path, idpath = Path(root) / "cross_service_reference.npz", Path(root) / "cross_service_reference.json"
        identity = read(idpath)
        passed = (sha(path) == ref["sha256"] and identity.get("scientific_manifest_hash") == manifest["manifest_hash"] and
                  identity.get("source_manifest_hash") == OLD_MANIFEST_HASH and
                  identity.get("source_npz", {}).get("path") == ref["path"] and
                  identity.get("source_npz", {}).get("sha256") == ref["sha256"] and
                  identity.get("reference_npz_sha256") == ref["sha256"] and
                  identity.get("historical_reference") == ref)
        return dict(passed=passed, path=str(path), sha256=sha(path), identity_path=str(idpath),
                    identity_sha256=sha(idpath), identity=identity)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(passed=False, error=repr(exc))


def verify_episodes(root, manifest, stage, request_audit, worker, condition_ids):
    records, errors = stage["records"], []
    official = {(x["suite"], x["task"]): x for x in manifest["official_init_availability"]["cases"]}
    configs = {x["id"]: x for x in manifest["conditions"]}
    receipts, identities, initial, pids = [], {}, {}, {c: {} for c in condition_ids}
    requests = {c: dict(cases=0, passed_cases=0, requests=0, actions=0, failures=[]) for c in condition_ids}
    false_outcomes = {c: [] for c in condition_ids}
    for condition in condition_ids:
        for case in stage["scenes"]:
            ck = key(case)
            directory = Path(root) / condition / "episodes" / case["suite"] / ("task%02d" % case["task"]) / ("init%02d" % case["init"])
            receipt_path = directory / "episode_receipt.json"
            try:
                receipt = read(receipt_path)
                identity, result = read(directory / "identity.json"), read(directory / "result.json")
                hashes = {name: dict(expected_sha256=expected, sha256=sha(directory / name),
                                     passed=sha(directory / name) == expected)
                          for name, expected in receipt["artifacts"].items()}
                good = (receipt.get("valid") is True and receipt.get("manifest_hash") == manifest["manifest_hash"] and
                    receipt.get("condition") == condition and receipt.get("case") == case and
                    set(ARTIFACTS).issubset(receipt["artifacts"]) and all(x["passed"] for x in hashes.values()) and
                    result.get("valid") is True and result.get("manifest_hash") == manifest["manifest_hash"] and
                    result.get("condition") == condition and result.get("case") == case and
                    result.get("success") == receipt.get("success") == records[condition, ck]["success"] and
                    identity.get("case") == case and identity.get("manifest_hash") == manifest["manifest_hash"] and
                    identity.get("config") == configs[condition])
                receipts.append(dict(condition=condition, case=case, receipt_sha256=sha(receipt_path),
                    valid=receipt.get("valid"), success=receipt.get("success"), artifacts=hashes, passed=good))
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
                    errors.append(dict(condition=condition, case=case, issue="Receipt protocol lacks serving PID"))
                gate = records[condition, ck]["initial_gate"]
                initial[condition, ck] = dict(gate=gate, physical={k: identity.get(k) for k in
                    ("init_vector_sha256", "bddl_sha256", "prompt", "env_seed", "settling", "policy_action_limit", "control_hz")})
                rows = [json.loads(line) for line in (directory / "infer.jsonl").read_text().splitlines()]
                request = request_audit.audit_requests(root, condition, case, configs[condition], records[condition, ck])
                worker.validate_action_artifacts(directory, rows, result)
                hashes_ok = all(isinstance(x.get("input_hash"), str) and len(x["input_hash"]) == 64 and
                                all(c in "0123456789abcdef" for c in x["input_hash"]) for x in rows)
                first_ok = bool(rows) and rows[0].get("input_hash") == gate.get("input_hash")
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

    paired, pair_errors = [], []
    for case in stage["scenes"]:
        ck, rows = key(case), []
        asset = official[case["suite"], case["task"]]
        expected = dict(init_vector_sha256=asset["init_sha256"][str(case["init"])], bddl_sha256=asset["bddl_sha256"],
                        prompt=asset["prompt"], env_seed=7, settling=10, policy_action_limit=400, control_hz=20.0)
        for condition in condition_ids:
            item = initial.get((condition, ck))
            if item is None:
                continue
            gate = item["gate"]
            ok = (item["physical"] == expected and gate.get("passed") is True and
                  gate.get("manifest_hash") == manifest["manifest_hash"] and gate.get("case") == case)
            rows.append(dict(condition=condition, physical=item["physical"], input_hash=gate.get("input_hash"),
                             seed=gate.get("seed"), passed=ok))
            if not ok:
                pair_errors.append(dict(condition=condition, case=case, issue="Physical/initial identity differs"))
        refpath = Path(root) / "initial_observations" / case["suite"] / ("task%02d" % case["task"]) / ("init%02d.json" % case["init"])
        try:
            ref = read(refpath)
            input_ids = {(x["input_hash"], x["seed"]) for x in rows}
            ref_sha = sha(refpath)
            ref_ok = (len(rows) == 3 and len({json.dumps(x["physical"], sort_keys=True) for x in rows}) == 1 and
                      len(input_ids) == 1 and (ref.get("input_hash"), ref.get("seed")) in input_ids and
                      ref.get("manifest_hash") == manifest["manifest_hash"] and ref.get("case") == case)
        except (OSError, ValueError, TypeError):
            ref_sha, ref_ok = None, False
        if not ref_ok:
            pair_errors.append(dict(case=case, issue="Initial input/noise is not paired across the three conditions"))
        paired.append(dict(case=case, passed=ref_ok and all(x["passed"] for x in rows),
                           conditions=rows, shared_reference_sha256=ref_sha))
    request_ok = (not any(requests[c]["failures"] for c in condition_ids) and
                  all(requests[c]["cases"] == N_CASES and requests[c]["passed_cases"] == N_CASES for c in condition_ids))
    return dict(receipts=dict(passed=not errors and len(receipts) == N_TOTAL, count=len(receipts), receipts=receipts,
                failures=errors, valid_false_outcomes_by_condition=false_outcomes, success_is_acceptance_gate=False),
        pairing=dict(passed=not pair_errors and len(paired) == N_CASES, cases_checked=len(paired), cases=paired, failures=pair_errors),
        requests=dict(passed=request_ok, cases_checked=sum(x["cases"] for x in requests.values()),
            request_rows_checked=sum(x["requests"] for x in requests.values()),
            executed_actions_checked=sum(x["actions"] for x in requests.values()), by_condition=requests,
            success_is_acceptance_gate=False), identities=identities, pid_cases=pids)


def service_audit(root, manifest, stage, identities, pid_cases, markers, historical_npz, output_ref, gate_helpers, condition_ids):
    common, _, _ = modules()
    expected_runtime = common.runtime_revision(manifest)
    gate_manifest = dict(manifest, suites=list(common.GATE_SUITES))
    results, issues = {}, []
    for condition in condition_ids:
        condition_root = Path(root) / condition
        protocol_paths = sorted((condition_root / "runs").glob("*/protocol.json"))
        by_pid = {}
        for path in protocol_paths:
            try:
                protocol = read(path)
                if protocol.get("condition_id") == condition and type(protocol.get("server_pid")) is int:
                    by_pid.setdefault(protocol["server_pid"], []).append(path.parent.resolve())
            except (OSError, ValueError, TypeError):
                pass
        binds, selected, local = {}, set(), []
        for pid in pid_cases[condition]:
            paths = sorted(set(by_pid.get(pid, [])))
            binds[str(pid)] = [str(p) for p in paths]
            if len(paths) != 1:
                local.append("Receipt PID must bind to exactly one service protocol: " + str(pid))
            selected.update(paths)
        marker = markers["conditions"][condition].get("marker", {})
        marker_dir = markers["conditions"][condition].get("run_dir")
        if marker_dir:
            selected.add(Path(marker_dir).resolve())
        run_rows = []
        for run_dir in sorted(selected, key=str):
            names = ("identity.json", "protocol.json", "gate.json", "cross_service.json", "gate_actions.npz", "result.json")
            paths = {name: run_dir / name for name in names}
            try:
                ident, proto, gate = read(paths["identity.json"]), read(paths["protocol.json"]), read(paths["gate.json"])
                cross, result = read(paths["cross_service.json"]), read(paths["result.json"])
                hashes = {name: sha(path) for name, path in paths.items()}
            except (OSError, ValueError, TypeError) as exc:
                run_rows.append(dict(path=str(run_dir), passed=False, issues=[repr(exc)]))
                continue
            run_pids = {int(pid) for pid, bound in binds.items() if str(run_dir) in bound}
            cfg = next(x for x in manifest["conditions"] if x["id"] == condition)
            protocol_ok = (run_dir.parent.resolve() == (condition_root / "runs").resolve() and
                ident.get("manifest_hash") == manifest["manifest_hash"] and ident.get("condition") == condition and
                ident.get("stateful_workers_per_server") == 1 and ident.get("technical_runtime_revision") == expected_runtime and
                ident.get("gpu_binding", {}).get("gpu_uuid") == proto.get("gpu_uuid") and
                ident.get("resource_execution_revision", {}).get("scientific_manifest_hash") == manifest["manifest_hash"] and
                proto.get("manifest_hash") == manifest["manifest_hash"] and proto.get("condition_id") == condition and
                proto.get("server_pid") in run_pids and proto.get("gpu_uuid", "").startswith("GPU-") and
                proto.get("config") == cfg and proto.get("fixed_skip_layers") == cfg["skip"] and
                proto.get("refresh_D") == 10 and proto.get("technical_runtime_revision") == expected_runtime and
                proto.get("checkpoint_sha256") == manifest["checkpoints"]["pi0"]["sha256"] and
                proto.get("real_input_parity_passed") is True and proto.get("qk_reference_passed") is True and
                proto.get("policy_action_limit") == 400 and proto.get("settling_actions") == 10 and
                proto.get("evaluation_suites") == ["libero_90"] and proto.get("internal_actual_NFE_exported") is False)
            errors = []
            if not protocol_ok:
                errors.append("Serving run PID/GPU/manifest/condition/source/checkpoint identity differs")
            if any(identities[condition, key(case)].get("protocol") != proto
                   for pid, group in pid_cases[condition].items() if pid in run_pids for case in group):
                errors.append("Receipt-embedded protocol differs from the serving run")
            source = gate_helpers.verify_source_bank(proto, manifest)
            if not source["passed"]:
                errors.extend(source["issues"])
            try:
                gate_result = gate_helpers.verify_gate_record(gate, gate_manifest, paths["gate.json"])
                action_result = gate_helpers.compare_gate_actions(paths["gate_actions.npz"], historical_npz, cross)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                gate_result = dict(passed=False, issues=[repr(exc)])
                action_result = dict(passed=False, issues=[repr(exc)])
            cross_ok = (cross.get("scientific_manifest_hash") == manifest["manifest_hash"] and
                        cross.get("reference_identity") == output_ref.get("identity") and
                        cross.get("reference_identity", {}).get("historical_reference") == manifest["preflight"]["cross_service_reference"])
            result_ok = (result.get("complete") is True and result.get("manifest_hash") == manifest["manifest_hash"] and
                         result.get("condition") == condition and result.get("logs") == str(run_dir) and
                         (result.get("completed_after") == N_CASES or result.get("already_completed") == N_CASES))
            if not gate_result.get("passed"):
                errors.extend(gate_result.get("issues", ["Persisted 332-check finite gate failed"]))
            if not action_result.get("passed"):
                errors.extend(action_result.get("issues", ["Gate actions differ from pinned historical Jac NPZ"]))
            if not cross_ok:
                errors.append("cross_service receipt does not bind the confirmation's pinned preflight reference")
            if not result_ok:
                errors.append("Serving run result receipt is incomplete or misbound")
            if marker_dir and Path(marker_dir).resolve() == run_dir and result != marker:
                errors.append("Completion marker and result receipt differ")
            run_rows.append(dict(path=str(run_dir), passed=not errors, server_pid=proto.get("server_pid"),
                gpu_uuid=proto.get("gpu_uuid"), artifact_sha256=hashes, source_bank=source,
                old_332_gate=gate_result, gate_actions_vs_preflight_jac=action_result,
                cross_service_reference_passed=cross_ok, issues=errors))
            if errors:
                local.append(dict(path=str(run_dir), issues=errors))
        passed = bool(run_rows) and not local and all(x["passed"] for x in run_rows)
        results[condition] = dict(passed=passed, receipt_linked_pids=sorted(pid_cases[condition]),
            cases_by_pid=pid_cases[condition], pid_run_bindings=binds, run_count=len(run_rows), runs=run_rows, issues=local)
        if not passed:
            issues.append(dict(condition=condition, issues=local))
    return dict(passed=not issues, conditions=results, issues=issues)


def source_hashes():
    paths = [Path(__file__), ROOT / "reports/ease_insights_20261103/vlm_within_suite_v1_preparation/full_discovery_audit/audit.py",
             ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/first_resource_session_cases/audit.py",
             ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/common.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/selection.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/worker.py",
             ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/analyze.py"]
    return [{"path": str(p.resolve()), "sha256": sha(p)} for p in paths]


def assemble_report(manifest_id, raw_sha, root, started, checks, failures, **sections):
    report = dict(revision="within-suite-confirmation-360-read-only-acceptance-v1", status="incomplete",
        started_unix=started, finished_unix=None, manifest_hash=manifest_id,
        manifest_file_sha256=raw_sha, root=str(Path(root).resolve()), audit_input_sources=source_hashes(),
        checks=checks, failed_checks=failures,
        scope=dict(expected_episodes=N_TOTAL, conditions_per_pair=3, cases_per_condition=N_CASES,
            gpu_used=False, model_started=False, discovery_reselection=False, success_is_acceptance_gate=False,
            scientific_statistics_recomputed=False, actual_internal_nfe_exported=False,
            actual_nfe_limitation="Exported D10/D2 nodes are checked; backend internal actual_NFE is not exported.",
            numerical_scope="Persisted old four-suite 332-check gate and finite old-bank reference only; no claim for all new prompts.",
            prohibited_mutators_not_called=["worker.completed", "worker.finalize", "recovery.recover_case", "runner.run", "GPU/model startup"]))
    report.update(sections)
    return report


def run_audit(root, expected_hash, expected_raw_sha):
    started, root = time.time(), Path(root).resolve()
    checks, failures = [], []
    result = assemble_report(expected_hash, None, root, started, checks, failures,
        frozen_binding=None, source_identity=None, preflight_reference=None, stage=None,
        category_mapping=None, completion_markers=None, episode_evidence=None,
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
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash or "")) or not re.fullmatch(r"[0-9a-f]{64}", str(expected_raw_sha or "")):
        check("explicit_manifest_pins", False, "Expected canonical manifest hash and raw file SHA must both be 64 lowercase hex characters")
        result["status"], result["finished_unix"] = "failed", time.time()
        return result
    manifest_path = root / "manifest.json"
    try:
        manifest = read(manifest_path)
        actual_raw_sha = sha(manifest_path)
    except (OSError, ValueError, TypeError) as exc:
        check("confirmation_manifest", False, repr(exc))
        result["status"], result["finished_unix"] = "failed", time.time()
        return result
    result["manifest_file_sha256"] = actual_raw_sha
    raw_pin = raw_manifest_pin(actual_raw_sha, expected_raw_sha)
    matrix = confirmation_matrix(manifest, expected_hash, expected_raw_sha)
    raw_pin["passed"] = raw_pin["passed"] and manifest.get("manifest_hash") == expected_hash
    check("explicit_canonical_and_raw_manifest_pins", raw_pin["passed"] and matrix["passed"],
          dict(raw_pin=raw_pin, matrix=matrix))
    if not (raw_pin["passed"] and matrix["passed"]):
        result["status"], result["failed_checks"], result["finished_unix"] = "failed", failures, time.time()
        return result

    common, selection_module, worker = modules()
    try:
        manifest = common.read_manifest(manifest_path, verify_sources=False)
        binding = verify_frozen_binding(root, manifest, DISCOVERY_HASH)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        binding = dict(passed=False, issues=[repr(exc)])
    result["frozen_binding"] = binding
    check("frozen_432_discovery_selection_pair_and_directions", binding["passed"], binding)
    source_helpers = import_file("within_suite_discovery_helpers_for_confirmation", REPORT_DIR.parent / "full_discovery_audit" / "audit.py")
    source = verify_confirmation_sources(manifest, source_helpers)
    historical = verify_preflight(manifest)
    result["source_identity"], result["preflight_reference"] = source, historical
    check("source_checkpoint_norm_protocol_design_official_assets", source["passed"], source)
    check("old_finite_jac_reference_pins", historical["passed"], historical)
    if not (binding["passed"] and source["passed"] and historical["passed"]):
        result["status"], result["failed_checks"], result["finished_unix"] = "failed", failures, time.time()
        return result

    try:
        stage = selection_module.load_stage(root, "confirm")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        check("strict_load_stage", False, repr(exc))
        result["status"], result["failed_checks"], result["finished_unix"] = "failed", failures, time.time()
        return result
    condition_ids = manifest["planned_conditions"]
    summary = stage_summary(stage, condition_ids)
    result["stage"] = summary
    check("complete_strict_360_receipts", summary["passed"], summary)
    if not summary["passed"]:
        result["status"], result["failed_checks"], result["finished_unix"] = "incomplete", failures, time.time()
        return result

    categories = category_summary(manifest, stage["scenes"])
    markers = verify_markers(root, manifest, condition_ids)
    out_ref = output_reference(root, manifest, historical)
    result["category_mapping"], result["completion_markers"], result["output_reference"] = categories, markers, out_ref
    check("confirmation_category_mapping", categories["passed"], categories)
    check("three_120_case_completion_markers", markers["passed"], markers)
    check("confirmation_cross_service_reference_copy", out_ref["passed"], out_ref)
    request_audit = import_file("within_suite_confirmation_saved_request_audit", ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/first_resource_session_cases/audit.py")
    artifacts = verify_episodes(root, manifest, stage, request_audit, worker, condition_ids)
    result["episode_evidence"], result["initial_input_pairing"], result["saved_requests"] = artifacts["receipts"], artifacts["pairing"], artifacts["requests"]
    result["scope_request_rows_checked"] = artifacts["requests"]["request_rows_checked"]
    check("all_360_current_receipt_artifact_hashes_valid_false_retained", artifacts["receipts"]["passed"],
          {k: v for k, v in artifacts["receipts"].items() if k != "receipts"})
    check("physical_initial_input_noise_paired_across_three", artifacts["pairing"]["passed"],
          {k: v for k, v in artifacts["pairing"].items() if k != "cases"})
    check("saved_actions_schedule_masks_history_noise_input_hash", artifacts["requests"]["passed"],
          {k: v for k, v in artifacts["requests"].items() if k != "by_condition"})
    gate_helpers = import_file("within_suite_old_gate_helpers_for_confirmation", ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.py")
    services = service_audit(root, manifest, stage, artifacts["identities"], artifacts["pid_cases"], markers,
                             manifest["preflight"]["cross_service_reference"]["path"], out_ref, gate_helpers, condition_ids)
    result["serving_runs"] = services
    result["scope_serving_run_gates_checked"] = sum(row["run_count"] for row in services["conditions"].values())
    check("receipt_linked_pids_protocol_source_332_gate_cross_service", services["passed"],
          {k: v for k, v in services.items() if k != "conditions"})
    result["failed_checks"], result["finished_unix"] = failures, time.time()
    result["status"] = "failed" if failures else "pass"
    return result


def write_report(result, output):
    output = Path(output).resolve()
    if output.parent != REPORT_DIR.resolve():
        raise ValueError("Report output must remain inside full_confirmation_audit")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--expected-manifest-hash", required=True)
    parser.add_argument("--expected-manifest-file-sha256", required=True)
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.root, args.expected_manifest_hash, args.expected_manifest_file_sha256)
    except Exception as exc:
        result = assemble_report(args.expected_manifest_hash, None, args.root, time.time(), [],
            [dict(name="unhandled_audit_error", passed=False, details=repr(exc))], frozen_binding=None,
            source_identity=None, preflight_reference=None, stage=None, category_mapping=None,
            completion_markers=None, episode_evidence=None, initial_input_pairing=None,
            saved_requests=None, serving_runs=None)
        result["status"], result["finished_unix"] = "failed", time.time()
    result["audit_script"] = dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))
    write_report(result, args.output)
    print(json.dumps(dict(status=result["status"], manifest_hash=result.get("manifest_hash"),
                          failed_checks=len(result.get("failed_checks", [])), report=str(Path(args.output).resolve())), indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
