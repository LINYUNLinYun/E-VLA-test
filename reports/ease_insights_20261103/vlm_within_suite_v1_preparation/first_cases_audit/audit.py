"""Bounded, readonly CPU audit for three already submitted within-suite case pairs.

This script reads exactly six episode records (three cases x Full/Jac) and two
serving-run artifacts. It never constructs an environment or initializes a
model. It extracts only the pure audit_requests helper and its pure arithmetic
dependencies from the previously accepted audit source.
"""
from collections import Counter
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

REPORT_DIR = Path(__file__).resolve().parent
REPORT_PATH = REPORT_DIR / "report.json"
SUMMARY_PATH = REPORT_DIR / "report.md"
STUDY = ROOT / "results/ease_insights_20261103/vlm_within_suite_v1/discovery"
MANIFEST_PATH = STUDY / "manifest.json"
MANIFEST_HASH = "6db11c5a8f1cdb5eab549f406ece3ba7b14661b2699da407c1ac87073e6c8a7d"
REQUEST_HELPER = ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/first_resource_session_cases/audit.py"
REQUEST_HELPER_SNAPSHOT = REPORT_DIR / "request_helper_source.py"
REFERENCE_STUDY = ROOT / "results/ease_insights_20261103/vlm_local_search_v1/discovery"
REFERENCE_REPORT = ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/audit.json"
CONDITIONS = ("full_d10", "jac_anchor")
CASES = (
    {"suite": "libero_90", "task": 15, "init": 38},
    {"suite": "libero_90", "task": 16, "init": 37},
    {"suite": "libero_90", "task": 16, "init": 38},
)
SERVICE_RUNS = {
    "full_d10": "discovery_shard00of01_1793868912852313137",
    "jac_anchor": "discovery_shard00of01_1793868913301389294",
}
EXPECTED_SERVERS = {
    "full_d10": {"pid": 753871, "gpu_uuid": "GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1"},
    "jac_anchor": {"pid": 753887, "gpu_uuid": "GPU-599477fb-0507-40dc-14df-d491f7a7b202"},
}
EXPECTED_CONTROLLERS = {
    "full_d10": {"pid": 753851, "gpu_uuid": "GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1"},
    "jac_anchor": {"pid": 753864, "gpu_uuid": "GPU-599477fb-0507-40dc-14df-d491f7a7b202"},
}
NORM_STATS_PATH = "/root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted/base/assets/physical-intelligence/libero/norm_stats.json"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def case_key(case):
    return (case["suite"], case["task"], case["init"])


def episode_dir(root, condition, case):
    return (Path(root) / condition / "episodes" / case["suite"] /
            ("task%02d" % case["task"]) / ("init%02d" % case["init"]))


def compile_functions(source_path, names, namespace):
    tree = ast.parse(Path(source_path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    found = {node.name for node in selected}
    if found != set(names):
        raise RuntimeError("Expected pure helper definitions are missing: %s" % sorted(set(names) - found))
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def pure_request_auditor():
    if not REQUEST_HELPER_SNAPSHOT.is_file() or sha(REQUEST_HELPER_SNAPSHOT) != sha(REQUEST_HELPER):
        raise RuntimeError("Saved request-helper source differs from the original helper")
    source = REQUEST_HELPER.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "audit_requests")
    loaded = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    forbidden = {"MANIFEST_HASH", "STUDY", "BASELINE", "NEW_CONDITIONS", "SAMPLED_RECEIPTS_PER_CONDITION", "audit_gate", "load_stage"}
    if loaded & forbidden:
        raise RuntimeError("Per-request helper unexpectedly depends on old stage/whitelist globals")

    namespace = {"Counter": Counter, "hashlib": hashlib, "json": json,
                 "np": np, "Path": Path, "math": __import__("math")}
    helper = compile_functions(REQUEST_HELPER, {"read", "episode_dir", "audit_requests"}, namespace)
    seed_source = ROOT / "preparations/idea1_v2_2/common.py"
    compile_functions(seed_source, {"digest", "seed"}, helper)
    recovery_source = ROOT / "preparations/ease_insights_20261103/vlm/recovery.py"
    compile_functions(recovery_source, {"scheduled_nfe"}, helper)
    return helper["audit_requests"], {
        "source_path": str(REQUEST_HELPER),
        "source_sha256": sha(REQUEST_HELPER),
        "snapshot_path": str(REQUEST_HELPER_SNAPSHOT),
        "snapshot_sha256": sha(REQUEST_HELPER_SNAPSHOT),
        "audit_requests_loaded_names": sorted(loaded),
        "old_stage_or_whitelist_globals_absent": True,
        "compiled_only": ["read", "episode_dir", "audit_requests", "digest", "seed", "scheduled_nfe"],
        "pure_dependencies": {
            "noise_seed_source": str(seed_source), "noise_seed_source_sha256": sha(seed_source),
            "schedule_source": str(recovery_source), "schedule_source_sha256": sha(recovery_source),
        },
    }


def runtime_revision(manifest):
    value = dict(
        name="vlm-within-suite-isolated-runtime-v1",
        scientific_manifest_hash=manifest["manifest_hash"],
        phase=manifest["phase"],
        protocol_evidence=manifest["protocol_evidence"],
        design_evidence=manifest["design_evidence"],
        wrapper_source_files=manifest["local_runtime_sources"],
        backend_reused="preparations.ease_insights_20261103.vlm.backend.Backend",
        mechanics_reused="preparations.ease_insights_20261103.vlm.policy.Policy.predict/reset/acknowledge",
        policy_action_limit=400,
        settling_actions=10,
        legacy_sources_changed=False,
        finite_gate_suites=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        gate_covers_new_tasks_or_all_prompts=False,
    )
    value["revision_hash"] = digest(value)
    return value


def add_check(checks, name, passed, **details):
    checks.append(dict(name=name, passed=bool(passed), **details))


def verify_pin_map(checks, name, pins):
    evidence = []
    for filename, expected in sorted(pins.items()):
        path = Path(filename)
        actual = sha(path) if path.is_file() else None
        evidence.append(dict(path=str(path), expected_sha256=expected,
                             sha256=actual, passed=actual == expected))
    add_check(checks, name, bool(evidence) and all(row["passed"] for row in evidence),
              count=len(evidence), mismatches=[row for row in evidence if not row["passed"]],
              inputs=evidence)
    return evidence


def layer_mask_matches(execution, mask):
    return (execution.get("kv_only_layers") == mask and
            execution.get("patched_layers") == mask and
            execution.get("qkv_layers") == [layer for layer in range(1, 19) if layer not in mask] and
            execution.get("hidden_update_layers") == [layer for layer in range(1, 18) if layer not in mask] and
            execution.get("grad_enabled") is False and execution.get("selector_calls") == 0)


def audit_saved_arrays(directory, request_count):
    path = directory / "arrays.npz"
    expected_keys = ({"actions_%d" % i for i in range(request_count)} |
                     {"raw_%d" % i for i in range(request_count)})
    with np.load(path, allow_pickle=False) as arrays:
        keys = set(arrays.files)
        shapes = {}
        issues = []
        if keys != expected_keys:
            issues.append("Saved action/raw array keys differ from the request count")
        for index in range(request_count):
            for prefix, shape in (("actions", (50, 7)), ("raw", (50, 32))):
                key = "%s_%d" % (prefix, index)
                if key not in arrays:
                    continue
                value = arrays[key]
                shapes[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                if value.shape != shape or value.dtype != np.float32:
                    issues.append("%s has unexpected saved shape/dtype" % key)
    return dict(passed=not issues, keys=sorted(keys), expected_key_count=2 * request_count,
                array_shapes=shapes, issues=issues)


def audit_case(root, manifest, revision, configs, official, assets, condition, case, request_audit):
    directory = episode_dir(root, condition, case)
    receipt_path = directory / "episode_receipt.json"
    result_path = directory / "result.json"
    identity_path = directory / "identity.json"
    complete_path = directory / "complete.json"
    receipt = read(receipt_path)
    result = read(result_path)
    identity = read(identity_path)
    complete = read(complete_path)
    protocol = identity.get("protocol", {})
    condition_config = configs[condition]
    task_key = (case["suite"], case["task"])
    official_case = official[task_key]
    task_asset = assets[task_key]
    physical = receipt.get("initial_gate", {}).get("physical_assets", {})
    expected_init_sha = official_case["init_sha256"][str(case["init"])]
    expected_server = EXPECTED_SERVERS[condition]

    issues = []
    def require(name, passed, **detail):
        if not passed:
            issues.append(name)
        return dict(name=name, passed=bool(passed), **detail)

    checks = []
    checks.append(require("receipt_result_complete_identity", all((
        receipt.get("valid") is True, result.get("valid") is True, complete.get("valid") is True,
        receipt.get("manifest_hash") == manifest["manifest_hash"],
        result.get("manifest_hash") == manifest["manifest_hash"],
        complete.get("manifest_hash") == manifest["manifest_hash"],
        receipt.get("condition") == condition, result.get("condition") == condition,
        receipt.get("case") == case, result.get("case") == case,
        identity.get("case") == case, identity.get("manifest_hash") == manifest["manifest_hash"],
        identity.get("config") == condition_config, result.get("config") == condition_config,
        receipt.get("initial_gate", {}).get("passed") is True,
        receipt.get("initial_gate", {}).get("case") == case,
    )), receipt_sha256=sha(receipt_path), result_sha256=sha(result_path),
        identity_sha256=sha(identity_path), complete_sha256=sha(complete_path)))
    checks.append(require("runtime_revision_and_server_protocol", all((
        receipt.get("technical_runtime_revision") == revision,
        protocol.get("technical_runtime_revision") == revision,
        protocol.get("condition_id") == condition,
        protocol.get("manifest_hash") == manifest["manifest_hash"],
        protocol.get("server_pid") == expected_server["pid"],
        protocol.get("gpu_uuid") == expected_server["gpu_uuid"],
        receipt.get("gpu_uuid") == expected_server["gpu_uuid"],
        protocol.get("qk_reference_passed") is True,
        protocol.get("real_input_parity_passed") is True,
        protocol.get("checkpoint_sha256") == manifest["checkpoints"]["pi0"]["sha256"],
    )), server_pid=protocol.get("server_pid"), gpu_uuid=protocol.get("gpu_uuid"),
        worker_pid=receipt.get("worker_pid"), runtime_revision_hash=protocol.get("technical_runtime_revision", {}).get("revision_hash")))
    checks.append(require("official_task_bddl_vector_prompt_and_policy_pins", all((
        physical.get("bddl_sha256") == official_case["bddl_sha256"] == task_asset["bddl_sha256"],
        physical.get("init_vector_sha256") == expected_init_sha,
        identity.get("bddl_sha256") == official_case["bddl_sha256"],
    )), expected_bddl_sha256=official_case["bddl_sha256"],
        receipt_bddl_sha256=physical.get("bddl_sha256"),
        expected_init_vector_sha256=expected_init_sha,
        receipt_init_vector_sha256=physical.get("init_vector_sha256"),
        identity_init_vector_sha256=identity.get("init_vector_sha256")))
    # The identity file stores BDDL and init-vector hashes at top level; compare
    # each independently so an unrelated identity mismatch remains explicit.
    checks[-1]["passed"] = all((
        physical.get("bddl_sha256") == official_case["bddl_sha256"] == task_asset["bddl_sha256"],
        identity.get("bddl_sha256") == official_case["bddl_sha256"],
        physical.get("init_vector_sha256") == expected_init_sha,
        identity.get("init_vector_sha256") == expected_init_sha,
        physical.get("prompt") == official_case["prompt"] == task_asset["benchmark_instruction"],
        identity.get("prompt") == official_case["prompt"],
        physical.get("env_seed") == 7 and identity.get("env_seed") == 7,
        physical.get("settling") == 10 and identity.get("settling") == 10,
        physical.get("policy_action_limit") == 400 and identity.get("policy_action_limit") == 400,
        physical.get("control_hz") == 20.0 and identity.get("control_hz") == 20.0,
        receipt.get("policy_action_limit") == 400 and receipt.get("settling_actions") == 10,
        result.get("policy_action_limit") == 400 and result.get("settling_actions") == 10,
        result.get("actions") == 400 and result.get("predictions") == receipt.get("predictions"),
    ))
    if not checks[-1]["passed"] and "official_task_bddl_vector_prompt_and_policy_pins" not in issues:
        issues.append("official_task_bddl_vector_prompt_and_policy_pins")

    artifacts = receipt.get("artifacts", {})
    artifact_hashes = []
    for name, expected_sha in sorted(artifacts.items()):
        path = directory / name
        actual_sha = sha(path) if path.is_file() else None
        artifact_hashes.append(dict(name=name, expected_sha256=expected_sha,
                                    sha256=actual_sha, passed=actual_sha == expected_sha))
    required_artifacts = {"arrays.npz", "complete.json", "identity.json", "infer.jsonl",
                          "input_0000.npz", "result.json", "telemetry.jsonl"}
    artifacts_ok = (required_artifacts.issubset(artifacts) and
                    len(artifact_hashes) == len(artifacts) and
                    all(row["passed"] for row in artifact_hashes))
    checks.append(require("receipt_artifacts_current_sha", artifacts_ok,
        episode_receipt_sha256=sha(receipt_path), current_artifact_hashes=artifact_hashes))

    first_reference_path = Path(receipt["initial_gate"]["reference"])
    reference = read(first_reference_path)
    reference_ok = all((
        first_reference_path.resolve() == (root / "initial_observations" / case["suite"] /
            ("task%02d" % case["task"]) / ("init%02d.json" % case["init"])).resolve(),
        reference.get("manifest_hash") == manifest["manifest_hash"],
        reference.get("case") == case,
        reference.get("physical_assets", {}).get("init_vector_sha256") == expected_init_sha,
        reference.get("physical_assets", {}).get("bddl_sha256") == official_case["bddl_sha256"],
        reference.get("physical_assets", {}).get("prompt") == official_case["prompt"],
        reference.get("physical_assets", {}).get("env_seed") == 7,
        reference.get("physical_assets", {}).get("settling") == 10,
        reference.get("physical_assets", {}).get("policy_action_limit") == 400,
        reference.get("input_hash") == receipt["initial_gate"].get("input_hash"),
        reference.get("seed") == receipt["initial_gate"].get("seed"),
    ))
    checks.append(require("official_initial_observation_reference", reference_ok,
        reference_path=str(first_reference_path), reference_sha256=sha(first_reference_path),
        expected_init_vector_sha256=expected_init_sha,
        reference_init_vector_sha256=reference.get("physical_assets", {}).get("init_vector_sha256")))

    infer_rows = [json.loads(line) for line in (directory / "infer.jsonl").read_text().splitlines()]
    if not infer_rows:
        raise RuntimeError("Selected episode has no infer rows: %s" % directory)
    input_path = directory / "input_0000.npz"
    request_input_sha_ok = infer_rows[0].get("input_file_sha256") == sha(input_path)
    value_digest_namespace = {"hashlib": hashlib, "json": json, "np": np}
    runtime_client = ROOT / "preparations/idea1_v2_1/runtime_client.py"
    compile_functions(runtime_client, {"value_digest", "input_digest"}, value_digest_namespace)
    with np.load(input_path, allow_pickle=False) as saved:
        saved_instruction = str(saved["instruction"].item())
        saved_seed = int(saved["seed"].item())
        saved_obs = {key: saved[key] for key in saved.files if key not in ("instruction", "seed")}
        calculated_input_hash = value_digest_namespace["input_digest"](saved_obs, saved_instruction)
    input_hash_ok = all((
        request_input_sha_ok,
        saved_instruction == official_case["prompt"],
        calculated_input_hash == infer_rows[0].get("input_hash"),
        calculated_input_hash == receipt["initial_gate"].get("input_hash"),
        saved_seed == infer_rows[0].get("seed") == receipt["initial_gate"].get("seed"),
        reference.get("input_hash") == calculated_input_hash,
    ))
    checks.append(require("saved_first_input_hash_and_seed", input_hash_ok,
        input_file_sha256=sha(input_path), infer_file_sha256=sha(directory / "infer.jsonl"),
        saved_instruction=saved_instruction, saved_seed=saved_seed,
        input_hash=calculated_input_hash, gate_input_hash=receipt["initial_gate"].get("input_hash"),
        infer_input_hash=infer_rows[0].get("input_hash"),
        noise_seed=receipt["initial_gate"].get("seed")))

    config = condition_config
    request_result = request_audit(root, condition, case, config, {"success": result.get("success")})
    arrays_result = audit_saved_arrays(directory, len(infer_rows))
    checks.append(require("saved_request_schedule_history_mask_noise", request_result["passed"],
        requests=request_result["requests"], actions=request_result["actions"],
        scheduled_requests=request_result["scheduled_requests"], issues=request_result["issues"],
        success=result.get("success"), success_used_as_acceptance_gate=False))
    checks.append(require("saved_action_array_shapes", arrays_result["passed"], arrays_result=arrays_result))

    checkpoint_assets = protocol.get("checkpoint_assets", {})
    norm_record = checkpoint_assets.get(NORM_STATS_PATH) if isinstance(checkpoint_assets, dict) else None
    norm_pinned_sha = manifest["checkpoints"]["pi0"]["identity_artifacts"].get(NORM_STATS_PATH)
    norm_bound = isinstance(norm_record, dict) and norm_record.get("sha256") == norm_pinned_sha
    checks.append(require("runtime_norm_asset_bound_to_manifest_pin", norm_bound,
        norm_stats_path=NORM_STATS_PATH, manifest_pin_sha256=norm_pinned_sha,
        episode_protocol_norm_record=norm_record))

    success = result.get("success")
    return dict(condition=condition, case=case, passed=all(row["passed"] for row in checks),
        success=success, successful_outcome_required=False, receipt_mtime_ns=receipt_path.stat().st_mtime_ns,
        worker_pid=receipt.get("worker_pid"), gpu_uuid=receipt.get("gpu_uuid"),
        request_count=len(infer_rows), action_count=result.get("actions"),
        result_sha256=sha(result_path), receipt_sha256=sha(receipt_path),
        identity_sha256=sha(identity_path), complete_sha256=sha(complete_path),
        checks=checks, issues=issues)


def audit_service(root, manifest, condition, reference_actions, reference_sha, baseline_info):
    run_name = SERVICE_RUNS[condition]
    run_dir = root / condition / "runs" / run_name
    gate_path = run_dir / "gate.json"
    protocol_path = run_dir / "protocol.json"
    identity_path = run_dir / "identity.json"
    cross_path = run_dir / "cross_service.json"
    actions_path = run_dir / "gate_actions.npz"
    preparation_path = run_dir / "preparation.json"
    gate = read(gate_path)
    protocol = read(protocol_path)
    identity = read(identity_path)
    cross = read(cross_path)
    issues = []
    expected = EXPECTED_SERVERS[condition]
    expected_configs = {row["id"]: row for row in manifest["conditions"]}
    checks = []
    checks.append(dict(name="saved_332_check_service_gate", passed=(
        gate.get("passed") is True and gate.get("manifest_hash") == manifest["manifest_hash"] and
        gate.get("condition_ids") == [item["id"] for item in manifest["conditions"]] and
        len(gate.get("checks", [])) == 332 and all(row.get("passed") is True for row in gate.get("checks", []))),
        gate_check_count=len(gate.get("checks", [])), gate_sha256=sha(gate_path)))
    checks.append(dict(name="service_protocol_manifest_server_gpu_and_parity", passed=all((
        protocol.get("condition_id") == condition,
        protocol.get("manifest_hash") == manifest["manifest_hash"],
        protocol.get("config") == expected_configs[condition],
        protocol.get("server_pid") == expected["pid"],
        protocol.get("gpu_uuid") == expected["gpu_uuid"],
        protocol.get("qk_reference_passed") is True,
        protocol.get("real_input_parity_passed") is True,
        protocol.get("checkpoint_sha256") == manifest["checkpoints"]["pi0"]["sha256"],
        identity.get("condition") == condition,
        identity.get("manifest_hash") == manifest["manifest_hash"],
        identity.get("stateful_workers_per_server") == 1,
    )), condition=condition, server_pid=protocol.get("server_pid"), gpu_uuid=protocol.get("gpu_uuid"),
        qk_reference_passed=protocol.get("qk_reference_passed"),
        real_input_parity_passed=protocol.get("real_input_parity_passed"),
        protocol_sha256=sha(protocol_path), identity_sha256=sha(identity_path)))
    selected_execution_name = ("libero_spatial/full_native" if condition == "full_d10" else
                               "libero_spatial/jac_anchor/actual_execution")
    selected = [row for row in gate.get("checks", []) if row.get("name") == selected_execution_name]
    if condition == "full_d10":
        execution_ok = (len(selected) == 1 and selected[0].get("passed") is True and
                         selected[0].get("decoded_max_abs") == 0.0 and
                         selected[0].get("raw_max_abs") == 0.0)
    else:
        execution = selected[0].get("execution_audit", {}) if len(selected) == 1 else {}
        mask = list(expected_configs[condition]["skip"])
        execution_ok = (len(selected) == 1 and layer_mask_matches(execution, mask) and
                        execution.get("captured_layers") == [] and selected[0].get("passed") is True)
    checks.append(dict(name="selected_actual_execution_check", passed=execution_ok,
        selected_name=selected_execution_name, count=len(selected),
        check=selected[0] if selected else None))
    cross_checks = cross.get("checks", [])
    checks.append(dict(name="saved_cross_service_receipt", passed=(
        cross.get("passed") is True and cross.get("scientific_manifest_hash") == manifest["manifest_hash"] and
        len(cross_checks) == 24 and all(row.get("passed") is True for row in cross_checks)),
        cross_check_count=len(cross_checks), cross_service_sha256=sha(cross_path)))

    numeric = []
    with np.load(reference_actions, allow_pickle=False) as reference, np.load(actions_path, allow_pickle=False) as actual:
        keys_match = set(reference.files) == set(actual.files)
        reference_key_count = len(reference.files)
        if keys_match:
            for key in sorted(reference.files):
                delta = np.abs(actual[key] - reference[key])
                passed = bool(np.allclose(actual[key], reference[key], atol=1e-5, rtol=1e-5))
                numeric.append(dict(key=key, shape=list(actual[key].shape),
                                    max_abs=float(delta.max()), passed=passed))
    numeric_ok = (keys_match and len(numeric) == reference_key_count and
                  all(row["passed"] for row in numeric))
    checks.append(dict(name="gate_actions_vs_old_accepted_jac_1e-5_reference", passed=numeric_ok,
        reference_path=str(reference_actions), reference_sha256=reference_sha,
        actual_path=str(actions_path), actual_sha256=sha(actions_path),
        rtol=1e-5, atol=1e-5, keys_match=keys_match, comparisons=numeric,
        old_acceptance=baseline_info))

    service_artifacts = []
    for path in (gate_path, protocol_path, identity_path, cross_path, actions_path, preparation_path):
        if path.is_file():
            service_artifacts.append(dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size))
    issues = [check["name"] for check in checks if not check["passed"]]
    return dict(condition=condition, run_path=str(run_dir), passed=not issues, checks=checks,
        issues=issues, current_service_artifact_sha=service_artifacts,
        server_pid=protocol.get("server_pid"), gpu_uuid=protocol.get("gpu_uuid"),
        request_count=protocol.get("measured_request_count"), gate_check_count=len(gate.get("checks", [])),
        cross_service_check_count=len(cross_checks))


def main():
    started = time.time()
    checks = []
    manifest_bytes_sha = sha(MANIFEST_PATH)
    manifest = read(MANIFEST_PATH)
    supplied_hash = manifest.get("manifest_hash")
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_hash", None)
    computed_hash = digest(payload)
    add_check(checks, "new_within_suite_manifest_identity",
        supplied_hash == MANIFEST_HASH and computed_hash == MANIFEST_HASH,
        manifest_hash_field=supplied_hash, computed_manifest_hash=computed_hash,
        raw_manifest_file_sha256=manifest_bytes_sha, output_root=str(STUDY))

    source_groups = {}
    source_groups["source_files"] = verify_pin_map(checks, "frozen_source_files", manifest.get("source_files", {}))
    source_groups["local_runtime_sources"] = verify_pin_map(checks, "frozen_new_runtime_sources", manifest.get("local_runtime_sources", {}))
    source_groups["analysis_sources"] = verify_pin_map(checks, "frozen_new_analysis_sources", manifest.get("analysis_sources", {}))
    for name in ("protocol_evidence", "design_evidence", "candidate_evidence", "candidate_input_evidence"):
        evidence = manifest.get(name, {})
        path = Path(evidence.get("path", ""))
        actual = sha(path) if path.is_file() else None
        add_check(checks, "frozen_evidence:" + name, actual == evidence.get("sha256"),
            path=str(path), sha256=actual, expected_sha256=evidence.get("sha256"))
    resource_evidence = manifest.get("resource_execution_revision", {}).get("evidence", {})
    resource_path = Path(resource_evidence.get("path", ""))
    resource_actual = sha(resource_path) if resource_path.is_file() else None
    add_check(checks, "frozen_resource_execution_evidence",
        resource_actual == resource_evidence.get("sha256"), path=str(resource_path),
        sha256=resource_actual, expected_sha256=resource_evidence.get("sha256"))

    checkpoint = manifest["checkpoints"]["pi0"]
    checkpoint_path = Path(checkpoint["path"])
    checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path.is_file() else None
    add_check(checks, "checkpoint_manifest_sha_pin", isinstance(checkpoint.get("sha256"), str) and
        len(checkpoint["sha256"]) == 64,
        path=str(checkpoint_path), expected_checkpoint_sha256=checkpoint["sha256"],
        content_rehashed=False, size_bytes=checkpoint_size)
    checkpoint_identity = verify_pin_map(checks, "checkpoint_identity_artifacts",
                                         checkpoint.get("identity_artifacts", {}))
    norm_pin = checkpoint.get("identity_artifacts", {}).get(NORM_STATS_PATH)
    add_check(checks, "new_manifest_norm_asset_pin", bool(norm_pin) and sha(NORM_STATS_PATH) == norm_pin,
        path=NORM_STATS_PATH, sha256=sha(NORM_STATS_PATH), manifest_pin_sha256=norm_pin)

    expected_revision = runtime_revision(manifest)
    request_audit, helper_identity = pure_request_auditor()
    add_check(checks, "isolated_pure_request_helper", helper_identity["old_stage_or_whitelist_globals_absent"],
        helper_identity=helper_identity)

    official = {(row["suite"], row["task"]): row for row in manifest["official_init_availability"]["cases"]}
    assets = {(row["suite"], row["task"]): row for row in manifest["task_assets"]}
    selected_task_assets = []
    selected_task_ok = True
    for task_key in sorted({(case["suite"], case["task"]) for case in CASES}):
        official_task = official[task_key]
        task_asset = assets[task_key]
        bddl_expected = official_task["bddl_sha256"]
        init_raw_expected = official_task["init_file_sha256"]
        bddl_path = Path(official_task["bddl_path"])
        init_path = Path(official_task["init_file"])
        bddl_actual = sha(bddl_path) if bddl_path.is_file() else None
        init_actual = sha(init_path) if init_path.is_file() else None
        passed = all((
            bddl_actual == bddl_expected == task_asset["bddl_sha256"],
            init_actual == init_raw_expected == task_asset["init_sha256"],
            official_task["prompt"] == task_asset["benchmark_instruction"],
        ))
        selected_task_ok = selected_task_ok and passed
        selected_task_assets.append(dict(suite=task_key[0], task=task_key[1], prompt=official_task["prompt"],
            bddl_path=str(bddl_path), bddl_sha256=bddl_actual, expected_bddl_sha256=bddl_expected,
            init_file=str(init_path), init_file_sha256=init_actual,
            expected_init_file_sha256=init_raw_expected,
            manifest_init_vectors={case["init"]: official_task["init_sha256"][str(case["init"])]
                                   for case in CASES if (case["suite"], case["task"]) == task_key},
            passed=passed))
    add_check(checks, "selected_official_task_bddl_raw_file_and_init_raw_file_pins",
        selected_task_ok, tasks=selected_task_assets)

    manifest_reference = manifest["preflight"]["cross_service_reference"]
    reference_actions = Path(manifest_reference["path"])
    reference_sha = sha(reference_actions) if reference_actions.is_file() else None
    old_reference_identity_path = REFERENCE_STUDY / "cross_service_reference.json"
    old_reference_identity = read(old_reference_identity_path)
    old_report = read(REFERENCE_REPORT)
    old_services = old_report.get("runtime_services", [])
    baseline_service = old_services[0].get("path") if len(old_services) == 1 else None
    baseline_ok = all((
        old_report.get("status") == "accepted",
        old_report.get("failed_checks") == [],
        old_report.get("manifest_hash") == manifest["preflight"]["cross_service_reference"]["manifest_hash"],
        len(old_services) == 1,
        Path(old_reference_identity.get("first_service", "")).resolve() == Path(baseline_service or "/nonexistent").resolve(),
        reference_sha == manifest_reference["sha256"],
    ))
    baseline_info = dict(report_path=str(REFERENCE_REPORT), report_sha256=sha(REFERENCE_REPORT),
        status=old_report.get("status"), manifest_hash=old_report.get("manifest_hash"),
        service_path=baseline_service, reference_identity_path=str(old_reference_identity_path),
        reference_identity_sha256=sha(old_reference_identity_path),
        reference_actions_path=str(reference_actions), reference_actions_sha256=reference_sha,
        expected_reference_sha256=manifest_reference["sha256"])
    add_check(checks, "old_accepted_jac_reference_identity", baseline_ok, reference=baseline_info)

    controllers = read(STUDY / "controllers/controller_processes.json")
    controller_rows = controllers.get("processes", [])
    controller_statuses = [read(path) for path in sorted((STUDY / "controllers").glob("GPU-*/status.json"))]
    launch_identity = {(row.get("pid"), row.get("gpu_uuid")) for row in controller_rows}
    expected_launch_identity = {(row["pid"], row["gpu_uuid"]) for row in EXPECTED_CONTROLLERS.values()}
    add_check(checks, "frozen_controller_manifest_and_keepalive", all((
        controllers.get("manifest_hash") == manifest["manifest_hash"],
        len(controller_rows) == 2,
        controllers.get("original_holders_retained") is True,
        controllers.get("legacy_broker_quiesced") is True,
        launch_identity == expected_launch_identity,
        len(controller_statuses) == 2 and all(row.get("manifest_hash") == manifest["manifest_hash"] for row in controller_statuses),
        {row.get("controller_pid") for row in controller_statuses} == {row["pid"] for row in EXPECTED_CONTROLLERS.values()},
    )), controller_processes_sha256=sha(STUDY / "controllers/controller_processes.json"),
        controller_launch_identity=[dict(pid=row.get("pid"), gpu_uuid=row.get("gpu_uuid")) for row in controller_rows],
        current_status=[dict(controller_pid=row.get("controller_pid"), current_condition=row.get("condition"),
                             phase=row.get("phase"), manifest_hash=row.get("manifest_hash")) for row in controller_statuses])

    configs = {row["id"]: row for row in manifest["conditions"]}
    case_results = []
    for case in CASES:
        for condition in CONDITIONS:
            case_results.append(audit_case(STUDY, manifest, expected_revision, configs, official,
                                           assets, condition, case, request_audit))
    # Cross-condition pairing is limited to the three selected physical cases.
    paired_cases = []
    records_by_case = {}
    for row in case_results:
        records_by_case.setdefault(case_key(row["case"]), {})[row["condition"]] = row
    for case in CASES:
        pair = records_by_case[case_key(case)]
        full = pair["full_d10"]
        jac = pair["jac_anchor"]
        full_gate = read(episode_dir(STUDY, "full_d10", case) / "episode_receipt.json")["initial_gate"]
        jac_gate = read(episode_dir(STUDY, "jac_anchor", case) / "episode_receipt.json")["initial_gate"]
        full_identity = read(episode_dir(STUDY, "full_d10", case) / "identity.json")
        jac_identity = read(episode_dir(STUDY, "jac_anchor", case) / "identity.json")
        same = all((
            full_gate.get("input_hash") == jac_gate.get("input_hash"),
            full_gate.get("seed") == jac_gate.get("seed"),
            full_gate.get("physical_assets") == jac_gate.get("physical_assets"),
            full_identity.get("prompt") == jac_identity.get("prompt"),
            full_identity.get("bddl_sha256") == jac_identity.get("bddl_sha256"),
            full_identity.get("init_vector_sha256") == jac_identity.get("init_vector_sha256"),
            full_identity.get("env_seed") == jac_identity.get("env_seed") == 7,
            full_identity.get("settling") == jac_identity.get("settling") == 10,
            full_identity.get("policy_action_limit") == jac_identity.get("policy_action_limit") == 400,
        ))
        paired_cases.append(dict(case=case, passed=same,
            input_hashes={"full_d10": full_gate.get("input_hash"), "jac_anchor": jac_gate.get("input_hash")},
            seeds={"full_d10": full_gate.get("seed"), "jac_anchor": jac_gate.get("seed")},
            physical_assets=full_gate.get("physical_assets"),
            success_outcomes={"full_d10": full["success"], "jac_anchor": jac["success"]},
            successful_outcome_required=False))
    add_check(checks, "same_case_first_input_prompt_seed7_settling10_action_limit400_and_physical_pins",
        all(row["passed"] for row in paired_cases), pairs=paired_cases)

    service_results = []
    for condition in CONDITIONS:
        service_results.append(audit_service(STUDY, manifest, condition,
            reference_actions, reference_sha, baseline_info))
    add_check(checks, "two_actual_service_gates", len(service_results) == 2 and
        all(row["passed"] for row in service_results), services=service_results)

    failed = [row for row in checks if not row["passed"]]
    total_requests = sum(row["request_count"] for row in case_results)
    result = dict(
        revision="vlm-within-suite-first-three-paired-cases-readonly-audit-v1",
        status="sample_pass" if not failed else "issues",
        started_unix=started,
        finished_unix=time.time(),
        manifest_hash=supplied_hash,
        manifest_file_sha256=manifest_bytes_sha,
        root=str(ROOT),
        output_root=str(STUDY),
        scope=dict(
            conditions=list(CONDITIONS),
            selected_cases=list(CASES),
            episode_records=len(case_results),
            case_count=len(CASES),
            episode_count_expected=6,
            actual_request_count=total_requests,
            serving_runs=len(service_results),
            sample_selection="Convenience sample: the first three paired Full/Jac case directories already observed in the initial bounded artifact listing; selected by case identity/completion availability before reading success values. This is not random and does not claim all-stage earliest ordering.",
            full_episode_scan=False,
            full_432_episode_validation=False,
            full_332_gate_recomputation=False,
            full_success_gate_used=False,
            environment_or_model_started=False,
            gpu_recomputed=False,
            permitted_writes=[str(REPORT_DIR)],
            limits=[
                "Only three physical cases x two conditions were audited.",
                "Only two saved service runs and their finite gate artifacts were read.",
                "No new prompts, unsampled episodes, complete matrix, environment/video, or model runtime were inspected.",
                "The 6.48 GB checkpoint binary was not reread for a second full SHA pass; its manifest pin is cross-checked in all six episode protocols and both service protocols, while its three small checkpoint identity artifacts were hashed from disk.",
                "The old finite Jac reference comparison is limited to its saved gate-action arrays at unchanged atol=rtol=1e-5.",
                "Gate JSON checks were read as saved evidence; they were not recomputed or extended to new tasks.",
            ],
        ),
        helper_identity=helper_identity,
        expected_runtime_revision=expected_revision,
        frozen_source_groups=source_groups,
        checkpoint_identity_artifacts=checkpoint_identity,
        selected_task_assets=selected_task_assets,
        old_accepted_reference=baseline_info,
        case_results=case_results,
        paired_case_inputs=paired_cases,
        service_results=service_results,
        checks=checks,
        failed_checks=failed,
        audit_script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
    )
    REPORT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = [
        "# Within-suite first paired cases audit",
        "",
        "Status: **%s**" % result["status"],
        "",
        "Manifest field hash: `%s`" % supplied_hash,
        "Raw manifest JSON SHA256: `%s`" % manifest_bytes_sha,
        "",
        "Audited %d physical cases x %d conditions = %d episode records, with %d saved inference requests; compared %d service runs." %
        (len(CASES), len(CONDITIONS), len(case_results), total_requests, len(service_results)),
        "",
        "Selected cases: " + ", ".join("task%02d/init%02d" % (case["task"], case["init"]) for case in CASES) + ".",
        "",
        "Sample-selection boundary: convenience sample of paired records already observed in the initial bounded listing; it is not random and is not claimed to be earliest across the entire phase.",
        "Success values were retained as outcomes and never used as an acceptance gate.",
        "No new episode, environment, or model was started; no GPU recomputation was done.",
        "",
        "The full 432-episode matrix, unselected prompts, and all-stage success/quality are not covered.",
        "",
        "Machine-readable details and per-file current SHA256 values are in `report.json`.",
    ]
    SUMMARY_PATH.write_text("\n".join(summary) + "\n")
    print(json.dumps(dict(status=result["status"], failed_checks=failed,
        cases=len(CASES), episode_records=len(case_results), requests=total_requests,
        service_runs=len(service_results), report=str(REPORT_PATH),
        report_sha256=sha(REPORT_PATH), audit_sha256=result["audit_script"]["sha256"]), indent=2))


if __name__ == "__main__":
    main()
