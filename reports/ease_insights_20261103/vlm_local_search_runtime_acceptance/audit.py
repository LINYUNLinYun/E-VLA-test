"""Read-only acceptance of the first real completed local-search condition.

This does not select paths, summarize task preferences, or change scientific
artifacts. Only this report directory receives output. Run with the FLASH venv.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from preparations.ease_insights_20261103.vlm_local_search_v1.selection import load_stage
from preparations.ease_insights_20261103.vlm_local_search_v1.common import sha, atomic_json
from preparations.idea1_v2_2.common import seed


def read(path):
    return json.loads(Path(path).read_text())


def main(root, output):
    started = time.time()
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.parent != Path(__file__).resolve().parent:
        raise ValueError("Output must remain in this acceptance-report directory")
    checks, inputs = [], {}
    def require(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))
    def evidence(path):
        path = Path(path).resolve()
        value = dict(path=str(path), sha256=sha(path))
        inputs[str(path)] = value
        return value
    audit = load_stage(root, "discovery")
    manifest = audit["manifest"]
    evidence(root / "manifest.json")
    cfg = next(c for c in manifest["conditions"] if c["id"] == "jac_anchor")
    source_groups = {}
    for group, expected_count in (("source_files", 46), ("analysis_sources", 6), ("local_runtime_sources", 9)):
        values = []
        for path, expected in manifest[group].items():
            item = evidence(ROOT / path)
            values.append(dict(item, expected_sha256=expected, passed=item["sha256"] == expected))
        source_groups[group] = values
        require(group, len(values) == expected_count and all(v["passed"] for v in values), count=len(values))
    for name in ("protocol_evidence", "design_evidence"):
        item = evidence(manifest[name]["path"])
        require(name, item["sha256"] == manifest[name]["sha256"])
    records = {case: record for (condition, case), record in audit["records"].items() if condition == "jac_anchor"}
    expected = {(suite, task, initial) for suite in manifest["suites"] for task in range(10) for initial in (36, 37, 38)}
    require("jac_exact_120_case_universe", set(records) == expected, valid=len(records), expected=120)
    require("load_stage_has_no_invalid_or_unexpected", not audit["invalid"] and not audit["unexpected"],
        invalid=audit["invalid"], unexpected=audit["unexpected"])
    require("jac_has_no_missing_receipt", not [r for r in audit["missing"] if r["condition"] == "jac_anchor"])
    marker = root / "jac_anchor/discovery_shard00of01_complete.json"
    done = read(marker); evidence(marker)
    require("condition_completion_marker", done.get("complete") is True and done.get("condition") == "jac_anchor" and done.get("manifest_hash") == manifest["manifest_hash"])
    runs = []
    for path in sorted((root / "jac_anchor/runs").glob("*/result.json")):
        result = read(path)
        if result.get("complete") is True:
            runs.append(path.parent)
    require("one_completed_first_service", len(runs) == 1, completed_services=len(runs))
    services = []
    condition_ids = [c["id"] for c in manifest["conditions"]]
    configs = {c["id"]: c for c in manifest["conditions"]}
    gate_suffixes = ("actual_capture", "residual_identity", "same_obs_reconstruct_full_d2", "reconstructed_kv",
        "actual_execution", "pure_native", "extrapolation_reference", "explicit_noise_repeat")
    gate_names = set()
    for suite in manifest["suites"]:
        gate_names.add(suite + "/full_native")
        for cid in condition_ids:
            gate_names.update(suite + "/" + cid + "/policy_round" + str(i) for i in range(6))
            gate_names.add(suite + "/" + cid + "/schedule")
            if cid != "full_d10":
                gate_names.update(suite + "/" + cid + "/" + suffix for suffix in gate_suffixes)
    for run in runs:
        values = {}
        for name in ("result.json", "protocol.json", "gate.json", "cross_service.json", "qk_receipt.json", "fixed_input_cost.json", "preparation.json"):
            values[name] = read(run / name); evidence(run / name)
        evidence(run / "gate_actions.npz")
        result, protocol, gate = (values[n] for n in ("result.json", "protocol.json", "gate.json"))
        require("service_identity", result["manifest_hash"] == manifest["manifest_hash"] == protocol["manifest_hash"] == gate["manifest_hash"] and
            result["condition"] == protocol["condition_id"] == "jac_anchor" and protocol["config"] == cfg and
            protocol["fixed_skip_layers"] == cfg["skip"] and protocol["refresh_D"] == 10 and
            protocol["real_input_parity_passed"] is True and protocol["qk_reference_passed"] is True and
            protocol["checkpoint_sha256"] == manifest["checkpoints"]["pi0"]["sha256"] and result["completed_after"] == 120)
        require("gate_332_exact_checks_all_six_conditions", gate["passed"] is True and len(gate["checks"]) == 332 and
            len({c["name"] for c in gate["checks"]}) == 332 and {c["name"] for c in gate["checks"]} == gate_names and
            all(c["passed"] is True for c in gate["checks"]) and gate["condition_ids"] == condition_ids and
            gate["atol"] == gate["rtol"] == 1e-5, checks=len(gate["checks"]))
        for check in gate["checks"]:
            if check["name"].endswith("/actual_execution"):
                cid = check["name"].split("/")[1]; mask = configs[cid]["skip"]
                execution = check["execution_audit"]
                require("gate_mask:" + check["name"], execution["kv_only_layers"] == mask and execution["patched_layers"] == mask and
                    execution["qkv_layers"] == [l for l in range(1,19) if l not in mask] and
                    execution["hidden_update_layers"] == [l for l in range(1,18) if l not in mask] and
                    execution["captured_layers"] == [] and execution["grad_enabled"] is False and execution["selector_calls"] == 0)
        qk = values["qk_receipt.json"]
        require("qk_reference_passed_unchanged_tolerance", qk["passed"] is True and qk["unsafe_legacy_kernel_executed"] is False and
            qk["atol"] == .0009765625 and qk["rtol"] == .0078125 and
            qk["source_sha256"] == manifest["source_files"]["preparations/idea1_v3/safe_qk.py"])
        cross = values["cross_service.json"]
        require("cross_service_reference_receipt", cross["passed"] is True and len(cross["checks"]) == 24 and
            all(c["passed"] is True for c in cross["checks"]) and cross["atol"] == cross["rtol"] == 1e-5)
        cost = values["fixed_input_cost.json"]
        warm, measured = 0, 0
        cycles = {cid: [] for cid in condition_ids}
        cost_errors = []
        if cost["manifest_hash"] != manifest["manifest_hash"] or [s["suite"] for s in cost["suites"]] != manifest["suites"]:
            cost_errors.append("Manifest/suite coverage")
        for suite in cost["suites"]:
            seeds = []
            if {b["condition"] for b in suite["conditions"]} != set(condition_ids):
                cost_errors.append("Condition coverage: " + suite["suite"])
            for block in suite["conditions"]:
                cid = block["condition"]
                rows = block["warmup"] + block["measured"]
                warm += len(block["warmup"]); measured += len(block["measured"])
                if (block["config"] != configs[cid] or len(block["warmup"]) != 6 or len(block["measured"]) != 12 or
                    [r["round"] for r in rows] != list(range(18))):
                    cost_errors.append("Config/round coverage: " + suite["suite"] + "/" + cid)
                seeds.append([(r["seed"], r["model_noise_sha256"]) for r in rows])
                for row in rows:
                    full = cid == "full_d10" or row["round"] % 3 == 0
                    if row["refresh"] != full or len(row["model_timing"]["nodes"]) - 1 != (10 if full else 2):
                        cost_errors.append("Schedule: " + suite["suite"] + "/" + cid)
                    noise = np.random.default_rng(row["seed"]).standard_normal((50,32), dtype=np.float32)
                    if hashlib.sha256(noise.tobytes()).hexdigest() != row["model_noise_sha256"]:
                        cost_errors.append("Noise identity: " + suite["suite"] + "/" + cid)
                cycles[cid].extend(sum(r["policy_wall_ms"] for r in block["measured"][i:i+3]) for i in range(0,12,3))
            if any(row != seeds[0] for row in seeds[1:]):
                cost_errors.append("Between-condition fixed-input probe noise pairing: " + suite["suite"])
        require("fixed_input_probe_144_warm_288_measured", warm == 144 == cost["total_warmup_requests"] and measured == 288 == cost["total_measured_requests"] and not cost_errors,
            warmup_requests=warm, measured_requests=measured, errors=cost_errors)
        services.append(dict(path=str(run), gpu_uuid=protocol["gpu_uuid"], server_pid=protocol["server_pid"],
            numerical_gate_checks=len(gate["checks"]), conditions=condition_ids, qk_reference=qk,
            fixed_input_cost=dict(warmup_requests=warm, measured_requests=measured,
                full_three_request_native_policy_ms=statistics.mean(cycles["full_d10"]),
                jac_three_request_native_policy_ms=statistics.mean(cycles["jac_anchor"]),
                full_over_jac_fixed_input_cycle=statistics.mean(cycles["full_d10"])/statistics.mean(cycles["jac_anchor"]),
                scope="One completed service, four fixed real suite inputs; native policy only, RPC/environment excluded; no rollout acceleration or path preference claim"),
            wall_costs=result, preparation=values["preparation.json"],
            legacy_metadata_notes=dict(mask_variant=protocol.get("mask_variant"), authoritative_fixed_skip_layers=protocol["fixed_skip_layers"],
                note="mask_variant is inherited legacy backend provenance; active config, fixed_skip_layers and every actual execution audit use the new mask. QK diagnostic_scope retains reused diagnostic wording, not an assertion that this service failed.")))
    request_counts = Counter(); residual_ratios = Counter(); runtime_revisions = Counter(); request_errors = []
    costs = dict(native_policy_ms=0., client_rpc_ms=0., executed_actions=0, summed_episode_wall_s=0.)
    initial_references = []; episode_evidence = []
    for scene_key, record in sorted(records.items()):
        suite, task, initial = scene_key
        directory = root / "jac_anchor/episodes" / suite / ("task%02d" % task) / ("init%02d" % initial)
        receipt = read(directory / "episode_receipt.json")
        episode_evidence.append(evidence(directory / "episode_receipt.json"))
        runtime_revisions[receipt["technical_runtime_revision"]["revision_hash"]] += 1
        for filename, expected_sha in receipt["artifacts"].items():
            item = evidence(directory / filename)
            if item["sha256"] != expected_sha:
                request_errors.append(dict(case=record["case"], issue="Artifact changed after load_stage", file=filename))
        ref = Path(receipt["initial_gate"]["reference"])
        evidence(ref)
        reference = read(ref)
        initial_references.append(dict(case=record["case"], reference=str(ref), created_by_condition=reference.get("condition"),
            input_hash=reference["input_hash"], noise_seed=reference["seed"]))
        rows = [json.loads(line) for line in (directory / "infer.jsonl").read_text().splitlines()]
        history = []; step = 0; per_case = Counter()
        for index, row in enumerate(rows):
            full = index % 3 == 0; mask = [] if full else cfg["skip"]
            capture = cfg["skip"] if full else []
            age = None if not history else step - history[-1]
            gap = None if len(history) < 2 else history[-1] - history[-2]
            ratio = 0. if gap is None else age/gap
            second_history = gap is not None
            if full:
                history = (history + [step])[-2:]
            route = row["route"]; execution = route["execution_audit"]
            expected_seed = seed(suite, task, initial, index)
            expected_noise = hashlib.sha256(np.random.default_rng(expected_seed).standard_normal((50,32), dtype=np.float32).tobytes()).hexdigest()
            valid = (row["round"] == index and row["step"] == step and row["replan"] == index and
                row["seed"] == expected_seed and row["model_noise_sha256"] == expected_noise and
                row["execute_length"] == 5 and type(row["executed_length"]) is int and 0 < row["executed_length"] <= 5 and
                route["refresh"] == full and route["path"] == ("full" if full else "extrapolate") and
                route["skip_layers"] == mask and route["capture_layers"] == capture and route["compensation"] == "extrapolate" and
                route["has_second_history"] == second_history and abs(route["extrapolation_ratio"] - ratio) < 1e-12 and
                route["history_steps"] == history and row["full_cache_age"] == age and route["selector_calls"] == 0 and
                execution["kv_only_layers"] == mask and execution["patched_layers"] == mask and execution["captured_layers"] == capture and
                execution["qkv_layers"] == [l for l in range(1,19) if l not in mask] and
                execution["hidden_update_layers"] == [l for l in range(1,18) if l not in mask] and
                execution["grad_enabled"] is False and execution["selector_calls"] == 0 and
                len(row["timing"]["nodes"]) - 1 == (10 if full else 2))
            if not valid:
                request_errors.append(dict(case=record["case"], round=index, issue="Execution/schedule/residual/noise identity mismatch"))
            request_counts["full_D10" if full else "fast_D2"] += 1
            per_case["full" if full else "draft"] += 1
            residual_ratios[("full:" if full else "fast:") + str(ratio)] += 1
            costs["native_policy_ms"] += row["timing"]["policy_wall_ms"]
            costs["client_rpc_ms"] += row["rpc_ms"]
            costs["executed_actions"] += row["executed_length"]
            step += row["executed_length"]
        result = read(directory / "result.json")
        if step != result["actions"] or per_case["full"] != result["counts"]["full"] or per_case["draft"] != result["counts"]["draft"]:
            request_errors.append(dict(case=record["case"], issue="Episode action/request accounting"))
        costs["summed_episode_wall_s"] += result["wall_s"]
    require("all_actual_requests_schedule_mask_residual_noise", not request_errors, requests=sum(request_counts.values()), errors=request_errors)
    require("single_frozen_runtime_revision_120", len(runtime_revisions) == 1 and sum(runtime_revisions.values()) == 120, revisions=dict(runtime_revisions))
    require("120_initial_observation_references", len(initial_references) == 120 and len({r["reference"] for r in initial_references}) == 120)
    for key in ("native_policy_ms", "client_rpc_ms"):
        costs[key + "_per_executed_action"] = costs[key] / costs["executed_actions"]
    # Snapshot all condition counts without reading them for scientific selection.
    counts = Counter(condition for condition, _ in audit["records"])
    cross_reference = root / "cross_service_reference.json"
    cross_identity = read(cross_reference); evidence(cross_reference); evidence(root / "cross_service_reference.npz")
    require("cross_service_reference_created_by_this_first_service", cross_identity["manifest_hash"] == manifest["manifest_hash"] and
        len(runs) == 1 and Path(cross_identity["first_service"]).resolve() == runs[0].resolve())
    failed = [c for c in checks if not c["passed"]]
    result = dict(revision="ease-first-real-local-search-runtime-acceptance-v1", started_unix=started, finished_unix=time.time(),
        status="accepted" if not failed else "blocked", condition="jac_anchor", manifest_hash=manifest["manifest_hash"],
        root=str(root), checks=checks, failed_checks=failed, source_identity=source_groups,
        strict_stage_loader=dict(valid_episodes_by_condition=dict(counts), valid_total=len(audit["records"]),
            missing_total=len(audit["missing"]), invalid=audit["invalid"], unexpected=audit["unexpected"],
            discovery_matrix_complete=audit["complete"], selection_called=False),
        coverage=dict(episodes=len(records), suites=manifest["suites"], tasks=list(range(10)), inits=[36,37,38],
            episodes_per_suite=dict(Counter(s for s,t,i in records)), episodes_per_init=dict(Counter(i for s,t,i in records))),
        runtime_services=services, execution=dict(requests=dict(request_counts), residual_ratios=dict(residual_ratios),
            scheduled_NFE_only=True, actual_internal_NFE_counter_exported=False, errors=request_errors),
        initial_identity=dict(first_input_npz_content_and_seed_validated_by_load_stage=True, references=initial_references,
            reference_creator_counts=dict(Counter(r["created_by_condition"] for r in initial_references)),
            independent_between_condition_rollout_pairs_claimed=0,
            interpretation="The first completed Jac condition establishes120 initial observation/noise references. This acceptance does not claim120 cross-condition pairs; other conditions must match them when they execute."),
        cross_service_reference=dict(identity=cross_identity,
            interpretation="The first service creates the reference then compares against it; this first receipt alone is not independent cross-service replication."),
        single_condition_run_cost=costs,
        limits=["Runtime acceptance only; no success/preference/selected-path conclusion from the partial matrix.",
            "Recorded actual layer execution and exported denoising time-node schedule were verified; the backend does not export an internal actual-NFE counter.",
            "Fixed-input cost is one shared-load service probe, not closed-loop acceleration or exclusive-GPU timing.",
            "No scientific output, frozen implementation, manifest, queue, GPU state, or unrelated worker was modified."],
        input_evidence=list(inputs.values()), audit_script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        reproduce="policies/realtime-vla-flash/.venv/bin/python reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/audit.py")
    atomic_json(output, result)
    print(json.dumps(dict(status=result["status"], failed_checks=failed, episodes=len(records), requests=dict(request_counts),
        initial_reference_creators=result["initial_identity"]["reference_creator_counts"], costs=costs,
        report=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", default=str(ROOT / "results/ease_insights_20261103/vlm_local_search_v1/discovery"))
    parser.add_argument("--output", default=str(Path(__file__).with_name("audit.json")))
    args = parser.parse_args()
    main(args.root, args.output)
