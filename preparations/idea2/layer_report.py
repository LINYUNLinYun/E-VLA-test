"""CPU-only, immutable-snapshot report for the two-step layer pipeline study.

Writes fresh CSV/PNG/plain-text artifacts. Pending, diagnostic, numerical-failed,
and freshness-failed evidence is retained but excluded from recommendations.
This report is a snapshot of evidence, never a declaration that the goal ended.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def stats(values):
    values = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    if not values:
        return dict(n=0, mean=None, p50=None, p95=None)
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)))


def metric(rows, name):
    return stats([row.get(name) for row in rows])


def model_span(row):
    if row.get("model_completion_event_ms") is not None:
        return row["model_completion_event_ms"]
    fields = ("both_completion_event_ms", "prefill_completion_event_ms", "prefill_gpu_event_ms")
    if all(isinstance(row.get(key), (int, float)) for key in fields):
        return row[fields[0]] - row[fields[1]] + row[fields[2]]
    return None


def family(data):
    args = data.get("args", {})
    if "JAX" in str(data.get("backend", "")):
        return "native"
    if "arrival_period_ms" in args and "boundary" in args:
        return "crossframe"
    if "boundary" in args:
        return "layer"
    if "steps" in args and "groups" in data:
        return "legacy"
    return "support"


def source_integrity(path, data):
    hashes = data.get("source_sha256")
    if isinstance(hashes, str):
        source = path.parent / "source.py"
        return source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() == hashes
    if not isinstance(hashes, dict) or not hashes:
        return None
    for name, expected in hashes.items():
        original = Path(name)
        options = (path.parent / "source" / original.name,
                   path.parent / "source" / (original.parent.name + "_" + original.name))
        if not any(candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected
                   for candidate in options):
            return False
    return True


def numerical_evidence(rows, *, allow_digest_only=False):
    errors = [row[key] for row in rows for key in ("normalized_error", "decoded_error", "first_actions_error")
              if isinstance(row.get(key), dict)]
    maxima = [item.get("max_abs") for item in errors]
    maes = [item.get("mae") for item in errors]
    finite = all(item.get("finite", True) is not False
                 and isinstance(item.get("max_abs"), (int, float)) and math.isfinite(item["max_abs"])
                 and isinstance(item.get("mae"), (int, float)) and math.isfinite(item["mae"]) for item in errors)
    complete = bool(rows) and all(row.get("correct") is True and
        ((allow_digest_only and "prefix_digest" in row) or
         all(isinstance(row.get(key), dict) for key in ("normalized_error", "decoded_error"))) for row in rows)
    kv = [row["kv_equal"] for row in rows if "kv_equal" in row]
    return dict(numerical_checks=len(errors), numerical_max_abs=max((value for value in maxima if isinstance(value, (int, float)) and math.isfinite(value)), default=None),
                numerical_max_mae=max((value for value in maes if isinstance(value, (int, float)) and math.isfinite(value)), default=None),
                numerical_complete=complete and finite, kv_digest_checks=len(kv),
                kv_equal=all(kv) if kv else None)


def expected_rows(data, kind):
    args = data.get("args", {})
    if kind == "native":
        return args.get("count")
    if kind == "layer":
        return (0 if args.get("profile") else 2 * args.get("count", 0)) + args.get("continuous_count", 0) * (1 if args.get("profile") else 2)
    if kind == "crossframe":
        modes = [args.get("profile_mode")] if args.get("profile") else args.get("modes", [])
        return len(modes) * (args.get("continuous_count", 0) +
            (0 if args.get("profile") else args.get("pairs", 0) * args.get("pair_length", 16)))
    if kind == "legacy":
        if args.get("kind") == "isolated":
            return 2 * args.get("count", 0) * len(args.get("continuous", []))
        return 2 * args.get("count", 0) * len(args.get("pairs", [])) + args.get("continuous_count", 0) * len(args.get("continuous", []))
    return None


def completion(path, data, job=None):
    kind = family(data)
    status = data.get("status")
    if job and (job.get("state") in ("failed", "process_failed") or job.get("returncode", 0) not in (None, 0)):
        return False, "manifest_process_failed"
    if kind == "support":
        partitions = data.get("partitions", [])
        checked = bool(partitions) and all(
            part.get("eager_disjoint") is True and part.get("observed_counts_match_resource") is True
            and part.get("graph_respects_observed_partition") is True and part.get("primary_tensors_survive_close") is True
            and part.get("external_cross_graph_wait", {}).get("passed") is True for part in partitions)
        return status in ("passed", "complete") and checked, "capability_probe_only" if checked else "support_not_verified"
    if status is not None and status != "passed":
        return False, "explicit_status=" + str(status)
    expected = expected_rows(data, kind)
    if not expected or len(data.get("rows", [])) != expected:
        return False, "missing_or_unexpected_rows"
    args = data.get("args", {})
    if kind in ("layer", "crossframe"):
        smoke = data.get("smoke", [])
        if (data.get("smoke_status") != "passed" or not smoke
                or data.get("external_events", {}).get("passed") is not True
                or any(row.get("kv_equal") is not True or not isinstance(row.get("first_actions_error"), dict) for row in smoke)):
            return False, "first_step_kv_or_event_smoke_incomplete"
        modes = ([args.get("profile_mode", "overlap")] if args.get("profile") else
                 (args.get("modes", []) if kind == "crossframe" else ["serial", "overlap"]))
        wanted = {( "continuous", mode): args.get("continuous_count", 0) for mode in modes}
        if not args.get("profile"):
            paired = args.get("count", 0) if kind == "layer" else args.get("pairs", 0) * args.get("pair_length", 16)
            if paired:
                wanted.update({("paired", mode): paired for mode in modes})
        if Counter((row.get("phase"), row.get("mode")) for row in data["rows"]) != wanted:
            return False, "arm_or_phase_sample_counts_incomplete"
    if source_integrity(path, data) is not True:
        return False, "source_snapshot_not_verified"
    if kind == "native":
        if data.get("repeatability", {}).get("passed") is not True or not (path.parent / "actions.npy").is_file():
            return False, "native_final_artifacts_missing"
        actions = np.load(path.parent / "actions.npy", mmap_mode="r", allow_pickle=False)
        if actions.shape[0] != expected or not np.isfinite(actions).all():
            return False, "native_actions_invalid"
    else:
        evidence = numerical_evidence(data.get("rows", []) + data.get("smoke", []), allow_digest_only=kind == "legacy")
        if not evidence["numerical_complete"] or evidence["kv_equal"] is False:
            return False, "numerical_or_kv_checks_incomplete_or_failed"
        tolerance = data.get("tolerance", {})
        if (evidence["numerical_max_abs"] is not None and evidence["numerical_max_abs"] > tolerance.get("max_abs", 1e-5)) or (
                evidence["numerical_max_mae"] is not None and evidence["numerical_max_mae"] > tolerance.get("mae", 1e-6)):
            return False, "numerical_tolerance_failed"
        groups = data.get("groups", [])
        if not groups or any(group.get("correct") is not True for group in groups):
            return False, "group_validation_incomplete"
        if kind == "legacy":
            args = data["args"]
            wanted = len(args.get("continuous", [])) if args.get("kind") == "isolated" else len(args.get("pairs", [])) + len(args.get("continuous", []))
            if len(groups) != wanted:
                return False, "legacy_groups_incomplete"
    if status == "passed":
        return True, "passed_status_and_complete_verified_artifacts"
    if job and job.get("state") in ("running", "process_running", "started"):
        return False, "manifest_process_still_running"
    if job and job.get("state") == "process_succeeded" and job.get("returncode", 0) == 0:
        return True, "process_succeeded_and_complete_verified_artifacts"
    return True, "complete_final_artifacts_and_source_snapshot_no_process_receipt"


def ranking_reasons(data, complete, rows, *, paired_count=None):
    args = data.get("args", {})
    reasons = []
    if not complete:
        reasons.append("case_not_proven_complete")
    if args.get("profile") or data.get("profile_run"):
        reasons.append("profile_only")
    if args.get("iteration_timing") or data.get("iteration_timing_diagnostic"):
        reasons.append("iteration_timing_diagnostic")
    if len(rows) < 100:
        reasons.append("fewer_than_100_samples")
    if paired_count is not None and paired_count < 100:
        reasons.append("fewer_than_100_matched_pairs")
    if args.get("warmup", 0) < 10:
        reasons.append("fewer_than_10_warmups")
    if args.get("smoke"):
        reasons.append("smoke_only")
    if any(isinstance(row, dict) and (row.get("graph_recaptures", 0) or row.get("steady_recapture", 0)) for row in rows):
        reasons.append("steady_graph_recapture")
    return reasons


def base_row(path, data, root, complete, proof):
    args = data.get("args", {})
    green = data.get("green_context") or {}
    # Green partitions need not correspond to AE then Prefix: a Prefix-only
    # cap leaves an unused remainder while AE runs in the primary context.
    branch_sms = dict(zip(green.get("branch_order") or [], green.get("sm_counts") or []))
    environment = data.get("environment") or {}
    evidence = numerical_evidence(data.get("rows", []) + data.get("smoke", []), allow_digest_only=family(data) == "legacy")
    return dict(case=str(path.parent.relative_to(root)), source_file=str(path), family=family(data),
                complete=complete, completion_evidence=proof,
                gpu_uuid=data.get("actual_cuda_uuid", data.get("gpu_uuid", args.get("gpu_uuid"))),
                steps=data.get("steps", data.get("num_steps", args.get("steps"))),
                bank_sha256=data.get("bank_sha256"), checkpoint=args.get("checkpoint"),
                noise_period=args.get("noise_period", len(data.get("selected_bank_indices", [])) if family(data) in ("layer", "crossframe") else None),
                cuda_device_max_connections=environment.get("CUDA_DEVICE_MAX_CONNECTIONS"),
                boundary=args.get("boundary"), group_size=args.get("group_size"), ae_layout=args.get("ae_layout"),
                ae_priority=args.get("ae_priority"), requested_ae_sm_min=args.get("ae_sm_min"),
                requested_prefix_sm_cap=args.get("prefix_sm_cap"),
                ae_resource_policy=green.get("ae_resource_policy", data.get("ae_resource_policy")),
                device_sm_count=green.get("device_sm_count", data.get("device_sm_count")),
                actual_ae_sms=branch_sms.get("ae"),
                actual_prefix_sms=green.get("prefix_actual_sm_count", data.get("prefix_actual_sm_count", branch_sms.get("prefix"))),
                hoist_modulation=args.get("hoist_modulation"), reset_every=args.get("reset_every"),
                numerical_scope="Maximum explicit numerical errors across all available case smoke and measured rows; completion is a separate gate",
                arrival_period_ms=args.get("arrival_period_ms"), lookahead_window_ms=args.get("lookahead_window_ms"),
                requested_lookahead_delay_ms=args.get("lookahead_delay_ms"),
                **evidence)


def add_metric_columns(target, rows, names):
    for name in names:
        measured = stats([model_span(row) for row in rows]) if name == "model_completion_event_ms" else metric(rows, name)
        for key in ("n", "mean", "p50", "p95"):
            target[name + "_" + key] = measured[key]


def matching_group(data, phase, mode):
    return next((group for group in data.get("groups", []) if group.get("phase") == phase
                 and (group.get("mode") == mode or mode in group.get("modes", []))), {})


def add_submission_metrics(target, rows):
    add_metric_columns(target, rows, (
        "prefill_completion_event_ms", "ae_completion_event_ms", "both_completion_event_ms",
        "prefill_gpu_event_ms", "prefill_cpu_submit_ms", "ae_cpu_submit_ms",
        "encode_cpu_submit_ms", "vlm_cpu_submit_ms", "ae_host_wait_ms", "slot_wait_ms",
        "ae_wait_after_lookahead_ms", "acquire_current_host_ms", "noise_cpu_ms", "output_cpu_ms"))
    recaptures = [row.get("graph_recaptures") for row in rows
                  if isinstance(row.get("graph_recaptures"), (int, float))
                  and math.isfinite(row["graph_recaptures"])]
    target.update(graph_recaptures_n=len(recaptures),
                  graph_recaptures_max=max(recaptures) if recaptures else None)


def add_boolean_count(target, rows, name):
    values = [row[name] for row in rows if isinstance(row.get(name), bool)]
    target[name + "_n"] = len(values)
    target[name + "_count"] = sum(values) if values else None


def collect_layer(base, data, tables):
    for phase, mode in sorted({(row["phase"], row["mode"]) for row in data["rows"]}):
        rows = [row for row in data["rows"] if (row["phase"], row["mode"]) == (phase, mode)]
        row = dict(base, phase=phase, mode=mode, samples=len(rows), kv_semantics="same_frame_fresh",
                   latency_scope="CPU input preparation through response; continuous includes startup/reset effects",
                   event_scope="CUDA spans include gaps/waits/contention; model span excludes initial input preparation, not kernel busy time")
        add_metric_columns(row, rows, ("cpu_cpu_ms", "model_completion_event_ms", "prefill_gpu_event_ms",
            "vision_embedding_event_ms", "vlm_event_ms", "ae_gpu_event_ms", "prepare_cpu_ms",
            "prefill_cpu_submit_ms", "ae_cpu_submit_ms", "ae_iteration1_event_ms", "ae_iteration2_event_ms"))
        add_submission_metrics(row, rows)
        group = matching_group(data, phase, mode)
        row.update(peak_allocated_bytes=group.get("peak_allocated_bytes"), peak_reserved_bytes=group.get("peak_reserved_bytes"),
                   all_work_hz=group.get("all_work_hz"))
        reasons = ranking_reasons(data, base["complete"], rows)
        row.update(ranking_eligible=not reasons, exclusion_reasons=";".join(reasons))
        tables["layer"].append(row)
    pairs = {}
    for row in data["rows"]:
        if row.get("phase") == "paired":
            pairs.setdefault(row.get("pair"), {})[row["mode"]] = row
    for name in ("cpu_cpu_ms", "model_completion_event_ms"):
        ratios = []
        for arms in pairs.values():
            if set(arms) != {"serial", "overlap"}:
                continue
            values = [model_span(arms[mode]) if name == "model_completion_event_ms" else arms[mode].get(name)
                      for mode in ("serial", "overlap")]
            if all(isinstance(value, (int, float)) and value > 0 for value in values):
                ratios.append(values[0] / values[1])
        reasons = ranking_reasons(data, base["complete"], list(pairs), paired_count=len(ratios))
        tables["paired_ratios"].append(dict(base, baseline="serial", candidate="overlap", metric=name,
            ratio_definition="mean/quantiles of per-pair serial divided by overlap; same resource restrictions", **stats(ratios),
            ranking_eligible=not reasons, exclusion_reasons=";".join(reasons)))


def collect_crossframe(base, data, tables):
    for group in data.get("groups", []):
        if group.get("phase") != "continuous":
            continue
        mode = group["mode"]
        rows = [row for row in data["rows"] if row.get("phase") == "continuous" and row["mode"] == mode]
        counts = group.get("arrival_counts", {})
        row = dict(base, phase="continuous", mode=mode, samples=len(rows), kv_semantics="same_frame_fresh",
            all_work_hz=group.get("all_work_hz"), response_hz=group.get("response_hz"),
            whole_trajectory_ms=group.get("all_work_trajectory_ms"),
            trajectory_scope="Includes fill/drain and configured resets; camera replay is not closed-loop robotics",
            event_scope="CUDA spans include waits/gaps/contention, not kernel busy time. Each frame has its own origin before its input preparation, including lookahead-admitted frames; completion spans are relative to that origin, not the previous action or camera capture. Model span starts before Encoder staging.",
            dropped=counts.get("dropped"), received=counts.get("received"),
            drop_fraction=counts["dropped"] / counts["received"] if counts.get("received") else None,
            buffered_unprocessed=counts.get("buffered_unprocessed"),
            completed_prefills=group.get("completed_prefills"), consumed_snapshots=group.get("consumed_snapshots"),
            completed_prefill_hz=group.get("completed_prefill_hz"),
            peak_allocated_bytes=group.get("peak_allocated_bytes"), peak_reserved_bytes=group.get("peak_reserved_bytes"),
            host_lookahead=mode != "fresh_serial")
        add_metric_columns(row, rows, ("admission_to_action_ms", "capture_to_action_ms", "cpu_cpu_ms",
            "camera_age_frames_at_output", "model_completion_event_ms", "prepare_cpu_ms", "lookahead_host_ms",
            "lookahead_poll_ms", "lookahead_delay_ms", "vision_embedding_event_ms", "vlm_event_ms", "ae_gpu_event_ms",
            "kv_conditioning_offset_frames"))
        add_submission_metrics(row, rows)
        for name in ("lookahead_admitted", "ae_done_after_lookahead"):
            add_boolean_count(row, rows, name)
        reasons = ranking_reasons(data, base["complete"], rows)
        row.update(ranking_eligible=not reasons, exclusion_reasons=";".join(reasons))
        tables["arrival" if (data["args"].get("arrival_period_ms") or 0) > 0 else "capacity"].append(row)
    paired = data.get("paired_summary", {}).get("comparisons", {})
    for baseline, candidates in paired.items():
        for candidate, metrics in candidates.items():
            for name, measured in metrics.items():
                if not isinstance(measured, dict):
                    continue
                reasons = ranking_reasons(data, base["complete"], data["rows"], paired_count=measured.get("n", 0))
                tables["paired_ratios"].append(dict(base, baseline=baseline, candidate=candidate, metric=name,
                    ratio_definition="matched whole-capacity trajectories; full includes fill/drain, interior excludes first two and last one frames",
                    **measured, ranking_eligible=not reasons, exclusion_reasons=";".join(reasons)))
    for mode in data["args"].get("modes", []):
        groups = [group for group in data.get("groups", []) if group.get("phase") == "paired" and group.get("mode") == mode]
        if not groups:
            continue
        row = dict(base, mode=mode, phase="paired_trajectories", trajectories=len(groups),
                   frames_per_trajectory=data["args"].get("pair_length"),
                   scope="Full trajectory includes fill/drain; trimmed interior is not established steady state")
        for name in ("all_work_trajectory_ms", "all_work_hz"):
            for key, value in stats([group.get(name) for group in groups]).items():
                row[name + "_" + key] = value
        for name, access in (("interior_interval_mean_ms", lambda item: item.get("output_interval_ms", {}).get("mean")),
                             ("interior_admission_p95_ms", lambda item: (item.get("latency", {}).get("admission_to_action_ms") or {}).get("p95"))):
            for key, value in stats([access(group.get("trimmed_interior") or {}) for group in groups]).items():
                row[name + "_" + key] = value
        reasons = ranking_reasons(data, base["complete"], groups, paired_count=len(groups))
        row.update(ranking_eligible=not reasons, exclusion_reasons=";".join(reasons))
        tables["capacity_trajectories"].append(row)


def collect_legacy(base, data, tables):
    for phase, mode in sorted({(row["phase"], row["mode"]) for row in data["rows"] if row.get("phase") != "paired"}):
        rows = [row for row in data["rows"] if (row["phase"], row["mode"]) == (phase, mode)]
        fresh = "fresh" in mode
        summary = data.get("summary", {}).get(phase + "/" + mode, {})
        ages = [row.get("observation_age") for row in rows]
        age_max = max((value for value in ages if isinstance(value, (int, float))), default=None)
        submitted = sum(row.get("prefill_gpu_event_ms") is not None for row in rows)
        freshness = (age_max is not None and age_max <= (0 if fresh else 1) and submitted == len(rows)) if phase == "continuous" else None
        row = dict(base, phase=phase, mode=mode, samples=len(rows), kv_semantics="same_frame_fresh" if fresh else "complete_old_KV",
                   freshness_ok=freshness, prefill_submissions=submitted,
                   prefill_fraction=submitted / len(rows), observation_age_max=age_max,
                   age_definition="KV conditioning observation offset, not camera acquisition age",
                   action_semantics="Different KV observations are not numerically interchangeable",
                   latency_scope="CPU response; legacy serial_reuse may finish background Prefill before returning action",
                   peak_allocated_bytes=summary.get("peak_allocated"), peak_reserved_bytes=summary.get("peak_reserved"),
                   prefill_hz=summary.get("prefill_submissions_per_second"))
        add_metric_columns(row, rows, ("cpu_cpu_ms", "ae_output_ms", "ae_gpu_event_ms", "prefill_gpu_event_ms", "source_age_at_response_ms", "observation_age"))
        reasons = ranking_reasons(data, base["complete"], rows)
        if phase == "continuous" and not freshness:
            reasons.append("freshness_or_prefill_coverage_failed")
        row.update(ranking_eligible=not reasons, exclusion_reasons=";".join(reasons))
        tables["legacy"].append(row)
    for label, measured in data.get("paired_speedups", {}).items():
        if measured is None:
            continue
        arms = label.split(":", 1)[-1].split("/", 1)
        reasons = ranking_reasons(data, base["complete"], data["rows"], paired_count=measured.get("n", 0))
        continuous_candidate = next((row for row in tables["legacy"] if row["case"] == base["case"]
                                     and row["mode"] == arms[-1] and row["phase"] == "continuous"), None)
        if continuous_candidate is None or continuous_candidate.get("freshness_ok") is not True:
            reasons.append("continuous_freshness_not_proven")
        tables["paired_ratios"].append(dict(base, baseline=arms[0], candidate=arms[-1], metric="cpu_cpu_ms",
            ratio_definition="Legacy same-old-KV bootstrapped paired request ratio; does not prove continuous freshness",
            **measured, ranking_eligible=not reasons, exclusion_reasons=";".join(reasons)))


def collect_native(base, data, tables):
    row = dict(base, mode="native_jax", samples=len(data["rows"]), kv_semantics="same_frame_fresh",
               numerical_max_abs=None, numerical_max_mae=None, numerical_complete=None,
               repeatability_max_abs=data.get("repeatability", {}).get("max_abs"),
               numerical_scope="Native repeatability only; no cross-framework exact-output equivalence claimed",
               latency_scope=data.get("boundary"), sampled_peak_device_used_mib=data.get("sampled_peak_device_used_mib"),
               memory_scope=data.get("memory_definition"))
    add_metric_columns(row, data["rows"], ("cpu_cpu_ms",))
    reasons = ranking_reasons(data, base["complete"], data["rows"])
    row.update(ranking_eligible=not reasons, exclusion_reasons=";".join(reasons))
    tables["native"].append(row)


def add_slo(tables):
    for candidate in tables["arrival"]:
        same_case = [row for row in tables["arrival"] if row["case"] == candidate["case"]]
        for baseline in ("fresh_serial", "lookahead_serial"):
            reference = next((row for row in same_case if row["mode"] == baseline), None)
            if reference is None:
                continue
            p95, base95 = candidate["capture_to_action_ms_p95"], reference["capture_to_action_ms_p95"]
            for budget in (0, 0.02, 0.05, 0.1):
                tables["slo_budgets"].append(dict(case=candidate["case"], mode=candidate["mode"], baseline=baseline,
                    arrival_period_ms=candidate["arrival_period_ms"], lookahead_window_ms=candidate["lookahead_window_ms"],
                    requested_lookahead_delay_ms=candidate.get("requested_lookahead_delay_ms"),
                    capture_p95_ms=p95, baseline_capture_p95_ms=base95, allowed_p95_increase=budget,
                    meets_budget=p95 <= base95 * (1 + budget) if p95 is not None and base95 is not None else None,
                    all_work_hz=candidate["all_work_hz"], drop_fraction=candidate["drop_fraction"],
                    ranking_eligible=candidate["ranking_eligible"] and reference["ranking_eligible"],
                    source_file=candidate["source_file"]))
    for candidate in tables["arrival"]:
        candidate["pareto_latency_throughput_drop"] = None
        if not candidate["ranking_eligible"] or any(candidate[key] is None for key in ("capture_to_action_ms_p95", "all_work_hz", "drop_fraction")):
            continue
        comparable = [row for row in tables["arrival"] if row["ranking_eligible"] and
            all(row[key] == candidate[key] for key in ("gpu_uuid", "arrival_period_ms", "bank_sha256", "checkpoint")) and
            all(row[key] is not None for key in ("capture_to_action_ms_p95", "all_work_hz", "drop_fraction"))]
        vector = lambda row: (row["capture_to_action_ms_p95"], -row["all_work_hz"], row["drop_fraction"])
        target = vector(candidate)
        candidate["pareto_latency_throughput_drop"] = not any(all(a <= b for a, b in zip(vector(row), target)) and
            any(a < b for a, b in zip(vector(row), target)) for row in comparable)


def capacity_comparisons(tables):
    for candidate in tables["capacity"]:
        for baseline in ("fresh_serial", "lookahead_serial"):
            if candidate["mode"] == baseline:
                continue
            reference = next((row for row in tables["capacity"] if row["case"] == candidate["case"] and row["mode"] == baseline), None)
            if reference is None:
                continue
            tables["capacity_comparisons"].append(dict(case=candidate["case"], baseline=baseline, candidate=candidate["mode"],
                all_work_hz_ratio=candidate["all_work_hz"] / reference["all_work_hz"] if candidate["all_work_hz"] and reference["all_work_hz"] else None,
                admission_p95_delta_ms=candidate["admission_to_action_ms_p95"] - reference["admission_to_action_ms_p95"],
                ratio_definition="Ratio of separate continuous whole-trajectory throughput, NOT a matched paired speedup",
                scope="fresh_serial to lookahead_serial measures input-preparation lookahead; lookahead_serial to overlap arms isolates additional GPU scheduling changes",
                ranking_eligible=candidate["ranking_eligible"] and reference["ranking_eligible"], source_file=candidate["source_file"]))


def aggregate(root):
    root = Path(root).resolve()
    tables = {name: [] for name in ("layer", "capacity", "capacity_comparisons", "capacity_trajectories", "arrival", "slo_budgets", "legacy", "native", "paired_ratios")}
    inventory = []
    manifest_path = root / "baseline_commands.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    jobs = {job["name"]: job for job in manifest.get("jobs", [])}
    seen = set()
    for path in sorted(root.rglob("result.json")):
        try:
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
                raise ValueError("Result changed while being read")
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("Result is not an object")
            seen.add(path.parent.name)
            complete, proof = completion(path, data, jobs.get(path.parent.name))
            base = base_row(path, data, root, complete, proof)
            inventory.append(dict(case=base["case"], source_file=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                                  family=base["family"], status=data.get("status"), complete=complete, completion_evidence=proof,
                                  observed_rows=len(data.get("rows", [])), expected_rows=expected_rows(data, base["family"]),
                                  source_integrity=source_integrity(path, data)))
            if base["family"] == "layer":
                collect_layer(base, data, tables)
            elif base["family"] == "crossframe":
                collect_crossframe(base, data, tables)
            elif base["family"] == "legacy":
                collect_legacy(base, data, tables)
            elif base["family"] == "native":
                collect_native(base, data, tables)
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            inventory.append(dict(case=str(path.parent.relative_to(root)), source_file=str(path), complete=False,
                                  completion_evidence="unreadable_or_incomplete_schema: " + str(error)))
    for name, job in jobs.items():
        if name not in seen:
            inventory.append(dict(case=name, source_file=None, complete=False,
                                  completion_evidence="no_result_file", status=job.get("state")))
    add_slo(tables)
    capacity_comparisons(tables)
    return dict(input_root=str(root), generated_unix=time.time(), inventory=inventory, tables=tables,
                scope="Evidence snapshot; active/pending experiments remain incomplete. No goal completion is inferred.")


def csv_table(path, rows):
    with path.open("w", newline="") as handle:
        keys = sorted(set().union(*(row.keys() for row in rows))) if rows else ["no_rows"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def figures(dest, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    formal = [row for row in tables["layer"] if row["ranking_eligible"] and row["phase"] == "continuous"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    cases = sorted({row["case"] for row in formal})
    for axis, metric_name, title in zip(axes, ("cpu_cpu_ms_p50", "model_completion_event_ms_p50"),
                                       ("CPU response P50", "Model Event span P50 (not kernel busy)")):
        for offset, mode in ((-0.18, "serial"), (0.18, "overlap")):
            values = [next((row[metric_name] for row in formal if row["case"] == case and row["mode"] == mode), np.nan) for case in cases]
            axis.bar(np.arange(len(cases)) + offset, values, width=0.35, label=mode)
        axis.set_xticks(np.arange(len(cases)), cases, rotation=55, ha="right", fontsize=8)
        axis.set_ylabel("ms"); axis.set_title(title); axis.legend()
    fig.tight_layout(); fig.savefig(dest / "layer_comparison.png", dpi=160); plt.close(fig)
    capacity = [row for row in tables["capacity"] if row["ranking_eligible"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    labels = [row["case"] + "/" + row["mode"] for row in capacity]
    for axis, name, title, unit in ((axes[0], "all_work_hz", "Capacity including fill/drain/reset", "outputs/s"),
                                  (axes[1], "admission_to_action_ms_p95", "Admission latency P95 (not camera age)", "ms")):
        axis.bar(np.arange(len(capacity)), [row[name] for row in capacity])
        axis.set_xticks(np.arange(len(capacity)), labels, rotation=55, ha="right", fontsize=8)
        axis.set_title(title); axis.set_ylabel(unit)
    fig.tight_layout(); fig.savefig(dest / "capacity_comparison.png", dpi=160); plt.close(fig)
    arrival = [row for row in tables["arrival"] if row["ranking_eligible"]]
    periods = sorted({row["arrival_period_ms"] for row in arrival})
    fig, axes = plt.subplots(1, max(1, len(periods)), figsize=(7 * max(1, len(periods)), 5), squeeze=False)
    if not periods:
        axes[0, 0].text(0.5, 0.5, "No completed formal camera-replay results yet", ha="center", va="center")
    for axis, period in zip(axes.flat, periods):
        selected = [row for row in arrival if row["arrival_period_ms"] == period]
        points = axis.scatter([row["capture_to_action_ms_p95"] for row in selected], [row["all_work_hz"] for row in selected],
                              c=[row["drop_fraction"] for row in selected], vmin=0, vmax=1, cmap="viridis")
        for row in selected:
            axis.annotate(row["case"] + "/" + row["mode"], (row["capture_to_action_ms_p95"], row["all_work_hz"]), fontsize=7)
        axis.set(xlabel="Capture-to-action P95 (ms)", ylabel="outputs/s", title=f"Camera replay: {period:g} ms arrivals")
        fig.colorbar(points, ax=axis, label="Dropped fraction")
    fig.tight_layout(); fig.savefig(dest / "arrival_pareto.png", dpi=160); plt.close(fig)


def plain_report(report):
    tables = report["tables"]
    lines = ["Two-step pi0.5 layer pipeline evidence snapshot", report["scope"], "",
             "Definitions: CUDA Event spans include waits, dispatch gaps and resource contention; they are not kernel busy time.",
             "model_completion = both_completion - prefill_completion + prefill_gpu_event, per request; removes initial input preparation gap.",
             "Continuous capacity includes startup/fill/drain/reset. Paired short trajectories include fill/drain; trimmed interiors are not proven pure steady state.",
             "Admission latency is not camera capture latency. Camera replay is not closed-loop robot validation; throughput and drops must be considered together.",
             "Old-KV action latency is a separate semantic category. Missing metrics/environment variables are unknown (blank CSV / null JSON), never assumed zero or default 8 connections.",
             "SM counts follow named Green Context branches. Unrestricted primary-context AE has no recorded partition quota; device_sm_count describes the GPU, not an exclusive AE allocation. Unused partition SMs are not assigned to AE.",
             "Paired ratios are computed per matched pair or trajectory; two aggregate medians are not a paired speedup.", ""]
    lines.append("Pareto flags describe individual observed runs, not repeat-validated SLO guarantees; consult slo_repeat_analysis and delay_repeat_analysis for independent rounds.")
    complete = [row for row in report["inventory"] if row.get("complete") and row.get("family") != "support"]
    pending = [row for row in report["inventory"] if not row.get("complete")]
    lines.append(f"Completed verified model-result files: {len(complete)}; pending/unverified/failed/missing: {len(pending)}. Capability probes are not model benchmarks.")
    formal_a = [row for row in tables["paired_ratios"] if row["family"] == "layer" and row["ranking_eligible"] and row["metric"] == "model_completion_event_ms"]
    for row in formal_a:
        lines.append(f"A {row['case']}: matched model-span speedup P50 {row['p50']:.4f}x over its own same-resource serial arm ({row['n']} pairs). Source: {row['source_file']}")
    for row in tables["legacy"]:
        if row["complete"] and row.get("freshness_ok") is False and set(row["exclusion_reasons"].split(";")) == {"freshness_or_prefill_coverage_failed"}:
            lines.append(f"Old-KV exclusion {row['case']}/{row['mode']}: Prefill {row['prefill_submissions']}/{row['samples']}, mean KV offset {row['observation_age_mean']}; low action latency does not meet freshness requirements.")
    formal_b = [row for row in tables["capacity"] if row["ranking_eligible"]]
    for row in formal_b:
        lines.append(f"Capacity {row['case']}/{row['mode']}: {row['all_work_hz']:.3f} outputs/s; admission P95 {row['admission_to_action_ms_p95']:.3f} ms (no camera-age claim).")
    for row in tables["paired_ratios"]:
        if row["family"] == "crossframe" and row["ranking_eligible"] and row["metric"] == "full_trajectory_speedup" and (
                (row["baseline"], row["candidate"]) in (("fresh_serial", "lookahead_serial"),
                                                         ("lookahead_serial", "encoder_overlap"), ("lookahead_serial", "combined"))):
            lines.append(f"Matched capacity trajectories {row['case']}: {row['baseline']} -> {row['candidate']} P50 speedup {row['p50']:.4f}x ({row['n']} pairs, including fill/drain).")
    if not any(row["ranking_eligible"] for row in tables["arrival"]):
        lines.append("No completed formal camera-replay SLO result is available in this snapshot; no SLO-optimal configuration is selected.")
    lines.extend(["", "Pending / excluded evidence:"])
    lines.extend(f"{row['case']}: {row.get('completion_evidence')}" for row in pending)
    lines.extend(["", "Tables: layer.csv, capacity.csv, capacity_comparisons.csv, capacity_trajectories.csv, arrival.csv, slo_budgets.csv, legacy.csv, native.csv, paired_ratios.csv, excluded.csv.",
                  "All diagnostic/profile/short-sample rows remain in tables with ranking_eligible=False and explicit exclusion reasons.",
                  "Re-run into a new output directory to discover later results. This report does not mark the study or goal complete."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/idea2/layer_pipeline_20261023")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    report = aggregate(args.input)
    report["reporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, rows in report["tables"].items():
        csv_table(dest / (name + ".csv"), rows)
    csv_table(dest / "excluded.csv", [dict(table=name, **row) for name, rows in report["tables"].items()
                                      for row in rows if row.get("ranking_eligible") is False])
    csv_table(dest / "inventory.csv", report["inventory"])
    (dest / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (dest / "report.txt").write_text(plain_report(report))
    figures(dest, report["tables"])
    print(json.dumps(dict(output=str(dest), result_files=len(report["inventory"]),
                          table_rows={name: len(rows) for name, rows in report["tables"].items()})))


if __name__ == "__main__":
    main()
