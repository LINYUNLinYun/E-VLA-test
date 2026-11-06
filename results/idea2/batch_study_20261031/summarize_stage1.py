"""CPU-only analysis of explicit eligible study runs; never infer missing results."""
import argparse
import csv
import json
import math
from pathlib import Path
import random


HERE = Path(__file__).resolve().parent


def pair_comparison(groups, *, draws=2000):
    arms = {}
    for group in groups:
        if group.get("phase") == "paired" and group["mode"] in ("serial", "pipeline"):
            key = group["pair"]
            if group["mode"] in arms.setdefault(key, {}):
                raise ValueError("Duplicate paired timing group")
            arms[key][group["mode"]] = group
    pairs = [row for row in arms.values() if set(row) == {"serial", "pipeline"}]
    if not pairs:
        return None
    if len(pairs) < 100:
        raise ValueError("Incomplete paired performance result")
    if any(row["serial"]["count"] != row["pipeline"]["count"] for row in pairs):
        raise ValueError("Unequal work in paired timing groups")
    left = [row["serial"]["all_work_ms"] for row in pairs]
    right = [row["pipeline"]["all_work_ms"] for row in pairs]
    rng = random.Random(20261031)
    ratios = []
    for _ in range(draws):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        ratios.append(math.fsum(left[i] for i in indices) / math.fsum(right[i] for i in indices))
    ratios.sort()
    return dict(pairs=len(pairs), throughput_speedup=math.fsum(left) / math.fsum(right),
                paired_bootstrap95=[ratios[int(draws * .025)], ratios[int(draws * .975)]],
                scope="Timing-repeat uncertainty on the fixed recorded bank, not task-success uncertainty")


def analyze(execution):
    rows, comparisons, incomplete = [], [], []
    for job in execution.get("jobs", []):
        if not job["id"].startswith("performance_"):
            continue
        path = Path(job["output"]) / "result.json"
        if job["status"] != "passed" or not path.is_file():
            incomplete.append(dict(id=job["id"], status=job["status"]))
            continue
        data = json.loads(path.read_text())
        strict = data.get("status") == "passed" and data.get("performance_eligible") and data.get("numerical_gates_passed")
        diagnostic = (data.get("status") == "diagnostic_passed" and data.get("diagnostic_only")
                      and data.get("scheduling_performance_eligible")
                      and data.get("same_batch_numerical_gates_passed")
                      and data.get("b1_equivalence_passed") is False)
        if (not (strict or diagnostic) or not data.get("source_unchanged")):
            raise ValueError(f"Result is not eligible for a performance claim: {path}")
        gpu = data.get("admission", {}).get("gpu_uuid")
        for mode, summary in data["summary"].items():
            if mode == "paired_comparison":
                continue
            mode_groups = [group for group in data["groups"] if group["mode"] == mode]
            mode_rows = [item for item in data["rows"] if item["mode"] == mode]
            def mean(field):
                value = summary.get(field)
                return value.get("mean") if isinstance(value, dict) else None
            row = dict(job=job["id"], phase=job.get("phase"), batch_size=data["args"]["batch_size"],
                       input_mode=data["args"]["input_mode"], mode=mode, gpu_uuid=gpu,
                       prepare_stream=data["args"].get("prepare_stream", "caller"),
                       lookahead_enabled=not data["args"].get("disable_lookahead", False),
                       eligibility="strict" if strict else "same_batch_diagnostic",
                       b1_batch_equivalence_passed=bool(data.get("b1_equivalence_passed", data.get("numerical_gates_passed"))),
                       diagnostic_only=bool(data.get("diagnostic_only")),
                       same_batch_numerical_gates_passed=bool(data.get("same_batch_numerical_gates_passed",
                                                                      data.get("numerical_gates_passed"))),
                       samples=summary["samples"], request_hz=summary["request_hz"],
                       requests=summary["requests"], timing_groups=len(mode_groups),
                       sample_hz=summary["sample_hz"], cpu_cpu_mean_ms=mean("cpu_cpu_ms"),
                       cpu_cpu_p50_ms=summary["cpu_cpu_ms"]["p50"],
                       cpu_cpu_p95_ms=summary["cpu_cpu_ms"]["p95"],
                       encoder_event_ms=mean("vision_ms"), vlm_event_ms=mean("vlm_ms"),
                       ae_event_ms=mean("ae_ms"), prepare_ms=mean("input_prepare_ms"),
                       lookahead_prepare_ms=mean("lookahead_prepare_ms"),
                       own_input_prepare_to_cpu_ms=mean("input_prepare_to_cpu_ms"),
                       own_input_prepare_age_at_admission_ms=mean("input_prepare_age_at_admission_ms"),
                       ae_submit_ms=mean("ae_submit_ms"),
                       prefill_completion_event_ms=mean("prefill_completion_event_ms"),
                       ae_completion_event_ms=mean("ae_completion_event_ms"),
                       both_completion_event_ms=mean("both_completion_event_ms"),
                       all_submitted_work_completion_event_ms=mean("all_submitted_work_completion_event_ms"),
                       exposed_ae_after_vlm_ms=mean("exposed_ae_after_vlm_ms"),
                       ae_finish_minus_next_encoder_ms=mean("ae_finish_minus_next_encoder_ms"),
                       new_layers_by_ae_step_json=json.dumps(summary.get("mean_new_layers_by_ae_step")),
                       mean_fresh_layer_fraction=(math.fsum(summary["mean_new_layers_by_ae_step"]) /
                                                  (data["num_steps"] * mode_rows[0]["num_vlm_layers"])
                                                  if "mean_new_layers_by_ae_step" in summary else None),
                       prefix_done_at_response_fraction=(sum(bool(item["prefix_done_at_response"]) for item in mode_rows)
                                                         / len(mode_rows)),
                       old_kv_age_requests=mean("old_kv_age_requests"),
                       bootstrap_mean_ms=math.fsum(group["bootstrap_ms"] for group in mode_groups) / len(mode_groups),
                       bank_prepare_setup_mean_ms=math.fsum(group["prepare_bank_setup_ms"] for group in mode_groups) / len(mode_groups),
                       model_load_ms=data["model_load_ms"], graph_setup_ms=data["graph_setup_ms"],
                       setup_peak_allocated_mib=data["setup_peak_allocated_bytes"] / (1024 ** 2),
                       setup_peak_reserved_mib=data["setup_peak_reserved_bytes"] / (1024 ** 2),
                       peak_allocated_mib=data["peak_allocated_bytes"] / (1024 ** 2),
                       peak_reserved_mib=data["peak_reserved_bytes"] / (1024 ** 2),
                       recaptures=summary["steady_graph_recaptures"],
                       requested_ae_sms=data["args"]["ae_sm_min"] if mode == "green" else None,
                       actual_ae_sms=data["resource_layout"].get("actual_ae_sms"),
                       actual_prefix_sms=data["resource_layout"].get("actual_prefix_sms"),
                       source=str(path))
            rows.append(row)
        comparison = pair_comparison(data["groups"])
        if comparison:
            comparisons.append(dict(job=job["id"], batch_size=data["args"]["batch_size"],
                                    input_mode=data["args"]["input_mode"], gpu_uuid=gpu, **comparison))
    index = {(row["batch_size"], row["input_mode"], row["lookahead_enabled"], row["mode"]): row
             for row in rows if row["mode"] == "serial"}
    for row in rows:
        baseline = index.get((row["batch_size"], row["input_mode"], row["lookahead_enabled"], "serial"))
        batch_one = index.get((1, row["input_mode"], row["lookahead_enabled"], "serial"))
        row["speedup_vs_same_batch_serial"] = row["sample_hz"] / baseline["sample_hz"] if baseline else None
        row["speedup_vs_b1_serial"] = row["sample_hz"] / batch_one["sample_hz"] if batch_one else None
        row["same_gpu_as_serial"] = row["gpu_uuid"] == baseline["gpu_uuid"] if baseline else None
    expected = {(batch, input_mode, mode) for batch in (1, 2, 4)
                for input_mode in ("prepared", "online") for mode in ("serial", "pipeline", "green16")}
    present = {(row["batch_size"], row["input_mode"], "green16" if row["mode"] == "green" else row["mode"])
               for row in rows if row["lookahead_enabled"] and (row["mode"] != "green" or row["requested_ae_sms"] == 16)}
    missing = [dict(batch_size=batch, input_mode=input_mode, mode=mode)
               for batch, input_mode, mode in sorted(expected - present)]
    return dict(status="available_results" if rows else "awaiting_performance_results", rows=rows,
                paired_comparisons=comparisons, incomplete=incomplete,
                expected_base_performance_jobs=12,
                scheduled_performance_jobs=sum(job["id"].startswith("performance_") for job in execution.get("jobs", [])),
                complete_performance_jobs=len({r["job"] for r in rows}),
                base_matrix_cells=len(expected), base_matrix_complete=not missing, missing_base_matrix_cells=missing,
                warnings=["Event spans are not busy-kernel intersections.",
                          "Green runs are separate-process comparisons.",
                          "Offline replay does not establish closed-loop quality or camera observation age."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    if args.catalog:
        catalog = json.loads(args.catalog.read_text())
        execution = dict(jobs=[])
        for item in catalog["performance"]:
            receipt = HERE / (item["name"] + ".execution.json")
            value = json.loads(receipt.read_text()) if receipt.is_file() else {}
            execution["jobs"].append(dict(id="performance_" + item["name"], phase=item["phase"],
                status="passed" if value.get("status") == "passed" and value.get("exit_code") == 0 else value.get("status", "pending"),
                output=str(HERE / item["name"])))
    else:
        execution = json.loads((HERE / "stage1_execution.json").read_text())
    result = analyze(execution)
    if args.output is not None and result["rows"]:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        with (args.output / "table.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
