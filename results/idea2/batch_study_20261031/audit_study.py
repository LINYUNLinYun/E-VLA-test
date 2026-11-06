"""Inspect actual receipts, actions, lane checks and traces before delivery."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .summarize_stage1 import analyze


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def checked_error(error):
    assert error["finite"] and math.isfinite(error["mae"]) and math.isfinite(error["max_abs"])
    assert error["mae"] <= 1e-6 and error["max_abs"] <= 1e-5


def inspect_run(name, category):
    execution = json.loads((HERE / (name + ".execution.json")).read_text())
    data = json.loads((HERE / name / "result.json").read_text())
    assert execution["exit_code"] == 0 and execution["status"] == "passed", name
    assert execution["source_unchanged"] and execution["source_before"] == execution["source_after"], name
    assert data["source_unchanged"] and data["source_before"] == data["source_after"], name
    assert data["status"] in ("passed", "diagnostic_passed"), name
    lease = json.loads((HERE / name / "gpu_lease.json").read_text())
    assert lease["status"] == "released" and lease["lease_active"] is False, name
    assert data["num_steps"] == data["action_horizon"] == 10
    identity = data["model_identity"]
    assert identity["num_steps"] == 10
    assert all(value in identity["config"] for value in
               ("paligemma_variant='gemma_2b'", "action_expert_variant='gemma_300m'", "pi05=True"))
    assert data["full_kv_slots"] == 2 and data["hidden_third_full_kv"] is False
    left, right = data["kv_addresses"]
    assert len(left) == len(right) == 36 and not set(left).intersection(right)
    batch = data["args"]["batch_size"]
    if data.get("diagnostic_only"):
        assert batch > 1 and data["b1_equivalence_passed"] is False
        assert data["numerical_gates_passed"] is False and data["performance_eligible"] is False
        assert data["same_batch_numerical_gates_passed"] is True
    else:
        assert data["numerical_gates_passed"] is True
    frames = data["input_manifest"]["frames"]
    mapping = {index: seed for frame in frames for index, seed in zip(frame["bank_indices"], frame["seeds"], strict=True)}
    assert len(mapping) == 12 and all(len(set(frame["bank_indices"])) == batch for frame in frames)
    all_rows = data["rows"] + data["verification_rows"]
    mixed_errors, serial_checks = [], 0
    errors_by_kind = {"normalized": [], "decoded": []}
    assert any(row.get("phase") == "same_address_updated_contents" and row.get("addresses_unchanged")
               for row in data["verification_rows"])
    for row in all_rows:
        assert row["full_kv_slots"] == 2 and row["graph_recaptures"] == 0
        assert row["batch_size"] == batch and row["lane_ids"] == frames[0]["lane_ids"]
        assert row["old_kv_age_requests"] == 1
        assert row["num_vlm_layers"] == 18 and len(row["iteration_ms"]) == 10
        assert len(row["frontiers"]) == 10 and all(0 <= value <= 18 for value in row["frontiers"])
        assert row["frontiers"] == sorted(row["frontiers"])
        assert row["seeds"] == frames[row["frame"]]["seeds"]
        assert row["bank_indices"] == frames[row["frame"]]["bank_indices"]
        if row["mode"] == "serial":
            assert row["frontiers"] == [18] * 10 and row["checks"]["passed"]
            prefix = "same_batch" if data.get("diagnostic_only") else "fresh"
            for field in (prefix + "_normalized", prefix + "_decoded"):
                lanes = row["checks"][field]["per_lane"]
                assert len(lanes) == batch
                for lane in lanes:
                    checked_error(lane)
                    serial_checks += 1
        numerical = row.get("numerical", {}).get("matched_schedule")
        if numerical is not None:
            assert numerical["passed"] and len(numerical["per_lane"]) == batch
            assert row["numerical"]["actual_steps"] == 10
            for lane in numerical["per_lane"]:
                for field in ("normalized", "decoded"):
                    checked_error(lane[field])
                    mixed_errors.append(lane[field])
                    errors_by_kind[field].append(lane[field])
    assert mixed_errors and serial_checks
    with np.load(HERE / name / "actions.npz", allow_pickle=False) as archive:
        normalized, decoded = archive["normalized"], archive["decoded"]
        assert normalized.shape == (data["action_archive"]["entries"], batch, 10, 32)
        assert decoded.shape[:3] == normalized.shape[:3] and decoded.shape[-1] == 7
        assert np.isfinite(normalized).all() and np.isfinite(decoded).all()
        assert all(0 <= row["output_index"] < len(normalized) for row in all_rows)
    for mode, summary in data.get("summary", {}).items():
        if mode == "paired_comparison":
            continue
        groups = [row for row in data["groups"] if row["mode"] == mode]
        count = sum(row["count"] for row in groups)
        expected = batch * count * 1000 / math.fsum(row["all_work_ms"] for row in groups)
        assert math.isclose(summary["sample_hz"], expected, rel_tol=1e-10)
        assert summary["requests"] == count and summary["samples"] == batch * count
        assert summary["kv_update_batches_hz"] == summary["request_hz"]
        if category == "performance":
            assert len(groups) >= 100 and count >= 800
    if data["args"].get("disable_lookahead"):
        for row in data["rows"]:
            assert row["lookahead_enabled"] is False and row["next_encoder"] is False
            assert row["lookahead_prepare_ms"] == 0 and row["input_prepare_age_at_admission_ms"] == 0
            assert 0 < row["input_prepare_to_cpu_ms"] <= row["cpu_cpu_ms"] + 1e-5
    return dict(name=name, category=category, batch=batch, status=data["status"], gpu=lease["gpu_uuid"],
                bank_sha256=data["bank_sha256"], sample_seed_map=mapping,
                checkpoint=data["args"]["checkpoint"],
                model_source_sha256={path: digest for path, digest in data["source_before"].items()
                                     if path.startswith("policies/openpi/")},
                same_schedule_max_mae=max(row["mae"] for row in mixed_errors),
                same_schedule_max_abs=max(row["max_abs"] for row in mixed_errors),
                matched_normalized_max_mae=max(row["mae"] for row in errors_by_kind["normalized"]),
                matched_normalized_max_abs=max(row["max_abs"] for row in errors_by_kind["normalized"]),
                matched_decoded_max_mae=max(row["mae"] for row in errors_by_kind["decoded"]),
                matched_decoded_max_abs=max(row["max_abs"] for row in errors_by_kind["decoded"]),
                checked_lane_errors=len(mixed_errors)+serial_checks, archived_outputs=data["action_archive"]["entries"],
                source=str(HERE / name / "result.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    catalog = json.loads((HERE / "study_catalog.json").read_text())
    checks, jobs, skipped = [], [], []
    for section, category in (("performance", "performance"), ("continuity", "continuity"), ("profiles", "profile")):
        for item in catalog.get(section, []):
            name = item["name"]
            receipt = json.loads((HERE / (name + ".execution.json")).read_text())
            if args.completed_only and receipt.get("exit_code") is None:
                skipped.append(name)
                continue
            row = inspect_run(name, category)
            checks.append(row)
            if category == "performance":
                jobs.append(dict(id="performance_"+name, status="passed", phase=item["phase"],
                                 output=str(HERE / name)))
            if category == "profile":
                proof = json.loads((HERE / (name+"_analysis") / "summary.json").read_text())
                assert proof["full_replay_ranges_verified"] == 24 and proof["real_kernels_total"] > 0
                assert all(proof["counts"][stage] == 8 for stage in ("encoder", "vlm", "AE"))
                assert Path(proof["trace"]).is_file() and Path(proof["trace"]).stat().st_size > 0
    assert checks
    assert len({row["bank_sha256"] for row in checks}) == 1
    assert len({row["checkpoint"] for row in checks}) == 1
    assert all(row["model_source_sha256"] and row["model_source_sha256"] == checks[0]["model_source_sha256"]
               for row in checks)
    assert all(row["sample_seed_map"] == checks[0]["sample_seed_map"] for row in checks)
    summary = analyze(dict(jobs=jobs))
    assert summary["base_matrix_complete"]
    if not args.completed_only:
        queue = json.loads((HERE / "stage2_tail_execution.json").read_text())
        assert queue["status"] == "passed" and all(job["status"] == "passed" for job in queue["jobs"])
        assert queue["source_before"] == queue["source_after"]
        boundary = json.loads((HERE / "latency_boundary_execution.json").read_text())
        assert boundary["status"] == "passed" and all(job["status"] == "passed" for job in boundary["jobs"])
        assert boundary["source_before"] == boundary["source_after"]
        fresh = {(row["batch_size"], row["mode"]) for row in summary["rows"]
                 if row["input_mode"] == "online" and not row["lookahead_enabled"]}
        assert fresh == {(batch, mode) for batch in (1, 2, 4) for mode in ("serial", "pipeline", "green")}
        report = ROOT / "docs/experiments/idea2/idea2_batch_pipeline_two_stage_report_20261101.md"
        assert report.is_file() and "状态：**初稿" not in report.read_text()
        final = json.loads((HERE / "final_analysis/summary.json").read_text())
        assert final["base_matrix_complete"] and not final["incomplete"]
        assert (HERE / "final_analysis/table.csv").is_file() and (HERE / "reproduction_commands.md").is_file()
    result = dict(status="completed_subset_verified" if args.completed_only else "passed", completed_only=args.completed_only,
                  inspected_runs=len(checks), skipped=skipped, base_matrix_complete=True, checks=checks,
                  limits="Finite fixed-bank numerical coverage; no cross-B1 equivalence for diagnostic rows or closed-loop quality claim")
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "inspected_runs", "skipped", "base_matrix_complete")}))


if __name__ == "__main__":
    main()
