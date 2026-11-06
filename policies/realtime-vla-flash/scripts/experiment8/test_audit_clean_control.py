"""CPU-only tests for the clean D3/D10 control auditor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.experiment8.audit_clean_control import (
    AuditError,
    audit_pair,
    audit_run,
    expected_noise_sha256,
    main,
    mcnemar_exact_two_sided,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path, *, count: int, denoise_steps: int, successes: set[int], name: str) -> tuple[Path, Path]:
    testset_path = tmp_path / "testset.json"
    states = tmp_path / "states"
    samples = []
    for index in range(count):
        state_path = states / f"state{index:03d}.npy"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = np.asarray([index, index + 0.5], dtype=np.float64)
        np.save(state_path, state)
        samples.append(
            {
                "sample_id": f"libero_90_task{index:03d}",
                "suite_name": "libero_90",
                "task_id": index,
                "init_state_path": str(state_path.relative_to(tmp_path)),
                "init_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            }
        )
    _write_json(testset_path, {"samples": samples})
    testset_hash = hashlib.sha256(testset_path.read_bytes()).hexdigest()

    run = tmp_path / name
    declared = [
        {"sample_index": index, "sample_id": row["sample_id"], "suite_name": row["suite_name"], "task_id": index}
        for index, row in enumerate(samples)
    ]
    _write_json(
        run / "manifest.json",
        {
            "run_id": name,
            "status": "completed" if count == 100 else "running",
            "testset_sha256": testset_hash,
            "sample_count": count,
            "samples": declared,
            "condition_id": f"Control-clean-D{denoise_steps}",
            "vlm_layer_pattern": "vlm_full18",
            "action_layer_pattern": "full18",
            "denoise_steps": denoise_steps,
            "global_noise_seed": 12345,
            "seed": 7,
            "completed_sample_count": count if count == 100 else None,
        },
    )
    episodes = []
    for index, sample in enumerate(samples):
        episode_dir = run / "episodes" / f"sample{index:03d}"
        infer_relative = episode_dir.relative_to(run) / "infer.jsonl"
        trace_relative = episode_dir.relative_to(run) / "trace.jsonl"
        action = [0.01 * (index + axis) for axis in range(7)]
        noise_hash = expected_noise_sha256(12345, index, 0)
        infer = {
            "sample_index": index,
            "sample_id": sample["sample_id"],
            "task_suite_name": sample["suite_name"],
            "task_id": index,
            "episode_idx": index,
            "infer_id": 0,
            "route_type": "full",
            "chunk_exec_len": 1,
            "chunk_actions": [action],
            "server_experiment": {
                "vlm_layer_pattern": "vlm_full18",
                "action_layer_pattern": "full18",
                "vlm_layer_indices_1based": list(range(1, 19)),
                "action_layer_indices_1based": list(range(1, 19)),
                "denoise_steps": denoise_steps,
                "ablation_noise_seed": 12345,
                "noise_sha256": noise_hash,
                "noise_seed_mode": "episode_infer",
                "noise_seed_rule": "SeedSequence(global_noise_seed, episode_idx, replan_idx)",
                "trace_episode_idx": index,
                "trace_infer_id": 0,
            },
        }
        trace = {
            "sample_index": index,
            "sample_id": sample["sample_id"],
            "task_suite_name": sample["suite_name"],
            "task_id": index,
            "episode_idx": index,
            "infer_id": 0,
            "action_offset_in_chunk": 0,
            "executed_action": action,
            "measured_eef_pos": [0.1, 0.2, 0.3],
            "measured_eef_quat": [1.0, 0.0, 0.0, 0.0],
            "measured_gripper_qpos": [0.04, -0.04],
            "measured_state_when": "post_action",
        }
        _write_jsonl(run / infer_relative, [infer])
        _write_jsonl(run / trace_relative, [trace])
        episodes.append(
            {
                "sample_index": index,
                "sample_id": sample["sample_id"],
                "suite_name": sample["suite_name"],
                "task_id": index,
                "episode_idx": index,
                "noise_episode_idx": index,
                "init_state_sha256": sample["init_state_sha256"],
                "success": index in successes,
                "failure_reason": None,
                "infer_calls": 1,
                "infer_record_count": 1,
                "trace_record_count": 1,
                "executed_action_count": 1,
                "trajectory_length": 1,
                "infer_path": str(infer_relative),
                "trace_path": str(trace_relative),
            }
        )
    _write_json(run / "episode_log.json", episodes)
    return run, testset_path


def test_single_incremental_run_verifies_noise_actions_and_telemetry(tmp_path: Path) -> None:
    run, testset = _fixture(tmp_path, count=1, denoise_steps=3, successes=set(), name="clean_d3")
    report = audit_run(run, testset, expected_count=1, expected_denoise_steps=3)
    assert report["status"] == "passed"
    assert report["strict_final_100"] is False
    assert report["inference_count"] == 1
    assert report["executed_action_count"] == 1
    assert report["episodes"][0]["noise_sha256_by_infer"]["0"] == expected_noise_sha256(12345, 0, 0)


def test_failure_writes_no_success_report(tmp_path: Path) -> None:
    run, testset = _fixture(tmp_path, count=1, denoise_steps=3, successes=set(), name="clean_bad")
    infer_path = next(run.glob("episodes/*/infer.jsonl"))
    infer = json.loads(infer_path.read_text(encoding="utf-8"))
    infer["server_experiment"]["noise_sha256"] = "0" * 64
    _write_jsonl(infer_path, [infer])
    output = tmp_path / "must_not_exist.json"
    result = main(
        [
            "--run-dir", str(run),
            "--testset-manifest", str(testset),
            "--expected-count", "1",
            "--expected-denoise-steps", "3",
            "--output", str(output),
        ]
    )
    assert result == 2
    assert not output.exists()


def test_rejects_trace_action_or_nonfinite_measured_pose(tmp_path: Path) -> None:
    run, testset = _fixture(tmp_path, count=1, denoise_steps=10, successes=set(), name="clean_bad_trace")
    trace_path = next(run.glob("episodes/*/trace.jsonl"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["executed_action"][0] += 1.0
    trace["measured_eef_pos"][2] = float("nan")
    _write_jsonl(trace_path, [trace])
    with pytest.raises(AuditError, match="executed_action differs"):
        audit_run(run, testset, expected_count=1, expected_denoise_steps=10)


def test_final_100_pair_reports_rates_noise_pairing_and_exact_mcnemar(tmp_path: Path) -> None:
    d3_success = set(range(10))
    d10_success = set(range(5, 20))
    d3, testset = _fixture(tmp_path, count=100, denoise_steps=3, successes=d3_success, name="clean_d3_100")
    d10, _ = _fixture(tmp_path, count=100, denoise_steps=10, successes=d10_success, name="clean_d10_100")
    report = audit_pair(d3, d10, testset, expected_count=100)
    assert report["strict_final_100"] is True
    assert report["d3"]["success_rate"] == 0.10
    assert report["d10"]["success_rate"] == 0.15
    assert report["paired_noise"]["common_inferences"] == 100
    assert report["paired_noise"]["all_common_hashes_equal"] is True
    table = report["mcnemar_exact"]
    assert table == {
        "both_success": 5,
        "d3_only_success": 5,
        "d10_only_success": 10,
        "both_failure": 80,
        "discordant_pairs": 15,
        "two_sided_p_value": mcnemar_exact_two_sided(5, 10),
        "method": "two-sided exact binomial McNemar test, p=2*lower tail, capped at 1",
    }
    assert report["scope"]["rollouts_per_task"] == 1
    assert "not a comprehensive" in report["scope"]["interpretation"]


def test_rejects_active_resume_marker(tmp_path: Path) -> None:
    run, testset = _fixture(tmp_path, count=1, denoise_steps=3, successes=set(), name="resumed")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resume"] = True
    _write_json(manifest_path, manifest)
    with pytest.raises(AuditError, match="active resume marker"):
        audit_run(run, testset, expected_count=1)
