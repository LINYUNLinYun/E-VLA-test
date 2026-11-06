from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_layer_denoise_ablation import DEFAULT_CONFIG  # noqa: E402
from run_layer_denoise_ablation import _condition_specs  # noqa: E402
from run_layer_denoise_ablation import _load_config  # noqa: E402
from summarize_layer_denoise_ablation import summarize_experiment  # noqa: E402


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_expected_condition_grid() -> None:
    conditions = _condition_specs(_load_config(DEFAULT_CONFIG))
    assert len(conditions) == 10
    assert sum(bool(item["is_baseline"]) for item in conditions) == 1
    assert {item["condition_id"] for item in conditions} == {
        f"{layer}__{flow}"
        for layer in ("first3_last3", "first6", "last6")
        for flow in ("full10", "first3", "last3")
    } | {"full18__full10"}


def test_synthetic_complete_sweep() -> None:
    conditions = _condition_specs(_load_config(DEFAULT_CONFIG))
    with tempfile.TemporaryDirectory(prefix="pi0-ablation-summary-") as temporary:
        temp_root = Path(temporary)
        original = __import__("summarize_layer_denoise_ablation")
        original_root = original.REPO_ROOT
        try:
            synthetic_repo = temp_root / "repo"
            synthetic_repo.mkdir()
            original.REPO_ROOT = synthetic_repo
            for condition_index, condition in enumerate(conditions):
                run_dir = synthetic_repo / "data/test" / f"run_{condition['condition_id'].replace('-', '_')}"
                condition["run_id"] = run_dir.name
                condition["run_dir"] = run_dir.relative_to(synthetic_repo).as_posix()
                condition["status"] = "completed"
                records = []
                task_dir = run_dir / "episodes/task05_push_the_plate_to_the_front_of_the_stove"
                for episode_idx in range(50):
                    success = (episode_idx + condition_index) % 4 != 0
                    status = "success" if success else "failure"
                    episode_dir = task_dir / f"ep{episode_idx:03d}_{status}"
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    (episode_dir / f"rollout_{status}.mp4").write_bytes(b"synthetic-video")
                    trace = {
                        "executed_action": [0.0] * 7,
                        "episode_idx": episode_idx,
                        "task_id": 5,
                    }
                    infer = {"infer_id": 0, "sample_actions_ms": 12.0 + condition_index}
                    (episode_dir / "trace.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
                    (episode_dir / "infer.jsonl").write_text(json.dumps(infer) + "\n", encoding="utf-8")
                    records.append(
                        {
                            "run_id": run_dir.name,
                            "task_suite_name": "libero_goal",
                            "task_id": 5,
                            "task_description": "push the plate to the front of the stove",
                            "episode_idx": episode_idx,
                            "success": success,
                            "failure_reason": None,
                            "trajectory_length": 1,
                            "executed_action_count": 1,
                            "infer_calls": 1,
                            "infer_latency_mean_ms": 12.0 + condition_index,
                            "infer_latency_sum_ms": 12.0 + condition_index,
                            "avg_latency_per_action_ms": 12.0 + condition_index,
                            "policy_time_mean_ms": 12.5 + condition_index,
                            "serve_time_mean_ms": 13.0 + condition_index,
                            "video_path": f"{episode_dir.relative_to(run_dir).as_posix()}/rollout_{status}.mp4",
                            "trace_path": f"{episode_dir.relative_to(run_dir).as_posix()}/trace.jsonl",
                            "infer_path": f"{episode_dir.relative_to(run_dir).as_posix()}/infer.jsonl",
                            "trace_record_count": 1,
                            "infer_record_count": 1,
                        }
                    )
                _write_json(run_dir / "manifest.json", {
                    "run_id": run_dir.name,
                    "task_suite_name": "libero_goal",
                    "selected_task_ids": [5],
                    "num_trials_per_task": 50,
                })
                _write_json(run_dir / "episode_log.json", records)
            manifest_path = temp_root / "master_manifest.json"
            _write_json(
                manifest_path,
                {
                    "experiment_id": "synthetic",
                    "status": "completed",
                    "task_suite_name": "libero_goal",
                    "task_id": 5,
                    "task_description": "push the plate to the front of the stove",
                    "num_trials_per_condition": 50,
                    "baseline_condition_id": "full18__full10",
                    "conditions": conditions,
                },
            )
            output_dir = temp_root / "results"
            errors = summarize_experiment(manifest_path, output_dir)
            assert errors == [], errors
            assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["episode_record_count"] == 500
            assert len((output_dir / "episode_results.csv").read_text(encoding="utf-8").splitlines()) == 501
            assert len((output_dir / "paired_vs_baseline.csv").read_text(encoding="utf-8").splitlines()) == 451
            assert len((output_dir / "video_index.csv").read_text(encoding="utf-8").splitlines()) == 501
            assert all((output_dir / name).is_file() for name in (
                "experiment_report.md", "manifest.json", "summary.json", "episode_results.csv",
                "paired_vs_baseline.csv", "video_index.csv", "success_rate_heatmap.png", "latency_heatmap.png",
            ))
            first_condition = conditions[0]
            first_log = synthetic_repo / first_condition["run_dir"] / "episode_log.json"
            first_records = json.loads(first_log.read_text(encoding="utf-8"))
            first_log.write_text(json.dumps(first_records[:-1]), encoding="utf-8")
            errors = summarize_experiment(manifest_path, output_dir)
            assert any("missing episode indices [49]" in error for error in errors)
            assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "incomplete"
        finally:
            original.REPO_ROOT = original_root


if __name__ == "__main__":
    test_expected_condition_grid()
    test_synthetic_complete_sweep()
    print("synthetic ablation summary tests passed")
