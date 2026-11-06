"""Small CPU-only tests for closed-loop D10 gallery discovery and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.experiment8.build_closed_loop_gallery import build_gallery


IDENTITY = {
    "semantic_task_key": "libero_10/synthetic_task",
    "init_state_sha256": "b" * 64,
    "init_source": "official",
    "official_init_index": 0,
    "noise_seed": 12345,
    "env_seed": 7,
    "warmup_steps": 10,
    "replan_steps": 12,
    "coordinate_frame": "MuJoCo world",
    "units": "m",
    "control_frequency_hz": 20,
}


def _summary(condition: str) -> dict[str, Any]:
    return {
        "condition_label": condition,
        **IDENTITY,
        "skip_attn_layers": [16] if condition == "skip_attn16" else [],
        "skip_ffn_layers": [2] if condition == "skip_ffn2" else [],
        "denoise_steps": 10,
        "success": condition != "skip_ffn2",
        "executed_actions": 2,
        "state_count": 3,
        "video_frame_count": 3,
        "termination_reason": "success" if condition != "skip_ffn2" else "step_limit",
    }


def _metadata(condition: str, infer_id: int, noise_sha256: str = "a" * 64) -> dict[str, Any]:
    return {
        "skip_attn_layers": [16] if condition == "skip_attn16" else [],
        "skip_ffn_layers": [2] if condition == "skip_ffn2" else [],
        "denoise_steps": 10,
        "action_space": "pi0_libero_unnormalized_7d_policy_command",
        "noise_sha256": noise_sha256,
        "noise_identity": {
            "semantic_task_key": IDENTITY["semantic_task_key"],
            "init_state_sha256": IDENTITY["init_state_sha256"],
            "noise_seed": IDENTITY["noise_seed"],
            "infer_id": infer_id,
        },
        "eager_diagnostic": True,
        "vlm_layer_pattern": "vlm_full18",
        "action_layer_pattern": "full18",
    }


def _write_condition(case_dir: Path, condition: str, *, noise_sha256: str = "a" * 64) -> None:
    condition_dir = case_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "summary.json").write_text(json.dumps(_summary(condition)), encoding="utf-8")
    (condition_dir / "rollout.mp4").write_bytes(b"synthetic source video")
    (condition_dir / "trajectory.csv").write_text("state_index,eef_x,eef_y,eef_z\n", encoding="utf-8")
    (condition_dir / "trajectory.npz").write_bytes(b"synthetic trajectory archive")
    rows = []
    for step in range(3):
        rows.append({
            "step": step,
            "when": "initial" if step == 0 else "post_action",
            "state_index": step,
            "video_frame_index": step,
            "action_index": None if step == 0 else step - 1,
            "eef_pos": [0.1 * step, (0.03 if condition == "skip_attn16" else 0.0), 0.8 + 0.02 * step],
        })
    (condition_dir / "telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    inferences = []
    for infer_id in range(2):
        inferences.append({
            "infer_id": infer_id,
            "start_action_index": infer_id,
            "metadata": _metadata(condition, infer_id, noise_sha256),
            "actions": [[0.0] * 7 for _ in range(50)],
        })
    (condition_dir / "inferences.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inferences), encoding="utf-8"
    )


def _fake_video_renderer(**kwargs: Any) -> Path:
    output = Path(kwargs["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"synthetic paired video")
    report = output.with_suffix(".json")
    report.write_text(json.dumps({"alignment": "state-synchronous N+1"}), encoding="utf-8")
    return report


def _input_root(tmp_path: Path, *, include_ffn: bool = False) -> Path:
    root = tmp_path / "closed_loop"
    case = root / "libero_10_task000_official0_seed12345"
    _write_condition(case, "baseline")
    _write_condition(case, "skip_attn16")
    if include_ffn:
        _write_condition(case, "skip_ffn2")
    return root


def test_gallery_outputs_all_available_pairs_and_explicitly_marks_incomplete(tmp_path: Path) -> None:
    source = _input_root(tmp_path, include_ffn=False)
    output = tmp_path / "gallery"
    rendered = []

    def fake_renderer(**kwargs: Any) -> Path:
        rendered.append(kwargs)
        return _fake_video_renderer(**kwargs)

    manifest = build_gallery(closed_loop_dir=source, output_dir=output, video_renderer=fake_renderer)

    assert manifest["pair_count"] == 2
    assert manifest["rendered_pair_count"] == 1
    assert manifest["skipped_incomplete_pair_count"] == 1
    assert [record["status"] for record in manifest["pairs"]] == ["rendered", "skipped_incomplete"]
    assert "missing condition directory" in manifest["pairs"][1]["reasons"][0]
    assert len(rendered) == 1
    assert output.joinpath("index.html").is_file()
    assert "explicitly skipped as incomplete" in output.joinpath("index.html").read_text(encoding="utf-8")
    assert output.joinpath("manifest.json").is_file()
    artifacts = manifest["pairs"][0]["artifacts"]
    for key in ("png", "pdf", "summary", "video", "video_report"):
        assert output.joinpath(artifacts[key]).is_file()
    summary = json.loads(output.joinpath(artifacts["summary"]).read_text(encoding="utf-8"))
    assert summary["baseline"]["label"] == manifest["baseline_label"]
    assert summary["variant"]["label"] == "Skip Attention L16 / D10"
    assert summary["step_coverage"]["shared_step_count"] == 3
    assert summary["position_source"]["action_command_integration_used"] is False


def test_gallery_rejects_missing_d10_configuration_before_output(tmp_path: Path) -> None:
    source = _input_root(tmp_path, include_ffn=True)
    summary_path = source / "libero_10_task000_official0_seed12345" / "baseline" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary["denoise_steps"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "gallery"

    with pytest.raises(ValueError, match="Missing D10 run configuration.*denoise_steps"):
        build_gallery(closed_loop_dir=source, output_dir=output, video_renderer=_fake_video_renderer)

    assert not output.exists()


def test_gallery_rejects_noise_mismatch_and_never_overwrites_output(tmp_path: Path) -> None:
    source = _input_root(tmp_path, include_ffn=True)
    attn_inferences = source / "libero_10_task000_official0_seed12345" / "skip_attn16" / "inferences.jsonl"
    records = [json.loads(line) for line in attn_inferences.read_text(encoding="utf-8").splitlines()]
    records[0]["metadata"]["noise_sha256"] = "c" * 64
    attn_inferences.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    output = tmp_path / "gallery"

    with pytest.raises(ValueError, match="paired inference noise differs"):
        build_gallery(closed_loop_dir=source, output_dir=output, video_renderer=_fake_video_renderer)
    assert not output.exists()

    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite gallery output directory"):
        build_gallery(closed_loop_dir=source, output_dir=output, video_renderer=_fake_video_renderer)
    assert marker.read_text(encoding="utf-8") == "unchanged"
