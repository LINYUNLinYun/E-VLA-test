#!/usr/bin/env python3
"""Build every available D10 baseline/Attention16/FFN2 telemetry comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment8.plot_paired_xyz import (  # noqa: E402
    compare_trajectories,
    plot_comparison,
    read_trajectory,
)


BASELINE = "baseline"
VARIANTS = ("skip_attn16", "skip_ffn2")
CONDITIONS = (BASELINE, *VARIANTS)
BASELINE_LABEL = "D10 baseline (Full18 VLM / Full18 action)"
VARIANT_LABELS = {
    "skip_attn16": "Skip Attention L16 / D10",
    "skip_ffn2": "Skip FFN L2 / D10",
}
ACTION_SPACE = "pi0_libero_unnormalized_7d_policy_command"
REQUIRED_RESULT_FILES = (
    "summary.json",
    "telemetry.jsonl",
    "inferences.jsonl",
    "rollout.mp4",
    "trajectory.csv",
    "trajectory.npz",
)
SUMMARY_CONFIG_FIELDS = (
    "condition_label",
    "semantic_task_key",
    "skip_attn_layers",
    "skip_ffn_layers",
    "denoise_steps",
    "init_state_sha256",
    "init_source",
    "official_init_index",
    "noise_seed",
    "env_seed",
    "warmup_steps",
    "replan_steps",
    "success",
    "executed_actions",
    "state_count",
    "video_frame_count",
    "coordinate_frame",
    "units",
    "control_frequency_hz",
    "termination_reason",
)
PAIR_IDENTITY_FIELDS = (
    "semantic_task_key",
    "init_state_sha256",
    "init_source",
    "official_init_index",
    "noise_seed",
    "env_seed",
    "warmup_steps",
    "replan_steps",
    "coordinate_frame",
    "units",
    "control_frequency_hz",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: record must be a JSON object")
                records.append(record)
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _int_field(value: Any, *, field: str, path: Path, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path}: {field} must be an integer >= {minimum}")
    return value


def _validate_summary_config(path: Path, condition: str) -> dict[str, Any]:
    summary = _read_json(path)
    missing = [field for field in SUMMARY_CONFIG_FIELDS if field not in summary]
    if missing:
        raise ValueError(f"Missing D10 run configuration in {path}: {', '.join(missing)}")

    expected_attn = [16] if condition == "skip_attn16" else []
    expected_ffn = [2] if condition == "skip_ffn2" else []
    if summary["skip_attn_layers"] != expected_attn:
        raise ValueError(
            f"{path}: condition {condition} requires skip_attn_layers={expected_attn}, "
            f"got {summary['skip_attn_layers']!r}"
        )
    if summary["skip_ffn_layers"] != expected_ffn:
        raise ValueError(
            f"{path}: condition {condition} requires skip_ffn_layers={expected_ffn}, "
            f"got {summary['skip_ffn_layers']!r}"
        )
    if summary["denoise_steps"] != 10:
        raise ValueError(f"{path}: expected the D10 experiment, got denoise_steps={summary['denoise_steps']!r}")
    if not isinstance(summary["condition_label"], str) or not summary["condition_label"].strip():
        raise ValueError(f"{path}: condition_label is missing or empty")
    if not isinstance(summary["semantic_task_key"], str) or not summary["semantic_task_key"].strip():
        raise ValueError(f"{path}: semantic_task_key is missing or empty")
    init_hash = summary["init_state_sha256"]
    if not isinstance(init_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", init_hash):
        raise ValueError(f"{path}: init_state_sha256 must be a SHA-256 digest")
    if not isinstance(summary["success"], bool):
        raise ValueError(f"{path}: success must be boolean")
    if summary["coordinate_frame"] != "MuJoCo world" or summary["units"] != "m":
        raise ValueError(f"{path}: expected measured MuJoCo world coordinates in meters")
    frequency = summary["control_frequency_hz"]
    if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f"{path}: control_frequency_hz must be positive and finite")
    for field in ("official_init_index", "noise_seed", "env_seed", "warmup_steps", "replan_steps",
                  "executed_actions", "state_count", "video_frame_count"):
        _int_field(summary[field], field=field, path=path)
    if summary["replan_steps"] < 1:
        raise ValueError(f"{path}: replan_steps must be positive")
    if summary["state_count"] != summary["executed_actions"] + 1:
        raise ValueError(f"{path}: state_count must equal executed_actions + 1")
    if summary["video_frame_count"] != summary["state_count"]:
        raise ValueError(f"{path}: state-synchronous source video must have one frame per saved state")
    return summary


def _validate_state_synchronous_telemetry(path: Path, expected_count: int) -> list[dict[str, Any]]:
    # read_trajectory validates that XYZ is present, finite, and unique.
    trajectory = read_trajectory(path)
    raw_rows = _read_jsonl(path)
    if len(raw_rows) != expected_count or len(trajectory) != expected_count:
        raise ValueError(
            f"{path}: telemetry state count {len(raw_rows)} does not match summary {expected_count}"
        )
    for index, row in enumerate(raw_rows):
        expected_when = "initial" if index == 0 else "post_action"
        if row.get("step") != index:
            raise ValueError(f"{path}: expected contiguous environment step {index}, got {row.get('step')!r}")
        if row.get("state_index") != index or row.get("video_frame_index") != index:
            raise ValueError(
                f"{path}: N+1 state/video alignment requires state_index and video_frame_index={index}"
            )
        if row.get("when") != expected_when:
            raise ValueError(f"{path}: state {index} must be marked when={expected_when!r}")
        expected_action_index = None if index == 0 else index - 1
        if row.get("action_index") != expected_action_index:
            raise ValueError(f"{path}: state {index} has incorrect action_index alignment")
    return trajectory


def _validate_inferences(path: Path, condition: str, summary: dict[str, Any]) -> dict[int, str]:
    records = _read_jsonl(path)
    expected_attn = [16] if condition == "skip_attn16" else []
    expected_ffn = [2] if condition == "skip_ffn2" else []
    hashes: dict[int, str] = {}
    for line_index, record in enumerate(records, start=1):
        infer_id = _int_field(record.get("infer_id"), field="infer_id", path=path)
        if infer_id in hashes:
            raise ValueError(f"{path}:{line_index}: duplicate infer_id {infer_id}")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}:{line_index}: diagnostic response metadata is missing")
        for field in ("skip_attn_layers", "skip_ffn_layers", "denoise_steps", "action_space",
                      "noise_sha256", "noise_identity", "eager_diagnostic",
                      "vlm_layer_pattern", "action_layer_pattern"):
            if field not in metadata:
                raise ValueError(f"{path}:{line_index}: D10 response config field {field!r} is missing")
        if metadata["skip_attn_layers"] != expected_attn or metadata["skip_ffn_layers"] != expected_ffn:
            raise ValueError(f"{path}:{line_index}: server response masks do not match {condition}")
        if metadata["denoise_steps"] != 10 or metadata["action_space"] != ACTION_SPACE:
            raise ValueError(f"{path}:{line_index}: response is not a D10 pi0_libero 7D command")
        if metadata["eager_diagnostic"] is not True:
            raise ValueError(f"{path}:{line_index}: response is not marked eager diagnostic")
        if metadata["vlm_layer_pattern"] != "vlm_full18" or metadata["action_layer_pattern"] != "full18":
            raise ValueError(f"{path}:{line_index}: expected Full18 VLM/action runtime")
        noise_hash = metadata["noise_sha256"]
        if not isinstance(noise_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", noise_hash):
            raise ValueError(f"{path}:{line_index}: invalid noise_sha256")
        identity = metadata["noise_identity"]
        if not isinstance(identity, dict):
            raise ValueError(f"{path}:{line_index}: noise_identity is missing")
        expected_identity = {
            "semantic_task_key": summary["semantic_task_key"],
            "init_state_sha256": summary["init_state_sha256"].lower(),
            "noise_seed": summary["noise_seed"],
            "infer_id": infer_id,
        }
        if identity != expected_identity:
            raise ValueError(f"{path}:{line_index}: noise identity does not match rollout summary")
        actions = np.asarray(record.get("actions"), dtype=float)
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError(f"{path}:{line_index}: expected finite transformed [50,7] actions")
        hashes[infer_id] = noise_hash.lower()
    if not hashes:
        raise ValueError(f"{path}: no inference records")
    return hashes


def _inspect_condition(case_dir: Path, condition: str) -> dict[str, Any]:
    condition_dir = case_dir / condition
    if not condition_dir.is_dir():
        return {"status": "incomplete", "condition": condition,
                "reason": f"missing condition directory: {condition_dir}"}

    summary_path = condition_dir / "summary.json"
    summary = None
    if summary_path.is_file():
        # A present but incomplete/misconfigured summary is a hard error. Only
        # absent result files are considered an unfinished rollout.
        summary = _validate_summary_config(summary_path, condition)

    missing_files = [name for name in REQUIRED_RESULT_FILES if not (condition_dir / name).is_file()]
    if missing_files:
        return {"status": "incomplete", "condition": condition,
                "reason": "missing result files: " + ", ".join(missing_files),
                "summary": summary,
                "directory": str(condition_dir.resolve())}

    assert summary is not None
    telemetry_path = condition_dir / "telemetry.jsonl"
    trajectory = _validate_state_synchronous_telemetry(telemetry_path, summary["state_count"])
    inference_hashes = _validate_inferences(condition_dir / "inferences.jsonl", condition, summary)
    if not (condition_dir / "rollout.mp4").stat().st_size:
        return {"status": "incomplete", "condition": condition,
                "reason": "source rollout.mp4 is empty", "summary": summary,
                "directory": str(condition_dir.resolve())}
    return {
        "status": "complete",
        "condition": condition,
        "directory": str(condition_dir.resolve()),
        "summary": summary,
        "telemetry_path": telemetry_path.resolve(),
        "trajectory": trajectory,
        "inference_noise_hashes": inference_hashes,
        "video_path": (condition_dir / "rollout.mp4").resolve(),
    }


def _validate_pair(baseline: dict[str, Any], variant: dict[str, Any], case_id: str) -> None:
    baseline_summary = baseline["summary"]
    variant_summary = variant["summary"]
    for field in PAIR_IDENTITY_FIELDS:
        if baseline_summary.get(field) != variant_summary.get(field):
            raise ValueError(
                f"{case_id}: baseline/variant mismatch for paired-run identity {field}: "
                f"{baseline_summary.get(field)!r} vs {variant_summary.get(field)!r}"
            )
    common_ids = baseline["inference_noise_hashes"].keys() & variant["inference_noise_hashes"].keys()
    if not common_ids:
        raise ValueError(f"{case_id}: baseline and variant have no shared inference IDs for paired noise")
    mismatched = [
        infer_id for infer_id in common_ids
        if baseline["inference_noise_hashes"][infer_id] != variant["inference_noise_hashes"][infer_id]
    ]
    if mismatched:
        raise ValueError(f"{case_id}: paired inference noise differs at infer IDs {sorted(mismatched)}")


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"Cannot create safe output name from {value!r}")
    return safe


def _render_pair_video(
    *, baseline_video: Path, variant_video: Path, baseline_telemetry: Path,
    variant_telemetry: Path, baseline_label: str, variant_label: str, output: Path,
) -> Path:
    renderer = Path(__file__).with_name("render_xyz_video.py")
    command = [
        sys.executable, str(renderer),
        "--baseline-video", str(baseline_video),
        "--variant-video", str(variant_video),
        "--baseline-telemetry", str(baseline_telemetry),
        "--variant-telemetry", str(variant_telemetry),
        "--baseline-label", baseline_label,
        "--variant-label", variant_label,
        "--alignment", "state-synchronous",
        "--output", str(output),
    ]
    subprocess.run(command, check=True)
    report_path = output.with_suffix(".json")
    if not output.is_file() or not report_path.is_file():
        raise RuntimeError(f"XYZ video renderer did not produce both {output} and {report_path}")
    return report_path


def _check_no_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError("Refusing to overwrite gallery artifact(s): " + ", ".join(existing))


def _gallery_html(records: list[dict[str, Any]], *, closed_loop_dir: Path) -> str:
    cards: list[str] = []
    for record in records:
        title = f"{record['case_id']} · {record['variant']}"
        if record["status"] == "rendered":
            artifacts = record["artifacts"]
            cards.append(
                '<article class="pair">'
                f"<h2>{html.escape(title)}</h2>"
                f"<p>Outcome: baseline={html.escape(str(record['baseline_success']))}, "
                f"variant={html.escape(str(record['variant_success']))}; "
                f"maximum paired XYZ deviation={record['maximum_deviation_m']:.4f} m; "
                f"shared steps={record['shared_steps']}.</p>"
                f'<a href="{html.escape(artifacts["pdf"], quote=True)}">PDF</a> · '
                f'<a href="{html.escape(artifacts["summary"], quote=True)}">JSON summary</a> · '
                f'<a href="{html.escape(artifacts["video_report"], quote=True)}">video report</a> · '
                f'<a href="{html.escape(artifacts["png"], quote=True)}">'
                f'<img loading="lazy" src="{html.escape(artifacts["png"], quote=True)}" '
                'alt="Measured EEF trajectory comparison"></a>'
                f'<video controls preload="metadata" src="{html.escape(artifacts["video"], quote=True)}"></video>'
                "</article>"
            )
        elif record["status"] == "skipped_incomplete":
            reasons = "<br>".join(html.escape(reason) for reason in record["reasons"])
            cards.append(
                '<article class="skipped">'
                f"<h2>{html.escape(title)} — explicitly skipped as incomplete</h2>"
                f"<p>{reasons}</p></article>"
            )
        else:
            cards.append(
                '<article class="skipped">'
                f"<h2>{html.escape(title)} — rendering failed</h2>"
                f"<p>{html.escape(record.get('error', 'unknown rendering error'))}</p></article>"
            )
    return """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>D10 closed-loop XYZ gallery</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem;color:#222}
article{margin:2.5rem 0;padding:1rem;border:1px solid #ddd;border-radius:8px}
article.skipped{border-color:#d98b20;background:#fff9ed}.pair img,.pair video{display:block;width:100%;height:auto;margin-top:1rem}
a{color:#145b96}
</style>
<h1>D10 closed-loop measured EEF trajectory gallery</h1>
<p>Blue is the paired eager D10 Full18/Full18 baseline. Orange is the specified Attention-16 or FFN-2
intervention. XYZ comes from simulator telemetry in MuJoCo world meters. Every plot pairs states by
environment step; videos use state-synchronous N+1 telemetry/video alignment. Comparison distances use
only shared environment steps. All discovered expected pairs are listed below; unfinished ones are
explicitly marked instead of silently omitted.</p>
<p>Input results: """ + html.escape(str(closed_loop_dir)) + """</p>
""" + "\n".join(cards) + "\n</html>\n"


def build_gallery(
    *,
    closed_loop_dir: Path,
    output_dir: Path,
    video_renderer: Callable[..., Path] = _render_pair_video,
) -> dict[str, Any]:
    source_root = Path(closed_loop_dir).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError(f"Closed-loop input must be a directory: {source_root}")
    supplied_output = Path(output_dir).expanduser()
    if supplied_output.name in ("", ".", ".."):
        raise ValueError(f"Output path must name a new directory: {output_dir}")
    final_output = supplied_output.parent.resolve() / supplied_output.name
    if os.path.lexists(final_output):
        raise FileExistsError(f"Refusing to overwrite gallery output directory: {final_output}")

    case_dirs = sorted(
        path for path in source_root.iterdir()
        if path.is_dir() and any((path / condition).is_dir() for condition in CONDITIONS)
    )
    if not case_dirs:
        raise ValueError(f"No closed-loop task directories found under {source_root}")

    inspected: dict[tuple[str, str], dict[str, Any]] = {}
    for case_dir in case_dirs:
        for condition in CONDITIONS:
            inspected[(case_dir.name, condition)] = _inspect_condition(case_dir, condition)

    planned: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        baseline = inspected[(case_dir.name, BASELINE)]
        for variant_name in VARIANTS:
            variant = inspected[(case_dir.name, variant_name)]
            reasons: list[str] = []
            for result in (baseline, variant):
                if result["status"] != "complete":
                    reasons.append(f"{result['condition']}: {result['reason']}")
            if reasons:
                planned.append({
                    "case_id": case_dir.name,
                    "variant": variant_name,
                    "status": "skipped_incomplete",
                    "reasons": reasons,
                    "baseline_source": baseline.get("directory"),
                    "variant_source": variant.get("directory"),
                })
                continue
            _validate_pair(baseline, variant, f"{case_dir.name}/{variant_name}")
            planned.append({
                "case_id": case_dir.name,
                "variant": variant_name,
                "status": "ready",
                "baseline": baseline,
                "variant_data": variant,
            })

    # Preflight succeeded, so reserve a fresh output directory. It is never
    # opened in append/replace mode and all child paths are checked before use.
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.mkdir(exist_ok=False)
    records: list[dict[str, Any]] = []
    for item in planned:
        if item["status"] == "skipped_incomplete":
            records.append({key: value for key, value in item.items() if key != "status"} | {
                "status": "skipped_incomplete",
                "baseline_label": BASELINE_LABEL,
                "variant_label": VARIANT_LABELS[item["variant"]],
                "alignment": "state-synchronous N+1 states/frames",
            })
            continue

        case_id = item["case_id"]
        variant_name = item["variant"]
        baseline = item["baseline"]
        variant = item["variant_data"]
        baseline_label = BASELINE_LABEL
        variant_label = VARIANT_LABELS[variant_name]
        stem = f"{_safe_name(case_id)}__{variant_name}"
        artifact_dir = final_output / "pairs" / _safe_name(case_id)
        prefix = artifact_dir / variant_name
        png_path = prefix.with_suffix(".png")
        pdf_path = prefix.with_suffix(".pdf")
        summary_path = prefix.with_suffix(".summary.json")
        video_path = prefix.with_suffix(".mp4")
        video_report_path = prefix.with_suffix(".json")
        _check_no_existing([png_path, pdf_path, summary_path, video_path, video_report_path])

        record: dict[str, Any] = {
            "case_id": case_id,
            "variant": variant_name,
            "status": "rendering",
            "baseline_label": baseline_label,
            "variant_label": variant_label,
            "alignment": "state-synchronous N+1 states/frames",
            "baseline_source": baseline["directory"],
            "variant_source": variant["directory"],
            "paired_identity": {
                field: baseline["summary"][field] for field in PAIR_IDENTITY_FIELDS
            },
            "baseline_success": baseline["summary"]["success"],
            "variant_success": variant["summary"]["success"],
        }
        try:
            comparison = compare_trajectories(
                baseline["trajectory"],
                variant["trajectory"],
                baseline_label=baseline_label,
                variant_label=variant_label,
                baseline_path=str(baseline["telemetry_path"]),
                variant_path=str(variant["telemetry_path"]),
            )
            plot_comparison(
                comparison,
                baseline_label=baseline_label,
                variant_label=variant_label,
                png_path=png_path,
                pdf_path=pdf_path,
            )
            video_report = video_renderer(
                baseline_video=baseline["video_path"],
                variant_video=variant["video_path"],
                baseline_telemetry=baseline["telemetry_path"],
                variant_telemetry=variant["telemetry_path"],
                baseline_label=baseline_label,
                variant_label=variant_label,
                output=video_path,
            )
            if Path(video_report) != video_report_path or not Path(video_report).is_file():
                raise RuntimeError(f"Expected video report at {video_report_path}, got {video_report}")
            summary = {key: value for key, value in comparison.items() if key != "_plot_data"}
            summary.update({
                "experiment": "closed_loop_D10_attention_ffn_pair",
                "alignment": "state-synchronous N+1 states/frames",
                "baseline_summary": str(Path(baseline["directory"]) / "summary.json"),
                "variant_summary": str(Path(variant["directory"]) / "summary.json"),
                "artifacts": {
                    "png": str(png_path.relative_to(final_output)),
                    "pdf": str(pdf_path.relative_to(final_output)),
                    "video": str(video_path.relative_to(final_output)),
                    "video_report": str(video_report_path.relative_to(final_output)),
                },
            })
            _check_no_existing([summary_path])
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            deviation = comparison["paired_deviation_m"]
            record.update({
                "status": "rendered",
                "shared_steps": comparison["step_coverage"]["shared_step_count"],
                "maximum_deviation_m": deviation["maximum"],
                "maximum_deviation_step": deviation["maximum_step"],
                "artifacts": {
                    "png": str(png_path.relative_to(final_output)),
                    "pdf": str(pdf_path.relative_to(final_output)),
                    "summary": str(summary_path.relative_to(final_output)),
                    "video": str(video_path.relative_to(final_output)),
                    "video_report": str(video_report_path.relative_to(final_output)),
                },
            })
        except Exception as exc:
            record.update({"status": "render_failed", "error": f"{type(exc).__name__}: {exc}"})
        records.append(record)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "closed_loop_D10_attention_ffn_gallery",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_directory": str(source_root),
        "output_directory": str(final_output.resolve()),
        "baseline_label": BASELINE_LABEL,
        "baseline_label_source": "gallery caller-defined D10 Full18/Full18 condition, not historical Control",
        "variant_labels": VARIANT_LABELS,
        "position_source": "simulator telemetry eef_pos; no action-command integration",
        "coordinate_frame": "MuJoCo world",
        "units": "m",
        "pairing": "same environment step; state-synchronous N+1 video/telemetry alignment",
        "denoise_steps": 10,
        "expected_conditions": list(CONDITIONS),
        "pair_count": len(records),
        "rendered_pair_count": sum(record["status"] == "rendered" for record in records),
        "skipped_incomplete_pair_count": sum(record["status"] == "skipped_incomplete" for record in records),
        "render_failed_pair_count": sum(record["status"] == "render_failed" for record in records),
        "pairs": records,
    }
    manifest_path = final_output / "manifest.json"
    index_path = final_output / "index.html"
    _check_no_existing([manifest_path, index_path])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index_path.write_text(_gallery_html(records, closed_loop_dir=source_root), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closed-loop-dir",
        type=Path,
        default=ROOT / "results/pi0_attention_ffn_diagnosis/closed_loop",
        help="Root containing task directories with baseline/skip_attn16/skip_ffn2 results.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be a new, non-existing directory.")
    args = parser.parse_args(argv)
    manifest = build_gallery(closed_loop_dir=args.closed_loop_dir, output_dir=args.output_dir)
    print(json.dumps({
        "output_dir": manifest["output_directory"],
        "manifest": str(Path(manifest["output_directory"]) / "manifest.json"),
        "index": str(Path(manifest["output_directory"]) / "index.html"),
        "pair_count": manifest["pair_count"],
        "rendered_pair_count": manifest["rendered_pair_count"],
        "skipped_incomplete_pair_count": manifest["skipped_incomplete_pair_count"],
        "render_failed_pair_count": manifest["render_failed_pair_count"],
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["render_failed_pair_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
