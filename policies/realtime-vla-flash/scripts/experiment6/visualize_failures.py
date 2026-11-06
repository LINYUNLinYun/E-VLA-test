#!/usr/bin/env python3
"""Render paired Control/candidate trajectories and failure-stage summaries."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 10.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"video has no frames: {path}")
    return frames, fps


def _resize(frame: np.ndarray, height: int = 360) -> np.ndarray:
    width = max(1, round(frame.shape[1] * height / frame.shape[0]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _telemetry_text(rows: list[dict[str, Any]], step: int) -> list[str]:
    if not rows:
        return ["phase: telemetry unavailable"]
    row = rows[min(max(step, 0), len(rows) - 1)]
    distance = row.get("min_eef_object_distance")
    distance_text = "n/a" if distance is None else f"{float(distance):.3f}m"
    return [
        f"phase: {row.get('semantic_phase', 'unknown')}",
        f"eef-target: {distance_text}",
        f"goal: {row.get('goal_progress', 0)}/{row.get('goal_predicate_count', 0)}",
        f"contact/grasp: {any(row.get('robot_contacts', {}).values())}/{any(row.get('robot_grasps', {}).values())}",
    ]


def _annotate(frame: np.ndarray, title: str, lines: list[str], *, highlight: bool = False) -> np.ndarray:
    canvas = frame.copy()
    overlay_height = 28 + 22 * len(lines)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], overlay_height), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 45 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    if highlight:
        cv2.rectangle(canvas, (2, 2), (canvas.shape[1] - 3, canvas.shape[0] - 3), (0, 0, 255), thickness=5)
    return canvas


def render_pair(
    row: dict[str, str],
    *,
    telemetry_dir: Path | None,
    output_path: Path,
    contact_sheet_path: Path,
) -> dict[str, Any]:
    control_frames, control_fps = _read_video(Path(row["control_video"]))
    candidate_frames, candidate_fps = _read_video(Path(row["candidate_video"]))
    condition = row["condition_id"]
    sample_id = row["sample_id"]
    control_telemetry = _jsonl(None if telemetry_dir is None else telemetry_dir / "Control" / f"{sample_id}.telemetry.jsonl")
    candidate_telemetry = _jsonl(None if telemetry_dir is None else telemetry_dir / condition / f"{sample_id}.telemetry.jsonl")
    length = max(len(control_frames), len(candidate_frames))
    first_divergence = None
    try:
        first_divergence = int(float(row["first_divergence_step"]))
    except (KeyError, TypeError, ValueError):
        pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    left_width = _resize(control_frames[0]).shape[1]
    right_width = _resize(candidate_frames[0]).shape[1]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        min(control_fps, candidate_fps, 10.0),
        (left_width + right_width, 360),
    )
    sheet_indices = sorted(set(round(value * (length - 1) / 5) for value in range(6)))
    sheet_rows = []
    for step in range(length):
        left = _resize(control_frames[min(step, len(control_frames) - 1)])
        right = _resize(candidate_frames[min(step, len(candidate_frames) - 1)])
        highlight = first_divergence is not None and step == first_divergence
        left = _annotate(left, f"Control | {sample_id} | step {step}", _telemetry_text(control_telemetry, step - 1), highlight=highlight)
        right = _annotate(right, f"{condition} | failure | step {step}", _telemetry_text(candidate_telemetry, step - 1), highlight=highlight)
        pair = np.concatenate([left, right], axis=1)
        writer.write(pair)
        if step in sheet_indices:
            thumb = cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"step {step}", (8, thumb.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            sheet_rows.append(thumb)
    writer.release()
    if sheet_rows:
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(contact_sheet_path), np.concatenate(sheet_rows, axis=0))
    return {
        "condition_id": condition,
        "sample_id": sample_id,
        "paired_video": str(output_path),
        "contact_sheet": str(contact_sheet_path),
        "frames": length,
        "telemetry_available": bool(control_telemetry and candidate_telemetry),
        "candidate_furthest_phase": row.get("candidate_furthest_phase", row.get("failure_phase", "")),
        "first_failed_transition": row.get("first_failed_transition", ""),
    }


def _plot_stage_counts(rows: list[dict[str, str]], path: Path) -> None:
    semantic = [row for row in rows if row.get("phase_source") == "simulator_telemetry"]
    source = semantic if semantic else rows
    conditions = sorted({row["condition_id"] for row in source})
    phases = sorted({row.get("candidate_furthest_phase") or row.get("failure_phase", "unknown") for row in source})
    counts = Counter((row["condition_id"], row.get("candidate_furthest_phase") or row.get("failure_phase", "unknown")) for row in source)
    bottom = np.zeros(len(conditions), dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, len(conditions) * 1.6), 4.8))
    for phase in phases:
        values = np.asarray([counts[(condition, phase)] for condition in conditions], dtype=float)
        ax.bar(conditions, values, bottom=bottom, label=phase)
        bottom += values
    ax.set_ylabel("new-failure episodes")
    ax.set_title("Furthest phase reached before failure" + (" (semantic replay subset)" if semantic else " (action proxies)"))
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_html(records: list[dict[str, Any]], output_dir: Path) -> None:
    cards = []
    for record in records:
        video = Path(record["paired_video"]).relative_to(output_dir)
        sheet = Path(record["contact_sheet"]).relative_to(output_dir)
        cards.append(
            "<article><h2>{condition} · {sample}</h2>"
            "<p>furthest phase: <b>{phase}</b>; failed transition: <b>{transition}</b></p>"
            "<video controls preload='metadata' src='{video}'></video>"
            "<a href='{sheet}'><img loading='lazy' src='{sheet}'></a></article>".format(
                condition=html.escape(record["condition_id"]),
                sample=html.escape(record["sample_id"]),
                phase=html.escape(str(record["candidate_furthest_phase"])),
                transition=html.escape(str(record["first_failed_transition"] or "not yet classified")),
                video=html.escape(str(video)),
                sheet=html.escape(str(sheet)),
            )
        )
    document = """<!doctype html><meta charset='utf-8'><title>Experiment 6 failure gallery</title>
<style>body{font:15px sans-serif;max-width:1500px;margin:auto;background:#111;color:#eee}article{margin:24px 0;padding:16px;background:#222}video,img{max-width:100%;display:block;margin-top:10px}a{color:#8cf}</style>
<h1>VLM pruning paired failure gallery</h1>
<p>Video frames are pre-action; telemetry overlays describe the preceding post-action state.</p>""" + "\n".join(cards)
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def select_balanced_cases(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Round-robin conditions while preferring new semantic phases per condition."""

    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(row["condition_id"], []).append(row)
    for condition_rows in buckets.values():
        condition_rows.sort(
            key=lambda row: (
                row.get("candidate_furthest_phase", ""),
                row.get("sample_id", ""),
            )
        )
    selected: list[dict[str, str]] = []
    conditions = sorted(buckets)
    while len(selected) < limit and any(buckets.values()):
        for condition in conditions:
            if not buckets[condition] or len(selected) >= limit:
                continue
            seen = {
                row.get("candidate_furthest_phase", "")
                for row in selected
                if row["condition_id"] == condition
            }
            index = next(
                (i for i, row in enumerate(buckets[condition]) if row.get("candidate_furthest_phase", "") not in seen),
                0,
            )
            selected.append(buckets[condition].pop(index))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path)
    parser.add_argument("--conditions")
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args()
    rows = [row for row in _csv(args.paired_analysis.resolve()) if row.get("cohort") == "new_failure"]
    if args.conditions:
        allowed = {value.strip() for value in args.conditions.split(",") if value.strip()}
        rows = [row for row in rows if row["condition_id"] in allowed]
    # Prefer rows backed by semantic replay, then cover each condition in stable order.
    rows.sort(key=lambda row: (row.get("phase_source") != "simulator_telemetry", row["condition_id"], row["sample_id"]))
    all_rows = rows
    rows = select_balanced_cases(rows, args.max_cases)
    output_dir = args.output_dir.resolve()
    records = []
    for row in rows:
        stem = f"{row['condition_id']}__{row['sample_id']}"
        records.append(
            render_pair(
                row,
                telemetry_dir=args.telemetry_dir.resolve() if args.telemetry_dir else None,
                output_path=output_dir / "paired_videos" / f"{stem}.mp4",
                contact_sheet_path=output_dir / "contact_sheets" / f"{stem}.jpg",
            )
        )
    # The gallery is intentionally capped, but the aggregate stage chart must
    # represent the complete filtered new-failure cohort.
    _plot_stage_counts(all_rows, output_dir / "failure_stage_counts.png")
    _write_html(records, output_dir)
    (output_dir / "gallery_manifest.json").write_text(
        json.dumps({"case_count": len(records), "cases": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(records), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
