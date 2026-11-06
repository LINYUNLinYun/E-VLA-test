#!/usr/bin/env python3
"""Recompute semantic phase labels without rerunning the simulator."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .trajectory_analysis import read_csv, read_jsonl, telemetry_summary, write_csv, write_jsonl
    from .trajectory_replay import assign_semantic_phases
except ImportError:
    from trajectory_analysis import read_csv, read_jsonl, telemetry_summary, write_csv, write_jsonl
    from trajectory_replay import assign_semantic_phases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry_dir", type=Path)
    args = parser.parse_args()
    root = args.telemetry_dir.resolve()
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    phase_counts: Counter[str] = Counter()
    for path in sorted(root.glob("*/*.telemetry.jsonl")):
        rows = read_jsonl(path)
        assign_semantic_phases(rows)
        write_jsonl(path, rows)
        summary = telemetry_summary(rows)
        phase_counts.update(str(row.get("semantic_phase", "unknown")) for row in rows)
        suffix = ".telemetry.jsonl"
        sample_id = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
        summaries[(path.parent.name, sample_id)] = summary

    validation_path = root / "replay_validation.csv"
    if validation_path.is_file():
        validation = read_csv(validation_path)
        for row in validation:
            summary = summaries.get((row["condition_id"], row["sample_id"]))
            if summary is not None:
                row["max_phase"] = str(summary["furthest_phase"])
        write_csv(validation_path, validation, validation[0].keys() if validation else None)

    report = {
        "episode_count": len(summaries),
        "phase_counts": dict(sorted(phase_counts.items())),
        "validation_updated": validation_path.is_file(),
    }
    (root / "reclassification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
