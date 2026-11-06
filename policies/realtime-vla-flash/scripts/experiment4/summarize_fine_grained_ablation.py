"""Summarize the fine-grained Action Expert × denoising experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment3"))
import summarize_skip_layers_experiment as stage1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = stage1._read_json(args.manifest.resolve())
    output_dir = args.output_dir.resolve()
    expected = int(manifest["num_trials_per_condition"])
    summaries = []
    episodes = []
    errors = []
    for condition in manifest["conditions"]:
        try:
            summary, condition_episodes, _ = stage1._condition_summary(condition, expected)
        except Exception as exc:  # noqa: BLE001 - collect all integrity errors
            errors.append(str(exc))
            continue
        summaries.append(summary)
        episodes.extend(condition_episodes)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1._write_csv(
        output_dir / "condition_results.csv",
        summaries,
        list(summaries[0]) if summaries else ["condition_id"],
    )
    stage1._write_csv(
        output_dir / "episode_results.csv",
        episodes,
        list(episodes[0]) if episodes else ["condition_id"],
    )
    payload = {
        "experiment_id": manifest["experiment_id"],
        "generated_at": stage1._now(),
        "status": "completed" if not errors and len(summaries) == 10 else "incomplete",
        "integrity_passed": not errors and len(summaries) == 10 and len(episodes) == 10 * expected,
        "integrity_errors": errors,
        "condition_count": len(summaries),
        "episode_record_count": len(episodes),
        "conditions": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# π0 Action Expert depth × denoising fine-grained ablation",
        "",
        f"- experiment_id: `{manifest['experiment_id']}`",
        f"- status: `{payload['status']}`",
        f"- conditions: {len(summaries)}/10",
        f"- episodes: {len(episodes)}/{10 * expected}",
        "",
    ]
    if errors:
        report.extend(["## Integrity errors", "", *[f"- {error}" for error in errors], ""])
    (output_dir / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")
    return 0 if payload["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
