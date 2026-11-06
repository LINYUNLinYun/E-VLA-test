#!/usr/bin/env python3
"""Paired success and trajectory comparison for pi0 baseline vs FLASH runs."""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any


def _load(pattern: str) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for run_dir_raw in sorted(glob.glob(pattern)):
        run_dir = pathlib.Path(run_dir_raw)
        log_path = run_dir / "episode_log.json"
        if not log_path.is_file():
            continue
        for row in json.loads(log_path.read_text()):
            key = (str(row["task_suite_name"]), int(row["task_id"]), int(row["episode_idx"]))
            if key in rows:
                raise ValueError(f"duplicate episode key {key} in {pattern}")
            rows[key] = row
    if not rows:
        raise FileNotFoundError(f"no episode records matched {pattern}")
    return rows


def _length(row: dict[str, Any]) -> int:
    return int(row.get("trajectory_length", row["executed_action_count"]))


def _mean(values: list[int | float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[int | float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _suite_label(suite: str) -> str:
    return {
        "libero_spatial": "Spatial",
        "libero_object": "Object",
        "libero_goal": "Goal",
        "libero_10": "10",
    }.get(suite, suite)


def _mean_ci95(values: list[int | float]) -> tuple[float | None, list[float] | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, [mean - half_width, mean + half_width]


def _mcnemar_exact_p(baseline_only: int, flash_only: int) -> float:
    discordant = int(baseline_only) + int(flash_only)
    if discordant == 0:
        return 1.0
    lower = min(int(baseline_only), int(flash_only))
    tail = sum(math.comb(discordant, i) for i in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _summarize_pair(
    keys: list[tuple[str, int, int]],
    baseline: dict[tuple[str, int, int], dict[str, Any]],
    flash: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    base_rows = [baseline[key] for key in keys]
    flash_rows = [flash[key] for key in keys]
    n = len(keys)
    base_successes = sum(bool(row["success"]) for row in base_rows)
    flash_successes = sum(bool(row["success"]) for row in flash_rows)
    contingency = {"both_success": 0, "baseline_only": 0, "flash_only": 0, "both_failure": 0}
    both_success_base_lengths: list[int] = []
    both_success_flash_lengths: list[int] = []
    for key in keys:
        base_ok = bool(baseline[key]["success"])
        flash_ok = bool(flash[key]["success"])
        if base_ok and flash_ok:
            contingency["both_success"] += 1
            both_success_base_lengths.append(_length(baseline[key]))
            both_success_flash_lengths.append(_length(flash[key]))
        elif base_ok:
            contingency["baseline_only"] += 1
        elif flash_ok:
            contingency["flash_only"] += 1
        else:
            contingency["both_failure"] += 1

    base_lengths = [_length(row) for row in base_rows]
    flash_lengths = [_length(row) for row in flash_rows]
    success_deltas = [int(bool(flash[key]["success"])) - int(bool(baseline[key]["success"])) for key in keys]
    both_success_deltas = [
        flash_len - base_len
        for base_len, flash_len in zip(both_success_base_lengths, both_success_flash_lengths)
    ]
    success_delta_mean, success_delta_ci = _mean_ci95(success_deltas)
    trajectory_delta_mean, trajectory_delta_ci = _mean_ci95(both_success_deltas)
    return {
        "episodes": n,
        "baseline_successes": base_successes,
        "flash_successes": flash_successes,
        "baseline_success_rate": base_successes / n,
        "flash_success_rate": flash_successes / n,
        "success_delta_pp": 100.0 * (flash_successes - base_successes) / n,
        "success_delta_pp_ci95_normal": (
            None if success_delta_ci is None else [100.0 * success_delta_ci[0], 100.0 * success_delta_ci[1]]
        ),
        "mcnemar_exact_p": _mcnemar_exact_p(contingency["baseline_only"], contingency["flash_only"]),
        "contingency": contingency,
        "trajectory_all": {
            "baseline_mean": _mean(base_lengths),
            "flash_mean": _mean(flash_lengths),
            "paired_delta_mean": _mean([f - b for b, f in zip(base_lengths, flash_lengths)]),
        },
        "trajectory_both_success": {
            "episodes": len(both_success_deltas),
            "baseline_mean": _mean(both_success_base_lengths),
            "flash_mean": _mean(both_success_flash_lengths),
            "paired_delta_mean": trajectory_delta_mean,
            "paired_delta_mean_ci95_normal": trajectory_delta_ci,
            "paired_delta_median": _median(both_success_deltas),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="glob for baseline run directories")
    parser.add_argument("--flash", required=True, help="glob for FLASH run directories")
    parser.add_argument("--json-out", type=pathlib.Path)
    args = parser.parse_args()

    baseline = _load(args.baseline)
    flash = _load(args.flash)
    baseline_keys = set(baseline)
    flash_keys = set(flash)
    if baseline_keys != flash_keys:
        missing_baseline = sorted(flash_keys - baseline_keys)[:10]
        missing_flash = sorted(baseline_keys - flash_keys)[:10]
        raise ValueError(f"episode keys differ; missing_baseline={missing_baseline}, missing_flash={missing_flash}")

    by_suite: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    by_task: dict[tuple[str, int], list[tuple[str, int, int]]] = defaultdict(list)
    for key in sorted(baseline_keys):
        by_suite[key[0]].append(key)
        by_task[(key[0], key[1])].append(key)

    summary = {
        "baseline_pattern": args.baseline,
        "flash_pattern": args.flash,
        "paired_episode_count": len(baseline_keys),
        "overall": _summarize_pair(sorted(baseline_keys), baseline, flash),
        "by_suite": {suite: _summarize_pair(keys, baseline, flash) for suite, keys in sorted(by_suite.items())},
        "by_task": {
            f"{suite}/task{task_id}": _summarize_pair(keys, baseline, flash)
            for (suite, task_id), keys in sorted(by_task.items())
        },
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print("| suite | π0 成功率 | FLASH 成功率 | ΔFLASH (pp) | π0/FLASH-only | 双成功轨迹 π0→FLASH | 配对 Δ步数 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for suite, row in [*summary["by_suite"].items(), ("全部", summary["overall"])]:
        c = row["contingency"]
        t = row["trajectory_both_success"]
        print(
            f"| {_suite_label(suite)} | {100 * row['baseline_success_rate']:.1f}% "
            f"({row['baseline_successes']}/{row['episodes']}) | {100 * row['flash_success_rate']:.1f}% "
            f"({row['flash_successes']}/{row['episodes']}) | {row['success_delta_pp']:+.1f} | "
            f"{c['baseline_only']}/{c['flash_only']} | {_fmt(t['baseline_mean'])}→{_fmt(t['flash_mean'])} "
            f"| {_fmt(t['paired_delta_mean'])} |"
        )

    print("\n| suite/task | π0 | FLASH | ΔFLASH (pp) | both / π0-only / FLASH-only / neither |")
    print("| --- | --- | --- | --- | --- |")
    for name, row in summary["by_task"].items():
        c = row["contingency"]
        print(
            f"| {name} | {row['baseline_successes']}/{row['episodes']} | "
            f"{row['flash_successes']}/{row['episodes']} | {row['success_delta_pp']:+.1f} | "
            f"{c['both_success']} / {c['baseline_only']} / {c['flash_only']} / {c['both_failure']} |"
        )


if __name__ == "__main__":
    main()
