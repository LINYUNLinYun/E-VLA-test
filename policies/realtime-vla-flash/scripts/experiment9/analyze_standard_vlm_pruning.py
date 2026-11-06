#!/usr/bin/env python3
"""Summarize Experiment 9 success, paired outcomes, phases, and XYZ artifacts."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "experiment9_standard_vlm_pruning_20261004"
REPORT = ROOT / "experiment9_standard_vlm_pruning_report.md"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
CONDITIONS = ("Full18", "Skip-L2", "Skip-L3", "Skip-L4", "Skip-L5", "Skip-L2-5", "V13-Custom", "V16", "V14", "V12", "V10-Mid")
AE_CONDITIONS = ("AE-D7", "AE-D5", "AE-D3")
ALL_CONDITIONS = CONDITIONS + AE_CONDITIONS
STAGES = ("initial", "approach", "contact", "grasp_or_manipulate", "transport", "goal_alignment", "release_or_final_action", "success")


def suite_label(suite: str) -> str:
    """Return a short suite label using Python 3.8-compatible string handling."""
    prefix = "libero_"
    return suite[len(prefix):] if suite.startswith(prefix) else suite


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = columns or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def complete_chunk(path: Path, start: int, end: int) -> bool:
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        episodes = json.loads((path / "episode_log.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if (manifest.get("episode_index_start"), manifest.get("episode_index_end")) != (start, end):
        return False
    expected = {(task, index) for task in range(10) for index in range(start, end)}
    observed = [(int(row["task_id"]), int(row["episode_idx"])) for row in episodes]
    return len(observed) == len(expected) and set(observed) == expected


def collect(phase: str) -> tuple[dict[tuple[str, str, int, int], dict], dict[tuple[str, str, int, int], list[dict]]]:
    rows_by_key: dict[tuple[str, str, int, int], dict] = {}
    telemetry_by_key: dict[tuple[str, str, int, int], list[dict]] = {}
    for condition in ALL_CONDITIONS:
        slug = condition.replace("-", "_")
        for suite in SUITES:
            ranges = [(0, 10)] if phase.endswith("pilot") else [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
            for start, end in ranges:
                candidates = sorted(RESULTS.glob(f"exp9_*_{slug}_{suite}_i{start:02d}_{end:02d}_a*"), reverse=True)
                chunk = next((path for path in candidates if complete_chunk(path, start, end)), None)
                if chunk is None:
                    continue
                episodes = json.loads((chunk / "episode_log.json").read_text())
                for episode in episodes:
                    task, index = int(episode["task_id"]), int(episode["episode_idx"])
                    key = (condition, suite, task, index)
                    if key in rows_by_key:
                        continue
                    row = dict(episode)
                    row["run_dir"] = str(chunk)
                    row["condition"] = condition
                    row["suite"] = suite
                    infer_rel = row.get("infer_path")
                    if infer_rel:
                        infer_path = chunk / infer_rel
                        if infer_path.is_file():
                            infer_rows = [json.loads(line) for line in infer_path.read_text().splitlines() if line]
                            row["_policy_call_ms"] = [float(item["policy_time_ms"]) for item in infer_rows
                                                       if item.get("policy_time_ms") is not None]
                            row["_action_denoise_ms"] = [float(item["action_denoise_ms"]) for item in infer_rows
                                                          if item.get("action_denoise_ms") is not None]
                        else:
                            row["_policy_call_ms"], row["_action_denoise_ms"] = [], []
                    else:
                        row["_policy_call_ms"], row["_action_denoise_ms"] = [], []
                    rows_by_key[key] = row
                    telemetry_rel = row.get("telemetry_path")
                    if telemetry_rel:
                        path = chunk / telemetry_rel
                        if path.is_file():
                            telemetry_by_key[key] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return rows_by_key, telemetry_by_key


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def cluster_bootstrap(values: list[float], *, seed: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(array), size=(10_000, len(array)))
    estimates = array[samples].mean(axis=1)
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def mcnemar_exact(new_failures: int, recoveries: int) -> float:
    n = new_failures + recoveries
    if n == 0:
        return 1.0
    lower = min(new_failures, recoveries)
    tail = sum(math.comb(n, k) for k in range(lower + 1)) / (2**n)
    return min(1.0, 2 * tail)


def summarize(rows: dict, telemetry: dict, phase: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    suite_rows, task_rows, stage_rows, paired_rows = [], [], [], []
    for condition in ALL_CONDITIONS:
        for suite in SUITES:
            selected = [row for key, row in rows.items() if key[:2] == (condition, suite)]
            if not selected:
                continue
            successes = sum(bool(row["success"]) for row in selected)
            low, high = wilson(successes, len(selected))
            policy_times = np.asarray([value for row in selected for value in row.get("_policy_call_ms", [])], dtype=float)
            denoise_times = np.asarray([value for row in selected for value in row.get("_action_denoise_ms", [])], dtype=float)
            baseline_tasks, condition_tasks = {}, {}
            for task in range(10):
                task_selected = [row for row in selected if int(row["task_id"]) == task]
                if not task_selected:
                    continue
                task_successes = sum(bool(row["success"]) for row in task_selected)
                task_rate = task_successes / len(task_selected)
                condition_tasks[task] = task_rate
                task_rows.append({"phase": phase, "condition": condition, "suite": suite,
                                  "task_id": task, "successes": task_successes,
                                  "episodes": len(task_selected), "success_rate": task_rate})
            baseline_rows = [row for key, row in rows.items() if key[:2] == ("Full18", suite)]
            for task in range(10):
                group = [row for row in baseline_rows if int(row["task_id"]) == task]
                if group:
                    baseline_tasks[task] = sum(bool(row["success"]) for row in group) / len(group)
            paired_diff = [condition_tasks[task] - baseline_tasks[task]
                           for task in sorted(condition_tasks.keys() & baseline_tasks.keys())]
            boot_low, boot_high = cluster_bootstrap(paired_diff, seed=20261004 + ALL_CONDITIONS.index(condition) * 11 + SUITES.index(suite))
            common_keys = rows.keys()
            baseline_lookup = {(key[2], key[3]): bool(row["success"]) for key, row in rows.items()
                               if key[:2] == ("Full18", suite)}
            new_failures = recoveries = common_success = common_failure = 0
            for key in common_keys:
                if key[0] != condition or key[1] != suite:
                    continue
                base_success = baseline_lookup.get((key[2], key[3]))
                if base_success is None:
                    continue
                variant_success = bool(rows[key]["success"])
                if base_success and variant_success:
                    common_success += 1
                elif not base_success and not variant_success:
                    common_failure += 1
                elif base_success:
                    new_failures += 1
                else:
                    recoveries += 1
            if condition == "Full18":
                common_success = common_failure = new_failures = recoveries = None
                exact_p = None
            else:
                exact_p = mcnemar_exact(new_failures, recoveries)
            suite_rows.append({
                "phase": phase, "condition": condition, "suite": suite,
                "successes": successes, "episodes": len(selected), "success_rate": successes / len(selected),
                "wilson95_low": low, "wilson95_high": high,
                "policy_time_mean_p50_ms": float(np.percentile(policy_times, 50)) if policy_times.size else math.nan,
                "policy_time_mean_p95_ms": float(np.percentile(policy_times, 95)) if policy_times.size else math.nan,
                "action_denoise_p50_ms": float(np.percentile(denoise_times, 50)) if denoise_times.size else math.nan,
                "action_denoise_p95_ms": float(np.percentile(denoise_times, 95)) if denoise_times.size else math.nan,
                "delta_vs_full18_pp": (successes / len(selected) - sum(bool(x["success"]) for x in baseline_rows) / len(baseline_rows)) * 100 if baseline_rows else math.nan,
                "task_cluster_bootstrap_delta_pp_low": boot_low * 100,
                "task_cluster_bootstrap_delta_pp_high": boot_high * 100,
                "at_least_90pct_observed": successes / len(selected) >= .90,
                "wilson_lower_bound_over_90pct": low >= .90,
                "paired_both_success": common_success, "paired_both_failure": common_failure,
                "paired_new_failure": new_failures, "paired_recovery": recoveries,
                "paired_mcnemar_exact_p": exact_p,
            })

            rank = {name: index for index, name in enumerate(STAGES)}
            for stage_index, stage in enumerate(STAGES):
                reached = 0
                denominator = len(selected)
                for row in selected:
                    key = (condition, suite, int(row["task_id"]), int(row["episode_idx"]))
                    states = telemetry.get(key, [])
                    phases = [str(state.get("semantic_phase", "unknown")) for state in states]
                    if stage == "initial":
                        reached += bool(phases)
                    elif stage == "success":
                        reached += bool(row["success"])
                    else:
                        reached += any(phase == stage or (phase in rank and rank[phase] > stage_index) for phase in phases)
                previous = STAGES[stage_index - 1] if stage_index else None
                previous_reached = 0
                if previous:
                    for row in selected:
                        if previous == "success":
                            previous_reached += bool(row["success"])
                        else:
                            key = (condition, suite, int(row["task_id"]), int(row["episode_idx"]))
                            phases = [str(state.get("semantic_phase", "unknown")) for state in telemetry.get(key, [])]
                            previous_reached += any(phase == previous or (phase in rank and rank[phase] > rank[previous]) for phase in phases)
                conditional = reached / previous_reached if previous_reached else math.nan
                stage_rows.append({"phase": phase, "condition": condition, "suite": suite,
                                   "stage": stage, "episodes": denominator, "reached": reached,
                                   "reach_rate": reached / denominator if denominator else math.nan,
                                   "previous_stage": previous or "", "previous_reached": previous_reached,
                                   "conditional_transition_rate": conditional})
    for key, row in rows.items():
        if key[0] == "Full18":
            continue
        base = rows.get(("Full18", key[1], key[2], key[3]))
        if base is None:
            continue
        if bool(base["success"]) and not bool(row["success"]):
            outcome = "new_failure"
        elif not bool(base["success"]) and bool(row["success"]):
            outcome = "recovery"
        elif bool(base["success"]):
            outcome = "both_success"
        else:
            outcome = "both_failure"
        telemetry_rows = telemetry.get(key, [])
        furthest = row.get("furthest_semantic_phase", "unknown")
        paired_rows.append({"phase": phase, "condition": key[0], "suite": key[1],
                            "task_id": key[2], "episode_idx": key[3],
                            "outcome": outcome, "baseline_success": bool(base["success"]),
                            "variant_success": bool(row["success"]),
                            "baseline_furthest_phase": base.get("furthest_semantic_phase", "unknown"),
                            "variant_furthest_phase": furthest,
                            "telemetry_states": len(telemetry_rows)})
    return suite_rows, task_rows, stage_rows, paired_rows


def make_figures(suite_rows: list[dict], stage_rows: list[dict], phase: str) -> list[str]:
    output = RESULTS / "analysis" / phase
    output.mkdir(parents=True, exist_ok=True)
    available = {(row["condition"], row["suite"]): row["success_rate"] for row in suite_rows}
    matrix = np.array([[available.get((condition, suite), np.nan) for suite in SUITES] for condition in ALL_CONDITIONS])
    fig, ax = plt.subplots(figsize=(9.5, 7.6), layout="constrained")
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(SUITES)), [suite_label(suite) for suite in SUITES])
    ax.set_yticks(range(len(ALL_CONDITIONS)), ALL_CONDITIONS)
    ax.set_title(f"Standard LIBERO success rate ({phase})")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j] * 100:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Success rate")
    fig.savefig(output / "suite_success_heatmap.png", dpi=180)
    plt.close(fig)

    reach = [row for row in stage_rows if row.get("stage") in STAGES]
    if reach:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
        x = np.arange(len(STAGES))
        for suite, ax in zip(SUITES, axes.flat):
            for condition in ALL_CONDITIONS:
                values = [next((r["reach_rate"] for r in reach if r["suite"] == suite and r["condition"] == condition and r["stage"] == stage), np.nan) for stage in STAGES]
                if np.isfinite(values).any():
                    ax.plot(x, values, marker="o", linewidth=1.2, markersize=3, label=condition)
            ax.set_title(suite)
            ax.set_xticks(x, [stage.replace("_or_", "/").replace("_", "\n") for stage in STAGES], fontsize=7)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("Episode reach rate")
            ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Task stage reach rates ({phase}); later phases imply prior reach")
        fig.savefig(output / "stage_reach_by_suite.png", dpi=180)
        plt.close(fig)
    return [str(output / "suite_success_heatmap.png"), str(output / "stage_reach_by_suite.png")]


def select_xyz_pairs(rows: dict, telemetry: dict, phase: str) -> list[dict]:
    output = RESULTS / "analysis" / phase
    plots = output / "xyz_plots"
    videos = output / "xyz_videos"
    plots.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    selected = []
    python = str(ROOT / ".venv/bin/python")
    plot_script = ROOT / "scripts/experiment8/plot_paired_xyz.py"
    video_script = ROOT / "scripts/experiment8/render_xyz_video.py"
    for condition in ALL_CONDITIONS:
        if condition == "Full18":
            continue
        for suite in SUITES:
            candidate_groups: dict[str, list[tuple[float, int, int, tuple, tuple]]] = {}
            for key, row in rows.items():
                if key[:2] != (condition, suite):
                    continue
                base_key = ("Full18", suite, key[2], key[3])
                if base_key not in rows or key not in telemetry or base_key not in telemetry:
                    continue
                base_success, variant_success = bool(rows[base_key]["success"]), bool(row["success"])
                outcome = (
                    "both_success" if base_success and variant_success else
                    "new_failure" if base_success else
                    "recovery" if variant_success else "both_failure"
                )
                base_by_step = {int(state["step"]): np.asarray(state["eef_pos"], dtype=float)
                                for state in telemetry[base_key]}
                variant_by_step = {int(state["step"]): np.asarray(state["eef_pos"], dtype=float)
                                   for state in telemetry[key]}
                shared = base_by_step.keys() & variant_by_step.keys()
                if not shared:
                    continue
                max_deviation = max(float(np.linalg.norm(base_by_step[step] - variant_by_step[step]))
                                    for step in shared)
                candidate_groups.setdefault(outcome, []).append((-max_deviation, key[2], key[3], key, base_key))
            for outcome, candidates in sorted(candidate_groups.items()):
                _, _, _, key, base_key = min(candidates)
                variant_row, base_row = rows[key], rows[base_key]
                slug = f"{condition.replace('-', '_')}__{suite}__{outcome}__task{key[2]:02d}_ep{key[3]:02d}"
                baseline_tel = Path(base_row["run_dir"]) / base_row["telemetry_path"]
                variant_tel = Path(variant_row["run_dir"]) / variant_row["telemetry_path"]
                prefix = plots / slug
                subprocess.run([python, str(plot_script), str(baseline_tel), str(variant_tel),
                                "--baseline-label", "Full18 baseline", "--variant-label", condition,
                                "--output-prefix", str(prefix)], check=True, cwd=ROOT,
                                stdout=subprocess.DEVNULL)
                base_video = Path(base_row["run_dir"]) / base_row["video_path"]
                variant_video = Path(variant_row["run_dir"]) / variant_row["video_path"]
                video = videos / f"{slug}.mp4"
                subprocess.run([python, str(video_script),
                                "--baseline-video", str(base_video), "--variant-video", str(variant_video),
                                "--baseline-telemetry", str(baseline_tel), "--variant-telemetry", str(variant_tel),
                                "--baseline-label", "Full18 baseline", "--variant-label", condition,
                                "--alignment", "state-synchronous", "--output", str(video)],
                               check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
                summary = json.loads(prefix.with_suffix(".summary.json").read_text())
                selected.append({"condition": condition, "suite": suite, "task_id": key[2],
                                 "episode_idx": key[3], "paired_outcome": outcome,
                                 "mean_deviation_m": summary["paired_deviation_m"]["mean"],
                                 "p95_deviation_m": summary["paired_deviation_m"]["p95"],
                                 "max_deviation_m": summary["paired_deviation_m"]["maximum"],
                                 "plot": str(prefix.with_suffix(".png").relative_to(RESULTS)),
                                 "video": str(video.relative_to(RESULTS)),
                                 "baseline_success": bool(base_row["success"]),
                                 "variant_success": bool(variant_row["success"])})
    write_csv(output / "xyz_case_index.csv", selected)
    cards = []
    for row in selected:
        image_path = "../../" + row["plot"]
        video_path = "../../" + row["video"]
        cards.append(
            f"<article><h3>{html.escape(row['condition'])} · {html.escape(row['suite'])} "
            f"task {row['task_id']} / state {row['episode_idx']} ({html.escape(row['paired_outcome'])})</h3>"
            f"<img src='{html.escape(image_path)}'><video controls preload='metadata' src='{html.escape(video_path)}'></video>"
            f"<p>XYZ mean / p95 / max: {row['mean_deviation_m']:.3f} / "
            f"{row['p95_deviation_m']:.3f} / {row['max_deviation_m']:.3f} m</p></article>"
        )
    (output / "xyz_gallery.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Experiment 9 XYZ pairs</title>"
        "<style>body{font:15px sans-serif;margin:2rem}article{margin:2rem 0}img,video{width:min(900px,95vw);display:block;margin:.7rem 0}</style>"
        + "\n".join(cards), encoding="utf-8",
    )
    return selected


def write_report(phase: str, suite_rows: list[dict], task_rows: list[dict], stage_rows: list[dict], xyz_rows: list[dict]) -> None:
    phase_complete = phase in {"full", "ae_pilot", "ae_full"} and all(
        next((row["episodes"] for row in suite_rows if row["condition"] == condition and row["suite"] == suite), 0) == 500
        for condition in CONDITIONS for suite in SUITES
    )
    total_baseline = sum(row["successes"] for row in suite_rows if row["condition"] == "Full18")
    total_baseline_n = sum(row["episodes"] for row in suite_rows if row["condition"] == "Full18")
    lines = [
        "# Experiment 9: LIBERO 标准套件 VLM 裁剪实验报告", "",
        f"数据阶段：**{'四套件全量' if phase_complete else '初筛 / 当前可用数据'}**。"
        f" 已完成 Full18 baseline {total_baseline}/{total_baseline_n}。", "",
        "VLM 裁剪组固定 Action Expert Full18、D10；Action Expert 降噪组固定 VLM Full18。"
        "四套件各 10 任务。实际层索引、noise hash、denoise steps 和服务端配置保存在 inference 元数据中。", "",
        "V13-Custom 保留 VLM 层 L1/L2/L5/L6/L8/L9/L10/L11/L13/L14/L16/L17/L18，"
        "跳过 L3/L4/L7/L12/L15；命令使用相同 seed、D10、Action Expert Full18。", "",
        "## 套件成功率", "",
        "| 变体 | 套件 | 成功 / episodes | Wilson 95% | 相对 Full18 (pp) | 任务聚类配对差 95% | policy p50 / p95 ms | denoise p50 / p95 ms | ≥90% | 新失败 / 恢复 | McNemar p |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in suite_rows:
        if row["condition"] == "Full18":
            delta = "—"
            paired = "—"
            pvalue = "—"
            cohort = "—"
        else:
            delta = f"{row['delta_vs_full18_pp']:+.1f}"
            paired = f"[{row['task_cluster_bootstrap_delta_pp_low']:+.1f}, {row['task_cluster_bootstrap_delta_pp_high']:+.1f}]"
            pvalue = f"{row['paired_mcnemar_exact_p']:.4g}" if row["paired_mcnemar_exact_p"] is not None else "—"
            cohort = f"{row['paired_new_failure']} / {row['paired_recovery']}"
        latency = "—" if not np.isfinite(row["policy_time_mean_p50_ms"]) else f"{row['policy_time_mean_p50_ms']:.1f} / {row['policy_time_mean_p95_ms']:.1f}"
        denoise_latency = "—" if not np.isfinite(row["action_denoise_p50_ms"]) else f"{row['action_denoise_p50_ms']:.1f} / {row['action_denoise_p95_ms']:.1f}"
        lines.append(
            f"| {row['condition']} | {suite_label(row['suite'])} | "
            f"{row['successes']}/{row['episodes']} ({row['success_rate']:.1%}) | "
            f"[{row['wilson95_low']:.1%}, {row['wilson95_high']:.1%}] | {delta} | {paired} | {latency} | {denoise_latency} | "
            f"{'是' if row['at_least_90pct_observed'] else '否'} | "
            f"{cohort} | {pvalue} |"
        )
    lines += ["", "≥90% 是观测点估计阈值，不等价于达到 baseline 或通过非劣检验。"
              "配对差区间以任务为聚类单位；初筛只有每任务 10 个状态，区间仍较宽。", "",
              "![套件成功率热图](results/experiment9_standard_vlm_pruning_20261004/analysis/" + phase + "/suite_success_heatmap.png)", "",
              "## 失败阶段与转换", "",
              "阶段来自仿真器末端、物体位姿、接触/抓取状态及 goal predicate。reach 表示到达该阶段或更后阶段；"
              "conditional transition 表示已到达前一阶段的 episode 中继续到达本阶段的比例。", "",
              "| 变体 | 套件 | 阶段 | 到达数 / episode | 到达率 | 前一阶段条件转换率 |", "|---|---|---|---:|---:|---:|"]
    for row in stage_rows:
        if row.get("stage") not in STAGES:
            continue
        transition = row["conditional_transition_rate"]
        transition_label = "—" if not np.isfinite(transition) else f"{transition:.1%}"
        lines.append(f"| {row['condition']} | {suite_label(row['suite'])} | {row['stage']} | "
                     f"{row['reached']}/{row['episodes']} | {row['reach_rate']:.1%} | "
                     f"{transition_label} |")
    lines += ["", "![套件阶段到达率](results/experiment9_standard_vlm_pruning_20261004/analysis/" + phase + "/stage_reach_by_suite.png)", "",
              "## 配对 XYZ 图与视频", "",
              "蓝色实线和圆点为 Full18，橙色虚线和三角点为变体；真实 MuJoCo 世界坐标，单位米。"
              "阶段标注、xyz 同步偏差和 N+1 帧视频见画廊。", "",
              f"[打开 XYZ 轨迹与配对视频画廊](results/experiment9_standard_vlm_pruning_20261004/analysis/{phase}/xyz_gallery.html)", "",
              "| 变体 | 套件 | 案例 | 配对结果 | XYZ mean / p95 / max (m) | 图 | 视频 |", "|---|---|---|---|---:|---|---|"]
    for row in xyz_rows:
        relative_plot = "results/experiment9_standard_vlm_pruning_20261004/analysis/" + phase + "/" + row["plot"]
        relative_video = "results/experiment9_standard_vlm_pruning_20261004/analysis/" + phase + "/" + row["video"]
        lines.append(f"| {row['condition']} | {suite_label(row['suite'])} | task {row['task_id']} / state {row['episode_idx']} | "
                     f"{row['paired_outcome']} | {row['mean_deviation_m']:.3f} / {row['p95_deviation_m']:.3f} / {row['max_deviation_m']:.3f} | "
                     f"[图]({relative_plot}) | [视频]({relative_video}) |")
    lines += ["", "## Action Expert 降噪步数补充实验", "",
              "本节保持 VLM Full18、Action Expert Full18，对比 D7、D5、D3 与同状态/噪声的 D10 Full18 baseline。"
              "初筛成功率 ≥90% 或距 D10 baseline 不超过 2 个百分点的变体×套件扩展到每任务 50 个状态。"
              "推理时间列为 episode 级 policy latency；各次 denoise timing 与实际步数记录在 infer.jsonl。", "",
              "已有 Action Expert early-exit 层探针（200 个真实观测）见 [Experiment 1 分析](results/pi0_intermediate_probe/experiment1_report.md)。", "",
              "## 可复核数据", "",
              f"- [套件汇总 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/{phase}/suite_success.csv)",
              f"- [逐任务汇总 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/{phase}/task_success.csv)",
              f"- [阶段到达与配对 cohort CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/{phase}/failure_stage_analysis.csv)",
              f"- [配对失败与恢复 episode CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/{phase}/paired_failure_cohorts.csv)",
              f"- [XYZ 案例索引 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/{phase}/xyz_case_index.csv)",
              f"- [运行器事件日志](results/experiment9_standard_vlm_pruning_20261004/runner_events.jsonl)", "",
              "标准套件 Full18 的历史基线（2026-10-03）为 Spatial 95.4%、Object 99.2%、Goal 96.8%、LIBERO-10 86.8%。"
              "本轮配对对照以本轮重跑 Full18 为准，历史数据只作背景。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full", "ae_pilot", "ae_full"), required=True)
    args = parser.parse_args()
    rows, telemetry = collect(args.phase)
    suite_rows, task_rows, stage_rows, paired_rows = summarize(rows, telemetry, args.phase)
    output = RESULTS / "analysis" / args.phase
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "suite_success.csv", suite_rows)
    write_csv(output / "task_success.csv", task_rows)
    write_csv(output / "failure_stage_analysis.csv", stage_rows)
    write_csv(output / "paired_failure_cohorts.csv", paired_rows)
    figures = make_figures(suite_rows, stage_rows, args.phase)
    xyz_rows = select_xyz_pairs(rows, telemetry, args.phase)
    write_report(args.phase, suite_rows, task_rows, stage_rows, xyz_rows)
    summary = {"phase": args.phase, "episode_count": len(rows),
               "conditions_present": sorted({key[0] for key in rows}),
               "suite_rows": len(suite_rows), "telemetry_episodes": len(telemetry),
               "xyz_pairs": len(xyz_rows), "figures": figures,
               "report": str(REPORT)}
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
