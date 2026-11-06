#!/usr/bin/env python3
"""Summarize Realtime-VLA FLASH LIBERO client runs into per-task tables.

Reads every ``episode_log.json`` under the given run directories and prints:
  1. per-task success table (markdown, with an empty 备注 column to fill in by hand)
  2. failure list with rollout paths, for replaying failures one by one
  3. trajectory-length statistics split by success/failure
  4. latency summary that drops the first episode of each task (one-off first-round cost)

Usage:
    summarize_flash_runs.py data/test/goal_t00 data/test/goal_t01 ...
    summarize_flash_runs.py 'data/test/goal_t*'
    summarize_flash_runs.py                      # defaults to 'data/test/*'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "policies",
    "realtime-vla-flash",
    "data",
    "test",
)


def _load(pattern: str):
    """Return (episodes, run_names) from a run dir or glob pattern."""
    episodes, runs = [], []
    for path in sorted(glob.glob(pattern)):
        log = os.path.join(path, "episode_log.json")
        if not os.path.isfile(log):
            continue
        with open(log) as fh:
            records = json.load(fh)
        runs.append(os.path.basename(os.path.normpath(path)))
        for rec in records:
            rec["__run_dir__"] = os.path.normpath(path)
            episodes.append(rec)
    return episodes, runs


def _fmt_pct(hit: int, total: int) -> str:
    return "n/a" if not total else f"{100.0 * hit / total:.1f}%"


def _suite_label(suite: str) -> str:
    return {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object",
            "libero_goal": "LIBERO-Goal", "libero_10": "LIBERO-10"}.get(suite, suite)


def _trajectory_length(rec: dict) -> int:
    """Return policy-action trajectory length, excluding warm-up dummy steps."""
    if "trajectory_length" in rec:
        return int(rec["trajectory_length"])
    return int(rec["executed_action_count"])


def _fmt_lengths(values: list[int]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.fmean(values):.1f} / {statistics.median(values):.1f} ({min(values)}–{max(values)})"


def _point_biserial(lengths: list[int], successes: list[bool]) -> str:
    """Pearson r(length, success), equivalent to point-biserial correlation."""
    if len(lengths) < 2 or len(set(lengths)) < 2 or len(set(successes)) < 2:
        return "n/a"
    mean_x = statistics.fmean(lengths)
    mean_y = statistics.fmean(int(value) for value in successes)
    numerator = sum((x - mean_x) * (int(y) - mean_y) for x, y in zip(lengths, successes))
    denom_x = sum((x - mean_x) ** 2 for x in lengths)
    denom_y = sum((int(y) - mean_y) ** 2 for y in successes)
    return f"{numerator / (denom_x * denom_y) ** 0.5:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="run dirs or globs (default: data/test/*)")
    args = ap.parse_args()

    patterns = args.paths or [os.path.join(DEFAULT_ROOT, "*")]
    episodes: list = []
    runs: list = []
    for pat in patterns:
        eps, rn = _load(pat)
        episodes += eps
        runs += rn
    if not episodes:
        print("no episode_log.json found under: %s" % ", ".join(patterns), file=sys.stderr)
        return 1

    print(f"runs: {', '.join(runs)}")
    print(f"episodes: {len(episodes)}")

    by_task = defaultdict(list)
    for rec in episodes:
        by_task[(rec["task_suite_name"], int(rec["task_id"]))].append(rec)

    print("\n## 成功率（分 task）\n")
    print("| suite | task | 任务描述 | 成功/总数 | 成功率 | 备注 |")
    print("| --- | --- | --- | --- | --- | --- |")
    suite_hit, suite_tot = defaultdict(int), defaultdict(int)
    for (suite, task_id) in sorted(by_task):
        recs = sorted(by_task[(suite, task_id)], key=lambda r: r["episode_idx"])
        hit = sum(1 for r in recs if r["success"])
        total = len(recs)
        suite_hit[suite] += hit
        suite_tot[suite] += total
        print(f"| {_suite_label(suite)} | {task_id} | {recs[0]['task_description']} "
              f"| {hit}/{total} | {_fmt_pct(hit, total)} | |")

    print("\n| suite | 平均成功率 |")
    print("| --- | --- |")
    for suite in sorted(suite_tot):
        print(f"| {_suite_label(suite)} | {_fmt_pct(suite_hit[suite], suite_tot[suite])} "
              f"({suite_hit[suite]}/{suite_tot[suite]}) |")

    failures = [r for r in episodes if not r["success"]]
    print(f"\n## 失败案例清单（{len(failures)} 条，可直接回放 mp4）\n")
    if not failures:
        print("（无）")
    else:
        print("| suite | task | ep | mp4 | env_steps | infer | draft占比 | accepted长度 | 异常 |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for rec in sorted(failures, key=lambda r: (r["task_suite_name"], r["task_id"], r["episode_idx"])):
            route = (rec.get("route_ratio_by_infer") or {}).get("draft")
            print(f"| {_suite_label(rec['task_suite_name'])} | {rec['task_id']} | {rec['episode_idx']} "
                  f"| {os.path.join(rec['__run_dir__'], rec['video_path'])} "
                  f"| {rec['env_steps_taken']} | {rec['infer_calls']} "
                  f"| {'n/a' if route is None else f'{route:.2f}'} "
                  f"| {rec.get('accepted_action_len_mean') or 0:.1f} "
                  f"| {rec.get('failure_reason') or ''} |")

    print("\n## 轨迹长度与任务成败\n")
    print("> `trajectory_length` 是实际执行的策略动作数，不含 `num_steps_wait` 的 warm-up dummy actions。失败 episode 通常运行至 suite 的最大步数，因此相关性描述的是终止行为，不应直接解释为因果。\n")
    print("| suite | 成功轨迹 mean / median (range) | 失败轨迹 mean / median (range) | r(length, success) |")
    print("| --- | --- | --- | --- |")
    trajectory_groups = defaultdict(list)
    for rec in episodes:
        trajectory_groups[rec["task_suite_name"]].append(rec)
    for suite in sorted(trajectory_groups):
        recs = trajectory_groups[suite]
        success_lengths = [_trajectory_length(r) for r in recs if r["success"]]
        failure_lengths = [_trajectory_length(r) for r in recs if not r["success"]]
        lengths = [_trajectory_length(r) for r in recs]
        successes = [bool(r["success"]) for r in recs]
        print(f"| {_suite_label(suite)} | {_fmt_lengths(success_lengths)} "
              f"| {_fmt_lengths(failure_lengths)} | {_point_biserial(lengths, successes)} |")
    success_lengths = [_trajectory_length(r) for r in episodes if r["success"]]
    failure_lengths = [_trajectory_length(r) for r in episodes if not r["success"]]
    print(f"| **全部** | **{_fmt_lengths(success_lengths)}** | **{_fmt_lengths(failure_lengths)}** "
          f"| **{_point_biserial([_trajectory_length(r) for r in episodes], [bool(r['success']) for r in episodes])}** |")

    # 每个 task 恰好有 1 集（一定是 episode_idx==0）的第一轮带一笔 5–15 s 的一次性成本，足以把该 task
    # 的延迟均值抬高，故剔除首集。成因已定位：Triton 运行时池按 prompt 的 token 长度缓存 runtime
    # (triton_pi0_runtime.py:1782 `_runtime_by_prompt_len`)，且 get_runtime 在查表前先 _evict_stale_runtimes
    # (:1853) 把长度不同的全删 —— 是单槽，见过的长度也不会复用；长度一变就新建 + 重做 CUDA graph capture
    # (同文件 1687-1689)。判据是"与该 task 上一个 token 长度是否相同"，不是新 prompt / 缓存未命中。
    # 注意 `did_prefill` 既不是路径以外的信息（与 route_type=="full" 完全同进同出），也预测不了这笔成本。
    print("\n## 延迟（已排除每个 task 的首集）\n")
    print("> 注意字段名陷阱：`episode_log.json` 的 `infer_latency_mean_ms` **不是客户端往返**，它是 `sample_actions_ms`（服务端采样耗时，"
          "见 `spec_client_libero.py:191`）；真正的客户端往返在每集的 `infer.jsonl: client_roundtrip_ms`，按轮统计口径见 `scripts/flash_fr_acc.py`。\n")
    print("| suite | 服务端 sample_actions_ms 按集 mean/median (ms) | 服务端 serve_time_ms 按集 mean (ms) | 服务端每动作 mean (ms) | 样本 |")
    print("| --- | --- | --- | --- | --- |")
    per_suite = defaultdict(list)
    for (suite, task_id), recs in by_task.items():
        recs = sorted(recs, key=lambda r: r["episode_idx"])
        per_suite[suite] += recs[1:]
    for suite in sorted(per_suite):
        vals = [r["infer_latency_mean_ms"] for r in per_suite[suite] if r.get("infer_latency_mean_ms") is not None]
        serve = [r["serve_time_mean_ms"] for r in per_suite[suite] if r.get("serve_time_mean_ms") is not None]
        per_act = [r["avg_latency_per_action_ms"] for r in per_suite[suite]
                   if r.get("avg_latency_per_action_ms") is not None]
        if not vals:
            continue
        print(f"| {_suite_label(suite)} | {statistics.fmean(vals):.1f} / {statistics.median(vals):.1f} "
              f"| {'n/a' if not serve else f'{statistics.fmean(serve):.1f}'} "
              f"| {'n/a' if not per_act else f'{statistics.fmean(per_act):.2f}'} | {len(vals)} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
