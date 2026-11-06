#!/usr/bin/env python3
"""Diffusion Policy 单次 forward 的 nsys 瓶颈分析。

分析范围严格限定在 NVTX 的 `policy_infer` 区间（10 次真实推理），不把程序启动、
checkpoint 加载、warmup 算进来。内部再把 `denoising_step`（1000 次）单独拆开。

产物（默认写到 profile_results/diffusion-policy/）：
    summary.txt            —— 人读的汇总
    kernel_breakdown.csv   —— 按 kernel name 聚合
    denoising_steps.csv    —— 逐去噪 step
    gpu_metrics.csv        —— 仅当报告里含 GPU Metrics 时才生成

用法：
    python3 profiling/analyze_diffusion_policy.py
    python3 profiling/analyze_diffusion_policy.py <rep|sqlite> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# CUDA memcpy 的 copyKind 取值，见 ENUM_CUDA_MEMCPY_OPER。
MEMCPY_KINDS = {
    1: "HtoD",
    2: "DtoH",
    3: "HtoA",
    4: "AtoH",
    5: "AtoA",
    6: "AtoD",
    7: "DtoA",
    8: "DtoD",
    9: "HtoH",
    10: "PtoP",
}

GRANULARITY_BUCKETS = [(0, 5), (5, 20), (20, 100), (100, 500), (500, float("inf"))]
CONV_KERNEL = "implicit_convolve_sgemm"


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def percentile(ordered: list[float], fraction: float) -> float | None:
    """最近秩百分位，与 profiling/common.py 的 timing_summary 保持一致。"""
    if not ordered:
        return None
    return ordered[round((len(ordered) - 1) * fraction)]


def stats(values_ms: list[float]) -> dict[str, float | int | None]:
    if not values_ms:
        return {"count": 0, "total_ms": 0.0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values_ms)
    return {
        "count": len(ordered),
        "total_ms": sum(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "max_ms": ordered[-1],
    }


def ensure_sqlite(path: Path) -> Path:
    if path.suffix == ".sqlite":
        return path
    sqlite_path = path.with_suffix(".sqlite")
    if not sqlite_path.exists() or sqlite_path.stat().st_mtime < path.stat().st_mtime:
        subprocess.run(
            [
                "nsys", "export", "--type=sqlite", "--force-overwrite=true",
                "--output", str(sqlite_path), str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return sqlite_path


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠区间，返回按 start 排序的不相交区间。"""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def covered_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in merge_intervals(intervals))


def clip(intervals: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    return [(max(start, s), min(end, e)) for s, e in intervals if e > start and s < end]


# --------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------


def load_ranges(connection: sqlite3.Connection, name: str) -> list[tuple[int, int]]:
    return connection.execute(
        """
        SELECT e.start, e.end FROM NVTX_EVENTS e
        LEFT JOIN StringIds s ON s.id = e.textId
        WHERE e.end IS NOT NULL AND COALESCE(e.text, s.value) = ?
        ORDER BY e.start
        """,
        (name,),
    ).fetchall()


class Report:
    def __init__(self, sqlite_path: Path, scope: list[tuple[int, int]], denoising: list[tuple[int, int]]):
        self.connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        self.scope = scope
        self.denoising = denoising
        self.scope_start = min(s for s, _ in scope)
        self.scope_end = max(e for _, e in scope)
        self.tables = {
            row[0]
            for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    def close(self) -> None:
        self.connection.close()

    @property
    def wall_ns(self) -> int:
        return sum(e - s for s, e in self.scope)

    def kernels(self) -> list[tuple[str, str, int, int]]:
        """范围内所有 kernel 实例：(shortName, demangledName, start, end)。

        只取 start 落在范围内的实例；跨边界的实例按实际重叠裁剪。
        """
        rows = self.connection.execute(
            """
            SELECT COALESCE(ss.value, '?'), COALESCE(ds.value, '?'), k.start, k.end
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds ss ON ss.id = k.shortName
            LEFT JOIN StringIds ds ON ds.id = k.demangledName
            WHERE k.end IS NOT NULL AND k.start < ? AND k.end > ?
            ORDER BY k.start
            """,
            (self.scope_end, self.scope_start),
        ).fetchall()
        out = []
        for short, demangled, start, end in rows:
            for range_start, range_end in self.scope:
                if end <= range_start or start >= range_end:
                    continue
                out.append((short, demangled, max(start, range_start), min(end, range_end)))
        out.sort(key=lambda item: item[2])
        return out

    def api_calls(self) -> list[tuple[str, int, int]]:
        rows = self.connection.execute(
            """
            SELECT COALESCE(s.value, '?'), r.start, r.end
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            LEFT JOIN StringIds s ON s.id = r.nameId
            WHERE r.end IS NOT NULL AND r.start < ? AND r.end > ?
            """,
            (self.scope_end, self.scope_start),
        ).fetchall()
        out = []
        for name, start, end in rows:
            for range_start, range_end in self.scope:
                if end <= range_start or start >= range_end:
                    continue
                out.append((name, max(start, range_start), min(end, range_end)))
        return out

    def memory_ops(self) -> tuple[list[tuple[str, int, int, int]], list[tuple[int, int]]]:
        memcpy = []
        for kind, start, end, nbytes in self.connection.execute(
            """
            SELECT copyKind, start, end, COALESCE(bytes, 0)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE end IS NOT NULL AND start < ? AND end > ?
            """,
            (self.scope_end, self.scope_start),
        ):
            for range_start, range_end in self.scope:
                if end <= range_start or start >= range_end:
                    continue
                memcpy.append(
                    (MEMCPY_KINDS.get(kind, f"kind{kind}"), max(start, range_start), min(end, range_end), nbytes)
                )
        memset = []
        for start, end in self.connection.execute(
            """
            SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET
            WHERE end IS NOT NULL AND start < ? AND end > ?
            """,
            (self.scope_end, self.scope_start),
        ):
            for range_start, range_end in self.scope:
                if end <= range_start or start >= range_end:
                    continue
                memset.append((max(start, range_start), min(end, range_end)))
        return memcpy, memset

    def has_gpu_metrics(self) -> bool:
        return any("GPU_METRICS" in table.upper() for table in self.tables)


# --------------------------------------------------------------------------
# 各节分析
# --------------------------------------------------------------------------


def forward_overview(report: Report, kernels, api, memcpy, memset) -> dict:
    wall_ns = report.wall_ns
    gpu_busy_ns = covered_ns([(s, e) for _, _, s, e in kernels])
    api_total_ns = sum(e - s for _, s, e in api)
    api_busy_ns = covered_ns([(s, e) for _, s, e in api])
    by_kind: dict[str, dict] = {}
    for kind, start, end, nbytes in memcpy:
        entry = by_kind.setdefault(kind, {"count": 0, "bytes": 0, "ns": 0, "intervals": []})
        entry["count"] += 1
        entry["bytes"] += nbytes
        entry["ns"] += end - start
        entry["intervals"].append((start, end))
    for entry in by_kind.values():
        entry["busy_ns"] = covered_ns(entry.pop("intervals"))
    memset_ns = sum(e - s for s, e in memset)

    return {
        "wall_ns": wall_ns,
        "gpu_busy_ns": gpu_busy_ns,
        "gpu_busy_ratio": gpu_busy_ns / wall_ns if wall_ns else 0.0,
        "gap_ns": wall_ns - gpu_busy_ns,
        "api_total_ns": api_total_ns,
        "api_busy_ns": api_busy_ns,
        "memcpy": by_kind,
        "memset_count": len(memset),
        "memset_ns": memset_ns,
        "memcpy_ns": sum(entry["ns"] for entry in by_kind.values()),
    }


def kernel_breakdown(kernels) -> list[dict]:
    grouped: dict[str, list[float]] = {}
    demangled: dict[str, set] = {}
    for short, demangled_name, start, end in kernels:
        grouped.setdefault(short, []).append((end - start) / 1e6)
        demangled.setdefault(short, set()).add(demangled_name)

    total_ns = sum(sum(values) for values in grouped.values()) * 1e6
    rows = []
    for name, values in grouped.items():
        entry = stats(values)
        entry["kernel"] = name
        entry["pct_of_kernel_time"] = (entry["total_ms"] / (total_ns / 1e6) * 100) if total_ns else 0.0
        entry["variants"] = len(demangled[name])
        rows.append(entry)
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows


def conv_variants(kernels) -> list[dict]:
    """implicit_convolve_sgemm 的 demangled 模板变体，用于和 nsys stats 对账。"""
    grouped: dict[str, list[float]] = {}
    for short, demangled_name, start, end in kernels:
        if short != CONV_KERNEL:
            continue
        grouped.setdefault(demangled_name, []).append((end - start) / 1e6)
    rows = []
    for name, values in grouped.items():
        entry = stats(values)
        entry["variant"] = name
        rows.append(entry)
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows


def denoising_analysis(report: Report, kernels) -> list[dict]:
    """逐去噪 step 统计，附带落在该 step 区间内的 kernel 与卷积占比。"""
    # kernel 已按 start 排序，用游标线性推进即可，不必对 1000 个区间各扫一遍。
    ordered = sorted(kernels, key=lambda item: item[2])
    cursor = 0
    rows = []
    for index, (step_start, step_end) in enumerate(report.denoising):
        while cursor < len(ordered) and ordered[cursor][3] <= step_start:
            cursor += 1
        scan = cursor
        kernel_count = 0
        kernel_ns = 0
        conv_calls = 0
        conv_ns = 0
        while scan < len(ordered) and ordered[scan][2] < step_end:
            # 元组是 (shortName, demangledName, start, end)，卷积判断要用 shortName。
            name, _, start, end = ordered[scan]
            overlap = min(end, step_end) - max(start, step_start)
            if overlap > 0:
                kernel_count += 1
                kernel_ns += overlap
                if name == CONV_KERNEL:
                    conv_calls += 1
                    conv_ns += overlap
            scan += 1
        wall_ns = step_end - step_start
        rows.append(
            {
                "step": index,
                "step_index_in_call": index % 100,
                "call": index // 100,
                "wall_ms": wall_ns / 1e6,
                "gpu_busy_ms": kernel_ns / 1e6,
                "gpu_busy_ratio": kernel_ns / wall_ns if wall_ns else 0.0,
                "kernels": kernel_count,
                "conv_calls": conv_calls,
                "conv_ms": conv_ns / 1e6,
                "conv_pct_of_step_kernel_time": (conv_ns / kernel_ns * 100) if kernel_ns else 0.0,
            }
        )
    return rows


def granularity(kernels) -> tuple[list[dict], dict]:
    # 分桶边界是微秒，逐实例时长必须换算成微秒再比较。
    durations_us = [(end - start) / 1e3 for _, _, start, end in kernels]
    total_us = sum(durations_us)
    rows = []
    for low, high in GRANULARITY_BUCKETS:
        selected = [d for d in durations_us if low <= d < high]
        label = f"<{high:g}us" if low == 0 else (f">{low:g}us" if high == float("inf") else f"{low:g}-{high:g}us")
        rows.append(
            {
                "bucket": label,
                "count": len(selected),
                "pct": (len(selected) / len(durations_us) * 100) if durations_us else 0.0,
                "total_ms": sum(selected) / 1e3,
                "pct_time": (sum(selected) / total_us * 100) if total_us else 0.0,
            }
        )
    return rows, stats([d / 1e3 for d in durations_us])


def kernel_gaps(report: Report, kernels) -> dict:
    """逐个 policy_infer 区间内，相邻 kernel 之间的空隙。"""
    gaps_ms: list[float] = []
    head_ms = 0.0
    tail_ms = 0.0
    by_start = sorted(kernels, key=lambda item: item[2])
    for range_start, range_end in report.scope:
        inside = [
            (max(s, range_start), min(e, range_end))
            for _, _, s, e in by_start
            if e > range_start and s < range_end
        ]
        merged = merge_intervals(inside)
        if not merged:
            continue
        head_ms += (merged[0][0] - range_start) / 1e6
        tail_ms += (range_end - merged[-1][1]) / 1e6
        for previous, current in zip(merged, merged[1:]):
            gaps_ms.append((current[0] - previous[1]) / 1e6)
    entry = stats(gaps_ms)
    entry["head_ms"] = head_ms
    entry["tail_ms"] = tail_ms
    return entry


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value, unit="ms", digits=3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f} {unit}"


def build_summary(
    report: Report,
    overview: dict,
    breakdown: list[dict],
    variants: list[dict],
    steps: list[dict],
    buckets: list[dict],
    kernel_stats: dict,
    gap: dict,
    calls: int,
) -> str:
    wall_ms = overview["wall_ns"] / 1e6
    gpu_ms = overview["gpu_busy_ns"] / 1e6
    conv = next((row for row in breakdown if row["kernel"] == CONV_KERNEL), None)
    step_wall = stats([row["wall_ms"] for row in steps])
    step_kernels = stats([float(row["kernels"]) for row in steps])
    step_conv_calls = stats([float(row["conv_calls"]) for row in steps])
    step_conv_ms = stats([row["conv_ms"] for row in steps])
    step_conv_pct = stats([row["conv_pct_of_step_kernel_time"] for row in steps])

    lines: list[str] = []
    add = lines.append
    add("Diffusion Policy — Nsight Systems 单次 forward 瓶颈分析")
    add("=" * 72)
    add(f"报告       : {report.sqlite_path.name if hasattr(report, 'sqlite_path') else 'diffusion-policy.nsys-rep'}")
    add(f"分析范围   : NVTX `policy_infer` x {calls} 次（不含启动 / 模型加载 / warmup）")
    add(f"内部拆分   : NVTX `denoising_step` x {len(steps)} 次")
    add(f"GPU Metrics: {'有' if report.has_gpu_metrics() else '无（本次采集未带 --gpu-metrics-device）'}")
    add("")
    add("1. Forward overview")
    add("-" * 72)
    add(f"  forward wall time          : {fmt(wall_ms)}  ({calls} 次合计, 均值 {fmt(wall_ms / calls)})")
    add(f"  GPU kernel covered time    : {fmt(gpu_ms)}")
    add(f"  GPU busy ratio             : {overview['gpu_busy_ratio'] * 100:.2f} %")
    add(f"  GPU kernel gap / idle time : {fmt(overview['gap_ns'] / 1e6)}  ({(1 - overview['gpu_busy_ratio']) * 100:.2f} %)")
    add(f"  CUDA API time (sum)        : {fmt(overview['api_total_ns'] / 1e6)}")
    add(f"  CUDA API busy (union)      : {fmt(overview['api_busy_ns'] / 1e6)}")
    add("  Memory operations:")
    for kind in ("HtoD", "DtoH", "DtoD", "AtoA", "HtoA", "AtoH", "DtoA", "AtoD", "HtoH", "PtoP"):
        entry = overview["memcpy"].get(kind)
        if not entry:
            continue
        add(
            f"    {kind:<5} count={entry['count']:<6} bytes={entry['bytes']:<12} "
            f"time={fmt(entry['ns'] / 1e6)} (busy {fmt(entry['busy_ns'] / 1e6)})"
        )
    add(f"    Memset count={overview['memset_count']:<6} {'':<12} time={fmt(overview['memset_ns'] / 1e6)}")
    add("")
    add("2. Kernel breakdown (top 15, 按 shortName 聚合)")
    add("-" * 72)
    add(f"  {'kernel':<44}{'total_ms':>11}{'%':>7}{'calls':>8}{'avg_us':>9}{'p50_us':>9}{'p95_us':>9}{'max_us':>9}")
    for row in breakdown[:15]:
        add(
            f"  {row['kernel'][:43]:<44}{row['total_ms']:>11.1f}{row['pct_of_kernel_time']:>7.1f}"
            f"{row['count']:>8}{row['mean_ms'] * 1000:>9.1f}{row['p50_ms'] * 1000:>9.1f}"
            f"{row['p95_ms'] * 1000:>9.1f}{row['max_ms'] * 1000:>9.1f}"
        )
    add("")
    if conv:
        add(f"  implicit_convolve_sgemm（重点）")
        add(f"    total      : {fmt(conv['total_ms'])}  = {conv['pct_of_kernel_time']:.2f}% of GPU kernel time")
        add(f"    calls      : {conv['count']}  (avg {conv['mean_ms'] * 1000:.1f} us, p50 {conv['p50_ms'] * 1000:.1f} us, "
            f"p95 {conv['p95_ms'] * 1000:.1f} us, max {conv['max_ms'] * 1000:.1f} us)")
        add(f"    per forward: {conv['count'] / calls:.0f} calls, {conv['total_ms'] / calls:.1f} ms")
        add(f"    模板变体   : {len(variants)} 个")
        for row in variants:
            add(f"      {row['count']:>6} calls  {row['total_ms']:>9.1f} ms  {row['variant'][:96]}")
    add("")
    add("3. Denoising step")
    add("-" * 72)
    add(f"  denoising steps            : {len(steps)}  ({len(steps) // calls} steps x {calls} calls)")
    add(f"  step latency avg/p50/p95   : {fmt(step_wall['mean_ms'])} / {fmt(step_wall['p50_ms'])} / {fmt(step_wall['p95_ms'])}")
    add(f"  step latency min/max       : {fmt(min(r['wall_ms'] for r in steps))} / {fmt(step_wall['max_ms'])}")
    add(f"  kernels per step           : avg {step_kernels['mean_ms']:.1f}, p50 {step_kernels['p50_ms']:.0f}, "
        f"p95 {step_kernels['p95_ms']:.0f}")
    add(f"  conv calls per step        : avg {step_conv_calls['mean_ms']:.1f}, p50 {step_conv_calls['p50_ms']:.0f}, "
        f"p95 {step_conv_calls['p95_ms']:.0f}")
    add(f"  conv time per step         : avg {fmt(step_conv_ms['mean_ms'])}, p50 {fmt(step_conv_ms['p50_ms'])}, "
        f"p95 {fmt(step_conv_ms['p95_ms'])}")
    add(f"  conv % of step kernel time : avg {step_conv_pct['mean_ms']:.2f} %, p50 {step_conv_pct['p50_ms']:.2f} %, "
        f"p95 {step_conv_pct['p95_ms']:.2f} %")
    total_step_ms = step_wall["total_ms"]
    add(f"  denoising 占 forward 比例  : {total_step_ms / wall_ms * 100:.2f} %")
    add("")
    add("4. Kernel granularity / launch overhead")
    add("-" * 72)
    add(f"  kernel instances           : {kernel_stats['count']}  ({kernel_stats['count'] / calls:.0f} per forward)")
    add(f"  kernel duration            : avg {fmt(kernel_stats['mean_ms'])}  p50 {fmt(kernel_stats['p50_ms'])}"
        f"  p95 {fmt(kernel_stats['p95_ms'])}  max {fmt(kernel_stats['max_ms'])}")
    add(f"  {'bucket':<12}{'count':>10}{'%':>8}{'total_ms':>12}{'%time':>8}")
    for row in buckets:
        add(f"  {row['bucket']:<12}{row['count']:>10}{row['pct']:>8.1f}{row['total_ms']:>12.1f}{row['pct_time']:>8.1f}")
    add(f"  adjacent kernel gap        : avg {fmt(gap['mean_ms'])}  p50 {fmt(gap['p50_ms'])}  p95 {fmt(gap['p95_ms'])}"
        f"  max {fmt(gap['max_ms'])}")
    add(f"  gap intervals              : {gap['count']}")
    add(f"  total gap time             : {fmt((overview['gap_ns']) / 1e6)}"
        f"  (区间头 {fmt(gap['head_ms'])} + 区间尾 {fmt(gap['tail_ms'])} + 相邻间隙 {fmt(gap['total_ms'])})")
    add(f"  kernel gap ratio           : {(1 - overview['gpu_busy_ratio']) * 100:.2f} %")
    # 下发吞吐：CPU 每秒能推多少个 kernel。低于 GPU 消化速度就是 CPU 下发受限。
    launch_rate = kernel_stats["count"] / (wall_ms / 1000) if wall_ms else 0.0
    add(f"  host launch rate           : {launch_rate:.0f} kernels/s"
        f"  (GPU 平均每个 kernel 占用 {kernel_stats['mean_ms'] * 1000:.1f} us)")
    add("")
    add("5. GPU hardware metrics")
    add("-" * 72)
    if report.has_gpu_metrics():
        add("  见 gpu_metrics.csv")
    else:
        add("  该 .nsys-rep 不含 GPU Metrics（采集命令未使用 --gpu-metrics-device）。")
        add("  因此不做 compute-bound / memory-bound 的判定。")
    add("")
    add("6. Memory operations")
    add("-" * 72)
    add("  见第 1 节的 Memory operations。注意这些只是显式 CUDA memcpy，")
    add("  不能代表 kernel 内部的 HBM traffic。")
    add("")
    add("7. Summary")
    add("-" * 72)
    add(f"  Forward latency            : {fmt(wall_ms / calls)}")
    add(f"  GPU busy ratio             : {overview['gpu_busy_ratio'] * 100:.2f} %")
    add(f"  Conv/GEMM percentage       : {conv['pct_of_kernel_time'] if conv else 0:.2f} %")
    add(f"  implicit_convolve_sgemm    : {conv['pct_of_kernel_time'] if conv else 0:.2f} %")
    add(f"  Conv calls                 : {conv['count'] if conv else 0}")
    add(f"  Denoising steps            : {len(steps)}")
    add(f"  Conv calls / denoising step: {step_conv_calls['mean_ms']:.1f}")
    add(f"  Average denoising latency  : {fmt(step_wall['mean_ms'])}")
    add("  SM Active                  : n/a (报告不含 GPU Metrics)")
    add("  SM Issue                   : n/a (报告不含 GPU Metrics)")
    add("  Tensor Active              : n/a (报告不含 GPU Metrics)")
    add("  DRAM bandwidth             : n/a (报告不含 GPU Metrics)")
    add(f"  Kernel gap ratio           : {(1 - overview['gpu_busy_ratio']) * 100:.2f} %")
    add("")
    add("  倾向性判断：")
    add("    当前 NSYS report 不足以做严格的 compute-vs-memory 判定：")
    add('    "Current NSYS report is insufficient for strict compute-vs-memory')
    add('     classification; Nsight Compute hardware counters are required."')
    add("")
    add("    仅凭 kernel 时序可以确定的部分：")
    add(f"      - kernel gap ratio {(1 - overview['gpu_busy_ratio']) * 100:.2f}%，"
        f"相邻 kernel 间隙 avg {gap['mean_ms']:.2f} us / p95 {gap['p95_ms']:.2f} us，")
    add("        与 D（launch / CPU 调度开销）不相符。" if (gap["p95_ms"] or 0) < 50 else
        "        需要结合 D（launch / CPU 调度开销）进一步确认。")
    add(f"      - GPU kernel 连续执行且占满 {overview['gpu_busy_ratio'] * 100:.2f}% 的 wall time，")
    add(f"        其中 {conv['pct_of_kernel_time'] if conv else 0:.2f}% 是 implicit_convolve_sgemm 卷积。")
    add("      - 要区分 A（compute 饱和）/ B（带宽饱和）/ C（尺寸不利、占用率低），")
    add("        必须用 Nsight Compute 采集 SM / DRAM 计数器。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", type=Path,
                        default=ROOT / "profile_results" / "diffusion-policy" / "diffusion-policy.nsys-rep")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "profile_results" / "diffusion-policy")
    args = parser.parse_args()

    sqlite_path = ensure_sqlite(args.report)
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    scope = load_ranges(connection, "policy_infer")
    denoising = load_ranges(connection, "denoising_step")
    connection.close()
    if not scope:
        print("没有找到 policy_infer 区间。", file=sys.stderr)
        return 1

    report = Report(sqlite_path, scope, denoising)
    report.sqlite_path = sqlite_path  # type: ignore[attr-defined]
    try:
        kernels = report.kernels()
        api = report.api_calls()
        memcpy, memset = report.memory_ops()
        overview = forward_overview(report, kernels, api, memcpy, memset)
        breakdown = kernel_breakdown(kernels)
        variants = conv_variants(kernels)
        steps = denoising_analysis(report, kernels)
        buckets, kernel_stats = granularity(kernels)
        gap = kernel_gaps(report, kernels)
        has_metrics = report.has_gpu_metrics()
    finally:
        report.close()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / "kernel_breakdown.csv",
        breakdown,
        ["kernel", "total_ms", "pct_of_kernel_time", "count", "mean_ms", "p50_ms", "p95_ms", "max_ms", "variants"],
    )
    write_csv(
        out / "denoising_steps.csv",
        steps,
        ["step", "call", "step_index_in_call", "wall_ms", "gpu_busy_ms", "gpu_busy_ratio",
         "kernels", "conv_calls", "conv_ms", "conv_pct_of_step_kernel_time"],
    )
    if has_metrics:
        print("检测到 GPU Metrics 表，但本脚本尚未解析，请用 nsys stats --report gpumetrics_* 导出。", file=sys.stderr)

    summary = build_summary(
        report, overview, breakdown, variants, steps, buckets, kernel_stats, gap, len(scope)
    )
    (out / "summary.txt").write_text(summary)
    print(summary)
    print(f"wrote: {out/'summary.txt'}")
    print(f"wrote: {out/'kernel_breakdown.csv'}")
    print(f"wrote: {out/'denoising_steps.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
