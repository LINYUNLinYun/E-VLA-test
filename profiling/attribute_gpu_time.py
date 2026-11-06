#!/usr/bin/env python3
"""把 GPU kernel 时间归属到 NVTX 区间，量化每个阶段真正占用了多少设备时间。

NVTX 区间是墙钟时间，里面可能包含 CPU 下发开销、GPU 空转、以及别的阶段的 kernel。
只看 nsys 时间线上的区间宽度容易得出错误结论：一个"耗时 57 ms"的区间里可能只有
12 ms 是真的在算。本脚本用 kernel 的起止时间戳与区间求交，给出每个区间的：

  wall_ms   区间墙钟总时长（nsys nvtx_pushpop_sum 的口径）
  gpu_ms    落在区间内的 kernel 时长之和
  busy       gpu_ms / wall_ms，越低说明该阶段越是 CPU/下发受限

用法：
  python profiling/attribute_gpu_time.py profile_results/openvla-oft/openvla-oft.sqlite
  python profiling/attribute_gpu_time.py <rep|sqlite> --range policy_infer --depth 3
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path


def ensure_sqlite(path: Path) -> Path:
    if path.suffix == ".sqlite":
        return path
    sqlite_path = path.with_suffix(".sqlite")
    if not sqlite_path.exists() or sqlite_path.stat().st_mtime < path.stat().st_mtime:
        subprocess.run(
            ["nsys", "export", "--type=sqlite", "--force-overwrite=true", "--output", str(sqlite_path), str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return sqlite_path


def load_ranges(connection: sqlite3.Connection) -> list[tuple[str, int, int]]:
    """返回 [(name, start, end)]，按 start 排序。

    NVTX_EVENTS 没有 parentId 列，嵌套关系是靠区间包含隐含表达的；这里不需要
    显式层级，归属计算按时间包含即可（父区间的 gpu_ms 天然包含子区间的 kernel）。
    """
    return connection.execute(
        """
        SELECT COALESCE(e.text, s.value, '') AS name, e.start, e.end
        FROM NVTX_EVENTS e
        LEFT JOIN StringIds s ON s.id = e.textId
        WHERE e.end IS NOT NULL
        ORDER BY e.start
        """
    ).fetchall()


def load_gpu_intervals(connection: sqlite3.Connection) -> list[tuple[int, int]]:
    """返回所有 GPU 执行片段，按 start 排序。

    必须同时读两张表：JAX/XLA 默认开启 command buffer（CUDA graph），这类计算只在
    `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` 里有记录，`CUPTI_ACTIVITY_KIND_KERNEL` 里一条都
    没有。只读后者会把 pi0.5 的 GPU 时间少算 86%（实测 75.4 ms vs 480.6 ms）。
    """
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    intervals: list[tuple[int, int]] = []
    sources = ["CUPTI_ACTIVITY_KIND_KERNEL"]
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        sources.append("CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
    for table in sources:
        intervals.extend(
            connection.execute(
                f"SELECT start, end FROM {table} WHERE end IS NOT NULL"
            ).fetchall()
        )
    intervals.sort()
    return intervals


def attribute(
    ranges: list[tuple[str, int, int]],
    intervals: list[tuple[int, int]],
) -> dict[str, dict[str, float]]:
    """对每个区间名累加墙钟，以及落在其内部（含嵌套）的 GPU 执行时间。

    对每个区间取候选片段的**并集**而不是直接求和：kernel 与 graph 节点可能相互覆盖，
    直接相加会重复计数。
    """
    import bisect

    starts = [start for start, _ in intervals]
    stats: dict[str, dict[str, float]] = {}
    for name, start, end in ranges:
        entry = stats.setdefault(name, {"wall_ns": 0.0, "gpu_ns": 0.0, "count": 0.0})
        entry["wall_ns"] += end - start
        entry["count"] += 1

        index = bisect.bisect_left(starts, start)
        # 片段可能早于区间 start 开始但仍在执行，因此从左边界往前回看。
        low = index
        while low > 0 and intervals[low - 1][1] > start:
            low -= 1
        high = index
        while high < len(intervals) and intervals[high][0] < end:
            high += 1

        clipped = sorted(
            (max(start, s), min(end, e))
            for s, e in intervals[low:high]
            if e > start and s < end
        )
        covered = 0
        current_start: int | None = None
        current_end: int | None = None
        for piece_start, piece_end in clipped:
            if current_end is None or piece_start > current_end:
                if current_end is not None:
                    covered += current_end - current_start  # type: ignore[operator]
                current_start, current_end = piece_start, piece_end
            else:
                current_end = max(current_end, piece_end)
        if current_end is not None:
            covered += current_end - current_start  # type: ignore[operator]
        entry["gpu_ns"] += covered
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help=".nsys-rep 或 .sqlite")
    parser.add_argument("--range", default=None, help="只显示某一个区间名")
    parser.add_argument("--min-wall-ms", type=float, default=1.0)
    args = parser.parse_args()

    sqlite_path = ensure_sqlite(args.report)
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        ranges = load_ranges(connection)
        intervals = load_gpu_intervals(connection)
    finally:
        connection.close()

    if not ranges:
        print("没有找到 NVTX 区间：该报告可能不是用 --trace=nvtx 采集的。", file=sys.stderr)
        return 1
    if not intervals:
        print("没有找到 GPU 执行记录。", file=sys.stderr)
        return 1

    stats = attribute(ranges, intervals)
    rows = []
    for name, entry in stats.items():
        wall_ms = entry["wall_ns"] / 1e6
        if wall_ms < args.min_wall_ms or (args.range and name != args.range):
            continue
        gpu_ms = entry["gpu_ns"] / 1e6
        rows.append((name, wall_ms, gpu_ms, gpu_ms / wall_ms if wall_ms else 0.0, int(entry["count"])))
    rows.sort(key=lambda row: row[2], reverse=True)

    print(f"{'range':<26}{'wall_ms':>11}{'gpu_ms':>11}{'gpu/wall':>10}{'count':>8}")
    for name, wall_ms, gpu_ms, busy, count in rows:
        print(f"{name:<26}{wall_ms:>11.1f}{gpu_ms:>11.1f}{busy:>10.2f}{count:>8}")
    print("\ngpu/wall 越低说明该区间越是 CPU 下发 / 等待受限，而不是真的在算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
