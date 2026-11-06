"""Kernel overlap plus per-kernel concurrent slowdown from independent nsys runs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from .scheduling_trace import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze(args.sqlite)
    db = sqlite3.connect(args.sqlite)
    # Attribute kernel launches, including graph child kernels whose correlation
    # points to cudaGraphLaunch. GPU time intervals (not host NVTX intersections)
    # are the measured units. Nested stage ranges refine the module attribution.
    ranges = db.execute("""SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE n.end IS NOT NULL ORDER BY n.start""").fetchall()
    modules = defaultdict(list)
    sections = defaultdict(list)
    for begin, end, tid, label in ranges:
        if label in ("ae", "prefill", "prefill/vision", "prefill/vlm"):
            modules[tid].append((begin, end, label))
        if label and (label.startswith("cewp/paired/") or label.startswith("cewp/continuous/")
                      or label.startswith("cewp/isolated_")):
            sections[tid].append((begin, end, label))
    stats = defaultdict(list)
    for begin, end, launch, launch_end, tid, name in db.execute("""
        SELECT k.start,k.end,r.start,r.end,r.globalTid,s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
        ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216)
        JOIN StringIds s ON k.demangledName=s.id ORDER BY r.start"""):
        section = next((r for r in reversed(sections[tid]) if r[0]<=launch and launch_end<=r[1]), None)
        module = next((r for r in reversed(modules[tid]) if r[0]<=launch and launch_end<=r[1]), None)
        if section and module:
            group = "/".join(section[2].split("/")[:3])
            stats[group, module[2], name].append((end-begin)/1e3)
    report["kernel_types"] = [dict(group=g, module=m, name=n, calls=len(v),
        total_ms=sum(v)/1000, mean_us=float(np.mean(v)), p50_us=float(np.median(v)))
        for (g,m,n),v in sorted(stats.items(), key=lambda kv: -sum(kv[1]))]
    report["caveat"] = "Kernel elapsed intervals include resource stalls; not hardware-counter utilization. Compare counts/shapes and matching modes before calling duration inflation contention. Profiler timings are not steady latency."
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
