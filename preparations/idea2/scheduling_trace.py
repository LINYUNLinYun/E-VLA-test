"""Attribute nsys kernel intervals through launch correlations and module NVTX.

CPU range intersections are never used as evidence of simultaneous GPU work.
Only ae/prefill kernel intersections on distinct streams of the same device and
context count. Copies and memsets are excluded from overlap statistics.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def union(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return merged


def duration(intervals):
    return sum(end-begin for begin, end in union(intervals))


def intersection(left, right):
    left, right = union(left), union(right)
    total = i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0, min(a[1], b[1])-max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path):
    db = sqlite3.connect(path)
    ranges = db.execute("""SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE n.end IS NOT NULL ORDER BY n.start,n.end DESC""").fetchall()
    by_thread = defaultdict(list)
    sections = []
    for begin, end, tid, label in ranges:
        if label in ("ae", "prefill"):
            by_thread[tid].append((begin, end, label))
        elif label and (label.startswith("pair/") or label.startswith("runtime/")
                        or label.startswith("cewp/paired/") or label.startswith("cewp/continuous/")
                        or label.startswith("cewp/isolated_")):
            sections.append((begin, end, label))
    starts = {tid:[r[0] for r in rows] for tid, rows in by_thread.items()}
    section_starts = [r[0] for r in sections]
    groups = defaultdict(list)
    unassigned = 0
    for begin, end, dev, ctx, stream, launch, launch_end, tid, corr in db.execute("""
        SELECT k.start,k.end,k.deviceId,k.contextId,k.streamId,r.start,r.end,r.globalTid,k.correlationId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
        ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216)
        ORDER BY r.start"""):
        section = bisect_right(section_starts, launch)-1
        if section < 0 or launch_end > sections[section][1]:
            continue
        label = sections[section][2]
        module_index = bisect_right(starts.get(tid, []), launch)-1
        module = None
        # Nested duplicate labels from the scheduler/wrapper have the same
        # module; scan outwards when the innermost range has just ended.
        for i in range(module_index, -1, -1):
            row = by_thread[tid][i]
            if row[0] <= launch and launch_end <= row[1]:
                module = row[2]
                break
        if module is None:
            unassigned += 1
            continue
        groups[label].append((begin, end, dev, ctx, stream, module))
    results = {}
    for label, kernels in groups.items():
        modules = {name:[r for r in kernels if r[5] == name] for name in ("ae", "prefill")}
        overlap = 0
        for device_context in set((r[2], r[3]) for r in kernels):
            streams = {name:set(r[4] for r in values if r[2:4] == device_context)
                       for name, values in modules.items()}
            intervals = []
            # Each module uses exactly one stream per pair. Reject assumptions
            # that could double-count intersections in a future implementation.
            assert all(len(s) <= 1 for s in streams.values()), (label, streams)
            if streams["ae"] and streams["prefill"] and streams["ae"] != streams["prefill"]:
                overlap += intersection(*([[r[0], r[1]] for r in modules[name]
                                          if r[2:4] == device_context] for name in ("ae", "prefill")))
        busy = duration([(r[0], r[1]) for r in kernels])
        results[label] = dict(kernel_count=len(kernels), kernel_union_ms=busy/1e6,
                             overlap_ms=overlap/1e6, overlap_fraction=overlap/busy if busy else 0,
                             modules={name:dict(kernel_count=len(values),
                                 kernel_busy_ms=duration([(r[0],r[1]) for r in values])/1e6,
                                 gpu_envelope_ms=(max(r[1] for r in values)-min(r[0] for r in values))/1e6 if values else 0,
                                 streams=sorted(set(r[4] for r in values))) for name,values in modules.items()})
    api = db.execute("""SELECT s.value,COUNT(*),SUM(r.end-r.start)/1e6
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
        GROUP BY s.value ORDER BY SUM(r.end-r.start) DESC""").fetchall()
    aggregate = {}
    for kind_mode in dict.fromkeys("/".join(label.split("/")[:3 if label.startswith("cewp/") else 2]) for label in results):
        rows = [r for label,r in results.items() if label.startswith(kind_mode+"/")]
        busy, overlap = sum(r["kernel_union_ms"] for r in rows), sum(r["overlap_ms"] for r in rows)
        aggregate[kind_mode] = dict(requests=len(rows),kernel_union_ms=busy,overlap_ms=overlap,
                                   overlap_fraction=overlap/busy if busy else 0,
                                   modules={name:dict(mean_kernel_busy_ms=sum(r["modules"][name]["kernel_busy_ms"] for r in rows)/len(rows),
                                                      mean_gpu_envelope_ms=sum(r["modules"][name]["gpu_envelope_ms"] for r in rows)/len(rows))
                                            for name in ("ae", "prefill")})
    db.close()
    return dict(source=str(path), unassigned_module_kernels=unassigned, aggregate=aggregate,
                sections=results, cuda_api=[dict(name=n,calls=c,total_cpu_ms=t) for n,c,t in api],
                definition="Same-GPU/context distinct-stream AE/prefill kernel intersection / kernel union; excludes copies/memsets. Event/envelope includes gaps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["aggregate"], indent=2))
