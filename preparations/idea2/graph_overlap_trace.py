"""Measure CUDA Graph kernel overlap from a node-level Nsight Systems export.

Use drained pair ranges only to select iterations; overlap itself comes from
CUPTI kernel intervals, not CPU NVTX ranges or CUDA Event envelopes.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end-start for start, end in merged(intervals))/1e6


def analyze(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    ranges = connection.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'pair/%' ORDER BY start").fetchall()
    kernels = [dict(row) for row in connection.execute(
        "SELECT k.*,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY start")]
    assert kernels and all(k["graphNodeId"] is not None for k in kernels)
    prefill_streams = {k["streamId"] for r in ranges if r["text"].startswith("pair/prefill_only/")
                       for k in kernels if r["start"] <= k["start"] and k["end"] <= r["end"]}
    assert len(prefill_streams) == 1, prefill_streams
    rows, covered, families = [], 0, defaultdict(lambda: defaultdict(list))
    for region in ranges:
        _, mode, iteration = region["text"].split("/")
        selected = [k for k in kernels if region["start"] <= k["start"] and k["end"] <= region["end"]]
        covered += len(selected)
        groups = {name:[k for k in selected if (k["streamId"] in prefill_streams) == (name == "prefill")]
                  for name in ("ae", "prefill")}
        row = dict(mode=mode, iteration=int(iteration), count=len(selected), modules={})
        for name, items in groups.items():
            intervals = [(k["start"], k["end"]) for k in items]
            envelope = (max(k["end"] for k in items)-min(k["start"] for k in items))/1e6 if items else 0
            busy = duration(intervals)
            row["modules"][name] = dict(count=len(items), busy_ms=busy, envelope_ms=envelope,
                                        gap_ms=envelope-busy,
                                        streams=sorted({k["streamId"] for k in items}))
            counts = Counter()
            times = Counter()
            for k in items:
                # Grid/CTA/register/smem shape differentiates GEMM dispatches.
                key = (k["name"], *(k[f"grid{x}"] for x in "XYZ"), *(k[f"block{x}"] for x in "XYZ"),
                       k["registersPerThread"], k["staticSharedMemory"]+k["dynamicSharedMemory"])
                counts[key] += 1
                times[key] += (k["end"]-k["start"])/1e6
            for key in counts:
                families[(mode,name)][key].append(dict(iteration=int(iteration),count=counts[key],total_ms=times[key]))
        union = duration([(k["start"], k["end"]) for k in selected])
        overlap = sum(row["modules"][n]["busy_ms"] for n in groups)-union
        row.update(kernel_union_ms=union, kernel_overlap_ms=overlap,
                   overlap_fraction=overlap/union if union else 0,
                   gpu_envelope_ms=(max(k["end"] for k in selected)-min(k["start"] for k in selected))/1e6)
        rows.append(row)
    warmups = connection.execute("SELECT start,end FROM NVTX_EVENTS WHERE text LIKE 'warmup/%'").fetchall()
    warmup_count = sum(1 for k in kernels if any(r["start"] <= k["start"] and k["end"] <= r["end"] for r in warmups))
    assert covered + warmup_count == len(kernels), (covered,warmup_count,len(kernels))
    summary = {}
    for mode in dict.fromkeys(r["mode"] for r in rows):
        items = [r for r in rows if r["mode"] == mode]
        summary[mode] = {field:statistics.mean(r[field] for r in items)
                         for field in ("kernel_union_ms","kernel_overlap_ms","overlap_fraction","gpu_envelope_ms")}
        summary[mode]["modules"] = {name:{field:statistics.mean(r["modules"][name][field] for r in items)
                                          for field in ("count","busy_ms","envelope_ms","gap_ms")}
                                   for name in ("ae","prefill")}
    family_rows = [dict(mode=mode,module=module,name=key[0],grid=list(key[1:4]),block=list(key[4:7]),
                        registers_per_thread=key[7],shared_bytes=key[8],
                        mean_count=statistics.mean(r["count"] for r in vals),
                        mean_total_ms=statistics.mean(r["total_ms"] for r in vals),rows=vals)
                   for (mode,module),group in families.items() for key,vals in group.items()]
    return dict(source=str(path),selection="Fully drained NVTX pair boundaries; kernel timestamps and isolated prefill stream determine overlap",
                kernel_count=len(kernels),covered_count=covered,warmup_count=warmup_count,
                rows=rows,summary=summary,families=family_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("output")
    args = parser.parse_args()
    report = analyze(args.sqlite)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report["summary"],indent=2))


if __name__ == "__main__":
    main()
