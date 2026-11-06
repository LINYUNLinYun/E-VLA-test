"""Real kernel interval overlap, using same-thread runtime NVTX/API correlation."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics


STAGES = {"encoder", "vlm", "AE", "preprocess"}


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end-start for start, end in union(intervals))


def overlap(left, right):
    left, right = union(left), union(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1])-max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]: i += 1
        else: j += 1
    return total


def analyze(path):
    with sqlite3.connect("file:"+str(path.resolve())+"?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        strings = dict(db.execute("select id,value from StringIds"))
        contexts = [dict(row) for row in db.execute("select * from TARGET_INFO_CUDA_CONTEXT_INFO")]
        apis = [dict(row) for row in db.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME")]
        kernels = [dict(row) for row in db.execute("select rowid as kernel_rowid,* from CUPTI_ACTIVITY_KIND_KERNEL")]
        all_ops = list(kernels)
        tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table in tables:
                all_ops.extend(dict(row) for row in db.execute("select * from " + table))
        ranges = []
        for row in db.execute("select rowid as nvtx_rowid,* from NVTX_EVENTS where domainId=0 and end is not null"):
            row = dict(row)
            row["stage"] = row["text"] or strings.get(row["textId"], "")
            if row["stage"] in STAGES: ranges.append(row)
    operations = defaultdict(list)
    all_operations = defaultdict(list)
    for operation in all_ops:
        all_operations[operation["globalPid"], operation["correlationId"]].append(operation)
    for kernel in kernels:
        kernel["name"] = strings.get(kernel["demangledName"], str(kernel["demangledName"]))
        kernel["short_name"] = strings.get(kernel["shortName"], kernel["name"])
        operations[kernel["globalPid"], kernel["correlationId"]].append(kernel)
    tagged = {}
    proofs, cpu_calls = [], []
    counters = Counter()
    for nvtx in sorted(ranges, key=lambda row: row["start"]):
        stage = nvtx["stage"]
        frame = counters[stage]
        counters[stage] += 1
        calls = [api for api in apis if api["globalTid"] == nvtx["globalTid"]
                 and nvtx["start"] <= api["start"] and api["end"] <= nvtx["end"]]
        linked = []
        linked_operations = []
        for api in calls:
            cpu_calls.append(dict(stage=stage, frame=frame, name=strings.get(api["nameId"], ""),
                                  duration_ms=(api["end"]-api["start"])/1e6))
            key = (api["globalTid"] & -16777216, api["correlationId"])
            linked_operations.extend(all_operations.get(key, []))
            for kernel in operations.get(key, []):
                identity = kernel["kernel_rowid"]
                if identity in tagged:
                    raise RuntimeError("A kernel was ambiguously linked to two stage ranges")
                tagged[identity] = dict(kernel, stage=stage, frame=frame)
                linked.append(kernel)
        if stage in ("encoder", "vlm", "AE") and not linked:
            raise RuntimeError(f"No real kernels for {stage} frame {frame}")
        if linked:
            proofs.append(dict(stage=stage, frame=frame, kernels=len(linked),
                cpu_start_ns=nvtx["start"], cpu_end_ns=nvtx["end"],
                gpu_first_ns=min(x["start"] for x in linked_operations),
                gpu_last_ns=max(x["end"] for x in linked_operations),
                cpu_submit_ms=(nvtx["end"]-nvtx["start"])/1e6,
                gpu_envelope_ms=(max(x["end"] for x in linked_operations)-min(x["start"] for x in linked_operations))/1e6,
                gpu_busy_union_ms=duration((x["start"], x["end"]) for x in linked)/1e6))
    selected = list(tagged.values())
    def intervals(stage, frame=None):
        return [(x["start"], x["end"]) for x in selected
                if x["stage"] == stage and (frame is None or x["frame"] == frame)]
    context_index = {(row["processId"], row["deviceId"], row["contextId"]): row for row in contexts}
    resources = Counter()
    signatures = defaultdict(list)
    for kernel in selected:
        green = kernel["greenContextId"]
        context = context_index.get(((kernel["globalPid"] >> 24) & 0xffffff, kernel["deviceId"], green)) if green else None
        actual = context["numMultiprocessors"] if context is not None else (0 if green == 0 else None)
        if green and actual is None: raise RuntimeError("Green kernel has no unique actual SM attribution")
        resources[kernel["stage"], green, actual] += 1
        signature = (kernel["stage"], kernel["short_name"], kernel["gridX"], kernel["gridY"], kernel["gridZ"],
                     kernel["blockX"], kernel["blockY"], kernel["blockZ"])
        signatures[signature].append((kernel["end"]-kernel["start"])/1000)
    groups = [dict(stage=key[0], kernel=key[1], grid=list(key[2:5]), block=list(key[5:8]),
                   count=len(values), mean_us=statistics.mean(values), p50_us=statistics.median(values),
                   sum_ms=sum(values)/1000)
              for key, values in signatures.items()]
    groups.sort(key=lambda row: row["sum_ms"], reverse=True)
    busy = {stage: duration(intervals(stage))/1e6 for stage in STAGES}
    pairs = []
    for frame in range(min(counters["AE"], counters["vlm"])):
        pairs.append(dict(frame=frame, ae_vlm_kernel_ms=overlap(intervals("AE", frame), intervals("vlm", frame))/1e6,
            ae_current_encoder_kernel_ms=overlap(intervals("AE", frame), intervals("encoder", frame))/1e6,
            ae_next_encoder_kernel_ms=(overlap(intervals("AE", frame), intervals("encoder", frame+1))/1e6
                                       if frame+1 < counters["encoder"] else None)))
    calls = defaultdict(list)
    for call in cpu_calls: calls[call["stage"], call["name"]].append(call["duration_ms"])
    return dict(source=str(path), counts=dict(counters), real_kernels_total=len(kernels),
                linked_core_and_preprocess_kernels=len(selected), unlinked_control_kernels=len(kernels)-len(selected),
                gpu_busy_union_ms=busy,
                ae_vlm_real_kernel_overlap_ms=overlap(intervals("AE"), intervals("vlm"))/1e6,
                ae_encoder_real_kernel_overlap_ms=overlap(intervals("AE"), intervals("encoder"))/1e6,
                frames=pairs, replay_envelopes=proofs, contexts=contexts,
                actual_resources=[dict(stage=key[0], green_context=key[1], eligible_sms=key[2], kernels=value)
                                  for key,value in resources.items()],
                cuda_api=[dict(stage=key[0], name=key[1], count=len(values), total_ms=sum(values),
                               mean_ms=statistics.mean(values)) for key,values in calls.items()],
                kernel_groups=groups,
                interpretation="Overlap is intersection of unions of actual GPU kernel intervals, not NVTX/Event envelopes. Primary eligible_sms=0 means unrestricted, not zero occupancy. Timing is profiled, not performance latency.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = analyze(args.sqlite)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    with (args.output / "kernel_groups.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["kernel_groups"][0]))
        writer.writeheader();writer.writerows(result["kernel_groups"])
    print(json.dumps({key:result[key] for key in ("counts","real_kernels_total","ae_vlm_real_kernel_overlap_ms","ae_encoder_real_kernel_overlap_ms","actual_resources")},indent=2))


if __name__ == "__main__": main()
