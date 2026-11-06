"""Measure actual GPU kernel overlap in a PyTorch Chrome trace."""
import json
from pathlib import Path


def summarize(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    edges = []
    stream_kernel_us = {}
    for row in events:
        args = row.get("args", {})
        if row.get("ph") != "X" or "kernel" not in row.get("cat", "").lower() or "stream" not in args:
            continue
        stream = str((args.get("device", row.get("pid")), args["stream"]))
        duration = row.get("dur", 0)
        if duration <= 0:
            continue
        stream_kernel_us[stream] = stream_kernel_us.get(stream, 0) + duration
        edges.extend([(row["ts"], stream, 1), (row["ts"] + duration, stream, -1)])
    edges.sort()
    active = {}
    previous = None
    union = overlap = 0
    for timestamp, stream, delta in edges:
        if previous is not None:
            count = sum(value > 0 for value in active.values())
            if count:
                union += timestamp - previous
            if count > 1:
                overlap += timestamp - previous
        active[stream] = active.get(stream, 0) + delta
        previous = timestamp
    return dict(kernel_streams=len(stream_kernel_us), gpu_kernel_union_ms=union / 1000,
                simultaneous_kernel_overlap_ms=overlap / 1000,
                overlap_fraction_of_union=overlap / union if union else 0,
                stream_kernel_ms={key: value / 1000 for key, value in stream_kernel_us.items()},
                definition="Wall duration with kernels active on at least two CUDA streams; excludes CPU and copies")
