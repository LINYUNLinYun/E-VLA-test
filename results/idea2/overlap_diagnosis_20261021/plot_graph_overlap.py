"""Plot unprofiled latency separately from node-level traced kernel timelines."""
from pathlib import Path
import json
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


root = Path(__file__).resolve().parent
workspace = root.parents[2]
result = json.loads((root / "graph_grouped/result.json").read_text())
trace = json.loads((root / "graph_grouped_kernel_overlap.json").read_text())
connection = sqlite3.connect(workspace / "nsys/idea2_experiment/graph_overlap_grouped_20261021.sqlite")
connection.row_factory = sqlite3.Row
prefill_stream = trace["rows"][next(i for i,r in enumerate(trace["rows"]) if r["mode"]=="prefill_only")]["modules"]["prefill"]["streams"][0]
modes = ["serial_ae_first", "concurrent_ae_first", "concurrent_ae_first_high_ae"]
labels = ["Serial", "Two streams", "Two streams / high AE priority"]
fig, axes = plt.subplots(4, 1, figsize=(12, 8.7), gridspec_kw={"height_ratios":[1.7,1,1,1]}, layout="constrained")
values = [result["summary"][m]["all_done_host_ms"]["p50"] for m in modes]
axes[0].barh(labels, values, color=["#64748b", "#0f766e", "#1d4ed8"], height=.55)
axes[0].invert_yaxis()
for i,value in enumerate(values):
    axes[0].text(value+.6, i, f"{value:.2f} ms  ({values[0]/value:.3f}x)", va="center")
axes[0].set_xlim(0,85)
axes[0].set_xlabel("Both tasks complete: P50 wall time (50 samples, NO profiler)")
axes[0].set_title("Fixed-input Graph diagnostic: overlap improves throughput, contention limits the gain", loc="left", fontsize=12)
for axis,mode,label in zip(axes[1:],modes,labels):
    region = connection.execute("SELECT start,end FROM NVTX_EVENTS WHERE text=?",(f"pair/{mode}/1",)).fetchone()
    kernels = connection.execute("SELECT start,end,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start",tuple(region)).fetchall()
    zero = min(k["start"] for k in kernels)
    for name,y,color in [("ae",1,"#ea580c"),("prefill",0,"#2563eb")]:
        pieces = [[((k["start"]-zero)/1e6,y),((k["end"]-zero)/1e6,y)]
                  for k in kernels if (k["streamId"]==prefill_stream)==(name=="prefill")]
        axis.add_collection(LineCollection(pieces, colors=color, linewidths=7))
    row = next(r for r in trace["rows"] if r["mode"]==mode and r["iteration"]==1)
    axis.set_title(f"{label}: actual kernel overlap {row['kernel_overlap_ms']:.2f} ms / union {row['kernel_union_ms']:.2f} ms",loc="left",fontsize=10)
    axis.set_yticks([0,1],["Prefill","AE"])
    axis.set_ylim(-.6,1.6)
    axis.set_xlim(0,80)
    axis.grid(axis="x",alpha=.2)
axes[-1].set_xlabel("GPU kernel time from first kernel (ms); nsys node tracing perturbs timing")
fig.savefig(root / "graph_overlap.png",dpi=180)
