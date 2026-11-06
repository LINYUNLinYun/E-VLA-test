from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
data=json.loads((root/"steady/result.json").read_text())["summary"]
labels=["Eager / original", "Eager / two host workers", "Graph / serial", "Graph / AE first",
        "Graph / Prefill first", "Graph / AE first / high AE priority", "Graph / Prefill first / high AE priority"]
p50=[v["response_host_ms"]["p50"] for v in data.values()]
p95=[v["response_host_ms"]["p95"] for v in data.values()]
fig,ax=plt.subplots(figsize=(11,5.3),layout="constrained")
for i,(a,b) in enumerate(zip(p50,p95)):
    ax.barh(i,b,color="#e2e8f0",height=.62,label="P95" if i==0 else None)
    ax.barh(i,a,color="#64748b" if i<2 else "#0f766e",height=.37,label="P50" if i==0 else None)
    ax.text(b+4,i,f"{a:.2f} / {b:.2f} ms",va="center",fontsize=9)
ax.set_yticks(range(len(labels)),labels)
ax.invert_yaxis()
ax.set_xlim(0,485)
ax.set_xlabel("Policy response time (ms), including input preparation and output decode")
ax.set_title("Dynamic real-observation replay: 39 steady requests per mode, no profiler")
ax.legend(loc="lower right")
ax.spines[["top","right"]].set_visible(False)
fig.savefig(root/"runtime_response.png",dpi=180)
