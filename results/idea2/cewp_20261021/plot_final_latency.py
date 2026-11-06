"""Primary-card latency plot; freshness failures and fresh-KV references are explicit."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/"comparison.csv").open()))
summary=json.loads((root/"summary.json").read_text())
cases=[("main","graph_serial","Graph serial / old KV","#7b8794"),
       ("main","graph_fresh_serial","Graph serial / fresh KV","#aab3bd"),
       ("main","graph_ae_high","Full overlap / old KV","#3e72a6"),
       ("wait_full","graph_ae_high","Full overlap + prior-KV wait","#8b6bb1"),
       ("vision_wait","graph_ae_high","Vision overlap + prior-KV wait","#229785"),
       ("vision_modulation_wait_repeat","graph_ae_high","Vision + wait + modulation (repeat)","#d18431")]
fig,axes=plt.subplots(1,3,figsize=(19,6),sharex=True,sharey=True,layout="constrained")
evidence=[]
maximum=0
for ax,steps in zip(axes,(1,5,10)):
    labels=[]
    for index,(suffix,mode,label,color) in enumerate(cases):
        run=f"d{steps}_{suffix}"
        found=[r for r in rows if r["run"]==run and r["mode"]==mode and r["phase"]=="continuous" and r["eligible"]=="True"]
        labels.append(label)
        if len(found)!=1:
            ax.text(1,index,"pending",va="center",color="gray")
            continue
        r=found[0];p50,p95=float(r["cpu_cpu_ms_p50"]),float(r["cpu_cpu_ms_p95"])
        fresh="fresh" in mode
        failed=not fresh and r.get("freshness_gate")!="True"
        ax.barh(index,p50,color=color,hatch="///" if fresh else None,edgecolor="#c43c39" if failed else "white",linewidth=2 if failed else .5)
        ax.plot([p50,p95],[index,index],color="black",lw=1)
        ax.plot([p95,p95],[index-.12,index+.12],color="black",lw=1)
        ax.text(p95+1,index,f"{p50:.1f}"+(" !" if failed else ""),va="center",fontsize=9)
        evidence.append(dict(steps=steps,run=run,mode=mode,p50=p50,p95=p95,fresh_KV=fresh,freshness_gate=r.get("freshness_gate")))
        maximum=max(maximum,p95+12)
    index=len(cases);labels.append("Original JAX / fresh KV")
    native=next((r for r in summary["native_fresh"] if r["run"]==f"native_d{steps}_gpu2" and r["status"]=="complete"),None)
    if native:
        p50,p95=native["cpu_cpu_ms_p50"],native["cpu_cpu_ms_p95"]
        ax.barh(index,p50,color="#b9b9b9",hatch="///",edgecolor="white")
        ax.plot([p50,p95],[index,index],color="black",lw=1)
        ax.plot([p95,p95],[index-.12,index+.12],color="black",lw=1)
        ax.text(p95+1,index,f"{p50:.1f}",va="center",fontsize=9)
        maximum=max(maximum,p95+12)
        evidence.append(dict(steps=steps,run=native["run"],p50=p50,p95=p95,fresh_KV=True))
    else:
        ax.text(1,index,"pending",va="center",color="gray")
    ax.set_yticks(range(len(labels)),labels)
    ax.set_title(f"Flow matching D={steps}")
    ax.set_xlabel("CPU-to-CPU latency (ms): bar=P50, whisker=P95")
    ax.grid(axis="x",alpha=.2)
axes[0].invert_yaxis()
axes[0].set_xlim(0,max(110,maximum))
fig.suptitle("Same RTX 4090 (GPU2), 100 continuous requests per configuration\nHatched: fresh-KV semantics; red outline / !: fails age or update-rate gate; whiskers are not confidence intervals",fontsize=12)
fig.savefig(root/"best_latency_comparison.png",dpi=180)
plt.close(fig)
(root/"best_latency_comparison.json").write_text(json.dumps(dict(source="comparison.csv + summary.json",measurements=evidence),indent=2))
print("Wrote best_latency_comparison.png")
