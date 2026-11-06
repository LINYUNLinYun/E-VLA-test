"""Publication-style final quality figure from the complete prospective grid."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent;P=R.parents[2]/"reports/idea2_continuous_encoder_20261026"
ARMS=["fresh18_no_encoder","sm16_no_encoder","sm16_encoder"]
LABELS=["Fresh KV; no next Encoder","AE16; no next Encoder","AE16 + next Encoder"]
COLORS=["#4477AA","#CC6677","#228833"]


def main():
    data=json.loads((P/"statistics.json").read_text())
    if data["status"]!="complete" or data["received_rows"]!=300:raise ValueError("Final figure requires complete300")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={"width_ratios":[1,.95]})
    for i,arm in enumerate(ARMS):
        row=next(r for r in data["configurations"] if r["configuration"]==arm and r["scope"]=="overall")
        rate=row["success_rate"]*100;low,high=[v*100 for v in row["wilson95"]]
        axes[0].errorbar(i,rate,yerr=[[max(0,rate-low)],[max(0,high-rate)]],fmt="o",color=COLORS[i],capsize=4)
        axes[0].annotate(f"{row['successes']}/100",(i,high),xytext=(0,7),textcoords="offset points",ha="center")
    axes[0].set_xticks(range(3),LABELS,rotation=15,ha="right");axes[0].set_ylim(0,105);axes[0].set_xlim(-.45,2.45)
    axes[0].set_ylabel("Success rate (%)");axes[0].set_title("Same100 initial scenarios per condition\nNominal Wilson95% intervals")
    selected=[next(r for r in data["comparisons"] if r["scope"]=="overall" and r["candidate"]=="sm16_encoder" and r["reference"]==ref)
              for ref in ["fresh18_no_encoder","sm16_no_encoder"]]
    bounds=[0.]
    for i,row in enumerate(selected):
        point=row["paired_success_rate_difference"]*100
        lo,hi=[v*100 for v in row["task_cluster_percentile95"]];bounds.extend([lo,hi])
        axes[1].errorbar(point,i,xerr=[[max(0,point-lo)],[max(0,hi-point)]],fmt="o",capsize=4,color=COLORS[2])
        axes[1].annotate(f"{point:+.1f} pp; gains {row['candidate_only_success_b']}, losses {row['reference_only_success_c']}",
                         (point,i),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9)
    axes[1].set_yticks(range(2),["Full pipeline vs fresh (primary)","Encoder vs no Encoder (ablation)"])
    axes[1].set_ylim(1.5,-.5);axes[1].set_xlim(min(bounds)-3,max(bounds)+3);axes[1].axvline(0,color="black",ls="--",lw=.8)
    axes[1].set_xlabel("Paired success-rate difference (percentage points)")
    axes[1].set_title("40-task weighted cluster bootstrap95%\nWithin-task samples retained (2 or3 inits)")
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("Full pi0.5: D10 / H10 / R5; continuous20Hz actor, single latest-prediction mailbox\nAll three conditions prepare the next input ahead")
    fig.tight_layout()
    for ext in ("png","svg"):fig.savefig(P/f"quality_and_paired_effect.{ext}",dpi=180,bbox_inches="tight")


if __name__=="__main__":main()
