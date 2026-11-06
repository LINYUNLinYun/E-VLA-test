"""Standalone final scientific plots from the complete prospective statistics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent
P=R.parents[2]/"reports/idea2_closedloop200_20261025"
ARMS=["fresh18","sm16_tail0","sm32_tail0","sm32_tail2"]
LABELS=["Fresh current KV","AE16, mixed KV","AE32, mixed KV","AE32, final2 fresh"]
COLORS=["#4477AA","#228833","#CC6677","#AA3377"]
SUITES=["libero_object","libero_spatial","libero_goal","libero_10"]


def save(fig,name):
    for extension in ["png","svg"]:fig.savefig(P/f"{name}.{extension}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    data=json.loads((P/"statistics.json").read_text())
    if data["status"]!="complete" or data["received_rows"]!=800:
        raise ValueError("Final plot requires the complete fixed800-case grid")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={"width_ratios":[1,1.1]})
    for index,arm in enumerate(ARMS):
        value=next(x for x in data["configurations"] if x["configuration"]==arm and x["scope"]=="overall")
        rate=value["success_rate"]*100;lo,hi=[n*100 for n in value["wilson95"]]
        axes[0].errorbar(index,rate,yerr=[[max(0,rate-lo)],[max(0,hi-rate)]],fmt="o",color=COLORS[index],capsize=4)
        axes[0].annotate(f"{value['successes']}/200",(index,rate),xytext=(0,9),textcoords="offset points",ha="center",fontsize=9)
    axes[0].set_xticks(range(4),LABELS,rotation=18,ha="right")
    axes[0].set_ylim(0,105);axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("200 fixed cases per condition\nNominal Wilson95% intervals")
    bounds=[0]
    for index,arm in enumerate(ARMS[1:]):
        value=next(x for x in data["comparisons"] if x["candidate"]==arm and x["reference"]=="fresh18" and x["scope"]=="overall")
        point=value["paired_success_rate_difference"]*100
        lo,hi=[n*100 for n in value["task_cluster_percentile95"]];bounds.extend([lo,hi])
        axes[1].errorbar(point,index,xerr=[[max(0,point-lo)],[max(0,hi-point)]],fmt="o",color=COLORS[index+1],capsize=4)
        axes[1].annotate(f"{point:+.1f} pp; gains {value['candidate_only_success_b']}, losses {value['reference_only_success_c']}",(point,index),xytext=(0,12),textcoords="offset points",ha="center",fontsize=8)
        if value["no_discordance"]:axes[1].text(.02,.02,"Zero discordances do not prove equivalence",transform=axes[1].transAxes,fontsize=8)
    axes[1].axvline(0,color="black",lw=.8,ls="--")
    axes[1].set_yticks(range(3),LABELS[1:]);axes[1].invert_yaxis();axes[1].set_ylim(2.5,-.5)
    axes[1].set_xlim(min(bounds)-3,max(bounds)+3)
    axes[1].set_xlabel("Paired success-rate difference vs fresh (percentage points)")
    axes[1].set_title("Same suite/task/init pairs\n40-task cluster bootstrap95% intervals")
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("Exploratory LIBERO4-suite study: D10 / H10 / R5; approximate KV pipelines")
    fig.tight_layout();save(fig,"success_and_paired_effect")
    values=np.array([[next(x for x in data["configurations"] if x["configuration"]==arm and x["scope"]==suite)["successes"] for suite in SUITES] for arm in ARMS])
    fig,ax=plt.subplots(figsize=(7.6,4.2));im=ax.imshow(values/50,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
    for i in range(4):
        for j in range(4):ax.text(j,i,f"{values[i,j]}/50",ha="center",va="center",color="white" if values[i,j]>36 else "black")
    ax.set_xticks(range(4),[x.replace("libero_","") for x in SUITES]);ax.set_yticks(range(4),LABELS)
    ax.set_title("Suite-level outcomes:10 tasks x5 official inits each")
    fig.colorbar(im,ax=ax,label="Success rate");fig.tight_layout();save(fig,"suite_success_counts")


if __name__=="__main__":main()
