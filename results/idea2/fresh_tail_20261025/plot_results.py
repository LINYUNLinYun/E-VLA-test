"""Static plots from terminal artifacts; no GPU or inferred quality equivalence."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent
P=R.parents[2]/"reports/idea2_fresh_tail_20261025"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})


def main():
    rows=list(csv.DictReader((P/"performance.csv").open()))
    qpath=R/"matched100_gpu3/quality_summary.csv"
    quality=list(csv.DictReader(qpath.open())) if qpath.exists() else []
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,sms in enumerate([16,32]):
        for repeat,color in [(False,"#4477AA"),(True,"#EE7733")]:
            values=[]
            for tail in [0,1,2]:
                selected=[x for x in rows if int(x["ae_sms"])==sms and int(x["tail"])==tail and x["phase"]=="paired" and x["lead"]=="3" and ("_repeat" in x["case"])==repeat]
                values.append(float(selected[0]["trajectory_ms_mean"]) if selected else np.nan)
            axes[0,col].plot([0,1,2],values,"o-",color=color,label="Reverse blocks" if repeat else "Forward blocks")
            if sms==32 and repeat and values[1]>600:
                axes[0,col].annotate("Host prepare anomaly retained",(1,values[1]),xytext=(0,-22),textcoords="offset points",ha="center",fontsize=8)
        axes[0,col].legend(fontsize=8)
        axes[0,col].set_title(f"AE {sms} SM: formal repeats on one GPU")
        axes[0,col].set_ylabel("8-request all-work time (ms)")
        if sms==16: axes[0,col].set_ylim(470,485)
        selected=[x for x in quality if x.get("ae_sm_min")==str(sms)]
        values={int(x["fresh_tail_steps"]):float(x["R5_mae"]) for x in selected}
        axes[1,col].plot([0,1,2],[values.get(t,np.nan) for t in [0,1,2]],"o-",color="#228833")
        axes[1,col].set_title("Matched observations/noise on GPU3; 100 samples")
        axes[1,col].set_ylabel("Decoded executed R5 MAE vs all-fresh")
        for ax in axes[:,col]:
            ax.set_xticks([0,1,2]);ax.set_xlabel("Mandatory all-fresh final rounds (of D10)");ax.grid(alpha=.2)
    fig.suptitle("Latency and approximation are separate workloads; full-fresh equality is not guaranteed")
    fig.tight_layout()
    for ext in ["png","svg"]:fig.savefig(P/f"fresh_tail_summary.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":main()
