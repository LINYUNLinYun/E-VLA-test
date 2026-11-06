"""Readable phase-chart styling; all scientific preparation uses frozen v1."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import phase_figures as core
from .common import atomic_json
from .same_process_parity import sha

NAMES = dict(uniform="uniform",front_middle="front-mid",back_middle="back-mid",
    short_runs="short-runs",seeded_random="random",historical_reference="historical ref")


def labels(group,layouts):
    """Unique short display labels, with reversible scientific-key mapping."""
    result,used = [],set()
    for row in group:
        cid = row["condition"]
        metadata = layouts.get(cid,{})
        if cid==row["model"]+"_full":
            name = "Full [reference]"
        else:
            layout = metadata.get("layout") or next((k for k in NAMES if cid.endswith(k)),None)
            name = NAMES.get(layout,cid.removeprefix(row["model"]+"_")[:18])
            if str(metadata.get("reference","False")).lower()=="true" and name!="historical ref":
                name += " [ref]"
        base,count = name,1
        while name in used:
            count += 1
            name = f"{base} [{count}]"
        used.add(name)
        detail = "MISSING" if row["status"]=="missing" else f"F{row['failures']}"
        detail += f" | {row['denominator']}/{row['expected']}"
        if row["status"]=="partial":
            detail += " partial"
        result.append(dict(condition=cid,short_label=name,tick_label=name+"\n"+detail))
    return result


def render(group,order,display,title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(max(15,len(group)*2.2),6.8))
    for ax,field in zip(axes,("counts","proportions")):
        bottom = np.zeros(len(group))
        for index,phase in enumerate(order):
            values = np.asarray([r[field][index] if r[field] is not None else 0 for r in group])
            bars = ax.bar(range(len(group)),values,bottom=bottom,color=plt.get_cmap("tab20")(index%20),label=phase.replace("_"," "))
            for bar,row in zip(bars,group):
                if not row["complete"]:
                    bar.set_hatch("//")
                    bar.set_alpha(.5)
            bottom += values
        for i,row in enumerate(group):
            if not row["denominator"] or not row["failures"]:
                ax.text(i,.025 if field=="proportions" else .1,
                    "missing" if not row["denominator"] else "no failures",ha="center",fontsize=8)
        ax.set_xticks(range(len(group)),[r["tick_label"] for r in display],rotation=25,ha="right",fontsize=8)
        ax.tick_params(axis="y",labelsize=8)
        ax.set_ylabel("Failed-episode count" if field=="counts" else "Proportion among FAILED episodes only",fontsize=9)
        ax.set_ylim(0,max(1,max(bottom))*1.15 if field=="counts" else 1.1)
    handles,names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="upper center",bbox_to_anchor=(.5,.94),ncol=4,fontsize=8,frameon=False)
    fig.suptitle(title+": "+core.CAPTION,fontsize=10,y=.99)
    fig.text(.02,.025,"F = failed episodes; fractions = observed/expected. Hatching/partial = incomplete observation. "
        "Missing or zero failed episodes have no failed-only proportion.",fontsize=8)
    fig.subplots_adjust(left=.06,right=.985,bottom=.27,top=.77,wspace=.16)
    return fig


def build(report,output,page_size=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report,output = Path(report),Path(output)
    if output.exists():
        raise FileExistsError("Choose a NEW immutable v2 phase-figure directory")
    sources = [report/"failure_progress_phases.csv",report/"quality_screen_init0.csv",report/"quality_combined_init0_4.csv"]
    rows,order = core.prepare(core.read_csv(sources[1])+core.read_csv(sources[2]),core.read_csv(sources[0]))
    layouts = {}
    if (report/"layout_masks.csv").exists():
        sources.append(report/"layout_masks.csv")
        layouts = {r["condition"]:r for r in core.read_csv(sources[-1])}
    output.mkdir(parents=True)
    artifacts = []
    for model,scope,budget,page,group in core.pages(rows,page_size):
        display = labels(group,layouts)
        fig = render(group,order,display,f"{model} / {scope} / mask budget {budget}")
        path = output/f"phases_{model}_{scope}_budget{budget}_page{page}.png"
        fig.savefig(path,dpi=150)
        plt.close(fig)
        artifacts.append(dict(path=path.name,sha256=sha(path),labels=display,conditions=[r["condition"] for r in group]))
    result = dict(source_sha256=sha(__file__),core_source_sha256=sha(core.__file__),
        csv_sources_sha256={str(p.absolute()):sha(p) for p in sources},phase_order=order,caption=core.CAPTION,
        plots=artifacts,observations=rows,page_size=page_size,
        scope="Styling only; frozen v1 prepare/pages define all counts/proportions/partial handling. No ranking, causality or completion claim.")
    atomic_json(output/"plot_index.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--page-size",type=int,default=10)
    args = parser.parse_args()
    result = build(args.report,args.output,args.page_size)
    print(json.dumps(dict(plots=len(result["plots"]),output=args.output)))


if __name__=="__main__":
    main()
