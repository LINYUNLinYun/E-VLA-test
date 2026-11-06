"""CPU-only final-stage progress figures from an existing report snapshot."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import atomic_json
from .same_process_parity import sha

PHASES = ("initial","approach","contact","grasp_or_manipulate","transport","goal_alignment","release_or_final_action","unknown")
CAPTION = "furthest observed progress, NOT cause of failure"


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def prepare(quality,phases):
    rows,counts = {},{}
    for raw in quality:
        if raw["suite"]!="ALL" or str(raw.get("benchmark_only","False")).lower()=="true":
            continue
        key = (raw["scope"],raw["condition"])
        assert key not in rows,"Duplicate quality condition/scope"
        row = dict(raw)
        for field in ("denominator","expected","successes","budget"):
            row[field] = int(row[field])
        assert 0<=row["successes"]<=row["denominator"]<=row["expected"]
        row["failures"] = row["denominator"]-row["successes"]
        row["complete"] = row["denominator"]==row["expected"]
        row["status"] = "missing" if not row["denominator"] else "complete" if row["complete"] else "partial"
        rows[key],counts[key] = row,{}
    for phase in phases:
        key = (phase["scope"],phase["condition"])
        if phase["suite"]!="ALL" or key not in rows:
            continue
        count,failed = int(phase["count"]),int(phase["failed_episode_denominator"])
        name = phase["progress_phase"] or "unknown"
        assert name not in counts[key] and count>0 and failed==rows[key]["failures"]
        assert np.isclose(float(phase["proportion"]),count/failed)
        counts[key][name] = count
    order = list(PHASES)+sorted({p for values in counts.values() for p in values}-set(PHASES))
    for key,row in rows.items():
        assert sum(counts[key].values())==row["failures"],"Missing/inconsistent failure progress data"
        row["counts"] = [counts[key].get(phase,0) for phase in order]
        row["proportions"] = [n/row["failures"] for n in row["counts"]] if row["failures"] else None
    return list(rows.values()),order


def pages(rows,page_size=10):
    assert 2<=page_size<=12
    groups = sorted({(r["model"],r["scope"],r["budget"]) for r in rows if r["denominator"]})
    result = []
    for model,scope,budget in groups:
        baseline = next((r for r in rows if r["model"]==model and r["scope"]==scope and r["condition"]==model+"_full"),None)
        assert baseline is not None,"Missing Native Full quality row; cannot draw an unlabelled reference"
        variants = [r for r in rows if (r["model"],r["scope"],r["budget"])==(model,scope,budget) and r is not baseline and r["denominator"]]
        if not variants:
            if baseline and budget==0:
                result.append((model,scope,budget,0,[baseline]))
            continue
        width = page_size-int(baseline is not None)
        for page,start in enumerate(range(0,len(variants),width)):
            result.append((model,scope,budget,page,([baseline] if baseline else [])+variants[start:start+width]))
    return result


def build(report,output,page_size=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report,output = Path(report),Path(output)
    if output.exists():
        raise FileExistsError("Choose a NEW immutable phase-figure directory")
    sources = [report/"failure_progress_phases.csv",report/"quality_screen_init0.csv",report/"quality_combined_init0_4.csv"]
    quality = read_csv(sources[1])+read_csv(sources[2])
    rows,order = prepare(quality,read_csv(sources[0]))
    layouts = report/"layout_masks.csv"
    references = set()
    if layouts.exists():
        sources.append(layouts)
        references = {r["condition"] for r in read_csv(layouts) if str(r.get("reference","False")).lower()=="true"}
    output.mkdir(parents=True)
    artifacts = []
    colors = [plt.get_cmap("tab20")(i%20) for i in range(len(order))]
    for model,scope,budget,page,group in pages(rows,page_size):
        fig,axes = plt.subplots(1,2,figsize=(max(10,len(group)*1.3),6))
        for ax,field in zip(axes,("counts","proportions")):
            bottom = np.zeros(len(group))
            for index,phase in enumerate(order):
                values = np.asarray([r[field][index] if r[field] is not None else 0 for r in group])
                bars = ax.bar(range(len(group)),values,bottom=bottom,color=colors[index],label=phase)
                for bar,row in zip(bars,group):
                    if not row["complete"]:
                        bar.set_hatch("//")
                        bar.set_alpha(.5)
                bottom += values
            for i,row in enumerate(group):
                annotation = ("MISSING" if row["status"]=="missing" else f"{row['failures']} failures")
                if row["denominator"] and not row["failures"]:
                    annotation += "; no failed episodes"
                ax.text(i,bottom[i]+(.025 if field=="proportions" else max(1,max(bottom)*.025)),annotation,
                    ha="center",rotation=25,fontsize=7)
            labels = [r["condition"]+(" [Native Full reference]" if r["condition"]==model+"_full" else " [reference]" if r["condition"] in references else "")+
                f"\nobserved {r['denominator']}/{r['expected']} ({r['status']})" for r in group]
            ax.set_xticks(range(len(group)),labels,rotation=35,ha="right",fontsize=7)
            ax.set_ylabel("Failed-episode count" if field=="counts" else "Proportion among FAILED episodes only")
            ax.set_ylim(0,max(1,max(bottom))*1.4 if field=="counts" else 1.4)
        axes[1].legend(loc="upper left",bbox_to_anchor=(1.01,1),fontsize=7)
        fig.suptitle(f"{model} / {scope} / mask budget {budget}: {CAPTION}",fontsize=10)
        fig.text(.01,.01,"Hatching = partial observation. Missing data and zero failed episodes have no failed-only proportion; no causal interpretation.",fontsize=8)
        fig.tight_layout(rect=(0,.04,.85,.94))
        path = output/f"phases_{model}_{scope}_budget{budget}_page{page}.png"
        fig.savefig(path,dpi=150)
        plt.close(fig)
        artifacts.append(dict(path=path.name,sha256=sha(path),conditions=[r["condition"] for r in group]))
    result = dict(source_sha256=sha(__file__),csv_sources_sha256={str(p.absolute()):sha(p) for p in sources},
        phase_order=order,caption=CAPTION,plots=artifacts,observations=rows,page_size=page_size,
        caveat="Counts describe observed failures only; proportions condition on failure. Missing conditions omitted from plots, retained in observations. No ranking or completion claim.")
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
