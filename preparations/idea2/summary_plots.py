"""Analysis-only figures; uncertainty and incomplete denominators stay visible."""
from pathlib import Path

import numpy as np


CONDITIONS = [("fresh_serial",1)]+[(mode,interval) for mode in ("serial_reuse","async_reuse") for interval in (1,2,4)]


def label(mode,interval):
    return "Fresh serial" if mode=="fresh_serial" else f"{'Serial' if mode=='serial_reuse' else 'Async'} reuse\nr={interval}"


def plot_sources(report,root,figures):
    root = Path(root).resolve()
    # Use exactly the analyzer's completed-episode snapshot. New episodes can
    # finish while plots render, and must not be attributed to an older figure.
    episode_results = [(root/row["path"]).parent/"result.json" for row in report["videos"]]
    replay = [path for path in root.rglob("e2.json") if "attempts" not in path.parts]
    entries = []
    for figure in figures:
        name = Path(figure).name
        if name.startswith("e3_"):
            sources,partial = episode_results,report["status"]["partial"]
        elif name.startswith("e2_"):
            sources,partial = replay,len(replay)!=4
        elif name.startswith("e1b_"):
            index = int(Path(name).stem.rsplit("_",1)[-1])-1
            sources,partial = [Path(report["e1b"][index]["path"]).resolve()],False
        else:
            sources,partial = [],False
            for record in report["videos"]:
                if record["task"]!=0 or record["init"]!=0:
                    continue
                expected = f"eef_{record['suite']}_{record['mode']}_r{record['interval']}.png"
                if name==expected:
                    sources.append((root/record["path"]).parent/"telemetry.jsonl")
                    baseline = root/"episodes/fresh_serial_r1"/record["suite"]/"task00/init00/telemetry.jsonl"
                    sources.append(baseline)
        entries.append(dict(figure=str(Path(figure).resolve()),source_paths=[str(path.resolve()) for path in sources],
                            partial=partial,e3_status=report["status"],
                            caveat="Figures describe recorded data; final audit and causal interpretation are separate"))
    return entries


def plot_summaries(report,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    destination = Path(destination)
    outputs = []
    def save(fig,name):
        path = destination/name
        fig.savefig(path,dpi=180,bbox_inches="tight",pad_inches=.35)
        plt.close(fig)
        outputs.append(str(path))
    data = {(row["mode"],row["interval"]):row for row in report["e3"]}
    x = np.arange(len(CONDITIONS))
    names = [label(*key) for key in CONDITIONS]
    colors = ["tab:blue"]+["tab:orange"]*6
    status = report["status"]
    qualifier = "Partial" if status["partial"] else "Recorded matrix"
    if data:
        fig,axis = plt.subplots(figsize=(10,4.8))
        values = [100*data[key]["success_rate"] if key in data else 0 for key in CONDITIONS]
        bars = axis.bar(x,values,color=colors,alpha=.75)
        for key,bar in zip(CONDITIONS,bars,strict=True):
            row = data.get(key)
            text = f"{row['successes']}/{row['episodes']}" if row else "pending"
            axis.annotate(text,(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,5),textcoords="offset points",ha="center")
        axis.set(xticks=x,xticklabels=names,ylabel="Task success (%)",ylim=(0,112),
                 title=f"E3 {qualifier.lower()}: {status['recorded_e3_episodes']}/112 recorded episodes")
        axis.text(.5,-.23,"Labels show successes / valid recorded episodes; target n=16 per condition. Final raw-data audit is separate.",transform=axis.transAxes,ha="center",fontsize=9)
        save(fig,"e3_success_counts.png")

        phases = ["initial","approach","contact","grasp_or_manipulate","transport","goal_alignment","release_or_final_action","success","unknown"]
        hatches = [".","/","\\","o","x","-","+","*","O"]
        fig,axis = plt.subplots(figsize=(11,5.3))
        bottom = np.zeros(len(CONDITIONS))
        handles = []
        for index,(phase,hatch) in enumerate(zip(phases,hatches,strict=True)):
            counts = np.array([data.get(key,{}).get("failure_furthest_phase",{}).get(phase,0) for key in CONDITIONS])
            if not counts.any():
                continue
            axis.bar(x,counts,bottom=bottom,color=colors,alpha=.3+.065*index,hatch=hatch,edgecolor="0.2",linewidth=.4)
            bottom += counts
            handles.append(Patch(facecolor="white",edgecolor="0.2",hatch=hatch,label=phase.replace("_"," ")))
        axis.set(xticks=x,xticklabels=names,ylabel="Failed episodes (count)",
                 title=f"E3 {qualifier.lower()}: farthest observed stage in failed episodes")
        axis.set_ylim(0,max(1,float(bottom.max()))+1)
        axis.legend(handles=handles,ncol=3,loc="upper center",bbox_to_anchor=(.5,-.17),fontsize=8)
        axis.text(.5,-.43,"Heuristic progress evidence, not a causal diagnosis of failure. Partial condition denominators match the success figure.",transform=axis.transAxes,ha="center",fontsize=9)
        save(fig,"e3_failure_furthest_stage.png")

    if report["e2_age"]:
        fig,axes = plt.subplots(1,2,figsize=(11,5))
        fig.subplots_adjust(bottom=.27,wspace=.26)
        for horizon,color,marker in (("first5","tab:blue","o"),("full50","tab:orange","^")):
            rows = sorted((row for row in report["e2_age"] if row["horizon"]==horizon),key=lambda row:row["request_age"])
            ages = [row["request_age"] for row in rows]
            name = "First 5 actions" if horizon=="first5" else "All 50 actions"
            axes[0].plot(ages,[row["pooled_position_rmse"] for row in rows],color=color,marker=marker,alpha=.75,label=name)
            axes[1].plot(ages,[100*row["gripper_sign_mismatch_rate"] for row in rows],color=color,marker=marker,alpha=.75,label=name)
        n = {row["request_age"]:row["unique_request_source_pairs"] for row in report["e2_age"]}
        for axis in axes:
            axis.set_xticks(sorted(n),[f"{age}\nn={n[age]}" for age in sorted(n)])
            axis.set_xlabel("Condition age (generation requests)")
            axis.grid(alpha=.2)
            axis.legend()
        axes[0].set(ylabel="Position action RMSE (decoded action units)",title="E2 fixed replay: position component")
        axes[1].set(ylabel="Gripper sign mismatches (%)",title="E2 fixed replay: gripper sign")
        fig.text(.5,.015,"Each point pools unique (suite, request, prefix source) pairs; n labels count those pairs.\nPosition pools n×horizon×3 coordinates; gripper rates divide mismatches by n×horizon. No conversion to millimeters.",ha="center",va="bottom",fontsize=9)
        save(fig,"e2_staleness_unique_pairs.png")

    for index,row in enumerate(report["e1b"]):
        if not row.get("eager_summary"):
            continue
        groups = [("Original eager",row.get("original_eager_summary"),"tab:blue",""),
                  ("Static metadata eager",row["eager_summary"],"tab:blue","//"),
                  ("CUDA Graph",row["summary"],"tab:orange","")]
        groups = [item for item in groups if item[1]]
        fig,axes = plt.subplots(1,2,figsize=(12,4.7))
        width = .8/len(groups)
        for axis,(metric,title) in zip(axes,(("action_ready_host_ms","Action chunk available"),("all_done_host_ms","AE and prefix both complete")),strict=True):
            for group_index,(name,summary,color,hatch) in enumerate(groups):
                values = [summary[mode][metric]["p50"] for mode in ("serial","two_stream")]
                upper = [max(0,summary[mode][metric]["p95"]-value) for mode,value in zip(("serial","two_stream"),values,strict=True)]
                centers = np.arange(2)+(group_index-(len(groups)-1)/2)*width
                bars = axis.bar(centers,values,width,color=color,alpha=.75,hatch=hatch,label=name,
                                yerr=np.array([np.zeros(2),upper]),capsize=3)
                for bar,value in zip(bars,values,strict=True):
                    axis.annotate(f"{value:.1f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),ha="center",xytext=(0,3),textcoords="offset points",fontsize=8)
            axis.set(xticks=[0,1],xticklabels=["Serial AE → prefix","Two streams"],ylabel="Completed host latency (ms)",title=title)
            axis.legend(fontsize=8)
        uuid = row["identity"].get("gpu_uuid","not recorded")
        fig.suptitle(f"E1b same-GPU controls (UUID {uuid})",fontsize=11)
        fig.text(.5,-.03,"Bars: P50; upper whiskers: P95. Static eager and CUDA Graph share the same hoisted metadata wrapper.\nOriginal eager is a separate control; improvements from metadata hoisting are not attributed to graph replay.",ha="center",fontsize=9)
        save(fig,f"e1b_same_gpu_latency_{index+1}.png")
    return outputs
