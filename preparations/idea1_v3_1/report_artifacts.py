"""CPU report artifacts from explicit dataset condition directories; no experiments."""
import argparse
from collections import Counter
import csv
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from urllib.parse import quote,unquote,urlsplit

import numpy as np

from .common import atomic_json,read_manifest
from .same_process_parity import sha


CAVEAT = "Exploratory observations; masks/confirmation selected using screen outcomes. Combined init0-4 includes selection data; not an independent held-out estimate. Partial conditions are never complete or ranked."
PROGRESS = "PROGRESS: furthest heuristic phase is descriptive progress evidence, not failure cause."
DISTANCE_CAVEAT = "Both distance metrics include route and progress-speed differences. Neither is path-only or a longitudinal/lateral decomposition. At LIBERO 20 Hz the panels may be identical."


def read_json(path,inputs):
    path = Path(path)
    inputs[str(path.absolute())] = sha(path)
    return json.loads(path.read_text())


def collect(index,manifest,inputs):
    """Explicit glob follows each mapped alias without recursive bundle traversal."""
    assert index["manifest_hash"]==manifest["manifest_hash"]
    definitions = {c["id"]:c for c in manifest["conditions"]+manifest.get("benchmark_references",[])}
    mappings,data = {},{}
    for mapping in index["conditions"]:
        cid = mapping["condition"]
        assert cid in definitions and cid not in mappings,"Unknown/duplicate scientific condition"
        assert mapping["model"]==definitions[cid]["model"]
        mappings[cid] = Path(mapping["actual_directory"])
        rows = {}
        for path in sorted(mappings[cid].glob("episodes/*/task*/init*/result.json")):
            if any(".invalid." in part for part in path.parts):
                continue
            row = read_json(path,inputs)
            marker = path.with_name("complete.json")
            if not marker.exists() or not row.get("valid"):
                continue
            complete = read_json(marker,inputs)
            if not complete.get("valid"):
                continue
            assert row["manifest_hash"]==complete["manifest_hash"]==manifest["manifest_hash"]
            assert row["condition"]==cid
            key = tuple(row[k] for k in ("suite","task","init"))
            assert key[0] in manifest["suites"] and key[1] in manifest["tasks"]
            assert key[2] in set(manifest["screen_inits"]+manifest["confirm_inits"])
            assert path.parent.name==f"init{key[2]:02d}" and path.parent.parent.name==f"task{key[1]:02d}"
            assert path.parent.parent.parent.name==key[0] and key not in rows,"Duplicate/wrong scene key"
            assert isinstance(row["success"],bool) and row.get("init_vector_sha256")
            rows[key] = dict(row,_path=path)
        data[cid] = rows
    return definitions,mappings,data


def quality_tables(manifest,definitions,data):
    quality,phases = [],[]
    scopes = {"screen_init0":manifest["screen_inits"],
        "combined_init0_4":sorted(set(manifest["screen_inits"]+manifest["confirm_inits"]))}
    for scope,inits in scopes.items():
        for cid,observed in data.items():
            condition = definitions[cid]
            if condition.get("benchmark_only",False):
                continue
            baseline = data.get(condition["model"]+"_full",{})
            for suite in ["ALL"]+list(manifest["suites"]):
                expected = {(s,t,i) for s in manifest["suites"] if suite=="ALL" or s==suite
                    for t in manifest["tasks"] for i in inits}
                rows = {k:v for k,v in observed.items() if k in expected}
                paired = {k:v for k,v in rows.items() if k in baseline and
                    v["init_vector_sha256"]==baseline[k]["init_vector_sha256"]}
                full_success = sum(baseline[k]["success"] for k in paired)
                paired_success = sum(v["success"] for v in paired.values())
                quality.append(dict(scope=scope,condition=cid,model=condition["model"],budget=len(condition["skip"]),
                    suite=suite,successes=sum(v["success"] for v in rows.values()),denominator=len(rows),expected=len(expected),
                    complete=set(rows)==expected,success_rate=sum(v["success"] for v in rows.values())/len(rows) if rows else None,
                    matched_full_denominator=len(paired),matched_full_successes=full_success,matched_variant_successes=paired_success,
                    exact_matching_full_retention=paired_success/full_success if full_success else None,
                    complete_matching_full=set(paired)==expected,caveat=CAVEAT))
                failures = [v for v in rows.values() if not v["success"]]
                for phase,count in sorted(Counter(v.get("furthest_phase","unknown") for v in failures).items()):
                    phases.append(dict(scope=scope,condition=cid,suite=suite,progress_phase=phase,count=count,
                        failed_episode_denominator=len(failures),proportion=count/len(failures),interpretation=PROGRESS))
    return quality,phases


def trajectory(path,inputs):
    inputs[str(path.absolute())] = sha(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows,"Missing actual telemetry"
    xyz = np.asarray([r["eef_pos"] for r in rows],float)
    times = np.asarray([r["sim_time_s"] for r in rows],float)
    steps = np.asarray([r["step"] for r in rows],int)
    assert xyz.shape==(len(rows),3) and np.isfinite(xyz).all() and np.isfinite(times).all()
    assert (np.diff(times)>=0).all() and (np.diff(steps)>0).all()
    return dict(xyz=xyz,times=times,steps=steps)


def distances(full,variant,tolerance=.025001):
    """Actual metres to mm; no padding, normalization or command comparisons."""
    common = sorted(set(full["steps"]) & set(variant["steps"]))
    fi = {int(s):i for i,s in enumerate(full["steps"])}
    vi = {int(s):i for i,s in enumerate(variant["steps"])}
    step_rows = [dict(step=int(s),full_sim_time_s=float(full["times"][fi[s]]),variant_sim_time_s=float(variant["times"][vi[s]]),
        distance_mm=float(np.linalg.norm(full["xyz"][fi[s]]-variant["xyz"][vi[s]])*1000)) for s in common]
    lower = max(float(full["times"][0]),float(variant["times"][0]))
    upper = min(float(full["times"][-1]),float(variant["times"][-1]))
    time_rows = []
    for i,t in enumerate(variant["times"]):
        if not lower<=t<=upper:
            continue
        j = int(np.argmin(np.abs(full["times"]-t)))
        delta = abs(float(full["times"][j]-t))
        if delta<=tolerance:
            time_rows.append(dict(variant_step=int(variant["steps"][i]),full_step=int(full["steps"][j]),
                variant_sim_time_s=float(t),full_sim_time_s=float(full["times"][j]),nearest_time_delta_s=delta,
                distance_mm=float(np.linalg.norm(full["xyz"][j]-variant["xyz"][i])*1000)))
    coverage = dict(common_start_s=lower,common_end_s=upper if upper>=lower else None,
        full_start_s=float(full["times"][0]),full_end_s=float(full["times"][-1]),
        variant_start_s=float(variant["times"][0]),variant_end_s=float(variant["times"][-1]),
        full_duration_s=float(full["times"][-1]-full["times"][0]),variant_duration_s=float(variant["times"][-1]-variant["times"][0]),
        same_step_points=len(step_rows),nearest_actual_time_points=len(time_rows),nearest_tolerance_s=tolerance,
        interpretation="Euclidean actual eef_pos distance in mm; no padding or normalized command error. "+DISTANCE_CAVEAT)
    return step_rows,time_rows,coverage


def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields or list(rows[0]) if rows else fields or ["no_observations"])
        writer.writeheader()
        writer.writerows(rows)


def plot_quality(output,quality):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    observed = [r for r in quality if r["denominator"]>0]
    groups = sorted({(r["scope"],r["model"],r["budget"]) for r in observed})
    paths = []
    for scope,model,budget in groups:
        candidates = [r for r in observed if (r["scope"],r["model"],r["budget"],r["suite"])==(scope,model,budget,"ALL")]
        # Bounded pages preserve readable labels even when many layouts exist.
        for page,start in enumerate(range(0,len(candidates),12)):
            rows = candidates[start:start+12]
            fig,ax = plt.subplots(figsize=(max(6,len(rows)*1.1),4))
            bars = ax.bar(range(len(rows)),[r["success_rate"] or 0 for r in rows],
                color=["tab:blue" if r["condition"]==model+"_full" else "tab:orange" for r in rows])
            for bar,row in zip(bars,rows):
                if not row["complete"]:
                    bar.set_hatch("//")
                    bar.set_alpha(.4)
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,
                    f"{row['successes']}/{row['denominator']} (expected {row['expected']})",ha="center",fontsize=7,rotation=25)
            ax.set_xticks(range(len(rows)),[r["condition"] for r in rows],rotation=35,ha="right",fontsize=8)
            ax.set_ylim(0,1.25)
            ax.set_ylabel("Observed success fraction")
            ax.set_title(f"{model}: {scope}, mask budget {budget}; hatched = partial")
            baseline = next((r for r in quality if r["condition"]==model+"_full" and r["scope"]==scope and r["suite"]=="ALL"),None)
            if budget and baseline and baseline["denominator"]:
                ax.axhline(baseline["success_rate"],color="tab:blue",alpha=.55,label=
                    f"Native Full {baseline['successes']}/{baseline['denominator']} of expected {baseline['expected']}"
                    +( " (partial)" if not baseline["complete"] else ""))
                ax.legend(fontsize=7)
            fig.text(.01,.01,"Exploratory; combined scope includes selection screen. No ranking.",fontsize=8)
            fig.tight_layout(rect=(0,.06,1,1))
            path = output/f"quality_{model}_{scope}_budget{budget}_page{page}.png"
            fig.savefig(path,dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_pair(path,full,variant,step_rows,time_rows,coverage,title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11,5))
    xyz = fig.add_subplot(121,projection="3d")
    for record,color,label,style in ((full,"tab:blue","Native Full","-"),(variant,"tab:orange","Variant","--")):
        points = record["xyz"]*1000
        xyz.plot(*points.T,color=color,linestyle=style,label=label,alpha=.55 if label=="Native Full" else 1,
            marker="o" if label=="Variant" else None,markevery=max(1,len(points)//15),markersize=2)
        xyz.scatter(*points[0],color=color,marker="o",s=30)
        xyz.scatter(*points[-1],color=color,marker="X",s=45)
    xyz.set(xlabel="X (mm)",ylabel="Y (mm)",zlabel="Z (mm)",title="Actual eef_pos: circle=start, X=end")
    xyz.legend()
    distance = fig.add_subplot(222)
    distance.plot([r["variant_sim_time_s"] for r in step_rows],[r["distance_mm"] for r in step_rows],color="tab:orange")
    distance.set(xlabel="Variant simulation time (s)",ylabel="Distance (mm)",title="Same actual step")
    nearest = fig.add_subplot(224)
    nearest.plot([r["variant_sim_time_s"] for r in time_rows],[r["distance_mm"] for r in time_rows],color="tab:orange")
    nearest.set(xlabel="Variant simulation time (s)",ylabel="Distance (mm)",title="Nearest actual Full timestep within 25 ms")
    fig.suptitle(title,fontsize=11)
    fig.text(.01,.035,f"Full duration {coverage['full_duration_s']:.3f}s; variant {coverage['variant_duration_s']:.3f}s; "
        f"common simulation coverage [{coverage['common_start_s']:.3f}, {coverage['common_end_s']}]s. No padding.",fontsize=8)
    fig.text(.01,.01,DISTANCE_CAVEAT,fontsize=7)
    fig.tight_layout(rect=(0,.08,1,.95))
    fig.savefig(path,dpi=150)
    plt.close(fig)


def relative_link(path,output):
    assert Path(path).is_file(),f"Missing media target: {path}"
    return quote(os.path.relpath(Path(path).absolute(),Path(output).absolute()),safe="/.-_")


def audit_links(output,links):
    checked = []
    for link in links:
        assert not urlsplit(link).scheme and not link.startswith("/")
        target = Path(output)/unquote(link)
        assert target.is_file(),f"Broken relative link {link}"
        checked.append(link)
    return checked


def audit_documents(output):
    """Audit actual rendered Markdown/HTML hrefs, not merely the planned list."""
    output = Path(output)
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self,tag,attrs):
            self.links.extend(value for name,value in attrs if name in ("href","src"))
    parser = Links()
    parser.feed((output/"media.html").read_text())
    markdown = re.findall(r"\]\(([^)\n]+)\)",(output/"media.md").read_text())
    return audit_links(output,sorted(set(parser.links+markdown)))


def build(index,manifest,output,conditions,inputs=None,plots=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Report artifacts are immutable snapshots; choose a new output")
    inputs = {} if inputs is None else dict(inputs)
    definitions,mappings,data = collect(index,manifest,inputs)
    assert conditions is not None,"Pass explicit conditions or authenticated promotion selection"
    assert len(conditions)==len(set(conditions)) and all(cid in mappings for cid in conditions)
    output.mkdir(parents=True)
    quality,phases = quality_tables(manifest,definitions,data)
    for scope in ("screen_init0","combined_init0_4"):
        write_csv(output/f"quality_{scope}.csv",[r for r in quality if r["scope"]==scope])
    write_csv(output/"failure_progress_phases.csv",phases)
    layouts = [dict(condition=cid,model=c["model"],layout=c.get("layout"),budget=len(c["skip"]),
        skip=json.dumps(c["skip"]),keep=json.dumps(c.get("keep",[])),reference=c.get("reference",False),benchmark_only=c.get("benchmark_only",False),
        actual_directory=str(mappings[cid])) for cid,c in definitions.items() if cid in mappings]
    write_csv(output/"layout_masks.csv",layouts)
    performance = []
    source_mappings = {r["condition"]:r for r in index["conditions"]}
    for cid,directory in mappings.items():
        mapping = source_mappings[cid]
        path = Path(mapping.get("benchmark_source") or directory/"benchmark.json")
        if path.exists():
            if mapping.get("benchmark_sha256"):
                assert sha(path)==mapping["benchmark_sha256"]
            record = read_json(path,inputs)
            assert record["manifest_hash"]==manifest["manifest_hash"] and record["condition"]["id"]==cid
            performance.append(dict(condition=cid,source_path=str(path.absolute()),source_sha256=sha(path),
                summary_json=json.dumps(record.get("summary",{}),sort_keys=True),
                caveat="Raw timing only; no new speedup ratio without authenticated same-GPU provenance."))
    write_csv(output/"performance_raw.csv",performance)
    links,media,issues = [],[],[]
    quality_plots = plot_quality(output,quality) if plots else []
    for path in quality_plots:
        links.append(relative_link(path,output))
    for cid in conditions:
        c = definitions[cid]
        if c.get("benchmark_only",False) or cid==c["model"]+"_full":
            continue
        baseline = c["model"]+"_full"
        for suite in manifest["suites"]:
            key = (suite,0,0)
            a,b = data.get(baseline,{}).get(key),data.get(cid,{}).get(key)
            if not a or not b or a["init_vector_sha256"]!=b["init_vector_sha256"]:
                issues.append(dict(condition=cid,scene=list(key),reason="Missing same-key native Full/variant or official init hash mismatch"))
                continue
            ap,bp = a["_path"].with_name("telemetry.jsonl"),b["_path"].with_name("telemetry.jsonl")
            if not ap.exists() or not bp.exists():
                issues.append(dict(condition=cid,scene=list(key),reason="Missing actual telemetry; no trajectory fabricated"))
                continue
            full,variant = trajectory(ap,inputs),trajectory(bp,inputs)
            step_rows,time_rows,coverage = distances(full,variant)
            prefix = f"{cid}_{suite}_task00_init00"
            write_csv(output/f"{prefix}_same_step_mm.csv",step_rows)
            write_csv(output/f"{prefix}_nearest_simtime_mm.csv",time_rows)
            row = dict(condition=cid,baseline=baseline,scene=list(key),coverage=coverage,links={})
            if plots:
                png = output/f"{prefix}.png"
                plot_pair(png,full,variant,step_rows,time_rows,coverage,f"{cid} vs Native Full: {suite}, task0/init0")
                row["links"]["XYZ and distance PNG"] = relative_link(png,output)
            for label,path in (("Full telemetry",ap),("Variant telemetry",bp),
                ("Full video",a["_path"].with_name("video.mp4")),("Variant video",b["_path"].with_name("video.mp4"))):
                if path.exists():
                    inputs[str(path.absolute())] = sha(path)
                    row["links"][label] = relative_link(path,output)
                else:
                    issues.append(dict(condition=cid,scene=list(key),reason=f"{label} unavailable"))
            links.extend(row["links"].values())
            media.append(row)
    markdown = ["# v3.1 observed data artifacts","",CAVEAT,"",PROGRESS,"",DISTANCE_CAVEAT,"","## Data tables",""]
    document = ["<!doctype html><meta charset='utf-8'><title>v3.1 observed artifacts</title>",
        "<h1>v3.1 observed data artifacts</h1>",f"<p>{html.escape(CAVEAT)}</p><p>{html.escape(PROGRESS)}</p><p>{html.escape(DISTANCE_CAVEAT)}</p>"]
    document.append("<h2>Data tables</h2>")
    for path in sorted(output.glob("*.csv")):
        link = relative_link(path,output)
        links.append(link)
        markdown.append(f"- [{path.name}]({link})")
        document.append(f'<p><a href="{html.escape(link,quote=True)}">{html.escape(path.name)}</a></p>')
    markdown.extend(["","## Quality plots",""])
    document.append("<h2>Quality plots</h2>")
    for path in quality_plots:
        link = relative_link(path,output)
        markdown.append(f"- [{path.name}]({link})")
        document.append(f'<p><a href="{html.escape(link,quote=True)}">{html.escape(path.name)}</a></p>')
    for row in media:
        title = f"{row['condition']}: {row['scene']}"
        markdown.extend(["",f"## {title}","",json.dumps(row["coverage"],sort_keys=True),""])
        document.append(f"<h2>{html.escape(title)}</h2><p>{html.escape(json.dumps(row['coverage'],sort_keys=True))}</p>")
        for label,link in row["links"].items():
            markdown.append(f"- [{label}]({link})")
            document.append(f'<p><a href="{html.escape(link,quote=True)}">{html.escape(label)}</a></p>')
    if issues:
        markdown.extend(["","## Missing evidence","",*['- '+json.dumps(issue,sort_keys=True) for issue in issues]])
        document.append("<h2>Missing evidence</h2><pre>"+html.escape(json.dumps(issues,indent=2))+"</pre>")
    (output/"media.md").write_text("\n".join(markdown)+"\n")
    (output/"media.html").write_text("\n".join(document)+"\n")
    planned = audit_links(output,links)
    checked = audit_documents(output)
    assert set(checked)==set(planned)
    artifacts = {str(p.relative_to(output)):sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    result = dict(manifest_hash=manifest["manifest_hash"],source_sha256=sha(__file__),inputs_sha256=inputs,
        artifacts_sha256=artifacts,relative_links_checked=checked,conditions_requested=conditions,media=media,missing_evidence=issues,
        quality_complete_conditions={scope:sum(r["complete"] for r in quality if r["scope"]==scope and r["suite"]=="ALL")
            for scope in ("screen_init0","combined_init0_4")},
        completion_is_not_implied_by_report=True,scientific_caveat=CAVEAT,phase_interpretation=PROGRESS,
        distance_interpretation=DISTANCE_CAVEAT)
    atomic_json(output/"artifact_index.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--dataset-index",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--conditions",nargs="+")
    parser.add_argument("--promotion")
    args = parser.parse_args()
    inputs = {}
    manifest = read_manifest(args.manifest,verify_sources=True)
    inputs[str(Path(args.manifest).absolute())] = sha(args.manifest)
    index = read_json(args.dataset_index,inputs)
    conditions = args.conditions
    promotion = args.promotion or index.get("promotion_path")
    if conditions is None and promotion and Path(promotion).is_file():
        selected = read_json(promotion,inputs)
        assert selected["manifest_hash"]==manifest["manifest_hash"]
        conditions = selected["selected_conditions"]
    if conditions is None:
        parser.error("XYZ selection requires existing promotion or explicit --conditions")
    result = build(index,manifest,args.output,conditions,inputs)
    print(json.dumps(dict(output=args.output,artifacts=len(result["artifacts_sha256"]),
        missing_evidence=len(result["missing_evidence"]),completion_is_not_implied_by_report=True)))


if __name__=="__main__":
    main()
