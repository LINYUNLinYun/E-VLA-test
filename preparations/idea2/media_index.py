"""CPU-only fixed-order media handoff; no selection by success or policy output."""
import argparse
import html
import itertools
import json
import os
from pathlib import Path
from urllib.parse import quote


SUITES = ("libero_spatial","libero_object","libero_goal","libero_10")
CONDITIONS = [("fresh_serial",1)]+[(mode,interval) for mode in ("serial_reuse","async_reuse") for interval in (1,2,4)]


def relative(path,destination):
    return quote(Path(os.path.relpath(Path(path).resolve(),Path(destination).resolve())).as_posix(),safe="/.")


def build(root,figures,destination):
    root,figures,destination = Path(root),Path(figures),Path(destination)
    scenes = list(itertools.product(SUITES,(0,5),(0,1)))
    records = {}
    for suite,task,init in scenes:
        for mode,interval in CONDITIONS:
            directory = root/"episodes"/f"{mode}_r{interval}"/suite/f"task{task:02d}"/f"init{init:02d}"
            if not (directory/"complete.json").exists() or not (directory/"result.json").exists():
                continue
            result = json.loads((directory/"result.json").read_text())
            if result.get("valid"):
                records[(suite,task,init,mode,interval)] = dict(directory=directory,result=result,
                    video=directory/"video.mp4" if (directory/"video.mp4").exists() else None)
    available = sum(bool(row["video"]) for row in records.values())
    state = "Partial" if len(records)!=112 or available!=112 else "Recorded matrix"
    title = f"Idea2 media — {state.lower()} ({len(records)}/112 episode records, {available}/112 videos)"
    md = [f"# {title}","","Fixed scene and condition order; no clips selected by success. This index does not replace the raw-data audit.",""]
    body = [f"<h1>{html.escape(title)}</h1><p>Fixed scene and condition order; no selection by success. Raw-data audit is separate.</p>"]
    source_paths = []
    for suite,task,init in scenes:
        heading = f"{suite} · task {task} · init {init}"
        md.extend([f"## {heading}",""])
        body.append(f"<section><h2>{html.escape(heading)}</h2><div class='videos'>")
        for mode,interval in CONDITIONS:
            label = f"{mode} r={interval}"
            row = records.get((suite,task,init,mode,interval))
            if row:
                source_paths.append(str((row["directory"]/"complete.json").resolve()))
            if row and row["video"]:
                url = relative(row["video"],destination)
                outcome = "success" if row["result"]["success"] else "action limit"
                md.append(f"- [{label} — {outcome}]({url})")
                body.append(f"<article><h3>{html.escape(label)} <small>{outcome}</small></h3><video controls preload='none' src='{html.escape(url,quote=True)}'></video><p><a href='{html.escape(url,quote=True)}'>Open video</a></p></article>")
            else:
                md.append(f"- {label} — pending / unavailable")
                body.append(f"<article><h3>{html.escape(label)}</h3><p>Pending / unavailable</p></article>")
        md.append("")
        body.append("</div></section>")
    md.extend(["## Representative XYZ comparisons","","All 24 predefined comparisons use task0/init0. Baseline is blue; variant is orange.","",
               "| Scene / variant | XYZ | Baseline video | Variant video |","|---|---|---|---|"])
    body.append("<section><h2>Representative XYZ comparisons</h2><p>All 24 task0/init0 comparisons; baseline blue, variant orange.</p><div class='figures'>")
    figure_count = 0
    for suite in SUITES:
        baseline = records.get((suite,0,0,"fresh_serial",1))
        for mode,interval in CONDITIONS[1:]:
            variant = records.get((suite,0,0,mode,interval))
            path = figures/f"eef_{suite}_{mode}_r{interval}.png"
            name = f"{suite} / {mode} r={interval}"
            image_url = relative(path,destination) if path.exists() else None
            baseline_url = relative(baseline["video"],destination) if baseline and baseline["video"] else None
            variant_url = relative(variant["video"],destination) if variant and variant["video"] else None
            figure_count += bool(image_url)
            def link(url,label):
                return f"[{label}]({url})" if url else "pending"
            md.append(f"| {name} | {link(image_url,'XYZ figure')} | {link(baseline_url,'baseline')} | {link(variant_url,'variant')} |")
            body.append(f"<article><h3>{html.escape(name)}</h3>")
            if image_url:
                body.append(f"<a href='{html.escape(image_url,quote=True)}'><img loading='lazy' src='{html.escape(image_url,quote=True)}' alt='{html.escape(name,quote=True)}'></a>")
            else:
                body.append("<p>XYZ figure pending</p>")
            body.append("<p>"+" · ".join(f"<a href='{html.escape(url,quote=True)}'>{label}</a>" if url else f"{label} pending" for url,label in ((baseline_url,"Baseline video"),(variant_url,"Variant video")))+"</p></article>")
    body.append("</div></section>")
    document = "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Idea2 media</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font:16px/1.5 system-ui;color:#20252b}h1{font-size:1.7rem}h2{border-top:1px solid #ccd4de;padding-top:24px}h3{font-size:1rem;margin:0 0 12px}small{font-weight:400;color:#606b78}.videos,.figures{display:grid;gap:20px}.videos{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}.figures{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}article{background:#f5f7fa;border-radius:8px;padding:16px;overflow:hidden}video,img{width:100%;height:auto}a{color:#165b9b}p{margin:8px 0}@media(max-width:480px){.figures{grid-template-columns:1fr}}</style>"+"".join(body)
    return "\n".join(md)+"\n",document,dict(completed_sources=len(records),available_videos=available,representative_figures=figure_count,
                                            expected_videos=112,expected_figures=24,partial=state=="Partial",sources=source_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--figures",required=True)
    parser.add_argument("--output",required=True,help="Directory for media_index.md and media_index.html")
    args = parser.parse_args()
    destination = Path(args.output)
    markdown,document,receipt = build(args.root,args.figures,destination)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"media_index.md").write_text(markdown)
    (destination/"media_index.html").write_text(document)
    (destination/"media_index.sources.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps({key:value for key,value in receipt.items() if key!="sources"}))


if __name__=="__main__":
    main()
