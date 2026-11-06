"""Browse saved episodes/videos/actual XYZ telemetry; no new scientific audit."""
import argparse
from collections import Counter,defaultdict
import csv
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import quote,unquote,urlsplit

from .common import ROOT,atomic_json,read_manifest
from .dataset_bundle_v3 import collect as collect_quality
from .same_process_parity import sha

SCOPE = "Inventory and saved-file identity/hash/link checks only; NOT a new scientific audit or video-decode audit. Raw XYZ links are original telemetry.jsonl eef_pos (metres) and sim_time_s (seconds), unchanged at full stored precision."


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def load(path,inputs=None,expected=None):
    path = absolute(path)
    digest = sha(path)
    if expected is not None:
        assert digest==expected,f"Saved input SHA mismatch: {path}"
    if inputs is not None:
        inputs[str(path.resolve())] = digest
    return json.loads(path.read_text())


def url(path,output):
    import os
    assert Path(path).is_file(),f"Missing link target: {path}"
    return quote(os.path.relpath(Path(path).resolve(),Path(output).absolute()),safe="/.-_")


def episode(path,condition,manifest,category,arm,output):
    path = Path(path).resolve()
    result,identity,marker = (load(path/name) for name in ("result.json","identity.json","complete.json"))
    assert result["valid"] and marker["valid"]
    assert all(row["manifest_hash"]==manifest["manifest_hash"] for row in (result,identity,marker))
    assert result["condition"]==condition["id"] and result["model"]==condition["model"] and identity["condition"]==condition
    key = tuple(result[k] for k in ("suite","task","init"))
    assert tuple(identity[k] for k in ("suite","task","init"))==key
    assert (path.parent.parent.name,path.parent.name,path.name)==(key[0],f"task{key[1]:02d}",f"init{key[2]:02d}")
    assert key[0] in manifest["suites"] and key[1] in manifest["tasks"] and key[2] in set(manifest["screen_inits"]+manifest["confirm_inits"])
    assert result["init_vector_sha256"]==identity["init_vector_sha256"] and result["init_vector_sha256"]
    assert isinstance(result["success"],bool) and isinstance(result["video"],bool)
    actions,predictions = result["actions"],result["predictions"]
    assert type(actions) is int and type(predictions) is int and actions>0 and predictions>0
    R = condition["execute_length"]
    R = R[key[0]] if isinstance(R,dict) else R
    assert 0<R<=condition["horizon"] and predictions==math.ceil(actions/R)
    assert math.isfinite(result["wall_s"]) and result["wall_s"]>0
    telemetry = path/"telemetry.jsonl"
    assert telemetry.is_file() and telemetry.stat().st_size>0
    # Read just the first actual sample for format/context. The linked complete
    # telemetry is NOT decoded/replayed/re-audited or copied into synthetic XYZ.
    with telemetry.open() as stream:
        first = json.loads(next(line for line in stream if line.strip()))
    xyz = first["eef_pos"]
    assert first["step"]==0 and len(xyz)==3 and all(math.isfinite(v) for v in xyz) and math.isfinite(first["sim_time_s"])
    video = path/"video.mp4"
    recorded = video.is_file()
    assert not result["video"] or recorded,"Result declares missing saved video"
    row = dict(category=category,model=condition["model"],condition=condition["id"],arm=arm,suite=key[0],task=key[1],init=key[2],
        success=result["success"],actions=actions,predictions=predictions,wall_s=result["wall_s"],
        init_vector_sha256=result["init_vector_sha256"],actual_episode_directory=str(path),
        first_eef_x_m=xyz[0],first_eef_y_m=xyz[1],first_eef_z_m=xyz[2],first_sim_time_s=first["sim_time_s"],
        video_declared=result["video"],video_recorded=recorded,recorded_video_decode_marker=marker.get("video_decoded",False),
        result_sha256=sha(path/"result.json"),identity_sha256=sha(path/"identity.json"),complete_sha256=sha(path/"complete.json"),
        telemetry_sha256=sha(telemetry),video_sha256=sha(video) if recorded else "",
        result_url=url(path/"result.json",output),identity_url=url(path/"identity.json",output),complete_url=url(path/"complete.json",output),
        telemetry_url=url(telemetry,output),xyz_raw_url=url(telemetry,output),video_url=url(video,output) if recorded else "")
    return key,row


def scan(directory):
    for path in sorted(Path(directory).glob("*/task*/init*/result.json")):
        if ".invalid." not in str(path) and ".invalid." not in str(path.resolve()) and (path.parent/"complete.json").is_file():
            yield path.parent


def inventory(index,manifest,promotion,output):
    assert index["manifest_hash"]==promotion["manifest_hash"]==manifest["manifest_hash"]
    definitions = {c["id"]:c for c in manifest["conditions"]+manifest.get("benchmark_references",[])}
    quality_ids = {c["id"] for c in manifest["conditions"] if not c.get("benchmark_only",False)}
    selected = promotion["selected_conditions"]
    assert len(selected)==len(set(selected)) and set(selected)<=quality_ids
    assert set(index["profile_origins"])=={definitions[cid]["model"] for cid in quality_ids}
    quality = collect_quality(index,manifest)  # frozen explicit episode_source collector + provenance guards
    rows,seen,expected_quality = [],set(),{}
    mappings = {r["condition"]:r for r in index["conditions"]}
    for cid,data in quality.items():
        condition = definitions[cid]
        if condition.get("benchmark_only",False):
            assert not data,"Benchmark-only reference must not be counted as quality"
            continue
        inits = manifest["confirm_inits"] if cid in selected else manifest["screen_inits"]
        expected = {(s,t,i) for s in manifest["suites"] for t in manifest["tasks"] for i in inits}
        assert set(data)==expected,f"Quality source does not match predeclared 200/40 selection: {cid}"
        expected_quality[cid] = len(expected)
        source = absolute(mappings[cid]["episode_source"])
        for key in sorted(data):
            path = source/key[0]/f"task{key[1]:02d}"/f"init{key[2]:02d}"
            assert path.resolve() not in seen,"Canonical quality episode duplicated through aliases"
            actual,row = episode(path,condition,manifest,"quality","episodes",output)
            assert actual==key
            row["expected_quality_condition_episodes"] = len(expected)
            row["selected_for_200"] = cid in selected
            rows.append(row)
            seen.add(path.resolve())
    assert set(expected_quality)==quality_ids
    profile_keys = {(s["suite"],s["task"],s["init"]) for s in manifest["e2"]["episodes"]}
    adopted_counts = {}
    for model,origin in index["profile_origins"].items():
        condition = definitions[model+"_full"]
        for variant in manifest["e2"]["variants"]:
            arm = variant["id"]
            keys = set()
            for path in scan(absolute(origin["root"])/(model+"_full")/arm):
                assert path.resolve() not in seen,"Adopted profile source duplicates another canonical episode"
                key,row = episode(path,condition,manifest,"adopted_e2_profile",arm,output)
                assert key not in keys and key in profile_keys
                keys.add(key)
                rows.append(row)
                seen.add(path.resolve())
            assert keys==profile_keys,f"Incomplete adopted E2 arm: {model}/{arm}"
            adopted_counts[f"{model}/{arm}"] = len(keys)
    original = absolute(index["original_root"])
    for cid in sorted(quality_ids):
        for variant in manifest["e2"]["variants"]:
            for path in scan(original/cid/variant["id"]):
                if path.resolve() in seen:
                    continue
                marker,result = load(path/"complete.json"),load(path/"result.json")
                if not marker.get("valid") or not result.get("valid"):
                    continue  # historical invalid completions are outside valid inventory
                key,row = episode(path,definitions[cid],manifest,"archived_not_adopted",variant["id"],output)
                assert key in profile_keys
                rows.append(row)
                seen.add(path.resolve())
    assert len(rows)==len(seen)
    # Optional fields are explicit in every CSV row, including archived profiles.
    for row in rows:
        row.setdefault("expected_quality_condition_episodes","")
        row.setdefault("selected_for_200",False)
    return rows,dict(expected_quality_by_condition=expected_quality,expected_quality_total=sum(expected_quality.values()),
        adopted_e2_by_model_arm=adopted_counts,counts_by_category=dict(Counter(r["category"] for r in rows)),
        recorded_videos_by_category=dict(Counter(r["category"] for r in rows if r["video_recorded"])),
        archived_not_adopted_excluded_from_quality_and_adopted_denominators=True)


def check_links(output,links):
    for link in set(links):
        assert not urlsplit(link).scheme and not link.startswith("/")
        assert (Path(output)/unquote(link)).is_file(),f"Broken inventory link: {link}"
    return len(set(links))


def documents(output,rows):
    groups = defaultdict(list)
    for row in rows:
        if row["video_recorded"]:
            groups[(row["category"],row["model"],row["condition"],row["arm"])].append(row)
    md = ["# Saved episode/video/actual XYZ inventory","",SCOPE,"","[All episodes CSV](episodes.csv) · [All episodes JSON](episodes.json)",""]
    doc = ["<!doctype html><html><meta charset='utf-8'><title>Saved episode inventory</title>",
        "<style>body{font:15px system-ui;max-width:1100px;margin:2em auto}summary{cursor:pointer;margin:1em 0}video{width:480px;max-width:100%}article{border-top:1px solid #ddd;padding:1em 0}</style>",
        "<h1>Saved episode/video/actual XYZ inventory</h1>",f"<p>{html.escape(SCOPE)}</p>",
        '<p><a href="episodes.csv">All episodes CSV</a> · <a href="episodes.json">All episodes JSON</a></p>']
    for group,records in sorted(groups.items()):
        title = " / ".join(group)
        md.extend(["",f"## {title} ({len(records)} recorded videos)",""])
        doc.append(f"<details><summary>{html.escape(title)} ({len(records)} recorded videos)</summary>")
        for row in records:
            context = f"{row['suite']} task{row['task']:02d}/init{row['init']:02d}; success={row['success']}; actions={row['actions']}; predictions={row['predictions']}"
            md.append(f"- {context}: [video]({row['video_url']}) · [raw XYZ / telemetry]({row['xyz_raw_url']}) · [result]({row['result_url']})")
            doc.extend([f"<article><p>{html.escape(context)}</p>",
                f'<video controls preload="none" src="{html.escape(row["video_url"],quote=True)}"></video>',
                f'<p><a href="{html.escape(row["xyz_raw_url"],quote=True)}">Raw XYZ / actual telemetry</a> · '
                f'<a href="{html.escape(row["result_url"],quote=True)}">Result and context</a></p></article>'])
        doc.append("</details>")
    doc.append("</html>")
    (output/"videos.md").write_text("\n".join(md)+"\n")
    (output/"videos.html").write_text("\n".join(doc)+"\n")
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self,tag,attrs):
            self.links.extend(value for name,value in attrs if name in ("href","src"))
    parser = Links()
    parser.feed((output/"videos.html").read_text())
    links = parser.links+re.findall(r"\]\(([^)\n]+)\)",(output/"videos.md").read_text())
    return check_links(output,links)


def build(index_path,promotion_path,manifest,output,manifest_path=None):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a NEW immutable inventory output directory")
    inputs = {}
    index,promotion = load(index_path,inputs),load(promotion_path,inputs)
    if manifest_path:
        inputs[str(absolute(manifest_path).resolve())] = sha(absolute(manifest_path))
    for origin in index["profile_origins"].values():
        for field in ("registration","validation","audit"):
            load(origin[field+"_path"],inputs,origin[field+"_sha256"])
    rows,counts = inventory(index,manifest,promotion,output)
    output.mkdir(parents=True)
    atomic_json(output/"episodes.json",dict(scope=SCOPE,manifest_hash=manifest["manifest_hash"],counts=counts,episodes=rows))
    with (output/"episodes.csv").open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    row_links = [row[field] for row in rows for field in row if field.endswith("_url") and row[field]]
    checked = check_links(output,row_links)
    rendered_links = documents(output,rows)
    result = dict(scope=SCOPE,source_sha256=sha(__file__),input_metadata_sha256=inputs,counts=counts,
        episode_raw_file_hashes="Per-episode result/identity/complete/telemetry/video SHA256 are recorded in episodes.csv and episodes.json",
        unique_episode_urls_checked=checked,rendered_relative_urls_checked=rendered_links,
        artifacts_sha256={p.name:sha(p) for p in sorted(output.iterdir())},
        scientific_audit_performed=False,video_decode_audit_performed=False,
        xyz_scope="All indexed episodes link original full-precision actual eef_pos/sim_time_s telemetry; no padded, synthetic, normalized or reconstructed trajectories.")
    atomic_json(output/"inventory_index.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--dataset-index",required=True)
    parser.add_argument("--promotion")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    promotion = args.promotion or str(absolute(args.dataset_index).parent/"view/promotion.json")
    result = build(args.dataset_index,promotion,read_manifest(args.manifest),args.output,args.manifest)
    print(json.dumps(dict(output=args.output,counts=result["counts"],scope=SCOPE)))


if __name__=="__main__":
    main()
