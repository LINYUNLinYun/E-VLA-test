"""Per-condition quality sources, separate authenticated per-model E2 origins.

Planning is read-only. Creation writes only a new index/composite symlink view;
it never activates resume aliases or changes actual result directories/queues.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .common import ROOT,atomic_json,read_manifest
from .same_process_parity import sha


def absolute(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT/p


def quality_rows(episodes,condition,manifest):
    """Explicit canonical glob follows an episodes alias; no recursive guessing."""
    rows = {}
    for p in Path(episodes).glob("*/task*/init*/result.json"):
        if ".invalid." in str(p) or not (p.parent/"complete.json").exists():
            continue
        result = json.loads(p.read_text())
        marker = json.loads((p.parent/"complete.json").read_text())
        assert marker.get("valid") and result.get("valid"),f"Invalid completion: {p}"
        assert result["manifest_hash"]==marker["manifest_hash"]==manifest["manifest_hash"]
        assert result["condition"]==condition["id"]
        key = tuple(result[k] for k in ("suite","task","init"))
        assert key[0] in manifest["suites"] and key[1] in manifest["tasks"] and key[2] in manifest["confirm_inits"]
        assert (p.parents[2].name,p.parent.parent.name,p.parent.name)==(key[0],f"task{key[1]:02d}",f"init{key[2]:02d}")
        assert key not in rows,f"Duplicate canonical scene: {p}"
        rows[key] = dict(result,actual_result_path=str(p.resolve()),actual_result_sha256=sha(p))
    return rows


def profile_provenance(root,model,manifest):
    """Read authenticated positive registration, independently bind arm timing."""
    root = absolute(root).resolve()
    p = root/"shared_registration.json"
    registration = json.loads(p.read_text())
    assert registration["manifest_hash"]==manifest["manifest_hash"]
    provenance = registration["provenance"]
    assert sha(provenance["validation_path"])==provenance["validation_sha256"] and sha(provenance["audit_path"])==provenance["audit_sha256"]
    validation = json.loads(Path(provenance["validation_path"]).read_text())
    audit = json.loads(Path(provenance["audit_path"]).read_text())
    assert validation["manifest_hash"]==audit["manifest_hash"]==manifest["manifest_hash"]
    assert validation["condition"]==model+"_full" and audit["passed"]
    assert audit["validation_receipt_sha256"]==provenance["validation_sha256"]
    assert Path(audit["validation_receipt"]).resolve()==Path(provenance["validation_path"]).resolve()
    suffix = "" if model=="pi05" else "_generic"
    assert audit["audit_source_sha256"]==sha(Path(__file__).with_name(f"audit_shared_e2{suffix}.py"))
    assert validation["helper_source_sha256"]==sha(Path(__file__).with_name(f"shared_service_e2{suffix}.py"))
    assert provenance["registration_source_sha256"]==sha(Path(__file__).with_name(f"register_shared_e2{suffix}.py"))
    variants = manifest["e2"]["variants"]
    assert audit["counts"]=={v["id"]:8 for v in variants}
    evidence = audit["adoption_evidence"]
    assert {r["variant"] for r in evidence}=={v["id"] for v in variants if v["id"]!="full_io"}
    assert all(r["evidence_supports_consideration"] for r in evidence)
    expected_targets = set()
    for arm,variant in zip(validation["arms"],variants,strict=True):
        assert arm["variant"]==variant["id"] and np.isfinite(arm["wall_s"]) and arm["wall_s"]>0
        target = root/(model+"_full")/f"throughput_profile_{variant['id']}.json"
        expected_targets.add(target.resolve())
        timing = json.loads(target.read_text())
        assert timing["manifest_hash"]==manifest["manifest_hash"] and timing["condition"]==model+"_full"
        assert timing["provenance"]==provenance and timing["source_kind"]=="derived_from_authenticated_shared_service_validation"
        assert timing["wall_s"]==arm["wall_s"] and timing["workers"]==variant["workers"] and timing["variant"]==variant["id"]
        assert timing["newly_completed_episodes"]==8 and timing["completed_before"]==0 and timing["completed_after"]==8
    assert {absolute(p).resolve() for p in registration["derived_timing_targets"]}==expected_targets
    for alias in registration["benchmark_aliases"]:
        assert sha(absolute(alias["source"]))==sha(absolute(alias["target"]))==alias["sha256"]
    return dict(root=str(root),registration_path=str(p),registration_sha256=sha(p),validation_path=provenance["validation_path"],
        validation_sha256=provenance["validation_sha256"],audit_path=provenance["audit_path"],audit_sha256=provenance["audit_sha256"])


def plan(manifest,original,adopted_roots):
    original = absolute(original).resolve()
    models = {c["model"] for c in manifest["conditions"]}
    assert set(adopted_roots)<=models
    profiles = {m:profile_provenance(root,m,manifest) for m,root in adopted_roots.items()}
    rows,aliases = [],[]
    screen = {(s,t,i) for s in manifest["suites"] for t in manifest["tasks"] for i in manifest["screen_inits"]}
    for c in manifest["conditions"]+manifest["benchmark_references"]:
        old = original/c["id"]
        runtime = Path(profiles[c["model"]]["root"])/c["id"] if c["model"] in profiles else old
        old_rows = quality_rows(old/"episodes",c,manifest)
        new_rows = quality_rows(runtime/"episodes",c,manifest) if runtime.resolve()!=old.resolve() else old_rows
        different = (runtime/"episodes").resolve()!=(old/"episodes").resolve()
        if different and old_rows and new_rows:
            raise ValueError(f"Split/duplicate condition data: {c['id']}; preserve a single canonical episode source")
        if different and old_rows and not screen<=old_rows.keys():
            raise ValueError(f"Condition boundary not complete: {c['id']}; finish/resume the old condition before switching roots")
        chosen = old if old_rows else runtime
        if different and old_rows:
            if not runtime.exists() and not runtime.is_symlink():
                aliases.append(dict(target=str(runtime),source=str(old),kind="completed condition directory alias"))
            elif not (runtime/"episodes").exists() and not (runtime/"episodes").is_symlink():
                aliases.append(dict(target=str(runtime/"episodes"),source=str(old/"episodes"),kind="completed episodes alias; preserve E2 arms/benchmark in runtime condition directory"))
            elif (runtime/"episodes").resolve()!=(old/"episodes").resolve():
                raise ValueError(f"Existing incompatible resume path: {runtime/'episodes'}")
        benchmark = old/"benchmark.json" if (old/"benchmark.json").exists() else runtime/"benchmark.json"
        rows.append(dict(condition=c["id"],model=c["model"],actual_directory=str(chosen),episode_source=str(chosen/"episodes"),
            runtime_directory=str(runtime),parity_source=str(chosen/"condition_parity.json"),
            benchmark_source=str(benchmark),benchmark_sha256=sha(benchmark) if benchmark.exists() else None,
            existing_quality_keys=len(old_rows if old_rows else new_rows),source_selection="completed original" if old_rows else "pending/adopted runtime",
            profile_root=profiles.get(c["model"],{}).get("root")))
    return dict(schema="idea1-v3.1-dataset-bundle-v3",manifest_hash=manifest["manifest_hash"],conditions=rows,
        original_root=str(original),profile_origins=profiles,resume_alias_plan=aliases,resume_aliases_activated=False,
        source_sha256=sha(__file__),scope="One explicit episode source per condition; E2/profile and benchmark origins are independent. No queue change or actual-directory alias is activated by this index.")


def validate(index,manifest):
    assert index["manifest_hash"]==manifest["manifest_hash"] and index["schema"]=="idea1-v3.1-dataset-bundle-v3"
    expected = {c["id"]:c for c in manifest["conditions"]+manifest["benchmark_references"]}
    assert {r["condition"] for r in index["conditions"]}==expected.keys() and len(index["conditions"])==len(expected)
    for model,origin in index["profile_origins"].items():
        assert profile_provenance(origin["root"],model,manifest)==origin
    for row in index["conditions"]:
        model = expected[row["condition"]]["model"]
        assert row["model"]==model
        old = Path(index["original_root"])/row["condition"]
        runtime = Path(index["profile_origins"][model]["root"])/row["condition"] if model in index["profile_origins"] else old
        assert Path(row["runtime_directory"])==runtime
        assert Path(row["actual_directory"]) in (old,runtime)
        assert Path(row["episode_source"])==Path(row["actual_directory"])/"episodes"
        assert Path(row["parity_source"])==Path(row["actual_directory"])/"condition_parity.json"
        assert Path(row["benchmark_source"]) in (old/"benchmark.json",runtime/"benchmark.json")
        if row["benchmark_sha256"]:
            assert sha(row["benchmark_source"])==row["benchmark_sha256"]


def collect(index,manifest):
    validate(index,manifest)
    conditions = {c["id"]:c for c in manifest["conditions"]+manifest["benchmark_references"]}
    return {r["condition"]:quality_rows(r["episode_source"],conditions[r["condition"]],manifest) for r in index["conditions"]}


def summarize(index,manifest):
    """Same frozen summarize quality formulas, explicit source collection only."""
    data = collect(index,manifest)
    mappings = {r["condition"]:r for r in index["conditions"]}
    quality = []
    for c in manifest["conditions"]:
        rows = list(data[c["id"]].values())
        if not rows:
            continue
        baseline = data.get(c["model"]+"_full",{})
        paired = [r for r in rows if tuple(r[k] for k in ("suite","task","init")) in baseline]
        full_successes = sum(baseline[tuple(r[k] for k in ("suite","task","init"))]["success"] for r in paired)
        variant_successes = sum(r["success"] for r in paired)
        retained = variant_successes/full_successes if full_successes else None
        quality.append(dict(condition=c["id"],model=c["model"],skip=c["skip"],valid=len(rows),successes=sum(r["success"] for r in rows),
            success_rate=float(np.mean([r["success"] for r in rows])),per_suite={s:dict(successes=sum(r["success"] for r in rows if r["suite"]==s),valid=sum(r["suite"]==s for r in rows)) for s in manifest["suites"]},
            paired_valid=len(paired),paired_variant_successes=variant_successes,paired_full_successes=full_successes,
            relative_retention=retained,passes_70_percent_floor=retained is not None and retained>=.70,
            failure_furthest_phase=dict(Counter(r.get("furthest_phase","unknown") for r in rows if not r["success"])),
            phase_caveat="Heuristic progress evidence, not a causal failure diagnosis",data_use="Exploratory screen and candidate selection share observations",
            actual_directory=mappings[c["id"]]["actual_directory"],episode_source=mappings[c["id"]]["episode_source"]))
    from .analyze import summarize as frozen_summarize
    optimization = [r for model,origin in index["profile_origins"].items() for r in frozen_summarize(manifest,origin["root"])["evaluation_optimizations"] if r["condition"]==model+"_full"]
    return dict(manifest_hash=manifest["manifest_hash"],quality=quality,evaluation_optimizations=optimization,
        analysis_source_sha256=sha(__file__),dataset_creation_source_sha256=index["source_sha256"],
        actual_condition_directories={r["condition"]:r["actual_directory"] for r in index["conditions"]},profile_origins=index["profile_origins"],
        benchmark_origins={r["condition"]:dict(path=r["benchmark_source"],sha256=r["benchmark_sha256"]) for r in index["conditions"]},
        canonical_quality_counts={cid:len(rows) for cid,rows in data.items() if rows},partial=sum(r["valid"]>=40 for r in quality)<len(manifest["conditions"]),
        scope="Quality formulas identical to frozen summarize; collection uses explicit episodes.glob so completed-condition and episodes aliases are included exactly once.")


def create_view(index,bundle,manifest):
    """Only bundle-local aliases; never create the proposed runtime resume links."""
    validate(index,manifest)
    bundle = absolute(bundle)
    bundle.mkdir(parents=True,exist_ok=False)
    view = bundle/"view"
    view.mkdir()
    for row in index["conditions"]:
        p = view/row["condition"]
        p.mkdir()
        (p/"episodes").symlink_to(row["episode_source"],target_is_directory=True)
        (p/"condition_parity.json").symlink_to(row["parity_source"])
        (p/"benchmark.json").symlink_to(row["benchmark_source"])
        (p/"loaded_protocol.json").symlink_to(Path(row["runtime_directory"])/"loaded_protocol.json")
        for stage in ("screen","confirm"):
            name = f"throughput_{stage}_default.json"
            (p/name).symlink_to(Path(row["actual_directory"])/name)
        if row["condition"]==row["model"]+"_full" and row["profile_root"]:
            profile_condition = Path(row["profile_root"])/row["condition"]
            for v in manifest["e2"]["variants"]:
                (p/v["id"]).symlink_to(profile_condition/v["id"],target_is_directory=True)
                name = f"throughput_profile_{v['id']}.json"
                (p/name).symlink_to(profile_condition/name)
    for model in ("pi0","pi05"):
        for suffix in ("vision_parity","vision_same_process_parity"):
            name = f"{model}_{suffix}.json"
            (view/name).symlink_to(Path(index["original_root"])/name)
    atomic_json(bundle/"index.json",index)


def audit(index,bundle,manifest,check_videos=True):
    from .final_audit_v2 import audit as scientific_audit
    validate(index,manifest)
    bundle = absolute(bundle)
    view = bundle/"view"
    for row in index["conditions"]:
        for name,source in (("episodes",row["episode_source"]),("benchmark.json",row["benchmark_source"]),("condition_parity.json",row["parity_source"])):
            link = view/row["condition"]/name
            assert link.is_symlink() and link.resolve()==Path(source).resolve()
        if row["condition"]==row["model"]+"_full" and row["profile_root"]:
            for v in manifest["e2"]["variants"]:
                for name in (v["id"],f"throughput_profile_{v['id']}.json"):
                    assert (view/row["condition"]/name).resolve()==(Path(row["profile_root"])/row["condition"]/name).resolve()
    gate_paths = {model:str(Path(index["original_root"])/f"{model}_vision_same_process_parity.json") for model in ("pi0","pi05")}
    report = scientific_audit(manifest,view,gate_paths=gate_paths,check_videos=check_videos,analysis_report=str(bundle/"summary.json"))
    actual_counts = {cid:len(rows) for cid,rows in collect(index,manifest).items() if rows}
    if (bundle/"summary.json").exists():
        summary = json.loads((bundle/"summary.json").read_text())
        if summary.get("canonical_quality_counts")!=actual_counts:
            report["missing_requirements"].append("Dataset analysis counts are stale; refresh explicit-source bundle analyze")
    report.update(actual_condition_directories={r["condition"]:r["actual_directory"] for r in index["conditions"]},
        profile_origins=index["profile_origins"],benchmark_origins={r["condition"]:r["benchmark_source"] for r in index["conditions"]},
        dataset_source_sha256=sha(__file__),dataset_creation_source_sha256=index["source_sha256"],dataset_index_sha256=sha(bundle/"index.json"),
        excluded_technical_attempts=sorted({str(p.resolve()) for row in index["conditions"] for p in Path(row["actual_directory"]).rglob("technical_error.json")}))
    if not check_videos:
        report["missing_requirements"].append("Strict video decode skipped")
    report["passed"] = not report["errors"] and not report["missing_requirements"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("plan","analyze","create","audit"))
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--original-root")
    parser.add_argument("--adopted-root",action="append",default=[],help="Explicit parent-approved MODEL=ROOT, each requiring authenticated registration")
    parser.add_argument("--index")
    parser.add_argument("--bundle")
    parser.add_argument("--output")
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--skip-video-decode",action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest,verify_sources=True)
    if args.index:
        index = json.loads(Path(args.index).read_text())
    else:
        if not args.original_root:
            parser.error("Planning requires --original-root")
        pairs = [s.split("=",1) for s in args.adopted_root]
        if any(len(p)!=2 for p in pairs) or len({p[0] for p in pairs})!=len(pairs):
            parser.error("Unique MODEL=ROOT adopted-root values required")
        index = plan(manifest,args.original_root,dict(pairs))
    if args.operation=="analyze":
        result = summarize(index,manifest)
    elif args.operation=="audit":
        if not args.bundle:
            parser.error("audit requires --bundle")
        result = audit(index,args.bundle,manifest,not args.skip_video_decode)
    else:
        result = index
    if args.operation=="create" and args.write:
        if not args.bundle:
            parser.error("create --write requires a NEW --bundle path")
        create_view(index,args.bundle,manifest)
    if args.output:
        atomic_json(args.output,result)
    print(json.dumps(dict(operation=args.operation,dry_run=not args.write,mappings=len(index["conditions"]),
        proposed_resume_aliases=len(index["resume_alias_plan"]),resume_aliases_activated=False,
        quality_conditions=len(result.get("quality",[])),partial=result.get("partial"),passed=result.get("passed")),indent=2))
    if args.operation=="audit" and not result["passed"] and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
