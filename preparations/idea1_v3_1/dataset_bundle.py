"""Explicit split-root dataset view; actual condition directories stay in place.

Recursive-symlink traversal is not assumed: analysis visits the actual roots;
audit uses explicit canonical globs and separately indexes technical attempts.
"""
import argparse
import json
from pathlib import Path

from .common import ROOT,atomic_json,read_manifest
from .same_process_parity import sha


GATES = tuple(f"{m}_{suffix}.json" for m in ("pi0","pi05") for suffix in ("vision_parity","vision_same_process_parity"))


def create(bundle,manifest,original,pi05,write=False):
    bundle,original,pi05 = (Path(p).resolve() for p in (bundle,original,pi05))
    if bundle.exists():
        raise FileExistsError("Bundle indices/views are immutable; use a new bundle path")
    registration = pi05/"shared_registration.json"
    assert registration.exists(),"Register authenticated pi05 shared-service evidence first"
    rows = []
    for c in manifest["conditions"]+manifest["benchmark_references"]:
        source_root = pi05 if c["model"]=="pi05" else original
        source = source_root/c["id"]
        rows.append(dict(condition=c["id"],model=c["model"],source_root=str(source_root),actual_directory=str(source),
            exists_at_creation=source.is_dir(),view_link=str(bundle/"view"/c["id"])))
    index = dict(manifest_hash=manifest["manifest_hash"],bundle_source_sha256=sha(__file__),conditions=rows,
        source_roots=dict(original=str(original),pi05=str(pi05)),
        registration=dict(path=str(registration),sha256=sha(registration)),
        gates=[dict(source=str(original/name),view_link=str(bundle/"view"/name),sha256=sha(original/name)) for name in GATES],
        promotion_path=str(bundle/"view/promotion.json"),
        scope="One primary actual directory per scientific condition: pi05 quality/profile evidence from shared-service root, pi0/XR/Cog from original root. Directory mappings are immutable, results may continue to arrive; pending links do not imply completion.",
        original_pi05_observations="Original pi05 profiles and benchmarks remain at original root. Exact registered benchmark copies are aliases, not independent measurements. No episodes are copied or moved.")
    if write:
        bundle.mkdir(parents=True,exist_ok=False)
        (bundle/"view").mkdir()
        for row in rows:
            Path(row["view_link"]).symlink_to(row["actual_directory"],target_is_directory=True)
        for row in index["gates"]:
            Path(row["view_link"]).symlink_to(row["source"])
        atomic_json(bundle/"index.json",index)
    return index


def load(bundle,manifest):
    bundle = Path(bundle).resolve()
    index = json.loads((bundle/"index.json").read_text())
    assert index["manifest_hash"]==manifest["manifest_hash"]
    expected = {c["id"]:c for c in manifest["conditions"]+manifest["benchmark_references"]}
    assert {r["condition"] for r in index["conditions"]}==expected.keys() and len(index["conditions"])==len(expected)
    for row in index["conditions"]:
        c = expected[row["condition"]]
        selected = index["source_roots"]["pi05" if c["model"]=="pi05" else "original"]
        assert row["actual_directory"]==str(Path(selected)/c["id"]) and row["model"]==c["model"]
        assert Path(row["view_link"]).is_symlink() and Path(row["view_link"]).resolve()==Path(row["actual_directory"])
    for row in index["gates"]:
        assert Path(row["view_link"]).is_symlink() and Path(row["view_link"]).resolve()==Path(row["source"])
        assert sha(row["source"])==row["sha256"]
    registered = index["registration"]
    assert sha(registered["path"])==registered["sha256"]
    registration = json.loads(Path(registered["path"]).read_text())
    assert registration["manifest_hash"]==manifest["manifest_hash"]
    provenance = registration["provenance"]
    assert sha(provenance["validation_path"])==provenance["validation_sha256"] and sha(provenance["audit_path"])==provenance["audit_sha256"]
    for alias in registration["benchmark_aliases"]:
        assert sha(alias["source"])==sha(alias["target"])==alias["sha256"]
    for filename in registration["derived_timing_targets"]:
        path = Path(filename) if Path(filename).is_absolute() else ROOT/filename
        row = json.loads(path.read_text())
        assert row["source_kind"]=="derived_from_authenticated_shared_service_validation" and row["provenance"]==provenance
        assert row["manifest_hash"]==manifest["manifest_hash"] and row["condition"]=="pi05_full"
    return index


def analyze(bundle,manifest):
    from .analyze import summarize
    index = load(bundle,manifest)
    reports = {name:summarize(manifest,path) for name,path in index["source_roots"].items()}
    ownership = {r["condition"]:("pi05" if r["model"]=="pi05" else "original") for r in index["conditions"]}
    actual = {r["condition"]:r["actual_directory"] for r in index["conditions"]}
    report = dict(manifest_hash=manifest["manifest_hash"],dataset_index=str((Path(bundle)/"index.json").resolve()),
        dataset_index_sha256=sha(Path(bundle)/"index.json"),actual_condition_directories=actual,
        completion_is_not_implied_by_code=True,analysis_source_sha256=sha(__file__))
    for section in ("quality","throughput","performance","evaluation_optimizations"):
        report[section] = [dict(row,actual_condition_directory=actual[row["condition"]]) for name,data in reports.items()
                           for row in data[section] if ownership.get(row["condition"])==name]
    report["frontier"] = reports["original"]["frontier"]
    report["technical_errors"] = [p for name,data in reports.items() for p in data["technical_errors"]
        if ownership.get(Path(p).relative_to(index["source_roots"][name]).parts[0])==name]
    report["retained_original_pi05_profile_observations"] = [r for r in reports["original"]["evaluation_optimizations"] if r["condition"]=="pi05_full"]
    report["partial"] = sum(r["valid"]>=40 for r in report["quality"])<len(manifest["conditions"])
    report["canonical_quality_counts"] = {r["condition"]:r["valid"] for r in report["quality"]}
    return report


def audit(bundle,manifest,check_videos=True):
    from .final_audit_v2 import audit as actual_audit
    index = load(bundle,manifest)
    summary = Path(bundle)/"summary.json"
    report = actual_audit(manifest,Path(bundle)/"view",check_videos=check_videos,analysis_report=str(summary))
    report["dataset_index"] = str((Path(bundle)/"index.json").resolve())
    report["dataset_index_sha256"] = sha(Path(bundle)/"index.json")
    report["actual_condition_directories"] = {r["condition"]:r["actual_directory"] for r in index["conditions"]}
    report["excluded_technical_attempts"] = sorted({str(p.resolve()) for row in index["conditions"]
        for p in Path(row["actual_directory"]).rglob("technical_error.json")})
    report["dataset_audit_source_sha256"] = sha(__file__)
    if not check_videos:
        report["missing_requirements"].append("Strict video decode skipped")
        report["passed"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("create","analyze","audit"))
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--bundle",required=True)
    parser.add_argument("--original-root")
    parser.add_argument("--pi05-root")
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--skip-video-decode",action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest,verify_sources=True)
    if args.operation=="create":
        if not args.original_root or not args.pi05_root:
            parser.error("create requires both actual roots")
        result = create(args.bundle,manifest,args.original_root,args.pi05_root,args.write)
    elif args.operation=="analyze":
        result = analyze(args.bundle,manifest)
    else:
        result = audit(args.bundle,manifest,not args.skip_video_decode)
    if args.output:
        atomic_json(args.output,result)
    print(json.dumps(dict(operation=args.operation,manifest_hash=manifest["manifest_hash"],passed=result.get("passed"),
        partial=result.get("partial"),quality_conditions=len(result.get("quality",[])),
        errors=result.get("errors",[]),missing_requirements=len(result.get("missing_requirements",[])),
        actual_directory_mappings=len(result.get("conditions",[]))),indent=2))
    if args.operation=="audit" and not result["passed"] and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
