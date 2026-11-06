"""Coverage-only exact aliases of missing CANDIDATE benchmarks after CPU audit.

Fresh native/matched references are never copied. Primary ratios remain in the
new cohort; frozen root-level comparisons are legacy and may cross card/epoch.
"""
import argparse
import json
from pathlib import Path

from .common import read_manifest
from .performance_cohort import audit
from .same_process_parity import sha


def prepare(cohort,runtime,manifest):
    cohort,runtime = Path(cohort).resolve(),Path(runtime).resolve()
    fresh = audit(cohort,manifest)
    audit_path = cohort/"audit.json"
    recorded = json.loads(audit_path.read_text())
    assert recorded==fresh and recorded["passed"],"Saved cohort strict audit does not match current source/evidence"
    copies,skipped = [],[]
    for phase in fresh["phases"]:
        if phase["role"]!="candidate":
            continue
        cid = phase["condition"]
        target = runtime/cid/"benchmark.json"
        sidecar = target.with_name("benchmark.performance_cohort_origin.json")
        if target.exists() or target.is_symlink():
            skipped.append(dict(condition=cid,path=str(target),reason="Existing benchmark preserved without overwrite"))
            continue
        if sidecar.exists() or sidecar.is_symlink():
            raise FileExistsError(f"Existing origin receipt without target benchmark: {sidecar}")
        source = Path(phase["benchmark_path"])
        directory = source.parent
        origin = dict(source_kind="coverage_only_exact_candidate_benchmark_alias",condition=cid,
            source_benchmark=str(source),source_sha256=sha(source),cohort_receipt=str(cohort/"cohort.json"),
            cohort_receipt_sha256=sha(cohort/"cohort.json"),cohort_audit=str(audit_path),cohort_audit_sha256=sha(audit_path),
            registrar_source_sha256=sha(__file__),original_benchmark_parity_snapshot=phase["original_benchmark_parity_snapshot"],
            affirmative_same_pid_proof=phase["affirmative_same_pid_proof"],
            origin_witnesses={str((directory/name).resolve()):sha(directory/name) for name in ("loaded_protocol.json","condition_parity.json")},
            fresh_full_references_copied=False,cohort_stable_for_primary_comparison=fresh["stable_for_primary_comparison"],
            warning="COVERAGE ONLY. Legacy frozen-analyze root-level comparisons may cross cards/epochs; main performance evidence must use this new cohort's native-BEFORE/matched P50 ratios and drift flags.")
        copies.append(dict(source=str(source),target=str(target),sidecar=str(sidecar),origin=origin))
    return dict(copies=copies,skipped=skipped,manifest_hash=manifest["manifest_hash"],primary_cohort=str(cohort),
        scope="Only previously missing candidate benchmark.json files; no Full/matched references, raw counters, protocols or old gates are changed")


def write(registration):
    for row in registration["copies"]:
        target,sidecar = Path(row["target"]),Path(row["sidecar"])
        if target.exists() or target.is_symlink() or sidecar.exists() or sidecar.is_symlink():
            raise FileExistsError("A planned alias target already exists; never overwrite")
    for row in registration["copies"]:
        source,target,sidecar = Path(row["source"]),Path(row["target"]),Path(row["sidecar"])
        assert sha(source)==row["origin"]["source_sha256"]
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        assert sha(target)==row["origin"]["source_sha256"]
        with sidecar.open("x") as stream:
            json.dump(row["origin"],stream,indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--cohort",required=True)
    parser.add_argument("--runtime-root",required=True)
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    registration = prepare(args.cohort,args.runtime_root,read_manifest(args.manifest,verify_sources=True))
    if args.write:
        write(registration)
    print(json.dumps(dict(dry_run=not args.write,planned_candidate_copies=[r["target"] for r in registration["copies"]],
        skipped=registration["skipped"],scope=registration["scope"]),indent=2))


if __name__ == "__main__":
    main()
