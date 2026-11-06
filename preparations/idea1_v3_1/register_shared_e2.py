"""Register truthful derived timing receipts after independent32-episode audit."""
import argparse
import json
from pathlib import Path
import time

from .common import atomic_json,read_manifest
from .same_process_parity import sha
from .audit_shared_e2 import verify


def prepare(root,manifest,benchmark_root=None):
    root = Path(root)
    audit = verify(root,manifest,check_videos=True)
    recorded_path = root/"audit.json"
    recorded = json.loads(recorded_path.read_text())
    assert recorded==audit and audit["passed"],"Recorded independent audit differs from current authenticated evidence"
    assert all(r["evidence_supports_consideration"] for r in audit["adoption_evidence"])
    receipt = json.loads((root/"validation.json").read_text())
    provenance = dict(source_kind="derived_from_authenticated_shared_service_validation",
        validation_path=str((root/"validation.json").resolve()),validation_sha256=sha(root/"validation.json"),
        audit_path=str(recorded_path.resolve()),audit_sha256=sha(recorded_path),registration_source_sha256=sha(__file__))
    targets = []
    for arm,variant in zip(receipt["arms"],manifest["e2"]["variants"],strict=True):
        assert arm["variant"]==variant["id"]
        row = dict(manifest_hash=manifest["manifest_hash"],condition="pi05_full",wall_s=arm["wall_s"],
            newly_completed_episodes=8,completed_before=0,completed_after=8,workers=variant["workers"],variant=variant["id"],
            scope="Derived eight-episode arm wall from one shared-service validation; includes simulation/RPC/common observational first-input capture/arm I/O, excludes startup/warm/control requests",
            provenance=provenance,source_kind=provenance["source_kind"],validation_sha256=provenance["validation_sha256"],audit_sha256=provenance["audit_sha256"],
            controller_process_receipt=False)
        targets.append((root/"pi05_full"/f"throughput_profile_{variant['id']}.json",row))
    aliases = []
    if benchmark_root:
        for cid in ("pi05_full","pi05_full_vision_scan_reference"):
            source = Path(benchmark_root)/cid/"benchmark.json"
            b = json.loads(source.read_text())
            assert b["manifest_hash"]==manifest["manifest_hash"] and b["condition"]["id"]==cid and b["warmup"]==20 and b["samples"]==100 and len(b["rows"])==100
            aliases.append(dict(source=str(source.resolve()),target=str(root/cid/"benchmark.json"),sha256=sha(source),
                kind="immutable derived exact artifact copy; original benchmark scientific key, not a new measurement"))
    registration = dict(manifest_hash=manifest["manifest_hash"],provenance=provenance,derived_timing_targets=[str(p) for p,_ in targets],
        benchmark_aliases=aliases,registered_host_s=time.time(),scope="Registration only; unchanged frozen reuse_gate recomputes actual profile evidence. Parent adoption is pi05 quality only.")
    planned = [p for p,_ in targets]+[Path(a["target"]) for a in aliases]+[root/"shared_registration.json"]
    if any(p.exists() or p.is_symlink() for p in planned):
        raise FileExistsError("Registration refuses to overwrite any existing derived receipt or alias")
    return targets,registration


def write(targets,registration,root):
    for target,row in targets:
        if target.exists():
            raise FileExistsError(target)
        atomic_json(target,row)
    for alias in registration["benchmark_aliases"]:
        source,target = Path(alias["source"]),Path(alias["target"])
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as output:
            output.write(source.read_bytes())
        assert sha(target)==alias["sha256"]
    atomic_json(Path(root)/"shared_registration.json",registration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--benchmark-root")
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    targets,registration = prepare(args.root,read_manifest(args.manifest,verify_sources=True),args.benchmark_root)
    if args.write:
        write(targets,registration,args.root)
    print(json.dumps(dict(dry_run=not args.write,derived_receipts=[str(p) for p,_ in targets],aliases=registration["benchmark_aliases"]),indent=2))


if __name__ == "__main__":
    main()
