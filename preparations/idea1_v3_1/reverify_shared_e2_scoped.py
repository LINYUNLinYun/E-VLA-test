"""Read-only authenticated E2 re-verification after quality keys are added.

The original verifier/source/receipt remain untouched. Its one broad discovery
glob is narrowed to the four manifest profile arms; every original numerical,
identity, control, episode, command, first-input and strict video check still runs.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from . import audit_shared_e2 as original
from .common import atomic_json,read_manifest
from .same_process_parity import sha


def scoped_path_type(root,variants):
    condition_root = (Path(root)/"pi05_full").resolve()
    variants = tuple(variants)
    assert variants==("full_io","lean_io","reuse_env","two_workers")
    class ProfileScopedPath(type(Path())):
        def glob(self,pattern):
            if str(pattern)=="*/*/task*/init*/complete.json" and self.resolve()==condition_root:
                for arm in variants:
                    # Original verifier still checks this discovered set against
                    # all32 exact expected keys, rejecting extras/missing arms.
                    yield from super().glob(f"{arm}/*/task*/init*/complete.json")
            else:
                yield from super().glob(pattern)
    return ProfileScopedPath


def verify(root,manifest):
    root = Path(root)
    recorded_path = root/"audit.json"
    recorded = json.loads(recorded_path.read_text())
    assert recorded["passed"] and recorded["video_verified"]
    assert recorded["audit_source_sha256"]==sha(original.__file__),"Original authenticated verifier source changed"
    assert recorded["validation_receipt_sha256"]==sha(root/"validation.json")
    variants = [v["id"] for v in manifest["e2"]["variants"]]
    path_class = scoped_path_type(root,variants)
    # Only this module's Path constructor is scoped; pathlib globally and all
    # raw data are unchanged. No validation result/boolean is patched.
    with patch.object(original,"Path",path_class):
        fresh = original.verify(root,manifest,check_videos=True)
    assert fresh==recorded,"Current four-arm raw evidence differs from original authenticated audit"
    markers = [p for p in path_class(root/"pi05_full").glob("*/*/task*/init*/complete.json") if ".invalid." not in str(p)]
    assert len(markers)==32
    quality = list((root/"pi05_full/episodes").glob("*/task*/init*/complete.json"))
    return dict(passed=True,manifest_hash=manifest["manifest_hash"],scope_adapter_source_sha256=sha(__file__),
        original_audit_path=str(recorded_path.resolve()),original_audit_sha256=sha(recorded_path),
        original_audit_source_sha256=recorded["audit_source_sha256"],validation_sha256=sha(root/"validation.json"),
        profile_episodes=32,profile_marker_hashes={str(p.resolve()):sha(p) for p in sorted(markers)},
        observed_unrelated_quality_markers=len(quality),
        scope="Only the original broad E2 marker-discovery glob is restricted to the four frozen profile arms; all original strict validation checks rerun unchanged. Quality keys are a separate stage, not extra E2 episodes.",
        original_verification=fresh,
        dependency_review=dict(immutable_proof_files=[str((root/p).resolve()) for p in ("validation.json","audit.json","shared_registration.json")],
            self_contained_profile_identity="Every profile identity.json retains its original service PID/protocol, and first-input receipts retain the original PID/hash; current quality metadata is not substituted.",
            shared_service_protocol_file=str((root/"loaded_protocol.json").resolve()),
            quality_protocol_file=str((root/"pi05_full/loaded_protocol.json").resolve()),
            quality_overwrites_proof_files=False,
            timing_names="Quality writes throughput_screen_default/throughput_confirm_default; derived E2 receipts are throughput_profile_<arm>.json.",
            benchmark_metadata_caveat="Quality may overwrite pi05_full/loaded_protocol.json and condition_parity.json. They are not E2 proof dependencies; historical benchmark PID proof must continue to use preserved receipts or explicitly limited source-guard inference."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",help="New receipt only; never overwrite the original audit")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise FileExistsError("Scoped re-verification receipts are append-only; choose a new output")
    result = verify(args.root,read_manifest(args.manifest,verify_sources=True))
    if args.output:
        atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("original_verification","profile_marker_hashes")},indent=2))


if __name__ == "__main__":
    main()
