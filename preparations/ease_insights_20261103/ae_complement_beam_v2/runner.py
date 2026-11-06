"""Broad-mask command plan; GPU lifecycle belongs to resource poolV4."""
import argparse
import json
from pathlib import Path
import time

from preparations.idea1_v3_1.common import environment, lock
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.ease_insights_20261103.vlm.recovery import recover_case
from preparations.ease_insights_20261103.vlm_local_search_v1.runner import resolve_binding
from .common import ROOT, MODULE, MODEL_PYTHON, ENV_PYTHON, CONDITIONS, atomic_json, sha, read_manifest
from .common import cases, planned_config, runtime_revision, validate_output, stage_for, LIMITS
from .server import finite_bank_scope


def pinned_reference(manifest):
    """Resolve only the explicitly pinned optional historical numerical fixture."""
    reference = manifest.get("preflight", {}).get("cross_service_reference")
    if reference is None:
        return None
    required = {"path", "sha256", "manifest_hash", "reference_manifest", "source_protocol"}
    if not isinstance(reference, dict) or set(reference) != required:
        raise ValueError("Cross-service reference needs exact artifact, manifest and protocol pins")
    for item in (reference, reference["reference_manifest"], reference["source_protocol"]):
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Historical numerical reference artifact changed")
    original = json.loads(Path(reference["reference_manifest"]["path"]).read_text())
    protocol = json.loads(Path(reference["source_protocol"]["path"]).read_text())
    if (original.get("manifest_hash") != reference["manifest_hash"] or protocol.get("manifest_hash") != reference["manifest_hash"] or
        protocol.get("real_input_parity_passed") is not True or protocol.get("qk_reference_passed") is not True or
        protocol.get("checkpoint_sha256") != manifest["checkpoints"]["pi0"]["sha256"]):
        raise ValueError("Historical reference manifest/protocol binding differs")
    return reference


def cross_service(output, logs, manifest_hash):
    """Pair old finite-bank outputs while recording both scientific identities."""
    import numpy as np
    output, logs = Path(output), Path(logs)
    manifest = read_manifest(output/"manifest.json")
    if manifest["manifest_hash"] != manifest_hash:
        raise ValueError("Cross-service scientific manifest differs")
    historical = pinned_reference(manifest)
    with lock(output/"cross_service.lock"):
        reference = output/"cross_service_reference.npz"
        identity = output/"cross_service_reference.json"
        if reference.exists() != identity.exists():
            raise ValueError("Incomplete cross-service reference requires technical recovery")
        if not reference.exists():
            source = Path(historical["path"]) if historical else logs/"gate_actions.npz"
            reference.write_bytes(source.read_bytes())
            atomic_json(identity, dict(scientific_manifest_hash=manifest_hash,
                source_manifest_hash=historical["manifest_hash"] if historical else manifest_hash,
                source_npz=dict(path=str(source), sha256=sha(source)),
                reference_npz_sha256=sha(reference),
                historical_reference=historical, first_service=str(logs),
                relation="finite old-bank numerical fixture; new task universe and manifest are independent"))
        bound = json.loads(identity.read_text())
        if (bound["scientific_manifest_hash"] != manifest_hash or bound.get("historical_reference") != historical or
            bound["reference_npz_sha256"] != sha(reference)):
            raise ValueError("Cross-service reference belongs to another identity or changed")
        checks = []
        with np.load(reference, allow_pickle=False) as ref, np.load(logs/"gate_actions.npz", allow_pickle=False) as actual:
            if set(ref.files) != set(actual.files):
                raise ValueError("Cross-service finite-bank gate keys differ")
            for key in ref.files:
                a, b = ref[key], actual[key]
                same_shape = a.shape == b.shape
                passed = same_shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=1e-5,rtol=1e-5)
                checks.append(dict(key=key, reference_shape=list(a.shape), actual_shape=list(b.shape),
                    max_abs=float(np.abs(a-b).max()) if same_shape else None, passed=bool(passed)))
        receipt = dict(passed=all(c["passed"] for c in checks), checks=checks, atol=1e-5, rtol=1e-5,
            scientific_manifest_hash=manifest_hash, reference_identity=bound,
            gate_scope=finite_bank_scope(manifest), new_scientific_protocol_unchanged_from_reference=False)
        atomic_json(logs/"cross_service.json", receipt)
        if not receipt["passed"]:
            raise RuntimeError("Cross-service finite real-input bank gate failed")
        return receipt


def run(args):
    if args.launch:
        raise RuntimeError("Direct --launch is disabled: use the external retained resource poolV4")
    if not isinstance(args.gpu_uuid, str) or not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("An explicitly assigned GPU UUID is required")
    if type(args.port) is not int or not 0 < args.port < 65536:
        raise ValueError("Invalid assigned local RPC port")
    output = Path(args.output).resolve()
    manifest = read_manifest(output/"manifest.json")
    validate_output(output, manifest)
    planned_config(manifest, args.condition)
    stage = stage_for(manifest)
    if getattr(args, "stage", None) not in (None, stage):
        raise ValueError("CLI stage differs from immutable manifest")
    args.stage = stage
    scenes = cases(manifest, stage, args.shard_index, args.shards)
    logs = output/args.condition/"runs"/f"{stage}_shard{args.shard_index:02d}of{args.shards:02d}_{time.time_ns()}"
    marker = output/args.condition/f"{stage}_shard{args.shard_index:02d}of{args.shards:02d}_complete.json"
    server = [str(MODEL_PYTHON), "-m", MODULE+".server", "--manifest", str(output/"manifest.json"),
        "--condition", args.condition, "--port", str(args.port), "--gpu-uuid", args.gpu_uuid, "--output", str(logs)]
    worker = [str(ENV_PYTHON), "-m", MODULE+".worker", "--output", str(output), "--condition", args.condition,
        "--port", str(args.port), "--stage", stage, "--shard-index", str(args.shard_index), "--shards", str(args.shards)]
    identity = dict(manifest_hash=manifest["manifest_hash"], condition=args.condition, stage=stage,
        stateful_workers_per_server=1, technical_runtime_revision=runtime_revision(manifest),
        command_builder=MODULE+".runner.run(launch=False)", scientific_protocol_is_new=True,
        gate_scope=finite_bank_scope(manifest), policy_action_limits=dict(LIMITS), settling_actions=10,
        resource_lifecycle="external borrower monitors retained original holders and owns only its new Popen children")
    plan = dict(dry_run=True, episodes=len(scenes), preflight_only=bool(getattr(args,"preflight_only",False)),
        server=server, worker=worker, logs=str(logs), complete_file=str(marker), identity=identity,
        manifest_hash=manifest["manifest_hash"], stage=stage, condition=args.condition)
    # The aliases preserve gpu_resource_session.Science.execute's exact shape;
    # task and holder ownership are supplied by the independent borrower.
    return dict(plan, args=args, output=output, manifest=manifest, scenes=scenes, marker=marker,
        logs=logs, server_argv=server, worker_argv=worker, plan=plan)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--stage", choices=("calibration", "discovery", "confirm"), default=None)
    parser.add_argument("--gpu-uuid", required=True); parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--launch", action="store_true")
    print(json.dumps(run(parser.parse_args())["plan"], indent=2))


if __name__ == "__main__":
    main()
