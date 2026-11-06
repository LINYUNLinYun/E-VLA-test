"""Independent confirmation protocol; never unlocks discovery execution."""
from pathlib import Path
from .common import digest, sha

PHASE = "independent_confirmation"
WRAPPERS = ("prepare_confirmation.py", "confirm_common.py", "confirm_runner.py", "confirm_worker.py")


def cases(manifest, stage, shard_index=0, shards=1):
    if manifest.get("phase") != PHASE or stage != "confirm":
        raise ValueError("Confirmation wrapper only accepts an independent confirmation manifest and confirm stage")
    if manifest.get("confirmation_inits") != list(range(31,36)):
        raise ValueError("Independent confirmation must use exactly init31-35")
    if not 0 <= shard_index < shards:
        raise ValueError("Invalid task shard")
    pairs = [(s,t) for s in manifest["suites"] for t in manifest["tasks"]]
    return [dict(suite=s,task=t,init=i) for index,(s,t) in enumerate(pairs) if index % shards == shard_index
            for i in manifest["confirmation_inits"]]


def planned_config(manifest, condition):
    if manifest.get("phase") != PHASE or condition not in manifest["planned_conditions"]:
        raise ValueError("Condition is not in the frozen independent confirmation plan")
    return next(c for c in manifest["conditions"] if c["id"] == condition)


def validate_output(output, manifest):
    output = Path(output).resolve()
    if manifest.get("phase") != PHASE or output != Path(manifest["confirmation_output"]).resolve():
        raise ValueError("Confirmation output differs from its frozen independent destination")
    if output == Path(manifest["discovery_reference"]["root"]).resolve():
        raise ValueError("Confirmation cannot execute in the discovery directory")
    evidence = manifest["selection_evidence"]
    if sha(evidence["path"]) != evidence["sha256"]:
        raise ValueError("Frozen selection file changed after confirmation preparation")


def runtime_revision(manifest):
    value = dict(name="vlm-independent-confirmation-v1", scientific_manifest_hash=manifest["manifest_hash"],
        discovery_manifest_hash=manifest["discovery_reference"]["manifest_hash"],
        selection_hash=manifest["selection_evidence"]["selection_hash"],
        wrapper_source_files=manifest["confirmation_wrapper_sources"],
        model_backend_policy_changed=False, discovery_runtime_changed=False)
    value["revision_hash"] = digest(value)
    return value
