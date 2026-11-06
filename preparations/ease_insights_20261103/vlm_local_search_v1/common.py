"""Immutable local-search identities; legacy experiment modules remain unchanged."""
import hashlib
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT, SUITES, atomic_json, digest
from preparations.idea1_v3_1.common import read_manifest as _read_manifest

MODULE = "preparations.ease_insights_20261103.vlm_local_search_v1"
MODEL_PYTHON = ROOT / "policies/realtime-vla-flash/.venv/bin/python"
ENV_PYTHON = ROOT / "policies/libero-eval-v21/.venv/bin/python"
DISCOVERY_PHASE = "local_search_discovery_v1"
CONFIRMATION_PHASE = "local_search_confirmation_v1"
MASKS = {
    "jac_anchor": (2, 3, 5, 14, 16, 17),
    "local_spatial": (2, 3, 5, 13, 16, 17),
    "local_object": (2, 3, 5, 14, 15, 17),
    "local_goal": (2, 3, 11, 14, 16, 17),
    "local_long": (2, 3, 5, 12, 16, 17),
}
CONDITIONS = ("full_d10", *MASKS)
RUNTIME_FILES = ("__init__.py", "common.py", "policy.py", "server.py", "gate.py", "benchmark.py", "runner.py", "worker.py", "prepare.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions():
    return [dict(id=cid, model="pi0", mode="full" if cid == "full_d10" else "periodic",
                 compensation=None if cid == "full_d10" else "extrapolate", period=2,
                 H=50, R=5, D=10 if cid == "full_d10" else 2, refresh_D=10,
                 skip=list(MASKS.get(cid, ()))) for cid in CONDITIONS]


def validate_condition(config):
    expected = next((c for c in conditions() if c["id"] == config.get("id")), None)
    if expected is None or config != expected:
        raise ValueError("Condition differs from the bounded local-search whitelist")
    return tuple(config["skip"])


def stage_for(manifest):
    phase = manifest.get("phase")
    if phase == DISCOVERY_PHASE:
        if manifest.get("discovery_inits") != [36, 37, 38]:
            raise ValueError("Discovery requires exactly init36-38")
        return "discovery"
    if phase == CONFIRMATION_PHASE:
        if manifest.get("confirmation_inits") != list(range(39, 44)):
            raise ValueError("Confirmation requires exactly init39-43")
        return "confirm"
    raise ValueError("Not a local-search stage manifest")


def validate_manifest(manifest):
    stage = stage_for(manifest)
    if manifest.get("conditions") != conditions() or manifest.get("suites") != list(SUITES) or manifest.get("tasks") != list(range(10)):
        raise ValueError("Local-search condition, suite, or task universe changed")
    planned = manifest.get("planned_conditions")
    if stage == "discovery":
        if planned != list(CONDITIONS) or manifest.get("cases_per_condition") != 120 or manifest.get("total_discovery_episodes") != 720:
            raise ValueError("Discovery must cover all six conditions and 720 cases")
    elif (not isinstance(planned, list) or len(planned) != 3 or planned[0] != "full_d10" or
          len(set(planned)) != 3 or any(cid not in MASKS for cid in planned[1:]) or
          manifest.get("cases_per_condition") != 200 or manifest.get("total_confirmation_episodes") != 600):
        raise ValueError("Confirmation requires Full and exactly two frozen candidates, 600 cases")
    return manifest


def cases(manifest, stage=None, shard_index=0, shards=1):
    actual = stage_for(manifest)
    if stage not in (None, actual):
        raise ValueError("Requested stage differs from immutable manifest")
    if not isinstance(shards, int) or not isinstance(shard_index, int) or not 0 <= shard_index < shards:
        raise ValueError("Invalid task shard")
    if manifest.get("suites") != list(SUITES) or manifest.get("tasks") != list(range(10)):
        raise ValueError("Case universe changed")
    inits = manifest["discovery_inits" if actual == "discovery" else "confirmation_inits"]
    pairs = [(suite, task) for suite in manifest["suites"] for task in manifest["tasks"]]
    return [dict(suite=suite, task=task, init=init) for index, (suite, task) in enumerate(pairs)
            if index % shards == shard_index for init in inits]


def planned_config(manifest, condition):
    validate_manifest(manifest)
    if condition not in manifest["planned_conditions"]:
        raise ValueError("Condition is not planned for this frozen stage")
    cfg = next(c for c in manifest["conditions"] if c["id"] == condition)
    validate_condition(cfg)
    return cfg


def validate_output(output, manifest):
    validate_manifest(manifest)
    if Path(output).resolve() != Path(manifest["output_root"]).resolve():
        raise ValueError("Output differs from its frozen stage destination")
    for name in ("protocol_evidence", "design_evidence"):
        evidence = manifest[name]
        if sha(evidence["path"]) != evidence["sha256"]:
            raise ValueError("Frozen " + name + " changed")
    if stage_for(manifest) == "confirm":
        evidence = manifest["selection_evidence"]
        if sha(evidence["path"]) != evidence["sha256"]:
            raise ValueError("Frozen selection file changed")
        selection = json.loads(Path(evidence["path"]).read_text())
        supplied = selection.pop("selection_hash", None)
        if supplied != evidence["selection_hash"] or digest(selection) != supplied:
            raise ValueError("Frozen selection digest differs")
        selected = selection["selection"]["selected"]
        if (selection["selection"]["status"] != "selected" or
                manifest["planned_conditions"] != ["full_d10", *selected["paths"]] or
                manifest["preregistered_hypotheses"] != selected["directions"]):
            raise ValueError("Confirmation differs from its frozen pair and directions")


def read_manifest(path, verify_sources=False):
    manifest = _read_manifest(path, verify_sources=verify_sources)
    validate_manifest(manifest)
    validate_output(Path(path).resolve().parent, manifest)
    return manifest


def runtime_revision(manifest):
    value = dict(name="vlm-local-search-isolated-runtime-v1", scientific_manifest_hash=manifest["manifest_hash"],
        phase=manifest["phase"], protocol_evidence=manifest["protocol_evidence"], design_evidence=manifest["design_evidence"],
        wrapper_source_files=manifest["local_runtime_sources"],
        backend_reused="preparations.ease_insights_20261103.vlm.backend.Backend",
        mechanics_reused="preparations.ease_insights_20261103.vlm.policy.Policy.predict/reset/acknowledge",
        legacy_sources_changed=False, new_mask_whitelist=True)
    value["revision_hash"] = digest(value)
    return value


def verify_analysis_sources(manifest):
    sources = manifest.get("analysis_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Frozen analysis source identities are missing")
    for filename, expected in sources.items():
        if sha(ROOT/filename) != expected:
            raise ValueError("Analysis source changed without a technical revision: " + filename)
    return sources
