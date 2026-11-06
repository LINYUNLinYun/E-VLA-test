"""Independent LIBERO90 metadata-category branch; frozen legacy files stay intact."""
import hashlib
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT, atomic_json, digest
from preparations.idea1_v3_1.common import read_manifest as _read_manifest

MODULE = "preparations.ease_insights_20261103.vlm_within_suite_v1"
MODEL_PYTHON = ROOT / "policies/realtime-vla-flash/.venv/bin/python"
ENV_PYTHON = ROOT / "policies/libero-eval-v21/.venv/bin/python"
SUITES = ("libero_90",)
GATE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIMITS = {"libero_90": 400}
TASKS = (15,16,27,29,36,51,59,65,67,73,80,83,0,6,7,11,20,22,23,28,33,35,39,44)
CATEGORIES = ("single_object_placement", "state_change_only")
TASK_CATEGORIES = {str(t): CATEGORIES[0 if i < 12 else 1] for i,t in enumerate(TASKS)}
DISCOVERY_PHASE = "within_suite_discovery_v1"
CONFIRMATION_PHASE = "within_suite_confirmation_v1"
MASKS = {
    "jac_anchor": (2, 3, 5, 14, 16, 17),
    "local_spatial": (2, 3, 5, 13, 16, 17),
    "local_object": (2, 3, 5, 14, 15, 17),
    "local_goal": (2, 3, 11, 14, 16, 17),
    "local_long": (2, 3, 5, 12, 16, 17),
}
CONDITIONS = ("full_d10", *MASKS)
RUNTIME_FILES = ("__init__.py", "common.py", "policy.py", "rollout.py", "worker.py", "server.py", "runner.py")
SCIENCE_FILES = ("prepare.py", "selection.py", "analyze.py", "test_science.py")
SHARED_ANALYSIS_FILES = ("preparations/ease_insights_20261103/vlm/analyze.py", "preparations/ease_insights_20261103/vlm/analyze_confirmation.py", "preparations/ease_insights_20261103/pipeline.py")


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
        raise ValueError("Condition differs from the six-condition within-suite whitelist")
    return tuple(config["skip"])


def stage_for(manifest):
    if manifest.get("phase") == DISCOVERY_PHASE and manifest.get("discovery_inits") == [36,37,38]:
        return "discovery"
    if manifest.get("phase") == CONFIRMATION_PHASE and manifest.get("confirmation_inits") == list(range(39,44)):
        return "confirm"
    raise ValueError("Not a fixed within-suite stage/init manifest")


def validate_manifest(manifest):
    stage = stage_for(manifest)
    if (manifest.get("conditions") != conditions() or manifest.get("suites") != list(SUITES) or
        manifest.get("tasks") != list(TASKS) or manifest.get("categories") != list(CATEGORIES) or
        manifest.get("task_categories") != TASK_CATEGORIES or
        manifest.get("case_table") != [dict(suite="libero_90", task=t) for t in TASKS]):
        raise ValueError("Fixed24 task/category/condition universe differs")
    science = manifest.get("science", {})
    if science.get("policy_action_limit") != 400 or science.get("settling_actions") != 10:
        raise ValueError("Both categories require400 policy actions plus10 settling actions")
    planned = manifest.get("planned_conditions")
    if stage == "discovery":
        if planned != list(CONDITIONS) or manifest.get("cases_per_condition") != 72 or manifest.get("total_discovery_episodes") != 432:
            raise ValueError("Discovery requires6x24x3=432 episodes")
    elif (not isinstance(planned,list) or len(planned)!=3 or planned[0]!="full_d10" or len(set(planned))!=3 or
          planned[1:] != sorted(planned[1:]) or any(c not in MASKS for c in planned[1:]) or
          manifest.get("cases_per_condition")!=120 or manifest.get("total_confirmation_episodes")!=360):
        raise ValueError("Confirmation requires Full and frozen pair,3x24x5=360 episodes")
    return manifest


def cases(manifest, stage=None, shard_index=0, shards=1):
    actual = stage_for(manifest)
    if stage not in (None,actual):
        raise ValueError("Requested stage differs from immutable manifest")
    if type(shards) is not int or type(shard_index) is not int or not 0 <= shard_index < shards:
        raise ValueError("Invalid task shard")
    if manifest.get("case_table") != [dict(suite="libero_90",task=t) for t in TASKS]:
        raise ValueError("Explicit24task case table differs")
    inits = manifest["discovery_inits" if actual=="discovery" else "confirmation_inits"]
    return [dict(**row,init=i) for index,row in enumerate(manifest["case_table"])
            if index % shards == shard_index for i in inits]


def episode_dir(output, condition, scene):
    return Path(output)/condition/"episodes"/scene["suite"]/("task%02d"%scene["task"])/("init%02d"%scene["init"])


def planned_config(manifest, condition):
    validate_manifest(manifest)
    if condition not in manifest["planned_conditions"]:
        raise ValueError("Condition is not planned for frozen stage")
    cfg=next(c for c in manifest["conditions"] if c["id"]==condition)
    validate_condition(cfg)
    return cfg


def validate_output(output, manifest):
    validate_manifest(manifest)
    if Path(output).resolve()!=Path(manifest["output_root"]).resolve():
        raise ValueError("Output differs from frozen stage destination")
    for name in ("protocol_evidence", "design_evidence"):
        evidence=manifest[name]
        if sha(evidence["path"])!=evidence["sha256"]:
            raise ValueError("Frozen "+name+" changed")
    for name in ("candidate_input_evidence",):
        if name in manifest and sha(manifest[name]["path"]) != manifest[name]["sha256"]:
            raise ValueError("Frozen "+name+" changed")
    resource=manifest.get("resource_execution_revision",{}).get("evidence")
    if resource and sha(resource["path"]) != resource["sha256"]:
        raise ValueError("Frozen resource execution evidence changed")
    if stage_for(manifest)=="confirm":
        ev=manifest["selection_evidence"]
        value=json.loads(Path(ev["path"]).read_text()); supplied=value.pop("selection_hash",None)
        if sha(ev["path"])!=ev["sha256"] or supplied!=ev["selection_hash"] or digest(value)!=supplied:
            raise ValueError("Frozen selection digest differs")
        choice=value["selection"]
        if (choice["status"]!="selected" or manifest["planned_conditions"] != ["full_d10",*choice["selected"]["paths"]] or
            manifest["preregistered_hypotheses"]!=choice["selected"]["directions"]):
            raise ValueError("Confirmation differs from frozen pair/directions")


def read_manifest(path, verify_sources=False):
    manifest=_read_manifest(path,verify_sources=verify_sources)
    validate_output(Path(path).resolve().parent,manifest)
    if verify_sources:
        verify_analysis_sources(manifest)
    return manifest


def runtime_revision(manifest):
    value=dict(name="vlm-within-suite-isolated-runtime-v1", scientific_manifest_hash=manifest["manifest_hash"],
        phase=manifest["phase"],protocol_evidence=manifest["protocol_evidence"],design_evidence=manifest["design_evidence"],
        wrapper_source_files=manifest["local_runtime_sources"],
        backend_reused="preparations.ease_insights_20261103.vlm.backend.Backend",
        mechanics_reused="preparations.ease_insights_20261103.vlm.policy.Policy.predict/reset/acknowledge",
        policy_action_limit=400,settling_actions=10,legacy_sources_changed=False,
        finite_gate_suites=list(GATE_SUITES),gate_covers_new_tasks_or_all_prompts=False)
    value["revision_hash"]=digest(value)
    return value


def verify_analysis_sources(manifest):
    sources=manifest.get("analysis_sources")
    required={str((Path(__file__).parent/name).relative_to(ROOT)) for name in SCIENCE_FILES}|set(SHARED_ANALYSIS_FILES)
    if not isinstance(sources,dict) or not required.issubset(sources):
        raise ValueError("Frozen analysis sources are missing")
    for filename,expected in sources.items():
        if sha(ROOT/filename)!=expected:
            raise ValueError("Analysis source changed without technical revision: "+filename)
    return sources
