"""Freeze local-search stages with CPU-only official-state and prior-use audits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from preparations.idea1_v3_1.common import environment
from preparations.idea1_v3_1.common import read_manifest as read_legacy_manifest
from .common import (ROOT, SUITES, MODULE, ENV_PYTHON, MASKS, CONDITIONS, RUNTIME_FILES,
    DISCOVERY_PHASE, CONFIRMATION_PHASE, conditions, atomic_json, digest, sha, read_manifest, validate_output)

PACKAGE = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "results/ease_insights_20261103/vlm_confirmation/manifest.json"
DEFAULT_PROTOCOL = PACKAGE / "protocol.json"
DEFAULT_DESIGN = PACKAGE.parent / "local_search_design/candidate_proposal_receipt.json"


def inspect_inits():
    """Deserialize only pinned official CPU state files; never instantiate a renderer."""
    from libero.libero import benchmark, get_libero_path
    from preparations.idea1_v2_1.runtime_client import official_init_states
    import numpy as np
    rows = []
    for suite in SUITES:
        selected = benchmark.get_benchmark_dict()[suite]()
        for task in range(10):
            definition = selected.get_task(task)
            states = official_init_states(selected, task)
            if len(states) <= 43:
                raise ValueError(f"Insufficient official inits: {suite}/{task}")
            bddl = Path(get_libero_path("bddl_files")) / definition.problem_folder / definition.bddl_file
            init_file = Path(get_libero_path("init_states")) / definition.problem_folder / definition.init_states_file
            rows.append(dict(suite=suite, task=task, available=len(states), prompt=definition.language,
                bddl_path=str(bddl), bddl_sha256=sha(bddl), init_file=str(init_file), init_file_sha256=sha(init_file),
                init_sha256={str(i): hashlib.sha256(np.asarray(states[i]).tobytes()).hexdigest() for i in range(36, 44)}))
    return dict(passed=True, cases=rows, discovery_inits=[36, 37, 38], reserved_confirmation_inits=list(range(39, 44)),
                method="Official pinned torch CPU state files; no environment/render/GPU construction")


def read_protocol(path):
    obj = json.loads(Path(path).read_text())
    if (obj.get("revision") != "ease-vlm-local-search-v1" or
        obj.get("status") != "prospectively_frozen_before_new_outcomes" or
        obj.get("conditions") != {c["id"]: c["skip"] for c in conditions()} or
        obj["discovery"]["inits"] != [36, 37, 38] or obj["confirmation"]["inits"] != list(range(39, 44)) or
        obj["discovery"]["episodes"] != 720 or obj["confirmation"]["episodes"] != 600):
        raise ValueError("Unexpected bounded local-search protocol")
    for stage in ("discovery", "confirmation"):
        if obj[stage]["suites"] != list(SUITES) or obj[stage]["tasks"] != list(range(10)):
            raise ValueError("Protocol case universe changed")
    return obj


def read_design(path):
    obj = json.loads(Path(path).read_text())
    payload = dict(obj); supplied = payload.pop("receipt_hash", None)
    if supplied is None or digest(payload) != supplied:
        raise ValueError("Candidate design receipt hash differs")
    if (obj.get("proposal_only") is not True or obj.get("new_initial_state_outcomes_read") is not False or
        obj["constraints"]["anchor"] != list(MASKS["jac_anchor"])):
        raise ValueError("Candidate proposal was not independent of new outcomes")
    mapping = {}
    for cid, mask in MASKS.items():
        if cid == "jac_anchor":
            continue
        matches = [r for r in obj["proposals"] if r["skip"] == list(mask)]
        if len(matches) != 1:
            raise ValueError("Candidate does not map uniquely to its historical proposal")
        mapping[cid] = dict(design_id=matches[0]["id"], skip=list(mask), proposal_suite=matches[0]["proposal_suite"],
                           label_meaning="Historical proposal provenance only; no observed preference asserted")
    for item in [*obj["inputs"].values(), obj["generator"]]:
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Historical candidate source changed: " + item["path"])
    return obj, mapping


def _config_from_document(obj):
    config = obj.get("config")
    if isinstance(config, dict):
        return config
    if isinstance(obj.get("protocol"), dict) and isinstance(obj["protocol"].get("config"), dict):
        return obj["protocol"]["config"]
    return obj


def semantic_identity(config):
    """Classify explicit model/mask evidence; unresolved identities stay unresolved."""
    model = config.get("model", config.get("model_id"))
    if model is not None and model not in ("pi0", "pi0_libero"):
        return dict(status="different_model", matches=[], model=model)
    skip = next((config[name] for name in ("skip", "skip_layers", "fixed_skip_layers", "skip_layers_1based") if name in config), None)
    inferred_full = False
    if skip is None and config.get("mode") == "full":
        skip = []; inferred_full = True
    if skip is None:
        return dict(status="unresolved_mask", matches=[], model=model)
    try:
        mask = tuple(sorted(int(x) for x in skip))
    except (TypeError, ValueError):
        return dict(status="unresolved_mask", matches=[], model=model)
    matches = [cid for cid in CONDITIONS if mask == MASKS.get(cid, ())]
    return dict(status="matching_mask" if matches else "different_mask", matches=matches,
                model=model, mask=list(mask), inferred_from_explicit_full_mode=inferred_full)


def semantic_match(config):
    # Public small helper retained for prior-use tests and downstream audits.
    return semantic_identity(config)["matches"]


def prior_use_audit(roots, inits, exclude_root=None):
    """Read actual episode identities/results, not unexecuted manifest case lists."""
    roots = [Path(p).resolve() for p in roots if Path(p).exists()]
    command = ["rg", "--files", "--hidden", "--no-ignore", "-g", "identity.json", "-g", "result.json", "-g", "episode_receipt.json", *map(str, roots)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise RuntimeError("Prior-use inventory failed: " + result.stderr)
    found, ambiguous, checked = [], [], 0
    for filename in sorted(set(result.stdout.splitlines())):
        path = Path(filename)
        if exclude_root is not None and (path == Path(exclude_root) or Path(exclude_root) in path.parents):
            continue
        # Most files explicitly identify their init in the canonical path. Skip
        # known out-of-range paths without reading potentially large run reports.
        part_inits = [int(m.group(1)) for part in path.parts if (m := re.match(r"init(\d+)(?:$|\.)", part))]
        if part_inits and not any(i in inits for i in part_inits):
            continue
        try:
            obj = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            ambiguous.append(dict(path=str(path), reason="Unreadable potential outcome identity", error=str(exc)))
            continue
        checked += 1
        if not isinstance(obj, dict):
            continue
        scene = obj.get("case", obj)
        if not isinstance(scene, dict) or scene.get("init") not in inits or scene.get("suite") not in SUITES or scene.get("task") not in range(10):
            continue
        config = _config_from_document(obj)
        resolved = semantic_identity(config)
        if resolved["status"] == "unresolved_mask":
            identity = path.parent / "identity.json"
            if identity.exists() and identity != path:
                try:
                    reference = json.loads(identity.read_text())
                    config = _config_from_document(reference)
                    resolved = semantic_identity(config)
                except (OSError, ValueError, TypeError):
                    pass
        matches = resolved["matches"]
        if matches:
            found.append(dict(path=str(path), sha256=sha(path), case={k:scene[k] for k in ("suite", "task", "init")},
                condition=obj.get("condition", config.get("id")), matching_runtime_conditions=matches,
                comparison="Explicit sorted skip mask or explicit full mode and Pi0/unspecified model, independent of condition alias; missing schedule metadata treated conservatively",
                semantic_identity=resolved))
        elif resolved["status"] == "unresolved_mask":
            ambiguous.append(dict(path=str(path), case=scene,
                reason="Potential Pi0 outcome lacks resolvable mask/full-mode evidence even after canonical identity lookup"))
    audit = dict(passed=not found and not ambiguous, roots=[str(p) for p in roots], requested_inits=list(inits),
        inventoried_files=len(result.stdout.splitlines()), parsed_documents=checked, matching_existing_outcomes=found,
        unresolved_potential_outcomes=ambiguous, excludes_new_output=str(exclude_root) if exclude_root else None,
        scope="Inventory of known explicit model/mask/full-mode evidence in workspace results and policy result roots only; identities/results/receipts, not planned manifest cases. This does not establish absence of outcomes outside these roots or in unrecognized schemas.",
        semantics="Mask aliases compared by actual skip set; jac_anchor collides with old fixed_v12_jacobian. Full collides by empty skip or explicit full mode. Potential Pi0 cases with unresolved mask are not considered unused.",
        outcome_success_not_used=True, inspected_unix=time.time())
    if not audit["passed"]:
        raise ValueError("New initial states are not verified unused: " + json.dumps(audit))
    return audit


def verify_selection(selection, discovery, report):
    value = copy.deepcopy(selection); supplied = value.pop("selection_hash", None)
    if supplied is None or digest(value) != supplied:
        raise ValueError("Selection hash is missing or changed")
    if (selection.get("discovery_manifest_hash") != discovery["manifest_hash"] or
        selection.get("complete_discovery_episodes") != 720 or selection.get("expected_discovery_episodes") != 720 or
        selection.get("independent_validation_launched") is not False or
        selection.get("protocol_evidence") != discovery["protocol_evidence"] or
        selection.get("design_evidence") != discovery["design_evidence"]):
        raise ValueError("Selection does not bind complete discovery before confirmation")
    choice = selection["selection"]
    if choice.get("status") != "selected" or choice.get("selected") is None:
        raise ValueError("No qualified pair: confirmation is forbidden")
    if not report.get("complete") or report.get("valid_episodes") != 720 or report.get("expected_episodes") != 720 or report["selection"] != choice or report["evidence"] != selection["evidence"]:
        raise ValueError("Discovery/selection evidence changed")
    chosen = choice["selected"]; pair = chosen["paths"]
    if chosen.get("pair") != pair or len(pair) != 2 or len(set(pair)) != 2 or any(cid not in MASKS for cid in pair):
        raise ValueError("Selected pair must contain two local-search candidates")
    directions = chosen["directions"]
    if (len(directions) != 2 or len({d["suite"] for d in directions}) != 2 or
        any(d["suite"] not in SUITES or {d["better"],d["worse"]} != set(pair) for d in directions) or
        {d["better"] for d in directions} != set(pair)):
        raise ValueError("Selection must freeze opposite directions in two different suites")
    return chosen


def prepare(output, stage="discovery", protocol=None, design=None, selection=None, discovery_root=None, base_manifest=None):
    if stage not in ("discovery", "confirm"):
        raise ValueError("Stage must be discovery or confirm")
    output = Path(output).resolve()
    protocol_path = Path(protocol or DEFAULT_PROTOCOL).resolve(); protocol_obj = read_protocol(protocol_path)
    design_path = Path(design or DEFAULT_DESIGN).resolve(); design_obj, mapping = read_design(design_path)
    stage_key = "discovery" if stage == "discovery" else "confirmation"
    expected_output = (ROOT/protocol_obj[stage_key]["root"]).resolve()
    if output != expected_output:
        raise ValueError("Output must be the destination frozen in protocol.json")
    target = output/"manifest.json"
    protocol_evidence = dict(path=str(protocol_path), sha256=sha(protocol_path))
    design_evidence = dict(path=str(design_path), sha256=sha(design_path), receipt_hash=design_obj["receipt_hash"])
    if target.exists():
        existing = read_manifest(target, verify_sources=True)
        validate_output(output, existing)
        if existing["protocol_evidence"] != protocol_evidence or existing["design_evidence"] != design_evidence:
            raise ValueError("Existing stage belongs to another protocol/design")
        if stage == "confirm" and (selection is None or existing["selection_evidence"]["path"] != str(Path(selection).resolve())):
            raise ValueError("Existing confirmation belongs to another selection")
        return dict(reused=True, path=str(target), manifest_hash=existing["manifest_hash"],
            planned_conditions=existing["planned_conditions"], episodes=existing["total_discovery_episodes" if stage == "discovery" else "total_confirmation_episodes"], gpu_launched=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError("First stage freeze requires an empty new output directory")
    if stage == "discovery" and selection is not None:
        raise ValueError("Discovery cannot be supplied a heldout selection")
    chosen = None; discovery = None; selection_path = None
    if stage == "confirm":
        if selection is None:
            raise ValueError("Confirmation requires frozen selection.json")
        discovery_root = Path(discovery_root or ROOT/protocol_obj["discovery"]["root"]).resolve()
        if output == discovery_root or output in discovery_root.parents or discovery_root in output.parents:
            raise ValueError("Discovery and confirmation must occupy separate trees")
        discovery = read_manifest(discovery_root/"manifest.json", verify_sources=True)
        if discovery["protocol_evidence"] != protocol_evidence or discovery["design_evidence"] != design_evidence:
            raise ValueError("Confirmation protocol/design differs from discovery")
        if discovery["phase"] != DISCOVERY_PHASE:
            raise ValueError("Confirmation reference is not local-search discovery")
        selection_path = Path(selection).resolve(); selected = json.loads(selection_path.read_text())
        # Validate predeclared selection digest/status before touching heldout state.
        check = dict(selected); supplied = check.pop("selection_hash", None)
        if supplied is None or digest(check) != supplied or selected.get("selection",{}).get("status") != "selected":
            raise ValueError("Confirmation requires a valid selected frozen choice")
        from .selection import analyze_discovery
        chosen = verify_selection(selected, discovery, analyze_discovery(discovery_root))
    base_path = Path(base_manifest or DEFAULT_BASE).resolve()
    base = read_legacy_manifest(base_path, verify_sources=True)
    expected_science = dict(H=50, R=5, draft_D=2, refresh_D=10, retained_layers=12, depth=18)
    if any(base["science"].get(k) != v for k,v in expected_science.items()):
        raise ValueError("Legacy asset manifest has different fixed scientific budgets")
    if base["noise"].get("condition_independent") is not True:
        raise ValueError("Legacy noise protocol is not paired")
    inits = [36,37,38] if stage == "discovery" else list(range(39,44))
    # Discovery also reserves every heldout init before any new result exists.
    audit_inits = list(range(36,44)) if stage == "discovery" else inits
    roots = [ROOT/"results", *sorted((ROOT/"policies").glob("*/results"))]
    prior = prior_use_audit(roots, audit_inits)
    env = environment(); env["CUDA_VISIBLE_DEVICES"] = ""
    probe = subprocess.check_output([str(ENV_PYTHON), "-m", MODULE+".prepare", "--inspect-inits"], cwd=ROOT, env=env, text=True)
    official = json.loads(probe.splitlines()[-1])
    if not official.get("passed") or len(official["cases"]) != 40:
        raise ValueError("Official state audit incomplete")
    # Keep the shared asset/scientific metadata only, not the old phase/selection.
    keys = ("checkpoints", "bank", "environment", "noise", "science", "quality_environment_reuse", "technical_max_attempts", "preflight")
    manifest = {key:copy.deepcopy(base[key]) for key in keys}
    source_files = dict(base["source_files"])
    local_files = [PACKAGE/name for name in RUNTIME_FILES]
    decision_files = [PACKAGE/name for name in ("selection.py", "analyze.py", "supervisor.py")]
    if any(not p.exists() for p in decision_files):
        raise ValueError("Selection, analysis, and supervisor must exist before source freeze")
    local_sources = {str(p.relative_to(ROOT)):sha(p) for p in local_files}
    source_files.update(local_sources)
    analysis_dependencies = [PACKAGE.parent/"vlm"/"analyze.py", PACKAGE.parent/"vlm"/"analyze_confirmation.py", PACKAGE.parent/"pipeline.py"]
    analysis_sources = {str(p.relative_to(ROOT)):sha(p) for p in decision_files+analysis_dependencies}
    # Recovery shares the old receipt-selection mechanics; actual wrappers are
    # separately identified in every new episode receipt by runtime_revision.
    for name in ("recovery.py", "worker_v2.py", "runner_v2.py"):
        p = PACKAGE.parent/"vlm"/name; source_files[str(p.relative_to(ROOT))] = sha(p)
    manifest.update(revision="ease-vlm-bounded-local-search-v1", phase=DISCOVERY_PHASE if stage=="discovery" else CONFIRMATION_PHASE,
        output_root=str(output), conditions=conditions(), planned_conditions=list(CONDITIONS) if stage=="discovery" else ["full_d10",*chosen["paths"]],
        suites=list(SUITES), tasks=list(range(10)), cases_per_condition=120 if stage=="discovery" else 200,
        official_init_availability=official, protocol_evidence=protocol_evidence, design_evidence=design_evidence,
        candidate_design_mapping=mapping, initial_state_prior_use_audit=prior,
        legacy_asset_reference=dict(path=str(base_path), sha256=sha(base_path), manifest_hash=base["manifest_hash"],
            purpose="Reuse checkpoint/bank/environment/noise/scientific budgets; prior null results remain unchanged"),
        local_runtime_sources=local_sources, source_files=source_files, analysis_sources=analysis_sources,
        fixed_input_cost=dict(real_observations_per_suite=1, suites=list(SUITES), conditions="All six frozen definitions",
            warmup_rounds_per_condition_per_suite=6, measured_rounds_per_condition_per_suite=12,
            total_warmup_requests=144, total_measured_requests=288,
            primary="Native completed policy Full/Fast/Fast cost; RPC and environment excluded",
            scope="Same-input descriptive timing only; no closed-loop quality or exclusive-GPU claim"),
        gate_condition_scope="Every frozen candidate and Full; all six definitions retained during either phase")
    if stage=="discovery":
        manifest.update(discovery_inits=inits,total_discovery_episodes=720,
            independent_validation=dict(status="locked_pending_qualified_selection", inits=list(range(39,44)),
                selected_pair=None, rule="Separate manifest only after all720 valid and frozen qualified pair/directions"))
    else:
        manifest.update(confirmation_inits=inits,total_confirmation_episodes=600,
            discovery_reference=dict(root=str(discovery_root), manifest_hash=discovery["manifest_hash"],
                manifest_sha256=sha(discovery_root/"manifest.json"), valid_episodes_at_freeze=720),
            selection_evidence=dict(path=str(selection_path),sha256=sha(selection_path),selection_hash=selected["selection_hash"],
                frozen_selection=selected["selection"],evidence=selected["evidence"]),
            preregistered_hypotheses=copy.deepcopy(chosen["directions"]),
            heldout_preparation=dict(no_heldout_data_at_first_freeze=True, independent_initial_states=inits,
                outcome_inspection_before_freeze=False))
    manifest["manifest_hash"] = digest(manifest)
    validate_output(output, manifest)
    atomic_json(target, manifest)
    return dict(reused=False,path=str(target),manifest_hash=manifest["manifest_hash"],planned_conditions=manifest["planned_conditions"],
        episodes=720 if stage=="discovery" else 600,gpu_launched=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage", choices=("discovery", "confirm"), default="discovery")
    parser.add_argument("--output"); parser.add_argument("--protocol"); parser.add_argument("--design")
    parser.add_argument("--selection"); parser.add_argument("--discovery-root"); parser.add_argument("--base-manifest")
    parser.add_argument("--inspect-inits", action="store_true")
    args = parser.parse_args()
    if args.inspect_inits:
        value = inspect_inits()
    else:
        if not args.output:
            parser.error("--output is required")
        value = prepare(args.output,args.stage,args.protocol,args.design,args.selection,args.discovery_root,args.base_manifest)
    print(json.dumps(value))


if __name__ == "__main__":
    main()
