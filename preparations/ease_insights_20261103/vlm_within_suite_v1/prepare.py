"""CPU asset/prior-use audits and deliberate post-review independent manifest freeze."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from preparations.idea1_v3_1.common import environment, read_manifest as read_legacy_manifest
from .common import (ROOT, SUITES, GATE_SUITES, TASKS, TASK_CATEGORIES, CATEGORIES, MODULE,
    ENV_PYTHON, MASKS, CONDITIONS, RUNTIME_FILES, SCIENCE_FILES, SHARED_ANALYSIS_FILES,
    DISCOVERY_PHASE, CONFIRMATION_PHASE, conditions, atomic_json, digest, sha, read_manifest,
    validate_output)

PACKAGE=Path(__file__).resolve().parent
DEFAULT_BASE=ROOT/"results/ease_insights_20261103/vlm_local_search_v1/discovery/manifest.json"
DEFAULT_PROTOCOL=PACKAGE/"protocol.json"
DEFAULT_DESIGN=ROOT/"reports/ease_insights_20261103/semantic_pool_feasibility/within_suite_candidate/candidate.json"
CANDIDATE_SHA256="478b637d0e688ca8178551ef4c18e548c70492045844b1f5374923aeeada47bc"
DEFAULT_INVENTORY=ROOT/"reports/ease_insights_20261103/semantic_pool_feasibility/task_inventory.csv"
RESOURCE_SOURCES=("preparations/ease_insights_20261103/vlm_within_suite_v1/borrowed_controller.py",
                  "preparations/ease_insights_20261103/retry_resource_session_port.py")


def accepted_cross_service_reference():
    """Exact old acceptedJac numerical artifact; finite four-suite scope only."""
    old=ROOT/"results/ease_insights_20261103/vlm_local_search_v1/discovery"
    reference=dict(path=str(old/"cross_service_reference.npz"),
        sha256="1ba90f16ffc16e2df9e74f3fc7a27004bfc182b8f47c1c177ab7e28a6e55621c",
        manifest_hash="032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9",
        reference_manifest=dict(path=str(old/"manifest.json"),sha256="7906d0fafa676e9072fe1eb2693c1eeac6b246c5f676a9ac79a0fa3b96b61fbf"),
        source_protocol=dict(path=str(old/"jac_anchor/runs/discovery_shard00of01_1793783749619082188/protocol.json"),
            sha256="f61249d89b55d7633136c45ab9edf419cc67ce904221b3ac03dbf4b706968078"))
    for evidence in (reference,reference["reference_manifest"],reference["source_protocol"]):
        if sha(evidence["path"])!=evidence["sha256"]:
            raise ValueError("Old acceptedJac reference evidence changed: "+evidence["path"])
    source=json.loads(Path(reference["source_protocol"]["path"]).read_text())
    old_manifest=json.loads(Path(reference["reference_manifest"]["path"]).read_text())
    if (old_manifest["manifest_hash"]!=reference["manifest_hash"] or
        source.get("manifest_hash")!=reference["manifest_hash"] or source.get("condition_id")!="jac_anchor" or
        source.get("qk_reference_passed") is not True or source.get("real_input_parity_passed") is not True or
        source.get("config")!=next(c for c in conditions() if c["id"]=="jac_anchor") or
        source.get("checkpoint_sha256")!=old_manifest["checkpoints"]["pi0"]["sha256"]):
        raise ValueError("Old acceptedJac source protocol identity differs")
    return reference


def pin_normalization_asset(checkpoint, preflight_path=None):
    """Add the pinned historical preflight's exact norm asset to direct checks."""
    historical=Path(preflight_path or ROOT/"policies/realtime-vla-flash/results/experiment11_layer_skip_prefix_20261010/provenance/preflight.json")
    artifacts=checkpoint["identity_artifacts"]
    expected=artifacts.get(str(historical))
    if expected is None or sha(historical)!=expected:
        raise ValueError("Historical normalization preflight identity differs")
    normalization=json.loads(historical.read_text())["normalization_asset"]
    asset_path=normalization["path"];asset_sha=normalization["sha256"]
    if sha(asset_path)!=asset_sha:
        raise ValueError("Normalization asset SHA differs from pinned historical preflight: "+asset_path)
    if asset_path in artifacts and artifacts[asset_path]!=asset_sha:
        raise ValueError("Normalization asset identity conflicts with an existing checkpoint artifact")
    artifacts[asset_path]=asset_sha
    return dict(path=asset_path,sha256=asset_sha)


def read_design(path=DEFAULT_DESIGN):
    path=Path(path)
    if sha(path)!=CANDIDATE_SHA256:
        raise ValueError("Fixed metadata candidate SHA256 differs")
    value=json.loads(path.read_text())
    if (value.get("total_tasks")!=24 or [r["task"] for r in value["tasks"]]!=list(TASKS) or
        {str(r["task"]):r["category"] for r in value["tasks"]}!=TASK_CATEGORIES or
        any(r["suite"]!="libero_90" for r in value["tasks"]) or
        len({r["full_bddl_identity"] for r in value["tasks"]})!=24):
        raise ValueError("Fixed candidate24 tasks/categories/physical identities differ")
    if sha(value["input_inventory"])!=value["input_sha256"]:
        raise ValueError("Candidate input inventory changed")
    return value


def read_protocol(path=DEFAULT_PROTOCOL):
    value=json.loads(Path(path).read_text())
    if (value.get("revision")!="ease-vlm-within-suite-v1" or
        value.get("status")!="prospectively_frozen_before_new_outcomes" or
        value.get("conditions")!={c["id"]:c["skip"] for c in conditions()} or
        value["candidate_generation"]["sha256"]!=CANDIDATE_SHA256):
        raise ValueError("Unexpected within-suite protocol/condition evidence")
    for stage,n,inits in (("discovery",432,[36,37,38]),("confirmation",360,list(range(39,44)))):
        row=value[stage]
        if (row["suites"]!=list(SUITES) or row["tasks"]!=list(TASKS) or row["categories"]!=list(CATEGORIES) or
            row["inits"]!=inits or row["episodes"]!=n or row["tasks_per_category"]!=12):
            raise ValueError("Protocol case universe differs")
    if (value["fixed_science"]["policy_action_limit"]!=400 or value["fixed_science"]["settling_actions"]!=10 or
        value["execution"]["new_scientific_episode_cap"]!=792 or
        value["confirmation"]["bootstrap_seed"]!=20261105 or value["confirmation"]["bootstrap_replicates"]!=20000 or
        value["confirmation"]["bootstrap_quantile_method"]!="linear"):
        raise ValueError("Protocol fixed budget/statistics differ")
    return value


def inspect_inits(design=DEFAULT_DESIGN):
    """Read official benchmark definitions and CPU state vectors; no env/render/model."""
    candidate=read_design(design)
    from libero.libero import benchmark, get_libero_path
    from preparations.idea1_v2_1.runtime_client import official_init_states
    import numpy as np
    selected=benchmark.get_benchmark_dict()["libero_90"]()
    rows=[]
    for target in candidate["tasks"]:
        task=target["task"]; definition=selected.get_task(task)
        bddl=Path(get_libero_path("bddl_files"))/definition.problem_folder/definition.bddl_file
        init_file=Path(get_libero_path("init_states"))/definition.problem_folder/definition.init_states_file
        if (definition.name!=target["task_name"] or definition.language!=target["benchmark_instruction"] or
            sha(bddl)!=target["bddl_sha256"] or sha(init_file)!=target["init_sha256"]):
            raise ValueError("Official task/BDDL/init asset differs from candidate: "+str(task))
        states=official_init_states(selected,task)
        if len(states)<=43:
            raise ValueError("Insufficient official states for task"+str(task))
        rows.append(dict(suite="libero_90",task=task,category=target["category"],available=len(states),
            task_name=definition.name,prompt=definition.language,bddl_path=str(bddl),bddl_sha256=sha(bddl),
            full_bddl_identity=target["full_bddl_identity"],init_file=str(init_file),init_file_sha256=sha(init_file),
            init_sha256={str(i):hashlib.sha256(np.asarray(states[i]).tobytes()).hexdigest() for i in range(36,44)}))
    return dict(passed=True,cases=rows,discovery_inits=[36,37,38],reserved_confirmation_inits=list(range(39,44)),
        task_count=24,unique_prompts=len({r["prompt"] for r in rows}),
        candidate_sha256=CANDIDATE_SHA256,
        method="Official benchmark metadata plus pinned BDDL/state files and CPU-deserialized vectors; no environment, rendering, model or GPU construction")


def prior_use_audit(roots,inits,design=DEFAULT_DESIGN,exclude_root=None,raise_on_collision=True):
    """Finite known EASE manifest/case metadata inventory, including physical aliases.

    Planned manifest case lists are indexed as provenance, never counted as
    outcomes. Any executed matching physical task+init blocks this branch,
    independent of model/condition or success. No global absence claim is made.
    """
    candidate=read_design(design)
    inventory=Path(candidate["input_inventory"])
    with inventory.open() as stream:
        catalog={(r["suite"],int(r["task"])):r for r in csv.DictReader(stream)}
    targets={r["full_bddl_identity"]:r for r in candidate["tasks"]}
    raw_targets={r["bddl_sha256"]:r for r in candidate["tasks"]}
    roots=[Path(p).resolve() for p in roots if Path(p).exists()]
    if not roots:
        raise ValueError("Known EASE output roots must exist for the finite prior-use audit")
    command=["rg","--files","--hidden","--no-ignore","-g","manifest.json","-g","identity.json","-g","result.json","-g","episode_receipt.json",*map(str,roots)]
    result=subprocess.run(command,capture_output=True,text=True,check=False)
    if result.returncode not in (0,1) or result.stderr.strip():
        raise RuntimeError("Prior-use inventory failed: "+result.stderr)
    found=[];ambiguous=[];manifests=[];parsed=0;case_documents=0
    for filename in sorted(set(result.stdout.splitlines())):
        path=Path(filename)
        if exclude_root and Path(exclude_root).resolve() in (path.resolve(),*path.resolve().parents):
            continue
        # Exact init paths outside the reserved range cannot collide by index.
        path_inits=[int(m.group(1)) for part in path.parts if (m:=re.fullmatch(r"init(\d+)(?:\.technical.*)?",part))]
        if path.name!="manifest.json" and path_inits and not any(i in inits for i in path_inits):
            continue
        try:
            value=json.loads(path.read_text());parsed+=1
        except (OSError,ValueError) as exc:
            ambiguous.append(dict(path=str(path),reason="Unreadable inventoried metadata",error=str(exc)));continue
        if not isinstance(value,dict):
            continue
        if path.name=="manifest.json":
            manifests.append(dict(path=str(path),sha256=sha(path),manifest_hash=value.get("manifest_hash"),phase=value.get("phase")))
            continue
        scene=value.get("case",value)
        if not isinstance(scene,dict) or not all(k in scene for k in ("suite","task","init")):
            continue
        case_documents+=1
        if scene["init"] not in inits:
            continue
        row=catalog.get((scene["suite"],scene["task"]))
        # Official catalog resolves suite aliases even when raw BDDL text differs.
        physical=row.get("full_bddl_identity") if row else value.get("full_bddl_identity")
        raw=value.get("bddl_sha256")
        target=targets.get(physical) or raw_targets.get(raw)
        if target:
            found.append(dict(path=str(path),sha256=sha(path),case={k:scene[k] for k in ("suite","task","init")},
                candidate_task=target["task"],category=target["category"],physical_bddl_identity=target["full_bddl_identity"],
                comparison="Canonical complete official BDDL identity or matching raw BDDL hash plus same official init index; any model/path blocks, success never used in matching"))
        elif scene["suite"]=="libero_90" and scene["task"] in TASKS and row is None:
            ambiguous.append(dict(path=str(path),case=scene,reason="Potential fixed candidate task lacks resolvable physical identity"))
    audit=dict(passed=not found and not ambiguous,roots=list(map(str,roots)),requested_inits=list(inits),
        inventoried_files=len(set(result.stdout.splitlines())),parsed_documents=parsed,case_documents=case_documents,
        manifest_evidence=manifests,matching_existing_outcomes=found,unresolved_potential_outcomes=ambiguous,
        catalog_evidence=dict(path=str(inventory),sha256=sha(inventory)),candidate_sha256=CANDIDATE_SHA256,
        scope="Only known explicit EASE output roots supplied here, manifest/identity/result/receipt schemas and official catalog physical aliases; no absence claim outside these roots or in unrecognized schemas. Manifest plans alone are not outcomes.",
        outcome_success_not_used=True,inspected_unix=time.time())
    if not audit["passed"] and raise_on_collision:
        raise ValueError("Reserved physical task+init is not verified unused: "+json.dumps(audit))
    return audit


def official_audit(design=DEFAULT_DESIGN):
    env=environment();env["CUDA_VISIBLE_DEVICES"]=""
    output=subprocess.check_output([str(ENV_PYTHON),"-m",MODULE+".prepare","--inspect-inits","--design",str(design)],cwd=ROOT,env=env,text=True)
    value=json.loads(output.splitlines()[-1])
    if not value.get("passed") or len(value.get("cases",[]))!=24 or value.get("unique_prompts")!=20:
        raise ValueError("Official24task/20prompt CPU asset audit incomplete")
    return value


def verify_selection(selection,discovery,report):
    payload=copy.deepcopy(selection);supplied=payload.pop("selection_hash",None)
    if supplied is None or digest(payload)!=supplied:
        raise ValueError("Selection hash missing or changed")
    if (selection.get("discovery_manifest_hash")!=discovery["manifest_hash"] or
        selection.get("complete_discovery_episodes")!=432 or selection.get("expected_discovery_episodes")!=432 or
        selection.get("independent_validation_launched") is not False or
        selection.get("protocol_evidence")!=discovery["protocol_evidence"] or selection.get("design_evidence")!=discovery["design_evidence"]):
        raise ValueError("Selection does not bind complete432episode discovery")
    choice=selection["selection"]
    if choice.get("status")!="selected" or not choice.get("selected"):
        raise ValueError("No qualified pair; confirmation forbidden")
    if (not report.get("complete") or report.get("valid_episodes")!=432 or report.get("expected_episodes")!=432 or
        report["selection"]!=choice or report["evidence"]!=selection["evidence"]):
        raise ValueError("Discovery selection evidence changed")
    chosen=choice["selected"];pair=chosen["paths"];directions=chosen["directions"]
    if chosen.get("pair")!=pair or len(pair)!=2 or pair!=sorted(set(pair)) or any(c not in MASKS for c in pair):
        raise ValueError("Selected pair must be two distinct sorted nonFull candidates")
    if (len(directions)!=2 or {d["category"] for d in directions}!=set(CATEGORIES) or
        {d["better"] for d in directions}!=set(pair) or any({d["better"],d["worse"]}!=set(pair) for d in directions)):
        raise ValueError("Two opposite directions across the fixed metadata classes required")
    return chosen


def prepare(output,stage="discovery",protocol=None,design=None,selection=None,discovery_root=None,
            base_manifest=None,extra_sources=(),freeze_reviewed=False,resource_evidence=None):
    if not freeze_reviewed:
        raise ValueError("Runnable freeze requires explicit freeze_reviewed=True after final runtime/controller review; use --audit-only for preparation")
    if stage not in ("discovery","confirm"):
        raise ValueError("Stage must be discovery or confirm")
    output=Path(output).resolve();protocol_path=Path(protocol or DEFAULT_PROTOCOL).resolve();protocol_obj=read_protocol(protocol_path)
    design_path=Path(design or DEFAULT_DESIGN).resolve();candidate=read_design(design_path)
    stage_key="discovery" if stage=="discovery" else "confirmation"
    if output!=(ROOT/protocol_obj[stage_key]["root"]).resolve():
        raise ValueError("Output must equal protocol destination")
    protocol_evidence=dict(path=str(protocol_path),sha256=sha(protocol_path))
    design_evidence=dict(path=str(design_path),sha256=sha(design_path),candidate_sha256=CANDIDATE_SHA256)
    target=output/"manifest.json"
    if target.exists():
        existing=read_manifest(target,verify_sources=True)
        if existing["protocol_evidence"]!=protocol_evidence or existing["design_evidence"]!=design_evidence:
            raise ValueError("Existing stage belongs to a different protocol/candidate")
        if stage=="confirm" and (selection is None or existing["selection_evidence"]["path"]!=str(Path(selection).resolve())):
            raise ValueError("Existing confirmation belongs to another selection")
        return dict(reused=True,path=str(target),manifest_hash=existing["manifest_hash"],gpu_launched=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError("First freeze requires empty independent output directory")
    chosen=None;discovery=None;selected=None;selection_path=None
    if stage=="discovery" and selection is not None:
        raise ValueError("Discovery cannot be supplied heldout selection")
    if stage=="confirm":
        if selection is None:
            raise ValueError("Confirmation requires frozen selection.json")
        discovery_root=Path(discovery_root or ROOT/protocol_obj["discovery"]["root"]).resolve()
        if output==discovery_root or output in discovery_root.parents or discovery_root in output.parents:
            raise ValueError("Discovery/confirmation must occupy separate trees")
        discovery=read_manifest(discovery_root/"manifest.json",verify_sources=True)
        if discovery["protocol_evidence"]!=protocol_evidence or discovery["design_evidence"]!=design_evidence:
            raise ValueError("Confirmation protocol/candidate differs from discovery")
        if discovery["phase"]!=DISCOVERY_PHASE:
            raise ValueError("Confirmation reference not within-suite discovery")
        selection_path=Path(selection).resolve();selected=json.loads(selection_path.read_text())
        check=copy.deepcopy(selected);supplied=check.pop("selection_hash",None)
        if supplied is None or digest(check)!=supplied or selected.get("selection",{}).get("status")!="selected":
            raise ValueError("Confirmation needs valid selected decision before heldout access")
        from .selection import analyze_discovery
        chosen=verify_selection(selected,discovery,analyze_discovery(discovery_root))
    base_path=Path(base_manifest or DEFAULT_BASE).resolve();base=read_legacy_manifest(base_path,verify_sources=True)
    if any(base["science"].get(k)!=v for k,v in dict(H=50,R=5,draft_D=2,refresh_D=10,retained_layers=12,depth=18).items()):
        raise ValueError("Legacy asset manifest has different scientific budget")
    if base["noise"].get("condition_independent") is not True:
        raise ValueError("Noise protocol must be condition-independent")
    inits=[36,37,38] if stage=="discovery" else list(range(39,44))
    prior=prior_use_audit([ROOT/"results/ease_insights_20261103"],list(range(36,44)) if stage=="discovery" else inits,design_path,exclude_root=output)
    official=official_audit(design_path)
    required=[PACKAGE/name for name in (*RUNTIME_FILES,*SCIENCE_FILES)]
    required.extend(ROOT/name for name in RESOURCE_SOURCES)
    required.append(PACKAGE/"test_runtime.py")
    required.extend(Path(name) if Path(name).is_absolute() else ROOT/name for name in extra_sources)
    if any(not p.is_file() for p in required):
        raise ValueError("All final runtime/science/resource sources must exist before freeze")
    if resource_evidence is None:
        raise ValueError("Final ended-study/resource execution evidence JSON is required before runnable freeze")
    evidence_path=Path(resource_evidence).resolve();resource=json.loads(evidence_path.read_text())
    if (not isinstance(resource,dict) or resource.get("schema")!="ease-borrowed-idle-holders-proof-v1" or
        resource.get("original_lease_ownership_unchanged") is not True or len(resource.get("holders",{}))!=2):
        raise ValueError("Resource evidence must be the reviewed two-holder external-borrower proof")
    resource_plan=ROOT/"results/ease_insights_20261103/gpu_resource_campaign_v3/plan.json"
    expected_plan_sha="18f3609856c026342bd79064a59a8a97f9e5ebdb9d8eb06240b2f8ab05fb5e42"
    old_plan=json.loads(resource_plan.read_text())
    if sha(resource_plan)!=expected_plan_sha or resource.get("plan_sha256")!=expected_plan_sha:
        raise ValueError("Original resource plan identity differs")
    expected_reports={
        "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_confirmation/audit.json":"347c6bb41a5c6589336e7b58ac4cc174825d7ab6c1072fed973cc7eb59dcd66e",
        "results/ease_insights_20261103/vlm_local_search_v1/confirmation/analysis/analysis.json":"48400436855e9b78cd5ebfa61ae409b765699dcc96280522db16145a861f946b",
        "reports/ease_insights_20261103/vlm_local_confirmation_statistics/audit.json":"096681189368855d25425691f80114f5f5ed15fc76ea3adc32cd9327bc4724da"}
    if resource.get("old_reports")!=expected_reports or any(sha(ROOT/name)!=expected for name,expected in expected_reports.items()):
        raise ValueError("Completed prior-study evidence changed")
    acceptance,analysis,stats=[json.loads((ROOT/name).read_text()) for name in expected_reports]
    if (acceptance.get("status")!="pass" or stats.get("status")!="pass" or stats.get("differences")!=[] or
        analysis.get("complete") is not True or analysis.get("valid_episodes")!=600 or analysis.get("conclusion")!="not_reproduced"):
        raise ValueError("Prior study is not a completed independently accepted negative study")
    # The caller supplies exact ended-study/holder/old-plan pins after review.
    resource_pins=resource.get("source_hashes",{})
    if not isinstance(resource_pins,dict) or resource_pins!=old_plan["source_hashes"] or len(resource_pins)!=8:
        raise ValueError("Resource evidence must preserve exactly the original eight plan source pins")
    for name,expected in resource_pins.items():
        if sha(ROOT/name)!=expected:
            raise ValueError("Frozen resource source changed: "+name)
    keys=("checkpoints","bank","environment","noise","science","quality_environment_reuse","technical_max_attempts","preflight")
    manifest={k:copy.deepcopy(base[k]) for k in keys}
    pin_normalization_asset(manifest["checkpoints"]["pi0"])
    manifest["science"].update(policy_action_limit=400,settling_actions=10,environment_seed=7,training=False)
    manifest["preflight"]["cross_service_reference"]=accepted_cross_service_reference()
    source_files=dict(base["source_files"])
    local_sources={str(p.relative_to(ROOT)):sha(p) for p in required}
    source_files.update(local_sources);source_files.update(resource_pins)
    analysis_sources={str((PACKAGE/name).relative_to(ROOT)):sha(PACKAGE/name) for name in SCIENCE_FILES}
    analysis_sources.update({name:sha(ROOT/name) for name in SHARED_ANALYSIS_FILES})
    source_files.update(analysis_sources)
    # Also bind finite bank gate wrappers and the explicit policy-limit authority.
    for name in ("preparations/ease_insights_20261103/vlm_local_search_v1/gate.py","preparations/ease_insights_20261103/vlm_local_search_v1/benchmark.py","policies/realtime-vla-flash/scripts/spec/spec_client_libero.py"):
        source_files[name]=sha(ROOT/name)
    manifest.update(revision="ease-vlm-bounded-within-suite-v1",phase=DISCOVERY_PHASE if stage=="discovery" else CONFIRMATION_PHASE,
        output_root=str(output),conditions=conditions(),planned_conditions=list(CONDITIONS) if stage=="discovery" else ["full_d10",*chosen["paths"]],
        suites=list(SUITES),tasks=list(TASKS),categories=list(CATEGORIES),task_categories=dict(TASK_CATEGORIES),
        case_table=[dict(suite="libero_90",task=t) for t in TASKS],task_assets=copy.deepcopy(candidate["tasks"]),
        cases_per_condition=72 if stage=="discovery" else 120,official_init_availability=official,
        protocol_evidence=protocol_evidence,design_evidence=design_evidence,candidate_evidence=design_evidence,
        candidate_input_evidence=dict(path=candidate["input_inventory"],sha256=candidate["input_sha256"]),
        initial_state_prior_use_audit=prior,
        legacy_asset_reference=dict(path=str(base_path),sha256=sha(base_path),manifest_hash=base["manifest_hash"],
            purpose="Only checkpoint/normalization/bank/environment/noise/fixed model budgets reused; this is a separate432/360 scientific case universe"),
        local_runtime_sources=local_sources,source_files=source_files,analysis_sources=analysis_sources,
        resource_execution_revision=dict(name="external-borrower-within-suite-v1",new_science=True,
            original_holder_lease_reused=True,original_scientific_manifests_changed=False,
            evidence=dict(path=str(evidence_path),sha256=sha(evidence_path)),details=resource),
        fixed_input_cost=dict(real_observations_per_suite=1,suites=list(GATE_SUITES),conditions="All six frozen definitions",
            warmup_rounds_per_condition_per_suite=6,measured_rounds_per_condition_per_suite=12,
            total_warmup_requests=144,total_measured_requests=288,
            primary="Native completed policy Full/Fast/Fast cost; RPC and environment excluded",
            scope="Existing finite four-suite bank only; no coverage of every new task/prompt, closed-loop acceleration or exclusive-GPU timing claim"),
        gate_condition_scope="All six conditions on old finite four-suite bank; new case first inputs/prompt/noise checked separately",
        prior_negative_results=copy.deepcopy(protocol_obj["previous_evidence"]),new_scientific_episode_cap=792)
    if stage=="discovery":
        manifest.update(discovery_inits=inits,total_discovery_episodes=432,
            independent_validation=dict(status="locked_pending_qualified_selection",inits=list(range(39,44)),selected_pair=None,
                rule="Separate confirmation manifest only after complete432 and frozen qualified pair/two category directions"))
    else:
        manifest.update(confirmation_inits=inits,total_confirmation_episodes=360,
            discovery_reference=dict(root=str(discovery_root),manifest_hash=discovery["manifest_hash"],manifest_sha256=sha(discovery_root/"manifest.json"),valid_episodes_at_freeze=432),
            selection_evidence=dict(path=str(selection_path),sha256=sha(selection_path),selection_hash=selected["selection_hash"],frozen_selection=selected["selection"],evidence=selected["evidence"]),
            preregistered_hypotheses=copy.deepcopy(chosen["directions"]),
            heldout_preparation=dict(no_heldout_data_at_first_freeze=True,independent_initial_states=inits,outcome_inspection_before_freeze=False))
    manifest["manifest_hash"]=digest(manifest);validate_output(output,manifest);atomic_json(target,manifest)
    return dict(reused=False,path=str(target),manifest_hash=manifest["manifest_hash"],planned_conditions=manifest["planned_conditions"],episodes=432 if stage=="discovery" else 360,gpu_launched=False)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage",choices=("discovery","confirm"),default="discovery")
    parser.add_argument("--output");parser.add_argument("--protocol");parser.add_argument("--design")
    parser.add_argument("--selection");parser.add_argument("--discovery-root");parser.add_argument("--base-manifest")
    parser.add_argument("--extra-source",action="append",default=[]);parser.add_argument("--resource-evidence")
    parser.add_argument("--inspect-inits",action="store_true");parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--freeze-reviewed",action="store_true",help="Caller has reviewed final runtime/controller sources before independent runnable freeze")
    args=parser.parse_args();design=args.design or DEFAULT_DESIGN
    if args.inspect_inits:
        value=inspect_inits(design)
    elif args.audit_only:
        if not args.output:parser.error("--output report directory is required")
        read_protocol(args.protocol or DEFAULT_PROTOCOL)
        prior=prior_use_audit([ROOT/"results/ease_insights_20261103"],list(range(36,44)),design,raise_on_collision=False)
        official=official_audit(design)
        value=dict(revision="within-suite-preparation-cpu-audit-v1",passed=prior["passed"] and official["passed"],
            prior_use=prior,official_assets=official,
            audit_source=dict(path=str(PACKAGE/"prepare.py"),sha256=sha(PACKAGE/"prepare.py")),
            protocol_evidence=dict(path=str(args.protocol or DEFAULT_PROTOCOL),sha256=sha(args.protocol or DEFAULT_PROTOCOL)),
            candidate_evidence=dict(path=str(design),sha256=sha(design)),
            runnable_manifest_created=False,gpu_launched=False)
        atomic_json(Path(args.output)/"cpu_audit.json",value)
    else:
        if not args.output:parser.error("--output is required")
        value=prepare(args.output,args.stage,args.protocol,args.design,args.selection,args.discovery_root,args.base_manifest,
            args.extra_source,args.freeze_reviewed,args.resource_evidence)
    if args.audit_only:
        print(json.dumps(dict(passed=value["passed"],audit_path=str(Path(args.output)/"cpu_audit.json"),
            inventoried_files=prior["inventoried_files"],matching_existing_outcomes=len(prior["matching_existing_outcomes"]),
            unresolved_potential_outcomes=len(prior["unresolved_potential_outcomes"]),official_tasks=24,unique_prompts=20,
            runnable_manifest_created=False,gpu_launched=False)))
    else:
        print(json.dumps(value))


if __name__=="__main__":
    main()
