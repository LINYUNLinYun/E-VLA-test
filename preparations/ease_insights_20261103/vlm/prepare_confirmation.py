"""Freeze independent init31-35 confirmation only after valid frozen discovery selection."""
import argparse
import copy
import json
from pathlib import Path
import re
from .common import ROOT, SUITES, atomic_json, digest, read_manifest, sha
from .confirm_common import PHASE, WRAPPERS, validate_output


def verify_selection(selection, manifest, report):
    payload = copy.deepcopy(selection)
    supplied = payload.pop("selection_hash",None)
    if supplied is None or digest(payload) != supplied:
        raise ValueError("Selection hash missing or invalid")
    if (selection.get("discovery_manifest_hash") != manifest["manifest_hash"] or
        selection.get("complete_discovery_episodes") != 160 or selection.get("independent_validation_launched") is not False):
        raise ValueError("Selection is not a pre-validation decision from this complete discovery")
    choice = selection["selection"]
    if choice.get("status") != "selected" or choice.get("selected") is None:
        raise ValueError("Confirmation requires a frozen selected pair; none/incomplete is not launchable")
    if (not report.get("complete") or report.get("valid_episodes") != 160 or report.get("expected_episodes") != 160 or
        report["selection"] != choice or report["evidence"] != selection["evidence"]):
        raise ValueError("Discovery is incomplete or frozen selection/evidence changed")
    chosen = choice["selected"]
    masks = [c["id"] for c in manifest["conditions"] if c["id"] != "full_d10"]
    pair = chosen["pair"]
    if len(pair) != 2 or len(set(pair)) != 2 or any(cid not in masks for cid in pair):
        raise ValueError("Selected pair is not two fixed candidate masks")
    directions = chosen["directions"]
    if (len(directions) != 2 or len({d["suite"] for d in directions}) != 2 or
        any(d["suite"] not in manifest["suites"] or {d["better"],d["worse"]} != set(pair) for d in directions) or
        {d["better"] for d in directions} != set(pair)):
        raise ValueError("Selection does not freeze two opposite suite preferences")
    return chosen


def reject_heldout_data(roots):
    found = []
    for root in roots:
        if not Path(root).exists(): continue
        for directory in Path(root).glob("*/episodes/*/task*/init*"):
            if directory.is_dir() and re.match(r"init3[1-5](?:$|\.)",directory.name):
                if any(directory.iterdir()): found.append(str(directory.resolve()))
    if found:
        raise ValueError("Heldout data already exists before confirmation freeze: "+str(found))


def prepare(discovery_root, selection_path, output):
    discovery_root, selection_path, output = (Path(p).resolve() for p in (discovery_root,selection_path,output))
    if output == discovery_root or output in discovery_root.parents or discovery_root in output.parents:
        raise ValueError("Confirmation requires a separate output tree")
    discovery = read_manifest(discovery_root/"manifest.json",verify_sources=True)
    if discovery.get("discovery_inits") != [30] or discovery.get("total_discovery_episodes") != 160:
        raise ValueError("Unexpected discovery scope")
    selection = json.loads(selection_path.read_text())
    # Fail unfrozen/edited choices before expensive evidence reads or creating output.
    check = dict(selection); supplied = check.pop("selection_hash",None)
    if supplied is None or digest(check) != supplied or selection.get("selection",{}).get("status") != "selected":
        raise ValueError("A valid frozen selected selection.json is required")
    from .analyze import analyze
    target = output/"manifest.json"
    if target.exists():
        existing = read_manifest(target,verify_sources=True)
        validate_output(output,existing)
        if (existing["selection_evidence"]["path"] != str(selection_path) or
            existing["selection_evidence"]["selection_hash"] != supplied or
            existing["discovery_reference"]["manifest_hash"] != discovery["manifest_hash"]):
            raise ValueError("Existing confirmation is frozen to another discovery selection")
        verify_selection(selection,discovery,analyze(discovery_root))
        return dict(reused=True,path=str(target),manifest_hash=existing["manifest_hash"],episodes=600,gpu_launched=False)
    reject_heldout_data([discovery_root,output])
    if output.exists() and any(output.iterdir()):
        raise ValueError("First confirmation freeze requires an empty new output directory")
    report = analyze(discovery_root)
    chosen = verify_selection(selection,discovery,report)
    manifest = copy.deepcopy(discovery)
    for name in ("manifest_hash","discovery_inits","total_discovery_episodes","independent_validation","analysis"):
        manifest.pop(name,None)
    wrappers = [Path(__file__).parent/name for name in WRAPPERS]
    additional = wrappers+[Path(__file__).parent/name for name in ("recovery.py","worker_v2.py","runner_v2.py","analyze.py")]
    manifest["source_files"].update({str(p.relative_to(ROOT)):sha(p) for p in additional})
    manifest.update(revision="ease-vlm-independent-confirmation-v1",phase=PHASE,
        confirmation_output=str(output),confirmation_inits=list(range(31,36)),
        planned_conditions=["full_d10",*chosen["pair"]],cases_per_condition=200,total_confirmation_episodes=600,
        discovery_reference=dict(root=str(discovery_root),manifest_hash=discovery["manifest_hash"],
            manifest_sha256=sha(discovery_root/"manifest.json"),valid_episodes_at_freeze=160),
        selection_evidence=dict(path=str(selection_path),sha256=sha(selection_path),selection_hash=selection["selection_hash"],
            frozen_selection=selection["selection"],evidence=selection["evidence"]),
        preregistered_hypotheses=copy.deepcopy(chosen["directions"]),
        confirmation_wrapper_sources={str(p.relative_to(ROOT)):sha(p) for p in wrappers},
        heldout_preparation=dict(no_heldout_data_at_first_freeze=True,checked_roots=[str(discovery_root),str(output)],
            outcome_inspection_before_freeze=False,independent_initial_states=list(range(31,36))),
        analysis=dict(primary="Paired success differences in the two frozen suite directions; report discordant counts and exact paired tests",
            independent_validation_inits=list(range(31,36)),selection="Pair and directions copied from discovery selection; never reselect or relabel from validation",
            planned_conditions=["full_d10",*chosen["pair"]],
            interpretation="Separate direction reproduction from statistical resolution; other suite/task comparisons descriptive",
            multiplicity="Holm adjustment for the two preregistered one-sided exact paired tests if significance claims are made",
            size="200 episodes per planned condition; exploratory independent confirmation, not official 50-init benchmark"),
        gate_condition_scope="All four defined conditions remain available to unchanged numerical gate/cost probe; only planned_conditions launch rollouts")
    manifest["manifest_hash"] = digest(manifest)
    atomic_json(target,manifest)
    return dict(reused=False,path=str(target),manifest_hash=manifest["manifest_hash"],planned_conditions=manifest["planned_conditions"],
        episodes=600,selection_hash=selection["selection_hash"],gpu_launched=False)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--discovery-root",required=True);parser.add_argument("--selection",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.discovery_root,args.selection,args.output)))


if __name__ == "__main__": main()
