"""Immutable scheduling-only dispatch for closed frozen screen groups."""
import argparse
import json
from pathlib import Path

from . import controller,queue
from .common import ROOT,read_manifest
from .same_process_parity import sha


def sources(manifest):
    controller_sha = sha(controller.__file__)
    assert controller_sha==manifest["source_files"][str(Path(controller.__file__).relative_to(ROOT))]
    return dict(frozen_controller_sha256=controller_sha,queue_source_sha256=sha(queue.__file__),helper_sha256=sha(__file__))


def screen_evidence(root,condition,manifest):
    """Only canonical init0 fingerprints; later confirmations are outside scope."""
    assert manifest["screen_inits"]==[0]
    assert queue.completed_quality(root,condition,manifest,"screen")
    expected = {(s,t,0) for s in manifest["suites"] for t in manifest["tasks"]}
    assert len(expected)==40
    files,initials = [],{}
    for suite,task,init in sorted(expected):
        path = Path(root)/condition["id"]/"episodes"/suite/f"task{task:02d}"/f"init{init:02d}"
        result,identity = (json.loads((path/name).read_text()) for name in ("result.json","identity.json"))
        assert result["init_vector_sha256"]==identity["init_vector_sha256"]
        assert isinstance(result["success"],bool)
        initials[f"{suite}/{task}/{init}"] = result["init_vector_sha256"]
        for name in ("result.json","identity.json","complete.json"):
            target = path/name
            files.append(dict(path=str(target.absolute()),actual_path=str(target.resolve()),sha256=sha(target)))
    return dict(condition=condition["id"],initials=initials,files=files)


def plan(manifest,root):
    root = Path(root).absolute()
    assert not any(c.get("benchmark_only",False) for c in manifest["conditions"])
    definitions = {c["id"]:c for c in manifest["conditions"]}
    assert len(definitions)==len(manifest["conditions"])
    # This is the frozen canonical identity/key validator, not a row-count gate.
    complete = {cid:queue.completed_quality(root,c,manifest,"screen") for cid,c in definitions.items()}
    original = controller.promotion(manifest,root)
    evidence,fulls,closed = {},[],[]
    for c in manifest["conditions"]:
        if not c["skip"] and complete[c["id"]]:
            assert c["id"]==c["model"]+"_full" and not c["reference"]
            fulls.append(c["id"])
            evidence[c["id"]] = screen_evidence(root,c,manifest)
    for decision in original["decisions"]:
        if "best" not in decision:
            continue
        model,budget = decision["model"],decision["budget"]
        candidates = [c for c in manifest["conditions"] if c["model"]==model and len(c["skip"])==budget and not c["reference"]]
        assert len(candidates)==5,"Frozen closed budget must include ALL five nonreference masks"
        full = model+"_full"
        if full not in fulls or not all(complete[c["id"]] for c in candidates):
            continue
        for c in candidates:
            evidence[c["id"]] = screen_evidence(root,c,manifest)
            assert evidence[c["id"]]["initials"]==evidence[full]["initials"],"Screen official init keys differ from native Full"
        closed.append(dict(model=model,budget=budget,candidates=sorted(c["id"] for c in candidates),decision=decision))
    allowed = set(fulls)|{r["decision"]["best"] for r in closed if r["decision"]["confirm"]}
    selected = [cid for cid in original["selected_conditions"] if cid in allowed]
    assert set(selected)==allowed
    return dict(manifest_hash=manifest["manifest_hash"],root=str(root),global_promotion=False,
        source_kind="partial_confirmation_dispatch_closed_frozen_screen_groups",scientific_scope_unchanged=True,
        selected_conditions=selected,complete_full_conditions=fulls,closed_budgets=closed,screen_evidence=evidence,
        sources=sources(manifest),
        scope="Scheduling only: frozen controller.promotion choices for exact Full40 and all-five closed budgets. No global promotion, extra episodes, altered variants or selection formula.",
        caveat=original["caveat"])


def verify(receipt,manifest,root):
    assert receipt["global_promotion"] is False and receipt["scientific_scope_unchanged"] is True
    assert receipt["manifest_hash"]==manifest["manifest_hash"] and receipt["root"]==str(Path(root).absolute())
    assert receipt["sources"]==sources(manifest)
    current = plan(manifest,root)
    for cid,recorded in receipt["screen_evidence"].items():
        assert current["screen_evidence"].get(cid)==recorded,"Screen content/source/identity changed"
    assert set(receipt["complete_full_conditions"])<=set(current["complete_full_conditions"])
    closed = {(r["model"],r["budget"]):r for r in current["closed_budgets"]}
    assert len(receipt["closed_budgets"])==len({(r["model"],r["budget"]) for r in receipt["closed_budgets"]})
    for row in receipt["closed_budgets"]:
        assert closed.get((row["model"],row["budget"]))==row,"Frozen closed-group decision changed"
    expected_evidence = set(receipt["complete_full_conditions"])|{cid for r in receipt["closed_budgets"] for cid in r["candidates"]}
    assert set(receipt["screen_evidence"])==expected_evidence
    allowed = set(receipt["complete_full_conditions"])|{r["decision"]["best"] for r in receipt["closed_budgets"] if r["decision"]["confirm"]}
    assert len(receipt["selected_conditions"])==len(set(receipt["selected_conditions"])) and set(receipt["selected_conditions"])==allowed
    assert allowed<=set(current["selected_conditions"])
    return dict(passed=True,manifest_hash=manifest["manifest_hash"],global_promotion=False,selected_conditions=receipt["selected_conditions"],
        closed_budgets=len(receipt["closed_budgets"]),scope="Original dispatch choices and screen-only fingerprints remain valid; later data may close other budgets.")


def write_new(path,result):
    path = Path(path)
    if path.name=="promotion.json":
        raise ValueError("Partial dispatch must never be named global promotion.json")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation preserves append-only selection receipts.
    with path.open("x") as stream:
        json.dump(result,stream,indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--output")
    parser.add_argument("--verify",metavar="DISPATCH")
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest,verify_sources=True)
    if args.verify:
        if args.write:
            parser.error("--verify is read-only")
        result = verify(json.loads(Path(args.verify).read_text()),manifest,args.root)
    else:
        result = plan(manifest,args.root)
        if args.write:
            if not args.output:
                parser.error("--write requires NEW --output partial-dispatch path")
            write_new(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
