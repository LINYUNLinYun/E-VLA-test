"""Fresh same-card benchmark cohort around the unchanged frozen controller.

No quality rollouts. Dry runs never query a GPU or launch children. Every phase
retains its own benchmark, affirmative same-PID health and parity evidence.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .common import ROOT,atomic_json,read_manifest
from .same_process_parity import sha


TIMERS = ("request_roundtrip_ms","server_request_ms","vision_ms")


def expected_inputs(banks):
    from .benchmark import load_banks
    from preparations.idea1_v2_1.runtime_client import input_digest
    records,receipts = load_banks([str(Path(p).resolve()) for p in banks])
    if len(records)<100:
        raise ValueError("Cohorts require100 distinct actual observations")
    return [dict(index=i,input_hash=input_digest(r["obs"],r["instruction"]),seed=int(r.get("seed",i+20)))
            for i,r in enumerate(records[:100])],receipts


def authenticate_vision_gate(plan,manifest):
    if plan["model"] not in ("pi0","pi05"):
        return None
    from .final_audit_v2 import verify_gate
    gate = plan["vision_parity"]
    assert sha(gate["path"])==gate["sha256"]
    proof = verify_gate(gate["path"],manifest,plan["model"])
    raw = json.loads(Path(gate["path"]).read_text())
    normalize = lambda rows:{str(Path(r["path"]).resolve()):r["sha256"] for r in rows}
    assert normalize(raw["banks"])==normalize(plan["banks"]),"Cohort and original vision E0 banks differ"
    return proof


def make_plan(manifest,manifest_path,root,model,candidates,banks,gpu=None,port_start=11000,require_four_budgets=False,launch=False,vision_parity=None):
    root = Path(root).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError("Cohort outputs are immutable; choose a new directory")
    if not candidates or len(candidates)!=len(set(candidates)):
        raise ValueError("Explicit unique candidate CIDs required; no automatic quality selection")
    conditions = {c["id"]:c for c in manifest["conditions"]+manifest["benchmark_references"]}
    for cid in candidates:
        if cid not in conditions or conditions[cid]["model"]!=model or not conditions[cid]["skip"]:
            raise ValueError(f"Candidate is not a pruned condition of {model}: {cid}")
    if require_four_budgets:
        if model not in ("pi0","pi05") or not {3,6,9,12}<={len(conditions[c]["skip"]) for c in candidates}:
            raise ValueError("Final vision cohort requires explicit coverage of skip3/6/9/12")
    expected,receipts = expected_inputs(banks)
    if model in ("pi0","pi05") and not vision_parity:
        raise ValueError("Pi cohorts require explicit ORIGINAL --vision-parity same-process artifact")
    phases = [("native_full_before",model+"_full")]
    if model in ("pi0","pi05"):
        phases.append(("matched_all27_full",model+"_full_vision_scan_reference"))
    phases += [("candidate",cid) for cid in candidates]
    phases.append(("native_full_after",model+"_full"))
    jobs = []
    for i,(role,cid) in enumerate(phases):
        port = port_start+i
        if not 1<=port<=65535:
            raise ValueError("Cohort TCP port range invalid")
        phase_root = root/f"phase_{i:02d}"
        command = [sys.executable,"-m","preparations.idea1_v3_1.controller","benchmark",
            "--manifest",str(Path(manifest_path).resolve()),"--output",str(phase_root),"--condition",cid,
            "--port",str(port),"--bank",*[str(Path(p).resolve()) for p in banks]]
        if gpu is not None:
            command += ["--gpu",str(gpu)]
        if vision_parity:
            command += ["--vision-parity",str(Path(vision_parity).resolve())]
        if launch:
            command += ["--launch"]
        jobs.append(dict(index=i,role=role,condition=cid,phase_root=str(phase_root),port=port,command=command,
            status="pending",controller_pid=None,exit_code=None))
    plan = dict(manifest_hash=manifest["manifest_hash"],manifest_path=str(Path(manifest_path).resolve()),model=model,
        source_sha256=sha(__file__),frozen_source_hashes=manifest["source_files"],checkpoint=manifest["checkpoints"][model],
        root=str(root),gpu=str(gpu) if gpu is not None else None,banks=receipts,expected_inputs=expected,
        candidates=candidates,require_four_budgets=require_four_budgets,phases=jobs,status="planned",dry_run=not launch,
        vision_parity=dict(path=str(Path(vision_parity).resolve()),sha256=sha(vision_parity)) if vision_parity else None,
        selection_scope="Candidate IDs are explicitly supplied by parent; this timing cohort does not select or certify quality candidates.",
        timing_scope="Standalone all-camera vision and whole completed RPC are measured independently by frozen benchmark; never add/subtract them. Fresh native and matched references bracket/report candidate P50 ratios; legacy comparisons remain legacy.")
    if launch:
        plan["vision_gate_verification"] = authenticate_vision_gate(plan,manifest)
    return plan


def gpu_identity(gpu):
    # Only called in launched execution, and queries only the assigned card.
    line = subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid,name,pci.bus_id","--format=csv,noheader"],text=True).strip()
    uuid,name,pci = [s.strip() for s in line.split(",",2)]
    if not uuid.startswith("GPU-"):
        raise ValueError("Assigned card UUID unavailable")
    return dict(uuid=uuid,name=name,pci_bus_id=pci,assigned_gpu=str(gpu))


def execute(plan,popen=subprocess.Popen,query=gpu_identity):
    if plan["dry_run"] or plan["gpu"] is None:
        raise ValueError("Execution requires explicit --launch and assigned --gpu")
    root = Path(plan["root"])
    root.mkdir(parents=True,exist_ok=False)
    path = root/"cohort.json"
    plan.update(status="running",queue_pid=os.getpid(),created_host_s=time.time(),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    atomic_json(path,plan)
    try:
        plan["gpu_identity"] = query(plan["gpu"])
        atomic_json(path,plan)
        for phase in plan["phases"]:
            phase["gpu_before"] = query(plan["gpu"])
            assert phase["gpu_before"]["uuid"]==plan["gpu_identity"]["uuid"]
            with (root/f"phase_{phase['index']:02d}.log").open("x") as log:
                begin = time.perf_counter()
                child = popen(phase["command"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                phase.update(controller_pid=child.pid,status="running",started_host_s=time.time())
                atomic_json(path,plan)
                code = child.wait()
                phase.update(exit_code=code,wall_s=time.perf_counter()-begin,finished_host_s=time.time(),status="completed" if code==0 else "failed")
            if code:
                plan["status"] = "stopped_on_child_failure"
                atomic_json(path,plan)
                return code
            phase["gpu_after"] = query(plan["gpu"])
            assert phase["gpu_after"]["uuid"]==plan["gpu_identity"]["uuid"]
            directory = Path(phase["phase_root"])/phase["condition"]
            phase["artifact_hashes"] = {name:sha(directory/name) for name in ("benchmark.json","loaded_protocol.json","condition_parity.json")}
            phase["server_pid"] = json.loads((directory/"loaded_protocol.json").read_text())["server_pid"]
            atomic_json(path,plan)
            print(json.dumps(dict(phase=phase["index"],role=phase["role"],condition=phase["condition"],exit_code=code)),flush=True)
        plan.update(status="completed",finished_host_s=time.time())
        atomic_json(path,plan)
        return 0
    except BaseException as exc:
        plan.update(status="interrupted_or_failed",error=repr(exc))
        atomic_json(path,plan)
        # No other process is signalled. Frozen controller owns service cleanup;
        # any interrupted own controller PID remains in the receipt for review.
        raise


def verify_phase(root,phase,manifest,condition,expected,banks):
    from .final_audit_v2 import schema_protocol_check,benchmark_parity_proof
    directory = Path(phase["phase_root"])/condition["id"]
    assert directory.parent==Path(root)/f"phase_{phase['index']:02d}" and not directory.is_symlink() and not directory.parent.is_symlink()
    artifacts = {name:sha(directory/name) for name in ("benchmark.json","loaded_protocol.json","condition_parity.json")}
    assert artifacts==phase["artifact_hashes"]
    b = json.loads((directory/"benchmark.json").read_text())
    loaded = json.loads((directory/"loaded_protocol.json").read_text())
    assert b["manifest_hash"]==manifest["manifest_hash"] and b["condition"]==condition and b["banks"]==banks
    assert b["warmup"]==20 and b["samples"]==100 and len(b["rows"])==100
    assert b["protocol"]["server_pid"]==loaded["server_pid"]==phase["server_pid"] and loaded["real_input_parity_passed"]
    proofs = []
    schema_protocol_check(b["protocol"],condition,manifest,directory.parent,proofs)
    if condition["model"] in ("pi0","pi05"):
        files = [ROOT/f"policies/openpi/src/openpi/models/{name}.py" for name in ("pi0","siglip","gemma")]
        assert b["protocol"]["openpi_sources"]=={str(p.relative_to(ROOT)):sha(p) for p in files}
    else:
        from preparations.idea1_v2_2.provenance import collect_scientific_sources
        assert b["protocol"]["scientific_code_hashes"]==collect_scientific_sources(condition["model"])
    affirmative = benchmark_parity_proof(b["protocol"],condition,manifest,directory.parent)
    assert affirmative["proof"]=="affirmative same-PID loaded protocol plus condition parity","Weaker historical inference is forbidden for fresh cohort"
    assert len({r["input_hash"] for r in b["rows"]})==100
    for row,wanted in zip(b["rows"],expected,strict=True):
        assert all(row[k]==wanted[k] for k in ("index","input_hash","seed"))
        assert isinstance(row["noise_hash"],str) and len(row["noise_hash"])==64
        assert all(np.isfinite(row[k]) and row[k]>0 for k in TIMERS)
        if row.get("parity_actions") is not None:
            actions = np.asarray(row["parity_actions"])
            assert actions.shape==(condition["horizon"],7) and np.isfinite(actions).all()
        assert (row.get("parity_actions") is not None)==(row["index"]<2)
        if condition["model"] in ("pi0","pi05"):
            vision = row["vision_details"]
            assert vision["cameras"]==len(vision["shapes"])==3
            assert all(shape[:2]==[1,256] for shape in vision["shapes"]) and vision["total_tokens"]==768
    recomputed = {key:dict(zip(("p50","p95","p99"),np.percentile([r[key] for r in b["rows"]],[50,95,99]).tolist())) for key in TIMERS}
    assert b["summary"]==recomputed
    return dict(condition=condition["id"],summary=recomputed,rows=b["rows"],affirmative_same_pid_proof=affirmative,
        original_benchmark_parity_snapshot=b["protocol"]["real_input_parity_passed"],artifacts=artifacts,benchmark_path=str((directory/"benchmark.json").resolve()))


def comparison(phases):
    before,after = phases[0]["summary"],phases[-1]["summary"]
    drift = {k:abs(after[k]["p50"]-before[k]["p50"])/before[k]["p50"] for k in TIMERS}
    matched = next((p["summary"] for p in phases if p["role"]=="matched_all27_full"),None)
    ratios = [dict(condition=p["condition"],native_before_p50_speedup={k:before[k]["p50"]/p["summary"][k]["p50"] for k in TIMERS},
        matched_full_p50_speedup={k:matched[k]["p50"]/p["summary"][k]["p50"] for k in TIMERS} if matched else None)
        for p in phases if p["role"]=="candidate"]
    flags = {k:v>.10 for k,v in drift.items()}
    return dict(native_full_after_p50_drift_fraction=drift,drift_exceeds_threshold_by_metric=flags,
        primary_drift_metrics=["request_roundtrip_ms","vision_ms"],unstable=any(flags[k] for k in ("request_roundtrip_ms","vision_ms")),drift_threshold=.10,
        candidates=ratios,primary_statistic="P50",interpretation="Standalone vision and whole completed RPC are independent measurements; no additive/subtractive attribution. Drift>10% flags unstable, with no automatic retries or cherry-picking.")


def audit(root,manifest,require_four_budgets=False):
    root = Path(root).resolve()
    plan = json.loads((root/"cohort.json").read_text())
    assert plan["status"]=="completed" and not plan["dry_run"] and plan["root"]==str(root)
    assert plan["manifest_hash"]==manifest["manifest_hash"] and plan["source_sha256"]==sha(__file__)
    assert plan["frozen_source_hashes"]==manifest["source_files"] and plan["checkpoint"]==manifest["checkpoints"][plan["model"]]
    assert all(sha(ROOT/name)==expected for name,expected in manifest["source_files"].items())
    assert all(sha(path)==expected for path,expected in manifest["checkpoints"][plan["model"]]["identity_artifacts"].items())
    expected,banks = expected_inputs([r["path"] for r in plan["banks"]])
    assert expected==plan["expected_inputs"] and banks==plan["banks"]
    definitions = {c["id"]:c for c in manifest["conditions"]+manifest["benchmark_references"]}
    ids = plan["candidates"]
    assert ids and len(set(ids))==len(ids) and all(definitions[c]["model"]==plan["model"] and definitions[c]["skip"] for c in ids)
    observed_budgets = sorted({len(definitions[c]["skip"]) for c in ids})
    complete_vision_budgets = {3,6,9,12}<=set(observed_budgets)
    if plan["require_four_budgets"] or require_four_budgets:
        assert plan["model"] in ("pi0","pi05") and {3,6,9,12}<={len(definitions[c]["skip"]) for c in ids}
    vision_proof = authenticate_vision_gate(plan,manifest)
    sequence = [("native_full_before",plan["model"]+"_full")]
    if plan["model"] in ("pi0","pi05"):
        sequence.append(("matched_all27_full",plan["model"]+"_full_vision_scan_reference"))
    sequence += [("candidate",cid) for cid in ids]+[("native_full_after",plan["model"]+"_full")]
    assert [(p["role"],p["condition"]) for p in plan["phases"]]==sequence
    assert [p["index"] for p in plan["phases"]]==list(range(len(sequence))) and len({p["port"] for p in plan["phases"]})==len(sequence)
    assert [p["port"] for p in plan["phases"]]==list(range(plan["phases"][0]["port"],plan["phases"][0]["port"]+len(sequence)))
    assert plan["gpu_identity"]["uuid"].startswith("GPU-")
    verified = []
    for phase in plan["phases"]:
        assert phase["status"]=="completed" and phase["exit_code"]==0 and phase["controller_pid"]>0
        assert np.isfinite(phase["wall_s"]) and phase["wall_s"]>0 and phase["server_pid"]>0 and phase["server_pid"]!=phase["controller_pid"]
        assert phase["gpu_before"]["uuid"]==phase["gpu_after"]["uuid"]==plan["gpu_identity"]["uuid"]
        assert phase["gpu_before"]["assigned_gpu"]==phase["gpu_after"]["assigned_gpu"]==plan["gpu"]
        command = [phase["command"][0],"-m","preparations.idea1_v3_1.controller","benchmark","--manifest",plan["manifest_path"],
            "--output",phase["phase_root"],"--condition",phase["condition"],"--port",str(phase["port"]),"--bank",*[r["path"] for r in banks],"--gpu",plan["gpu"]]
        if plan["vision_parity"]:
            command += ["--vision-parity",plan["vision_parity"]["path"]]
        assert phase["command"]==command+["--launch"]
        result = verify_phase(root,phase,manifest,definitions[phase["condition"]],expected,banks)
        verified.append(dict(result,role=phase["role"],index=phase["index"]))
    baseline = verified[0]["rows"]
    assert all(all(a[k]==b[k] for k in ("index","input_hash","noise_hash","seed")) for p in verified[1:] for a,b in zip(baseline,p["rows"],strict=True))
    summary = comparison(verified)
    return dict(passed=True,stable_for_primary_comparison=not summary["unstable"],manifest_hash=manifest["manifest_hash"],
        cohort_path=str(root),cohort_receipt_sha256=sha(root/"cohort.json"),source_sha256=sha(__file__),gpu_identity=plan["gpu_identity"],
        banks=banks,phases=[{k:v for k,v in p.items() if k!="rows"} for p in verified],comparison=summary,
        vision_gate_verification=vision_proof,
        vision_budget_coverage=dict(observed=observed_budgets,required=[3,6,9,12],complete=complete_vision_budgets) if vision_proof else None,
        final_vision_cohort=bool(vision_proof and (plan["require_four_budgets"] or require_four_budgets) and complete_vision_budgets),
        passed_definition="Data/provenance integrity, not a quality gate or stability guarantee; unstable timing remains a valid flagged result.",
        candidate_actions_comparison="Not compared across pruning conditions; only input/noise/index/seed pairing is required.",
        limitations=["20warmup is authenticated by frozen controller/benchmark code and header; individual warmup outputs are not persisted.",
            "No automatic candidate-quality selection, retry, drift correction, or legacy benchmark overwrite."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("run","audit"))
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--model",choices=("pi0","pi05","xiaomi-robotic0","db-cogact"))
    parser.add_argument("--candidate",nargs="+")
    parser.add_argument("--bank",nargs="+")
    parser.add_argument("--gpu")
    parser.add_argument("--port-start",type=int,default=11000)
    parser.add_argument("--require-four-budgets",action="store_true")
    parser.add_argument("--vision-parity",help="Explicit ORIGINAL same-process E0 artifact; mandatory for pi models")
    parser.add_argument("--launch",action="store_true")
    parser.add_argument("--output",help="Explicit new CPU audit receipt; never overwrites")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest,verify_sources=True)
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.operation=="audit":
        result = audit(args.root,manifest,args.require_four_budgets)
        if args.output:
            atomic_json(args.output,result)
        print(json.dumps(result,indent=2))
        return
    if not args.model or not args.candidate or not args.bank:
        parser.error("run requires explicit --model, --candidate and real --bank")
    if args.launch and args.gpu is None:
        parser.error("--launch requires assigned --gpu")
    plan = make_plan(manifest,args.manifest,args.root,args.model,args.candidate,args.bank,args.gpu,args.port_start,args.require_four_budgets,args.launch,args.vision_parity)
    if not args.launch:
        print(json.dumps(plan,indent=2))
        return
    code = execute(plan)
    if code:
        raise SystemExit(code)
    result = audit(args.root,manifest)
    atomic_json(Path(args.root)/"audit.json",result)
    print(json.dumps(dict(passed=result["passed"],stable_for_primary_comparison=result["stable_for_primary_comparison"],comparison=result["comparison"]),indent=2))


if __name__ == "__main__":
    main()
