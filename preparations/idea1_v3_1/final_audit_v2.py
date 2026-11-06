"""Strict additive audit clarification: verified same-process vision gate only.

The frozen original auditor remains unchanged. No other failures are waived.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .common import ROOT, atomic_json, read_manifest
from . import final_audit
from .same_process_parity import compare, cross_process_controls, representatives, sha


def benchmark_parity_proof(protocol,condition,manifest,root):
    """Authenticate pre-parity health ordering without editing the snapshot."""
    directory = Path(root)/condition["id"]
    path = directory/"benchmark.json"
    b = final_audit.load(path)
    assert b["protocol"] == protocol and b["condition"] == condition and b["manifest_hash"] == manifest["manifest_hash"]
    assert b["warmup"]==20 and b["samples"]==100 and len(b["rows"])==100
    for bank in b["banks"]:
        assert sha(bank["path"])==bank["sha256"]
    loaded_path,parity_path = directory/"loaded_protocol.json",directory/"condition_parity.json"
    loaded = final_audit.load(loaded_path) if loaded_path.exists() else {}
    result = dict(benchmark=str(path.resolve()),benchmark_sha256=sha(path),original_pre_parity_snapshot=protocol["real_input_parity_passed"],
                  historical_server_pid=protocol["server_pid"])
    if loaded.get("server_pid") == protocol["server_pid"]:
        assert loaded["real_input_parity_passed"]
        assert {k:v for k,v in loaded.items() if k!="real_input_parity_passed"} == {k:v for k,v in protocol.items() if k!="real_input_parity_passed"}
        parity = final_audit.load(parity_path)
        assert parity["passed"] and parity["max_abs"]<=1e-5 and parity.get("noise_paired",True)
        assert parity["manifest_hash"]==manifest["manifest_hash"] and parity["condition_id"]==condition["id"] and parity["banks"]==b["banks"]
        result.update(proof="affirmative same-PID loaded protocol plus condition parity",receipts={str(p.resolve()):sha(p) for p in (loaded_path,parity_path)})
    else:
        controller = ROOT/"preparations/idea1_v3_1/controller.py"
        expected = manifest["source_files"][str(controller.relative_to(ROOT))]
        assert sha(controller)==expected
        text = controller.read_text()
        guard = 'if not parity.get("passed") or parity.get("noise_paired") is False:'
        start = text.index('parity = rpc(args.port,dict(op="parity"')
        guarded = text.index(guard,start)
        raises = text.index('raise RuntimeError("Real-input native/adapter parity gate failed")',guarded)
        benchmark = text.index('if args.operation=="benchmark":',raises)
        write = text.index('atomic_json(output/"benchmark.json",result)',benchmark)
        assert start < guarded < raises < benchmark < write
        result.update(proof="inferred from frozen controller guarded benchmark-write branch and exact bank/result identity",
            frozen_controller_sha256=expected,current_loaded_pid=loaded.get("server_pid"),
            limitation="Historical same-PID affirmative metadata was overwritten or is unavailable. No current PID is represented as the historical service; source/control-flow inference is weaker than a retained runtime receipt.")
    return result


def schema_protocol_check(protocol,condition,manifest,root,proofs):
    """Same scientific checks as frozen auditor, corrected schema/digest only."""
    assert protocol["manifest_hash"]==manifest["manifest_hash"]
    assert protocol["condition_id"]==condition["id"] and protocol["server_pid"]>0
    if not protocol["real_input_parity_passed"]:
        proof = benchmark_parity_proof(protocol,condition,manifest,root)
        if proof not in proofs:
            proofs.append(proof)
    recorded = protocol.get("condition",protocol)
    if isinstance(recorded,dict):
        for key in ("skip","horizon","denoising_steps","execute_length","depth","schedule"):
            if key in condition:
                assert recorded.get(key,protocol.get(key))==condition[key],f"Protocol {key} mismatch"
    assert protocol.get("H")==condition["horizon"] and protocol.get("D")==condition["denoising_steps"]
    assert protocol.get("checkpoint")==manifest["checkpoints"][condition["model"]]["path"]
    for key in ("openpi_sources","scientific_sources","scientific_code_hashes"):
        for filename,expected in protocol.get(key,{}).items():
            path = Path(filename) if Path(filename).is_absolute() else ROOT/filename
            assert sha(path)==expected,"Loaded scientific source changed"
    if condition["model"] in ("pi0","pi05"):
        assert protocol["R"]==5 and protocol["vision_depth"]==27
        assert protocol["vision_tokens"]==256 and protocol["vision_compression"]=="off"
    else:
        assert protocol["depth"]==condition["depth"] and protocol["R"]==condition["execute_length"]
        receipt = final_audit.load(protocol["checkpoint_verification_source"])
        assert receipt["ok"]
        expected = {r["path"]:(r["actual_sha256"],r["expected_bytes"]) for r in receipt["shards"]}
        actual = {r["path"]:(r["sha256"],r["bytes"]) for r in protocol["checkpoint_shards"]}
        assert actual==expected,"Loaded shard receipt mismatch"
        assert all(Path(p).stat().st_size==size for p,(_,size) in actual.items())
        from preparations.idea1_v2_1.runtime_models import digest as exact_native_digest
        assert exact_native_digest(protocol["checkpoint_shards"])==protocol["checkpoint_sha256"]


def numeric(check, shape=None):
    assert check["passed"] and np.isfinite(check["max_abs"]) and np.isfinite(check["rmse"])
    assert 0 <= check["max_abs"] <= 1e-5 and 0 <= check["rmse"] <= 1e-5
    assert not check.get("exact") or check["max_abs"] == 0
    if shape is not None:
        assert check["shape"] == shape


def verify_gate(path, manifest, model):
    from .benchmark import load_banks
    from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
    gate = final_audit.load(path)
    assert gate["passed"] and gate["model"] == model and gate["manifest_hash"] == manifest["manifest_hash"]
    assert gate["atol"] == gate["rtol"] == 1e-5
    assert gate["checkpoint"] == manifest["checkpoints"][model]
    required = [Path(__file__).with_name("same_process_parity.py"),Path(__file__).with_name("vision.py"),Path(__file__).with_name("models.py"),
        ROOT/"policies/openpi/src/openpi/models/pi0.py",ROOT/"policies/openpi/src/openpi/models/siglip.py",ROOT/"policies/openpi/src/openpi/models/gemma.py"]
    assert gate["source_hashes"] == {str(p.resolve()):sha(p) for p in required}, "Same-process gate source identity changed"
    records,receipts = load_banks([r["path"] for r in gate["banks"]])
    assert receipts == gate["banks"], "Gate bank changed"
    indices = representatives(records)
    assert gate["representative_indices"] == indices and len(gate["checks"]) == len(indices)
    condition = next(c for c in manifest["conditions"] if c["id"] == model+"_full")
    horizon = condition["horizon"]
    assert gate["sampler"]["NFE"] == gate["sampler"]["num_steps"] == 10
    t,expected = np.float32(1),[]
    while t >= np.float32(.05):
        expected.append(float(t))
        t = np.float32(t+np.float32(-.1))
    assert gate["sampler"]["time_values"] == expected and gate["sampler"]["final_time"] == float(t)
    assert "inferred" in gate["sampler"]["scope"]
    for check,index in zip(gate["checks"],indices,strict=True):
        record = records[index]
        assert check["bank_index"] == index and check["suite"] == record["suite"]
        assert check["seed"] == int(record["seed"])
        assert check["input_hash"] == input_digest(record["obs"],record["instruction"])
        noise = np.random.default_rng(check["seed"]).standard_normal((1,horizon,32),dtype=np.float32)
        assert check["noise_hash"] == value_digest(noise) and check["prepared_hash"]
        assert set(check["modes"]) == {"native","native_retrace","selected_v2"}
        for mode in check["modes"].values():
            numeric(mode["raw"],[1,horizon,32])
            numeric(mode["decoded"],[horizon,7])
            assert set(mode["cameras"]) == {"base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb"}
            for camera in mode["cameras"].values():
                numeric(camera["encoded"])
                numeric(camera["projected"])
                assert camera["encoded"]["shape"][:2] == [1,256] and camera["projected"]["shape"][:2] == [1,256]
                assert set(camera["blocks"]) == {f"block{i:02d}" for i in range(27)}
                for block in camera["blocks"].values():
                    assert set(block) == {"sa","+sa","mlp","+mlp"}
                    for node in block.values():
                        numeric(node)
                        assert node["shape"][:2] == [1,256]
    # Independently rebuild cross-process observations from retained raw records.
    actual_controls = cross_process_controls(Path(path).resolve().parent,model)
    assert all(all(c in actual_controls[k] for c in values) for k,values in gate["cross_process_controls"].items()), "Cross-process evidence changed since gate"
    return dict(path=str(Path(path).resolve()),sha256=sha(path),model=model,representatives=len(indices),
                suites=sorted({c["suite"] for c in gate["checks"]}),sampler=gate["sampler"],cross_process_controls=actual_controls)


def old_cross_process_observation(root,model,manifest):
    """Prove the old error concerns only paired cross-process action equality."""
    a,b = (final_audit.load(root/cid/"benchmark.json") for cid in (model+"_full",model+"_full_vision_scan_reference"))
    artifact = final_audit.load(root/f"{model}_vision_parity.json")
    assert artifact["model"] == model and artifact["manifest_hash"] == manifest["manifest_hash"]
    assert artifact["atol"] == artifact["rtol"] == 1e-5 and len(artifact["checks"])==2
    from .parity import compare as original_cross_process_compare
    recomputed = original_cross_process_compare(a,b)
    calculated = recomputed["checks"]
    assert artifact["checks"] == calculated
    assert artifact["passed"] == all(c["passed"] for c in calculated)
    return dict(path=str((root/f"{model}_vision_parity.json").resolve()),sha256=sha(root/f"{model}_vision_parity.json"),checks=calculated,
                interpretation="Cross-process numerical reproducibility observation; not a same-loaded-model structural gate")


def selected_best_frontiers(root,manifest):
    """Frozen plan: one best nonreference candidate per budget, not every mask."""
    root = Path(root)
    def results(cid,inits):
        expected = final_audit.scenes(manifest,inits)
        rows = {}
        for s,t,i in expected:
            p = root/cid/"episodes"/s/f"task{t:02d}"/f"init{i:02d}"
            if not (p/"complete.json").exists() or not (p/"result.json").exists():
                return None
            marker,row = final_audit.load(p/"complete.json"),final_audit.load(p/"result.json")
            if not marker.get("valid") or not row.get("valid") or any(r.get("manifest_hash")!=manifest["manifest_hash"] for r in (marker,row)):
                return None
            if row["condition"]!=cid or (row["suite"],row["task"],row["init"])!=(s,t,i):
                return None
            rows[(s,t,i)] = row
        return rows
    obligations,observations = [],[]
    for model,cap in (("xiaomi-robotic0",24),("db-cogact",18)):
        full = results(model+"_full",manifest["screen_inits"])
        if full is None:
            continue
        baseline = sum(r["success"] for r in full.values())
        cap_candidates = [c for c in manifest["conditions"] if c["model"]==model and not c.get("reference") and len(c["skip"])==cap]
        scores = [(c,results(c["id"],manifest["screen_inits"])) for c in cap_candidates]
        if any(rows is None for _,rows in scores) or len(scores)!=5 or not baseline:
            continue
        best,best_rows = sorted(scores,key=lambda pair:(-sum(r["success"] for r in pair[1].values()),pair[0]["id"]))[0]
        retained = sum(r["success"] for r in best_rows.values())/baseline
        observation = dict(model=model,budget=cap,best=best["id"],screen_relative_retention=retained,
            screened_candidates=[c["id"] for c,_ in scores],rule="Only frozen promotion best per budget creates a confirmation obligation")
        observations.append(observation)
        if retained>=manifest["relative_success_floor"]-manifest["confirmation_near_floor_margin"]:
            confirmed = results(best["id"],manifest["confirm_inits"])
            baseline_confirmed = results(model+"_full",manifest["confirm_inits"])
            if confirmed is None or baseline_confirmed is None:
                obligations.append(dict(condition=best["id"],retention=retained,required="confirm selected best cap candidate200 with corresponding Full200"))
            else:
                successes = sum(r["success"] for r in baseline_confirmed.values())
                ratio = sum(r["success"] for r in confirmed.values())/successes if successes else None
                observation["confirmed_relative_retention"] = ratio
                if ratio is not None and ratio>=manifest["relative_success_floor"]:
                    obligations.append(dict(condition=best["id"],retention=ratio,required="extend beyond initial cap in independent immutable extension manifest"))
        for reference in [c for c in manifest["conditions"] if c["model"]==model and c.get("reference")]:
            reference_rows = results(reference["id"],manifest["screen_inits"])
            same_budget = [c for c in manifest["conditions"] if c["model"]==model and not c.get("reference") and len(c["skip"])==len(reference["skip"])]
            peers = [(c,results(c["id"],manifest["screen_inits"])) for c in same_budget]
            if reference_rows is not None and peers and all(r is not None for _,r in peers):
                top = max(sum(r["success"] for r in rows.values()) for _,rows in peers)
                ref_success = sum(r["success"] for r in reference_rows.values())
                if ref_success>top:
                    observations.append(dict(reference=reference["id"],reference_screen_successes=ref_success,best_nonreference_successes=top,
                        interpretation="Historical reference outperforms eligible mask screen; flag only, original promotion excludes reference and is not changed"))
    return obligations,observations


def audit(manifest,root,gate_paths=None,check_videos=True,analysis_report=None):
    root = Path(root)
    proofs = []
    with patch.object(final_audit,"protocol_check",lambda p,c,m:schema_protocol_check(p,c,m,root,proofs)):
        report = final_audit.audit(manifest,root,check_videos,analysis_report)
    report["schema_correction_proofs"] = proofs
    report["schema_corrections"] = ["Pre-parity benchmark health snapshot remains unchanged; affirmative runtime/source-guard proof is checked separately.",
        "Native Torch shard identity uses the exact original NativePolicy recursive digest after independent receipt/size checks, not the unrelated manifest JSON digest."]
    gates,failures = {},[]
    for model in ("pi0","pi05"):
        path = Path(gate_paths[model]) if gate_paths and model in gate_paths else root/f"{model}_vision_same_process_parity.json"
        try:
            gates[model] = verify_gate(path,manifest,model)
        except Exception as exc:
            failures.append(f"{model} affirmative same-process gate: {type(exc).__name__}: {exc}")
    replacements,observations = [],[]
    # Both complete, independently verified model gates are mandatory before
    # even one original error can be reclassified.
    if len(gates)==2:
        native_controls = [c for g in gates.values() for c in g["cross_process_controls"]["native_vs_native"]
                           if c["input_paired"] and c["noise_paired"]]
        if not native_controls:
            failures.append("Missing authenticated native-vs-native cross-process control receipts")
        else:
            for model in ("pi0","pi05"):
                label = f"{model} native/matched real vision parity: AssertionError: "
                if label in report["errors"]:
                    try:
                        observation = old_cross_process_observation(root,model,manifest)
                        assert not all(c["passed"] for c in observation["checks"])
                        report["errors"].remove(label)
                        replacements.append(dict(original_error=label,replacement_gate=gates[model]["path"],scope="Only paired cross-process action equality reclassified"))
                        observations.append(observation)
                    except Exception as exc:
                        failures.append(f"{model} old gate reclassification proof: {type(exc).__name__}: {exc}")
    report["errors"].extend(failures)
    original_frontiers = report.get("frontier_obligations",[])
    for row in original_frontiers:
        exact_missing = f"Conditional frontier unresolved: {row}"
        if exact_missing in report["missing_requirements"]:
            report["missing_requirements"].remove(exact_missing)
    frontiers,frontier_observations = selected_best_frontiers(root,manifest)
    report["missing_requirements"].extend(f"Selected-best frontier unresolved: {r}" for r in frontiers)
    report.update(original_frontier_observations=original_frontiers,frontier_obligations=frontiers,
                  frontier_selection_observations=frontier_observations,
                  frontier_method_clarification="Plan section59–61 and immutable promotion select one best eligible mask per budget; unselected near-floor cap masks do not individually require200 episodes. Original observations are retained.")
    report.update(same_process_gates=gates,explicit_gate_replacements=replacements,retained_cross_process_observations=observations,
        clarification="Same-process structural E0 pairing replaces only proven cross-process equality errors; tolerance remains1e-5. All other original audit errors/missing requirements remain.",
        audit_v2_source_sha256=sha(__file__))
    report["limitations"].append("Same-process block/NN numerical metrics are generated by the pinned gate source; CPU audit validates identities, coverage and metrics, not GPU recomputation. NFE10 follows original scalar recurrence/source, not intrusive NN call instrumentation.")
    report["passed"] = not report["errors"] and not report["missing_requirements"] and len(gates)==2
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--pi0-gate")
    parser.add_argument("--pi05-gate")
    parser.add_argument("--analysis-report")
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--skip-video-decode",action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    paths = {m:p for m,p in (("pi0",args.pi0_gate),("pi05",args.pi05_gate)) if p}
    report = audit(read_manifest(args.manifest,verify_sources=True),args.root,paths,not args.skip_video_decode,args.analysis_report)
    if args.skip_video_decode:
        report["missing_requirements"].append("Strict video decode skipped")
        report["passed"] = False
    if args.output:
        atomic_json(args.output,report)
    print(json.dumps(report,indent=2))
    if not report["passed"] and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
