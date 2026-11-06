"""CPU-only read-only audit of raw idea2 records, with optional explicit receipt.

Markers are necessary but never sufficient: inputs, action arrays, telemetry,
cache lineage and videos are checked directly. No model imports or GPU calls.
"""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np


SUITES = ("libero_spatial","libero_object","libero_goal","libero_10")
CONDITIONS = [("fresh_serial",1)]+[(mode,interval) for mode in ("serial_reuse","async_reuse") for interval in (1,2,4)]
WORKSPACE = Path(__file__).resolve().parents[2]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def require(condition,message):
    if not condition:
        raise ValueError(message)


def finite_array(value,shape,label):
    array = np.asarray(value)
    require(array.shape==shape and np.isfinite(array).all(),f"Invalid {label} shape or nonfinite values: {array.shape}")
    return array


def artifact(path):
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE/value


def video(path,frames):
    process = subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(path),"-f","null","-"],
                             check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    require(not process.stderr.strip(),"Video decoder reported errors")
    probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                                                "-show_entries","stream=nb_read_frames,avg_frame_rate,r_frame_rate","-of","json",str(path)],text=True))
    stream = probe["streams"][0]
    require(int(stream["nb_read_frames"])==frames,"Decoded video frame count differs from actual telemetry")
    require(Fraction(stream["avg_frame_rate"])==20,"Video FPS differs from native20Hz")
    return dict(decoded_frames=int(stream["nb_read_frames"]),fps=float(Fraction(stream["avg_frame_rate"])),sha256=sha(path))


def audit_episode(directory,key):
    from preparations.idea1_v2_1.runtime_client import input_digest,model_seed
    mode,interval,suite,task,init = key
    result,complete,identity,protocol = [load(directory/name) for name in ("result.json","complete.json","identity.json","native_protocol.json")]
    require(result["valid"] and complete["valid"],"Invalid episode marker")
    require((result["mode"],result["interval"],result["suite"],result["task"],result["init"])==key,"Result key mismatch")
    require(complete["all_background_work_drained"],"Background work was not drained")
    require(protocol["H"]==50 and protocol["R"]==5 and protocol["num_steps"]==10,"Original H/R/D changed")
    require(protocol["execution"]=="eager" and protocol["attention"]=="eager" and protocol["cache_slots"]==2,"Scientific execution configuration changed")
    require(protocol["mode"]==mode and protocol["interval"]==interval,"Native protocol mode mismatch")
    require(identity["suite"]==suite and identity["task"]==task and identity["init"]==init and identity["env_seed"]==7,"Initial-state identity mismatch")
    require(sha(identity["init_file"])==identity["init_file_sha256"],"Official initial-state file changed")
    infers,telemetry = lines(directory/"infer.jsonl"),lines(directory/"telemetry.jsonl")
    require(len(infers)==result["predictions"],"Prediction count mismatch")
    require(len(telemetry)==result["actions"]+1==result["frames"],"Telemetry/action/frame count mismatch")
    require(sum(row["executed_length"] for row in infers)==result["actions"],"Executed-length sum mismatch")
    require(np.isfinite(result["all_work_wall_s"]) and result["all_work_wall_s"]>=result["wall_s"],"End-to-end accounting excludes pending work")
    for index,state in enumerate(telemetry):
        require(state["step"]==index,"Nonconsecutive actual action indices")
        finite_array(state["eef_pos"],(3,),"true EEF XYZ")
        require(np.isfinite(state["sim_time_s"]) and np.isfinite(state["wall_timestamp_s"]),"Nonfinite execution timestamp")
        if index:
            finite_array(state["action"],(7,),"actual executed command")
    epochs,snapshots = set(),{}
    executed = 0
    for request,row in enumerate(infers):
        require(row["request"]==request==row["replan"]==row["current_observation_id"],"Reset/request index mismatch")
        require(row["step"]==executed,"Request action-index mismatch")
        require(row["mode"]==mode and row["interval"]==interval,"Request mode mismatch")
        require(row["generated_length"]==50 and row["submitted_length"]==5 and row["execute_length"]==5,"Generated/submitted horizon mismatch")
        n = row["executed_length"]
        require(1<=n<=5,"Invalid actual controller consumption length")
        actions = finite_array(row["chunk_actions"],(50,7),"generated chunk").astype(np.float32)
        with np.load(artifact(row["prediction_path"]),allow_pickle=False) as saved:
            require(np.array_equal(actions,saved["actions"]),"Saved stable action chunk differs from logged output")
        require(sha(artifact(row["prediction_path"]))==row["prediction_sha256"],"Prediction artifact SHA mismatch")
        require(sha(artifact(row["input_path"]))==row["input_file_sha256"],"Observation artifact SHA mismatch")
        with np.load(artifact(row["input_path"]),allow_pickle=False) as saved:
            observation = {name:saved[name] for name in saved.files if name not in ("instruction","seed")}
            instruction = str(saved["instruction"].item())
            require(input_digest(observation,instruction)==row["input_hash"],"Actual saved model observation differs from request hash")
            require(int(saved["seed"].item())==row["seed"]==model_seed(suite,task,init,request),"Paired request noise seed rule changed")
        for offset,action in enumerate(actions[:n],start=1):
            require(np.array_equal(action,np.asarray(telemetry[executed+offset]["action"],np.float32)),"Actual execution differs from selected stable chunk")
        executed += n
        source = row["prefix_observation_id"]
        require(0<=source<=request and row["observation_request_age"]==request-source,"Future/cross-reset prefix source")
        require(row["suffix_state_id"]==request==row["reference_state_id"],"State/reference source mismatch")
        if request==0:
            require(source==0,"Episode did not bootstrap its own prefix")
        snapshot = row["snapshot"]
        epochs.add(snapshot["epoch"])
        require(snapshot["slot"] in (0,1) and snapshot["generation"]>0,"Invalid two-slot identity")
        identity_key = (snapshot["epoch"],snapshot["slot"],snapshot["generation"])
        require(identity_key not in snapshots or snapshots[identity_key]==source,"A snapshot identity changed source data")
        snapshots[identity_key] = source
        require(1<=row["pool"]["live_snapshots"]<=2,"Two-slot live snapshot bound exceeded")
        require(row["camera_capture_age_ms"] is None,"Unavailable camera capture age was fabricated")
        origin = row["source_host_receipt_monotonic_s"]
        ready,published,start = [row[name] for name in ("prefix_ready_observed_monotonic_s","prefix_published_monotonic_s","generation_start_monotonic_s")]
        require(origin<=ready<=published<=start,"Published before ready or negative source receipt age")
        ages = [row[name] for name in ("source_receipt_age_at_generation_ms","source_receipt_age_at_internal_output_ms","source_receipt_age_at_response_ms")]
        require(all(np.isfinite(value) and value>=0 for value in ages) and ages==sorted(ages),"Invalid source receipt age ordering")
        require(abs(ages[0]-(start-origin)*1000)<1e-3,"Generation receipt age does not match its clock")
        for name in ("execution_source_receipt_age_ms_start","execution_source_receipt_age_ms_end"):
            require(np.isfinite(row[name]) and row[name]>=0,"Invalid execution receipt age")
        require(row["execution_source_receipt_age_ms_end"]>=row["execution_source_receipt_age_ms_start"],"Execution age decreased within chunk")
        require(all(np.isfinite(row["timing"][name]) and row["timing"][name]>=0 for name in ("action_ready_host_ms","response_host_ms","ae_gpu_ms")),"Invalid inference timing")
    require(len(epochs)==1,"Episode contains old epoch completion results")
    require(executed==result["actions"],"Actual execution count mismatch")
    proof = video(directory/"video.mp4",len(telemetry))
    return dict(key=list(key),valid=True,success=result["success"],requests=len(infers),executed_actions=executed,
                epoch=next(iter(epochs)),init_vector_hash=identity["init_vector_hash"],video=proof,
                infer_sha256=sha(directory/"infer.jsonl"),telemetry_sha256=sha(directory/"telemetry.jsonl"))


def audit_micro(root,require_graph=False):
    details = {}
    mapping = load(root/"converted_mapping_audit.json")
    require(mapping["passed"] and not mapping["missing_trained_keys"] and not mapping["shape_mismatches"] and not mapping["unexpected_keys"],"Converted trained mapping audit failed")
    require(mapping["expected_trained_tensors"]==776,"Unexpected trained tensor mapping count")
    details["mapping"] = dict(passed=True,expected_trained_tensors=776,receipt_sha256=sha(root/"converted_mapping_audit.json"))
    for suite in SUITES:
        directory = root/"micro"/suite
        e0 = load(directory/"e0.json")
        require(e0["passed"] and e0["kv_unchanged"] and e0["prefix_masks_equal"] and e0["positions_equal"] and e0["original_denoise_calls"]==10,f"{suite} E0 failed")
        e2 = load(directory/"e2.json")
        require(len(e2["rows"])==24,f"{suite} E2 incomplete")
        for interval in (1,2,4):
            rows = [row for row in e2["rows"] if row["interval"]==interval]
            require(sorted(row["request"] for row in rows)==list(range(8)),f"{suite} E2 request coverage mismatch")
            for row in rows:
                require(row["serial_async_error"]["max_abs"]<=1e-5,f"{suite} E2 fixed-version mismatch")
                require(row["prefix_source"]==row["request"]//interval*interval,"E2 fixed version policy changed")
                require(set(row["by_horizon"])=={"first5","full50"},"E2 component horizon summaries missing")
        details[suite] = dict(e0=True,e2=True,e0_sha256=sha(directory/"e0.json"),e2_sha256=sha(directory/"e2.json"))
    e1 = load(root/"micro/libero_spatial/e1.json")
    require(Counter(row["mode"] for row in e1["rows"])=={"serial":100,"two_stream":100},"E1 sample count mismatch")
    for mode in ("serial","two_stream"):
        require(sorted(row["iteration"] for row in e1["rows"] if row["mode"]==mode)==list(range(100)),"E1 duplicated/missing iterations")
    require(all(row["output_error"]["max_abs"]<=1e-5 for row in e1["rows"]),"E1 same-KV paired mismatch")
    details["e1"] = dict(passed=True,samples_per_arm=100,trace_summary=e1.get("trace_summary"),
                        receipt_sha256=sha(root/"micro/libero_spatial/e1.json"))
    graphs = [path for path in root.rglob("e1b.json") if "attempts" not in path.parts]
    require(not require_graph or bool(graphs),"Committed E1b graph experiment is still pending")
    details["e1b"] = dict(status="pending_optional" if not graphs else "present",artifacts=[])
    for path in graphs:
        result = load(path)
        require(all(result["audit"].get(name) for name in ("stable_output","distinct_output_addresses","old_kv_content_unchanged","captured_prefill_equals_eager","graph_pools_distinct")),"E1b graph ownership/parity audit failed")
        require(Counter(row["mode"] for row in result["rows"])=={"serial":100,"two_stream":100},"E1b paired sample count mismatch")
        require(all(row["paired_output_error"]["max_abs"]<=1e-5 for row in result["rows"]),"E1b same-input serial/async mismatch")
        uuid = result["identity"].get("gpu_uuid")
        require(uuid and uuid!="None","E1b physical GPU UUID is missing")
        require(not require_graph or result.get("args",{}).get("eager_reference"),"Committed same-GPU eager control arms were omitted")
        require(result["audit"]["eager_error"]["max_abs"]<=1e-5,"E1b graph differs from the matched eager wrapper")
        for name in ("eager_rows","original_eager_rows"):
            if result.get(name) is not None:
                require(Counter(row["mode"] for row in result[name])=={"serial":100,"two_stream":100},f"E1b {name} sample count mismatch")
                require(all(row["paired_output_error"]["max_abs"]<=1e-5 for row in result[name]),f"E1b {name} output mismatch")
                for mode in ("serial","two_stream"):
                    require(sorted(row["iteration"] for row in result[name] if row["mode"]==mode)==list(range(100)),f"E1b {name} duplicated/missing iterations")
        if result.get("args",{}).get("eager_reference"):
            require(result.get("eager_rows") and result.get("original_eager_rows"),"Original/static eager reference arms missing")
            reference = load(path.parent/"eager_reference.json")
            require(reference["rows"]==result["eager_rows"],"Matched static eager artifact differs from final same-GPU result")
            require(reference["fixed_noise_seed"]==0 and reference["fixed_output_sha256"]==result["eager_fixed_output_sha256"],"E1b fixed-input/noise control identity mismatch")
            original_reference = load(path.parent/"original_eager_reference.json")
            require(original_reference["rows"]==result["original_eager_rows"] and original_reference["fixed_noise_seed"]==0,"Original eager raw control differs from final result")
            for candidate in (result.get("eager_identity",{}).get("gpu_uuid"),reference.get("gpu_uuid"),reference.get("identity",{}).get("gpu_uuid")):
                require(candidate is None or candidate==uuid,"Eager and graph reference GPU UUIDs differ")
        if result.get("graph_safe_e0"):
            check = result["graph_safe_e0"]
            require(check["passed"] and check["original_nfe"]==check["graph_safe_nfe"]==10 and check["time_nodes_exact"],"Static-metadata wrapper E0 failed")
            require(all(check[name]["max_abs"]<=1e-5 for name in ("normalized_error","decoded_error","velocity_error")),"Static-metadata wrapper numerical gate failed")
            require(result["identity"]["graph_sampler_sha256"]==sha(Path(__file__).with_name("graph_sampler.py")),"Graph sampler source differs from captured identity")
        require(not require_graph or result.get("graph_safe_e0",{}).get("passed"),"Committed graph-safe wrapper E0 is missing")
        details["e1b"]["artifacts"].append(dict(path=str(path),sha256=sha(path),passed=True,gpu_uuid=uuid,
                                                eager_gpu_identity="Shared run UUID applies to original/static eager and graph arms; separately recorded UUIDs checked when present"))
    return details


def audit(root,require_graph=False):
    root = Path(root).resolve()
    expected = list(itertools.product(CONDITIONS,SUITES,(0,5),(0,1)))
    keys = [(mode,interval,suite,task,init) for (mode,interval),suite,task,init in expected]
    report = dict(root=str(root),expected_episodes=112,episodes=[],missing=[],errors=[],micro=None)
    try:
        report["micro"] = audit_micro(root,require_graph=require_graph)
    except Exception as exc:
        report["errors"].append(dict(scope="micro",error=repr(exc)))
    hashes = defaultdict(set)
    for key in keys:
        mode,interval,suite,task,init = key
        directory = root/"episodes"/f"{mode}_r{interval}"/suite/f"task{task:02d}"/f"init{init:02d}"
        if not (directory/"complete.json").exists():
            report["missing"].append(list(key))
            continue
        try:
            result = audit_episode(directory,key)
            report["episodes"].append(result)
            hashes[(suite,task,init)].add(result["init_vector_hash"])
        except Exception as exc:
            report["errors"].append(dict(key=list(key),error=repr(exc)))
    for scene,values in hashes.items():
        if len(values)>1:
            report["errors"].append(dict(scene=list(scene),error="Initial vectors differ across paired conditions"))
    observed = []
    for path in (root/"episodes").glob("*/*/task*/init*/complete.json"):
        if "attempts" not in path.parts:
            observed.append(str(path.parent.relative_to(root/"episodes")))
    expected_paths = {str(Path(f"{m}_r{r}")/s/f"task{t:02d}"/f"init{i:02d}") for m,r,s,t,i in keys}
    extras = sorted(set(observed)-expected_paths)
    if extras:
        report["errors"].append(dict(scope="matrix",unexpected_complete_episodes=extras))
    sources = [WORKSPACE/"preparations/idea2"/name for name in ("scheduler.py","openpi_backend.py","experiments.py","policy.py","server.py","libero_runner.py","graph_bench.py","final_audit.py")]
    if Path(__file__).with_name("graph_sampler.py").exists():
        sources.append(Path(__file__).with_name("graph_sampler.py"))
    sources += [WORKSPACE/"policies/openpi/src/openpi/models_pytorch"/name for name in ("pi0_pytorch.py","gemma_pytorch.py")]
    for packages in (WORKSPACE/"policies/openpi/.venv/lib").glob("python*/site-packages"):
        sources += [packages/"transformers"/name for name in ("cache_utils.py","models/gemma/modeling_gemma.py",
                                                             "models/siglip/modeling_siglip.py","models/paligemma/modeling_paligemma.py")
                    if (packages/"transformers"/name).exists()]
    report["current_source_sha256"] = {str(path.relative_to(WORKSPACE)):sha(path) for path in sources}
    report["source_receipt_scope"] = "Current on-disk scientific sources; original per-run identities remain authoritative for loaded versions"
    report["completed_and_validated"] = len(report["episodes"])
    report["graph_required"] = require_graph
    report["available_records_passed"] = not report["errors"]
    report["passed"] = not report["errors"] and not report["missing"] and len(report["episodes"])==112
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",help="Explicitly requested audit receipt; without this option all artifacts remain read-only")
    parser.add_argument("--allow-partial",action="store_true",help="Permit a progress check with missing planned episodes; never sets passed=true")
    parser.add_argument("--require-graph",action="store_true",help="Committed E1b must have valid graph and same-GPU eager results before final pass")
    args = parser.parse_args()
    report = audit(args.root,require_graph=args.require_graph)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=report["passed"],available_records_passed=report["available_records_passed"],
                          validated=report["completed_and_validated"],missing=len(report["missing"]),errors=report["errors"]),indent=2))
    if report["errors"] or (report["missing"] and not args.allow_partial):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
