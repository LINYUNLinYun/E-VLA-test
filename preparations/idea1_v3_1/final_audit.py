"""CPU-only, read-only evidence audit. Missing work never becomes a pass."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .common import read_manifest, seed


def load(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def scenes(manifest, inits):
    return {(s,t,i) for s in manifest["suites"] for t in manifest["tasks"] for i in inits}


def finite(value, shape=None):
    array = np.asarray(value, dtype=float)
    return (shape is None or array.shape == shape) and bool(np.isfinite(array).all())


def video_check(path, frames):
    subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(path),"-f","null","-"],check=True,capture_output=True)
    data = json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
        "-show_entries","stream=nb_read_frames,avg_frame_rate","-of","json",str(path)],text=True))["streams"][0]
    numerator,denominator = map(float,data["avg_frame_rate"].split("/"))
    assert int(data["nb_read_frames"]) == frames and numerator/denominator == 20, "Video frame count/FPS mismatch"


def protocol_check(protocol, condition, manifest):
    assert protocol["manifest_hash"] == manifest["manifest_hash"]
    assert protocol["condition_id"] == condition["id"] and protocol["server_pid"] > 0
    assert protocol["real_input_parity_passed"]
    recorded = protocol.get("condition", protocol)
    if isinstance(recorded,dict):
        for key in ("skip","horizon","denoising_steps","execute_length","depth","schedule"):
            if key in condition:
                assert recorded.get(key,protocol.get(key)) == condition[key], f"Protocol {key} mismatch"
    assert protocol.get("H") == condition["horizon"] and protocol.get("D") == condition["denoising_steps"]
    assert protocol.get("checkpoint") == manifest["checkpoints"][condition["model"]]["path"]
    for key in ("openpi_sources","scientific_sources"):
        for filename,expected in protocol.get(key,{}).items():
            from .common import ROOT
            source = Path(filename) if Path(filename).is_absolute() else ROOT/filename
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, "Loaded scientific source changed"
    for filename,expected in protocol.get("scientific_code_hashes",{}).items():
        from .common import ROOT
        source = Path(filename) if Path(filename).is_absolute() else ROOT/filename
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    if condition["model"] in ("pi0","pi05"):
        assert protocol["R"] == 5 and protocol["vision_depth"] == 27
        assert protocol["vision_tokens"] == 256 and protocol["vision_compression"] == "off"
    else:
        assert protocol["depth"] == condition["depth"] and protocol["R"] == condition["execute_length"]
        receipt = load(protocol["checkpoint_verification_source"])
        assert receipt["ok"]
        expected = {r["path"]:(r["actual_sha256"],r["expected_bytes"]) for r in receipt["shards"]}
        actual = {r["path"]:(r["sha256"],r["bytes"]) for r in protocol["checkpoint_shards"]}
        assert actual == expected, "Loaded shard receipt mismatch"
        from .common import digest
        assert digest(protocol["checkpoint_shards"]) == protocol["checkpoint_sha256"]


def audit_episode(path, condition, manifest, check_videos=True):
    complete,identity,result = (load(path/name) for name in ("complete.json","identity.json","result.json"))
    assert complete["valid"] and result["valid"]
    for row in (complete,identity,result):
        assert row["manifest_hash"] == manifest["manifest_hash"]
    assert identity["condition"] == condition and result["condition"] == condition["id"]
    scene = tuple(result[k] for k in ("suite","task","init"))
    assert path.name == f"init{scene[2]:02d}" and path.parent.name == f"task{scene[1]:02d}" and path.parent.parent.name == scene[0]
    assert tuple(identity[k] for k in ("suite","task","init")) == scene
    assert identity["init_vector_sha256"] == result["init_vector_sha256"]
    assert identity["env_seed"] == 7 and identity["control_hz"] == 20
    scales = identity["controller"]
    assert np.allclose(scales["input_min"],-1) and np.allclose(scales["input_max"],1)
    assert np.allclose(scales["output_max"],[.05,.05,.05,.5,.5,.5])
    assert np.allclose(scales["output_min"],[-.05,-.05,-.05,-.5,-.5,-.5])
    protocol_check(identity["protocol"],condition,manifest)
    infer,telemetry = lines(path/"infer.jsonl"),lines(path/"telemetry.jsonl")
    assert len(infer) == result["predictions"] and len(telemetry) == result["actions"]+1
    executed = 0
    for index,row in enumerate(infer):
        assert row["request"] == index and row["step"] == executed
        assert row["seed"] == seed(*scene,index)
        assert row["manifest_hash"] == manifest["manifest_hash"] and row["condition_id"] == condition["id"]
        assert finite(row["actions"],(condition["horizon"],7))
        n = row["executed_length"]
        R = condition["execute_length"]
        R = R[scene[0]] if isinstance(R,dict) else R
        assert 0 < n <= R and n <= condition["horizon"]
        assert row["audit"]["NFE"] == condition["denoising_steps"]
        if condition["model"] in ("pi0","pi05"):
            assert row["audit"]["vision_skip"] == condition["skip"] and row["audit"]["original_scanned_ae"]
        else:
            assert row["audit"]["skip"] == condition["skip"] and row["audit"]["schedule"] == condition["schedule"]
        assert row.get("input_hash") and row.get("noise_hash")
        for offset in range(n):
            command = np.asarray(row["actions"][offset]).copy()
            if condition["model"] == "db-cogact":
                command[-1] = -1 if command[-1] < 0 else 1
            actual = telemetry[executed+offset+1].get("executed_action")
            assert finite(actual,(7,)) and np.allclose(command,actual,atol=1e-5,rtol=1e-5), "Executed command mismatch"
        observation = path/f"observation_{index:04d}.npz"
        if observation.exists():
            from preparations.idea1_v2_1.runtime_client import input_digest
            with np.load(observation,allow_pickle=False) as saved:
                obs = {k:saved[k] for k in saved.files if k not in ("instruction","seed")}
                assert input_digest(obs,str(saved["instruction"])) == row["input_hash"]
        elif result.get("save_observations"):
            raise AssertionError("Missing required observation NPZ")
        executed += n
    assert executed == result["actions"]
    for index,row in enumerate(telemetry):
        assert row["step"] == index and finite(row["eef_pos"],(3,))
        assert finite(row["sim_time_s"]) and finite(row["host_time_s"])
        assert row.get("phase_interpretation") and row.get("semantic_phase")
        if index:
            assert row["sim_time_s"] >= telemetry[index-1]["sim_time_s"]
    video = path/"video.mp4"
    if result["video"]:
        assert video.exists() and complete["video_decoded"]
    if check_videos and video.exists():
        video_check(video,len(telemetry))
    return dict(result,infer=infer,path=str(path))


def paired(reference, variant):
    if reference["init_vector_sha256"] != variant["init_vector_sha256"] or reference["success"] != variant["success"]:
        return False
    if len(reference["infer"]) != len(variant["infer"]):
        return False
    return all(all(a[k] == b[k] for k in ("seed","input_hash","noise_hash")) and
               np.allclose(a["actions"],b["actions"],atol=1e-5,rtol=1e-5)
               for a,b in zip(reference["infer"],variant["infer"]))


def audit(manifest, root, check_videos=True, analysis_report=None):
    root = Path(root)
    missing,errors,rows,benchmarks = [],[],{},{}
    def need(ok, message):
        if not ok:
            missing.append(message)
    def attempt(label, function):
        try:
            return function()
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    conditions = {c["id"]:c for c in manifest["conditions"]}
    need(len(conditions)==91,"Original initial matrix must contain exactly91 unique conditions")
    excluded = [str(p) for p in root.rglob("technical_error.json")]
    for p in root.glob("*/episodes/*/task*/init*/complete.json"):
        if ".invalid." not in str(p) and p.parents[4].name not in conditions:
            errors.append(f"Unknown quality condition: {p}")
    for cid,condition in conditions.items():
        data = {}
        for path in (root/cid/"episodes").glob("*/task*/init*"):
            if ".invalid." in str(path) or not (path/"complete.json").exists():
                continue
            row = attempt(str(path),lambda p=path,c=condition:audit_episode(p,c,manifest,check_videos))
            if row:
                key = tuple(row[k] for k in ("suite","task","init"))
                if key in data:
                    errors.append(f"Duplicate scene {cid}: {key}")
                data[key] = row
        rows[cid] = data
        need(data.keys() <= scenes(manifest,manifest["confirm_inits"]),f"{cid}: unexpected scene keys")
        need(scenes(manifest,manifest["screen_inits"]) <= data.keys(),f"{cid}: incomplete screen40")
        parity_path = root/cid/"condition_parity.json"
        if parity_path.exists():
            def parity():
                p = load(parity_path)
                assert p["passed"] and p["max_abs"] <= 1e-5
                assert p["manifest_hash"] == manifest["manifest_hash"] and p["condition_id"] == cid
            attempt(str(parity_path),parity)
        else:
            missing.append(f"{cid}: missing real-input parity")
    initial = {}
    for cid,data in rows.items():
        for key,row in data.items():
            if key in initial and initial[key] != row["init_vector_sha256"]:
                errors.append(f"Cross-condition initial state mismatch: {cid} {key}")
            initial[key] = row["init_vector_sha256"]
    promotion = root/"promotion.json"
    selected = []
    if promotion.exists():
        p = load(promotion)
        need(p["manifest_hash"] == manifest["manifest_hash"],"Promotion identity mismatch")
        selected = p["selected_conditions"]
        from .controller import promotion as recompute
        expected = attempt("Recompute immutable promotion",lambda:recompute(manifest,root))
        if expected:
            need(set(selected)==set(expected["selected_conditions"]),"Promotion does not match frozen screen selection")
        for cid in selected:
            need(cid in rows and scenes(manifest,manifest["confirm_inits"]) <= rows[cid].keys(),f"{cid}: incomplete selected confirmation200")
        for cid,data in rows.items():
            need(cid in selected or not any(k[2]!=0 for k in data),f"{cid}: unselected confirmation records")
    else:
        missing.append("Missing immutable promotion.json")
    for condition in manifest["conditions"]+manifest["benchmark_references"]:
        cid = condition["id"]
        path = root/cid/"benchmark.json"
        if not path.exists():
            continue
        def benchmark():
            b = load(path)
            assert b["manifest_hash"] == manifest["manifest_hash"] and b["condition"] == condition
            assert b["warmup"] == 20 and b["samples"] == 100 and len(b["rows"]) == 100
            from .benchmark import load_banks
            from preparations.idea1_v2_1.runtime_client import input_digest
            records,receipts = load_banks([r["path"] for r in b["banks"]])
            assert receipts == b["banks"] and len(records)>=100
            assert len({r["input_hash"] for r in b["rows"]}) == 100
            assert [r["index"] for r in b["rows"]] == list(range(100))
            for r,record in zip(b["rows"],records[:100]):
                assert r["input_hash"] == input_digest(record["obs"],record["instruction"])
                assert r["seed"] == int(record.get("seed",r["index"]+20))
                assert all(finite(r[k]) and r[k]>=0 for k in ("request_roundtrip_ms","server_request_ms","vision_ms"))
                if condition["model"] in ("pi0","pi05"):
                    detail = r["vision_details"]
                    assert detail["cameras"] == len(detail["shapes"]) and detail["cameras"]>0
                    assert all(s[0]==1 and s[1]==256 for s in detail["shapes"])
                    assert detail["total_tokens"] == detail["cameras"]*256
            protocol_check(b["protocol"],condition,manifest)
            return b
        b = attempt(str(path),benchmark)
        if b:
            benchmarks[cid] = b
    for model in ("pi0","pi05","xiaomi-robotic0","db-cogact"):
        need(model+"_full" in benchmarks,f"{model}: missing Full benchmark")
    for model in ("pi0","pi05"):
        native,matched = benchmarks.get(model+"_full"),benchmarks.get(model+"_full_vision_scan_reference")
        need(bool(matched),f"{model}: missing matched Full benchmark")
        if native and matched:
            def vision_parity():
                for a,b in zip(native["rows"][:2],matched["rows"][:2]):
                    assert a["input_hash"] == b["input_hash"] and a["noise_hash"] == b["noise_hash"]
                    assert a["parity_actions"] is not None and np.allclose(a["parity_actions"],b["parity_actions"],atol=1e-5,rtol=1e-5)
                artifact = load(root/f"{model}_vision_parity.json")
                assert artifact["passed"] and artifact["manifest_hash"] == manifest["manifest_hash"]
            attempt(model+" native/matched real vision parity",vision_parity)
        for keep in (24,21,18,15):
            need(any(c["model"] == model and len(c["skip"]) == 27-keep and c["id"] in benchmarks for c in conditions.values()),f"{model}: no Keep{keep} representative benchmark")
    e2 = []
    for cid,c in conditions.items():
        if c["skip"] or not any((root/cid/v["id"]).exists() for v in manifest["e2"]["variants"]):
            continue
        arms,times = {},{}
        expected = {tuple(s[k] for k in ("suite","task","init")) for s in manifest["e2"]["episodes"]}
        for variant in manifest["e2"]["variants"]:
            name = variant["id"]
            arms[name] = {}
            for scene in expected:
                s,t,i = scene
                path = root/cid/name/s/f"task{t:02d}"/f"init{i:02d}"
                if not (path/"complete.json").exists():
                    continue
                row = attempt(str(path),lambda p=path:audit_episode(p,c,manifest,check_videos))
                if row:
                    need(row["save_observations"]==variant["save_observations"] and row["video"]==(variant["video"]=="all"),f"{cid}/{name}: profile I/O mismatch")
                    need(not row["environment_reused"] or variant["environment_reuse"],f"{cid}/{name}: unexpected environment reuse")
                    arms[name][scene] = row
            need(arms[name].keys()==expected,f"{cid}/{name}: incomplete paired E2 eight scenes")
            candidates = list((root/cid).glob(f"throughput_*_{name}.json"))
            valid = [load(p) for p in candidates if load(p).get("newly_completed_episodes")==8]
            need(len(valid)==1,f"{cid}/{name}: require one clean eight-episode throughput receipt")
            if len(valid)==1:
                t = valid[0]
                need(t["manifest_hash"]==manifest["manifest_hash"] and t["wall_s"]>0 and t["workers"]==variant["workers"],f"{cid}/{name}: throughput identity/workers")
                times[name] = t["wall_s"]
        for name in ("lean_io","reuse_env","two_workers"):
            same = arms[name].keys()==expected and arms["full_io"].keys()==expected and all(paired(arms["full_io"][s],arms[name][s]) for s in expected)
            speedup = times.get("full_io",0)/times[name] if name in times else None
            e2.append(dict(condition=cid,variant=name,paired=same,speedup=speedup,eligible_for_adoption=same and speedup is not None and speedup>1))
    need(bool(e2),"Missing E2 four-arm experiment")
    report_path = Path(analysis_report) if analysis_report else root/"summary.json"
    if report_path.exists():
        report = load(report_path)
        for actual in e2:
            matches = [r for r in report.get("evaluation_optimizations",[]) if r["condition"]==actual["condition"] and r["variant"]==actual["variant"]]
            need(len(matches)==1 and matches[0]["eligible_for_adoption"]==actual["eligible_for_adoption"] and matches[0]["same_initial_states_outcomes_inputs_noise_outputs"]==actual["paired"],f"E2 adoption report mismatch: {actual['condition']}/{actual['variant']}")
    else:
        missing.append("Missing analysis/adoption report")
    for cid,data in rows.items():
        if any(r.get("environment_reused") for r in data.values()):
            need(any(r["condition"]==conditions[cid]["model"]+"_full" and r["variant"]=="reuse_env" and r["eligible_for_adoption"] for r in e2),f"{cid}: quality reused environments without validated E2 adoption")
    frontier = []
    for model,cap in (("xiaomi-robotic0",24),("db-cogact",18)):
        full = rows.get(model+"_full",{})
        for cid,c in conditions.items():
            if c["model"] != model or len(c["skip"]) != cap:
                continue
            for inits,threshold,obligation in (([0],.60,"confirm near-floor cap candidate"),(manifest["confirm_inits"],.70,"extend beyond initial cap")):
                keys = scenes(manifest,inits)
                if keys <= full.keys() and keys <= rows[cid].keys():
                    baseline = sum(full[k]["success"] for k in keys)
                    ratio = sum(rows[cid][k]["success"] for k in keys)/baseline if baseline else None
                    if ratio is not None and ratio>=threshold:
                        if len(inits)==1 and scenes(manifest,manifest["confirm_inits"]) <= rows[cid].keys():
                            continue
                        frontier.append(dict(condition=cid,retention=ratio,required=obligation))
    missing.extend(f"Conditional frontier unresolved: {r}" for r in frontier)
    return dict(passed=not errors and not missing,manifest_hash=manifest["manifest_hash"],errors=errors,missing_requirements=missing,
                quality_counts={cid:dict(total=len(data),per_suite=dict(Counter(k[0] for k in data))) for cid,data in rows.items()},
                benchmark_conditions=sorted(benchmarks),e2=e2,frontier_obligations=frontier,excluded_technical_attempts=excluded,
                limitations=["Lean camera contents cannot be reconstructed without saved NPZ; hashes and executed commands are audited.",
                    "Initial-state hashes are cross-compared, not independently reconstructed from simulator state.",
                    "Warmup20 is recorded by benchmark header/frozen loop; individual warmup outputs are not persisted.",
                    "Failure phases are evidence-backed progress heuristics, not causal failure diagnoses.",
                    "Frozen checkpoint identity receipts are verified; this audit does not rehash every multi-GB weight shard.",
                    "Frontier extensions outside the original manifest require a separately frozen, independently audited revision."],
                video_decoding_enabled=check_videos,audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--analysis-report")
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--skip-video-decode",action="store_true",help="Diagnostic only; never certifies final completion")
    parser.add_argument("--output",help="Only explicit output writes an audit receipt")
    args = parser.parse_args()
    result = audit(read_manifest(args.manifest,verify_sources=True),args.root,not args.skip_video_decode,args.analysis_report)
    if args.skip_video_decode:
        result["missing_requirements"].append("Strict stored-video decode intentionally skipped")
        result["passed"] = False
    if args.output:
        from .common import atomic_json
        atomic_json(args.output,result)
    print(json.dumps(result,indent=2))
    if not result["passed"] and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
