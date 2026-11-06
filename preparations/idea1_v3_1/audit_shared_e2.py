"""CPU-only independent raw-evidence audit of the bounded shared-service E2."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import ROOT,atomic_json,read_manifest
from .final_audit import audit_episode
from .same_process_parity import representatives,sha
from .shared_service_e2 import client_commands,fixed_checks,summarize


def adoption_evidence(summary,first_pairing=None):
    """Scientific negative outcomes remain valid data, never technical errors."""
    rows = []
    for variant in summary["comparisons"]:
        scenes = []
        for pair in variant["pairs"]:
            deltas = pair["first_input_deltas"]
            images = {k:v for k,v in deltas.items() if "image" in k.lower()}
            states = {k:v for k,v in deltas.items() if "image" not in k.lower()}
            exact = all(v["max_abs"]==0 and v["different_fraction"]==0 for v in deltas.values())
            assert len(pair["paired_action_checks"])==pair["matched_requests"]
            first_paired = (first_pairing[variant["variant"]][tuple(pair["scene"])] if first_pairing is not None else exact and pair["matched_requests"]>0)
            scenes.append(dict(scene=pair["scene"],same_official_initial=pair["same_official_initial"],same_outcome=pair["same_outcome"],
                matched_requests=pair["matched_requests"],nonvacuous=pair["matched_requests"]>0,first_input_exact=exact,
                first_request_input_noise_seed_paired=first_paired,
                first_state_max_abs=max((v["max_abs"] for v in states.values()),default=None),
                first_image_max_abs=max((v["max_abs"] for v in images.values()),default=None),
                first_image_max_different_fraction=max((v["different_fraction"] for v in images.values()),default=None),
                input_delta_classification="identical" if exact else ("image-only difference" if all(v["max_abs"]==0 for v in states.values()) else "numeric state difference (possibly also images)"),
                matched_actions_pass=all(c["passed"] for c in pair["paired_action_checks"])))
        rows.append(dict(variant=variant["variant"],scenes=scenes,matched_requests_total=sum(s["matched_requests"] for s in scenes),
            matched_scenes=sum(s["nonvacuous"] for s in scenes),first_input_exact_scenes=sum(s["first_input_exact"] for s in scenes),
            first_request_paired_scenes=sum(s["first_request_input_noise_seed_paired"] for s in scenes),
            all8_initials_outcomes_match=len(scenes)==8 and all(s["same_official_initial"] and s["same_outcome"] for s in scenes),
            evidence_supports_consideration=summary["fixed_controls_pass"] and len(scenes)==8 and
                all(s["same_official_initial"] and s["same_outcome"] and s["nonvacuous"] and s["first_input_exact"] and s["first_request_input_noise_seed_paired"] and s["matched_actions_pass"] for s in scenes)))
    return rows


def verify(root,manifest,check_videos=True):
    from .benchmark import load_banks
    from preparations.idea1_v2_1.runtime_client import input_digest
    root = Path(root)
    receipt = json.loads((root/"validation.json").read_text())
    assert receipt["status"]=="completed" and receipt["server_exit_code"]==0
    assert receipt["manifest_hash"]==manifest["manifest_hash"] and receipt["condition"]=="pi05_full"
    assert receipt["helper_source_sha256"]==sha(Path(__file__).with_name("shared_service_e2.py"))
    assert receipt["gpu"]=="0" and receipt["warmup"]==20 and receipt["raw_control_actions_available"] is False
    condition = next(c for c in manifest["conditions"] if c["id"]=="pi05_full")
    assert receipt["protocol"]["server_pid"]==receipt["server_pid"] and receipt["protocol"]["condition_id"]==condition["id"]
    assert receipt["parity"]["passed"] and receipt["parity"]["max_abs"]<=1e-5
    records,banks = load_banks([b["path"] for b in receipt["banks"]])
    assert banks==receipt["banks"]
    indices = representatives(records)
    assert indices==receipt["indices"] and len(indices)==5
    reference = receipt["fixed_reference"]
    assert [r["bank_index"] for r in reference]==indices
    for row,index in zip(reference,indices,strict=True):
        assert row["input_hash"]==input_digest(records[index]["obs"],records[index]["instruction"])
        assert row["seed"]==int(records[index]["seed"]) and np.asarray(row["actions"]).shape==(10,7)
    assert len(receipt["controls"])==8
    assert [(r["variant"],r["when"]) for r in receipt["controls"]]==[(v["id"],when) for v in manifest["e2"]["variants"] for when in ("before","after")]
    for row in receipt["controls"]:
        recomputed = fixed_checks(reference,row["results"])
        assert row["comparison"]==recomputed and recomputed["passed"]
    assert len(receipt["arms"])==4
    command_output = receipt["server_command"][receipt["server_command"].index("--output")+1]
    assert Path(command_output).resolve()==root.resolve()
    expected_paths = {root/condition["id"]/v["id"]/s["suite"]/f"task{s['task']:02d}"/f"init{s['init']:02d}"
                      for v in manifest["e2"]["variants"] for s in manifest["e2"]["episodes"]}
    actual_paths = {p.parent for p in (root/condition["id"]).glob("*/*/task*/init*/complete.json") if ".invalid." not in str(p)}
    assert actual_paths==expected_paths,"Require exactly32 canonical completed keys; technical attempts excluded"
    counts,successes,first_requests = {},{},{}
    for arm,variant in zip(receipt["arms"],manifest["e2"]["variants"],strict=True):
        name = variant["id"]
        assert arm["variant"]==name and arm["commands"]==client_commands(receipt["server_command"][receipt["server_command"].index("--manifest")+1],command_output,receipt["port"],variant)
        assert len(arm["pids"])==variant["workers"] and len(set(arm["pids"]))==variant["workers"]
        assert arm["exit_codes"]==[0]*variant["workers"] and np.isfinite(arm["wall_s"]) and arm["wall_s"]>0
        count,success = 0,0
        first_requests[name] = {}
        for scene in manifest["e2"]["episodes"]:
            p = root/condition["id"]/name/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"
            row = audit_episode(p,condition,manifest,check_videos)
            identity = json.loads((p/"identity.json").read_text())
            assert identity["protocol"]["server_pid"]==receipt["server_pid"]
            assert row["video"]==(variant["video"]=="all") and row["save_observations"]==variant["save_observations"]
            assert not row["environment_reused"] or variant["environment_reuse"]
            first_requests[name][tuple(scene[k] for k in ("suite","task","init"))] = row["infer"][0]
            count += 1
            success += row["success"]
        counts[name],successes[name] = count,success
    assert sum(counts.values())==32
    summary = summarize(root,manifest,receipt["controls"])
    assert summary==receipt["summary"]
    first_pairing = {name:{scene:all(row[k]==first_requests["full_io"][scene][k] for k in ("input_hash","noise_hash","seed"))
                          for scene,row in rows.items()} for name,rows in first_requests.items() if name!="full_io"}
    evidence = adoption_evidence(summary,first_pairing)
    speedups = {}
    baseline = receipt["arms"][0]["wall_s"]
    for arm in receipt["arms"]:
        speedup = baseline/arm["wall_s"]
        assert arm["aggregate_throughput_speedup"]==speedup
        assert arm["episodes_per_hour"]==8*3600/arm["wall_s"]
        speedups[arm["variant"]] = speedup
    return dict(passed=check_videos,manifest_hash=manifest["manifest_hash"],validation_receipt=str((root/"validation.json").resolve()),
        validation_receipt_sha256=sha(root/"validation.json"),audit_source_sha256=sha(__file__),counts=counts,successes=successes,
        speedups=speedups,summary=summary,adoption_evidence=evidence,adoption_automatically_authorized=False,
        fixed_control_max_abs=max(c["max_abs"] for r in receipt["controls"] for c in r["comparison"]["checks"]),
        passed_definition="passed certifies raw data integrity/completion, NOT optimization adoption; changed outcomes are a valid negative experiment",
        limitations=["Decoded fixed-input actions are tested; raw sampler outputs are unavailable from unchanged RPC.",
            "Common observational first-input capture is included in every timed arm.",
            "Only identical input/noise requests support pointwise action parity. Different closed-loop observations are not treated as scheduler error.",
            "This bounded32-episode validation does not modify quality defaults or the frozen adoption gate."],video_verified=check_videos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--output")
    parser.add_argument("--skip-video-decode",action="store_true")
    args = parser.parse_args()
    result = verify(args.root,read_manifest(args.manifest,verify_sources=True),not args.skip_video_decode)
    if args.output:
        atomic_json(args.output,result)
    print(json.dumps(result,indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
