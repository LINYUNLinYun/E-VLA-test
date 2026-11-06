"""Pair native Full quality controls and distinguish measured XYZ from commands."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from preparations.idea1_v3_1.common import ROOT,atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def errors(a,b):
    delta = np.asarray(a,np.float64)-np.asarray(b,np.float64)
    return dict(samples=len(delta),mean_l2=float(np.linalg.norm(delta,axis=-1).mean()),
                rmse_axis=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
                max_abs_axis=np.max(np.abs(delta),axis=0).tolist())


def analyze(quality,control):
    rows,failures = [],[]
    sources = {str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    seen = set()
    for result_path in sorted(Path(quality).rglob("init00/result.json")):
        relative = Path(*result_path.parts[-4:])
        if str(relative) in seen:
            raise ValueError("Duplicate episode in quality roots: "+str(relative))
        seen.add(str(relative))
        old_path = Path(control)/relative
        if not old_path.exists():
            failures.append(dict(episode=str(relative),reason="missing Full control"))
            continue
        new,old = result_path.parent,old_path.parent
        ni,oi = read(new/"identity.json"),read(old/"identity.json")
        nf,of = lines(new/"infer.jsonl"),lines(old/"infer.jsonl")
        reasons = [key for key in ("suite","task","init","init_vector_sha256","bddl_sha256","env_seed","control_hz","controller")
                   if ni[key]!=oi[key]]
        for key,value in (("H",50),("R",5)):
            if oi["protocol"].get(key)!=value or ni["protocol"].get(key)!=value:
                reasons.append("protocol_"+key)
        expected_steps = 2 if ni["protocol"]["condition_id"]=="d2" else 10
        if ni["protocol"].get("D")!=expected_steps or oi["protocol"].get("D")!=10:
            reasons.append("protocol_D")
        if ni["protocol"]["checkpoint"]!=oi["protocol"]["checkpoint"]:
            reasons.append("checkpoint_path")
        for filename,expected in ni["protocol"]["checkpoint_identity_artifacts"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
                reasons.append("checkpoint_"+filename)
        if oi["condition"]["skip"]:
            reasons.append("control pruning")
        for filename,expected in oi["protocol"]["openpi_sources"].items():
            if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=expected:
                reasons.append("source_"+filename)
        for key in ("noise_hash","seed"):
            if nf[0][key]!=of[0][key]:
                reasons.append("first_infer_"+key)
        if reasons:
            failures.append(dict(episode=str(relative),reason=reasons))
            continue
        nt,ot = lines(new/"telemetry.jsonl"),lines(old/"telemetry.jsonl")
        nsteps,osteps = {x["step"]:x for x in nt},{x["step"]:x for x in ot}
        paired = sorted(nsteps.keys()&osteps.keys())
        eef = errors([[1000*v for v in nsteps[i]["eef_pos"]] for i in paired],
                     [[1000*v for v in osteps[i]["eef_pos"]] for i in paired])
        commands = [i for i in paired if i>0 and nsteps[i].get("executed_action") is not None and osteps[i].get("executed_action") is not None]
        command = errors([nsteps[i]["executed_action"] for i in commands],[osteps[i]["executed_action"] for i in commands])
        nr,orr = read(result_path),read(old_path)
        row = dict(suite=nr["suite"],task=nr["task"],init=0,candidate_mode=ni["protocol"]["condition_id"],candidate_success=nr["success"],full_success=orr["success"],
            first_observation_hash_matched=nf[0]["input_hash"]==of[0]["input_hash"],
            pairing="official init, BDDL, controller, native checkpoint/source, first noise and seed; initial render hash separately recorded",
            candidate_actions=nr["actions"],full_actions=orr["actions"],measured_eef_xyz_mm_at_equal_step=eef,
            executed_controller_commands_native_units_at_equal_step=command,
            candidate_phase_counts=dict(Counter(x["semantic_phase"] for x in nt)),
            full_phase_counts=dict(Counter(x["semantic_phase"] for x in ot)),
            candidate_final_phase=nr["final_phase"],full_final_phase=orr["final_phase"],
            candidate_furthest_phase=nr["furthest_phase"],full_furthest_phase=orr["furthest_phase"],
            video=str(new/"video.mp4") if (new/"video.mp4").exists() else None)
        initial_through5 = [i for i in paired if i<=5]
        row["measured_eef_xyz_mm_initial_through5"] = errors(
            [[1000*v for v in nsteps[i]["eef_pos"]] for i in initial_through5],
            [[1000*v for v in osteps[i]["eef_pos"]] for i in initial_through5])
        first5 = [i for i in paired if 1<=i<=5]
        row["measured_eef_xyz_mm_post_actions1to5"] = errors(
            [[1000*v for v in nsteps[i]["eef_pos"]] for i in first5],
            [[1000*v for v in osteps[i]["eef_pos"]] for i in first5])
        rows.append(row)
        for base in (new,old):
            for name in ("identity.json","infer.jsonl","telemetry.jsonl","result.json"):
                p = base/name
                sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(schema_revision=2,rows=rows,paired_episodes=len(rows),candidate_successes=sum(r["candidate_success"] for r in rows),
        early_eef_scope="Executed global episode steps1..5 only, immediately after episode reset; not future offsets1..5 at every replan, and not a late-stage safety test",
        candidate_modes=sorted(set(r["candidate_mode"] for r in rows)),
        full_successes=sum(r["full_success"] for r in rows),incompatible_controls=failures,source_sha256=sources,
        caveat="Equal-step closed-loop trajectory divergence is descriptive, not same-observation error. Heuristic phases do not establish causes of failure. Measured EEF XYZ is distinct from executed controller commands.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality",required=True)
    parser.add_argument("--control",default="results/idea1_v3_1/run/pi0_full/episodes")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = analyze(args.quality,args.control)
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","source_sha256")},indent=2))


if __name__ == "__main__":
    main()
