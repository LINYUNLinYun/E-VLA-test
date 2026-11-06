"""Build the bounded E2 Full / residual03 / canonical D2 evidence table."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from preparations.idea1_v3_1.common import SUITES,atomic_json


def paired(left,right):
    return dict(both_success=sum(a and b for a,b in zip(left,right)),
        left_only=sum(a and not b for a,b in zip(left,right)),
        right_only=sum(b and not a for a,b in zip(left,right)),
        both_failure=sum(not a and not b for a,b in zip(left,right)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate",required=True)
    parser.add_argument("--reuse",required=True)
    parser.add_argument("--d2")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    paths = [Path(args.gate),Path(args.reuse)]+([Path(args.d2)] if args.d2 else [])
    values = [json.loads(p.read_text()) for p in paths]
    gate,reuse = values[:2]
    modes = {"reuse03":reuse}
    if args.d2:
        modes["d2"] = values[2]
    if gate["status"]!="complete" or not gate["parity_passed"]:
        raise ValueError("No completed real-model gate")
    cohorts = {}
    for mode,analysis in modes.items():
        if analysis["paired_episodes"]!=40 or analysis["incompatible_controls"] or analysis["candidate_modes"]!=[mode]:
            raise ValueError("Full40 compatible cohort required: "+mode)
        cohorts[mode] = {(r["suite"],r["task"],r["init"]):r for r in analysis["rows"]}
    keys = sorted(cohorts["reuse03"])
    if any(sorted(rows)!=keys for rows in cohorts.values()):
        raise ValueError("Different episode keys between controls")
    full = [cohorts["reuse03"][k]["full_success"] for k in keys]
    vectors = {"full":full}|{mode:[rows[k]["candidate_success"] for k in keys] for mode,rows in cohorts.items()}
    quality = {name:dict(successes=sum(vector),episodes=40,rate=sum(vector)/40,
        suites={suite:dict(successes=sum(value for k,value in zip(keys,vector) if k[0]==suite),episodes=10)
                for suite in SUITES}) for name,vector in vectors.items()}
    timings = gate["timing_summary"]
    latency = {}
    for mode in ("reuse03","d2"):
        latency[mode] = dict(request_p50_ms=timings[mode]["request_ms"]["p50"],
            ae_p50_ms=timings[mode]["ae_ms"]["p50"],
            request_speedup_vs_native=timings["native"]["request_ms"]["p50"]/timings[mode]["request_ms"]["p50"],
            ae_speedup_vs_matched_full=timings["refresh_all"]["ae_ms"]["p50"]/timings[mode]["ae_ms"]["p50"])
    report = dict(quality=quality,latency=latency,timing_summary=timings,
        paired_outcomes={f"{left}_vs_{right}":paired(vectors[left],vectors[right])
                         for left,right in (("full","reuse03"),("full","d2"),("reuse03","d2"))
                         if left in vectors and right in vectors},
        stages={mode:dict(failed_final_phase_counts=dict(Counter(r["candidate_final_phase"] for r in rows.values() if not r["candidate_success"])),
            failed_furthest_phase_counts=dict(Counter(r["candidate_furthest_phase"] for r in rows.values() if not r["candidate_success"])),
            full_success_candidate_failure_final_phase_counts=dict(Counter(r["candidate_final_phase"] for r in rows.values() if r["full_success"] and not r["candidate_success"])),
            full_success_candidate_failure_furthest_phase_counts=dict(Counter(r["candidate_furthest_phase"] for r in rows.values() if r["full_success"] and not r["candidate_success"])),
            full_success_candidate_failure_episodes=[dict(suite=r["suite"],task=r["task"],init=r["init"],final=r["candidate_final_phase"],furthest=r["candidate_furthest_phase"]) for r in rows.values() if r["full_success"] and not r["candidate_success"]],
            initial_observation_hash_matches=sum(r["first_observation_hash_matched"] for r in rows.values()),
            measured_eef_post_actions1to5_mean_episode_l2_mm=float(np.mean([r["measured_eef_xyz_mm_post_actions1to5"]["mean_l2"] for r in rows.values()])),
            measured_eef_initial_through5_mean_episode_l2_mm=float(np.mean([r["measured_eef_xyz_mm_initial_through5"]["mean_l2"] for r in rows.values()])),
            measured_eef_all_equal_steps_mean_episode_l2_mm=float(np.mean([r["measured_eef_xyz_mm_at_equal_step"]["mean_l2"] for r in rows.values()])),
            selected_videos=[r["video"] for r in rows.values() if r["video"]]) for mode,rows in cohorts.items()},
        approximation="Full AE scanned Transformer pre-final-RMSNorm residual cached at updates0/3, FP32 subtraction from BF16 suffix input. At each10Euler step recompute state/action/time suffix, finalRMSNorm and head. Fresh full prefix per request; request-local cache.",
        native_gate=dict(inputs=len(gate["checks"]),passed=gate["parity_passed"],runtime_audit=gate["runtime_audit"],request_cache_isolation=gate["request_cache_isolation"],
            full_refresh_raw_max_abs=max(r["modes"]["refresh_all"]["raw"]["max_abs"] for r in gate["checks"]),
            full_refresh_decoded_max_abs=max(r["modes"]["refresh_all"]["decoded"]["max_abs"] for r in gate["checks"]),
            canonical_policy_infer_decoded_max_abs=max(r["canonical_policy_infer_decoded"]["max_abs"] for r in gate["checks"]),
            d2_refresh_raw_max_abs=max(r["d2_native_parity"]["raw"]["max_abs"] for r in gate["checks"]),
            d2_refresh_decoded_max_abs=max(r["d2_native_parity"]["decoded"]["max_abs"] for r in gate["checks"])),
        timing_scope=gate["latency_scope"],timing_inputs_by_suite=dict(Counter(r["suite"] for r in gate["checks"])),
        early_eef_scope="Only executed global episode steps1..5 after reset, not a repeated horizon-offset or late-stage check",
        warmup=gate["warmup"],timing_samples=gate["samples"],gpu_preflight=gate["gpu_preflight"],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]},
        limitations="Single init0 per task,40pairs screening only; no uncertainty claim. Latency24inputs/3warmups, initialcompiles excluded. No same-time live rollout speedup inferred from reused quality controls. Equal-step closed-loop EEF divergence includes route, pacing and accumulation differences, not same-observation action error or Full-driven shadow comparison. Commands use separate native units; heuristic phases are not failure causes.")
    atomic_json(args.output,report)
    print(json.dumps({k:report[k] for k in ("quality","latency","paired_outcomes","stages")},indent=2))


if __name__ == "__main__":
    main()
