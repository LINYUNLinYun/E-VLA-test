#!/usr/bin/env python3
"""Extract a small, unlaunched fixed-mask VLM fallback proposal from local evidence."""
from pathlib import Path
import csv
import hashlib
import json


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "vlm_fallback_proposal.json"


def main():
    paths = [
        ROOT / "results/idea1_v3_4_E2_v12_candidates/selection.json",
        ROOT / "results/idea1_v3_4_v12_compensation_200/manifest.json",
        ROOT / "results/idea1_v3_4_v12_compensation_200/summary.csv",
        ROOT / "preparations/idea1_v3_4_v12_compensation/common.py",
        ROOT / "preparations/idea1_v3_4_v12_compensation/policy.py",
        ROOT / "preparations/idea1_v3_4_v12_compensation/backend.py",
        Path(__file__).resolve(),
    ]
    selection = json.loads(paths[0].read_text())
    manifest = json.loads(paths[1].read_text())
    condition = next(c for c in manifest["conditions"] if c["id"] == "periodic_extrapolate")
    rows = [{k: r[k] for k in ("condition", "suite", "n", "success")}
            for r in csv.DictReader(paths[2].open())
            if r["condition"] == "periodic_extrapolate" and r["cohort"] == "all200"]
    candidates = []
    for method in ("jacobian", "action_mse", "cosine"):
        row = selection["candidates"][method]
        skip = row["skip_layers_1based"]
        assert len(skip) == 6
        candidates.append({"id": "fixed_v12_" + method,
                           "skip_layers_1based": skip,
                           "keep_layers_1based": row["retained_indices_1based"],
                           "closed_loop_evidence_in_checked_sources": "baseline_200" if method == "jacobian" else "not_found",
                           "rationale": {"jacobian": "Historical usable anchor.",
                                         "action_mse": "One swap from the anchor: replace skipped L14 with L11.",
                                         "cosine": "Distinct early-layer allocation at the same six-layer skip budget."}[method]})
    result = {
        "status": "proposal_only_not_launched_not_frozen",
        "purpose": "A small fixed-mask search if the vision pair does not reproduce; no evidence currently establishes complementary VLM task preferences for this trio.",
        "candidates": candidates,
        "matched_protocol_proposal": {k: v for k, v in condition.items() if k not in ("id", "skip")},
        "schedule_interpretation": "One Full-D10 refresh followed by two draft-D2 requests; H50/R5; identical residual extrapolation for every candidate.",
        "baseline_historical_successes": rows,
        "baseline_historical_inits": sorted({c["init"] for c in manifest["cases"]}),
        "baseline_is_mixed_full_and_draft": True,
        "discovery_proposal": {"inits": [30], "tasks": list(range(10)),
                               "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                               "conditions": "three fixed masks plus Full-D10", "episodes": 160,
                               "requires_official_init_availability_check": True},
        "confirmation_proposal": "Freeze a pair and its category directions from discovery, then use unused init31-35; do not rename preferences using validation outcomes.",
        "implementation_requirements": [
            "Use a new isolated adapter. The historical policy uses imported MASK rather than config['skip']; changing the manifest alone does not change its path.",
            "Historical residual capture is also restricted to imported MASK. The new adapter must capture the chosen mask's residuals from its own Full path and validate per-layer identities.",
            "Keep mask, compensation, refresh schedule, AE D, H/R, noise, and initialization protocol fixed across the comparison; preserve the original implementation and results.",
            "The two new masks have only calibration provenance in the checked selection file. Its closedloop_tested=false predates the later anchor-only evaluation and is not evidence of failure.",
            "Report this as a fixed-mask periodic Full/draft system. It does not establish the standalone submodel success rate or adaptive-path benefit.",
        ],
        "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "reproduce": f"python3 {Path(__file__).resolve()}",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"proposal": str(OUT), "masks": [c["skip_layers_1based"] for c in candidates],
                      "historical_anchor_all": next(r for r in rows if r["suite"] == "all")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
