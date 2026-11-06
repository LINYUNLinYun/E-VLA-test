# Progressive KV requirement audit — completed

Status: **passed**. Pending items: **none**.

Independent CPU evidence audit completed 85,612 checks without failure; the final report passed numeric/scope review and all 29 local links resolve. No GPU work was launched by the auditor. Scoped limitations remain: small closed-loop quality subset, approximate conditioning, screen-only SM32, and no cross-GPU performance/SLO-budget claims.

Final report: [idea2_progressive_kv_report_20261025.md](../../../docs/experiments/idea2/idea2_progressive_kv_report_20261025.md). The JSON audit records its content hash and closure details.

## Historical checkpoints

The following sections preserve the audit chronology; their earlier pending states were superseded by the final evidence and report checks above.

## Confirmed findings

- Original bank contains 100 records and 8 instructions. gate_v2/screen used cross-task fallback within one epoch and must be excluded from intended same-task ranking; their exact matched-frontier result remains useful diagnostic evidence.
- Corrected same_task_bank contains 13 exact source records with one instruction; source/subset SHA-256 and original indices verified.
- Original screen: 1600 paired rows, 20 verification rows; per-pair source/noise/input signatures match, frontier histories are monotonic with 10 steps, fresh18 uses all-new KV, matched-oracle max error is zero. Runtime source hashes still match.
- One shared K/V-layer scratch is reported as 991232 bytes; two sets of 36 KV addresses are disjoint. This is address evidence and source review, not an allocator/race-sanitizer proof.
- Export admission_to_action_ms as loop_iteration_host_ms. Prefetched admission precedes this timestamp; waiting for a future frame can instead be included in it.
- Original periodic driver is not bounded latest-wins: lookahead retains source+1 despite newer arrivals. It must not support official bounded-camera SLO claims. A results-only replacement driver is pending.
- Closed-loop policy resets on suite/instruction change and bootstraps fresh; the performance bank flaw does not invalidate its task ownership design.

## Remaining closure

1. Corrected same-task gate: 12 requests passed with zero matched-schedule error, 10 steps each, both slots/two epochs and 3 claimed ahead tickets. Inspect both formal rounds and their embedded bank/source hashes next.
2. Inspect actual representative trace analyses and signature/correlation coverage.
3. Review corrected SLO driver and received/admitted/dropped/buffered conservation, current/old capture ages.
4. Inspect all 40 closed-loop case outcomes and provenance; failures remain valid outcomes.
5. Regenerate final report/CSV and exclusions, then reconcile every requirement against terminal evidence.

Machine-readable evidence and live checkpoint status: `requirements_audit.json`.

## Completed closed-loop raw-artifact audit (still in progress overall)

CPU-only audit checked all 40 episode terminal files and 1,187 request rows, original deterministic seeds, input/prediction file hashes, tensor shapes/finiteness, per-round frontiers, two-slot bounds, and old/new observation IDs. All checks passed. Each of five modes succeeded in 8/8 cases; step totals lead0/1/3/6/fresh18 = 1149/1146/1200/1178/1181. All 32 bootstrap comparisons have identical input/seed, normalized and decoded actions. All 71 source hashes match the recorded terminal manifest. This is a small closed-loop quality subset with no future Encoder; it is not a matched-input speed comparison or full LIBERO score.

Corrected same-task screen bank hash matches the verified 13-record subset and before/after runtime hashes match current files. GPU1 `formal_same_task` is excluded in full due to documented foreign GPU load (`gpu1_interference.json`). GPU2 formal/repeat/trace/SLO and GPU0 resource-screen audit remain pending; no completion claim.

## Final evidence audit — report review pending

{"status": "passed_evidence_report_pending", "checks": 85612, "errors": [], "quality_cases": 56, "quality_requests": 1708}

Two GPU2 formal runs each have 5412 rows (4800 paired, 600 continuous, 12 oracle). Same-GPU0 full-card blocks each contain three arms / 2706 rows; SM16 blocks each contain one arm / 902 rows (800 paired, 100 continuous, 2 oracle per arm); they are independent-process ABBA blocks, not within-capture paired resource switching. All counted processes exit 0 and recorded runtime sources are unchanged. Main SLO has 2800 rows; resource absolute-age SLO has 400 rows. Camera causality/source/drop ledger and null bootstrap age checks pass. All seven nsys reports have complete graph ancestry and retain actual Kernel tails. All 56 closed-loop cases pass their recorded protocol and artifact checks, each configuration 8/8 successful.

Excluded: original cross-task gate/screen/formal, entire GPU1 interference formal, initial SM16 exit -6 cleanup failure, and zero-case SM32 port19086 preflight. No direct cross-GPU performance ranking; GPU3 resource SLO has no same-device full-card baseline and cannot establish a 5% budget verdict. Final report/link review remains pending.

## Final report closure

Report numbers were checked against performance, paired-trajectory, SLO and quality CSVs. Paired module spans, 8-request all-work, continuous throughput and profile-only endpoints are explicitly distinguished. The 62 ms fresh+Encoder P95 variation (179.54/131.76 ms) is retained. GPU3 SLO gives absolute ages only; SM32 remains screen-only. Approximate-vs-fresh differences are separated from the exact matched-frontier implementation oracle. All 29 local links resolve. Final status: passed; pending: none.
