# Future VLM path diversity: planning note

**Status: planning only.** This note does not decide whether the active branch is negative, freeze a protocol, or change any current plan. I read frozen path and policy source plus prior completed reports; I did not open the current `vlm_within_suite_v1` results tree or use its outcomes.

The current five fastpaths form a narrow layer family. All five skip six of 18 one-based positions, and every named local path is one layer replacement away from `jac_anchor`. For equal-size masks, one replacement gives Hamming/symmetric-difference distance 2. All five share skipped layers `{2, 3, 17}`.

|  | Jac | Spatial | Object | Goal | Long |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jac | 0 | 2 | 2 | 2 | 2 |
| Spatial | 2 | 0 | 4 | 4 | 2 |
| Object | 2 | 4 | 0 | 4 | 4 |
| Goal | 2 | 4 | 4 | 0 | 4 |
| Long | 2 | 2 | 4 | 4 | 0 |

The four new masks below spread skips across different depth regions. Each still skips six layers, retains 12 indices nominally, keeps layers 1 and 18 out of the skip set, and sits at least six bits from every existing fastpath. Their pairwise distances range from 6 to 12. These are geometric proposals, not evidence that any path works or that paths complement one another.

| Proposed path | One-based skipped layers | Minimum distance from current paths |
| --- | --- | ---: |
| `p1_early_even` | 2, 4, 6, 8, 10, 12 | 8 |
| `p2_mid_stride` | 3, 6, 9, 12, 14, 16 | 6 |
| `p3_late_stride` | 4, 7, 10, 13, 15, 17 | 8 |
| `p4_mid_high_mix` | 4, 6, 8, 11, 13, 15 | 10 |

The execution code constrains how to interpret “12/18 retained.” The residual wrapper accepts skip and capture IDs 1–17, and history dictionaries must match the fixed mask exactly. The encoder updates hidden state only in layers 1–17; layer 18 has no attention/FFN hidden-state update and cannot be skipped or captured by the wrapper. Thus six skips leave 12 unmasked indices, while the code performs hidden updates in 11 of the first 17 layers plus its existing terminal prefix-K/V behavior. The proposed masks preserve this existing meaning. Layer 17 is the last position with an attention/FFN update; the implementation permits skipping it and the current masks already do, but code alone does not establish semantic criticality. `p3_late_stride` deliberately includes it as an execution-pattern axis that requires the same strict usability gate. The source permits skipping layer 1; this initial panel keeps it retained as a conservative choice, not because the implementation declares it critical. The layer intervention remains in the VLM prefix path; keep vision processing and the inherited action-expert sampler unchanged.

If the owner later opens a separate branch, a bounded design would use six fixed arms: Full-D10, `jac_anchor`, and the four proposed masks. Keep H50/R5, Full-D10 followed by Fast-D2/Fast-D2, own-path residual extrapolation, environment seed 7, 10 settling actions, 400 policy actions, and the existing 1e-5 numerical tolerances. Do not tune tolerances to make a new path pass.

| Stage | Proposed fixed budget | Separation |
| --- | --- | --- |
| Static usability | No episodes or model requests | Confirm six unique supported skip IDs and the unchanged schedule. A failure stops before a cohort. |
| Fixed-input calibration | Two task prompts, one per metadata class; two official initial observations per prompt; at most 72 requests across six arms | For each fast arm, exercise `Full-D10, D2, D2, Full-D10, D2, D2`; the Full control makes six D10 requests. Check protocol, mask, request history, and numerical implementation gates only. No success- or latency-based path selection. |
| Discovery | 12 tasks, six per metadata class, two inits per task; 24 episodes/arm, 144 total | Split on exact official instruction groups so duplicate prompts stay together. Assign groups with a deterministic SHA-256 rule, fixed before seeing outcomes. |
| Heldout | Complementary 12 tasks, six per class, two different inits per task; 24 episodes/arm, 144 total | Prompt/task groups and init IDs are disjoint from discovery. Run the entire fixed panel; predeclare category analyses and familywise handling. |

The resulting maximum is 288 closed-loop episodes, 115,200 policy actions at the 400-action cap, and 2,880 settling actions, plus at most 72 calibration requests. The 24-task metadata report has 12 tasks per class and 20 distinct instruction strings; grouping duplicate prompts makes a balanced six-task-per-class split possible, but tasks and scenes still are not statistically independent. This is a screen, not an official full-init benchmark.

Tentative init IDs are 44–45 for calibration, 46–47 for discovery, and 48–49 for heldout. The prior CPU asset audit found 50 official initial states per task, and the frozen current source defines its own discovery/confirmation range as 36–43. These future IDs are not asserted to be unused; a separate source and case-availability check would be required before any future protocol review. No current cases were inspected to establish that.

One owner-provided implementation risk belongs in any future resource review: separate fresh model loads can encounter an external memory-availability gap and OOM even when a logical lease is held. A later persistent-backend/config-switch review could reduce that gap, but it would need unchanged numerical parity and a residual-state reset audit before execution. This plan proposes no runtime change.

The machine-readable masks, exact distances, stage counts, caveats, and source hashes are in [plan.json](plan.json). The principal source evidence is:

- [`common.py`](../../../preparations/ease_insights_20261103/vlm_within_suite_v1/common.py#L19): frozen masks, shared schedule, whitelist, and stage scope; SHA256 `8568591ae47d9bd938a968d3f693da0b7e468f2c50d6b499a80d65fbf8a5d1ec`.
- [`policy.py`](../../../preparations/ease_insights_20261103/vlm/policy.py#L26): Full/Fast cadence and per-path residual history; SHA256 `91312c72c131c25b243859e8c07efcbdede5a4a6f6aed0a1a1e530b186433845`.
- [`backend.py`](../../../preparations/ease_insights_20261103/vlm/backend.py#L13): VLM skip/capture range, residual identity, hidden update behavior, and inherited sampler; SHA256 `33cbc9bd57f146e3b9a5e457fa78dbd075655f86e08469b5ee2c477d6cd561c6`.
- [`pi0_infer.py`](../../../policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py#L934): zero-based Triton indices and terminal layer behavior; SHA256 `c554399a9adf25b1ee512a73325cbe05f56943b1a4cbc1748934d2f413cd56b8`.
- [`candidate.json`](../semantic_pool_feasibility/within_suite_candidate/candidate.json): prior metadata-only 24-task proposal, not a path-quality result; SHA256 `478b637d0e688ca8178551ef4c18e548c70492045844b1f5374923aeeada47bc`.
- [`runtime_feasibility.json`](../semantic_pool_feasibility/within_suite_candidate/runtime_feasibility.json): prior finding that the finite old numeric gate does not cover new prompts; SHA256 `d75890f995586c075b2e63298891096fcf1e35861576439aee32e4c9083532a5`.

Prior runtime acceptance was explicitly runtime-only and did not select a path or establish closed-loop benefit. This planning note makes no quality, latency, or complementarity claim.
