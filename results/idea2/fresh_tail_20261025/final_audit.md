# Fresh-tail final independent audit

Status: **passed**. Pending: **none**. CPU only; auditor launched no GPU work and changed no frozen runtime or adapter.

79,253 checks passed. Twelve valid formal processes contain 21,600 measured outputs plus48 implementation oracle checks, all exact zero. Each resource family is compared only on its own GPU. All forced tail frontiers are18; source, bank, quota, seed/input identity and cleanup checks pass. The SM32 tail1 repeat at653.692ms remains valid; the separate20-group diagnostic at654.393ms is not merged into formal results.

Matched quality contains600 outputs sharing explicit current/actual-old/noise identities, with48 exact-zero schedule oracles. The600 causal AE replays only change the final frontier entries;200 tail0 normalized/decoded replay checks are exact zero. SM16 causal output remains identical; these serial replays do not count as timing.

The unique valid EGL1/GPU2 quality run contains56 valid cases and1782 requests. All input/prediction hashes, original model seeds, finite shapes, tail frontiers, two-slot limits and render guards pass.48 bootstrap comparisons are exact. SM32 tail1's libero_spatial task5 failure at220 actions is retained as a valid task failure; seven other cases succeed. All other configurations succeed8/8. Episode work sums are reconstructed from case receipts and exclude shared server startup/loading; free trajectories are not matched latency experiments.

Two actual-Kernel traces each validate8 current-generation publication/snapshot orderings. Only the first mandatory snapshot is about5–9us after final publication; later fresh snapshots remain after it. Profile disturbance and runtime Event timing scope are explicit. Tail2's first request costs about6.8ms, so continuous throughput does not establish single-request latency neutrality.

Excluded without deleting history: GPU0 foreign preflight and load interruption; initial CPU-JAX omission/OOM; old wrong-EGL quality run; our wrong-EGL renderer's interruption of SM32 tail1 repeat loading; planned-only quality directories. The first four passed SM32 GPU3 cases remain valid despite their parent matrix's later failure; the remaining two passed cases complete the formal set.

Report local links and major tables were checked. Small report fixes clarify averaging, snapshot timing, excluded runs, baseline quota, and executable reproduction environment. No cross-GPU global performance winner, full LIBERO guarantee or new camera-age SLO claim is made. Exact report hash and evidence counts are in final_audit.json.
