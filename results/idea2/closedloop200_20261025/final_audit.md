# Final independent CPU audit: closedloop200

Status: **passed**. Pending items: **none**.

Completed 874,589 checks over800 unique cells:768 new episodes and32 compatible reused pilot episodes. Every arm has200 cases, each suite50 and each task five initializations. All28,507 request records were inspected once after the dispatcher and statistics reached terminal state. No GPU/model/simulator was started by the auditor.

| Configuration | Success/200 | Valid task failures | Steps | Requests |
|---|---:|---:|---:|---:|
| fresh18 | 191/200 | 9 | 32685 | 6611 |
| sm16_tail0 | 193/200 | 7 | 33379 | 6752 |
| sm32_tail0 | 184/200 | 16 | 38656 | 7813 |
| sm32_tail2 | 190/200 | 10 | 36254 | 7331 |

All42 task failures remain valid outcomes. No selected episode had a technical retry, and no valid outcome was rerun. The zero-episode `gpu2_chunk1` port preflight failure remains excluded; its replacement has a separate directory. The original first chunk was not rerun.

The10 chunk parent/server exits and160 passed client batches are clean. Each batch has a matching receipt, drain success and render/compute ownership evidence. Policy episode IDs advance across batches; Pool epochs advance within each batch and may restart after Graph/Pool reconstruction. Each episode restarts observation/request0 with full fresh KV.

Validated77 frozen source hashes, model configuration, checkpoint path/size/mtime, environment/model seeds, environment assets, settling10,20Hz control,D10/H10/R5, SM quotas, no lookahead, old/new source mapping, mandatory fresh tails, two KV slots and zero steady-state recapture. Every saved input/action file SHA was checked; saved decoded arrays match finite10x7 actions and normalized outputs are finite10x32. Initial observation NPZ contents/digests were independently checked.

All200 initial states match across four arms on the same physical GPU within each paired case.600 candidate-vs-fresh bootstrap comparisons pass max1e-5/MAE1e-6; observed maximum and MAE are both0. Reused32 source directories, protocol hashes, identity and original process/render evidence match the reuse proof.

Full200 and pilot-excluded192 tables were reconciled with raw outcomes. New-only successes are fresh183, SM16-tail0=185, SM32-tail0=176, SM32-tail2=182 out of192. All30 paired comparisons in each table have correct b/c, signed delta, exact McNemar arithmetic and valid bootstrap bounds; Wilson intervals and fixed seed20261025/10,000 draws match the prospective implementation. Statistical source/input hashes match the final files. This is not a second blinded replication; known pilot successes dilute full200 deltas by4%.

An initial auditor-only schema error was retained as `final_audit_initial_adapter_error.json`: the old pilot execution uses numeric `arm` for ordering and textual `label` for configuration. The audit parser was corrected to configuration→label→arm before raw episode scanning, then the complete CPU pass succeeded. No experimental outcome or frozen execution source was changed.

Evidence limits: no full checkpoint weight-byte hash, no raw GPU noise-byte hash, and no videos scanned. Weight identity uses original path/file metadata; noise uses recorded original seeds and frozen generator source. Free trajectories are not matched-input latency comparisons; non-significant p does not establish no effect or a user-approved noninferiority margin.

Primary artifacts: [final_audit.json](final_audit.json), [audit source](final_audit.py), [prospective statistics](prospective_statistics.md), [statistical results](../../../reports/idea2_closedloop200_20261025/statistics.json).
