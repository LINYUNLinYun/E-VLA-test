# COMPLETED continuous closed-loop study

FINAL: primary300/300 complete. successes fresh75,SM16noEncoder72,fullEncoder84. Stats and full audit passed,53928predictions/69763actions/100bootstrapgroups max0. Both own GPU leases released and coordinator notified. Report docs/experiments/idea2/idea2_continuous_encoder_report_20261026.md; completion.json is authoritative. Do not restart jobs. Historical instructions below describe the now completed execution.

Latest user explicitly asks to validate full lead3 + mixed old/new KV D10 AE + next Encoder closed-loop success. We are actively running fixed100 scenes per3arms=300 episodes. No further parameter tuning. Complete grid, statistics, final audit, docs report and GPU release before final.

## Primary protocol
Three arms: fresh18_no_encoder / sm16_no_encoder / sm16_encoder. Full pi05_libero BF16 body B1 D10 H10 R5, weights unchanged; AE16/Prefix112 lead3 for SM16 arms. All arms perform identical backend.prepare ahead; only next Vision graph replay differs in Encoder ablation. No third KV, one GPU Host, original Pool/Lease/epoch/generation/Event.

Client main thread owns MuJoCo/EGL, paced native20Hz, exactly first R5 per consumed H10 chunk. Background network thread continuously receives real later control-step frames. Single latest candidate mailbox, select only at actual R5 application boundary. Prediction request_id != target_chunk_id. Original model_seed(suite,task,init,consumed_chunk_i) is fixed for all candidates of target i; selected chunk i must use seed(i). Superseded/expired candidates never execute; no H10 tail/noop/midchunk replacement. Missing result waits+records controlgap (sim pauses), not extra action. Pending ahead must be same raw/Prepared on next request; terminal ahead can be unused, drain then reset. No fixed250ms lag. This is new online protocol, cannot pool prior synchronous/250ms results.

## Fixed grid
Each suite object/spatial/goal/10: task0..9 init0/1, even tasks0/2/4/6/8 additionally init2 =>25suite,100arm,300cells. Seed7, settling10, original limits280/220/300/520, nativeOSC20Hz. No task failures retried. Partial success counts must use common paired scenarios.

## Active orchestration
R=results/idea2/continuous_encoder_20261026
P=reports/idea2_continuous_encoder_20261026
Plan docs/experiments/idea2/idea2_continuous_encoder_plan_20261026.md
Final intended report docs/experiments/idea2/idea2_continuous_encoder_report_20261026.md (not yet written)
Read R/dispatch_execution.json, P/progress.json, R/normalized_cases.json for authoritative state. Main dispatcher exec session39493; actual pid in state. It owns unique chunks0..9, each task_ids[t,10+t,20+t,30+t], max4task/model lifetime. Two workers GPU2/3. No extra GPU scans; watch own processes/status. Source core frozen run.py/policy.py/server.py/client.py + original77files + oldonline policy/wire/client,84hashes. No source changes during runs; postprocessing/dispatch helpers excluded and safe to fix if needed.

Both gates gate_gpu2_v3/gate_gpu3_v3 passed, runner/serverexit0/source unchanged. Unified sessions13146/85401 already collected exit0. Primary gates6 count toward grid; main --completed-runs gates deduplicates them. Main dispatcher uses unique19400+2*chunk+shard ports and new dirs gpu2_chunkN/gpu3_chunkN. Any child failure stops new scheduling and status needs_review; do not rerun valid episodes automatically.

Gate independent audit passed: R/gate_audit/audit.json/summary.md. Six episodes,967predictions,1236actual actions; bootstrap inputs/seeds/decoded/normalized max/MAE0 acrossarms; protocol/targetnoise/R5/real-frame chain/mailbox1/worker stop/two slots/zero recap/drain-reset passed. SM16Encoder355 next jobs,355 AEalive at receive,353 AEalive afterprepare and353 positive Event envelope intersections mean11.127ms. These are Event windows, not Kernel busy overlap. Exact same frozen GPU graph previously has standalone nsys true two-part overlap evidence (old report), keep claims separate.

## CPU finalizers
Statistics watcher session17777: R/finish_statistics.py -> on dispatch passed executes statistics.py --input R/normalized_cases.json --output P/statistics.json --csv-dir P. Writes R/statistics_execution.json; complete300 required. Fixed prospective statistics R/prospective_statistics.md,27 CPU tests passed. Only primary fullEncoder vs fresh; encoder/noencoder ablation secondary. Wilson, paired b/c exact McNemar,40task ratio-weighted bootstrap10000 seed20261026 (2/3inits), no noninferiority margin, p>.05 != no harm.

Agent /root/model_audit owns audit.py/runtime summaries. Its CPU final watcher session10641 taskset64-95,CUDAVISIBLEempty,4h bounded, waits dispatch passed then once full scan300. Output R/full_audit (audit.json/runtime_summary.json/csv etc) and R/final_audit.log/final_audit_watcher.json. Do not redo gate audit or scan running data. Summary includes perrequest Event windows, capture/response/application ages, used oldKVsource age (bootstrap/null fresh excludes old), actual actorHz vs predictionHz, discarded candidates, gap/deadline, prep/cold cost. Check summary completion and audit pending before final. Agent /root/layer_audit owns statistics.py, completed CPU27checks, idle. Agent /root/bench owns client.py/cpu_selftest.py,14checks after boundaryselection+socketabort fixes, frozen SHA0915cbbdb172e30c8b0169655d7ae002ba9caa0e301aae14f396c71752032f11. No agents should start GPU or change core.

Root plot helper R/plot_results.py ready; only run after P/statistics complete300, saves P/quality_and_paired_effect.png/svg. Inspect PNG once then final report. R/historical_synchronous_reference.json stores old same100 initialcases fresh96/100, quality-only descriptive different-protocol reference; no old timing pooled or claim independent confirmation. New300 main remains same-online-protocol three arms.

## GPU leases and cooperation
GPU2 UUID83fdf473... =>EGL1 CPU0-15, shard0.
GPU3 UUID555cde36... =>EGL0 CPU96-111, shard1.
Both explicitly re-leased under results/idea1_v3_3/external_gpu_leases.json owner 'user-authorized idea2 continuous closedloop100', idle C/G checked, R/leases.json. GPU0/1 excluded from use. Dispatcher releases each own lease as its work ends; notify coordinator promptly then, no GPUs held for CPU summary.
Coordinator Codex thread01a16d3b-4889-7e81-a0a3-71c41b600625, tools.mcp__codex_tui__send_message_to_thread max1000bytes. It knows both are running this authorized study. No foreign process kills. Release only own matching owner.

## Preserved preliminary attempts (not primary)
Initial gate_gpu2/gate_gpu3 empty outputs + console logs: root dict duplicate env keyword TypeError, zero episodes/no GPU loaded. Fixed in run.py dict unpack overrides.
Then v2 actual6gate episodes passed but noencoder only raw-next vs encoder prepared-next, preparation confound. Preserved all6 (including valid failures), excluded solely protocol mismatch, R/preparation_protocol_revision.json; not outcome-based. V3 reran6 corrected all-ahead-prepare, source frozen, counted primary. No unreported episode technical retries.
Old online_encoder_20261025 250ms onechunklag gate failed, frozen separate. Prior closedloop200_20261025 completed800 reference; not resumed. No current goal created (ordinary request), old get_goal refers already completed task.

## Tools and communication
exec_command default sandbox bwrap fails; use sandbox_permissions require_escalated(auto-approved). apply_patch normal works. Sessions are exec_command handles via write_stdin, not functions.wait. Keep concise commentary ~60sec, sleep <=50sec. User dislikes permission, none needed for authorized300; persist to final not just progress. User may steer meanwhile; answer briefly then continue unless canceled.
