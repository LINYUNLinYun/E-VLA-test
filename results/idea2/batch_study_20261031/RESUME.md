# Active goal: two experimental stages, not yet completed

## AUTHORITATIVE LIVE STATE — one finite queue waiting for external GPUs

The dedicated B2/SM24 short test **passed** same-B gates and clean teardown,~18.1033observations/s (only2groups×4; notformal),lookaheadprepare~32ms peractuallookahead. B1/SM16 andB4/SM20 dedicatedsmokes were **rejected before CUDA admission**, because externalprocesses tookGPU1/0; B2's firstformal attempt then wasrejected onGPU3 aswell. These are resource-admission failures, not numericalfailures. Allthree rawfailures arein catalog. All manual unified sessions6558/2011/26830/49843 areterminal; do notpoll/restartthem.

**Only current launcher:** unifiedexec **75167**, actualPID **57822**, script `run_stage2_tail.py`; receipt `stage2_tail_execution.json`. Last verified `os.kill(pid,0)` succeeded and30-secondpolltimestamps advanced (poll37 at2026-10-31T18:49UTC). Atthatpoint all14taskswerepending, noGPUmodelchildrunning. FouractualforeignGPUprocesses were4082707/4082710/4082711/4090464; do nottouchthem. RevalidatecurrentPID/state ratherthanassumingthoseoccupanciespersist.

The queue is the **single GPU monitor and catalog writer**. It polls all4cards every30s, admits onlyno-compute-process/no-activelease/low-memory/low-utilization candidates, then the benchmark atomically rechecksgraphics+compute. It canrunatmost3independentcardjobs. Nevermanuallydispatchduplicatequeuedtasks. ParentPopenhandlesarepreserved; a timeoutisnotterminal. Itisboundedto14jobs and12hwait, withatmost3attemptsonactualadmission/interferencefailures; neverretriesvalidnumericalfailures. Everyattemptgetsanewname. Itmovesinvalidattemptstocatalogfailures. ProfilesareCPU-exportedautomaticallyaftercleanexit; exporthandles/statusesareinqueue receipt. `stage2_tail_preflight.json` validatedall14commands+dependencyDAGwithoutqueryingGPUsorlaunchingprocesses.

The14jobs coverB1SM16/B2SM24/B4SM20 dedicatedformal100groups, B2/B4SM16dedicatedablation, same-boardB1ordinary/serial100ABBAreversecontrol, B4preparedSM20then24same-boardreverse-orderrepeat, B1/B4longcontinuousGreenandmatchedserial/ordinary (2×100requests,validationonly), andB1/B4dedicatedonlineprofiles8requests. B1/B4didnotrunnewdedicatedsmokesdueadmission; theirformaljobsstillperformallfresh/same-schedule/inplace/resetnumericalgatesbeforeanytiming. Profile/longjobsdependonpassedformalparents. The queue appendsactualattempts to`study_catalog.json` underperformance/profiles/continuity. Resultnamesstart`stage2_tail_...`.

### Frozen source for pending jobs

FullCPUprotocol91passed(`stage2_dedicated_input_cpu_validation.json`). One additionalmetadata-onlyfix nowsets`gpu_work_started=True`justbeforeCUDAdeviceinitialization; historicalsuccessfulrunsaccidentallyleftthisFalse, so **neverinterpretoldFalseasnoGPUwork**. Oldactualexecutionisprovedbyidentity/Events/archives/receipts. Targeted26CPUchecksafterthatfixpassed: `stage2_gpu_started_metadata_cpu.json`; **its sourcehashmap iscurrent**. No modelmath orGraphmath changed. Do notedit`preparations/idea2/*.py` whilethequeue maylaunchmodels. Resultanalysishelpers/docsarenotinmodelfrozensourcesandcanbeedited. Do notwritecatalogmanuallywhilequeuelive.

### Completed report/evidence work while waiting

- `base_complete_analysis/summary.json`+`table.csv`: all17originalformalruns/23moderows, real18basecellscomplete, no unfinishedadoptedruns; `original_input_comparison.png/.svg` inspectedvisually.
- `base_complete_analysis/kernel_slowdown.csv`:282same-stage/name/grid/block/countmatchedkernelgroupsacrossactualB4traces. Encoderflash~15.38→31.55us,AErepresentativeGEMM~11.68→15.08us; VLMlargestGEMMs~1.014×/1.041×. AE's~121msqueuegapscannotbeexplainedbyauniform3.6×kernelslowdown.
- `base_completed_audit.json`: **21completedruns/profiles independentlyverified** (17performance+4profiles): realexits/sourcebeforeafter, releasedleases, actionsNPZfinite/fullBshape, allrecordsseeds/laneIDs, 2disjoint36-tensorKVslots, oldage1, frontierssorted/10steps, per-lanenumericalerrorsandactualthroughputdenominators, official24corereplayenvelopes. Thenfullmodel/D10identityverifiedagain. `audit_study.py` implementscurrentchecksandfinalcompletiongates; finalrunwithout`--completed-only`requiresqueuepassed, finalreport, `final_analysis/summary.json/table.csv`, reproductioncommands.
- `stage2_plan_chain_audit.json`: originalsecond-stageplanand6first-stageevidencefiles**all7frozenSHAverifiedunchanged**. B2/B4originalstrictnumericalfailuresstillmax0.00758684/0.00747871,tolmax1e-5/MAE1e-6; firstGreenabort-6retained.
- **Draft report nowexists**: `docs/experiments/idea2/idea2_batch_pipeline_two_stage_report_20261101.md`. Explicitlymarked初稿/pendingtail; containsbaseprepared/online tables, moduletimes, numerics, nsysproofandlimitations. Updatewithrealqueuedresultsbeforecallingitfinal. Do notclaimgoalcompletebasedonthisdraft.
- `summarize_stage1.py` hasnewCSVfieldsforrequest/groupcounts, prepareplacement, Eventcompletiontimes, freshlayerfraction, prefixdone-at-response, bootstrapprepare/model/Graphsetup, steadyandsetupmemory. OldbaseCSVwascreatedbeforetheseadditionalcolumns; finalaggregationwillusethem. Grouplength8throughputexcludesbootstrap/banksetup (rawgroupsrecordthosecosts),includesdrain; allGPUprofilertimingseparate.

### Required finish after queue

Inspectactualqueue exits/results/exports; ifanyneedsinspection, diagnose/preservefailures. SmallSM20-vs24differencecouldbeatie; reportboundedrepeat, notglobaloptimal. MatchsameGPUflagswhenclaimingimprovements; cross-GPUrowsmustremainqualified. ConsideroneadditionalmatchedcallerB2controlifneededtoattributeinput-streamgainonadifferentassignedGPU; donotblindlyrepeatallmatrix.

Generate`final_analysis`CSV/JSON+finalplotsandcontinuitytables, actualreprocommandsMD; finalize reportwithCPU/P50/P95, AE/prefill/allworkEventtimes, memory, KVfrequency/frontiers, numericallimits, reset/inplace/recaptureproof. Createclearreal `.nsys-rep`downloadaliasesunder`nsys/idea2_experiment/batch_study_20261101/` (notcreatedyet), plusSCPcommands(hostsmc-gpu,localDesktop/nsys). Runfull`audit_study`andinspectownCUDAprocesses/leasesinactiveafterqueueexit. Onlythenmarkgoalcomplete. NoexactB1equivalence/closedloopsuccessclaimforB>1; mixedKVscheduleisapprovedapproximationandmustnotbecalledfreshserial-equivalent.

## CURRENT GOAL CONTINUATION — online base matrix complete; input-stream investigation underway

The active goal explicitly resumed experiment execution after the preparation turn. GPUs0/1/3 became genuinely free and were admitted via atomic leases; GPU2 had an external process and was untouched. All previous jobs/profiles in the sections below are **terminal**.

New completed formal online runs (100groups each, clean exit/source/lease):

- `stage2_b1_sm16_online_full`: strict passed,11.237826/s,CPUmean88.974ms,lookaheadprepare75.138ms.
- `stage2_b2_sm16_online_full`: same-B diagnostic,12.849887/s.
- `stage2_b4_sm16_online_full`: diagnostic,16.200877/s,CPU246.888ms,lookahead211.373ms.
- `stage2_b2_sm24_online_full`: diagnostic,14.468979/s,CPU138.214ms.
- `stage2_b4_sm20_online_full`: diagnostic,17.405731/s,CPU229.797ms.

The B1/B2/B4 ×prepared/online ×serial/ordinary/SM16 **18-cell base matrix is now complete**. `stage2_before_input_stream_analysis/` is an interim CSV/JSON saved before the finalSM20online run ended; its base-matrix assertion is true, but lastSM20online was still pending then. `summarize_stage1.py` now emits real base-cell coverage and extended stage/frontier/latency metrics, never treats extra quota jobs as filling missingbasecells. Catalog includes all17 formal runs.

Prepared B4SM20 formal finished20.705868/s vs24 20.471826 (~1.14%); tiny difference still needs sameGPU3 reverse-process-order repeat before choosing a unique best. Prepared ordinary pipeline has real AE queue gaps (earlier evidencebelow).

**New direct input-wait evidence:** `stage2_profile_b4_sm16_online_trace.nsys-rep` and `_analysis/summary.json` were captured/exported successfully:8 complete ranges each preprocess/encoder/vlm/AE,114480kernels,actual16/112. AE×VLM1064.452181ms total8, AE×nextEncoder **0**. Next-prepareframe1 longest `cudaStreamSynchronize`111.773705→296.078801ms (~184.305ms), immediately after previousAE finishes296.066245. `stage2_preparation_wait_evidence.json` records all7 windows (meanlongwait181.978ms). This is a real model input-submission wait, not evidence that CPUtransforms consume222ms or that all default streams always synchronize Green work.

`stage2_profile_b4_sm20_prepared_trace.nsys-rep` also completed/exported:114384kernels,24completecoreenvelopes,actual20/108; realAE×VLM1110.424647ms, AE×nextEncoder158.517786ms (7windows). Catalog includes bothnewprofiles.

### Input-placement code and current live jobs

Written-before-test amendment: `stage2_input_stream_amendment.md`. Added default-off benchmark `--prepare-stream {caller,prefix,dedicated}`. `caller` preserves old behavior. `prepare_in_stream()` runs unchanged prepare on selected stream, records and retains a per-Prepared readyEvent, and joins it into caller stream so existing Graph input-ready chain stayscorrect. `dedicated` creates one ordinary inputstream, explicitly drained in close_engine (also exceptionbeforeencode); Prefix reuses existingstream. No model math, Graph math/captures, Pool/Lease, KV allocation, weights/dtype/noise/D/H changed. `run_one.py` forwardsflag. New source frozen at **`stage2_dedicated_input_cpu_validation.json` (91/91 passed, CUDAuninitialized)**. Earlier Prefix-only CPU90 record retained.

Prefix-placement short tests allpassed numerical/lifetime gates but did **not** recover prepared-input throughput: `stage2_input_stream_b1_sm16_smoke`, B2SM24smoke, B4SM20smoke (2groups×4requests). These are short tests, not formal results. Prefix upload waits currentVLM and splits CPUwork aroundthatwait. No formalPrefix-placement runs started.

Three synthetic probes allcleanlycompleted (`input_stream_probe.py`, `input_graph_stream_probe.py`, `input_copy_tail_probe.py`, outputsamebaseprefixedstage2). Default/dedicated H2D remained~0.04ms while sleepingGreenwork wasunfinished, even withGraph andtailD2D; Prefix waiteditsown~11ms. These **rule out generic eagerGreen/Graph/tailD2D-alone explanations**. Do not claim confirmedcopy-engine/global-default-stream root cause. Real-modelspecificwaiting chain remainsknown; dedicatedplacementisthe nextminimaltest.

**Only current model jobs** (revalidate handles/PIDs; do not restart on observationtimeout):

- Unified exec **6558**, `stage2_input_dedicated_b1_sm16_smoke`,GPU1, modelPID57317 atstart.
- Unified exec **2011**, `stage2_input_dedicated_b2_sm24_smoke`,GPU3, modelPID57329 atstart.
- Unified exec **26830**, `stage2_input_dedicated_b4_sm20_smoke`,GPU0 (inspectreceiptforPID).

All use2groups,length4,warm2 and retainfull numerical gates. If valid/useful, formally measure dedicatedinput placement100groups atB1SM16/B2SM24/B4SM20 on the sameGPU as correspondingcallermodelcontrols; considerSM16ablation forB2/B4onlyifneeded. Ifnotuseful, retainnegativeevidence andstopunboundedinput-streamtuning. Needcandidateactualnsys (online) to verifywhetherAE×nextEncoderreturns, notjustNVTXoverlap. KeepB>1diagnosticrestriction andquota-dependentfreshKVfractionexplicit (~B4prepared16 .656,20 .589,24 .528).

### Required remaining delivery

1. Finish actualinputplacementsmokes; formalboundedcomparison + representativeactualtrace ifpromising. Alltimingwithoutprofiler.
2. SameGPU3 reverse-orderB4SM20then24 formalrepeat (originalorder24then20); tiny1.1%maybetie. B1strictkeyresult same-board/reversecontrolrepeat ifneeded.
3. Longcontinuous100-request sequences for selectedcandidate + matchedserial/ordinarycontrol ifreportingspeedupthere; canuse`--pairs2 --length100` ascontinuityvalidation(notformal100pairscapacity).
4. FinalCSV/report/plots/reprocommands/representativetracedownloadpaths; finalauditofactualreceipts/exits/leases, strictvsdiagnosticgates, everylaneerrors, inputs/noiseidentity,two-slot/inplace/reset/recapture/memory. Reportunder`docs/experiments/idea2/idea2_batch_pipeline_two_stage_report_20261101.md` notyetcreated. Largergoalstillactive, incomplete; donotmarkcompleteuntilartifactsandauditdone.

No newagentswereused. Do noteditpreparationssourceswhileanymodeljobisrunning. Resultanalysishelpers/docsarenotfrozensourceandcanbeupdated.

## LATEST USER STEERING — prepare code and plan first; no new GPU launches this turn

The current user asked: “你先准备好代码，因为现在没有卡空闲，你先plan一下，然后准备好代码，方便后续实验”. This preparation turn did not poll/claim GPUs, launch models, or start a background waiter. Do not treat the older dispatch instructions below as a request to launch work immediately.

New handoff: `docs/experiments/idea2/idea2_batch_pipeline_preparation_20261101.md` and `results/idea2/batch_preparation_20261101/`. Reused the existing batch backend/runtime/benchmark. Only `batch_plan.py` and its tests changed: the planner now supports the benchmark's existing explicit `--diagnostic-batch-drift` switch for B>1, preserving strict B1 jobs, fixed tolerances and per-job numerical-policy metadata. Both generated plans are unstarted: strict B1 (12 candidates), B2/B4 same-B diagnostic (24 candidates). CPU **87/87 passed**, CUDA uninitialized; all36 commands parse and match the new source hashes. No model/runtime math changed.

The three previous live jobs listed below were inspected: `stage2_b4_sm20_prepared_full`, `stage2_b2_full_online_full`, `stage2_b4_full_online_full` all have exit_code0, source_unchanged=true, terminal PIDs, and diagnostic_passed results (strict B1-equivalence gate still false). They were already complete; none was restarted. Later experiment execution should consume their actual receipts/results before dispatching missing jobs. Source hashes have now changed for the planner; do not confuse historical frozen runs with the new preparation snapshot. The larger two-stage experiment/report goal remains unfinished.

## MOST RECENT DISPATCH — online matrix and new real nsys traces underway

The quota24 formal runs both completed cleanly: B2 **19.50726 samples/s** (sameGPU2 vsSM16 16.97286), B4 **20.47183** (sameGPU3 vsSM16 19.72122). B4 SM32 short control is slower~19.6593, VLM169.07ms/AE110.03ms. B4 SM20 is a distinct actual20/108 split; short screen~20.8596 withVLM154.55/AE156.03ms. `stage2_sm20_rationale.md` documents the evidence before this extra point. Do not call the SM20 short screen a formal improvement yet.

**Current live jobs** (source remains frozen at stage2_cpu_validation84):

- Unified exec **39471**, `stage2_b4_sm20_prepared_full`,GPU3,modelPID47654:100groups same-board SM20 vs16/24.
- Unified exec **10603**, `stage2_b2_full_online_full`,GPU2,modelPID47674:100paired serial/ordinary with actual input preparation.
- Unified exec **1061**, `stage2_b4_full_online_full`,GPU0,modelPID47673:100paired serial/ordinary online-prepare.
- CPU export/analysis exec **81206**: `export_trace stage2_profile_b4_pipeline_prepared`.

Update:81206 has now **completed successfully**. Ordinary B4 profile has24 official complete core envelopes/114384kernels. Real AE×VLM kernel intersection138.392271ms total over8requests(~17.30ms/request); AE×nextEncoder98.78247ms over7windows(~14.11ms/window); current-Encoder intersection0. AE mean GPU envelope186.676ms but actual kernel-interval union65.200ms; serial envelope51.850/busy51.239ms. VLM pipeline envelope137.377/busy135.933ms vsserial130.188/130.002ms. This directly shows long AE queue gaps behind near-continuous VLM. **Profiler overhead inflates CPU replay launch to~22ms; do not infer normal Host bottleneck from that.** Formal unprofiled ae_submit_ms is sub-ms. Profile timing is diagnostic only.

**New actual study traces now exist**, not just reference:

- `stage2_profile_b4_serial_prepared_trace.nsys-rep`, successful capture and `..._analysis/summary.json`: official24 complete core replay envelopes,8 each encoder/vlm/AE,114384real kernels; AE×VLM=0,AE×Encoder=0 as expected. `export_trace.py` matched exact first/last all-GPU-operation endpoints against officialnvtx_gpu_proj_trace. Prepared has no timedpreprocess.
- `stage2_profile_b4_pipeline_prepared_trace.nsys-rep`, successful capture; CPU export/analysis currently running via81206. The analysis computes actual kernel interval union intersections, per-frame current/next Encoder pairing, actual Green eligibleSM attribution, CUDA API durations and kernel name/grid aggregates.
- GPU profile exporter helpers live inresults,not model source: `trace_analysis.py` and `export_trace.py`. Exporter invocation `taskset -c48-63 python3 -m results.idea2.batch_study_20261031.export_trace NAME`. It refuses failed executions and existing analysisdirs, so inspect partialdirs before recovery, never blindly overwrite. It writesSQLite,officialprojectionCSV,summaryJSON,kernel_groupsCSV.

Completed first online case: `stage2_b1_full_online_full/`:passed100pairs; serial13.98998samples/s(meanCPU71.4681ms),ordinary15.54504(mean64.3176ms),vsameB+11.12%. These are recorded raw input replay with actual prepare, not real camera/controlage. It is already in study_catalog.

Next prioritize the missing **online Green16 baselines B1/B2/B4** (each100groups), and tunedGreen24 forB2 / selected20or24 forB4 online. Prepared base matrix9cells is complete; online serial+ordinaryB1done,B2/B4running. After a card frees, capture **B4 Green16 prepared**, **best quota prepared**, and **best quota online** separately (profile jobtemplate `profile_b4_green_sm16_prepared/online` plus `--ae-sm-min N --diagnostic-batch-drift`). Never use profiler timing asperformance.

Then validate chosen long continuous sequence and repeat the small B4 quota improvement in reverse process order/sameGPU if needed; generate final two-stageCSV/report/commands/traces and audit actual exits/leases/source/gates. Goal remains active and incomplete. Most earlier session IDs below havefinished; rely on this section and actual receipt/PID checks.

---

## LATEST LIVE STATE — quota testing after successful formal baseline runs

Source is frozen at `stage2_cpu_validation.json` (84 passed). The three smoke jobs listed further below all **finished successfully**. Full Green cleanup is now hardware-verified with model runs; no more edits to preparations while jobs are live.

Completed formal results (all100 groups, paired full-GPU runs100 AB/BA pairs;800requests per mode):

| Path | Serial samples/s | Pipeline/Green samples/s | Notes |
|---|---:|---:|---|
| Stage1 B1 full prepared |14.8553|16.5660|strict,passed|
| Stage2 B1 SM16 prepared |—|18.07445|strict,clean exit+lease released|
| Stage2 B2 full prepared |16.47428|17.41751|diagnostic_passed;failed B1 equivalence retained|
| Stage2 B4 full prepared |18.73231|19.35426|same-B diagnostic|
| Stage2 B2 SM16 prepared |—|16.97286|same-B diagnostic,slower than ordinary pipeline|
| Stage2 B4 SM16 prepared |—|19.72122|same-B diagnostic|

SM24 short screens (not formal performance): B2~19.343 samples/s,AE86.96ms,VLM80.64ms; B4~20.761,AE134.65ms,VLM159.33ms. Actual Green splits8/120,16/112,24/104,32/96,48/80,64/64 were measured; see `stage2_sm_granularity_probe/`. Evidence-based amendment `stage2_quota_amendment.md` chooses24 as a midpoint because B2 AE16 is still exposed, rather than blindly increasing to64 or reducing to8.

**Current live jobs, revalidate actual handles/PIDs:**

- Session **92350**, name `stage2_b2_sm24_prepared_full`,GPU2,model PID43554: formal24-vs16 same-board comparison.
- Session **5603**, name `stage2_b4_sm24_prepared_full`,GPU3,model PID43705: formal24-vs16 same-board comparison.
- Session **84086**, name `stage2_b4_sm32_screen`,GPU0,model PID43573: neighboring32-SM short screen.

All use `run_one.py` and have `<name>.execution.json`/`.console.log`/`<name>/result.json`. These are the only current model jobs. Sessions23979,17577,50477,20854,13375 and the original watchers/runners are finished.

**Remaining work is substantial; do not mark complete:**

1. Inspect quota24 formal results and32 screen, retain negative results; repeat key candidate if needed, unchanged lead3 first. B1 strict best current18.074, B>1 remains diagnostic rather than B1-equivalent.
2. Complete **online input preparation** matrix B1/B2/B4 serial+ordinary andSM16 (6jobs) and test tuned24 forB2/B4 online if promising. No online-mode formal measurements have yet run. Use free GPUs0/2/3;GPU1 stays external. Old plan IDs `performance_b{B}_paired_online`, `performance_b{B}_green_sm16_online`; diagnostic flag only forB>1. Keep >=100pairs/groups. Add actual started formal runs to `study_catalog.json`.
3. Collect **new actual nsys traces**, currently NONE for this batch study. Use profile prepared-plan IDs via `run_one.py`, e.g. `profile_b4_serial_prepared`, `profile_b4_pipeline_prepared`, `profile_b4_green_sm16_prepared` with `--ae-sm-min24 --diagnostic-batch-drift`; add an online representative to inspect CPU/H2D waits. The wrapper redirects trace to `<name>_trace.nsys-rep`, flagsnode:host-only:nvtx-live preserve true replay ranges and individual kernels. Performance and traces separate.
4. Export SQLite, run `trace_analysis.py` for real kernel interval union/intersection and actual Green SM attribution. It has been tested only on the **old reference** runtime_ranges20261026 trace:AE×VLM269.479ms,AE×Encoder0,104936kernels,actual16/112. `reference_trace_analyzer_check/` is **not a new study trace**. Current `trace_analysis.py` correlates exact4 stage labels to same-thread CUDA API and kernel correlation IDs; nested scheduler ranges ignored. Official `nsys stats --report nvtx_gpu_proj_trace` can separately verify8 full replay envelopes/stage; prepared has no timed preprocess.
5. Test a longer continuous sequence for selected best configs (wrapper now accepts `--pairs`, `--length`,`--warmup` overrides, e.g.2×100requests; these have performance_eligiblefalse due<100pairs and are separate continuity validation).
6. Aggregate CSV and final two-stage report underdocs with best tested config, errors/failed gates, latency-throughput tradeoff, scope. Offline recorded replay does not prove closed-loop quality/camera freshness. B1 full timing already excludes preparation and uses Graph; never compare against native103ms as matching single-frame latency.

`study_catalog.json` lists explicit adopted/ongoing formal jobs and failures. `summarize_stage1.py --catalog ... --output NEW_DIR` accepts strict and explicitly qualified same-B diagnostic results, never unfinished/failed jobs. It emits per-mode rows, same-B speedups, cross-B speedups with eligibility/GPU fields, and paired timing bootstrap. Original default automatic-stage1 execution only contains failedB2 and should not be used to infer current completion.

---

## CURRENT UPDATE — 2026-11-01, supersedes the original waiting instructions below

Cards became free after the other study completed 800/800. Initial B1 smoke passed. The original automatic stage1 runner **terminated** at the genuine B2 cross-batch numerical failure; do not restart it. Initial exec session15888 and stage1 session94145 are finished.

First-stage actual evidence:

- `stage1_original_b1_full_prepared/`: **passed**, 100 AB/BA pairs,800 requests/arm. Serial14.8553 samples/s, ordinary pipeline16.5660, speedup1.11516, timing-repeat95%CI[1.11389,1.11639]. CPUCPU means67.3059/60.3542ms. AE Event25.5266→54.8028ms; VLM30.9112→33.6423ms. Source unchanged, lease released.
- `stage1_smoke_b2_paired_prepared/` and `stage1_original_b4_numeric/`: **real strict B1-equivalence failures**, retained unchanged. Same-B Graph vs eager outputs are exactly equal, but normalized cross-B maxima0.00758684/0.00747871. Do not rerun these failures to hunt for passing samples.
- `stage1_batch_shape_diagnosis/`: passed diagnostic; inputs/noise/masks/positions bitwise equal, Encoder embeddings already differ, identical-KV AE velocity and time MLP are shape sensitive. Max two logical KV objects. Evidence points to batch-shaped numerical computation rather than data mixing or graph scheduling.
- `stage1_original_b1_sm16_prepared/`:100 groups measured then SIGABRT (`context is destroyed`) during Tensor destruction. **Not adopted as a successful performance run.** Original active lease receipt preserved. Own stale registry lease was safely recovered only after PID33847 was terminal and actual GPU2 empty; see `.recovery.json`.

Second-stage plan was written **after** these results: `docs/experiments/idea2/idea2_batch_pipeline_stage2_plan_20261101.md`, frozen with six evidence hashes in `stage2_plan_evidence.json`.

Implemented fixes (no model math or precision changes):

1. `BatchProgressiveRuntime.close()` now returns its GreenContextPair instead of destroying streams before caller tensors. Benchmark frees runtime/inputs/noise/model refs, clears cuBLAS workspaces, waits/empties allocator cache, then closes Green resources before releasing GPU lease. Constructor failure defers context destruction to process exit to avoid traceback-held Tensor destruction against dead streams.
2. Explicit default-off `--diagnostic-batch-drift` adds original **same-B eager oracle**. Strict cross-B1 failure is still recorded; `numerical_gates_passed=false`, `performance_eligible=false`, `diagnostic_only=true` if it fails. Same-B scheduling gates retain MAE1e-6/max1e-5; successful diagnostic execution is `diagnostic_passed` and has separate `scheduling_performance_eligible`. Never call it B1-equivalent or deployment ready. Default strict mode still fails/stops exactly as before.

Verification: `stage2_cpu_validation.json`: **84/84 passed**, CUDA not initialized, source hashes unchanged. `stage2_green_cleanup_probe/`: real GPU record_stream/Graph/cuBLAS lifetime probe **passed**, primary context still works after closing Green resources. This is synthetic, not model timing.

### Three live real-model second-stage smoke jobs (revalidate before relying on handles)

- Unified exec **88345**, wrapper name `stage2_cleanup_b1_sm16_smoke`, GPU2, model PID37647 at start: strict B1 Green16 smoke verifies real teardown.
- Unified exec **11550**, name `stage2_diagnostic_b2_smoke`, GPU3, model PID37611: diagnostic same-B B2 ordinary serial/pipeline smoke.
- Unified exec **3464**, name `stage2_diagnostic_b4_smoke`, GPU0, model PID37629: diagnostic same-B B4 ordinary serial/pipeline smoke.

All were started through `run_one.py`; each has `<name>.execution.json`, `<name>.console.log`, and `<name>/result.json`. The module source is frozen while these jobs run. Never mutate preparation/runtime sources during a model job, because snapshots are checked.

### Next actions

1. Inspect these actual process exits/results; fix only concrete failures in new directories, keep strict failures. If all pass, run formal100-pair B2/B4 ordinary prepared comparisons with explicit diagnostic flag, and a clean B1 Green16 repeat. Default B1 original formal already passed, do not repeat solely to inflate data.
2. Complete prepared vs online-prepare replay conditions and SM16 baseline/targeted quota32 (then64 only if evidence merits), keeping lead3 first. Use available exclusively idle GPUs with shared lease; cards0/2/3 were released, GPU1 stays external. `run_one.py` supports `--job <prepared-plan-id> --gpu-uuid ... --gpu-index ... --name NEW --diagnostic-batch-drift --ae-sm-min N --lead N --reverse`. For B1 **omit diagnostic flag**.
3. Profile representative serial/ordinary/Green candidates separately using prepared profile job IDs and `run_one.py`. Runtime graph labels are instance-local real replay envelopes. Analyze actual kernel unions/intersections via CUDA API correlation and Green SM attribution, not Event spans. Prior working reference: `results/idea2/runtime_ranges_20261026/audit.py`; adapt its range-count assumptions (prepared has no timed preprocess, and scheduler parent NVTX ranges remain).
4. Produce a two-stage report/CSV/commands/traces with best tested configurations and limitations. **Offline batch replay does not prove closed-loop success or fresh camera observation age; B2/B4 remain non-equivalent to B1 at the original strict tolerance.** Do not mark goal complete before both-stage measurement/analysis/artifacts are actually finished and audited.

`summarize_stage1.py` currently reads only original automatic stage1 execution; it will not include manual `run_one` jobs or diagnostic-qualified results. Update/replace the analysis input explicitly, preserving separate strict versus diagnostic eligibility. Numerical and lifecycle failures must remain in the report.

---

User objective: wait for another session to release a GPU; produce first-stage batch/pipeline results; analyze them and write an improvement plan; implement and validate a second stage; deliver both stages by the next day if resources permit. Keep the goal active until both stages and the report are actually verified.

## Current authoritative execution

- Live initial launcher: unified exec session **15888**, PID **29451** at creation. Revalidate the handle before assuming it is still running.
- Launcher: `results/idea2/batch_study_20261031/wait_then_run.py`.
- Receipt: `initial_execution.json`; output: `stage1_smoke_b1_paired/`; console: `stage1_smoke_b1_paired.console.log`.
- The live launcher polls actual NVML processes, memory/utilization and the shared lease every 30 seconds. It starts **only the initial B1 smoke**, once a card is genuinely free. The benchmark atomically claims the shared lease and rechecks graphics/compute occupancy. Never restart this launcher merely because a tool observation times out.
- A dependent **first-stage runner is now live**: unified exec session **94145**, PID **29874**. Script `run_stage1.py`, receipt `stage1_execution.json`. It waits for the initial B1 smoke to pass, then automatically runs the remaining B2/B4/full-GPU/Green smoke cases and all 12 profiler-free performance conditions from the prepared matrix, one model child at a time. **Do not manually launch duplicate first-stage jobs while this runner is live.** It stops on any failure with `needs_inspection`; numerical failures are never retried. Profile capture and stage2 remain deliberate analysis steps after stage1 evidence.
- At admission it binds CPU affinity matching the chosen GPU and uses the already validated OpenPI environment.
- No batch model experiment has run yet as of this file's creation. CPU preparation was 81/81 passed, CUDA uninitialized, bridge compile passed; `candidate_filter_test` also passed.

## External work confirmed live

All four physical cards are occupied. GPU0/2/3 are being used by the other session's running `results/idea1_v3_4_E2_adjacency_extrapolate_200/` study; its status at 2026-10-31 15:28 UTC was 63/800 new episodes completed, active GPUs [0,2,3]. GPU1 has an external process as well. Actual compute PIDs (host namespace) were 3292049, 3292050, 3292051 and 4130181. Do not terminate them or treat 0% utilization as release. Cross-thread messaging tools are currently unavailable; no message was sent through another channel.

## After the initial launcher finishes

1. Inspect `initial_execution.json`, the actual child exit and the full B1 `result.json`/console. Technical launch failures and numerical failures are different; preserve them. Fix concrete bugs minimally and run a new output directory, never overwrite a valid or failed result.
2. Complete the first-stage matrix B1/B2/B4: full-GPU serial/ordinary pipeline, separate Green SM16/lead3; prepared and online-prepare boundaries separate. First numeric smoke, then profiler-free 100 AB/BA pairs (length8, warmup20, verify4). Green comparisons use separate processes, clearly labelled. Use the shared lease and actually idle cards only. Current prepared commands live in `results/idea2/batch_preparation_20261031/experiment_plan/commands.md`; UUID placeholders must be replaced.
3. First-stage analysis must inspect same-B serial/pipeline throughput, batch capacity, CPU latency and input preparation, GPU Event stage spans, old KV request age/frontiers, memory and exact per-lane normalized/decoded gates. Run selected nsys captures separately with real runtime labels and actual kernel overlap analysis. Do not confuse Event/NVTX overlap with busy kernel intersections.
4. Write the **second-stage plan after observing stage1 evidence**, then make the targeted improvement and validate it. Candidate knobs are SM16/32/64, lead0/1/3/6 or a concrete preparation/submission bottleneck; select by evidence instead of blindly running a full grid. If batch fails strict numerical gates, diagnose first and retain the failure; never silently relax tolerances.
5. Repeat key winning/negative results under reversed order where needed, retain CSV/JSON/action archives/representative traces. Deliver a concise technical report under `docs/experiments/idea2/` with both stages, commands, scope/limitations, and real results. No success-rate claim from offline input replay. Mark the goal complete only after both stages and the report are verified.

## Invariants

Full pi05_libero checkpoint, D10/H10, original precision/FP32 Euler schedule and explicit per-record noise. Fixed independent batch lanes; whole-batch reset only; two full batched KV slots, unchanged Pool/Lease/epoch/generation/Event protocol. No draft/verify, model pruning or hidden third full KV. Mixed KV output compares to the same observed frontier schedule, while fresh batch serial compares against independent B1 eager oracle. MAE<=1e-6/max<=1e-5 per lane, normalized and decoded; failed gates are not performance eligible.

All 20 numerical/runtime source hashes from `batch_preparation_20261031/cpu_validation.json` matched at this goal turn's initial check. The wait launcher lives in the results directory and changes no model or scheduling source.
