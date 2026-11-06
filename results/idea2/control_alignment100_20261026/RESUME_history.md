# ACTIVE — fixed100 control-alignment expansion

User explicitly requested a larger evaluation after the eight-case pilot.

- Objective: complete 100 paired scenarios for async unaligned and async offset, 200 cells total.
- Reuse16 pilot cells exactly once; execute184 new cells. Do not retry valid failures or drop horizon-exhaustion failures.
- Freeze pilot model/client/server; only the new runner expands cases/devices/order.
- Single dispatcher `dispatch.py` owns GPU leases, prefers GPU3 and checks GPU2 every30s if pending work needs capacity. GPU0/1 excluded. Respect other processes and leases.
- Dispatcher receipt: `dispatch_execution.json`. Progress and unique cases: `normalized_cases.json`.
- There are10 task groups, at most4 different tasks/model lifetime. Same case's two modes use same GPU.
- Pilot documentation and previous `control_alignment_20261026` completion remain historical; do not relaunch those dispatchers.
- Synchronous historical reference: `synchronous_reference.json`, same100 identities selected from prior200 per arm; no new synchronous-control experiment needed unless compatibility fails.
- Statistics and protocol fixed in `manifest.json` and `prospective_statistics.md` before new cases.
- Do not finalize at launch or after a partial result. Finish expanded statistics, report, GPU releases and receipt.

## Current continuation

- Initial dispatcher ended with one zero-episode GPU2 model-load OOM caused by an external process racing idle admission; no valid episode was retried.
- Original GPU3 chunk00 finished16 new cells normally. Together with pilot16, first32 cells are valid.
- `initial_dispatch_execution.json` preserves the first attempt. `dispatch_resume.py` now owns the queue and writes the canonical `dispatch_execution.json`.
- Current unified exec session: 94275 (initial session44485 has exited1). Do not launch another dispatcher.
- GPU2 is disabled until an explicitly authorized `GPU2_EXPLICIT_HANDOFF.json` exists; the resource coordinator could not guarantee its ownership. GPU3 continues alone. A question about another exclusively available GPU was sent to the user; keep working onGPU3 while awaiting an optional reply.
- `resource_revision.json` records this scheduling-only amendment. No model/control/scientific configuration changed.
- Final analysis command: `python3 -m results.idea2.control_alignment100_20261026.analyze`; `--partial` checks completed pairs without claiming final statistics.
- New report: `docs/experiments/idea2/idea2_control_alignment100_report_20261026.md`.

## Latest resource steering — supersedes GPU2 handoff question

- User answered: “有人占卡那就等一下或者去别的空卡跑”. GPU0 is repeatedly idle and lease inactive; now eligible along withGPU3. GPU1/2 remain occupied and are excluded.
- `runner_alt.py` differs only in eligible GPU0 mapping (UUID GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1, EGL3, CPU32–47) and reversed condition order on the second GPU. It imports the same frozen pilot server/client.
- `resource_gpu0_revision.json` records authorization and unchanged scientific grid.
- Sent SIGINT ONLY to dispatcher_resume PID518367. Its existing exception cleanup waits for GPU3 chunk01 to finish normally, then merges valid cells/releases; the model/episode subprocess is untouched.
- Unified exec session8343 waits for that old dispatcher to exit and then runs `dispatch_gpu0.py` automatically; do NOT start another dispatcher. Its log is `dispatch_gpu0.console.log`; exit receipt `dispatch_gpu0_exit.json`.
- New dispatcher archives previous receipt as `dispatch_before_gpu0.json`, continues only pending whole pairs, writes canonical `dispatch_execution.json` and `normalized_cases.json`, and uses GPUs3/0 after lease/process checks. Old session94275 should exit1 due intentional orchestration KeyboardInterrupt; this is not an episode failure.

## Latest observed live state

- Old dispatcher session94275 has exited130 (intentional SIGINT); all its GPU3 episodes finished normally.
- New dispatcher PID522810 is running under unified session8343. GPUs0 and3 both admitted successfully, running chunks03 and02 respectively. At07:13UTC,48/200 cells valid (24 pairs), previous chunks00 and01 complete.
- Historic synchronous compatibility check: all77 source hashes match; checkpoint sizes/mtime match, no full weight rehash. Receipt `synchronous_source_compatibility.json`.
- Continue monitoring canonical dispatcher; do not restart or edit runner_alt.py/manifest/prospective files while services run.

- At07:24UTC,85/200 cells valid; GPUs0/3 healthy, completed pilot and chunks00/01/03, chunk02 (GPU3) and chunk04 (GPU0) active.41 completed bootstrap pairs audited, max error0.
- A terminal-analysis watcher is running: it waits for dispatcher `passed`, runs `python3 -m ...analyze`, writes `analysis.console.log` and `statistics_execution.json`. Do not duplicate final analysis while it is running. The tool session ID is recorded in the conversation immediately after this note.

## Latest recovery — foreign process appeared on GPU0

- At07:51UTC, chunk09 GPU0 guard detected foreign CUDA processPID3759859 (578MiB). Our runner stopped its own client/model; server exit0 and sources unchanged. Seven completed chunk09 cells remain valid; one cell interrupted, eight unstarted.
- GPU3 chunk08 continues normally; dispatcher_gpu0 waits for it to finish and retains all its results. Canonical status temporarily says technical_failure until continuation takes over.
- `pending_final_cells.json` fixes the nine remaining cells. `runner_finish.py` can run ONLY explicit cell keys, so it cannot rerun a valid partner.
- Unified session71528 runs `finish_pending.py`: waits for old dispatcher to exit, verifies exactly nine pending cells, then finishes onGPU3. Do not start another dispatcher/runner. Logs `finish_pending.console.log`, `final9_gpu3.console.log`.
- One pair, Spatial/task9/init1, will have valid off result onGPU0 and resumed offset result onGPU3. The pre-result amendment `resource_final_revision.json` and `cross_gpu_exception.json` authorize this documented exception, require bootstrap numerical agreement, and add a same-GPU99-pair sensitivity. The100-cell primary still includes every valid failure.
- `analyze.py` now reports full100, excluding-pilot92, and same-GPU99 sensitivities; only one explicitly listed cross-GPU pair is allowed.
- Original terminal watcher session19817 exited1 on the dispatcher stop, before running statistics. A new terminal watcher was launched after this recovery (session ID in conversation); it waits through the expected old dispatcher failure and then runs final analysis once.
- No model, control client, GPU runtime, weights, D/H/R, or case universe changed.
