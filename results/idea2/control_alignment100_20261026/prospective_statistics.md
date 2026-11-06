# Expanded control-alignment study, fixed before new GPU cases

Primary comparison: `control_steps` minus `off`, with identical continuous control and SM16 lead3 D10/H10/R5 + next Encoder model scheduling. The only control change remains the pilot's prediction-prefix offset. No new model/control implementation is introduced.

Fixed universe: four LIBERO suites; task0–9/init0,1, plus init2 for even tasks. 25 cases per suite, 100 per condition. Sixteen compatible pilot cells (eight paired cases) count once; all their failures are retained. New cells: 184. The pilot was observed before expansion, so this remains exploratory. Report a separate sensitivity result excluding the eight pilot pairs.

Same case, both conditions run on the same physical GPU. Order rotates by suite/task/init and reverses on GPU2. One GPU scheduling Host and one environment process per service. At most four distinct tasks per model lifetime. GPU3 and GPU2 are eligible only after process and lease checks; GPU0/1 excluded. A single dispatcher checks unavailable GPU2 about every30 seconds while work remains. Unique ports/output directories; no valid failure retries.

Primary outcome: success fraction and paired gained/lost cases. Report two-sided exact McNemar and 40-task ratio-weighted cluster bootstrap95% interval (10000 draws, seed20261026); episode bootstrap is sensitivity. Within-suite numbers are descriptive. Do not interpret nonsignificance as noninferiority; no noninferiority margin was set. Excluding pilot leaves unequal task cluster sizes, retained intact in the resampling.

Keep the existing synchronous fresh/SM16 200-case evidence. Select the same100 identities as a historical descriptive reference, with protocol/source validation. Do not pool synchronous and asynchronous latency or call their difference a pure GPU scheduling effect.

Check fixed input/seed bootstrap normalized and decoded MAE<=1e-6 and max<=1e-5, actual action indices and values, R5/H10 bounds, slot/resource protocols, graph recaptures and episode reset. Horizon exhaustion is a valid scheduling failure included in the denominator. Preserve errors and original paths.

Report success, task steps, episode work time, RPC mean/P50/P95, first-action input age, skipped-prefix distribution, residual index delay, horizon exhaustion, and GPU Event envelope windows. Performance runs do not use nsys. No claim of new Kernel-level overlap measurement.
