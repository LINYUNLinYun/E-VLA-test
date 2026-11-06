# Full discovery acceptance

This directory contains a CPU-only, read-only acceptance entrypoint for the frozen discovery matrix. Run it once after all six conditions have committed all 120 receipts. It first requires the original strict `selection.load_stage(root, "discovery")` result to be complete with exactly 720 valid records, zero missing/invalid/unexpected cases, and exactly 120 cases per condition. It hashes the manifest source groups, including the nine `local_runtime_sources`, then audits every saved inference row with the reusable per-request validator from `first_resource_session_cases/audit.py`.

The service check follows episode identity server PIDs and the condition completion marker to the run directories actually used. It validates their persisted service protocol, source-bank evidence, all 332 saved gate records, mask checks, and saved `cross_service.json`; it compares each actual `gate_actions.npz` with the Jac reference at the frozen `atol=rtol=1e-5`. It never starts a model, recomputes gates on a GPU, or changes a case. A missing/invalid/incomplete matrix returns a non-PASS status and writes its report only under this directory.

The audit records `success: false` outcomes without using success as a gate. It verifies the exported D2/D10 time-node schedule, not the backend's internal actual NFE counter, which is not exported. If cases were served by multiple service runs, each run must be linked by an episode identity PID and pass the persisted gate checks. Each condition also needs a 120-case completion marker; when the marker names a run, its result receipt must match it.

After the matrix is complete, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 policies/realtime-vla-flash/.venv/bin/python reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.py --root results/ease_insights_20261103/vlm_local_search_v1/discovery --output reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.json
```

The command exits zero only for `status: "pass"`. It still writes `audit.json` when the matrix is incomplete or a check fails.

The small CPU-only contract checks do not open discovery artifacts and may be run while episodes are active:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 policies/realtime-vla-flash/.venv/bin/python reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/test_audit_contract.py
```
