# Full confirmation acceptance

This CPU-only, read-only entrypoint is for the frozen confirmation manifest at `results/ease_insights_20261103/vlm_local_search_v1/confirmation/`. It is pinned to manifest hash `80176d9f4b01e68abb022345bd6dc1a7bfe4e5aef5d69d13257801ca1eb6bdc0`, its current selection file SHA, the selected pair `jac_anchor/local_long`, and confirmation inits 39–43. It requires `selection.load_stage(root, "confirm")` to return exactly 600 valid receipts, 200 per planned condition, 200 cases, and no missing, invalid, or unexpected records. Any partial or mismatched input writes a non-PASS report.

For a complete matrix, it hashes the frozen source groups including all nine `local_runtime_sources`, checkpoint identity artifacts, protocol/design evidence, manifest, and selection evidence. It checks all 600 saved request rows with the reusable per-request validator, input/noise pairing, action lengths, and schedule/mask/history constraints. It retains `success: false` case outcomes without using success as an acceptance gate.

Service validation follows each receipt's server PID and the `confirm_shard00of01_complete.json` marker to the serving run. It checks source-bank identity, service protocol/mask, all 332 persisted gates (the six-condition gate whitelist remains intact even though only three conditions have confirmation episodes), the saved cross-service receipt, and each serving run's `gate_actions.npz` against the accepted Jac reference from full discovery at the frozen `atol=rtol=1e-5`. Gate calculations are read from saved CPU artifacts; no GPU gate is rerun. Only receipt-linked runs, the completion-marker run, and the Jac reference source are examined.

The report records source/input hashes, coverage counts, failure cases, gate/service evidence, and limitations. It does not make a new scientific selection or compute outcome statistics. The backend does not export its internal actual NFE counter; only the frozen exported time-node schedule is verified.

Run once the 600 confirmation cases and their service markers are complete, from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 policies/realtime-vla-flash/.venv/bin/python reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_confirmation/audit.py --root results/ease_insights_20261103/vlm_local_search_v1/confirmation --output reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_confirmation/audit.json
```

The command exits zero only for `status: "pass"`. It still writes `audit.json` when coverage is incomplete or a check fails.

The small CPU-only contract checks use in-memory confirmation records and do not open active confirmation cases:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 policies/realtime-vla-flash/.venv/bin/python reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_confirmation/test_audit_contract.py
```
