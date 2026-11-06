# COMPLETED — control alignment pilot

The user's two requested routes have both been tested. No GPU work remains queued.

- 32/32 episodes: eight fixed cases × sync fresh / sync SM16 / async unaligned / async offset.
- Successes 8/8, 8/8, 6/8, 8/8 respectively. Both valid failures are retained.
- No prediction-horizon exhaustion. Offset uses actual control-step age, retaining R5 and H10.
- Three services exited normally with unchanged sources during execution; GPU3 lease released.
- 3,013 predictions, 5,383 applied actions; 48 bootstrap numerical checks all zero error.
- Report: `docs/experiments/idea2/idea2_control_alignment_report_20261026.md`.
- Summary: `analysis/summary.json`; full commands in each stage's `execution.json`.
- Post-run launcher-only environment-log whitelist change: measured launcher archived in `frozen_source/run_async.py`, identity in `source_archive.json`.

This is an eight-case exploratory pilot, not a hundred-case no-harm result.
The next suggested study is a fixed paired 100-case unaligned/offset comparison;
it has not been launched. Do not resume completed dispatchers or reuse their ports.
