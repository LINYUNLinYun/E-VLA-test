"""Backfill only missing V13-Custom initial-state 0-9 chunks; skip report generation."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_standard_vlm_pruning as runner


def main():
    config = runner.load_config()
    jobs = [{"stage": "pilot", "condition": "V13-Custom", "suites": [suite]}
            for suite in runner.SUITES
            if runner.latest_complete("V13-Custom", suite, 0, 10) is None]
    runner.log_event({"event": "completion_backfill_started", "condition": "V13-Custom",
                      "jobs": jobs, "report_generation": False})
    runner.run_jobs(jobs, config)
    missing = [suite for suite in runner.SUITES
               if runner.latest_complete("V13-Custom", suite, 0, 10) is None]
    if missing:
        raise RuntimeError(f"Backfill still missing suites: {missing}")
    runner.log_event({"event": "completion_backfill_complete", "condition": "V13-Custom",
                      "init_range": [0, 10], "episodes": 400, "report_generation": False})
    audit = HERE / "audit_round_completion.py"
    return subprocess.run([str(runner.ROOT / ".venv/bin/python"), str(audit)], cwd=runner.ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
