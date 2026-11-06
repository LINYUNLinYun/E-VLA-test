#!/usr/bin/env python3
"""Start the independent Full18/Object expansion on a verified idle GPU."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_standard_vlm_pruning as runner  # noqa: E402


def main() -> int:
    config = runner.load_config()
    job = {"stage": "full", "condition": "Full18", "suites": ["libero_object"]}
    gpu = 2
    runner.log_event({"event": "early_full_job_started", "gpu": gpu,
                      "condition": job["condition"], "suites": job["suites"]})
    try:
        result = runner.run_condition_job(job, gpu, config)
    except Exception as exc:
        runner.log_event({"event": "job_error", "gpu": gpu, "condition": job["condition"],
                          "suites": job["suites"], "error": str(exc)})
        return 1
    runner.log_event({"event": "job_complete", **result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
