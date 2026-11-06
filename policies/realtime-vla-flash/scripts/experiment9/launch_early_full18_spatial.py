#!/usr/bin/env python3
"""Start Full18/Spatial expansion on the GPU released by the AE-D5 pilot."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_standard_vlm_pruning as runner  # noqa: E402


def main() -> int:
    config = runner.load_config()
    job = {"stage": "full", "condition": "Full18", "suites": ["libero_spatial"]}
    gpu = 3
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
