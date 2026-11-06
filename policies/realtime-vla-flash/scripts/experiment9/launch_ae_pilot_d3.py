#!/usr/bin/env python3
"""Use GPU 0 for the remaining D3 action-expert pilot while VLM pilot wraps up."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_standard_vlm_pruning as runner  # noqa: E402


def main() -> int:
    config = runner.load_config()
    job = {"stage": "ae_pilot", "condition": "AE-D3", "suites": list(runner.SUITES), "denoise_steps": 3}
    runner.log_event({"event": "fixed_gpu_ae_pilot_started", "gpu": 0, "condition": "AE-D3"})
    try:
        result = runner.run_condition_job(job, 0, config)
    except Exception as exc:
        runner.log_event({"event": "job_error", "gpu": 0, "condition": "AE-D3",
                          "suites": job["suites"], "error": str(exc)})
        return 1
    runner.log_event({"event": "job_complete", **result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
