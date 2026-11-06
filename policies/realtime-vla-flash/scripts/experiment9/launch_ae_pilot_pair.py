#!/usr/bin/env python3
"""Use currently idle GPUs for the D7/D5 action-expert pilot pair."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_standard_vlm_pruning as runner  # noqa: E402


def main() -> int:
    config = runner.load_config()
    assignments = [
        ({"stage": "ae_pilot", "condition": "AE-D7", "suites": list(runner.SUITES), "denoise_steps": 7}, 1),
        ({"stage": "ae_pilot", "condition": "AE-D5", "suites": list(runner.SUITES), "denoise_steps": 5}, 3),
    ]
    runner.log_event({
        "event": "fixed_gpu_ae_pilot_pair_started",
        "assignments": [{"gpu": gpu, "condition": job["condition"]} for job, gpu in assignments],
    })
    failures = 0
    with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
        futures = {pool.submit(runner.run_condition_job, job, gpu, config): (job, gpu)
                   for job, gpu in assignments}
        for future in as_completed(futures):
            job, gpu = futures[future]
            try:
                runner.log_event({"event": "job_complete", **future.result()})
            except Exception as exc:
                failures += 1
                runner.log_event({"event": "job_error", "gpu": gpu,
                                  "condition": job["condition"], "suites": job["suites"],
                                  "error": str(exc)})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
