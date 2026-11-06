#!/usr/bin/env python3
"""Run one Full18 suite's remaining official init chunks on a released GPU."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_standard_vlm_pruning as runner  # noqa: E402


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: launch_early_full_suite.py <condition> <libero_suite> <gpu_index>")
    condition, suite, gpu_text = sys.argv[1:]
    gpu = int(gpu_text)
    config = runner.load_config()
    if condition not in config["conditions"] or suite not in runner.SUITES or gpu not in range(4):
        raise SystemExit(f"invalid condition/suite/GPU assignment: {condition} / {suite} / {gpu}")
    free = runner.free_gpus(config)
    if gpu not in free:
        raise SystemExit(f"GPU {gpu} no longer meets the configured idle threshold; free GPUs: {free}")
    job = {"stage": "full", "condition": condition, "suites": [suite]}
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
