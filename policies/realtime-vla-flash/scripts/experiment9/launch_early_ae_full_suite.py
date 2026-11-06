#!/usr/bin/env python3
"""Run one already-qualified action-expert full-suite job on an idle GPU."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_standard_vlm_pruning as runner  # noqa: E402


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: launch_early_ae_full_suite.py <AE-D7|AE-D5|AE-D3> <suite> <gpu_index>")
    condition, suite, gpu_text = sys.argv[1:]
    gpu = int(gpu_text)
    config = runner.load_config()
    denoise_steps = int(condition.removeprefix("AE-D")) if condition.startswith("AE-D") else -1
    allowed_steps = {int(value) for value in config["action_expert_followup"]["denoise_steps"]}
    if denoise_steps not in allowed_steps or suite not in runner.SUITES or gpu < 0:
        raise SystemExit(f"invalid condition/suite/GPU assignment: {condition} / {suite} / {gpu}")
    free = runner.free_gpus(config)
    if gpu not in free:
        raise SystemExit(f"GPU {gpu} no longer meets the configured idle threshold; free GPUs: {free}")
    job = {"stage": "ae_full", "condition": condition, "suites": [suite],
           "denoise_steps": denoise_steps}
    runner.log_event({"event": "early_ae_full_job_started", "gpu": gpu,
                      "condition": condition, "suites": [suite], "denoise_steps": denoise_steps})
    try:
        result = runner.run_condition_job(job, gpu, config)
    except Exception as exc:
        runner.log_event({"event": "job_error", "gpu": gpu, "condition": condition,
                          "suites": [suite], "stage": "ae_full", "error": str(exc)})
        return 1
    runner.log_event({"event": "job_complete", **result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
