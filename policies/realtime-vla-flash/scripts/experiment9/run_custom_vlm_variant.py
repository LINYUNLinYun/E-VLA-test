#!/usr/bin/env python3
"""Run the registered V13-Custom standard-suite evaluation on idle GPUs.

The first GPU status check can be delayed to stay aligned with another
five-minute monitor interval. A GPU released by one of this runner's jobs is
reused immediately; otherwise idle-GPU checks are spaced 300 seconds apart.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_standard_vlm_pruning as runner  # noqa: E402


CONDITION = "V13-Custom"
POLL_SECONDS = 300


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-check-in-seconds", type=float, default=0.0,
        help="Delay the initial GPU status check to honor an existing five-minute cadence.",
    )
    parser.add_argument(
        "--confirmed-free-gpu", action="append", type=int, default=[],
        help="Immediately assign a GPU just released by a logged experiment job, without repolling it.",
    )
    args = parser.parse_args()
    if args.first_check_in_seconds < 0:
        raise SystemExit("--first-check-in-seconds must be nonnegative")
    if any(gpu < 0 for gpu in args.confirmed_free_gpu) or len(set(args.confirmed_free_gpu)) != len(args.confirmed_free_gpu):
        raise SystemExit("--confirmed-free-gpu values must be unique nonnegative GPU indices")

    config = runner.load_config()
    if CONDITION not in config["conditions"]:
        raise RuntimeError(f"{CONDITION} is not registered in experiment9_config.json")
    pending = [
        {"stage": "pilot", "condition": CONDITION, "suites": [suite]}
        for suite in runner.SUITES
        if runner.latest_complete(CONDITION, suite, 0, 10) is None
    ] + [
        {"stage": "full", "condition": CONDITION, "suites": [suite]}
        for suite in runner.SUITES
        if any(runner.latest_complete(CONDITION, suite, start, start + 10) is None
               for start in range(10, 50, 10))
    ]
    if not pending:
        runner.log_event({"event": "custom_variant_already_complete", "condition": CONDITION})
        return 0

    runner.log_event({
        "event": "custom_variant_monitor_started",
        "condition": CONDITION,
        "vlm_layer_pattern": config["conditions"][CONDITION],
        "queued_suites": [job["suites"][0] for job in pending],
        "first_check_in_seconds": args.first_check_in_seconds,
        "gpu_check_interval_seconds": POLL_SECONDS,
    })
    next_gpu_poll = time.monotonic() + args.first_check_in_seconds
    active: dict[concurrent.futures.Future, tuple[int, dict]] = {}
    retry_count: dict[tuple[str, str], int] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for gpu in args.confirmed_free_gpu:
            if not pending:
                break
            job = pending.pop(0)
            future = executor.submit(runner.run_condition_job, job, gpu, config)
            active[future] = (gpu, job)
            runner.log_event({
                "event": "custom_variant_job_assigned_on_release",
                "gpu": gpu,
                "condition": CONDITION,
                "suite": job["suites"][0],
                "release_source": "confirmed prior job_complete/server_stopped event",
            })

        while pending or active:
            if pending and time.monotonic() >= next_gpu_poll:
                used = {gpu for gpu, _ in active.values()}
                free = [gpu for gpu in runner.free_gpus(config) if gpu not in used]
                while pending and free:
                    job = pending.pop(0)
                    gpu = free.pop(0)
                    future = executor.submit(runner.run_condition_job, job, gpu, config)
                    active[future] = (gpu, job)
                    runner.log_event({
                        "event": "custom_variant_job_assigned",
                        "gpu": gpu,
                        "condition": CONDITION,
                        "suite": job["suites"][0],
                    })
                next_gpu_poll = time.monotonic() + POLL_SECONDS
                if not free and pending:
                    runner.log_event({
                        "event": "custom_variant_gpu_busy",
                        "pending_suites": [job["suites"][0] for job in pending],
                        "next_check_seconds": POLL_SECONDS,
                    })

            if active:
                timeout = min(60.0, max(0.0, next_gpu_poll - time.monotonic())) if pending else 60.0
                done, _ = concurrent.futures.wait(
                    active, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    gpu, job = active.pop(future)
                    try:
                        result = future.result()
                        runner.log_event({"event": "job_complete", **result})
                        if pending:
                            next_job = pending.pop(0)
                            next_future = executor.submit(runner.run_condition_job, next_job, gpu, config)
                            active[next_future] = (gpu, next_job)
                            runner.log_event({
                                "event": "custom_variant_job_assigned_on_release",
                                "gpu": gpu,
                                "condition": CONDITION,
                                "suite": next_job["suites"][0],
                            })
                    except Exception as exc:
                        key = (job["stage"], job["suites"][0])
                        retry_count[key] = retry_count.get(key, 0) + 1
                        runner.log_event({
                            "event": "job_error",
                            "gpu": gpu,
                            "condition": CONDITION,
                            "suites": job["suites"],
                            "retry": retry_count[key],
                            "error": str(exc),
                        })
                        if retry_count[key] > 1:
                            raise
                        pending.insert(0, job)
            elif pending:
                time.sleep(min(60.0, max(0.0, next_gpu_poll - time.monotonic())))

    runner.log_event({"event": "custom_variant_complete", "condition": CONDITION})
    report_script = ROOT / "scripts" / "experiment9" / "analyze_standard_vlm_pruning.py"
    if report_script.is_file():
        subprocess_result = subprocess.run(
            [str(runner.CLIENT_PYTHON), str(report_script), "--phase", "full"],
            cwd=ROOT,
        )
        if subprocess_result.returncode != 0:
            raise RuntimeError("Full Experiment 9 report generation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
