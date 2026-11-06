"""Continue stage1 after the live initial smoke, preserving every result.

One model subprocess at a time. Stops for inspection after any real failure;
never retries a numerical failure or modifies runtime code while it is running.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from results.idea2.batch_study_20261031.wait_then_run import (
    AFFINITIES, HERE, ROOT, candidates, save, snapshot,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def frozen_sources():
    baseline = json.loads((ROOT / "results/idea2/batch_preparation_20261031/cpu_validation.json").read_text())["source_sha256"]
    actual = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in baseline}
    if actual != baseline:
        raise RuntimeError("Prepared numerical/runtime source changed; inspect before running stage1")
    return actual


def main():
    execution = HERE / "stage1_execution.json"
    if execution.exists():
        raise RuntimeError("Existing stage1 execution must be inspected; do not restart blindly")
    state = dict(status="waiting_for_initial_smoke", pid=os.getpid(), start_utc=now(),
                 source_sha256=frozen_sources(), jobs=[],
                 next_stage="Analyze first-stage evidence, write stage2 plan, then improve and validate")
    save(execution, state)
    try:
        while True:
            initial = json.loads((HERE / "initial_execution.json").read_text())
            if initial["status"] in ("passed", "failed", "launcher_failed", "wait_deadline"):
                break
            if not alive(initial["pid"]):
                raise RuntimeError("Initial launcher PID is gone; inspect its actual child and receipt")
            time.sleep(15)
        if initial["status"] != "passed" or initial.get("exit_code") != 0:
            raise RuntimeError("Initial smoke did not pass; preserve failure and inspect before more jobs")
        initial_result = json.loads(Path(initial["output"]).joinpath("result.json").read_text())
        if initial_result.get("status") != "passed" or not initial_result.get("numerical_gates_passed"):
            raise RuntimeError("Initial B1 benchmark has no passing full numerical gate")
        state["initial_smoke"] = dict(output=initial["output"], verified=True)
        plan = json.loads((ROOT / "results/idea2/batch_preparation_20261031/experiment_plan/plan.json").read_text())
        ids = ["smoke_b2_paired_prepared", "smoke_b2_green_sm16_prepared",
               "smoke_b4_paired_prepared", "smoke_b4_green_sm16_prepared",
               "smoke_b1_green_sm16_prepared"]
        ids += [row["id"] for row in plan["jobs"] if row["stage"] == "performance"]
        jobs = {row["id"]: row for row in plan["jobs"]}
        preferred = initial["selected_gpu"]["uuid"]
        base_affinity = os.sched_getaffinity(0)
        for key in ids:
            frozen_sources()
            job = jobs[key]
            output = HERE / ("stage1_" + key)
            if output.exists():
                raise RuntimeError(f"Output already exists; inspect instead of rerunning: {output}")
            row = dict(id=key, status="waiting_for_gpu", output=str(output), start_utc=now())
            state["jobs"].append(row)
            state["status"] = "running_stage1"
            save(execution, state)
            while True:
                rows = snapshot()
                free = candidates(rows)
                free.sort(key=lambda item: item["uuid"] != preferred)
                if free:
                    device = free[0]
                    break
                row.update(last_wait_utc=now(), devices=rows)
                save(execution, state)
                time.sleep(30)
            argv = list(job["argv"])
            argv[argv.index("--output") + 1] = str(output)
            argv[argv.index("--gpu-uuid") + 1] = device["uuid"]
            environment = dict(os.environ, **job["environment"])
            environment["CUDA_VISIBLE_DEVICES"] = device["uuid"]
            affinity = set(AFFINITIES.get(device["index"], base_affinity)) & base_affinity
            if affinity:
                os.sched_setaffinity(0, affinity)
            row.update(status="running", gpu=device, command=argv,
                       cpu_affinity=sorted(os.sched_getaffinity(0)))
            log = HERE / ("stage1_" + key + ".console.log")
            with log.open("x") as console:
                child = subprocess.Popen(argv, cwd=ROOT, env=environment,
                                         stdout=console, stderr=subprocess.STDOUT)
                row["child_pid"] = child.pid
                save(execution, state)
                print("STARTED", key, device["uuid"], child.pid, flush=True)
                child.wait()
            os.sched_setaffinity(0, base_affinity)
            row.update(exit_code=child.returncode, end_utc=now())
            result_file = output / "result.json"
            if result_file.is_file():
                result = json.loads(result_file.read_text())
                row.update(benchmark_status=result.get("status"),
                           numerical_gates_passed=result.get("numerical_gates_passed"),
                           summary=result.get("summary"), error=result.get("error"))
            passed = child.returncode == 0 and row.get("benchmark_status") == "passed"
            row["status"] = "passed" if passed else "failed"
            save(execution, state)
            print("FINISHED", key, row["status"], flush=True)
            if not passed:
                raise RuntimeError(f"Stage1 job {key} failed; inspect before continuing")
            # A just-exited process can leave the previous utilization sample.
            # Let it settle once; normal unavailable-GPU polling remains 30s.
            time.sleep(2)
        state.update(status="stage1_performance_complete", end_utc=now(),
                     remaining="Select representative nsys captures, analyze, write stage2 plan and execute improvements")
        save(execution, state)
    except BaseException as error:
        state.update(status="needs_inspection", error=f"{type(error).__name__}: {error}", end_utc=now())
        save(execution, state)
        raise


if __name__ == "__main__":
    main()
