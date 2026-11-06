#!/usr/bin/env python3
"""Run resumable, paired standard-suite LIBERO evaluations for Experiment 9."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("experiment9_config.json")
RESULTS = ROOT / "results" / "experiment9_standard_vlm_pruning_20261004"
BASE_DIR = Path("/root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted/base")
DRAFT_SOURCE = Path("/root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted/draft_10/draft_triton.pkl")
OPENPI_DATA = Path("/root/nfs/embodied/benchmark/shared/checkpoints/openpi")
SERVER_PYTHON = ROOT / ".venv/bin/python"
CLIENT_PYTHON = ROOT / "examples/libero/.venv/bin/python"
LIBERO_ROOT = Path("/root/nfs/embodied/benchmark/LIBERO")
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
NEW_CONDITIONS = ("Skip-L2", "Skip-L3", "Skip-L4", "Skip-L5", "Skip-L2-5", "V13-Custom")
OLD_CONDITIONS = ("V16", "V14", "V12", "V10-Mid")
POLL_SECONDS = 300
NEXT_GPU_POLL: float | None = None


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def log_event(event: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    event = {"timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **event}
    with (RESULTS / "runner_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def free_gpus(config: dict[str, Any]) -> list[int]:
    limits = config["gpu_idle_threshold"]
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    available = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or "[N/A]" in fields:
            continue
        gpu, used_mib, utilization = map(int, fields)
        if used_mib <= int(limits["max_used_memory_mib"]) and utilization <= int(limits["max_utilization_percent"]):
            available.append(gpu)
    return available


def gpu_cache(gpu: int) -> Path:
    cache = RESULTS / "prompt_cache" / f"gpu{gpu}"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "draft_triton.pkl"
    if destination.exists():
        if destination.stat().st_size != DRAFT_SOURCE.stat().st_size:
            raise RuntimeError(f"Cached draft artifact has the wrong size: {destination}")
    else:
        os.link(DRAFT_SOURCE, destination)
    return cache


def chunk_complete(run_dir: Path, *, expected_start: int, expected_end: int, condition: str) -> bool:
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        rows = json.loads((run_dir / "episode_log.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("episode_index_start") != expected_start or manifest.get("episode_index_end") != expected_end:
        return False
    expected = {(task_id, episode) for task_id in range(10) for episode in range(expected_start, expected_end)}
    observed = [(int(row["task_id"]), int(row["episode_idx"])) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        return False
    expected_pattern = load_config()["conditions"].get(condition, "vlm_full18")
    expected_denoise = int(condition.removeprefix("AE-D")) if condition.startswith("AE-D") else 10
    for row in rows:
        telemetry_rel = row.get("telemetry_path")
        infer_rel = row.get("infer_path")
        video_rel = row.get("video_path")
        if not telemetry_rel or not infer_rel or not video_rel:
            return False
        episode_dir = run_dir / telemetry_rel
        if not episode_dir.is_file():
            return False
        telemetry = [json.loads(line) for line in episode_dir.read_text(encoding="utf-8").splitlines() if line]
        if len(telemetry) != int(row["executed_action_count"]) + 1:
            return False
        if int(row["frame_count"]) != len(telemetry):
            return False
        if not (run_dir / video_rel).is_file():
            return False
        infer_path = run_dir / infer_rel
        if not infer_path.is_file():
            return False
        inference_rows = [json.loads(line) for line in infer_path.read_text(encoding="utf-8").splitlines() if line]
        if len(inference_rows) != int(row["infer_record_count"]) or not inference_rows:
            return False
        for inference in inference_rows:
            metadata = inference.get("server_experiment", {})
            if (metadata.get("vlm_layer_pattern") != expected_pattern
                    or metadata.get("action_layer_pattern") != "full18"
                    or int(metadata.get("denoise_steps", -1)) != expected_denoise
                    or int(metadata.get("ablation_noise_seed", -1)) != 12345
                    or not metadata.get("noise_sha256")):
                return False
    return True


def latest_complete(condition: str, suite: str, start: int, end: int) -> Path | None:
    prefix = f"exp9_*_{condition.replace('-', '_')}_{suite}_i{start:02d}_{end:02d}_a*"
    for path in sorted(RESULTS.glob(prefix), reverse=True):
        if chunk_complete(path, expected_start=start, expected_end=end, condition=condition):
            return path
    return None


def run_chunk(
    *, gpu: int, port: int, stage: str, condition: str, suite: str, start: int, end: int,
    attempt: int, config: dict[str, Any],
) -> None:
    run_name = f"exp9_{stage}_{condition.replace('-', '_')}_{suite}_i{start:02d}_{end:02d}_a{attempt:02d}"
    run_dir = RESULTS / run_name
    log_dir = RESULTS / "client_logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "LIBERO_CONFIG_PATH": str(ROOT / ".libero"),
        "PYTHONPATH": f"{LIBERO_ROOT}:{ROOT}",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "CUDA_VISIBLE_DEVICES": str(gpu),
    })
    command = [
        str(CLIENT_PYTHON), "scripts/spec/spec_client_libero.py",
        "--host", "127.0.0.1", "--port", str(port),
        "--task-suite-name", suite, "--task", "0-9",
        "--num-trials-per-task", str(end - start),
        "--episode-index-start", str(start), "--episode-index-end", str(end),
        "--num-steps-wait", str(config["warmup_steps"]),
        "--replan-steps", str(config["replan_steps"]), "--seed", str(config["seed"]),
        "--record-semantic-telemetry",
        "--video-out-path", str(RESULTS), "--run-name", run_name,
    ]
    log_event({"event": "client_start", "gpu": gpu, "condition": condition, "suite": suite,
               "init_range": [start, end], "attempt": attempt, "run_dir": str(run_dir)})
    with (log_dir / "client.log").open("w", encoding="utf-8") as output:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError(f"Client exited {process.returncode}; see {log_dir / 'client.log'}")
    if not chunk_complete(run_dir, expected_start=start, expected_end=end, condition=condition):
        raise RuntimeError(f"Client exited without a complete, aligned chunk: {run_dir}")
    log_event({"event": "client_complete", "gpu": gpu, "condition": condition,
               "suite": suite, "init_range": [start, end], "run_dir": str(run_dir)})


def run_condition_job(job: dict[str, Any], gpu: int, config: dict[str, Any]) -> dict[str, Any]:
    condition = str(job["condition"])
    stage = str(job["stage"])
    suites = list(job["suites"])
    patterns = config["conditions"]
    layer_pattern = str(patterns.get(condition, "vlm_full18"))
    denoise_steps = int(job.get("denoise_steps", config["denoise_steps"]))
    port = 18000 + gpu * 100 + (os.getpid() % 80)
    cache = gpu_cache(gpu)
    session_id = f"{stage}_{condition.replace('-', '_')}_gpu{gpu}_{int(time.time())}"
    server_log = RESULTS / "server_logs" / f"{session_id}.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OPENPI_DATA_HOME": str(OPENPI_DATA),
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
    })
    server_command = [
        str(SERVER_PYTHON), "scripts/spec/spec_serve_policy.py",
        "--config", "pi0_libero", "--base-triton-path", str(BASE_DIR),
        "--draft-triton-path", str(cache), "--backend", "triton",
        "--max-exec-steps", str(config["replan_steps"]), "--force-full-each-round",
        "--vlm-layer-pattern", layer_pattern,
        "--action-layer-pattern", str(config["action_layer_pattern"]),
        "--denoise-steps", str(denoise_steps),
        "--ablation-noise-seed", str(config["noise_seed"]), "--port", str(port),
    ]
    log_event({"event": "server_start", "gpu": gpu, "condition": condition,
               "suites": suites, "server_log": str(server_log), "port": port})
    with server_log.open("w", encoding="utf-8") as output:
        server = subprocess.Popen(server_command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    try:
        ready = False
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited with code {server.returncode}; see {server_log}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    ready = True
                    break
            except OSError:
                time.sleep(2)
        if not ready:
            raise TimeoutError(f"Server did not become ready; see {server_log}")

        stage_chunks = ((0, 10),) if stage.endswith("pilot") else tuple((start, start + 10) for start in range(10, 50, 10))
        for suite in suites:
            for start, end in stage_chunks:
                if latest_complete(condition, suite, start, end):
                    continue
                for attempt in range(1, 3):
                    run_name_prefix = f"exp9_{stage}_{condition.replace('-', '_')}_{suite}_i{start:02d}_{end:02d}_a"
                    existing_attempts = [
                        path.name for path in RESULTS.glob(run_name_prefix + "*")
                        if path.name[len(run_name_prefix):].isdigit()
                    ]
                    next_attempt = max((int(name[len(run_name_prefix):]) for name in existing_attempts), default=0) + 1
                    try:
                        run_chunk(
                            gpu=gpu, port=port, stage=stage, condition=condition, suite=suite,
                            start=start, end=end, attempt=next_attempt, config=config,
                        )
                        break
                    except Exception as exc:
                        log_event({"event": "chunk_error", "gpu": gpu, "condition": condition,
                                   "suite": suite, "init_range": [start, end],
                                   "attempt": next_attempt, "error": str(exc)})
                        if attempt == 2:
                            raise
        return {"condition": condition, "stage": stage, "suites": suites, "gpu": gpu, "status": "complete"}
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
        log_event({"event": "server_stopped", "gpu": gpu, "condition": condition,
                   "returncode": server.returncode, "server_log": str(server_log)})


def enqueue_pilot(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"stage": "pilot", "condition": name, "suites": list(SUITES)}
        for name in config["conditions"]
        if any(latest_complete(name, suite, 0, 10) is None for suite in SUITES)
    ]


def pilot_rate(condition: str, suite: str) -> float | None:
    successes = total = 0
    stage = "ae_pilot" if condition.startswith("AE-") else "pilot"
    for path in RESULTS.glob(f"exp9_{stage}_{condition.replace('-', '_')}_{suite}_i00_10_a*"):
        if not chunk_complete(path, expected_start=0, expected_end=10, condition=condition):
            continue
        rows = json.loads((path / "episode_log.json").read_text(encoding="utf-8"))
        successes = sum(bool(row["success"]) for row in rows)
        total = len(rows)
        break
    return successes / total if total else None


def enqueue_full(config: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = list(config["conditions"])
    queue: list[tuple[str, str]] = []
    queue.extend(("Full18", suite) for suite in SUITES)
    promising = [
        (condition, suite)
        for condition in conditions if condition != "Full18"
        for suite in SUITES
        if (pilot_rate(condition, suite) or 0.0) >= 0.90
    ]
    if ("V12", "libero_object") not in promising:
        promising.append(("V12", "libero_object"))
    queue.extend(sorted(promising, key=lambda pair: (pair[0] not in NEW_CONDITIONS, pair[1] != "libero_object", pair)))
    seen = set(queue)
    queue.extend(
        (condition, suite)
        for condition in (*NEW_CONDITIONS, *OLD_CONDITIONS)
        for suite in SUITES
        if (condition, suite) not in seen
    )
    return [
        {"stage": "full", "condition": condition, "suites": [suite]}
        for condition, suite in queue
        if any(latest_complete(condition, suite, start, start + 10) is None for start in range(10, 50, 10))
    ]


def enqueue_ae_full(config: dict[str, Any], ae_config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for steps in ae_config["denoise_steps"]:
        condition = f"AE-D{steps}"
        for suite in SUITES:
            candidate = pilot_rate(condition, suite)
            baseline = pilot_rate("Full18", suite)
            if candidate is None or baseline is None:
                continue
            if candidate >= float(ae_config["expand_if_success_rate_at_least"]) or (
                candidate >= baseline - float(ae_config["expand_if_within_baseline_pp"]) / 100.0
            ):
                if any(latest_complete(condition, suite, start, start + 10) is None for start in range(10, 50, 10)):
                    jobs.append({"stage": "ae_full", "condition": condition, "suites": [suite],
                                 "denoise_steps": int(steps)})
    return jobs


def run_jobs(jobs: list[dict[str, Any]], config: dict[str, Any]) -> None:
    global NEXT_GPU_POLL
    pending = list(jobs)
    active: dict[concurrent.futures.Future, tuple[int, dict[str, Any]]] = {}
    retry_count: dict[tuple[str, tuple[str, ...]], int] = {}
    if NEXT_GPU_POLL is None:
        # This is the first check in this scheduler process; subsequent checks
        # remain spaced at least POLL_SECONDS apart when jobs are waiting.
        NEXT_GPU_POLL = time.monotonic()
        log_event({"event": "gpu_monitor_started", "first_status_check_in_seconds": 0,
                   "subsequent_check_interval_seconds": POLL_SECONDS})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while pending or active:
            if pending and time.monotonic() >= NEXT_GPU_POLL:
                used_gpus = {gpu for gpu, _ in active.values()}
                free = [gpu for gpu in free_gpus(config) if gpu not in used_gpus]
                while pending and free:
                    job = pending.pop(0)
                    gpu = free.pop(0)
                    future = executor.submit(run_condition_job, job, gpu, config)
                    active[future] = (gpu, job)
                    log_event({"event": "job_assigned", "gpu": gpu, "stage": job["stage"],
                               "condition": job["condition"], "suites": job["suites"]})
                NEXT_GPU_POLL = time.monotonic() + POLL_SECONDS
                if not free and pending:
                    log_event({"event": "gpu_busy", "pending_jobs": len(pending),
                               "next_check_seconds": POLL_SECONDS})
            if active:
                timeout = min(60.0, max(0.0, NEXT_GPU_POLL - time.monotonic())) if pending else 60.0
                done, _ = concurrent.futures.wait(active, timeout=timeout,
                                                   return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    gpu, job = active.pop(future)
                    try:
                        result = future.result()
                        log_event({"event": "job_complete", **result})
                    except Exception as exc:
                        key = (str(job["condition"]), tuple(job["suites"]))
                        retry_count[key] = retry_count.get(key, 0) + 1
                        log_event({"event": "job_error", "gpu": gpu, "condition": job["condition"],
                                   "suites": job["suites"], "retry": retry_count[key],
                                   "error": str(exc), "traceback": traceback.format_exc()})
                        if retry_count[key] > 1:
                            raise
                        pending.insert(0, job)
            elif pending:
                time.sleep(min(60.0, max(0.0, NEXT_GPU_POLL - time.monotonic())))


def main() -> int:
    if not SERVER_PYTHON.is_file() or not CLIENT_PYTHON.is_file():
        raise FileNotFoundError("The established Triton server or LIBERO client virtual environment is missing")
    if not BASE_DIR.is_dir() or not DRAFT_SOURCE.is_file():
        raise FileNotFoundError("Converted Full18 and draft checkpoint artifacts are required")
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_config()
    log_event({"event": "experiment_start", "config": str(CONFIG_PATH),
               "results": str(RESULTS), "gpu_policy": config["gpu_idle_threshold"],
               "gpu_check_interval_seconds": POLL_SECONDS})
    run_jobs(enqueue_pilot(config), config)
    log_event({"event": "pilot_complete", "action": "build_interim_report_and_expand"})
    report_script = ROOT / "scripts/experiment9/analyze_standard_vlm_pruning.py"
    if report_script.is_file():
        subprocess.run([str(CLIENT_PYTHON), str(report_script), "--phase", "pilot"], cwd=ROOT, check=True)
    run_jobs(enqueue_full(config), config)
    if report_script.is_file():
        subprocess.run([str(CLIENT_PYTHON), str(report_script), "--phase", "full"], cwd=ROOT, check=True)
    ae_config = config.get("action_expert_followup", {})
    if ae_config.get("enabled", False):
        ae_pilot_jobs = [
            {"stage": "ae_pilot", "condition": f"AE-D{steps}", "suites": list(SUITES),
             "denoise_steps": int(steps)}
            for steps in ae_config["denoise_steps"]
            if any(latest_complete(f"AE-D{steps}", suite, 0, 10) is None for suite in SUITES)
        ]
        run_jobs(ae_pilot_jobs, config)
        if report_script.is_file():
            subprocess.run([str(CLIENT_PYTHON), str(report_script), "--phase", "ae_pilot"], cwd=ROOT, check=True)
        run_jobs(enqueue_ae_full(config, ae_config), config)
        if report_script.is_file():
            subprocess.run([str(CLIENT_PYTHON), str(report_script), "--phase", "ae_full"], cwd=ROOT, check=True)
    log_event({"event": "experiment_complete", "results": str(RESULTS)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
