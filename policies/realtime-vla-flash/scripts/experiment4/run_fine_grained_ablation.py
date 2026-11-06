"""Run the fine-grained Action Expert depth × denoising matrix on four GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment3"))
import run_skip_layers_experiment as stage1


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
DEFAULT_CONFIG = Path(__file__).with_name("fine_grained_ablation_config.json")
DEFAULT_RESULTS = REPO_ROOT / "results/pi0_ae_denoise_fine_ablation"
DEFAULT_LOGS = WORKSPACE_ROOT / "logs/pi0_ae_denoise_fine_ablation"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _args(values: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend([flag, str(value)])
    return result


def _gpu_memory_mib(gpu: str) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", gpu],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return int(result.stdout.strip())


def _run_one(
    condition: dict[str, Any],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    lock: threading.Lock,
    start_timeout_s: float,
) -> None:
    gpu = str(condition["gpu"])
    used = _gpu_memory_mib(gpu)
    if used > 1024:
        raise RuntimeError(f"GPU {gpu} is not exclusive: {used} MiB used")
    server_log = WORKSPACE_ROOT / condition["server_log"]
    client_log = WORKSPACE_ROOT / condition["client_log"]
    server_log.parent.mkdir(parents=True, exist_ok=True)
    server: subprocess.Popen | None = None
    with lock:
        condition["status"] = "running"
        condition["started_at"] = stage1._now()
        stage1._atomic_json(manifest_path, manifest)
    try:
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = gpu
        client_env = os.environ.copy()
        client_env["CUDA_VISIBLE_DEVICES"] = gpu
        client_env["MUJOCO_GL"] = "egl"
        client_env["LIBERO_CONFIG_PATH"] = str(REPO_ROOT / ".libero")
        client_env["PYTHONPATH"] = str(REPO_ROOT / "third_party/libero")
        with server_log.open("w", encoding="utf-8") as stream:
            server = subprocess.Popen(
                condition["server_command"], cwd=REPO_ROOT, env=server_env,
                stdout=stream, stderr=subprocess.STDOUT, text=True,
            )
        stage1._wait_server(server, int(condition["port"]), start_timeout_s, server_log)
        with client_log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                condition["client_command"], cwd=REPO_ROOT, env=client_env,
                stdout=stream, stderr=subprocess.STDOUT, text=True, check=False,
            )
        validation = stage1._artifact_check(
            REPO_ROOT / condition["run_dir"], condition["run_id"], bool(manifest["smoke_only"])
        )
        if result.returncode != 0:
            raise RuntimeError(f"client exited with {result.returncode}\n{stage1._tail(client_log)}")
        if not validation.get("ok"):
            raise RuntimeError("artifact validation failed: " + "; ".join(validation["errors"]))
        with lock:
            condition["artifact_validation"] = validation
            condition["status"] = "completed"
            condition["return_code"] = 0
            condition["finished_at"] = stage1._now()
            manifest["updated_at"] = stage1._now()
            stage1._atomic_json(manifest_path, manifest)
    finally:
        stage1._stop(server)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--port-base", type=int, default=8620)
    parser.add_argument("--start-timeout-s", type=float, default=600.0)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    config = _read_json(args.config.resolve())
    experiment_id = args.experiment_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    results_dir = args.results_dir.resolve()
    logs_root = args.logs_root.resolve() / experiment_id
    manifest_path = logs_root / "smoke_manifest.json" if args.smoke_only else results_dir / "manifest.json"
    server_python = str((REPO_ROOT / ".venv/bin/python").absolute())
    client_python = str((REPO_ROOT / "examples/libero/.venv/bin/python").absolute())
    trials = 1 if args.smoke_only else int(config["num_trials_per_task"])
    conditions: list[dict[str, Any]] = []
    for index, raw in enumerate(config["conditions"]):
        condition = dict(raw)
        condition_id = condition["condition_id"]
        run_id = f"pi0_ae_denoise_{'smoke_' if args.smoke_only else ''}{experiment_id}_{condition_id}"
        port = int(args.port_base) + index
        condition.update({
            "vlm_layer_pattern": "vlm_full18",
            "action_condition_id": condition_id,
            "is_baseline": condition_id == "E18_D10",
            "run_id": run_id,
            "run_dir": str(Path("data/test") / run_id),
            "port": port,
            "status": "pending",
            "server_log": str((logs_root / condition_id / "server.log").relative_to(WORKSPACE_ROOT)),
            "client_log": str((logs_root / condition_id / "client.log").relative_to(WORKSPACE_ROOT)),
        })
        server_args = {
            "config": "pi0_libero",
            "base_triton_path": WORKSPACE_ROOT / "shared/checkpoints/realtime-vla-flash/converted/base",
            "draft_triton_path": WORKSPACE_ROOT / "shared/checkpoints/realtime-vla-flash/converted/draft",
            "backend": "triton",
            "task_suite_name": config["task_suite_name"],
            "tokenizer_source": "local",
            "max_exec_steps": 12,
            "force_full_each_round": True,
            "port": port,
            "vlm_layer_pattern": "vlm_full18",
            "action_layer_pattern": condition["action_layer_pattern"],
            "denoise_steps": condition["denoise_steps"],
            "ablation_noise_seed": config["global_noise_seed"],
        }
        client_args = {
            "task_suite_name": config["task_suite_name"], "task": config["task_id"],
            "num_trials_per_task": trials, "num_steps_wait": config["num_steps_wait"],
            "replan_steps": config["replan_steps"], "resize_size": config["resize_size"],
            "seed": config["seed"], "host": "127.0.0.1", "port": port,
            "video_out_path": REPO_ROOT / "data/test", "run_name": run_id,
        }
        condition["server_command"] = [server_python, "scripts/spec/spec_serve_policy.py", *_args(server_args)]
        condition["client_command"] = [client_python, "scripts/spec/spec_client_libero.py", *_args(client_args)]
        conditions.append(condition)

    if not args.smoke_only and results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"results directory is not empty: {results_dir}")
    for condition in conditions:
        run_dir = REPO_ROOT / condition["run_dir"]
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
    for gpu in sorted({str(condition["gpu"]) for condition in conditions}):
        stage1._require_cuda(server_python, gpu, f"server GPU{gpu}")
        if _gpu_memory_mib(gpu) > 1024:
            raise RuntimeError(f"GPU {gpu} is not available")

    manifest = {
        "experiment_id": experiment_id, "title": config["title"],
        "created_at": stage1._now(), "updated_at": stage1._now(), "status": "running",
        "smoke_only": args.smoke_only, "task_suite_name": config["task_suite_name"],
        "task_id": config["task_id"], "task_description": config["task_description"],
        "num_trials_per_condition": trials, "condition_count": len(conditions),
        "global_noise_seed": config["global_noise_seed"], "conditions": conditions,
        "server_python": server_python, "client_python": client_python,
    }
    stage1._atomic_json(manifest_path, manifest)
    lock = threading.Lock()
    failures: list[str] = []

    def run_gpu(gpu: str) -> None:
        for condition in [item for item in conditions if str(item["gpu"]) == gpu]:
            try:
                _run_one(
                    condition, manifest=manifest, manifest_path=manifest_path,
                    lock=lock, start_timeout_s=args.start_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 - persist exact condition failure
                with lock:
                    condition["status"] = "failed"
                    condition["error"] = str(exc)
                    condition["finished_at"] = stage1._now()
                    failures.append(f"{condition['condition_id']}: {exc}")
                    manifest["updated_at"] = stage1._now()
                    stage1._atomic_json(manifest_path, manifest)
                return

    gpu_ids = sorted({str(condition["gpu"]) for condition in conditions})
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(run_gpu, gpu) for gpu in gpu_ids]
        for future in futures:
            future.result()
    manifest["status"] = "completed" if not failures else "incomplete"
    manifest["failures"] = failures
    manifest["updated_at"] = stage1._now()
    stage1._atomic_json(manifest_path, manifest)
    if failures or args.smoke_only:
        return 1 if failures else 0
    summary = subprocess.run(
        [server_python, "scripts/experiment4/summarize_fine_grained_ablation.py",
         "--manifest", str(manifest_path), "--output-dir", str(results_dir)],
        cwd=REPO_ROOT, check=False,
    )
    return summary.returncode


if __name__ == "__main__":
    raise SystemExit(main())
