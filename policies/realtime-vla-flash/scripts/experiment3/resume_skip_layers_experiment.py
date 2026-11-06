"""Resume an interrupted skip-layer experiment without overwriting completed runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import run_skip_layers_experiment as base


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "results/pi0_vlm_action_denoise_ablation/manifest.json"


def _gpu_memory_mib(gpu: str) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return int(result.stdout.strip())


def _replace_flag(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command)
    if flag not in command:
        command.extend([flag, value])
        return command
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"missing value after {flag}")
    command[index + 1] = value
    return command


def _condition_run_dir(condition: dict[str, Any]) -> Path:
    path = Path(str(condition["run_dir"]))
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_completed(manifest: dict[str, Any]) -> None:
    expected = int(manifest["num_trials_per_condition"])
    for condition in manifest["conditions"]:
        if condition.get("status") != "completed":
            continue
        validation = base._artifact_check(
            _condition_run_dir(condition), str(condition["run_id"]), smoke=False
        )
        if not validation.get("ok") or int(validation.get("episode_count", -1)) != expected:
            raise RuntimeError(
                f"completed condition {condition['condition_id']} no longer validates: {validation}"
            )
        condition["artifact_validation"] = validation


def _prepare_pending(
    manifest: dict[str, Any], *, attempt_id: str, logs_root: Path, port: int
) -> list[dict[str, Any]]:
    pending = [condition for condition in manifest["conditions"] if condition.get("status") != "completed"]
    for offset, condition in enumerate(pending):
        condition_id = str(condition["condition_id"])
        run_id = (
            f"pi0_vlm_action_denoise_{manifest['experiment_id']}_retry_"
            f"{attempt_id}_{condition_id}"
        )
        condition_port = int(port) + offset
        condition_logs = logs_root / attempt_id / condition_id
        condition["run_id"] = run_id
        condition["run_dir"] = str((REPO_ROOT / "data/test" / run_id).relative_to(REPO_ROOT))
        condition["port"] = condition_port
        condition["server_log"] = str((condition_logs / "server.log").relative_to(WORKSPACE_ROOT))
        condition["client_log"] = str((condition_logs / "client.log").relative_to(WORKSPACE_ROOT))
        condition["server_command"] = _replace_flag(
            list(condition["server_command"]), "--port", str(condition_port)
        )
        client_command = _replace_flag(
            list(condition["client_command"]), "--port", str(condition_port)
        )
        condition["client_command"] = _replace_flag(client_command, "--run-name", run_id)
        condition["status"] = "pending"
        condition["resume_attempts"] = int(condition.get("resume_attempts", 0))
        for key in (
            "error",
            "artifact_validation",
            "return_code",
            "started_at",
            "finished_at",
        ):
            condition.pop(key, None)
    return pending


def _run_condition(
    condition: dict[str, Any],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    server_gpu: str,
    client_gpu: str,
    max_gpu_memory_mib: int,
    start_timeout_s: float,
) -> None:
    used = _gpu_memory_mib(server_gpu)
    if used > max_gpu_memory_mib:
        raise RuntimeError(
            f"GPU {server_gpu} is no longer available: {used} MiB used "
            f"(limit {max_gpu_memory_mib} MiB)"
        )
    base._port_available(int(condition["port"]))
    condition_logs = WORKSPACE_ROOT / Path(str(condition["server_log"])).parent
    condition_logs.mkdir(parents=True, exist_ok=True)
    server_log = WORKSPACE_ROOT / str(condition["server_log"])
    client_log = WORKSPACE_ROOT / str(condition["client_log"])
    server: subprocess.Popen | None = None
    condition["status"] = "running"
    condition["started_at"] = base._now()
    condition["resume_attempts"] = int(condition.get("resume_attempts", 0)) + 1
    manifest["status"] = "running"
    manifest["updated_at"] = base._now()
    base._atomic_json(manifest_path, manifest)
    try:
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = server_gpu
        client_env = os.environ.copy()
        client_env["CUDA_VISIBLE_DEVICES"] = client_gpu
        client_env["MUJOCO_GL"] = "egl"
        client_env["LIBERO_CONFIG_PATH"] = str(REPO_ROOT / ".libero")
        libero_pythonpath = str(REPO_ROOT / "third_party/libero")
        old_pythonpath = client_env.get("PYTHONPATH")
        client_env["PYTHONPATH"] = (
            libero_pythonpath
            if not old_pythonpath
            else f"{libero_pythonpath}{os.pathsep}{old_pythonpath}"
        )
        with server_log.open("w", encoding="utf-8") as handle:
            server = subprocess.Popen(
                condition["server_command"],
                cwd=REPO_ROOT,
                env=server_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        base._wait_server(server, int(condition["port"]), start_timeout_s, server_log)
        with client_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                condition["client_command"],
                cwd=REPO_ROOT,
                env=client_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        validation = base._artifact_check(
            _condition_run_dir(condition), str(condition["run_id"]), smoke=False
        )
        condition["artifact_validation"] = validation
        if result.returncode != 0:
            raise RuntimeError(f"client exited with {result.returncode}\n{base._tail(client_log)}")
        if not validation.get("ok"):
            raise RuntimeError(
                "artifact validation failed: " + "; ".join(validation.get("errors", []))
            )
        condition["status"] = "completed"
        condition["return_code"] = 0
        condition["finished_at"] = base._now()
    finally:
        base._stop(server)
        manifest["updated_at"] = base._now()
        base._atomic_json(manifest_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--server-gpu", required=True)
    parser.add_argument("--client-gpu", default=None)
    parser.add_argument("--port", type=int, default=8520)
    parser.add_argument("--max-gpu-memory-mib", type=int, default=1024)
    parser.add_argument("--start-timeout-s", type=float, default=600.0)
    parser.add_argument("--max-attempts-per-condition", type=int, default=3)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("conditions", [])) != 20:
        raise ValueError("resume requires the original 20-condition manifest")
    _validate_completed(manifest)
    if all(condition.get("status") == "completed" for condition in manifest["conditions"]):
        return 0

    server_gpu = str(args.server_gpu)
    client_gpu = str(args.client_gpu if args.client_gpu is not None else args.server_gpu)
    server_python = str(manifest["server_python"])
    client_python = str(manifest["client_python"])
    base._require_cuda(server_python, server_gpu, "server")
    base._require_cuda(client_python, client_gpu, "client")

    attempt_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    logs_root = Path(str(manifest["logs_root"]))
    if not logs_root.is_absolute():
        logs_root = WORKSPACE_ROOT / logs_root
    pending = _prepare_pending(
        manifest, attempt_id=attempt_id, logs_root=logs_root, port=args.port
    )
    manifest["server_gpu"] = server_gpu
    manifest["client_gpu"] = client_gpu
    manifest["resume_attempt_id"] = attempt_id
    manifest["failures"] = []
    base._atomic_json(manifest_path, manifest)

    for condition in pending:
        if int(condition.get("resume_attempts", 0)) >= args.max_attempts_per_condition:
            condition["status"] = "failed"
            condition["error"] = "maximum resume attempts reached"
            manifest["status"] = "incomplete"
            manifest["failures"] = [f"{condition['condition_id']}: {condition['error']}"]
            base._atomic_json(manifest_path, manifest)
            return 1
        try:
            _run_condition(
                condition,
                manifest=manifest,
                manifest_path=manifest_path,
                server_gpu=server_gpu,
                client_gpu=client_gpu,
                max_gpu_memory_mib=args.max_gpu_memory_mib,
                start_timeout_s=args.start_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - persist exact failure for the monitor
            condition["status"] = "failed"
            condition["error"] = str(exc)
            condition["finished_at"] = base._now()
            manifest["status"] = "incomplete"
            manifest["failures"] = [f"{condition['condition_id']}: {exc}"]
            manifest["updated_at"] = base._now()
            base._atomic_json(manifest_path, manifest)
            print(manifest["failures"][0], file=sys.stderr)
            return 75

    manifest["status"] = "completed"
    manifest["failures"] = []
    manifest["updated_at"] = base._now()
    base._atomic_json(manifest_path, manifest)
    summary_command = [
        server_python,
        "scripts/experiment3/summarize_skip_layers_experiment.py",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(manifest_path.parent),
    ]
    summary = subprocess.run(
        summary_command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    manifest["summary_command"] = summary_command
    manifest["summary_output"] = summary.stdout[-4000:]
    if summary.returncode != 0:
        manifest["status"] = "incomplete"
        manifest["failures"] = [f"summary failed with exit code {summary.returncode}"]
    manifest["updated_at"] = base._now()
    base._atomic_json(manifest_path, manifest)
    return summary.returncode


if __name__ == "__main__":
    raise SystemExit(main())
