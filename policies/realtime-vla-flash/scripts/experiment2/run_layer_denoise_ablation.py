# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "layer_denoise_ablation_config.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "pi0_layer_denoise_ablation"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {path}")
    return config


def _resolve_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_template(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_template(item) for item in value]
    if isinstance(value, str):
        return value.replace("$WORKSPACE_ROOT", str(WORKSPACE_ROOT)).replace("$REPO_ROOT", str(REPO_ROOT))
    return value


def _slug(value: str) -> str:
    return value.replace("-", "_")


def _condition_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    layers = list(config["layer_patterns"])
    flows = list(config["flow_patterns"])
    conditions = [
        {
            "condition_id": f"{layer}__{flow}",
            "layer_pattern": layer,
            "flow_pattern": flow,
            "is_baseline": False,
        }
        for layer, flow in itertools.product(layers, flows)
    ]
    baseline = dict(config["baseline"])
    conditions.append(
        {
            "condition_id": f"{baseline['layer_pattern']}__{baseline['flow_pattern']}",
            "layer_pattern": baseline["layer_pattern"],
            "flow_pattern": baseline["flow_pattern"],
            "is_baseline": True,
        }
    )
    if len(conditions) != 10 or len({item["condition_id"] for item in conditions}) != 10:
        raise ValueError("Experiment config must produce exactly ten unique conditions (3×3 + baseline).")
    return conditions


def _cli_args(values: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
            else:
                result.append("--no-" + str(key).replace("_", "-"))
        elif isinstance(value, list | tuple):
            result.append(flag)
            result.extend(str(item) for item in value)
        else:
            result.extend((flag, str(value)))
    return result


def _tail(path: Path, line_count: int = 30) -> str:
    if not path.exists():
        return "<log not created>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:])


def _assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is not available; choose another with --port.") from exc


def _wait_for_server(process: subprocess.Popen, port: int, timeout_s: float, interval_s: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited early (code {process.returncode}).\n{_tail(log_path)}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(interval_s)
    raise TimeoutError(f"Server did not listen on port {port} within {timeout_s:.0f}s.\n{_tail(log_path)}")


def _stop_server(process: subprocess.Popen, timeout_s: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _validate_smoke_artifacts(run_dir: Path, expected_run_id: str) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = run_dir / "manifest.json"
    episode_log_path = run_dir / "episode_log.json"
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {manifest_path.name}"]}
    if not episode_log_path.is_file():
        return {"ok": False, "errors": [f"missing {episode_log_path.name}"]}

    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("run_id") != expected_run_id:
        errors.append(f"manifest run_id is {run_manifest.get('run_id')!r}, expected {expected_run_id!r}")
    if int(run_manifest.get("num_trials_per_task", -1)) != 1:
        errors.append("client manifest does not record exactly one trial per task")
    if run_manifest.get("selected_task_ids") != [5]:
        errors.append(f"client manifest selected_task_ids={run_manifest.get('selected_task_ids')!r}, expected [5]")

    episodes = json.loads(episode_log_path.read_text(encoding="utf-8"))
    if not isinstance(episodes, list) or len(episodes) != 1:
        count = len(episodes) if isinstance(episodes, list) else type(episodes).__name__
        return {"ok": False, "errors": [*errors, f"expected exactly one episode log row; got {count}"]}
    episode = episodes[0]
    if int(episode.get("task_id", -1)) != 5 or int(episode.get("episode_idx", -1)) != 0:
        errors.append("episode log row is not task 5, episode 0")

    paths: dict[str, Path] = {}
    for key in ("video_path", "trace_path", "infer_path"):
        raw = episode.get(key)
        if not isinstance(raw, str) or not raw:
            errors.append(f"episode log has no {key}")
            continue
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe {key}: {raw!r}")
            continue
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            errors.append(f"{key} escapes run directory: {raw!r}")
            continue
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {key} artifact: {path}")
            continue
        paths[key] = path

    trace: list[dict[str, Any]] = []
    infer: list[dict[str, Any]] = []
    try:
        if "trace_path" in paths:
            trace = _jsonl_records(paths["trace_path"])
        if "infer_path" in paths:
            infer = _jsonl_records(paths["infer_path"])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid trace/infer JSONL: {exc}")

    if "trace_path" in paths and len(trace) != int(episode.get("trace_record_count", -1)):
        errors.append("trace line count differs from episode_log trace_record_count")
    if "infer_path" in paths:
        if len(infer) != int(episode.get("infer_record_count", -1)):
            errors.append("infer line count differs from episode_log infer_record_count")
        if len(infer) != int(episode.get("infer_calls", -1)):
            errors.append("infer line count differs from episode_log infer_calls")
    executed = sum(record.get("executed_action") is not None for record in trace)
    if executed != int(episode.get("executed_action_count", -1)):
        errors.append("trace executed_action count differs from episode_log executed_action_count")
    if executed != int(episode.get("trajectory_length", -1)):
        errors.append("trace executed_action count differs from episode_log trajectory_length")

    return {
        "ok": not errors,
        "errors": errors,
        "episode_idx": episode.get("episode_idx"),
        "success": episode.get("success"),
        "artifact_paths": {key: path.relative_to(run_dir.resolve()).as_posix() for key, path in paths.items()},
        "trace_records": len(trace),
        "infer_records": len(infer),
        "executed_actions": executed,
    }


def _run_full_path_bitwise_preflight(
    *, server_python: str, server_gpu: str, config: dict[str, Any], noise_seed: int, output_path: Path
) -> dict[str, Any]:
    server_args = _resolve_template(config["server_args"])
    observation_path = REPO_ROOT / "results/pi0_intermediate_probe/libero_observations_200.npz"
    command = [
        server_python,
        str(SCRIPT_DIR / "verify_full_path_bitwise.py"),
        "--base-artifact",
        str(server_args["base_triton_path"]),
        "--draft-artifact",
        str(server_args["draft_triton_path"]),
        "--observations",
        str(observation_path),
        "--noise-seed",
        str(noise_seed),
        "--output-path",
        str(output_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(server_gpu)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    report = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else {}
    if completed.returncode != 0 or report.get("bitwise_equal") is not True or report.get("max_abs_diff") != 0.0:
        raise RuntimeError(
            "full18/full10 GPU bitwise preflight failed; formal experiment was not started. "
            f"exit={completed.returncode}, report={report}"
        )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ten π0 layer × denoising LIBERO ablations serially.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--server-python", default=os.environ.get("PI0_ABLATION_SERVER_PYTHON"))
    parser.add_argument("--client-python", default=os.environ.get("PI0_ABLATION_CLIENT_PYTHON"))
    parser.add_argument("--server-gpu", default=os.environ.get("PI0_ABLATION_SERVER_GPU", "0"))
    parser.add_argument("--client-gpu", default=os.environ.get("PI0_ABLATION_CLIENT_GPU", "1"))
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--start-timeout-s", type=float, default=600.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--server-shutdown-timeout-s", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run one isolated LIBERO episode per condition and validate rollout artifacts; do not create formal summaries.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    return parser.parse_args()


def _resolve_python(configured: str | None, preferred: Path) -> str:
    candidate = configured or str(preferred)
    resolved = shutil.which(candidate) if not Path(candidate).is_absolute() else candidate
    if resolved is None or not Path(resolved).is_file():
        raise FileNotFoundError(f"Python executable not found: {candidate}")
    # Keep the virtualenv entry-point path intact.  Resolving its ``python``
    # symlink to the base interpreter bypasses pyvenv.cfg and silently drops
    # all packages installed in the experiment environment (notably JAX).
    return str(Path(resolved).expanduser().absolute())


def _preflight(config: dict[str, Any], server_python: str, client_python: str) -> dict[str, Any]:
    resolved = _resolve_template(config)
    for key in ("base_triton_path", "draft_triton_path"):
        value = Path(str(resolved["server_args"][key])).expanduser()
        if not value.exists():
            raise FileNotFoundError(f"Required Triton artifact for {key} is missing: {value}")
    if not (REPO_ROOT / "scripts/spec/spec_serve_policy.py").is_file():
        raise FileNotFoundError("spec_serve_policy.py was not found under the expected repo root.")
    if not (REPO_ROOT / "scripts/spec/spec_client_libero.py").is_file():
        raise FileNotFoundError("spec_client_libero.py was not found under the expected repo root.")
    if not Path(server_python).is_file() or not Path(client_python).is_file():
        raise FileNotFoundError("The configured server/client Python executable does not exist.")
    return resolved


def _validate_server_cli(server_python: str) -> str:
    completed = subprocess.run(
        [server_python, "scripts/spec/spec_serve_policy.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    help_text = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"Could not inspect spec_serve_policy.py CLI:\n{help_text[-6000:]}")
    expected_flags = (
        "--ablation-layer-pattern",
        "--ablation-flow-pattern",
        "--ablation-noise-seed",
    )
    missing = [flag for flag in expected_flags if flag not in help_text]
    if missing:
        raise RuntimeError(
            "Server CLI is missing the ablation interface required by this runner: " + ", ".join(missing)
        )
    expected_patterns = ("full18", "first3_last3", "first6", "last6", "full10", "first3", "last3")
    missing_patterns = [pattern for pattern in expected_patterns if pattern not in help_text]
    if missing_patterns:
        raise RuntimeError("Server CLI does not advertise expected pattern values: " + ", ".join(missing_patterns))
    return help_text


def _manifest_payload(
    *, experiment_id: str, config: dict[str, Any], conditions: list[dict[str, Any]], created_at: str
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "title": str(config["title"]),
        "created_at": created_at,
        "updated_at": created_at,
        "status": "planned",
        "task_suite_name": config["task_suite_name"],
        "task_id": int(config["task_id"]),
        "task_description": config["task_description"],
        "num_trials_per_condition": int(config["num_trials_per_task"]),
        "num_steps_wait": int(config["num_steps_wait"]),
        "replan_steps": int(config["replan_steps"]),
        "resize_size": int(config["resize_size"]),
        "video_fps": int(config["video_fps"]),
        "environment_seed": int(config["seed"]),
        "ablation_noise_seed": int(config["noise_seed"]),
        "flow_semantics": {
            "full10": "execute original step indices 0-9 at t=1.0 through 0.1",
            "first3": "execute original step indices 0-2 at t=1.0, 0.9, 0.8, then return",
            "last3": "preserve initial noise without state updates for indices 0-6, then execute indices 7-9 at t=0.3, 0.2, 0.1",
        },
        "layer_semantics": {
            "first3_last3": "Action Expert layers 1-3 followed by 16-18",
            "first6": "Action Expert layers 1-6",
            "last6": "Action Expert layers 13-18 from the original suffix embedding",
            "full18": "Action Expert layers 1-18",
        },
        "baseline_condition_id": "full18__full10",
        "conditions": conditions,
    }


def run() -> int:
    args = _parse_args()
    args.config = args.config.expanduser().resolve()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    config = _load_config(args.config)
    server_python = _resolve_python(args.server_python, REPO_ROOT / ".venv/bin/python")
    client_preferred = (
        REPO_ROOT / "examples/libero/.venv/bin/python"
        if (REPO_ROOT / "examples/libero/.venv/bin/python").is_file()
        else REPO_ROOT / ".venv/bin/python"
    )
    client_python = _resolve_python(args.client_python, client_preferred)
    resolved_config = _preflight(config, server_python, client_python)
    conditions = _condition_specs(config)
    experiment_id = args.experiment_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in experiment_id):
        raise ValueError("experiment-id may contain only letters, numbers, hyphens, and underscores.")
    results_dir = args.results_dir if args.results_dir.is_absolute() else REPO_ROOT / args.results_dir
    results_dir = results_dir.resolve()
    manifest_path = results_dir / "manifest.json"
    data_root = (REPO_ROOT / "data/test").resolve()
    experiment_logs_root = (WORKSPACE_ROOT / "logs/pi0_layer_denoise_ablation" / experiment_id).resolve()
    logs_root = experiment_logs_root / "smoke" if args.smoke_only else experiment_logs_root
    if args.smoke_only:
        manifest_path = logs_root / "smoke_manifest.json"

    if not args.dry_run:
        _validate_server_cli(server_python)
        for candidate_port in range(args.port, args.port + len(conditions)):
            _assert_port_available(candidate_port)
        if not args.smoke_only and results_dir.exists() and any(results_dir.iterdir()) and not args.overwrite_results:
            raise FileExistsError(
                f"Results directory is not empty: {results_dir}; choose --overwrite-results only if replacement is intended."
            )
        run_dirs = [
            data_root
            / f"pi0_layer_denoise_{'smoke_' if args.smoke_only else ''}{experiment_id}_{_slug(item['condition_id'])}"
            for item in conditions
        ]
        existing = [str(path) for path in run_dirs if path.exists()]
        if existing:
            raise FileExistsError("Run output directories already exist; refusing to overwrite: " + ", ".join(existing))

    created_at = _utc_now()
    for index, condition in enumerate(conditions):
        condition.update(
            {
                "run_id": f"pi0_layer_denoise_{'smoke_' if args.smoke_only else ''}{experiment_id}_{_slug(condition['condition_id'])}",
                "run_dir": f"data/test/pi0_layer_denoise_{'smoke_' if args.smoke_only else ''}{experiment_id}_{_slug(condition['condition_id'])}",
                "port": int(args.port) + index,
                "status": "pending",
                "server_log": str((logs_root / condition["condition_id"] / "server.log").relative_to(WORKSPACE_ROOT)),
                "client_log": str((logs_root / condition["condition_id"] / "client.log").relative_to(WORKSPACE_ROOT)),
            }
        )
    manifest = _manifest_payload(experiment_id=experiment_id, config=resolved_config, conditions=conditions, created_at=created_at)
    if args.smoke_only:
        manifest["experiment_type"] = "preflight_smoke_only"
        manifest["formal_num_trials_per_condition"] = manifest["num_trials_per_condition"]
        manifest["num_trials_per_condition"] = 1
        manifest["num_trials_per_task"] = 1
    manifest["config_path"] = str(args.config)
    manifest["server_python"] = server_python
    manifest["client_python"] = client_python
    manifest["server_gpu"] = args.server_gpu
    manifest["client_gpu"] = args.client_gpu
    manifest["results_dir"] = None if args.smoke_only else _relative(results_dir)
    manifest["data_root"] = str(data_root.relative_to(REPO_ROOT))
    manifest["logs_root"] = str(logs_root.relative_to(WORKSPACE_ROOT))

    for condition in conditions:
        server_args = dict(resolved_config["server_args"])
        server_args.update(
            {
                "port": condition["port"],
                "ablation_layer_pattern": condition["layer_pattern"],
                "ablation_flow_pattern": condition["flow_pattern"],
                "ablation_noise_seed": int(config["noise_seed"]),
            }
        )
        server_command = [server_python, "scripts/spec/spec_serve_policy.py", *_cli_args(server_args)]
        client_args = dict(resolved_config["client_args"])
        client_args.update(
            {
                "host": "127.0.0.1",
                "port": condition["port"],
                "video_out_path": str(data_root),
                "run_name": condition["run_id"],
            }
        )
        if args.smoke_only:
            client_args["num_trials_per_task"] = 1
        client_command = [client_python, "scripts/spec/spec_client_libero.py", *_cli_args(client_args)]
        condition["server_command"] = server_command
        condition["client_command"] = client_command

    if args.dry_run:
        for condition in conditions:
            trials = 1 if args.smoke_only else int(config["num_trials_per_task"])
            print(f"[{condition['condition_id']}] trials={trials} port={condition['port']} run={condition['run_dir']}")
            print("  server: " + shlex.join(condition["server_command"]))
            print("  client: " + shlex.join(condition["client_command"]))
        print(f"validated {len(conditions)} conditions; dry-run did not create outputs or start processes")
        return 0

    if not args.smoke_only:
        regression_report_path = logs_root / "full18_full10_bitwise.json"
        regression_report = _run_full_path_bitwise_preflight(
            server_python=server_python,
            server_gpu=str(args.server_gpu),
            config=resolved_config,
            noise_seed=int(config["noise_seed"]),
            output_path=regression_report_path,
        )
        manifest["full_path_bitwise_preflight"] = {
            "report_path": _relative(regression_report_path),
            "bitwise_equal": regression_report["bitwise_equal"],
            "max_abs_diff": regression_report.get("max_abs_diff"),
        }

    _atomic_json(manifest_path, manifest)
    logs_root.mkdir(parents=True, exist_ok=True)
    manifest["status"] = "running"
    manifest["updated_at"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    failures: list[str] = []

    try:
        for condition in conditions:
            condition_log_dir = logs_root / condition["condition_id"]
            condition_log_dir.mkdir(parents=True, exist_ok=True)
            server_log_path = condition_log_dir / "server.log"
            client_log_path = condition_log_dir / "client.log"
            server_env = os.environ.copy()
            server_env["CUDA_VISIBLE_DEVICES"] = str(args.server_gpu)
            client_env = os.environ.copy()
            client_env["CUDA_VISIBLE_DEVICES"] = str(args.client_gpu)
            client_env.setdefault("MUJOCO_GL", "egl")
            client_env["LIBERO_CONFIG_PATH"] = str(REPO_ROOT / ".libero")
            libero_pythonpath = str(REPO_ROOT / "third_party/libero")
            inherited_pythonpath = client_env.get("PYTHONPATH")
            client_env["PYTHONPATH"] = (
                libero_pythonpath
                if not inherited_pythonpath
                else libero_pythonpath + os.pathsep + inherited_pythonpath
            )
            condition["status"] = "running"
            condition["started_at"] = _utc_now()
            condition["server_command"] = condition["server_command"]
            manifest["updated_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            server_process: subprocess.Popen | None = None
            try:
                _assert_port_available(int(condition["port"]))
                with server_log_path.open("w", encoding="utf-8") as server_log:
                    server_process = subprocess.Popen(
                        condition["server_command"],
                        cwd=REPO_ROOT,
                        env=server_env,
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                _wait_for_server(
                    server_process,
                    int(condition["port"]),
                    args.start_timeout_s,
                    args.poll_interval_s,
                    server_log_path,
                )
                with client_log_path.open("w", encoding="utf-8") as client_log:
                    result = subprocess.run(
                        condition["client_command"],
                        cwd=REPO_ROOT,
                        env=client_env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                run_dir = REPO_ROOT / condition["run_dir"]
                smoke_validation = _validate_smoke_artifacts(run_dir, condition["run_id"]) if args.smoke_only else None
                if smoke_validation is not None:
                    condition["artifact_validation"] = smoke_validation
                if result.returncode != 0:
                    raise RuntimeError(f"LIBERO client exited with code {result.returncode}.\n{_tail(client_log_path)}")
                if smoke_validation is not None and not smoke_validation["ok"]:
                    raise RuntimeError("smoke artifact validation failed: " + "; ".join(smoke_validation["errors"]))
                condition_manifest = {
                    "experiment_id": experiment_id,
                    "condition_id": condition["condition_id"],
                    "layer_pattern": condition["layer_pattern"],
                    "flow_pattern": condition["flow_pattern"],
                    "is_baseline": condition["is_baseline"],
                    "ablation_noise_seed": int(config["noise_seed"]),
                    "server_command": condition["server_command"],
                }
                _atomic_json(run_dir / "ablation_condition.json", condition_manifest)
                condition["status"] = "completed"
                condition["finished_at"] = _utc_now()
                condition["return_code"] = 0
            except Exception as exc:
                condition["status"] = "failed"
                condition["finished_at"] = _utc_now()
                condition["error"] = str(exc)
                failures.append(f"{condition['condition_id']}: {exc}")
                if not (args.continue_on_error or args.smoke_only):
                    manifest["status"] = "failed"
                    manifest["failure"] = failures[-1]
                    break
            finally:
                if server_process is not None:
                    _stop_server(server_process, args.server_shutdown_timeout_s)
            manifest["updated_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["updated_at"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        raise

    if not failures:
        manifest["status"] = "completed"
    elif manifest["status"] != "failed":
        manifest["status"] = "incomplete"
    manifest["updated_at"] = _utc_now()
    _atomic_json(manifest_path, manifest)

    integrity_errors: list[str] = []
    if args.smoke_only:
        integrity_errors = [
            f"{condition['condition_id']}: {condition.get('artifact_validation', {}).get('errors', condition.get('error', 'not completed'))}"
            for condition in conditions
            if condition.get("status") != "completed"
        ]
        manifest["smoke_validation_passed"] = not integrity_errors
        manifest["updated_at"] = _utc_now()
        _atomic_json(manifest_path, manifest)
    else:
        sys.path.insert(0, str(SCRIPT_DIR))
        from summarize_layer_denoise_ablation import summarize_experiment

        integrity_errors = summarize_experiment(manifest_path, results_dir)
    if failures:
        print("\n".join(failures), file=sys.stderr)
    if integrity_errors:
        print("Artifact integrity check failed:", file=sys.stderr)
        for error in integrity_errors:
            print(f"- {error}", file=sys.stderr)
    return 1 if failures or integrity_errors else 0


if __name__ == "__main__":
    raise SystemExit(run())
