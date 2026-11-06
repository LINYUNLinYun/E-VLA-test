"""Run the 20-condition VLM/Action-Expert skip-layer experiment serially."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
from pathlib import Path
import random
import shlex
import socket
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
DEFAULT_CONFIG = Path(__file__).with_name("skip_layers_experiment_config.json")
DEFAULT_RESULTS = REPO_ROOT / "results/pi0_vlm_action_denoise_ablation"
DEFAULT_LOGS = WORKSPACE_ROOT / "logs/pi0_vlm_action_denoise_ablation"

_VLM_INDICES = {
    "vlm_first3_last3": (1, 2, 3, 16, 17, 18),
    "vlm_first5_last5": (1, 2, 3, 4, 5, 14, 15, 16, 17, 18),
    "vlm_first6": (1, 2, 3, 4, 5, 6),
    "vlm_last6": (13, 14, 15, 16, 17, 18),
    "vlm_full18": tuple(range(1, 19)),
}
_ACTION_INDICES = {"full18": tuple(range(1, 19)), "first5": (1, 2, 3, 4, 5)}


def _experiment_fields(condition: dict[str, Any], global_noise_seed: int) -> dict[str, Any]:
    vlm_indices = _VLM_INDICES[condition["vlm_layer_pattern"]]
    action_indices = _ACTION_INDICES[condition["action_layer_pattern"]]
    steps = int(condition["denoise_steps"])
    return {
        "vlm_layer_indices_1based": list(vlm_indices),
        "vlm_full_block_indices_1based": list(vlm_indices),
        "vlm_kv_shell_indices_1based": [index for index in range(1, 19) if index not in vlm_indices],
        "prefix_kv_indices_1based": list(range(1, 19)),
        "num_vlm_layers": len(vlm_indices),
        "vlm_full_block_visits": len(vlm_indices),
        "vlm_kv_shell_visits": 18 - len(vlm_indices),
        "kv_projection_visits": 18,
        "action_layer_indices_1based": list(action_indices),
        "num_action_layers": len(action_indices),
        "denoise_times": [round(1.0 - (0.2 if steps == 5 else 0.1) * i, 1) for i in range(steps)],
        "flow_dt": -0.2 if steps == 5 else -0.1,
        "num_flow_steps": steps,
        "action_layer_visits": len(action_indices) * steps,
        "denoise_updates": steps,
        "physical_vlm_layers": 18,
        "physical_action_expert_layers": 18,
        "global_noise_seed": int(global_noise_seed),
        "noise_seed_rule": "SeedSequence(global_noise_seed, episode_idx, replan_idx)",
    }


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _resolve(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item) for item in value]
    if isinstance(value, str):
        return value.replace("$WORKSPACE_ROOT", str(WORKSPACE_ROOT)).replace("$REPO_ROOT", str(REPO_ROOT))
    return value


def _cli_args(values: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                out.append(flag)
        elif isinstance(value, (list, tuple)):
            out.extend([flag, *(str(item) for item in value)])
        else:
            out.extend([flag, str(value)])
    return out


def _slug(value: str) -> str:
    return value.replace("-", "_")


def _port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError as exc:
            raise RuntimeError(f"port {port} is not available") from exc


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return "<log missing>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def _wait_server(process: subprocess.Popen, port: int, timeout: float, log: Path) -> None:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}\n{_tail(log)}")
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"server did not listen on {port}\n{_tail(log)}")


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_check(run_dir: Path, expected_run_id: str, smoke: bool) -> dict[str, Any]:
    errors: list[str] = []
    manifest = run_dir / "manifest.json"
    episode_log = run_dir / "episode_log.json"
    if not manifest.is_file():
        errors.append("missing manifest.json")
    if not episode_log.is_file():
        errors.append("missing episode_log.json")
    if errors:
        return {"ok": False, "errors": errors}
    run_manifest = _json(manifest)
    if run_manifest.get("run_id") != expected_run_id:
        errors.append(f"run_id={run_manifest.get('run_id')!r}, expected {expected_run_id!r}")
    episodes = _json(episode_log)
    expected = 1 if smoke else int(run_manifest.get("num_trials_per_task", 0))
    if not isinstance(episodes, list) or len(episodes) != expected:
        errors.append(f"episode count={len(episodes) if isinstance(episodes, list) else type(episodes).__name__}, expected {expected}")
        return {"ok": False, "errors": errors}
    successes = 0
    for ep in episodes:
        successes += int(bool(ep.get("success", False)))
        for key in ("video_path", "trace_path", "infer_path"):
            raw = ep.get(key)
            if not isinstance(raw, str) or Path(raw).is_absolute() or ".." in Path(raw).parts:
                errors.append(f"invalid {key}")
                continue
            artifact = (run_dir / raw).resolve()
            if not artifact.is_relative_to(run_dir.resolve()) or not artifact.is_file() or artifact.stat().st_size == 0:
                errors.append(f"missing {key}: {raw}")
    return {"ok": not errors, "errors": errors, "episode_count": len(episodes), "successes": successes, "success_rate": successes / len(episodes)}


def _conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    patterns = list(config["vlm_layer_patterns"])
    actions = list(config["action_conditions"])
    baseline = config["baseline"]
    conditions: list[dict[str, Any]] = []
    for vlm, action in itertools.product(patterns, actions):
        condition_id = f"V{patterns.index(vlm) + 1}_{action['id']}"
        full_name = (
            f"{vlm}__ae_{action['action_layer_pattern']}__"
            f"denoise{int(action['denoise_steps'])}"
        )
        conditions.append(
            {
                "condition_id": condition_id,
                "condition_name": full_name,
                "vlm_layer_pattern": vlm,
                "action_condition_id": action["id"],
                "action_layer_pattern": action["action_layer_pattern"],
                "denoise_steps": int(action["denoise_steps"]),
                "is_baseline": vlm == baseline["vlm_layer_pattern"] and action["id"] == baseline["action_condition_id"],
            }
        )
        conditions[-1].update(_experiment_fields(conditions[-1], int(config["global_noise_seed"])))
    if len(conditions) != 20 or sum(item["is_baseline"] for item in conditions) != 1:
        raise ValueError("config must produce exactly 20 conditions and one baseline")
    return conditions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--server-python", default=os.environ.get("PI0_ABLATION_SERVER_PYTHON"))
    parser.add_argument("--client-python", default=os.environ.get("PI0_ABLATION_CLIENT_PYTHON"))
    parser.add_argument("--server-gpu", default=None)
    parser.add_argument("--client-gpu", default=None)
    parser.add_argument("--port", type=int, default=8220)
    parser.add_argument("--start-timeout-s", type=float, default=600.0)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _python(value: str | None, preferred: Path) -> str:
    candidate = value or str(preferred)
    path = Path(candidate).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"python executable not found: {path}")
    # Keep the venv entry point itself; resolving its symlink bypasses
    # pyvenv.cfg and drops the experiment dependencies.
    return str(path.absolute())


def _require_cuda(python: str, visible_gpu: str, role: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)
    probe = subprocess.run(
        [python, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 2)"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()[-1:]
        suffix = f" ({detail[0]})" if detail else ""
        raise RuntimeError(f"{role} CUDA preflight failed for CUDA_VISIBLE_DEVICES={visible_gpu}{suffix}")


def _run_v5_a_bitwise_preflight(
    *, server_python: str, server_gpu: str, config: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Reuse the independent full-path reference check before formal rollout."""
    observation_path = REPO_ROOT / "results/pi0_intermediate_probe/libero_observations_200.npz"
    if not observation_path.is_file():
        raise FileNotFoundError(f"V5_A preflight observations are missing: {observation_path}")
    command = [
        server_python,
        "scripts/experiment2/verify_full_path_bitwise.py",
        "--base-artifact",
        str(config["server_args"]["base_triton_path"]),
        "--draft-artifact",
        str(config["server_args"]["draft_triton_path"]),
        "--observations",
        str(observation_path),
        "--noise-seed",
        str(config["global_noise_seed"]),
        "--output-path",
        str(output_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(server_gpu)
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    report = _json(output_path) if output_path.is_file() else {}
    if result.returncode != 0 or report.get("bitwise_equal") is not True or report.get("max_abs_diff") != 0.0:
        raise RuntimeError(
            "V5_A baseline bitwise preflight failed; formal experiment was not started: "
            f"exit={result.returncode}, report={report}"
        )
    return report


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config = _resolve(_json(config_path))
    conditions = _conditions(config)
    condition_order_seed = int(config.get("condition_order_seed", 20260930))
    random.Random(condition_order_seed).shuffle(conditions)
    server_python = _python(args.server_python, REPO_ROOT / ".venv/bin/python")
    client_python = _python(args.client_python, REPO_ROOT / "examples/libero/.venv/bin/python")
    experiment_id = args.experiment_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    results_dir = args.results_dir.expanduser().resolve()
    logs_root = args.logs_root.expanduser().resolve() / experiment_id
    data_root = (REPO_ROOT / "data/test").resolve()
    server_gpu = str(args.server_gpu if args.server_gpu is not None else config.get("server_gpu", "0"))
    client_gpu = str(args.client_gpu if args.client_gpu is not None else config.get("client_gpu", "1"))
    for index, condition in enumerate(conditions):
        name = f"pi0_vlm_action_denoise_{'smoke_' if args.smoke_only else ''}{experiment_id}_{condition['condition_id']}"
        condition.update(
            {
                "run_id": name,
                "run_dir": str((data_root / name).relative_to(REPO_ROOT)),
                "port": int(args.port) + index,
                "server_log": str((logs_root / condition["condition_id"] / "server.log").relative_to(WORKSPACE_ROOT)),
                "client_log": str((logs_root / condition["condition_id"] / "client.log").relative_to(WORKSPACE_ROOT)),
                "status": "pending",
            }
        )

    manifest_path = (logs_root / "smoke_manifest.json") if args.smoke_only else (results_dir / "manifest.json")
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "title": config["title"],
        "created_at": _now(),
        "updated_at": _now(),
        "status": "planned",
        "task_suite_name": config["task_suite_name"],
        "task_id": int(config["task_id"]),
        "task_description": config["task_description"],
        "expected_trials_per_condition": int(config["num_trials_per_task"]),
        "num_trials_per_condition": 1 if args.smoke_only else int(config["num_trials_per_task"]),
        "condition_count": len(conditions),
        "baseline_condition_id": "V5_A",
        "global_noise_seed": int(config["global_noise_seed"]),
        "noise_seed_rule": "SeedSequence(global_noise_seed, episode_idx, replan_idx)",
        "condition_order_seed": condition_order_seed,
        "execution_order": [condition["condition_id"] for condition in conditions],
        "conditions": conditions,
        "server_python": server_python,
        "client_python": client_python,
        "server_gpu": server_gpu,
        "client_gpu": client_gpu,
        "results_dir": None if args.smoke_only else str(results_dir),
        "logs_root": str(logs_root),
        "runtime_backend": "triton",
    }

    for condition in conditions:
        server_args = dict(config["server_args"])
        server_args.update(
            {
                "port": condition["port"],
                "vlm_layer_pattern": condition["vlm_layer_pattern"],
                "action_layer_pattern": condition["action_layer_pattern"],
                "denoise_steps": condition["denoise_steps"],
                "ablation_noise_seed": int(config["global_noise_seed"]),
            }
        )
        condition["server_command"] = [server_python, "scripts/spec/spec_serve_policy.py", *_cli_args(server_args)]
        client_args = dict(config["client_args"])
        client_args.update(
            {
                "host": "127.0.0.1",
                "port": condition["port"],
                "video_out_path": str(data_root),
                "run_name": condition["run_id"],
                "num_trials_per_task": 1 if args.smoke_only else int(config["num_trials_per_task"]),
            }
        )
        condition["client_command"] = [client_python, "scripts/spec/spec_client_libero.py", *_cli_args(client_args)]

    if args.dry_run:
        for condition in conditions:
            print(f"[{condition['condition_id']}] {condition['vlm_layer_pattern']} × {condition['action_layer_pattern']} × {condition['denoise_steps']}")
            print("  server: " + shlex.join(condition["server_command"]))
            print("  client: " + shlex.join(condition["client_command"]))
        print(f"validated {len(conditions)} conditions; no processes started")
        return 0

    try:
        _require_cuda(server_python, server_gpu, "server")
        _require_cuda(client_python, client_gpu, "client")
    except RuntimeError as exc:
        print(f"preflight failed: {exc}")
        return 1

    for condition in conditions:
        _port_available(int(condition["port"]))
    # The experiment plan is intentionally kept in the result directory. Only
    # refuse to overwrite materialized run outputs from an earlier experiment.
    preserved = {"experiment_plan.md", "skip_layers_experiment_plan.md"}
    existing_result_outputs = (
        [path for path in results_dir.iterdir() if path.name not in preserved]
        if results_dir.exists()
        else []
    )
    if not args.smoke_only and existing_result_outputs:
        raise FileExistsError(f"results directory is not empty: {results_dir}")
    existing = [condition["run_dir"] for condition in conditions if (REPO_ROOT / condition["run_dir"]).exists()]
    if existing:
        raise FileExistsError("refusing to overwrite run directories: " + ", ".join(existing))

    if not args.smoke_only:
        preflight_path = logs_root / "preflight_v5_a.json"
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight = _run_v5_a_bitwise_preflight(
            server_python=server_python,
            server_gpu=server_gpu,
            config=config,
            output_path=preflight_path,
        )
        manifest["v5_a_baseline_preflight"] = {
            "report_path": str(preflight_path.relative_to(WORKSPACE_ROOT)),
            "bitwise_equal": bool(preflight["bitwise_equal"]),
            "max_abs_diff": preflight.get("max_abs_diff"),
        }

    _atomic_json(manifest_path, manifest)
    if not args.smoke_only:
        _atomic_json(
            results_dir / "execution_order.json",
            {
                "experiment_id": experiment_id,
                "condition_order_seed": condition_order_seed,
                "condition_ids": [condition["condition_id"] for condition in conditions],
            },
        )
    manifest["status"] = "running"
    _atomic_json(manifest_path, manifest)
    failures: list[str] = []
    for condition in conditions:
        condition_dir = logs_root / condition["condition_id"]
        condition_dir.mkdir(parents=True, exist_ok=True)
        server_log = condition_dir / "server.log"
        client_log = condition_dir / "client.log"
        server: subprocess.Popen | None = None
        condition["status"] = "running"
        condition["started_at"] = _now()
        _atomic_json(manifest_path, manifest)
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
                libero_pythonpath if not old_pythonpath else f"{libero_pythonpath}{os.pathsep}{old_pythonpath}"
            )
            with server_log.open("w", encoding="utf-8") as handle:
                server = subprocess.Popen(condition["server_command"], cwd=REPO_ROOT, env=server_env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            _wait_server(server, int(condition["port"]), args.start_timeout_s, server_log)
            with client_log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(condition["client_command"], cwd=REPO_ROOT, env=client_env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
            validation = _artifact_check(REPO_ROOT / condition["run_dir"], condition["run_id"], args.smoke_only)
            condition["artifact_validation"] = validation
            if result.returncode != 0:
                raise RuntimeError(f"client exited with {result.returncode}\n{_tail(client_log)}")
            if not validation["ok"]:
                raise RuntimeError("artifact validation failed: " + "; ".join(validation["errors"]))
            condition["status"] = "completed"
            condition["return_code"] = 0
            condition["finished_at"] = _now()
        except Exception as exc:
            condition["status"] = "failed"
            condition["error"] = str(exc)
            condition["finished_at"] = _now()
            failures.append(f"{condition['condition_id']}: {exc}")
        finally:
            _stop(server)
            manifest["updated_at"] = _now()
            _atomic_json(manifest_path, manifest)
        if failures and not (args.continue_on_error or args.smoke_only):
            break

    manifest["status"] = "completed" if not failures else "incomplete"
    manifest["updated_at"] = _now()
    manifest["failures"] = failures
    _atomic_json(manifest_path, manifest)
    if not args.smoke_only and not failures:
        summary_command = [
            server_python,
            "scripts/experiment3/summarize_skip_layers_experiment.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(results_dir),
        ]
        summary_result = subprocess.run(
            summary_command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        manifest["summary_command"] = summary_command
        manifest["summary_output"] = summary_result.stdout[-4000:]
        if summary_result.returncode != 0:
            failures.append(f"summary validation failed with exit code {summary_result.returncode}")
            manifest["status"] = "incomplete"
            manifest["failures"] = failures
        manifest["updated_at"] = _now()
        _atomic_json(manifest_path, manifest)
    if failures:
        print("\n".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
