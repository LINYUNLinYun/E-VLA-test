"""Run the five-condition paired VLM pruning evaluation over a frozen testset."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import socket
import subprocess
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent.parent

DEFAULT_CONFIG = Path(__file__).with_name("experiment5_config.json")
DEFAULT_TESTSET = Path(__file__).with_name("testset_manifest.json")
DEFAULT_RESULTS = REPO_ROOT / "results/pi0_vlm_backbone_fine_pruning"
DEFAULT_LOGS = WORKSPACE_ROOT / "logs/pi0_vlm_backbone_fine_pruning"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _resolve(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item) for item in value]
    if isinstance(value, str):
        return value.replace("$WORKSPACE_ROOT", str(WORKSPACE_ROOT)).replace("$REPO_ROOT", str(REPO_ROOT))
    return value


def _python(value: str | None, preferred: Path) -> str:
    path = Path(value or preferred).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"python executable not found: {path}")
    return str(path.absolute())


def _require_cuda(python: str, gpu: str, role: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    probe = subprocess.run([python, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 2)"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if probe.returncode:
        detail = (probe.stderr or probe.stdout).strip().splitlines()[-1:]
        raise RuntimeError(f"{role} CUDA preflight failed{': ' + detail[0] if detail else ''}")


def _port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError as exc:
            raise RuntimeError(f"port {port} is unavailable") from exc


def _tail(path: Path, count: int = 40) -> str:
    if not path.is_file():
        return "<log missing>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _read_inputs(config_path: Path, testset_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str, str]:
    config_bytes = config_path.read_bytes()
    testset_bytes = testset_path.read_bytes()
    config = json.loads(config_bytes)
    testset = json.loads(testset_bytes)
    samples = testset.get("samples", testset.get("tasks"))
    if not isinstance(samples, list) or len(samples) != 100:
        raise ValueError(f"testset must contain exactly 100 task samples; found {len(samples) if isinstance(samples, list) else 'invalid'}")
    seen_tasks: set[tuple[str, int]] = set()
    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"testset sample {index} must be an object")
        suite = sample.get("suite_name", sample.get("task_suite_name", sample.get("suite")))
        task_id = sample.get("task_id")
        if not isinstance(suite, str) or not suite or not isinstance(task_id, int):
            raise ValueError(f"testset sample {index} must have suite_name and integer task_id")
        key = (suite, int(task_id))
        if key in seen_tasks:
            raise ValueError(f"duplicate task in testset: {suite}:{task_id}")
        sample_id = str(sample.get("sample_id", sample.get("id", f"{suite}:{task_id}")))
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen_tasks.add(key)
        seen_ids.add(sample_id)
        state = sample.get("init_state")
        state_path = sample.get("init_state_path", sample.get("init_state_file", sample.get("initial_state_path")))
        if isinstance(state, dict) and state_path is None:
            state_path = state.get("path", state.get("file"))
            state = state.get("data")
        if state is None and not state_path:
            raise ValueError(f"sample {sample_id} lacks an initial state")
        if state_path:
            resolved_state = Path(str(state_path))
            if not resolved_state.is_absolute():
                resolved_state = testset_path.parent / resolved_state
            if not resolved_state.is_file():
                raise FileNotFoundError(f"initial state missing for {sample_id}: {resolved_state}")
        normalized.append({**sample, "sample_index": index, "sample_id": sample_id, "suite_name": suite, "task_id": int(task_id)})
    manifest_noise_seed = testset.get("seed_policy", {}).get("noise_seed")
    if manifest_noise_seed is not None and int(config.get("global_noise_seed", -1)) != int(manifest_noise_seed):
        raise ValueError("config global_noise_seed differs from frozen testset seed_policy.noise_seed")
    sample_noise_seeds = {int(item["noise_seed"]) for item in normalized if item.get("noise_seed") is not None}
    if sample_noise_seeds and sample_noise_seeds != {int(config["global_noise_seed"])}:
        raise ValueError(f"sample noise_seed values do not match config global_noise_seed: {sample_noise_seeds}")
    manifest_env_seed = testset.get("seed_policy", {}).get("env_seed")
    sample_env_seeds = {int(item["env_seed"]) for item in normalized if item.get("env_seed") is not None}
    if manifest_env_seed is not None and sample_env_seeds and sample_env_seeds != {int(manifest_env_seed)}:
        raise ValueError("sample env_seed values differ from frozen testset seed_policy.env_seed")
    if len(config.get("conditions", [])) != 5:
        raise ValueError("experiment config must define exactly five conditions, including Control")
    condition_ids = [str(item["condition_id"]) for item in config["conditions"]]
    if len(set(condition_ids)) != 5 or "Control" not in condition_ids:
        raise ValueError("condition IDs must be unique and include Control")
    for item in config["conditions"]:
        if item.get("action_layer_pattern", config.get("action_layer_pattern")) != "full18" or int(item.get("denoise_steps", config.get("denoise_steps", 0))) != 3:
            raise ValueError(f"{item['condition_id']} must use full18 Action Expert and 3 denoising steps")
        if not item.get("vlm_layer_indices_1based"):
            raise ValueError(f"{item['condition_id']} is missing its explicit VLM layer list")
    if config["conditions"][-1]["condition_id"] != "Control" or config["conditions"][-1].get("vlm_layer_pattern") != "vlm_full18":
        # Keep the condition identifiable; randomization happens after manifest creation.
        raise ValueError("Control must be configured with vlm_full18")
    suite_counts = {suite: sum(item["suite_name"] == suite for item in normalized) for suite in {item["suite_name"] for item in normalized}}
    if suite_counts != {"libero_90": 90, "libero_10": 10}:
        raise ValueError(f"testset must contain 90 libero_90 and 10 libero_10 tasks; found {suite_counts}")
    return config, testset, normalized, _sha256_bytes(config_bytes), _sha256_bytes(testset_bytes)


def _validate_run_artifacts(run_dir: Path, run_id: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = run_dir / "manifest.json"
    episodes_path = run_dir / "episode_log.json"
    if not manifest_path.is_file() or not episodes_path.is_file():
        return {"ok": False, "errors": ["missing client manifest or episode_log.json"]}
    client_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    if client_manifest.get("run_id") != run_id:
        errors.append("client run_id mismatch")
    expected = {sample["sample_id"]: sample for sample in samples}
    actual = {str(row.get("sample_id")): row for row in episodes if isinstance(row, dict)} if isinstance(episodes, list) else {}
    if not isinstance(episodes, list) or len(episodes) != len(samples):
        errors.append(f"episode count={len(episodes) if isinstance(episodes, list) else 'invalid'}, expected={len(samples)}")
    if set(actual) != set(expected):
        errors.append(f"sample ID mismatch: missing={sorted(set(expected) - set(actual))[:5]}, extra={sorted(set(actual) - set(expected))[:5]}")
    for sample_id, sample in expected.items():
        row = actual.get(sample_id)
        if row is None:
            continue
        if row.get("suite_name") != sample["suite_name"] or int(row.get("task_id", -1)) != int(sample["task_id"]):
            errors.append(f"task identity mismatch for {sample_id}")
        if row.get("noise_episode_idx") != sample["sample_index"]:
            errors.append(f"noise episode index mismatch for {sample_id}")
        for key in ("video_path", "trace_path", "infer_path"):
            relative = Path(str(row.get(key, "")))
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{sample_id}: invalid {key}")
                continue
            artifact = (run_dir / relative).resolve()
            if not artifact.is_relative_to(run_dir.resolve()) or not artifact.is_file() or artifact.stat().st_size == 0:
                errors.append(f"{sample_id}: missing {key}={relative}")
    return {
        "ok": not errors, "errors": errors, "episode_count": len(episodes) if isinstance(episodes, list) else 0,
        "successes": sum(int(bool(row.get("success"))) for row in actual.values()),
    }


def _smoke_samples(samples: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    scenes: set[tuple[str, str]] = set()
    for sample in samples:
        task_name = str(sample.get("task_name", sample.get("task_description", sample["sample_id"])))
        match = re.match(r"^(.*?SCENE\d+)", task_name, flags=re.IGNORECASE)
        scene = match.group(1).lower() if match else task_name.lower()
        scene_key = (str(sample["suite_name"]), scene)
        if scene_key in scenes:
            continue
        selected.append(sample)
        scenes.add(scene_key)
        if len(selected) == count:
            return selected
    raise ValueError(f"smoke test needs {count} distinct scene groups; found {len(selected)}")


def _cli_args(values: dict[str, Any]) -> list[str]:
    output: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                output.append(flag)
        elif isinstance(value, (list, tuple)):
            output.extend([flag, *(str(item) for item in value)])
        else:
            output.extend([flag, str(value)])
    return output


def _wait_server(process: subprocess.Popen, port: int, timeout: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited {process.returncode}: {_tail(log_path)}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"server did not listen on port {port}: {_tail(log_path)}")


def _run_condition(
    condition: dict[str, Any], *, manifest: dict[str, Any], manifest_path: Path,
    samples: list[dict[str, Any]], config: dict[str, Any], testset_path: Path,
    server_python: str, client_python: str, server_gpu: str, client_gpu: str,
    results_dir: Path, logs_root: Path, smoke_only: bool, resume: bool,
    start_timeout_s: float,
) -> None:
    condition_id = condition["condition_id"]
    condition_log = logs_root / condition_id
    condition_log.mkdir(parents=True, exist_ok=True)
    server_log, client_log = condition_log / "server.log", condition_log / "client.log"
    port = int(condition["port"])
    run_id = condition["run_id"]
    run_dir = REPO_ROOT / condition["run_dir"]
    _port_available(port)
    server_args = {
        **config.get("server_args", {}), "port": port,
        "vlm_layer_pattern": condition["vlm_layer_pattern"],
        "action_layer_pattern": "full18", "denoise_steps": 3,
        "ablation_noise_seed": int(config["global_noise_seed"]),
    }
    server_args = _resolve(server_args)
    server_command = [server_python, "scripts/spec/spec_serve_policy.py", *_cli_args(server_args)]
    client_command = [
        client_python, "scripts/experiment5/eval_cross_task.py",
        "--testset-manifest", str(testset_path), "--host", "127.0.0.1", "--port", str(port),
        "--video-out-path", str(REPO_ROOT / "data/test"), "--run-name", run_id,
        "--condition-id", condition_id, "--vlm-layer-pattern", condition["vlm_layer_pattern"],
        "--global-noise-seed", str(config["global_noise_seed"]), "--seed", str(config.get("seed", 7)),
        "--num-steps-wait", str(config.get("num_steps_wait", 10)),
        "--replan-steps", str(config.get("replan_steps", 12)),
        "--resize-size", str(config.get("resize_size", 224)),
    ]
    if smoke_only:
        client_command.extend(["--sample-indices", ",".join(str(sample["sample_index"]) for sample in samples)])
    if resume:
        client_command.append("--resume")
    condition.update({
        "server_command": server_command, "client_command": client_command,
        "status": "running", "started_at": _now(),
    })
    manifest["updated_at"] = _now()
    _atomic_json(manifest_path, manifest)
    server: subprocess.Popen | None = None
    try:
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = server_gpu
        client_env = os.environ.copy()
        client_env["CUDA_VISIBLE_DEVICES"] = client_gpu
        client_env["MUJOCO_GL"] = "egl"
        client_env["LIBERO_CONFIG_PATH"] = str(REPO_ROOT / ".libero")
        libero_pythonpath = str(REPO_ROOT / "third_party/libero")
        old_pythonpath = client_env.get("PYTHONPATH")
        client_env["PYTHONPATH"] = libero_pythonpath if not old_pythonpath else f"{libero_pythonpath}{os.pathsep}{old_pythonpath}"
        with server_log.open("w", encoding="utf-8") as stream:
            server = subprocess.Popen(server_command, cwd=REPO_ROOT, env=server_env, stdout=stream, stderr=subprocess.STDOUT, text=True)
        _wait_server(server, port, start_timeout_s, server_log)
        with client_log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(client_command, cwd=REPO_ROOT, env=client_env, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
        validation = _validate_run_artifacts(run_dir, run_id, samples)
        condition["artifact_validation"] = validation
        if result.returncode != 0:
            raise RuntimeError(f"client exited {result.returncode}; see {client_log}")
        if not validation["ok"]:
            raise RuntimeError("artifact validation failed: " + "; ".join(validation["errors"][:10]))
        condition["status"] = "completed"
        condition["return_code"] = 0
    except Exception as exc:
        condition["status"] = "failed"
        condition["error"] = str(exc)
        raise
    finally:
        _stop(server)
        condition["finished_at"] = _now()
        manifest["updated_at"] = _now()
        _atomic_json(manifest_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--testset-manifest", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--experiment-id")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--server-python", default=os.environ.get("PI0_ABLATION_SERVER_PYTHON"))
    parser.add_argument("--client-python", default=os.environ.get("PI0_ABLATION_CLIENT_PYTHON"))
    parser.add_argument("--server-gpu", default="0")
    parser.add_argument("--client-gpu", default="1")
    parser.add_argument("--port-base", type=int, default=8720)
    parser.add_argument("--start-timeout-s", type=float, default=600.0)
    parser.add_argument("--smoke-only", action="store_true", help="Run five samples per condition.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    testset_path = args.testset_manifest.expanduser().resolve()
    config, testset, all_samples, config_hash, testset_hash = _read_inputs(config_path, testset_path)
    smoke_sample_indices: list[int] | None = None
    if args.smoke_only:
        samples = _smoke_samples(all_samples, int(config.get("smoke_sample_count", 5)))
        smoke_sample_indices = [int(item["sample_index"]) for item in samples]
    else:
        samples = all_samples
    experiment_id = args.experiment_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = args.results_dir.expanduser().resolve()
    logs_root = args.logs_root.expanduser().resolve() / experiment_id
    server_python = _python(args.server_python, REPO_ROOT / ".venv/bin/python")
    client_python = _python(args.client_python, REPO_ROOT / "examples/libero/.venv/bin/python")
    order_seed = int(config.get("condition_order_seed", 20261002))
    conditions = [dict(item) for item in config["conditions"]]
    random.Random(order_seed).shuffle(conditions)
    manifest_path = logs_root / "smoke_manifest.json" if args.smoke_only else results_dir / "manifest.json"
    old_manifest: dict[str, Any] | None = None
    if args.resume:
        if args.smoke_only or not manifest_path.is_file():
            raise FileNotFoundError(f"no formal manifest to resume: {manifest_path}")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("config_sha256") != config_hash or old_manifest.get("testset_sha256") != testset_hash:
            raise ValueError("cannot resume: experiment config or testset changed")
        experiment_id = str(old_manifest["experiment_id"])
        logs_root = args.logs_root.expanduser().resolve() / experiment_id
        previous = {item["condition_id"]: item for item in old_manifest.get("conditions", [])}
        for condition in conditions:
            if condition["condition_id"] in previous:
                condition.update(previous[condition["condition_id"]])
    else:
        existing = [p for p in results_dir.iterdir()] if results_dir.exists() and not args.smoke_only else []
        if existing:
            raise FileExistsError(f"results directory is not empty: {results_dir}; use --resume for an existing run")

    for index, condition in enumerate(conditions):
        condition.setdefault("status", "pending")
        condition.setdefault("port", args.port_base + index)
        condition.setdefault("run_id", f"pi0_experiment5_{'smoke_' if args.smoke_only else ''}{experiment_id}_{condition['condition_id']}")
        condition.setdefault("run_dir", str(Path("data/test") / condition["run_id"]))
        condition.setdefault("is_control", condition["condition_id"] == "Control")
    manifest = old_manifest or {
        "experiment_id": experiment_id, "title": config["title"], "created_at": _now(),
        "updated_at": _now(), "status": "planned", "config_path": str(config_path),
        "config_sha256": config_hash, "testset_manifest": str(testset_path), "testset_sha256": testset_hash,
        "total_testset_samples": len(all_samples), "selected_sample_count": len(samples),
        "selected_sample_ids": [sample["sample_id"] for sample in samples],
        "smoke_only": bool(args.smoke_only), "condition_count": len(conditions),
        "baseline_condition_id": "Control", "global_noise_seed": int(config["global_noise_seed"]),
        "noise_seed_rule": "SeedSequence(global_noise_seed, sample_index, infer_id)",
        "condition_order_seed": order_seed, "execution_order": [item["condition_id"] for item in conditions],
        "conditions": conditions, "server_python": server_python, "client_python": client_python,
        "server_gpu": str(args.server_gpu), "client_gpu": str(args.client_gpu), "logs_root": str(logs_root),
    }
    # On resume, ``conditions`` contains updated copies of the manifest
    # entries.  Keep the manifest bound to those same dictionaries so status,
    # commands, and timestamps written during subsequent conditions are not
    # lost when the manifest is atomically rewritten.
    manifest["conditions"] = conditions
    if args.dry_run:
        for condition in conditions:
            server_args = _resolve({
                **config.get("server_args", {}), "port": condition["port"],
                "vlm_layer_pattern": condition["vlm_layer_pattern"],
                "action_layer_pattern": "full18", "denoise_steps": 3,
                "ablation_noise_seed": int(config["global_noise_seed"]),
            })
            server_command = [server_python, "scripts/spec/spec_serve_policy.py", *_cli_args(server_args)]
            client_command = [
                client_python, "scripts/experiment5/eval_cross_task.py", "--testset-manifest", str(testset_path),
                "--host", "127.0.0.1", "--port", str(condition["port"]), "--run-name", condition["run_id"],
                "--condition-id", condition["condition_id"], "--vlm-layer-pattern", condition["vlm_layer_pattern"],
                "--global-noise-seed", str(config["global_noise_seed"]),
            ]
            if args.smoke_only:
                client_command.extend(["--sample-indices", ",".join(str(value) for value in smoke_sample_indices or [])])
            print(f"[{condition['condition_id']}] {len(samples)} paired tasks; {condition['vlm_layer_pattern']}")
            print("  server: " + shlex.join(server_command))
            print("  client: " + shlex.join(client_command))
        print("validated five conditions and 100-task manifest; no processes started")
        return 0

    if not args.resume:
        _atomic_json(manifest_path, manifest)
    try:
        _require_cuda(server_python, str(args.server_gpu), "server")
        if str(args.client_gpu) != str(args.server_gpu):
            _require_cuda(client_python, str(args.client_gpu), "client")
    except RuntimeError as exc:
        print(f"CUDA preflight failed: {exc}")
        return 1
    failures: list[str] = []
    for condition in conditions:
        if condition.get("status") == "completed":
            validation = _validate_run_artifacts(REPO_ROOT / condition["run_dir"], condition["run_id"], samples)
            if validation["ok"]:
                continue
            condition["status"] = "failed"
            condition["error"] = "completed artifacts invalid: " + "; ".join(validation["errors"][:5])
        condition_resume = args.resume and (REPO_ROOT / condition["run_dir"] / "manifest.json").is_file()
        try:
            _run_condition(
                condition, manifest=manifest, manifest_path=manifest_path, samples=samples,
                config=config, testset_path=testset_path, server_python=server_python,
                client_python=client_python, server_gpu=str(args.server_gpu), client_gpu=str(args.client_gpu),
                results_dir=results_dir, logs_root=logs_root, smoke_only=args.smoke_only,
                resume=condition_resume, start_timeout_s=args.start_timeout_s,
            )
        except Exception as exc:
            failures.append(f"{condition['condition_id']}: {exc}")
            if not args.continue_on_error:
                break
    manifest["status"] = "incomplete" if failures or any(item.get("status") != "completed" for item in conditions) else "completed"
    manifest["failures"] = failures
    manifest["updated_at"] = _now()
    _atomic_json(manifest_path, manifest)
    if failures or args.smoke_only or manifest["status"] != "completed":
        print("\n".join(failures) if failures else f"run status: {manifest['status']}")
        return 1 if failures or manifest["status"] != "completed" else 0
    summary = subprocess.run(
        [server_python, "scripts/experiment5/summarize_cross_task_ablation.py", "--manifest", str(manifest_path), "--output-dir", str(results_dir)],
        cwd=REPO_ROOT, check=False,
    )
    return summary.returncode


if __name__ == "__main__":
    raise SystemExit(main())
