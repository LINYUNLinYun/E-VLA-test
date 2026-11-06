"""Plan-first real LIBERO O0/O1 smoke. GPU/environment work requires --execute.

Frozen runtime sources are imported, never edited. One persistent server serves
one client episode at a time. This is an 8-case smoke per mode, not a latency
benchmark or an official 50-init evaluation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shlex
import socket
import struct
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
MODES = {"O0": "serial_reuse", "O1": "async_reuse"}
GPU_UUID = "GPU-599477fb-0507-40dc-14df-d491f7a7b202"


class DiscardVideoWriter:
    """Retain capture/telemetry calls without encoding or storing video frames."""
    def __init__(self):
        self.frames = 0

    def append_data(self, frame):
        self.frames += 1

    def close(self):
        pass


def client_gpu_ownership(snapshot, server_host_pids, child_container_pid):
    """Attribute one EGL client by before/active/after isolation, not PID equality."""
    if {process["pid"] for process in snapshot["processes"]} != server_host_pids:
        raise RuntimeError("Unexpected CUDA process during closed-loop smoke")
    graphics = snapshot["graphics_processes"]
    if len(graphics) > 1:
        raise RuntimeError("Expected at most one graphics process for the sole client")
    if not graphics:
        return None
    return dict(observed_graphics_host_pid=graphics[0]["pid"], client_container_pid=child_container_pid,
                namespace_caveat="nvidia-smi reports host PIDs; subprocess.pid is in this container. They need not match.",
                attribution="Exclusive GPU lease; zero G before client, one G while sole client runs, zero G after exit; server C set unchanged.",
                snapshot=snapshot)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_hashes():
    paths = [p for p in (ROOT / "preparations/idea2").glob("*.py") if not p.name.startswith("test_")]
    paths += [p for p in (ROOT / "preparations/idea1_v2_1").glob("*.py") if not p.name.startswith("test_")]
    paths += [ROOT / "policies/openpi/src/openpi" / relative
              for relative in ("training/config.py", "models_pytorch/pi0_pytorch.py", "models_pytorch/gemma_pytorch.py")]
    paths.append(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def episode_cases(reverse=False):
    rows = []
    for index, (suite, task) in enumerate((suite, task) for suite in SUITES for task in (0, 5)):
        ordered = ("O0", "O1") if (index + int(reverse)) % 2 == 0 else ("O1", "O0")
        for arm, label in enumerate(ordered):
            rows.append(dict(case=index, arm=arm, label=label, mode=MODES[label], suite=suite, task=task, init=0,
                             relative_directory=f"episodes/{label}/{suite}/task{task:02d}/init00"))
    return rows


def expected_protocol(mode=None):
    result = dict(H=10, R=5, num_steps=10, cache_slots=2, execution="cuda_graph", interval=1,
                  submission_order="ae_first", ae_priority=0, selective_overlap="vision_only",
                  await_previous_prefix=True, prepared_schedule=True, host_workers=0, diagnostics=True,
                  preparation_core="PreparedModulationCore")
    if mode is not None:
        result["mode"] = mode
    return result


def check_protocol(protocol, mode=None):
    for key, value in expected_protocol(mode).items():
        if protocol.get(key) != value:
            raise RuntimeError(f"Protocol mismatch {key}: {protocol.get(key)!r} != {value!r}")
    if "pi05=True" not in protocol.get("config", ""):
        raise RuntimeError("Expected the frozen pi05 model configuration")


def environments(args):
    common = dict(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                  JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", PYTHONUNBUFFERED="1")
    server = dict(common, CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_MAX_CONNECTIONS="32",
                  OPENPI_DATA_HOME=str(ROOT / "shared/checkpoints/openpi"),
                  PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "policies/openpi/src"))))
    client = dict(common, CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(args.egl_device),
                  LIBERO_CONFIG_PATH=str(ROOT / "policies/realtime-vla-flash/.libero"),
                  PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "LIBERO"),
                      str(ROOT / "policies/realtime-vla-flash/packages/openpi-client/src"))))
    return server, client


def dependency_probe(python, environment, names):
    code = ("import importlib.util,json; names=" + repr(names)
            + "; print(json.dumps({name:importlib.util.find_spec(name) is not None for name in names}))")
    try:
        result = subprocess.run([python, "-c", code], env=dict(os.environ, **(environment | {"CUDA_VISIBLE_DEVICES": ""})),
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        modules = json.loads(result.stdout) if result.returncode == 0 else {}
        return dict(passed=result.returncode == 0 and all(modules.values()), modules=modules,
                    returncode=result.returncode, stderr=result.stderr)
    except Exception as error:
        return dict(passed=False, error=repr(error))


def make_plan(args):
    server_env, client_env = environments(args)
    checkpoint = Path(args.checkpoint).resolve()
    command = [args.server_python, "-u", "-m", "preparations.idea2.server", "--checkpoint", str(checkpoint),
               "--model-config", "pi05_libero", "--num-steps", "10", "--cuda-graphs",
               "--submission-order", "ae_first", "--ae-priority", "0", "--selective-overlap", "vision_only",
               "--hoist-modulation", "--await-previous-prefix", "--diagnostics", "--port", str(args.port)]
    checks = dict(server=dependency_probe(args.server_python, server_env, ["torch", "numpy", "openpi", "transformers"]),
                  client=dependency_probe(args.client_python, client_env,
                                          ["numpy", "libero", "mujoco", "robosuite", "imageio", "PIL", "bddl"]),
                  checkpoint_exists=checkpoint.is_dir(),
                  weights_exists=(checkpoint / "model.safetensors").is_file(),
                  libero_config_exists=Path(client_env["LIBERO_CONFIG_PATH"]).is_dir())
    return dict(created_utc=now_utc(), status="planned", execute_required=True, cases=episode_cases(args.reverse_order),
                protocol=expected_protocol(), source_sha256=source_hashes(), checkpoint=str(checkpoint),
                checkpoint_files={path.name:dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                                  for path in checkpoint.glob("*") if path.is_file()},
                checkpoint_identity_note="Path and file metadata are recorded; no new full weights hash is claimed.",
                server=dict(argv=command, environment=server_env, python=args.server_python),
                client=dict(python=args.client_python, environment=client_env),
                gpu_uuid=args.gpu_uuid, egl_device=args.egl_device, port=args.port, dependency_checks=checks,
                env_seed=7, init=0, settling_steps=10, D=10, H=10, R=5,
                suite_step_limits=dict(zip(SUITES, (280, 220, 300, 520))),
                seed_rule="runtime_client.model_seed: sha256('idea1-v21/7/{suite}/{task}/{init}/{replan}'), first4bytes little endian",
                noise_identity="Per-request seed in infer.jsonl and inputs NPZ; GPU FP32 torch.randn((1,10,action_dim)) using a fresh CUDA Generator.manual_seed(seed). No noise tensor bytes/hash are exposed by this server.",
                schedule="Eight matched suite/task/init cases, alternating whole-episode O0/O1 order; one persistent service, one client at a time.",
                semantics="Both arms consume the complete previous observation KV after bootstrap. This is a separate old-KV algorithmic condition from fresh same-frame inference. Freely evolving closed-loop inputs may diverge between arms.",
                video_policy="No full videos: worker-scoped imageio writer discards frames, retaining original capture/telemetry and action execution calls. Result video field is explicitly null.",
                measurements="Native run_episode preserves actions, chunk NPZ, inputs and hashes, EEF telemetry, success, steps and request latency. Added all-work wall includes final drain and client/simulator/I/O overhead; it is not isolated GPU latency.",
                limitations=["8 episodes per mode is exploratory smoke, not official LIBERO evaluation.",
                             "CUDA_VISIBLE_DEVICES is empty in clients, but EGL rendering intentionally uses the allocated GPU.",
                             "EGL index2 corresponds to GPU1 in this host's current mapping; it is not a portable numbering rule.",
                             "First actual request may include Graph capture; report its cold cost separately, without bank-based warmup.",
                             "LIBERO pauses while inference runs; this smoke does not establish real-robot control frequency or camera SLO.",
                             "Server/model sources must stay frozen from plan through shutdown."])


def rpc(port, message, timeout=600):
    def exact(sock, count):
        value = bytearray()
        while len(value) < count:
            part = sock.recv(count - len(value))
            if not part:
                raise ConnectionError("RPC connection ended early")
            value.extend(part)
        return value
    payload = pickle.dumps(message, protocol=5)
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as client:
        client.sendall(struct.pack("!I", len(payload)) + payload)
        count = struct.unpack("!I", exact(client, 4))[0]
        if count > 128 * 1024 * 1024:
            raise ValueError("Oversized RPC response")
        response = pickle.loads(exact(client, count))
    if not response.get("ok"):
        raise RuntimeError(str(response))
    return response


def validate_artifacts(directory, client, case):
    import numpy as np
    result = json.loads((directory / "result.json").read_text())
    protocol = json.loads((directory / "native_protocol.json").read_text())
    check_protocol(protocol, case["mode"])
    infers = [json.loads(line) for line in (directory / "infer.jsonl").read_text().splitlines()]
    telemetry = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    if len(infers) != result["predictions"] or len(telemetry) != result["actions"] + 1:
        raise RuntimeError("Incomplete action/inference telemetry")
    for index, row in enumerate(infers):
        if row["seed"] != client.model_seed(case["suite"], case["task"], case["init"], index):
            raise RuntimeError("Model noise seed deviated from the existing protocol")
        if row["observation_request_age"] != (0 if index == 0 else 1):
            raise RuntimeError("Old-KV observation age contract failed")
        if row["graph_kv_buffers"] != 2 or row["pool"]["live_snapshots"] > 2:
            raise RuntimeError("Two-slot KV bound failed")
        actions = np.asarray(row["chunk_actions"])
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError("Invalid full H10 action chunk")
        if index and row["graph_setup_ms"] != 0:
            raise RuntimeError("Unexpected within-episode Graph recapture")
    if any("eef_pos" not in row or "eef_quat" not in row for row in telemetry):
        raise RuntimeError("Missing measured EEF state")
    def stats(values):
        return (dict(n=len(values), mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                     p95=float(np.percentile(values, 95))) if values else None)
    steady = [row for index, row in enumerate(infers) if index > 0 and row["graph_setup_ms"] == 0]
    return dict(valid=True, success=result["success"], steps=result["actions"], predictions=len(infers),
                cold_capture_requests=sum(row["graph_setup_ms"] > 0 for row in infers),
                first_input_hash=infers[0]["input_hash"] if infers else None,
                latency_scope="Real closed-loop smoke RPC timings with rendering and recording; excludes bootstrap/capture only in steady_oldkv fields. Not formal isolated GPU performance.",
                all_request_roundtrip_ms=stats([row["client_roundtrip_ms"] for row in infers]),
                steady_oldkv_roundtrip_ms=stats([row["client_roundtrip_ms"] for row in steady]),
                steady_oldkv_action_ready_ms=stats([row["timing"]["action_ready_host_ms"] for row in steady]),
                runtime_result=result, noise_identity="Seeds recorded, not noise tensor byte hashes")


def run_worker(case_path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("LIBERO client must have CUDA_VISIBLE_DEVICES empty")
    sys.path.insert(0, str(ROOT))
    from preparations.idea1_v2_1 import runtime_client as client
    import imageio.v2 as imageio
    from unittest.mock import patch
    case = json.loads(Path(case_path).read_text())
    directory = Path(case["directory"])
    receipt = dict(status="running", case=case, start_utc=now_utc())
    begin = time.perf_counter()
    failure = None
    try:
        check_protocol(client.rpc(case["port"], dict(op="health"))["protocol"], case["mode"])
        # Original code owns environment seed/init, settling, model seeds,
        # H10 prediction validation, R5 execution, termination, and all telemetry.
        writer = DiscardVideoWriter()
        with patch.object(imageio, "get_writer", return_value=writer):
            native_result = client.run_episode(directory, case["port"], "pi05", case["suite"], case["task"], case["init"], [],
                                               kind="idea2_pi05_d10_oldkv_smoke", reference=False)
        native_result.update(video=None, video_recorded=False, video_frames_discarded=writer.frames,
                             video_policy="DiscardVideoWriter: capture, observation, telemetry and actions unchanged")
        write_json(directory / "result.json", native_result)
        receipt["response_episode_wall_s"] = time.perf_counter() - begin
    except BaseException as error:
        failure = error
        receipt.update(status="technical_failure", error=repr(error), traceback=traceback.format_exc())
    finally:
        drain_begin = time.perf_counter()
        try:
            receipt["drain"] = client.rpc(case["port"], dict(op="drain"))
            receipt["drain_wall_s"] = time.perf_counter() - drain_begin
        except BaseException as error:
            receipt.update(status="technical_failure", drain_error=repr(error))
            failure = failure or error
        receipt["all_work_wall_s"] = time.perf_counter() - begin
        receipt["end_utc"] = now_utc()
        # A failed episode can terminate before run_episode creates its leaf.
        write_json(directory.parent / (directory.name + ".worker_receipt.json"), receipt)
    if failure is not None:
        raise failure
    try:
        validation = validate_artifacts(directory, client, case)
    except BaseException as error:
        receipt.update(status="technical_failure", validation_error=repr(error), traceback=traceback.format_exc())
        write_json(directory / "smoke_receipt.json", receipt)
        raise
    receipt.update(status="passed", validation=validation)
    write_json(directory / "smoke_receipt.json", receipt)
    write_json(directory / "complete.json", dict(status="valid", **validation,
        all_work_wall_s=receipt["all_work_wall_s"], drain_wall_s=receipt["drain_wall_s"]))
    return receipt


def execute(args, plan, dest):
    sys.path.insert(0, str(ROOT))
    from preparations.idea2.layer_bench import require_idle_gpu, require_same_processes, gpu_snapshot
    if (dest / "execution.json").exists():
        raise RuntimeError("Execution output already exists; use a fresh output for reruns")
    checks = plan["dependency_checks"]
    if not (checks["server"]["passed"] and checks["client"]["passed"] and checks["checkpoint_exists"]
            and checks["weights_exists"] and checks["libero_config_exists"]):
        raise RuntimeError("Dependency checks failed; inspect plan.json")
    if source_hashes() != plan["source_sha256"]:
        raise RuntimeError("Frozen sources differ from plan")
    with socket.socket() as test:
        test.bind(("127.0.0.1", args.port))
    state = dict(status="running", start_utc=now_utc(), cases=[], performance_result=False,
                 gpu_before_init=require_idle_gpu(args.gpu_uuid))
    save = lambda: write_json(dest / "execution.json", state)
    save()
    server = None
    ready = False
    try:
        with (dest / "server.log").open("x") as log:
            server = subprocess.Popen(plan["server"]["argv"], cwd=ROOT,
                env=dict(os.environ, **plan["server"]["environment"]), stdout=log, stderr=subprocess.STDOUT)
        state["server_pid"] = server.pid
        save()
        deadline = time.monotonic() + args.server_ready_timeout
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("Server exited during loading")
            try:
                protocol = rpc(args.port, dict(op="health"), timeout=2)["protocol"]
                check_protocol(protocol)
                ready = True
                break
            except (ConnectionError, OSError, TimeoutError):
                time.sleep(0.5)
        if not ready:
            raise TimeoutError("Server readiness timed out")
        state["protocol"] = protocol
        occupancy = gpu_snapshot(args.gpu_uuid)
        own_pids = {row["pid"] for row in occupancy["processes"]}
        if len(own_pids) != 1 or occupancy["graphics_processes"]:
            raise RuntimeError("Expected one owned model process before rendering")
        state["gpu_after_load"] = occupancy
        save()
        for case in plan["cases"]:
            if source_hashes() != plan["source_sha256"]:
                raise RuntimeError("Source changed while the service was running")
            row = dict(case, status="running", start_utc=now_utc(),
                       gpu_before=require_same_processes(args.gpu_uuid, own_pids))
            state["cases"].append(row)
            row["configure"] = rpc(args.port, dict(op="configure", mode=case["mode"], interval=1,
                submission_order="ae_first", cuda_graphs=True, ae_priority=0, diagnostics=True, host_workers=False))
            check_protocol(row["configure"]["protocol"], case["mode"])
            case_path = dest / "requests" / f"case{case['case']:02d}_{case['label']}.json"
            write_json(case_path, dict(case, directory=str(dest / case["relative_directory"]), port=args.port))
            command = [args.client_python, "-u", str(Path(__file__).resolve()), "--execute", "--worker-case", str(case_path)]
            row["command"] = shlex.join(command)
            client_log = dest / "logs" / (case_path.stem + ".log")
            client_log.parent.mkdir(exist_ok=True)
            started = time.monotonic()
            with client_log.open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **plan["client"]["environment"]),
                                         stdout=log, stderr=subprocess.STDOUT)
                row["client_pid"] = child.pid
                save()
                try:
                    render_deadline = started + 90
                    while child.poll() is None:
                        if time.monotonic() - started > args.episode_timeout:
                            raise TimeoutError("Episode subprocess timeout")
                        snapshot = gpu_snapshot(args.gpu_uuid)
                        ownership = client_gpu_ownership(snapshot, own_pids, child.pid)
                        if "render_gpu_verified" not in row:
                            if ownership is not None:
                                row["render_gpu_verified"] = ownership
                                save()
                            elif time.monotonic() > render_deadline:
                                raise RuntimeError("Client EGL context not observed on assigned GPU; verify EGL index mapping")
                        time.sleep(1)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=15)
                row["client_exit_code"] = child.returncode
            row["client_process_wall_s"] = time.monotonic() - started
            row["parent_drain"] = rpc(args.port, dict(op="drain"))
            row["gpu_after"] = require_same_processes(args.gpu_uuid, own_pids)
            complete = dest / case["relative_directory"] / "complete.json"
            if child.returncode != 0 or not complete.exists() or "render_gpu_verified" not in row:
                raise RuntimeError("Episode technically invalid; retain logs and do not count it as task failure")
            row["result"] = json.loads(complete.read_text())
            row["status"] = "valid"
            row["end_utc"] = now_utc()
            save()
        state["summary"] = {label:dict(episodes=8,
            successes=sum(row["result"]["success"] for row in state["cases"] if row["label"] == label),
            executed_steps=sum(row["result"]["steps"] for row in state["cases"] if row["label"] == label))
            for label in MODES}
        state["status"] = "passed"
    except BaseException as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        for row in state["cases"]:
            if row["status"] == "running":
                row["status"] = "technical_failure"
        raise
    finally:
        if server is not None:
            if ready and server.poll() is None:
                try:
                    state["shutdown"] = rpc(args.port, dict(op="shutdown"), timeout=120)
                except Exception as error:
                    state["shutdown_error"] = repr(error)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=15)
            state["server_exit_code"] = server.returncode
            if server.returncode != 0:
                state["status"] = "failed"
        state["source_unchanged"] = source_hashes() == plan["source_sha256"]
        if not state["source_unchanged"]:
            state["status"] = "excluded_source_drift"
        state["end_utc"] = now_utc()
        save()
    if state["status"] != "passed":
        raise RuntimeError("Smoke shutdown/provenance failed; inspect execution.json")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--output", default=str(HERE / "closedloop_d10_o0_o1"))
    result.add_argument("--checkpoint", default=str(ROOT / "shared/checkpoints/openpi/pi05_libero_pytorch_idea2"))
    result.add_argument("--gpu-uuid", default=GPU_UUID)
    result.add_argument("--egl-device", type=int, default=2)
    result.add_argument("--port", type=int, default=19082)
    result.add_argument("--server-python", default=str(ROOT / "policies/openpi/.venv/bin/python"))
    result.add_argument("--client-python", default=str(ROOT / "policies/libero-eval-v21/.venv/bin/python"))
    result.add_argument("--reverse-order", action="store_true")
    result.add_argument("--server-ready-timeout", type=float, default=600)
    result.add_argument("--episode-timeout", type=float, default=1800)
    result.add_argument("--worker-case", help=argparse.SUPPRESS)
    return result


def main():
    args = parser().parse_args()
    if not (0 < args.port < 65536 and args.egl_device >= 0
            and args.server_ready_timeout > 0 and args.episode_timeout > 0):
        raise ValueError("Invalid port, EGL device or timeout")
    if args.worker_case:
        if not args.execute:
            raise RuntimeError("Worker execution also requires explicit --execute")
        run_worker(args.worker_case)
        return
    dest = Path(args.output).resolve()
    if not args.execute:
        if (dest / "execution.json").exists():
            raise RuntimeError("Refusing to modify an existing execution's plan")
        plan = make_plan(args)
        write_json(dest / "plan.json", plan)
        print(json.dumps(dict(status="planned_only", cases=len(plan["cases"]),
                              plan=str(dest / "plan.json"), dependency_checks=plan["dependency_checks"]), indent=2))
        return
    # Explicit execution may create a new plan, or consume a previously reviewed
    # one. Require command/env/case agreement; never silently replace old plans.
    candidate = make_plan(args)
    plan_path = dest / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else candidate
    for key in ("server", "client", "cases", "checkpoint", "checkpoint_files", "gpu_uuid", "egl_device", "port", "source_sha256"):
        if plan[key] != candidate[key]:
            raise RuntimeError(f"Execution differs from saved plan ({key}); create a new output plan")
    plan["dependency_checks"] = candidate["dependency_checks"]
    write_json(plan_path, plan)
    execute(args, plan, dest)


if __name__ == "__main__":
    main()
