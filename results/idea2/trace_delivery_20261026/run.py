"""Collect the requested three traces sequentially on an exclusively idle GPU3."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

R = Path(__file__).resolve().parent
ROOT = R.parents[2]
GPU = "GPU-555cde36-5e41-bc44-0c69-ea47358fd703"
LEASES = ROOT / "results/idea1_v3_3/external_gpu_leases.json"
OWNER = "user-authorized idea2 three-path nsys delivery"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    p = Path(path)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(p)


def lease(active, snapshot):
    with LEASES.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        values = json.load(f)
        current = values.get("3", {})
        if active and current.get("active"):
            raise RuntimeError("GPU3 has an active lease")
        if not active and current.get("owner") != OWNER:
            raise RuntimeError("GPU lease ownership changed")
        values["3"] = dict(gpu=3, uuid=GPU, owner=OWNER, active=active,
            owner_pid=os.getpid(), time_utc=now(), snapshot=snapshot,
            purpose="Three bounded nsys captures, not a new performance/quality study")
        f.seek(0); json.dump(values, f, indent=2); f.write("\n"); f.truncate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=("native_jax", "fresh_prepared", "pipeline"),
                   default=("native_jax", "fresh_prepared", "pipeline"))
    p.add_argument("--receipt-name", default="execution.json")
    p.add_argument("--stage-suffix", default="")
    p.add_argument("--pipeline-alignment", choices=("off", "control_steps"), default="control_steps")
    args = p.parse_args()
    receipt_path = R / args.receipt_name
    from preparations.idea2.layer_bench import require_idle_gpu, gpu_snapshot
    from preparations.idea1_v2_1.runtime_client import rpc
    from results.idea2.d10_dynamic_sm_20261024.closedloop_smoke import client_gpu_ownership
    if receipt_path.exists():
        raise RuntimeError("Do not rerun a completed or in-flight capture collection")
    os.sched_setaffinity(0, set(range(96, 112)))
    prior = json.loads((ROOT / "results/idea2/closedloop200_20261025/gpu2_chunk0/plan.json").read_text())
    env = dict(os.environ, **prior["server"]["environment"])
    env.update(CUDA_VISIBLE_DEVICES=GPU, PROGRESSIVE_EXPECTED_CPUS=",".join(map(str, range(96,112))))
    client_env = dict(os.environ, **prior["client"]["environment"])
    client_env.update(MUJOCO_EGL_DEVICE_ID="0", PROGRESSIVE_EXPECTED_CPUS=env["PROGRESSIVE_EXPECTED_CPUS"])
    py = prior["server"]["python"]
    dest = ROOT / "nsys/idea2_experiment"
    names = dict(native_jax="pi05_threepaths_01_native_jax_d10_20261026",
                 fresh_prepared="pi05_threepaths_02_fresh_prepared_d10_20261026",
                 pipeline="pi05_threepaths_03_pipeline_sm16_lead3_d10_20261026")
    state = dict(status="starting", start_utc=now(), gpu_uuid=GPU, stages=[],
        note="Native is fixed recorded-observation replay; other two are real continuous LIBERO episodes. Trace timing is diagnostic, not replacement for profiler-free measurements.")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    frozen_paths = set(json.loads((ROOT / "results/idea2/control_alignment100_20261026/manifest.json").read_text())["source_sha256"])
    frozen = {p:sha(ROOT/p) for p in sorted(frozen_paths)}
    state["source_before"] = frozen
    claimed = False
    server = child = None
    try:
        lease(True, require_idle_gpu(GPU)); claimed = True
        for kind in args.only:
            require_idle_gpu(GPU)
            out = R / (kind + args.stage_suffix)
            out.mkdir(exist_ok=False)
            prefix = dest / names[kind]
            if Path(str(prefix)+".nsys-rep").exists():
                raise RuntimeError("Trace already exists")
            command = ["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node:nvtx-precapture",
                "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop", "--kill=none", "--force-overwrite=false",
                "--output="+str(prefix), py, "-u", "-m", "results.idea2.trace_delivery_20261026.capture",
                "--kind", kind, "--receipt", str(out/"capture.json")]
            stage = dict(kind=kind, status="starting", start_utc=now(), command=command,
                         trace=str(prefix)+".nsys-rep")
            state["stages"].append(stage); save(receipt_path, state)
            if kind == "native_jax":
                command += ["--output", str(out/"native_result"), "--steps", "10", "--warmup", "10", "--count", "100", "--noise-period", "13"]
                native_env = dict(env, JAX_PLATFORMS="cuda", JAX_PLATFORM_NAME="gpu")
                with (out/"console.log").open("x") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=native_env, stdout=log, stderr=subprocess.STDOUT)
                    stage["pid"] = child.pid; save(receipt_path, state)
                    child.wait(timeout=600)
                if child.returncode:
                    raise RuntimeError("Native capture failed; see native_jax/console.log")
                stage["exit_code"] = child.returncode
            else:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
                command += ["--port", str(port), "--cleanup", str(out/"cleanup.json")]
                if kind == "fresh_prepared":
                    command += ["--arm", "fresh18_no_encoder"]
                arm = "fresh18_no_encoder" if kind == "fresh_prepared" else "sm16_encoder"
                alignment = "off" if kind == "fresh_prepared" else args.pipeline_alignment
                module = ("results.idea2.continuous_encoder_20261026.client" if kind == "fresh_prepared"
                          else "results.idea2.control_alignment_20261026.client")
                client_command = [prior["client"]["python"], "-u", "-m", module,
                    "--port", str(port), "--suite", "libero_object", "--task", "0", "--init", "0",
                    "--arm", arm, "--output", str(out/"episode")]
                if kind == "pipeline":
                    client_command += ["--alignment", alignment]
                stage.update(client_command=client_command, port=port, arm=arm, alignment=alignment)
                with (out/"server.log").open("x") as log:
                    server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                stage["server_pid"] = server.pid; save(receipt_path, state)
                deadline = time.monotonic()+300
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("Profiled model server exited")
                    try:
                        health = rpc(port, dict(op="health")); break
                    except (OSError, TimeoutError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("Model startup deadline")
                        time.sleep(1)
                if kind == "pipeline":
                    rpc(port, dict(op="configure", arm=arm, alignment=alignment))
                stage["health"] = health
                own = {p["pid"] for p in gpu_snapshot(GPU)["processes"]}
                if len(own) != 1:
                    raise RuntimeError("Expected one model process")
                stage["owned_gpu_pids"] = sorted(own)
                with (out/"client.log").open("x") as log:
                    child = subprocess.Popen(client_command, cwd=ROOT, env=client_env, stdout=log, stderr=subprocess.STDOUT)
                    stage["client_pid"] = child.pid; save(receipt_path, state)
                    started = time.monotonic()
                    while child.poll() is None:
                        if time.monotonic()-started > 600:
                            raise TimeoutError("Profile episode deadline")
                        ownership = client_gpu_ownership(gpu_snapshot(GPU), own, child.pid)
                        if ownership is not None:
                            stage["render_gpu_verified"] = ownership
                        time.sleep(2)
                stage["client_exit_code"] = child.returncode
                if child.returncode:
                    raise RuntimeError("Profile client failed")
                rpc(port, dict(op="shutdown")); server.wait(timeout=120)
                stage["server_exit_code"] = server.returncode
                if server.returncode or not json.loads((out/"cleanup.json").read_text())["passed"]:
                    raise RuntimeError("Server/profile cleanup failed")
                stage["episode"] = json.loads((out/"episode/complete.json").read_text())
                server = None
            receipt = json.loads((out/"capture.json").read_text())
            if receipt["captured"] != 8 or not Path(stage["trace"]).is_file():
                raise RuntimeError("Missing/incomplete trace")
            stage.update(status="passed", captured_requests=8, end_utc=now(),
                bytes=Path(stage["trace"]).stat().st_size)
            save(receipt_path, state)
            print(kind, "passed", stage["bytes"], flush=True)
        after = {p:sha(ROOT/p) for p in sorted(frozen_paths)}
        if after != frozen:
            raise RuntimeError("Frozen runtime/model/control source changed")
        state.update(status="passed", source_unchanged=True, end_utc=now())
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), end_utc=now())
        if child is not None and child.poll() is None:
            child.terminate(); child.wait(timeout=20)
        if server is not None and server.poll() is None:
            try:
                rpc(port, dict(op="shutdown")); server.wait(timeout=60)
            except Exception:
                server.terminate(); server.wait(timeout=20)
        raise
    finally:
        if claimed:
            lease(False, gpu_snapshot(GPU))
            state["lease_released"] = True
        save(receipt_path, state)


if __name__ == "__main__":
    main()
