"""Assigned-GPU runner for one immutable local-search stage."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from preparations.idea1_v3_1.common import environment, lock
from preparations.idea1_v2_1.runtime_client import rpc
from .common import ROOT, MODULE, MODEL_PYTHON, ENV_PYTHON, CONDITIONS, atomic_json, read_manifest
from preparations.ease_insights_20261103.vlm.recovery import recover_case
from .common import cases, planned_config, runtime_revision, validate_output, stage_for


def resolve_binding(gpu_uuid):
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("A complete explicitly assigned --gpu-uuid is required")
    env = environment(); env.pop("CUDA_VISIBLE_DEVICES", None)
    text = subprocess.check_output([str(ENV_PYTHON), "-m", "preparations.ease_insights_20261103.fastpath.gpu_binding"],
                                   cwd=ROOT, env=env, text=True)
    mapping = json.loads(text)
    selected = [row for row in mapping if row["gpu_uuid"] == gpu_uuid]
    if len(selected) != 1:
        raise ValueError("Assigned GPU UUID must map uniquely to an EGL device")
    return dict(selected[0], mapping=mapping)


def cross_service(output, logs, manifest_hash):
    import numpy as np
    with lock(output/"cross_service.lock"):
        reference = output/"cross_service_reference.npz"
        identity = output/"cross_service_reference.json"
        if not reference.exists():
            reference.write_bytes((logs/"gate_actions.npz").read_bytes())
            atomic_json(identity, dict(manifest_hash=manifest_hash, first_service=str(logs)))
        if json.loads(identity.read_text())["manifest_hash"] != manifest_hash:
            raise ValueError("Cross-service reference belongs to another manifest")
        checks = []
        with np.load(reference) as ref, np.load(logs/"gate_actions.npz") as actual:
            if set(ref.files) != set(actual.files):
                raise ValueError("Cross-service gate keys differ")
            for key in ref.files:
                checks.append(dict(key=key, max_abs=float(np.abs(ref[key]-actual[key]).max()),
                    passed=bool(np.allclose(ref[key], actual[key], atol=1e-5, rtol=1e-5))))
        receipt = dict(passed=all(c["passed"] for c in checks), checks=checks, atol=1e-5, rtol=1e-5)
        atomic_json(logs/"cross_service.json", receipt)
        if not receipt["passed"]:
            raise RuntimeError("Cross-service real-input gate failed")


def run(args):
    output = Path(args.output).resolve()
    manifest = read_manifest(output/"manifest.json", verify_sources=args.launch)
    validate_output(output, manifest)
    planned_config(manifest, args.condition)
    stage = stage_for(manifest)
    if getattr(args, "stage", None) not in (None, stage):
        raise ValueError("CLI stage differs from immutable manifest")
    args.stage = stage
    scenes = cases(manifest, stage, args.shard_index, args.shards)
    if args.condition not in [c["id"] for c in manifest["conditions"]]:
        raise ValueError("Unknown frozen condition")
    logs = output/args.condition/"runs"/f"{args.stage}_shard{args.shard_index:02d}of{args.shards:02d}_{time.time_ns()}"
    server_cmd = [str(MODEL_PYTHON), "-m", MODULE+".server", "--manifest", str(output/"manifest.json"),
        "--condition", args.condition, "--port", str(args.port), "--gpu-uuid", args.gpu_uuid, "--output", str(logs)]
    client_cmd = [str(ENV_PYTHON), "-m", MODULE+".worker", "--output", str(output), "--condition", args.condition,
        "--port", str(args.port), "--stage", args.stage, "--shard-index", str(args.shard_index), "--shards", str(args.shards)]
    if not args.launch:
        return dict(dry_run=True, episodes=len(scenes), preflight_only=args.preflight_only, server=server_cmd, worker=client_cmd)
    from .worker import completed, finalize
    for scene in scenes:
        recover_case(output, args.condition, scene, manifest, finalize)
    missing = [s for s in scenes if not completed(output, args.condition, s, manifest["manifest_hash"])]
    marker = output/args.condition/f"{args.stage}_shard{args.shard_index:02d}of{args.shards:02d}_complete.json"
    if not missing and not args.preflight_only:
        receipt = dict(complete=True, manifest_hash=manifest["manifest_hash"], condition=args.condition, already_completed=len(scenes))
        atomic_json(marker, receipt); return receipt
    binding = resolve_binding(args.gpu_uuid)
    with lock(output/args.condition/f"shard{args.shard_index:02d}of{args.shards:02d}.lock", blocking=False):
        # Global GPU lease belongs to the main scheduler. Do not scan or claim other GPUs.
        processes = subprocess.check_output(["nvidia-smi", "--id="+args.gpu_uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        if processes.strip():
            raise RuntimeError("Assigned GPU already has a compute process")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        env = environment()
        env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, MUJOCO_EGL_DEVICE_ID=str(binding["egl_index"]),
                   EASE_VLM_GPU_UUID=args.gpu_uuid, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        env["PYTHONPATH"] = str(ROOT/"policies/realtime-vla-flash/src")+":"+env["PYTHONPATH"]
        logs.mkdir(parents=True)
        atomic_json(logs/"identity.json", dict(manifest_hash=manifest["manifest_hash"], condition=args.condition,
            gpu_binding=binding, controller_pid=os.getpid(), server_command=server_cmd, worker_command=client_cmd,
            stateful_workers_per_server=1, technical_runtime_revision=runtime_revision(manifest),
            command_semantics="server_command and worker_command are the actual argv passed to Popen; worker case assignment is inferred from the immutable local-search phase; original backend and policy mechanics are reused",
            global_gpu_lock_owner="external scheduler",
            global_gpu_lock=f"/tmp/embodied-benchmark-gpu-{args.gpu_uuid}.lock"))
        started = time.perf_counter(); server = client = None; handles = []
        try:
            handle = (logs/"server.log").open("a"); handles.append(handle)
            server = subprocess.Popen(server_cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+1800
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {logs/'server.log'}")
                try:
                    protocol = rpc(args.port, dict(op="health"))["protocol"]
                    if (protocol["server_pid"] != server.pid or protocol["manifest_hash"] != manifest["manifest_hash"] or
                        protocol["condition_id"] != args.condition or protocol["gpu_uuid"] != args.gpu_uuid or not protocol["real_input_parity_passed"]):
                        raise ValueError("Wrong service identity or failed gate")
                    break
                except (ConnectionError, OSError):
                    time.sleep(1)
            else:
                raise TimeoutError("Server preparation exceeded 1800 seconds")
            cross_service(output, logs, manifest["manifest_hash"])
            if args.preflight_only:
                receipt = dict(preflight_passed=True, manifest_hash=manifest["manifest_hash"], gpu_binding=binding,
                    condition=args.condition, logs=str(logs), total_wall_s=time.perf_counter()-started, episodes_run=0)
                atomic_json(logs/"result.json", receipt); return receipt
            tick = time.perf_counter()
            handle = (logs/"worker.log").open("a"); handles.append(handle)
            client = subprocess.Popen(client_cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            while client.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Model server exited during rollout")
                time.sleep(1)
            if client.returncode:
                raise RuntimeError(f"Worker exited {client.returncode}; inspect {logs/'worker.log'}")
            after = sum(completed(output, args.condition, scene, manifest["manifest_hash"]) for scene in scenes)
            if after != len(scenes):
                raise RuntimeError("Worker did not complete all assigned cases")
            receipt = dict(complete=True, manifest_hash=manifest["manifest_hash"], condition=args.condition,
                gpu_binding=binding, completed_after=after, new_episodes=after-len(scenes)+len(missing),
                closed_loop_wall_s=time.perf_counter()-tick, total_wall_s=time.perf_counter()-started, logs=str(logs))
            atomic_json(logs/"result.json", receipt); atomic_json(marker, receipt); return receipt
        finally:
            for process in (client, server):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
            for handle in handles:
                handle.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--stage", choices=("discovery", "confirm"), default=None)
    parser.add_argument("--gpu-uuid", required=True); parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--launch", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
