"""Local-only production CLI/RPC smoke for the final PI0.5 graph candidate."""
import argparse
import hashlib
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

import numpy as np
import psutil


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def rpc(port, message):
    from preparations.idea1_v2_1.runtime_client import recv_exact
    payload = pickle.dumps(message, protocol=5)
    with socket.create_connection(("127.0.0.1", port), timeout=180) as client:
        client.sendall(struct.pack("!I", len(payload))+payload)
        size = struct.unpack("!I", recv_exact(client, 4))[0]
        return pickle.loads(recv_exact(client, size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=19012)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dest = Path(args.output).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    bank_bytes = Path(args.bank).read_bytes()
    bank = pickle.loads(bank_bytes)
    selected = [i for i, r in enumerate(bank) if r["instruction"] == bank[0]["instruction"]]
    if len(selected) < 13:
        raise ValueError("Smoke requires 13 dynamic observations of the same task")
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    command = [sys.executable, "-u", "-m", "preparations.idea2.server",
               "--checkpoint", "shared/checkpoints/openpi/pi05_libero_pytorch_idea2",
               "--model-config", "pi05_libero", "--num-steps", "10", "--cuda-graphs",
               "--submission-order", "ae_first", "--ae-priority", "-1",
               "--selective-overlap", "vision_only", "--hoist-modulation",
               "--await-previous-prefix", "--diagnostics", "--port", str(args.port)]
    settings = dict(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu",
                    OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                    OPENPI_DATA_HOME=str(root/"shared/checkpoints/openpi"),
                    PYTHONPATH=str(root)+":"+str(root/"policies/openpi/src"))
    env = dict(os.environ, **settings)
    (dest/"server_command.json").write_text(json.dumps(dict(argv=command, env=settings, cwd=str(root)), indent=2))
    (dest/"reproduce_server.sh").write_text("#!/bin/bash\ncd "+shlex.quote(str(root))+"\n"+
        shlex.join(["env", *(f"{k}={v}" for k, v in settings.items()), *command])+"\n")
    (dest/"reproduce_smoke.sh").write_text("#!/bin/bash\ncd "+shlex.quote(str(root))+"\n"+
        shlex.join([sys.executable, "-m", "preparations.idea2.cewp_rpc_smoke", "--output", str(dest)+"_rerun",
                    "--gpu", args.gpu, "--port", str(args.port), "--bank", args.bank])+"\n")
    source_names = ("server.py", "policy.py", "scheduler.py", "submission.py", "openpi_backend.py",
                    "experiments.py", "runtime_graph.py", "graph_sampler.py", "cewp_segmented.py",
                    "cewp_preparation.py", "cewp_rpc_smoke.py")
    (dest/"source").mkdir()
    hashes = {}
    for name in source_names:
        content = (Path(__file__).parent/name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        (dest/"source"/name).write_bytes(content)
    receipt = dict(passed=False, source_sha256=hashes, selected_bank_indices=selected,
                   bank_sha256=hashlib.sha256(bank_bytes).hexdigest(), calls=[], predictions=[],
                   performance_result=False, scope="Production localhost CLI and RPC smoke; GPU0 timings excluded from performance tables")
    def save():
        (dest/"receipt.json").write_text(json.dumps(receipt, indent=2, default=json_default))
    save()
    responses = []
    def call(operation, **kwargs):
        response = rpc(args.port, dict(op=operation, **kwargs))
        ordinal = len(responses)
        (dest/f"response_{ordinal:02d}_{operation}.json").write_text(json.dumps(response, indent=2, default=json_default))
        responses.append(response)
        receipt["calls"].append(dict(index=ordinal, operation=operation, ok=response.get("ok")))
        save()
        if not response.get("ok"):
            raise AssertionError(response)
        return response
    with (dest/"server.log").open("x") as log:
        server = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["server_pid"] = server.pid
        save()
        print("Server started", server.pid, flush=True)
        ready, shutdown = False, False
        try:
            deadline = time.monotonic()+300
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("Server exited before localhost readiness")
                listeners = [c for c in psutil.Process(server.pid).net_connections(kind="tcp")
                             if c.status == psutil.CONN_LISTEN and c.laddr.port == args.port]
                if listeners:
                    assert all(c.laddr.ip == "127.0.0.1" for c in listeners)
                    receipt["owned_loopback_listener_verified"] = True
                    ready = True
                    break
                time.sleep(0.5)
            if not ready:
                raise TimeoutError("Server readiness timed out")
            protocol = call("health")["protocol"]
            expected = dict(H=10, R=5, num_steps=10, cache_slots=2, execution="cuda_graph",
                submission_order="ae_first", ae_priority=-1, selective_overlap="vision_only",
                prepared_schedule=True, await_previous_prefix=True, host_workers=0, diagnostics=True)
            for key, value in expected.items():
                assert protocol[key] == value, (key, protocol.get(key), value)
            assert "pi05=True" in protocol["config"]
            if "prepared_modulation" in protocol:
                assert protocol["prepared_modulation"] is True
            receipt["verified_protocol"] = protocol
            receipt["modulation_cli_explicit"] = "--hoist-modulation" in command
            def predict(index, phase):
                record = bank[selected[index]]
                response = call("predict", obs=record["obs"], instruction=record["instruction"],
                                suite=record.get("suite", "libero_spatial"), seed=1701)
                assert np.asarray(response["actions"]).shape == (10, 7)
                assert np.isfinite(response["actions"]).all()
                assert response["graph_kv_buffers"] == 2 and response["pool"]["live_snapshots"] <= 2
                assert response["observation_request_age"] in (0, 1)
                assert response["snapshot"]["slot"] in (0, 1)
                if receipt["predictions"]:
                    assert response["graph_setup_ms"] == 0, "Unexpected steady recapture"
                else:
                    assert response["graph_setup_ms"] > 0
                if response["mode"] == "fresh_serial" or response["request"] == 0:
                    assert response["observation_request_age"] == 0
                else:
                    assert response["observation_request_age"] == 1
                receipt["predictions"].append(dict(phase=phase, bank_index=selected[index], seed=1701,
                    input_hash=response["input_hash"], request=response["request"], snapshot=response["snapshot"],
                    observation_age=response["observation_request_age"], graph_setup_ms=response["graph_setup_ms"],
                    live_snapshots=response["pool"]["live_snapshots"]))
                save()
            for i in range(6):
                predict(i, "continuous_async")
            call("reset")
            for i in range(6, 9):
                predict(i, "after_reset")
            fresh = call("configure", mode="fresh_serial", interval=1)
            assert fresh["protocol"]["mode"] == "fresh_serial"
            for i in range(9, 11):
                predict(i, "fresh_serial")
            async_response = call("configure", mode="async_reuse", interval=1)
            assert async_response["protocol"]["mode"] == "async_reuse"
            call("reset")
            for i in range(11, 13):
                predict(i, "async_after_configure_and_reset")
            call("drain")
            call("shutdown")
            shutdown = True
            server.wait(timeout=30)
            assert server.returncode == 0
            assert not psutil.pid_exists(server.pid)
            with socket.socket() as check:
                assert check.connect_ex(("127.0.0.1", args.port)) != 0
            receipt.update(passed=True, server_exit_code=server.returncode, process_reaped=True,
                           port_closed=True, steady_recaptures=0, request_count=len(receipt["predictions"]))
        except BaseException as exc:
            receipt["failure"] = repr(exc)
            raise
        finally:
            if server.poll() is None:
                if ready and not shutdown:
                    try:
                        call("shutdown")
                    except Exception:
                        pass
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.terminate()
                    server.wait(timeout=15)
            receipt["server_exit_code"] = server.returncode
            save()
    print(json.dumps(dict(passed=receipt["passed"], requests=receipt["request_count"],
                         server_exit_code=server.returncode)), flush=True)


if __name__ == "__main__":
    main()
