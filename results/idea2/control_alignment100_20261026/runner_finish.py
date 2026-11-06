"""Expanded fixed100 paired study; frozen pilot model and control clients."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

import numpy as np

R = Path(__file__).resolve().parent
ROOT = R.parents[2]
DEVICES = {0: ("GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1", 3, range(32, 48)),
           2: ("GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25", 1, range(0, 16)),
           3: ("GPU-555cde36-5e41-bc44-0c69-ea47358fd703", 0, range(96, 112))}
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(path)


def sources():
    from results.idea2.control_alignment_20261026.run_async import sources as original_sources
    result = original_sources()
    for p in (Path(__file__), R/"manifest.json", R/"prospective_statistics.md", R/"resource_gpu0_revision.json", R/"resource_final_revision.json"):
        result[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def validate(leaf, mode):
    from preparations.idea1_v2_1.runtime_client import model_seed
    value = json.loads((leaf/"complete.json").read_text())
    if not value.get("valid") or value.get("control_alignment") != mode:
        raise ValueError("Invalid or mismatched control episode")
    records = [json.loads(x) for x in (leaf/"infer.jsonl").read_text().splitlines()]
    applications = [json.loads(x) for x in (leaf/"telemetry.jsonl").read_text().splitlines()][1:]
    if len(applications) != value["steps"]:
        raise ValueError("Action count mismatch")
    byid = {r["request_id"]:r for r in records}
    for record in records:
        if record["seed"] != model_seed(value["suite"], value["task"], value["init"], record["target_chunk_id"]):
            raise ValueError("Actuator noise identity changed")
        protocol = record["protocol"]
        actual = protocol["resource_adapter"]["actual_resources"]
        if not actual or any(x["sm_counts"] != [16, 112] for x in actual):
            raise ValueError("SM16/112 allocation changed")
        if protocol.get("control_alignment") != mode or record["graph_kv_buffers"] != 2:
            raise ValueError("Alignment or two-slot contract changed")
    for a in applications:
        r = byid[a["consumed_request_id"]]
        offset = r["execution_start_step"]-r["input_step"] if mode == "control_steps" else 0
        if a["action_index"] != offset+a["action_position_in_block"]:
            raise ValueError("Executed action index is not the declared offset")
        expected = np.asarray(r["chunk_actions"][a["action_index"]], np.float32)
        if not np.array_equal(expected, np.asarray(a["action"], np.float32)):
            raise ValueError("Executed action differs from model output")
        if mode == "control_steps" and a["alignment_residual_steps"] != 0:
            raise ValueError("Residual control-step index delay")
    return dict(value, checked_predictions=len(records), checked_applications=len(applications))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--gpu", type=int, choices=(0, 2, 3), required=True)
    p.add_argument("--cell-keys", nargs="+", required=True, help="mode/suite/task/init; only unfinished cells")
    args = p.parse_args()
    cases = []
    selected_modes = {}
    for cell in args.cell_keys:
        mode, suite, task, init = cell.split("/")
        task, init = int(task), int(init)
        if mode not in ("off", "control_steps") or suite not in SUITES or not 0 <= task < 10 or init not in range(3 if task%2 == 0 else 2):
            raise ValueError("Invalid pending cell")
        key = suite, task, init
        if key not in selected_modes:
            cases.append(key)
            selected_modes[key] = []
        if mode in selected_modes[key]:
            raise ValueError("Duplicate pending cell")
        selected_modes[key].append(mode)
    if len(set(cases)) != len(cases) or not cases or len({(s,t) for s,t,i in cases}) > 4:
        raise ValueError("Select unique cases belonging to at most four tasks per service")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    GPU, egl, cpu_range = DEVICES[args.gpu]
    cpus = set(cpu_range)
    os.sched_setaffinity(0, cpus)
    prior = json.loads((ROOT/"results/idea2/closedloop200_20261025/gpu2_chunk0/plan.json").read_text())
    common = dict(PROGRESSIVE_EXPECTED_CPUS=",".join(map(str, sorted(cpus))))
    env_server = dict(os.environ, **prior["server"]["environment"])
    env_server.update(common, CUDA_VISIBLE_DEVICES=GPU)
    env_client = dict(os.environ, **prior["client"]["environment"])
    env_client.update(common, MUJOCO_EGL_DEVICE_ID=str(egl))
    from preparations.idea1_v2_1.runtime_client import rpc
    from preparations.idea2.layer_bench import require_idle_gpu, require_same_processes, gpu_snapshot
    from results.idea2.d10_dynamic_sm_20261024.closedloop_smoke import client_gpu_ownership
    leases = json.loads((ROOT/"results/idea1_v3_3/external_gpu_leases.json").read_text())
    if leases[str(args.gpu)].get("owner") != "user-authorized idea2 control alignment100" or not leases[str(args.gpu)].get("active"):
        raise RuntimeError("Assigned GPU is not reserved for this study")
    with socket.socket() as test:
        test.bind(("127.0.0.1", args.port))
    frozen = sources()
    state = dict(status="starting", start_utc=now(), args=vars(args), cases=[], source_before=frozen,
                 gpu_uuid=GPU, gpu_before=require_idle_gpu(GPU), cpu_affinity=sorted(cpus),
                 environment_server={k:env_server[k] for k in prior["server"]["environment"]},
                 environment_client={k:env_client[k] for k in prior["client"]["environment"]})
    checkpoint = lambda: save(out/"execution.json", state)
    server = child = None
    try:
        command = [prior["server"]["python"], "-u", "-m", "results.idea2.control_alignment_20261026.server",
                   "--port", str(args.port), "--cleanup", str(out/"cleanup.json")]
        state["server_command"] = command
        with (out/"server.log").open("x") as log:
            server = subprocess.Popen(command, cwd=ROOT, env=env_server, stdout=log, stderr=subprocess.STDOUT)
        state["server_pid"] = server.pid
        checkpoint()
        deadline = time.monotonic()+300
        while True:
            if server.poll() is not None:
                raise RuntimeError("Model server exited during startup")
            try:
                state["health"] = rpc(args.port, dict(op="health"))
                break
            except (OSError, TimeoutError):
                if time.monotonic()>deadline:
                    raise TimeoutError("Startup deadline")
                time.sleep(1)
        snapshot = gpu_snapshot(GPU)
        own = {r["pid"] for r in snapshot["processes"]}
        if len(own) != 1 or snapshot["graphics_processes"]:
            raise RuntimeError("Expected one exclusively owned model process")
        state.update(status="running", owned_pids=sorted(own))
        checkpoint()
        for suite, task, init in cases:
            modes = ("off", "control_steps") if (SUITES.index(suite)+task+init+int(args.gpu in (0, 2)))%2 == 0 else ("control_steps", "off")
            for mode in modes:
                if mode not in selected_modes[(suite,task,init)]: continue
                configured = rpc(args.port, dict(op="configure", arm="sm16_encoder", alignment=mode))
                leaf = out/"episodes"/mode/suite/f"task{task:02d}"/f"init{init:02d}"
                leaf.parent.mkdir(parents=True, exist_ok=True)
                row = dict(configuration="sm16_encoder_"+mode, alignment=mode, suite=suite, task=task, init=init,
                           status="running", directory=str(leaf), configured=configured, start_utc=now(),
                           gpu_before=require_same_processes(GPU, own))
                state["cases"].append(row)
                command = [prior["client"]["python"], "-u", "-m", "results.idea2.control_alignment_20261026.client",
                           "--port", str(args.port), "--suite", suite, "--task", str(task), "--init", str(init),
                           "--arm", "sm16_encoder", "--alignment", mode, "--output", str(leaf)]
                row["command"] = command
                with (leaf.parent/f"init{init:02d}.log").open("x") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env_client, stdout=log, stderr=subprocess.STDOUT)
                    row["client_pid"] = child.pid
                    checkpoint()
                    start = time.monotonic()
                    while child.poll() is None:
                        if time.monotonic()-start>600:
                            raise TimeoutError("Episode deadline")
                        ownership = client_gpu_ownership(gpu_snapshot(GPU), own, child.pid)
                        if ownership is not None and "render_gpu_verified" not in row:
                            row["render_gpu_verified"] = ownership
                            checkpoint()
                        if "render_gpu_verified" not in row and time.monotonic()-start>90:
                            raise RuntimeError("EGL ownership unverified")
                        time.sleep(1)
                row.update(client_exit_code=child.returncode, end_utc=now(), gpu_after=require_same_processes(GPU, own))
                if child.returncode or "render_gpu_verified" not in row:
                    raise RuntimeError("Client exit or EGL verification failed")
                row.update(result=validate(leaf, mode), status="valid")
                checkpoint()
                print(suite, task, mode, row["result"]["success"], row["result"]["steps"], flush=True)
        rpc(args.port, dict(op="shutdown"))
        server.wait(timeout=60)
        cleanup = json.loads((out/"cleanup.json").read_text())
        if server.returncode or not cleanup["passed"]:
            raise RuntimeError("Model/context cleanup failed")
        state.update(status="passed", server_exit_code=server.returncode, cleanup=cleanup)
    except BaseException:
        state.update(status="technical_failure", error=traceback.format_exc())
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=15)
        if server is not None and server.poll() is None:
            try:
                rpc(args.port, dict(op="shutdown"))
                server.wait(timeout=30)
            except Exception:
                server.terminate()
                server.wait(timeout=15)
        if server is not None:
            state["server_exit_code"] = server.returncode
        raise
    finally:
        state.update(end_utc=now(), source_after=sources())
        state["source_unchanged"] = frozen == state["source_after"]
        if state["status"] == "passed" and not state["source_unchanged"]:
            state.update(status="technical_failure", error="Source changed during execution")
        checkpoint()


if __name__ == "__main__":
    main()
