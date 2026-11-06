"""Dry-run-first stage orchestration, condition locks and resumable rollouts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import socket
import time

from .common import PYTHONS, ROOT, atomic_json, environment, lock, read_manifest
from .server import rpc


def completed(output, condition, inits=(0,), manifest_hash=None):
    rows = []
    for path in (Path(output)/condition/"episodes").glob("*/task*/init*/result.json"):
        row = json.loads(path.read_text())
        if ".invalid." in str(path):
            continue
        if row.get("valid") and row["init"] in inits and (path.parent/"complete.json").exists():
            if manifest_hash is not None and row.get("manifest_hash")!=manifest_hash:
                raise ValueError("Promotion cannot mix results from another frozen manifest")
            rows.append(row)
    return rows


def promotion(manifest, output):
    selected,decisions = [],[]
    def records(cid):
        return completed(output,cid,manifest_hash=manifest["manifest_hash"])
    for model in ("pi0","pi05","xiaomi-robotic0","db-cogact"):
        conditions = [c for c in manifest["conditions"] if c["model"]==model]
        full = next(c for c in conditions if not c["skip"])
        rows = records(full["id"])
        if len(rows) != 40:
            decisions.append(dict(model=model,status="incomplete_full_screen",valid=len(rows)))
            continue
        baseline = sum(row["success"] for row in rows)
        selected.append(full["id"])
        groups = defaultdict(list)
        for condition in conditions:
            if condition["skip"] and not condition["reference"]:
                groups[len(condition["skip"])].append(condition)
        for budget,candidates in groups.items():
            scores = []
            if any(len(records(c["id"]))!=40 for c in candidates):
                decisions.append(dict(model=model,budget=budget,status="incomplete_candidate_screen"))
                continue
            for condition in candidates:
                rows = records(condition["id"])
                scores.append((sum(row["success"] for row in rows),condition["id"]))
            score,cid = sorted(scores,key=lambda row:(-row[0],row[1]))[0]
            ratio = score/baseline if baseline else None
            eligible = (baseline>0 and ratio >= manifest["relative_success_floor"]-manifest["confirmation_near_floor_margin"]
                        if model in ("xiaomi-robotic0","db-cogact") else score>=manifest["confirmation_min_successes_screen"])
            if eligible:
                selected.append(cid)
            decisions.append(dict(model=model,budget=budget,best=cid,screen_successes=score,full_successes=baseline,
                                  relative_retention=ratio,confirm=eligible,status="selected" if eligible else "below_frozen_confirmation_gate"))
    return dict(manifest_hash=manifest["manifest_hash"],selected_conditions=selected,decisions=decisions,
                caveat="Exploratory selection; confirmation includes the same40 screening episodes and160 additional official scenes")


def assert_gpu_free(gpu):
    uuid = subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    lines = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True).splitlines()
    if any(line.split(",")[0].strip()==uuid for line in lines):
        raise RuntimeError(f"Assigned GPU {gpu} has an existing compute process")


def condition_commands(args, manifest, condition):
    from .reuse_gate import quality_reuse
    stage = args.stage
    workers = args.workers
    if stage=="profile":
        variant = next(v for v in manifest["e2"]["variants"] if v["id"]==args.variant)
        workers = variant["workers"]
    else:
        quality_reuse(manifest, getattr(args,"environment_reuse",None), output=args.output, model=condition["model"])
    server = [str(PYTHONS[condition["model"]]),"-m","preparations.idea1_v3_1.server",
              "--manifest",args.manifest,"--condition",condition["id"],"--port",str(args.port),
              "--protocol-output",str(Path(args.output)/condition["id"]/"loaded_protocol.json"),"--launch"]
    if args.operation=="benchmark":
        command = [str(PYTHONS[condition["model"]]),"-m","preparations.idea1_v3_1.benchmark",
                   "--manifest",args.manifest,"--condition",condition["id"],"--port",str(args.port),
                   "--output",str(Path(args.output)/condition["id"]/"benchmark.json"),"--bank",*(args.bank or ["REAL_BANKS"]),"--launch"]
        return server,[command]
    clients = []
    for index in range(workers):
        command = [str(PYTHONS["environment"]),"-m","preparations.idea1_v3_1.rollout","--manifest",args.manifest,
                   "--condition",condition["id"],"--output",args.output,"--port",str(args.port),"--stage",stage,
                   "--shard-index",str(index),"--shards",str(workers),"--launch"]
        if args.variant:
            command += ["--variant",args.variant]
        elif getattr(args,"environment_reuse",None) is not None:
            command += ["--environment-reuse",args.environment_reuse]
        if args.vision_parity:
            command += ["--vision-parity",args.vision_parity]
        clients.append(command)
    return server,clients


def run_condition(args, manifest, condition):
    server_command,client_commands = condition_commands(args,manifest,condition)
    if not args.launch:
        return dict(condition=condition["id"],server=server_command,clients=client_commands,dry_run=True)
    if args.gpu is None:
        raise ValueError("--launch requires one assigned --gpu")
    if not args.bank:
        raise ValueError("Launched work requires --bank for real-input parity and warmup")
    if condition["experiment"]=="E3" and condition["skip"] and args.operation!="benchmark":
        if not args.vision_parity:
            raise ValueError("Vision variants require --vision-parity from native/matched Full artifacts")
        parity = json.loads(Path(args.vision_parity).read_text())
        if not parity.get("passed") or parity.get("manifest_hash")!=manifest["manifest_hash"] or parity.get("model")!=condition["model"]:
            raise ValueError("Full vision parity gate does not match this model/manifest")
    output = Path(args.output)/condition["id"]
    output.mkdir(parents=True,exist_ok=True)
    with lock(output/"condition.lock",blocking=False):
        # Reject occupied ports before allocating model weights. A later race is
        # still caught by the exact service PID and condition health checks.
        with socket.socket() as check:
            check.bind(("127.0.0.1",args.port))
        assert_gpu_free(args.gpu)
        env = environment(args.gpu)
        with (output/f"service_{args.stage}_{args.variant or 'default'}.log").open("a") as service_log:
            server = subprocess.Popen(server_command,cwd=ROOT,env=env,stdout=service_log,stderr=subprocess.STDOUT)
            ready = False
            clients = []
            try:
                deadline = time.monotonic()+1800
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError("Model service exited before readiness; see service log")
                    try:
                        health = rpc(args.port,dict(op="health"))
                        protocol = health["protocol"]
                        if (protocol["manifest_hash"]!=manifest["manifest_hash"] or protocol["condition_id"]!=condition["id"]
                                or protocol.get("server_pid")!=server.pid):
                            raise RuntimeError("Port belongs to a different manifest, condition, or service PID; stop own launch only")
                        ready = True
                        break
                    except (ConnectionError,OSError):
                        time.sleep(1)
                if not ready:
                    raise TimeoutError("Model service readiness deadline exceeded")
                from .benchmark import load_banks,benchmark
                records,banks = load_banks(args.bank)
                first = records[0]
                parity = rpc(args.port,dict(op="parity",obs=first["obs"],instruction=first["instruction"],seed=int(first.get("seed",0))))
                atomic_json(output/"condition_parity.json",dict(parity,banks=banks))
                if not parity.get("passed") or parity.get("noise_paired") is False:
                    raise RuntimeError("Real-input native/adapter parity gate failed")
                if args.operation=="benchmark":
                    result = benchmark(args.port,records)
                    result.update(manifest_hash=manifest["manifest_hash"],condition=condition,banks=banks,
                                  protocol=health["protocol"])
                    atomic_json(output/"benchmark.json",result)
                    return dict(condition=condition["id"],benchmark_complete=True,summary=result["summary"])
                # Warm every service on the same20 recorded observations before
                # throughput/quality timing. Compilation is outside rollout wall.
                for index in range(20):
                    record = records[index%len(records)]
                    rpc(args.port,dict(op="predict",obs=record["obs"],instruction=record["instruction"],seed=int(record.get("seed",index))))
                before = len(list(output.rglob("complete.json")))
                begin = time.perf_counter()
                clients = [subprocess.Popen(command,cwd=ROOT,env=env) for command in client_commands]
                exits = [client.wait() for client in clients]
                if any(exits):
                    raise RuntimeError(f"Rollout worker failed with exits {exits}; incomplete episodes are resumable")
                total = time.perf_counter()-begin
                after = len(list(output.rglob("complete.json")))
                atomic_json(output/f"throughput_{args.stage}_{args.variant or 'default'}.json",
                            dict(manifest_hash=manifest["manifest_hash"],condition=condition["id"],wall_s=total,
                                 newly_completed_episodes=after-before,completed_before=before,completed_after=after,
                                 workers=len(clients),variant=args.variant,scope="Aggregate rollout wall time; includes simulation, RPC, optional I/O; excludes service startup"))
            finally:
                for client in clients:
                    if client.poll() is None:
                        client.terminate()
                for client in clients:
                    if client.poll() is None:
                        client.wait(timeout=60)
                if server.poll() is None:
                    stopped = False
                    if ready:
                        try:
                            protocol = rpc(args.port,dict(op="health"))["protocol"]
                            if (protocol.get("server_pid")==server.pid and protocol.get("condition_id")==condition["id"]
                                    and protocol.get("manifest_hash")==manifest["manifest_hash"]):
                                rpc(args.port,dict(op="shutdown"))
                                stopped = True
                        except (ConnectionError,OSError,RuntimeError):
                            pass
                    if not stopped:
                        server.terminate()
                    server.wait(timeout=60)
    return dict(condition=condition["id"],complete=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation",choices=("run","promote","benchmark"))
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--stage",choices=("screen","confirm","profile"),default="screen")
    parser.add_argument("--condition")
    parser.add_argument("--model",choices=("pi0","pi05","xiaomi-robotic0","db-cogact"))
    parser.add_argument("--experiment",choices=("E1","E3"))
    parser.add_argument("--selection",help="Frozen promotion JSON required for confirm stage")
    parser.add_argument("--variant",choices=("full_io","lean_io","reuse_env","two_workers"))
    parser.add_argument("--workers",type=int,choices=(1,2),default=1)
    parser.add_argument("--port",type=int,default=9131)
    parser.add_argument("--gpu")
    parser.add_argument("--bank",nargs="+")
    parser.add_argument("--vision-parity")
    parser.add_argument("--environment-reuse",choices=("on","off"),help="Quality only; on requires paired E2 adoption evidence")
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest,verify_sources=args.launch)
    if args.operation=="promote":
        result = promotion(manifest,args.output)
        if args.launch:
            path = Path(args.output)/"promotion.json"
            if path.exists():
                raise FileExistsError("Promotion is immutable; use the existing selection or a new output root")
            atomic_json(path,result)
        print(json.dumps(result,indent=2))
        return
    candidates = manifest["conditions"]+(manifest["benchmark_references"] if args.condition or args.operation=="benchmark" else [])
    conditions = [c for c in candidates if (not args.condition or c["id"]==args.condition)
                  and (not args.model or c["model"]==args.model) and (not args.experiment or c["experiment"]==args.experiment)]
    if args.stage=="confirm":
        if not args.selection:
            raise ValueError("Confirm requires an explicit immutable --selection promotion.json")
        selection = json.loads(Path(args.selection).read_text())
        if selection["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("Promotion/manifest identity mismatch")
        conditions = [c for c in conditions if c["id"] in selection["selected_conditions"]]
    if args.stage=="profile" and (not args.variant or not args.condition):
        raise ValueError("Profile requires explicit Full --condition and --variant")
    for condition in conditions:
        print(json.dumps(run_condition(args,manifest,condition)),flush=True)


if __name__ == "__main__":
    main()
