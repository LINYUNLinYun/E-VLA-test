"""Freeze the protocol, then run one explicitly assigned condition/GPU at a time."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from preparations.idea1_v3_1.common import ROOT, SUITES, PYTHONS, atomic_json, digest, environment, lock, read_manifest
from preparations.idea1_v3_1.server import rpc

MODULE = "preparations.ease_insights_20261103.fastpath"
OLD_MANIFEST = ROOT / "results/idea1_v3_1/manifest_v2.json"
OLD_PARITY = ROOT / "results/idea1_v3_1/run/pi0_vision_same_process_parity.json"
BANK = ROOT / "results/idea1_v3_1/input_bank.pkl"
CONDITIONS = ("pi0_full", "pi0_keep24_2_front_middle", "pi0_keep24_1_uniform")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    output = Path(output).resolve()
    target = output / "manifest.json"
    if target.exists():
        manifest = read_manifest(target, verify_sources=True)
        return dict(reused=True, path=str(target), manifest_hash=manifest["manifest_hash"])
    old = read_manifest(OLD_MANIFEST)
    parity = json.loads(OLD_PARITY.read_text())
    # This receipt concerns the exact unchanged selected-scan implementation;
    # unrelated historical source files are not prerequisites for a new run.
    dependencies = {p: h for p, h in parity["source_hashes"].items() if not p.endswith("same_process_parity.py")}
    dependencies.update(old["checkpoints"]["pi0"]["identity_artifacts"])
    changes = [p for p, h in dependencies.items() if sha(p) != h]
    if not parity["passed"] or changes:
        raise ValueError(f"Need a new selected-scan Full parity gate; changed dependencies: {changes}")
    conditions = [copy.deepcopy(next(c for c in old["conditions"] if c["id"] == cid)) for cid in CONDITIONS]
    files = [Path(__file__).parent / name for name in ("__init__.py", "runner.py", "worker.py", "gpu_binding.py")]
    files += [ROOT / "preparations/idea1_v3_1" / name for name in ("common.py", "models.py", "vision.py", "server.py", "rollout.py", "benchmark.py", "manifest.py")]
    files += [ROOT / p for p in ("preparations/idea1_v2_1/runtime_client.py", "preparations/idea1_v2_1/runtime_models.py")]
    files += [Path(p) for p in dependencies if p.endswith(".py")]
    manifest = dict(revision="ease-insights-fastpath-confirmation-v1", model="pi0", conditions=conditions,
        benchmark_references=[], checkpoints={"pi0": old["checkpoints"]["pi0"]},
        suites=list(SUITES), tasks=list(range(10)), screen_inits=[20], confirm_inits=list(range(20,25)),
        quality_environment_reuse=False, technical_max_attempts=3,
        environment=dict(seed=7, seed_before_construction=True, settling_steps=10, control_hz=20, image_size=256,
            limits=dict(zip(SUITES,(220,280,300,520))), io="no video or observation dumps; retain telemetry and actions"),
        noise=dict(function="preparations.idea1_v3_1.common.seed", namespace="idea1-v3.1", seed=1701,
            key=["suite","task","init","request"], condition_independent=True),
        science=dict(module="vision", keep_blocks=24, ae_steps=10, H=50, R=5, residual_compensation=False,
            full_refresh=False, fixed_path=True, layer_numbering="original one-based"),
        preregistered_hypotheses=[dict(suite="libero_goal", better=CONDITIONS[1], worse=CONDITIONS[2]),
                                 dict(suite="libero_10", better=CONDITIONS[2], worse=CONDITIONS[1])],
        analysis=dict(primary="paired success differences in predeclared Goal and Long suites; exact paired tests and intervals",
            discovery_inits=[0,1,2,3,4], independent_validation_inits=list(range(20,25)),
            pilot_counts_toward_confirmation=True, selection="All three conditions confirm regardless of pilot direction",
            other_tasks="descriptive only; no post-hoc relabeling as preregistered categories",
            size="exploratory 200 episodes/condition, not official 50-init evaluation",
            interpretation="A universal winner supports a fixed path; no complementarity claim without reproduced crossing"),
        parity_evidence=dict(path=str(OLD_PARITY), sha256=sha(OLD_PARITY), historical_manifest_hash=old["manifest_hash"],
            passed=True, scope=parity["method"], compatibility_dependencies=dependencies,
            reuse="exact implementation/checkpoint identity; historical parity is not a new validation result"),
        bank=dict(path=str(BANK), sha256=sha(BANK)),
        historical_reference=dict(manifest=str(OLD_MANIFEST), manifest_hash=old["manifest_hash"],
            candidate_selection="init0: A 35/40, B 36/40; Goal A +2, Long B +2; exploratory evidence"),
        source_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))})
    manifest["manifest_hash"] = digest(manifest)
    atomic_json(target, manifest)
    return dict(reused=False,path=str(target),manifest_hash=manifest["manifest_hash"], conditions=list(CONDITIONS),
                pilot_cases_per_condition=40,confirmation_cases_per_condition=200)


def cases(manifest, stage, shard_index=0, shards=1):
    if not 0 <= shard_index < shards:
        raise ValueError("Invalid task shard")
    inits = manifest["screen_inits" if stage == "pilot" else "confirm_inits"]
    pairs = [(s,t) for s in manifest["suites"] for t in manifest["tasks"]]
    return [dict(suite=s,task=t,init=i) for index,(s,t) in enumerate(pairs) if index % shards == shard_index for i in inits]


def resolve_binding(gpu, gpu_uuid):
    if not gpu_uuid:
        if gpu is None:
            raise ValueError("An explicitly assigned --gpu-uuid or --gpu is required")
        gpu_uuid = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],text=True).strip()
    clean = environment()
    clean.pop("CUDA_VISIBLE_DEVICES",None)
    raw = subprocess.check_output([str(PYTHONS["environment"]),"-m",MODULE+".gpu_binding"],cwd=ROOT,env=clean,text=True)
    rows = json.loads(raw)
    selected = [r for r in rows if r["gpu_uuid"] == gpu_uuid]
    if len(selected) != 1:
        raise ValueError(f"GPU UUID does not map uniquely to an EGL device: {gpu_uuid}, {rows}")
    return dict(selected[0],mapping=rows)


def commands(args, manifest):
    output = Path(args.output).resolve()
    server = [str(PYTHONS["pi0"]),"-m","preparations.idea1_v3_1.server","--manifest",str(output/"manifest.json"),
              "--condition",args.condition,"--port",str(args.port),"--protocol-output",str(output/args.condition/"loaded_protocol.json"),"--launch"]
    clients=[]
    for index in range(args.workers):
        clients.append([str(PYTHONS["environment"]),"-m",MODULE+".worker","--output",str(output),"--condition",args.condition,
            "--port",str(args.port),"--stage",args.stage,"--shard-index",str(args.shard_index+args.shards*index),
            "--shards",str(args.shards*args.workers),"--launch"])
    return server,clients


def run(args):
    output=Path(args.output).resolve()
    manifest=read_manifest(output/"manifest.json",verify_sources=args.launch)
    if args.condition not in [c["id"] for c in manifest["conditions"]]:
        raise ValueError("Unknown frozen condition")
    scenes=cases(manifest,args.stage,args.shard_index,args.shards)
    server_cmd,client_cmds=commands(args,manifest)
    if not args.launch:
        return dict(dry_run=True,condition=args.condition,stage=args.stage,cases=len(scenes),server=server_cmd,clients=client_cmds)
    from .worker import completed
    missing=[s for s in scenes if not completed(output,args.condition,s,manifest["manifest_hash"])]
    marker=output/args.condition/f"{args.stage}_shard{args.shard_index:02d}of{args.shards:02d}_complete.json"
    fixed_receipt=dict(complete=True,manifest_hash=manifest["manifest_hash"],condition=args.condition,
        stage=args.stage,case_count=len(scenes),shard_index=args.shard_index,shards=args.shards)
    if not missing:
        atomic_json(marker,dict(fixed_receipt,already_completed=len(scenes)))
        return dict(fixed_receipt,already_completed=len(scenes))
    binding=resolve_binding(args.gpu,args.gpu_uuid)
    gpu_uuid=binding["gpu_uuid"]
    owner=output/args.condition
    owner.mkdir(parents=True,exist_ok=True)
    # The outer scheduler owns the global GPU flock. This local lock prevents
    # duplicate launches of the same condition/task partition.
    run_id=f"{args.stage}_shard{args.shard_index:02d}of{args.shards:02d}_{time.time_ns()}"
    with lock(owner/f"shard{args.shard_index:02d}of{args.shards:02d}.lock",blocking=False):
        with socket.socket() as check:
            check.bind(("127.0.0.1",args.port))
        processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True)
        if any(line.split(",")[0].strip()==gpu_uuid for line in processes.splitlines()):
            raise RuntimeError("Assigned GPU has an existing compute process")
        env=environment()
        env.update(CUDA_VISIBLE_DEVICES=gpu_uuid,MUJOCO_EGL_DEVICE_ID=str(binding["egl_index"]),
            EASE_FASTPATH_GPU_UUID=gpu_uuid,EASE_FASTPATH_BINDING=json.dumps(binding),
            XLA_PYTHON_CLIENT_PREALLOCATE="false")
        logs=owner/"runs"/run_id
        logs.mkdir(parents=True)
        atomic_json(logs/"identity.json",dict(manifest_hash=manifest["manifest_hash"],condition=args.condition,
            gpu_binding=binding,controller_pid=os.getpid(),server_command=server_cmd,client_commands=client_cmds,
            global_gpu_lock=f"/tmp/embodied-benchmark-gpu-{gpu_uuid}.lock",global_lock_owner="external scheduler"))
        started=time.perf_counter()
        clients=[]; ready=False; handles=[]; server=None
        try:
            log=(logs/"server.log").open("a"); handles.append(log)
            server=subprocess.Popen(server_cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+1800
            while time.monotonic()<deadline:
                if server.poll() is not None: raise RuntimeError(f"Service exited; inspect {logs/'server.log'}")
                try:
                    health=rpc(args.port,dict(op="health"))
                    p=health["protocol"]
                    if (p["server_pid"]!=server.pid or p["condition_id"]!=args.condition or p["manifest_hash"]!=manifest["manifest_hash"]):
                        raise ValueError("Wrong model service identity")
                    ready=True; break
                except (ConnectionError,OSError): time.sleep(1)
            if not ready: raise TimeoutError("Service startup exceeded 1800 seconds")
            loaded=time.perf_counter()
            from preparations.idea1_v3_1.benchmark import load_banks
            records,banks=load_banks([manifest["bank"]["path"]])
            if banks[0]["sha256"]!=manifest["bank"]["sha256"]: raise ValueError("Warmup bank changed")
            first=records[0]
            tick=time.perf_counter()
            parity=rpc(args.port,dict(op="parity",obs=first["obs"],instruction=first["instruction"],seed=int(first["seed"])))
            parity_wall=time.perf_counter()-tick
            atomic_json(logs/"condition_parity.json",parity)
            if not parity.get("passed"): raise RuntimeError("Explicit-noise repeatability failed")
            warmup=[]
            for index in range(20):
                row=records[index%len(records)]; tick=time.perf_counter()
                response=rpc(args.port,dict(op="predict",obs=row["obs"],instruction=row["instruction"],seed=int(row["seed"])))
                warmup.append(dict(index=index,wall_ms=(time.perf_counter()-tick)*1000,timing=response.get("timing")))
            atomic_json(logs/"preparation.json",dict(service_load_s=loaded-started,parity_s=parity_wall,
                parity_includes_first_compilation=True,warmup=warmup,total_preparation_s=time.perf_counter()-started,
                note="These cold and warmup costs are excluded from closed-loop request latency; all reported separately"))
            tick=time.perf_counter()
            for index,command in enumerate(client_cmds):
                handle=(logs/f"worker{index}.log").open("a"); handles.append(handle)
                clients.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in clients):
                bad=[p.returncode for p in clients if p.poll() not in (None,0)]
                if bad: raise RuntimeError(f"Worker failure {bad}; inspect {logs}")
                time.sleep(1)
            exits=[p.returncode for p in clients]
            if any(exits): raise RuntimeError(f"Worker failure {exits}; inspect {logs}")
            after=sum(completed(output,args.condition,s,manifest["manifest_hash"]) for s in scenes)
            receipt=dict(complete=after==len(scenes),condition=args.condition,manifest_hash=manifest["manifest_hash"],
                run_id=run_id,gpu_binding=binding,closed_loop_wall_s=time.perf_counter()-tick,total_wall_s=time.perf_counter()-started,
                completed_before=len(scenes)-len(missing),completed_after=after,new_episodes=after-len(scenes)+len(missing),workers=args.workers)
            atomic_json(logs/"result.json",receipt)
            if after!=len(scenes):
                raise RuntimeError(f"Worker returned without completing assigned cases: {after}/{len(scenes)}")
            atomic_json(marker,dict(fixed_receipt,run_id=run_id))
            return receipt
        finally:
            for child in clients:
                if child.poll() is None: child.terminate()
            for child in clients:
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            if server is not None and server.poll() is None:
                server.terminate()
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired: server.kill(); server.wait()
            for handle in handles: handle.close()


def main():
    parser=argparse.ArgumentParser(__doc__)
    sub=parser.add_subparsers(dest="operation",required=True)
    prep=sub.add_parser("prepare"); prep.add_argument("--output",required=True)
    launch=sub.add_parser("run"); launch.add_argument("--output",required=True); launch.add_argument("--condition",required=True,choices=CONDITIONS)
    launch.add_argument("--stage",choices=("pilot","confirm"),default="pilot")
    launch.add_argument("--gpu"); launch.add_argument("--gpu-uuid"); launch.add_argument("--port",type=int,required=True)
    launch.add_argument("--workers",type=int,choices=(1,2),default=1)
    launch.add_argument("--shard-index",type=int,default=0); launch.add_argument("--shards",type=int,default=1)
    launch.add_argument("--launch",action="store_true")
    args=parser.parse_args()
    print(json.dumps(prepare(args.output) if args.operation=="prepare" else run(args),indent=2))


if __name__=="__main__": main()
