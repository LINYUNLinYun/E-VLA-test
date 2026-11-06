"""Bounded experimental profile: native two-workers vs sparse RGB, 8 cases each.

Default is read-only planning. No quality execution or migration is implemented.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .common import ROOT,PYTHONS,atomic_json,environment,read_manifest
from .same_process_parity import representatives,sha
from .server import rpc
from .shared_service_e2_generic import fixed_checks,gpu_identity,assert_assigned_gpu_free
from .sparse_camera import revision,worker

ARMS = ("baseline","sparse")


def commands(args,arm):
    return [[str(PYTHONS["environment"]),"-m","preparations.idea1_v3_1.sparse_camera_probe","--worker",
        "--manifest",args.manifest,"--model",args.model,"--root",str(args.output),"--arm",arm,
        "--port",str(args.port),"--shard-index",str(i),"--launch"] for i in range(2)]


def stop(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=30)


def run(args,manifest):
    from .benchmark import load_banks
    condition = next(c for c in manifest["conditions"] if c["id"]==args.model+"_full")
    assert condition["skip"]==[] and not condition["reference"]
    records,banks = load_banks([args.bank])
    indices = representatives(records)
    assert len(indices)==5 and len(records)>=20
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    with socket.socket() as check:
        check.bind(("127.0.0.1",args.port))
    assert_assigned_gpu_free(args.gpu)
    receipt = dict(status="starting",manifest_hash=manifest["manifest_hash"],model=args.model,condition=condition,
        execution_revision=revision(),coordinator_source_sha256=sha(__file__),coordinator_pid=os.getpid(),
        gpu=gpu_identity(args.gpu),port=args.port,banks=banks,indices=indices,arms=[],controls=[],
        profile_only=True,quality_episodes=0,raw_control_actions_available=False,
        observational_scope="Both arms record every policy-input NPZ and per-step physics/noncamera/RNG hashes; common probe overhead is timed. Startup/warm20/fixed controls excluded from arm wall.")
    path = output/"probe.json"
    service_command = [str(PYTHONS[args.model]),"-m","preparations.idea1_v3_1.server","--manifest",args.manifest,
        "--condition",condition["id"],"--port",str(args.port),"--protocol-output",str(output/"loaded_protocol.json"),"--launch"]
    receipt["server_command"] = service_command
    atomic_json(path,receipt)
    env,children = environment(args.gpu),[]
    with (output/"service.log").open("x") as log:
        service = subprocess.Popen(service_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt["server_pid"] = service.pid
        atomic_json(path,receipt)
        try:
            deadline = time.monotonic()+1800
            while True:
                if service.poll() is not None:
                    raise RuntimeError("Own native service exited")
                try:
                    protocol = rpc(args.port,dict(op="health"))["protocol"]
                    assert protocol["server_pid"]==service.pid and protocol["manifest_hash"]==manifest["manifest_hash"] and protocol["condition_id"]==condition["id"]
                    break
                except (ConnectionError,OSError):
                    if time.monotonic()>deadline:
                        raise TimeoutError("Native service startup")
                    time.sleep(.5)
            first = records[indices[0]]
            parity = rpc(args.port,dict(op="parity",obs=first["obs"],instruction=first["instruction"],seed=int(first["seed"])))
            assert parity["passed"] and parity.get("noise_paired",True) and parity["max_abs"]<=1e-5
            receipt["parity"] = parity
            receipt["protocol"] = rpc(args.port,dict(op="health"))["protocol"]
            receipt["warm_started_monotonic_s"] = time.perf_counter()
            for record in records[:20]:
                rpc(args.port,dict(op="predict",obs=record["obs"],instruction=record["instruction"],suite=record["suite"],seed=int(record["seed"])))
            receipt["warmup"] = 20
            receipt["warm_finished_monotonic_s"] = time.perf_counter()
            def controls():
                rows = []
                for index in indices:
                    record = records[index]
                    row = rpc(args.port,dict(op="predict",obs=record["obs"],instruction=record["instruction"],suite=record["suite"],seed=int(record["seed"])))
                    rows.append(dict(bank_index=index,seed=int(record["seed"]),input_hash=row["input_hash"],noise_hash=row["noise_hash"],actions=np.asarray(row["actions"]).tolist()))
                return rows
            reference = controls()
            receipt["fixed_reference"] = reference
            for arm in ARMS:
                for when in ("before","after"):
                    tick = time.perf_counter()
                    actual = controls()
                    control = dict(arm=arm,when=when,results=actual,comparison=fixed_checks(reference,actual,condition["horizon"]),
                        started_monotonic_s=tick,finished_monotonic_s=time.perf_counter())
                    receipt["controls"].append(control)
                    atomic_json(path,receipt)
                    assert control["comparison"]["passed"],"Fixed same-input/noise controls changed"
                    if when=="after":
                        continue
                    row = dict(arm=arm,commands=commands(args,arm),pids=[])
                    receipt["arms"].append(row)
                    streams = []
                    row["started_monotonic_s"] = time.perf_counter()
                    try:
                        for i,command in enumerate(row["commands"]):
                            stream = (output/f"{arm}_worker{i}.log").open("x")
                            streams.append(stream)
                            child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                            children.append(child)
                            row["pids"].append(child.pid)
                        atomic_json(path,receipt)
                        deadline = time.monotonic()+5*3600
                        while any(c.poll() is None for c in children):
                            if service.poll() is not None or any(c.poll() not in (None,0) for c in children) or list((output/arm).rglob("probe_failure.json")):
                                raise RuntimeError("Own worker/service technical failure; preserve probe raw evidence")
                            if time.monotonic()>deadline:
                                raise TimeoutError("Bounded probe arm deadline")
                            time.sleep(.25)
                        row["exit_codes"] = [c.returncode for c in children]
                        assert row["exit_codes"]==[0,0]
                    finally:
                        for child in children:
                            stop(child)
                        children.clear()
                        for stream in streams:
                            stream.close()
                    row["finished_monotonic_s"] = time.perf_counter()
                    row["wall_s"] = row["finished_monotonic_s"]-row["started_monotonic_s"]
                    atomic_json(path,receipt)
            receipt["status"] = "completed"
        except BaseException as exc:
            receipt.update(status="failed",error=repr(exc))
            raise
        finally:
            for child in children:
                stop(child)
            if service.poll() is None:
                try:
                    assert rpc(args.port,dict(op="health"))["protocol"]["server_pid"]==service.pid
                    rpc(args.port,dict(op="shutdown"))
                    service.wait(timeout=30)
                except Exception:
                    stop(service)
            receipt["server_exit_code"] = service.returncode
            atomic_json(path,receipt)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare_pair(a,b):
    """Negative pairing is valid evidence, but cannot pass the adoption gate."""
    fields = ("input_hash","noise_hash","seed","actions","executed_length")
    paired = len(a["infer"])==len(b["infer"])>0 and all(all(x[k]==y[k] for k in fields)
        for x,y in zip(a["infer"],b["infer"]))
    telemetry = lambda rows:[{k:v for k,v in row.items() if k!="host_time_s"} for row in rows]
    return dict(same_outcome=a["success"]==b["success"],same_official_initial=a["init_vector_sha256"]==b["init_vector_sha256"],
        all_requests_exact=paired,all_physics_noncamera_rng_reward_done_exact=a["physics"]==b["physics"],
        all_telemetry_except_host_time_exact=telemetry(a["telemetry"])==telemetry(b["telemetry"]))


def audit(root,manifest):
    from . import final_audit
    from .final_audit_v2 import schema_protocol_check
    from .audit_shared_e2_generic import official_scene_identities
    from .benchmark import load_banks
    from preparations.idea1_v2_1.runtime_client import input_digest
    root = Path(root)
    receipt = json.loads((root/"probe.json").read_text())
    assert receipt["status"]=="completed" and receipt["server_exit_code"]==0 and receipt["warmup"]==20
    assert receipt["manifest_hash"]==manifest["manifest_hash"] and receipt["execution_revision"]==revision()
    assert receipt["coordinator_source_sha256"]==sha(__file__) and receipt["profile_only"] and receipt["quality_episodes"]==0
    condition = next(c for c in manifest["conditions"] if c["id"]==receipt["model"]+"_full")
    assert condition==receipt["condition"] and condition["skip"]==[]
    assert receipt["gpu"]["uuid"].startswith("GPU-") and receipt["gpu"]["assigned"]
    assert receipt["parity"]["passed"] and receipt["parity"]["max_abs"]<=1e-5 and receipt["parity"].get("noise_paired",True)
    command = receipt["server_command"]
    manifest_path = command[command.index("--manifest")+1]
    recorded_root = Path(command[command.index("--protocol-output")+1]).parent
    assert recorded_root.resolve()==root.resolve()
    assert command==[str(PYTHONS[receipt["model"]]),"-m","preparations.idea1_v3_1.server","--manifest",manifest_path,
        "--condition",condition["id"],"--port",str(receipt["port"]),"--protocol-output",str(recorded_root/"loaded_protocol.json"),"--launch"]
    protocol = json.loads((root/"loaded_protocol.json").read_text())
    assert protocol==receipt["protocol"] and protocol["server_pid"]==receipt["server_pid"] and protocol["real_input_parity_passed"]
    schema_protocol_check(protocol,condition,manifest,root,[])
    records,banks = load_banks([r["path"] for r in receipt["banks"]])
    assert banks==receipt["banks"] and representatives(records)==receipt["indices"] and len(receipt["fixed_reference"])==5
    for row,index in zip(receipt["fixed_reference"],receipt["indices"],strict=True):
        assert row["bank_index"]==index and row["seed"]==int(records[index]["seed"])
        assert row["input_hash"]==input_digest(records[index]["obs"],records[index]["instruction"])
    assert [(r["arm"],r["when"]) for r in receipt["controls"]]==[(a,w) for a in ARMS for w in ("before","after")]
    for control in receipt["controls"]:
        assert control["comparison"]==fixed_checks(receipt["fixed_reference"],control["results"],condition["horizon"]) and control["comparison"]["passed"]
    official = official_scene_identities(manifest)
    data,input_sets = {},{}
    for row,arm in zip(receipt["arms"],ARMS,strict=True):
        assert row["arm"]==arm and row["exit_codes"]==[0,0] and len(set(row["pids"]))==2
        assert row["commands"]==commands(SimpleNamespace(manifest=manifest_path,model=receipt["model"],output=str(recorded_root),port=receipt["port"]),arm)
        assert row["wall_s"]==row["finished_monotonic_s"]-row["started_monotonic_s"]>0
        before,after = [c for c in receipt["controls"] if c["arm"]==arm]
        assert receipt["warm_finished_monotonic_s"]<=before["started_monotonic_s"]<=before["finished_monotonic_s"]<=row["started_monotonic_s"]
        assert row["finished_monotonic_s"]<=after["started_monotonic_s"]<=after["finished_monotonic_s"]
        expected = {root/arm/condition["id"]/"two_workers"/s["suite"]/f"task{s['task']:02d}"/f"init{s['init']:02d}" for s in manifest["e2"]["episodes"]}
        actual = {p.parent for p in (root/arm/condition["id"] / "two_workers").glob("*/task*/init*/complete.json")}
        assert actual==expected and not list((root/arm).rglob("probe_failure.json")) and not list((root/arm).rglob("technical_error.json"))
        data[arm],input_sets[arm] = {},{}
        for path in sorted(expected):
            with patch.object(final_audit,"protocol_check",lambda p,c,m:schema_protocol_check(p,c,m,root,[])):
                episode = final_audit.audit_episode(path,condition,manifest,True)
            identity = json.loads((path/"identity.json").read_text())
            key = tuple(episode[k] for k in ("suite","task","init"))
            assert all(identity[k]==v for k,v in official[key].items()) and identity["protocol"]==protocol
            execution = json.loads((path/"probe_execution.json").read_text())
            assert execution==identity["active_environment_execution_revision"]==episode["active_environment_execution_revision"]
            assert execution["revision"]==revision() and execution["arm"]==arm and execution["worker_pid"] in row["pids"]
            assert execution["warm_steps"]==10 and execution["requests"]==len(episode["infer"])
            assert episode["video"] is False and episode["save_observations"] is False
            physics = lines(path/"probe_physics.jsonl")
            assert len(physics)==10+episode["actions"]==execution["physics_steps"]
            assert [r["index"] for r in physics]==list(range(len(physics)))
            assert [r["phase"] for r in physics]==["warmup"]*10+["quality"]*episode["actions"]
            assert all(r["physics"] and r["noncamera"] and r["rng"] for r in physics)
            episode.update(physics=physics,telemetry=lines(path/"telemetry.jsonl"))
            data[arm][key] = episode
            inputs = []
            assert len(list((path/"probe_inputs").glob("*.npz")))==len(episode["infer"])
            for index,infer in enumerate(episode["infer"]):
                with np.load(path/"probe_inputs"/f"request{index:04d}.npz",allow_pickle=False) as saved:
                    observation = {k:saved[k] for k in saved.files if k not in ("instruction","seed")}
                    assert input_digest(observation,str(saved["instruction"]))==infer["input_hash"] and int(saved["seed"])==infer["seed"]
                    inputs.append({k:saved[k].copy() for k in saved.files})
            input_sets[arm][key] = inputs
    pairs = []
    for key,a in data["baseline"].items():
        b = data["sparse"][key]
        pair = compare_pair(a,b)
        left,right = input_sets["baseline"][key],input_sets["sparse"][key]
        pair["all_policy_pixels_numeric_inputs_exact"] = len(left)==len(right) and all(x.keys()==y.keys() and
            all(x[k].dtype==y[k].dtype and np.array_equal(x[k],y[k]) for k in x) for x,y in zip(left,right))
        pairs.append(dict(scene=list(key),checks=pair,passed=all(pair.values())))
    speedup = receipt["arms"][0]["wall_s"]/receipt["arms"][1]["wall_s"]
    raw_files = {str(p.resolve()):sha(p) for arm in ARMS for p in (root/arm).rglob("*") if p.is_file() and p.suffix in (".json",".jsonl",".npz")}
    suppressed = sum(sum(json.loads(p.read_text())["skipped_rgb_samples"].values()) for p in (root/"sparse").rglob("probe_execution.json"))
    assert suppressed>0,"Vacuous sparse arm did not suppress a single RGB sample"
    return dict(integrity_passed=True,pairs=pairs,all8_exact=len(pairs)==8 and all(p["passed"] for p in pairs),
        measured_same_service_same_gpu_profile_speedup=speedup,
        suppressed_rgb_samples=suppressed,raw_profile_file_sha256=raw_files,
        explicit_adoption_gate="All16 valid; all8 paired outcomes/official init/all policy pixels+numeric inputs/noise/actions/executed lengths and every physics/noncamera/RNG/reward/done hash exact; before/after5controls<=1e-5; positive same-service/GPU arm-wall improvement; video bypass unchanged. Parent approval and separate immutable execution revision required before ANY quality migration.",
        adoption_automatically_authorized=False,validation_sha256=sha(root/"probe.json"),audit_source_sha256=sha(__file__),execution_revision=revision())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--model",choices=("pi0","pi05","xiaomi-robotic0","db-cogact"))
    parser.add_argument("--gpu")
    parser.add_argument("--port",type=int)
    parser.add_argument("--output")
    parser.add_argument("--bank")
    parser.add_argument("--launch",action="store_true")
    parser.add_argument("--audit",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--root",help=argparse.SUPPRESS)
    parser.add_argument("--arm",choices=ARMS,help=argparse.SUPPRESS)
    parser.add_argument("--shard-index",type=int,help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest,verify_sources=True)
    if args.audit:
        if args.launch or not args.output:
            parser.error("Audit is read-only and requires --output existing-probe-root")
        print(json.dumps(audit(args.output,manifest),indent=2))
        return
    if not args.model or not args.port:
        parser.error("Explicit --model and --port required")
    if args.worker:
        if not args.launch or not args.root or args.arm is None or args.shard_index not in (0,1):
            parser.error("Internal profile worker requires own coordinator receipt")
        worker(args)
        return
    if not args.launch:
        print(json.dumps(dict(dry_run=True,episodes=16,profile_only=True,quality_episodes=0,arms=ARMS,
            condition=args.model+"_full",workers=2,execution_revision=revision(),launch_automatically_authorized=False)))
        return
    if not args.output or not args.bank or not args.gpu:
        parser.error("Launch requires explicit NEW --output, --bank and assigned --gpu")
    run(args,manifest)


if __name__=="__main__":
    main()
