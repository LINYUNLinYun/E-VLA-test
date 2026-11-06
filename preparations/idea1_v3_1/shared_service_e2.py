"""Bounded E2 validation:32 episodes, four frozen arms, one native pi05 service."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

from .common import ROOT,PYTHONS,atomic_json,environment,read_manifest,seed
from .server import rpc
from .same_process_parity import representatives,sha,compare


def client_commands(manifest_path,output,port,variant):
    return [[str(PYTHONS["environment"]),"-m","preparations.idea1_v3_1.rollout","--manifest",manifest_path,
        "--condition","pi05_full","--output",output,"--port",str(port),"--stage","profile","--variant",variant["id"],
        "--shard-index",str(i),"--shards",str(variant["workers"]),"--launch"] for i in range(variant["workers"])]


def fixed_checks(reference,actual):
    assert len(reference)==len(actual)==5
    checks = []
    for a,b in zip(reference,actual,strict=True):
        assert a["input_hash"]==b["input_hash"] and a["noise_hash"]==b["noise_hash"] and a["seed"]==b["seed"]
        assert np.asarray(a["actions"]).shape == np.asarray(b["actions"]).shape == (10,7)
        checks.append(compare(a["actions"],b["actions"]))
    return dict(passed=all(c["passed"] for c in checks),checks=checks,raw_actions_available=False,
                raw_scope="Unchanged frozen RPC exposes decoded actions only; raw sampler outputs are not claimed")


def service(args):
    """Wrap persistence only; original factory, server and NN remain unchanged."""
    from . import models,server
    from preparations.idea1_v2_1.runtime_client import input_digest
    manifest = read_manifest(args.manifest,verify_sources=True)
    original_load = models.load
    scene_seeds = {seed(s["suite"],s["task"],s["init"],0):s for s in manifest["e2"]["episodes"]}
    class ObservationalPolicy:
        def __init__(self,policy):
            self.policy,self.phase,self.seen = policy,"controls",set()
        def __getattr__(self,name):
            return getattr(self.policy,name)
        def predict(self,**request):
            self.phase = request.pop("_validation_phase",self.phase)
            control = request.pop("_validation_control",False)
            tick = time.perf_counter()
            scene = scene_seeds.get(int(request["seed"]))
            if not control and self.phase.startswith("arm:") and scene and (self.phase,int(request["seed"])) not in self.seen:
                variant = self.phase.split(":",1)[1]
                p = Path(args.output)/"first_inputs"/variant/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"
                p.mkdir(parents=True,exist_ok=False)
                np.savez_compressed(p/"observation.npz",**request["obs"],instruction=np.asarray(request["instruction"]),seed=np.asarray(request["seed"]))
                atomic_json(p/"identity.json",dict(scene,seed=int(request["seed"]),input_hash=input_digest(request["obs"],request["instruction"]),
                    server_pid=os.getpid(),manifest_hash=manifest["manifest_hash"],npz_sha256=sha(p/"observation.npz")))
                self.seen.add((self.phase,int(request["seed"])))
            capture_ms = (time.perf_counter()-tick)*1000
            result = self.policy.predict(**request)
            result["validation_observational_capture_ms"] = capture_ms
            return result
    models.load = lambda condition:ObservationalPolicy(original_load(condition))
    sys.argv = ["server","--manifest",args.manifest,"--condition","pi05_full","--port",str(args.port),
                "--protocol-output",str(Path(args.output)/"loaded_protocol.json"),"--launch"]
    server.main()


def summarize(output,manifest,controls):
    root = Path(output)/"pi05_full"
    arms = {}
    server_pids = set()
    for variant in manifest["e2"]["variants"]:
        rows = {}
        for scene in manifest["e2"]["episodes"]:
            key = (scene["suite"],scene["task"],scene["init"])
            p = root/variant["id"]/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"
            complete,result,identity = (json.loads((p/f).read_text()) for f in ("complete.json","result.json","identity.json"))
            assert complete["valid"] and result["valid"] and complete["manifest_hash"]==manifest["manifest_hash"]
            assert identity["manifest_hash"]==manifest["manifest_hash"] and result["manifest_hash"]==manifest["manifest_hash"]
            server_pids.add(identity["protocol"]["server_pid"])
            infer = [json.loads(s) for s in (p/"infer.jsonl").read_text().splitlines()]
            rows[key] = dict(result=result,identity=identity,infer=infer)
        arms[variant["id"]] = rows
    assert len(server_pids)==1,"All32 episodes must use the same loaded model service"
    comparisons = []
    for variant,rows in arms.items():
        if variant=="full_io":
            continue
        pairs = []
        for key,row in rows.items():
            base = arms["full_io"][key]
            initials = row["identity"]["init_vector_sha256"]==base["identity"]["init_vector_sha256"]
            outcome = row["result"]["success"]==base["result"]["success"]
            same,unpaired,errors = 0,0,[]
            for a,b in zip(base["infer"],row["infer"]):
                if a["input_hash"]==b["input_hash"] and a["noise_hash"]==b["noise_hash"] and a["seed"]==b["seed"]:
                    same += 1
                    errors.append(compare(a["actions"],b["actions"]))
                else:
                    unpaired += 1
            unpaired += abs(len(base["infer"])-len(row["infer"]))
            first = {}
            for arm in ("full_io",variant):
                p = Path(output)/"first_inputs"/arm/key[0]/f"task{key[1]:02d}"/f"init{key[2]:02d}"
                metadata = json.loads((p/"identity.json").read_text())
                assert metadata["npz_sha256"]==sha(p/"observation.npz")
                assert metadata["input_hash"]==arms[arm][key]["infer"][0]["input_hash"]
                assert metadata["server_pid"] in server_pids
                with np.load(p/"observation.npz",allow_pickle=False) as saved:
                    first[arm] = {k:saved[k] for k in saved.files if k not in ("instruction","seed")}
                    from preparations.idea1_v2_1.runtime_client import input_digest
                    assert input_digest(first[arm],str(saved["instruction"]))==metadata["input_hash"]
            deltas = {}
            assert first["full_io"].keys()==first[variant].keys()
            for field,a in first["full_io"].items():
                b = first[variant][field]
                assert a.shape==b.shape
                difference = np.abs(a.astype(np.float64)-b.astype(np.float64))
                deltas[field] = dict(shape=list(a.shape),dtype=str(a.dtype),max_abs=float(difference.max()) if difference.size else 0.,
                    mean_abs=float(difference.mean()) if difference.size else 0.,different_fraction=float(np.mean(a!=b)) if a.size else 0.)
            pairs.append(dict(scene=list(key),same_official_initial=initials,same_outcome=outcome,matched_requests=same,
                unmatched_or_drifted_requests=unpaired,paired_action_checks=errors,first_input_deltas=deltas))
        comparisons.append(dict(variant=variant,pairs=pairs,all8_initials_outcomes_match=all(p["same_official_initial"] and p["same_outcome"] for p in pairs),
            matched_request_actions_pass=all(c["passed"] for p in pairs for c in p["paired_action_checks"])))
    return dict(complete_episodes=32,fixed_controls_pass=all(c["comparison"]["passed"] for c in controls),comparisons=comparisons,
        validation_passed=all(c["comparison"]["passed"] for c in controls) and all(c["all8_initials_outcomes_match"] and c["matched_request_actions_pass"] for c in comparisons),
        caveat="Only identical-input/noise requests are compared pointwise. Free closed-loop input drift is reported, not attributed to scheduling. This is not an automatic quality-adoption decision.")


def run(args,manifest):
    from .benchmark import load_banks
    from .controller import assert_gpu_free
    records,banks = load_banks([args.bank])
    indices = representatives(records)
    assert len(indices)==5
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    with socket.socket() as check:
        check.bind(("127.0.0.1",args.port))
    assert_gpu_free(args.gpu)
    receipt = dict(manifest_hash=manifest["manifest_hash"],helper_source_sha256=sha(__file__),banks=banks,indices=indices,
        condition="pi05_full",gpu=args.gpu,port=args.port,queue_pid=os.getpid(),status="starting",controls=[],arms=[],
        observational_io="One first-input NPZ per scene/arm is additionally saved in service wrapper, identically for all arms and included in timed wall. NN/server and rollout source unchanged.",
        raw_control_actions_available=False)
    path = output/"validation.json"
    command = [sys.executable,"-m","preparations.idea1_v3_1.shared_service_e2","--service","--manifest",args.manifest,
        "--output",args.output,"--port",str(args.port),"--launch"]
    receipt["server_command"] = command
    atomic_json(path,receipt)
    env = environment(args.gpu)
    live = []
    with (output/"service.log").open("x") as log:
        server = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt["server_pid"] = server.pid
        atomic_json(path,receipt)
        try:
            deadline = time.monotonic()+1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Own shared service exited during load")
                try:
                    protocol = rpc(args.port,dict(op="health"))["protocol"]
                    assert protocol["server_pid"]==server.pid and protocol["condition_id"]=="pi05_full" and protocol["manifest_hash"]==manifest["manifest_hash"]
                    break
                except (ConnectionError,OSError):
                    if time.monotonic()>deadline:
                        raise TimeoutError("Shared service load deadline")
                    time.sleep(1)
            receipt["protocol"] = protocol
            first = records[0]
            parity = rpc(args.port,dict(op="parity",obs=first["obs"],instruction=first["instruction"],seed=int(first["seed"])))
            assert parity["passed"] and parity["max_abs"]<=1e-5
            receipt["parity"] = parity
            for record in records[:20]:
                rpc(args.port,dict(op="predict",obs=record["obs"],instruction=record["instruction"],suite=record["suite"],seed=int(record["seed"]),_validation_control=True))
            receipt["warmup"] = 20
            def control(phase):
                results = []
                for index in indices:
                    record = records[index]
                    row = rpc(args.port,dict(op="predict",obs=record["obs"],instruction=record["instruction"],suite=record["suite"],seed=int(record["seed"]),
                        _validation_control=True,_validation_phase=phase))
                    results.append(dict(bank_index=index,seed=int(record["seed"]),input_hash=row["input_hash"],noise_hash=row["noise_hash"],actions=np.asarray(row["actions"]).tolist()))
                return results
            reference = control("controls:reference")
            receipt["fixed_reference"] = reference
            for variant in manifest["e2"]["variants"]:
                for when in ("before","after"):
                    if when=="before":
                        actual = control("arm:"+variant["id"])
                    else:
                        actual = control("controls:after:"+variant["id"])
                    check = dict(variant=variant["id"],when=when,results=actual,comparison=fixed_checks(reference,actual))
                    receipt["controls"].append(check)
                    atomic_json(path,receipt)
                    assert check["comparison"]["passed"],"Fixed same-service input/noise decoded actions changed"
                    if when=="after":
                        continue
                    commands = client_commands(args.manifest,args.output,args.port,variant)
                    arm = dict(variant=variant["id"],commands=commands,pids=[],exit_codes=[])
                    receipt["arms"].append(arm)
                    tick = time.perf_counter()
                    streams = []
                    try:
                        for i,cmd in enumerate(commands):
                            stream = (output/f"{variant['id']}_worker{i}.log").open("x")
                            streams.append(stream)
                            child = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                            live.append(child)
                            arm["pids"].append(child.pid)
                        atomic_json(path,receipt)
                        for child in live:
                            code = child.wait()
                            arm["exit_codes"].append(code)
                            if code:
                                raise RuntimeError(f"Own rollout worker exited {code}")
                        live.clear()
                    finally:
                        for child in live:
                            if child.poll() is None:
                                child.terminate()
                        for child in live:
                            child.wait()
                        live.clear()
                        for stream in streams:
                            stream.close()
                    arm["wall_s"] = time.perf_counter()-tick
                    atomic_json(path,receipt)
                    print(json.dumps(dict(variant=variant["id"],wall_s=arm["wall_s"])),flush=True)
            receipt["summary"] = summarize(args.output,manifest,receipt["controls"])
            base = receipt["arms"][0]["wall_s"]
            for arm in receipt["arms"]:
                arm["episodes_per_hour"] = 8*3600/arm["wall_s"]
                arm["aggregate_throughput_speedup"] = base/arm["wall_s"]
            receipt["status"] = "completed"
        except BaseException as exc:
            receipt.update(status="failed",error=repr(exc))
            raise
        finally:
            for child in live:
                if child.poll() is None:
                    child.terminate()
                child.wait()
            if server.poll() is None:
                try:
                    health = rpc(args.port,dict(op="health"))["protocol"]
                    assert health["server_pid"]==server.pid
                    rpc(args.port,dict(op="shutdown"))
                    server.wait(timeout=30)
                except Exception:
                    server.terminate()
                    server.wait(timeout=30)
            receipt["server_exit_code"] = server.returncode
            atomic_json(path,receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--bank")
    parser.add_argument("--gpu")
    parser.add_argument("--port",type=int,default=9700)
    parser.add_argument("--service",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if args.service:
        if not args.launch:
            parser.error("Service requires explicit --launch")
        service(args)
        return
    manifest = read_manifest(args.manifest,verify_sources=args.launch)
    if not args.launch:
        print(json.dumps(dict(dry_run=True,episodes=32,condition="pi05_full",variants=manifest["e2"]["variants"],port=args.port)))
        return
    if args.gpu!="0" or not args.bank:
        parser.error("This bounded validation is authorized only on --gpu 0 with --bank")
    run(args,manifest)


if __name__ == "__main__":
    main()
