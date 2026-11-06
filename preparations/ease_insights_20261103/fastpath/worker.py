"""Thin worker around the original native pi0 LIBERO episode implementation."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time
import uuid

from preparations.idea1_v3_1.common import atomic_json, lock, read_manifest
from .runner import cases, sha


def episode_dir(output, condition, scene):
    return Path(output)/condition/"episodes"/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"


def completed(output, condition, scene, manifest_hash):
    directory=episode_dir(output,condition,scene)
    if not (directory/"episode_receipt.json").exists(): return False
    receipt=json.loads((directory/"episode_receipt.json").read_text())
    if receipt["manifest_hash"]!=manifest_hash or receipt["condition"]!=condition or receipt["case"]!=scene:
        raise ValueError("Completed episode belongs to another condition, case, or manifest")
    for name,expected in receipt["artifacts"].items():
        if sha(directory/name)!=expected: raise ValueError(f"Completed episode artifact changed: {directory/name}")
    row=json.loads((directory/"result.json").read_text())
    done=json.loads((directory/"complete.json").read_text())
    if row.get("manifest_hash")!=manifest_hash or done.get("manifest_hash")!=manifest_hash:
        raise ValueError("Mixed manifest in episode")
    if row.get("condition")!=condition or any(row.get(k)!=v for k,v in scene.items()):
        raise ValueError("Result identity differs from episode path")
    return bool(row.get("valid") and done.get("valid") and receipt["valid"])


def initial_gate(output, manifest_hash, condition, scene, observation_hash, noise_seed):
    directory=Path(output)/"initial_observations"/scene["suite"]/f"task{scene['task']:02d}"
    path=directory/f"init{scene['init']:02d}.json"
    payload=dict(manifest_hash=manifest_hash,case=scene,input_hash=observation_hash,seed=noise_seed)
    with lock(str(path)+".lock"):
        if path.exists():
            reference=json.loads(path.read_text())
            if any(reference.get(k)!=v for k,v in payload.items()):
                raise ValueError(f"Paired initial observation/noise mismatch for {scene}; reference {reference['condition']}, current {condition}")
        else:
            atomic_json(path,dict(payload,condition=condition))
    return dict(**payload,reference=str(path),passed=True)


def finalize(output, condition, scene, manifest_hash):
    directory=episode_dir(output,condition,scene)
    result=json.loads((directory/"result.json").read_text())
    done=json.loads((directory/"complete.json").read_text())
    identity=json.loads((directory/"identity.json").read_text())
    if not result.get("valid") or not done.get("valid"): raise ValueError("Cannot finalize invalid scientific episode")
    if any(obj.get("manifest_hash")!=manifest_hash for obj in (result,done,identity)):
        raise ValueError("Cannot finalize another manifest")
    if result["condition"]!=condition or identity["condition"]["id"]!=condition:
        raise ValueError("Cannot finalize another condition")
    if any(result.get(k)!=v or identity.get(k)!=v for k,v in scene.items()):
        raise ValueError("Cannot finalize another case")
    infers=[json.loads(line) for line in (directory/"infer.jsonl").read_text().splitlines()]
    if len(infers)!=result["predictions"] or not infers: raise ValueError("Incomplete inference record")
    gate=initial_gate(output,manifest_hash,condition,scene,infers[0]["input_hash"],infers[0]["seed"])
    model=[r["timing"]["request_wall_ms"] for r in infers]
    rpc=[r["client_roundtrip_ms"] for r in infers]
    receipt=dict(valid=True,manifest_hash=manifest_hash,condition=condition,case=scene,initial_gate=gate,
        success=result["success"],executed_steps=result["actions"],predictions=result["predictions"],
        latency=dict(model_request_ms_mean=statistics.mean(model),model_request_ms_median=statistics.median(model),
            rpc_request_ms_mean=statistics.mean(rpc),model_request_ms_total=sum(model),
            scope="completed outputs after service parity and 20 warmup requests; cold compilation recorded per run; new shape compilation may remain"),
        gpu_uuid=os.environ.get("EASE_FASTPATH_GPU_UUID"),worker_pid=os.getpid(),
        artifacts={name:sha(directory/name) for name in ("identity.json","result.json","complete.json","infer.jsonl","telemetry.jsonl")})
    atomic_json(directory/"episode_receipt.json",receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",required=True); parser.add_argument("--condition",required=True)
    parser.add_argument("--port",type=int,required=True); parser.add_argument("--stage",choices=("pilot","confirm"),required=True)
    parser.add_argument("--shard-index",type=int,default=0); parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--launch",action="store_true")
    args=parser.parse_args(); output=Path(args.output).resolve()
    manifest=read_manifest(output/"manifest.json",verify_sources=args.launch)
    condition=next(c for c in manifest["conditions"] if c["id"]==args.condition)
    scenes=cases(manifest,args.stage,args.shard_index,args.shards)
    if not args.launch:
        print(json.dumps(dict(dry_run=True,episodes=len(scenes),condition=condition["id"]))); return
    evidence=manifest["parity_evidence"]
    if sha(evidence["path"])!=evidence["sha256"]: raise ValueError("Historical parity receipt changed")
    for path,expected in evidence["compatibility_dependencies"].items():
        if sha(path)!=expected: raise ValueError(f"Historical selected-scan parity no longer applies: {path}")
    import numpy as np
    from preparations.idea1_v3_1 import rollout
    from preparations.idea1_v2_1.runtime_client import input_digest
    class SeededEnvironments(rollout.Environments):
        def acquire(self,suite,task):
            random.seed(7); np.random.seed(7)
            return super().acquire(suite,task)
    pool=SeededEnvironments(reuse=False)
    original_rpc=rollout.rpc
    current={}
    def gated_rpc(port,request):
        if request.get("op")=="predict" and not current.get("checked"):
            initial_gate(output,manifest["manifest_hash"],condition["id"],current["scene"],
                input_digest(request["obs"],request["instruction"]),request["seed"])
            current["checked"]=True
        return original_rpc(port,request)
    rollout.rpc=gated_rpc
    try:
        for scene in scenes:
            directory=episode_dir(output,condition["id"],scene)
            with lock(str(directory)+".lock"):
                if completed(output,condition["id"],scene,manifest["manifest_hash"]): continue
                # A crash after atomic scientific completion does not authorize
                # rerunning its outcome: finish only the receipt bookkeeping.
                if (directory/"complete.json").exists():
                    print(json.dumps(finalize(output,condition["id"],scene,manifest["manifest_hash"])),flush=True); continue
                previous_errors=len(list(directory.parent.glob(directory.name+".invalid.*/technical_error.json")))
                if (directory/"technical_error.json").exists(): previous_errors+=1
                for attempt in range(previous_errors,manifest["technical_max_attempts"]):
                    if directory.exists(): directory.rename(directory.with_name(directory.name+".invalid."+uuid.uuid4().hex))
                    current.clear(); current.update(scene=scene,checked=False)
                    try:
                        rollout.run_episode(directory,args.port,condition,**scene,pool=pool,manifest_hash=manifest["manifest_hash"],video=False,save_observations=False)
                        print(json.dumps(finalize(output,condition["id"],scene,manifest["manifest_hash"])),flush=True)
                        break
                    except Exception as exc:
                        directory.mkdir(parents=True,exist_ok=True)
                        atomic_json(directory/"technical_error.json",dict(valid=False,error=repr(exc),attempt=attempt+1,time=time.time()))
                        pool.close()
                        # Protocol identity errors require diagnosis, not retries.
                        if isinstance(exc,ValueError) or attempt+1>=manifest["technical_max_attempts"]: raise
                else:
                    raise RuntimeError(f"Technical attempt budget exhausted: {directory}")
    finally:
        rollout.rpc=original_rpc; pool.close()


if __name__=="__main__": main()
