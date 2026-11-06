"""Same-service request and all-camera vision timings on >=100 real inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from .common import atomic_json, read_manifest
from .server import rpc


def load_banks(paths):
    from preparations.idea1_v2_1.runtime_client import input_digest
    records,identities = [],[]
    for path in paths:
        raw = Path(path).read_bytes()
        bank = pickle.loads(raw)
        records.extend(bank)
        identities.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),observations=len(bank)))
    unique,seen = [],set()
    for record in records:
        identity = input_digest(record["obs"],record["instruction"])
        if identity not in seen:
            unique.append(record)
            seen.add(identity)
    return unique,identities


def benchmark(port,records,warmup=20,samples=100):
    if len(records)<samples:
        raise ValueError("Supply at least100 real observation records; do not label repeats as100 distinct observations")
    rows,cold = [],{}
    for index in range(warmup+samples):
        record = records[(index-warmup)%samples] if index>=warmup else records[index%len(records)]
        request = dict(obs=record["obs"],instruction=record["instruction"],suite=record.get("suite"),seed=int(record.get("seed",index)))
        tick = time.perf_counter()
        response = rpc(port,dict(op="predict",**request))
        total = (time.perf_counter()-tick)*1000
        vision = rpc(port,dict(op="vision",obs=request["obs"],instruction=request["instruction"]))
        if index==0:
            cold = dict(first_completed_request_ms=total,first_vision_ms=vision["vision_ms"],
                        definition="Includes any first-trace compilation; excluded from steady-state measured rows")
        if index>=warmup:
            rows.append(dict(index=index-warmup,seed=request["seed"],request_roundtrip_ms=total,
                             server_request_ms=response["server_dispatch_ms"],vision_ms=vision["vision_ms"],
                             noise_hash=response.get("noise_hash"),input_hash=response["input_hash"],
                             stage_timing=response.get("timing",{}),vision_details=vision,
                             parity_actions=np.asarray(response["actions"]).tolist() if index-warmup<2 else None))
    summary = {key:dict(zip(("p50","p95","p99"),np.percentile([row[key] for row in rows],[50,95,99]).tolist()))
               for key in ("request_roundtrip_ms","server_request_ms","vision_ms")}
    return dict(rows=rows,summary=summary,warmup=warmup,samples=samples,cold=cold,
                scope="Vision standalone stage and completed request measured separately; vision latency is sum over all policy cameras")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--condition",required=True)
    parser.add_argument("--bank",required=True,nargs="+")
    parser.add_argument("--port",required=True,type=int)
    parser.add_argument("--output",required=True)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    condition = next(c for c in manifest["conditions"]+manifest["benchmark_references"] if c["id"]==args.condition)
    if not args.launch:
        print(json.dumps(dict(dry_run=True,condition=condition,bank=args.bank,warmup=20,samples=100)))
        return
    records,banks = load_banks(args.bank)
    first = records[0]
    parity = rpc(args.port,dict(op="parity",obs=first["obs"],instruction=first["instruction"],seed=int(first.get("seed",0))))
    if not parity.get("passed") or parity.get("noise_paired") is False:
        raise RuntimeError("Real-input parity failed before benchmarking")
    protocol = rpc(args.port,dict(op="health"))["protocol"]
    if protocol["manifest_hash"]!=manifest["manifest_hash"] or protocol["condition_id"]!=condition["id"]:
        raise ValueError("Benchmark service does not match frozen condition")
    result = benchmark(args.port,records)
    result.update(manifest_hash=manifest["manifest_hash"],condition=condition,banks=banks,protocol=protocol)
    atomic_json(args.output,result)
    print(json.dumps(result["summary"],indent=2))


if __name__ == "__main__":
    main()
