"""Thin sequential controller queue, with a distinct port for every job.

Dry runs do not load models, inspect GPUs, launch children, or write receipts.
This orchestration source is identified in each receipt, outside the v2 manifest.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .common import atomic_json, read_manifest


VARIANTS = ("full_io", "lean_io", "reuse_env", "two_workers")


def load(path):
    return json.loads(Path(path).read_text())


def completed_quality(root, condition, manifest, stage, variant=None):
    """Check canonical keys and identities, not a count or success outcome."""
    inits = manifest["screen_inits"] if stage == "screen" else manifest["confirm_inits"]
    expected = ({(r["suite"], r["task"], r["init"]) for r in manifest["e2"]["episodes"]}
                if stage == "profile" else
                {(s,t,i) for s in manifest["suites"] for t in manifest["tasks"] for i in inits})
    directory = Path(root)/condition["id"]/(variant or "episodes")
    actual = set()
    for marker in directory.glob("*/task*/init*/complete.json"):
        if ".invalid." in str(marker):
            continue
        p = marker.parent
        complete, result, identity = (load(p/name) for name in ("complete.json", "result.json", "identity.json"))
        for row in (complete,result,identity):
            if row.get("manifest_hash") != manifest["manifest_hash"]:
                raise ValueError(f"Existing completion belongs to another manifest: {p}")
        if not complete.get("valid") or not result.get("valid") or result.get("condition") != condition["id"] or identity.get("condition") != condition:
            raise ValueError(f"Invalid existing completion must be audited, not skipped: {p}")
        key = tuple(result[k] for k in ("suite", "task", "init"))
        if (p.parent.parent.name,p.parent.name,p.name) != (key[0],f"task{key[1]:02d}",f"init{key[2]:02d}"):
            raise ValueError(f"Completion scene/path mismatch: {p}")
        if tuple(identity[k] for k in ("suite","task","init")) != key:
            raise ValueError(f"Completion scene/identity mismatch: {p}")
        if stage == "profile" or key[2] in inits:
            if key in actual:
                raise ValueError(f"Duplicate completion key: {p}")
            actual.add(key)
    if actual-expected:
        raise ValueError(f"Unexpected completed keys for {condition['id']}: {sorted(actual-expected)}")
    return actual == expected


def completed_benchmark(root, condition, manifest, banks=None):
    path = Path(root)/condition["id"]/"benchmark.json"
    if not path.exists():
        return False
    b = load(path)
    valid = (b.get("manifest_hash") == manifest["manifest_hash"] and b.get("condition") == condition
             and b.get("warmup") == 20 and b.get("samples") == 100 and len(b.get("rows",[])) == 100
             and b.get("protocol",{}).get("manifest_hash") == manifest["manifest_hash"]
             and b.get("protocol",{}).get("condition_id") == condition["id"])
    rows = b.get("rows",[])
    valid = valid and [r.get("index") for r in rows] == list(range(100))
    valid = valid and len({r.get("input_hash") for r in rows}) == 100 and all(r.get("input_hash") for r in rows)
    if banks:
        requested = {str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in banks}
        recorded = {str(Path(r["path"]).resolve()):r["sha256"] for r in b.get("banks",[])}
        valid = valid and requested == recorded
    if not valid:
        raise ValueError(f"Existing benchmark is incomplete or has another identity; refusing overwrite: {path}")
    return True


def plan(args, manifest):
    candidates = manifest["conditions"] + (manifest["benchmark_references"] if args.operation == "benchmark" else [])
    known = {c["id"] for c in candidates}
    if args.condition and set(args.condition)-known:
        raise ValueError(f"Unknown conditions: {sorted(set(args.condition)-known)}")
    selected = [c for c in candidates if (not args.condition or c["id"] in args.condition)
                and (not args.model or c["model"] in args.model)
                and (not args.experiment or c["experiment"] == args.experiment)]
    if args.stage == "confirm" and args.operation == "run":
        if not args.selection:
            raise ValueError("Confirm queue requires immutable --selection")
        promotion = load(args.selection)
        if promotion["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("Promotion/manifest identity mismatch")
        selected = [c for c in selected if c["id"] in promotion["selected_conditions"]]
    if args.stage == "profile" and args.operation == "run":
        if not args.variants:
            raise ValueError("Profile queue requires --variants (use all for ordered four arms)")
        if args.condition and any(c["skip"] for c in selected):
            raise ValueError("Profile explicit conditions must be Full-only")
        selected = [c for c in selected if not c["skip"]]
        variants = list(VARIANTS) if args.variants == ["all"] else args.variants
        if len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
            raise ValueError("Profile variants must be unique frozen arm names, or all")
    else:
        if args.variants:
            raise ValueError("--variants is only valid for run --stage profile")
        variants = [None]
    if not selected:
        raise ValueError("No conditions matched the explicit filters/selection")
    jobs = []
    for c in selected:
        for variant in variants:
            port = args.port_start + len(jobs)
            if not 1 <= port <= 65535:
                raise ValueError("Queue port range exceeds valid TCP ports")
            command = [sys.executable,"-m","preparations.idea1_v3_1.controller",args.operation,
                       "--manifest",args.manifest,"--output",args.output,"--condition",c["id"],
                       "--stage",args.stage,"--port",str(port),"--workers",str(args.workers)]
            for option, value in (("--gpu",args.gpu),("--vision-parity",args.vision_parity),
                                  ("--environment-reuse",args.environment_reuse),("--selection",args.selection),
                                  ("--variant",variant)):
                if value is not None:
                    command += [option,str(value)]
            if args.bank:
                command += ["--bank",*args.bank]
            if args.launch:
                command += ["--launch"]
            skip = (completed_benchmark(args.output,c,manifest,args.bank) if args.operation == "benchmark" else
                    completed_quality(args.output,c,manifest,args.stage,variant))
            jobs.append(dict(condition=c["id"],variant=variant,port=port,command=command,
                             status="skipped_complete" if skip else "pending",pid=None,exit_code=None))
    return jobs


def execute(jobs, receipt, receipt_path, popen=subprocess.Popen):
    """Only own sequential controller subprocesses are launched; failures stop."""
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        raise FileExistsError("Batch receipts are never overwritten by another invocation")
    atomic_json(receipt_path,receipt)
    for index,job in enumerate(jobs):
        if job["status"] == "skipped_complete":
            continue
        log = receipt_path.parent/f"{receipt_path.stem}.job{index:03d}.log"
        with log.open("x") as stream:
            job.update(status="starting",log=str(log),started_host_s=time.time())
            atomic_json(receipt_path,receipt)
            try:
                child = popen(job["command"],stdout=stream,stderr=subprocess.STDOUT)
                job.update(pid=child.pid,status="running")
                atomic_json(receipt_path,receipt)
                print(json.dumps(dict(condition=job["condition"],variant=job["variant"],pid=child.pid,port=job["port"],log=str(log))),flush=True)
                code = child.wait()
            except BaseException as exc:
                job.update(status="interrupted_or_launch_error",error=repr(exc),finished_host_s=time.time())
                receipt["status"] = "interrupted_or_launch_error"
                # No signalling or killing: an interrupted own controller's PID
                # remains recorded for the operator to inspect safely.
                atomic_json(receipt_path,receipt)
                raise
        job.update(exit_code=code,status="completed" if code == 0 else "failed",finished_host_s=time.time())
        receipt["status"] = "running" if code == 0 else "stopped_on_failure"
        atomic_json(receipt_path,receipt)
        if code:
            return code
    receipt.update(status="completed",finished_host_s=time.time())
    atomic_json(receipt_path,receipt)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("run","benchmark"))
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--condition",nargs="+")
    parser.add_argument("--model",nargs="+",choices=("pi0","pi05","xiaomi-robotic0","db-cogact"))
    parser.add_argument("--experiment",choices=("E1","E3"))
    parser.add_argument("--stage",choices=("screen","confirm","profile"),default="screen")
    parser.add_argument("--variants",nargs="+",choices=(*VARIANTS,"all"))
    parser.add_argument("--bank",nargs="+")
    parser.add_argument("--gpu")
    parser.add_argument("--workers",type=int,choices=(1,2),default=1)
    parser.add_argument("--vision-parity")
    parser.add_argument("--environment-reuse",choices=("on","off"))
    parser.add_argument("--selection")
    parser.add_argument("--port-start",type=int,default=9200)
    parser.add_argument("--receipt",help="New batch JSON path; launch defaults to output/batch_receipts/unique.json")
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if args.launch and (args.gpu is None or not args.bank):
        parser.error("--launch requires one assigned --gpu and real --bank")
    manifest = read_manifest(args.manifest,verify_sources=args.launch)
    jobs = plan(args,manifest)
    receipt = dict(queue_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   manifest_hash=manifest["manifest_hash"],manifest_path=str(Path(args.manifest).resolve()),
                   argv=sys.argv,queue_pid=os.getpid(),created_host_s=time.time(),jobs=jobs,
                   status="dry_run" if not args.launch else "planned",dry_run=not args.launch,
                   scope="Sequential unchanged controller CLI; unique ports; no new scientific factors")
    if not args.launch:
        if args.receipt:
            path = Path(args.receipt)
            if path.exists():
                raise FileExistsError(path)
            atomic_json(path,receipt)
        print(json.dumps(receipt,indent=2))
        return
    path = Path(args.receipt) if args.receipt else Path(args.output)/"batch_receipts"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex+".json")
    print(json.dumps(dict(batch_receipt=str(path),jobs=len(jobs))),flush=True)
    raise SystemExit(execute(jobs,receipt,path))


if __name__ == "__main__":
    main()
