"""Run one explicit prepared job on one assigned GPU, with an immutable receipt."""
import argparse
import json
import os
import subprocess

from preparations.idea2.batch_plan import source_hashes
from results.idea2.batch_study_20261031.wait_then_run import AFFINITIES, HERE, ROOT, now, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--gpu-index", required=True, type=int)
    p.add_argument("--name", required=True)
    p.add_argument("--diagnostic-batch-drift", action="store_true")
    p.add_argument("--ae-sm-min", type=int)
    p.add_argument("--lead", type=int)
    p.add_argument("--pairs", type=int)
    p.add_argument("--length", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--prepare-stream", choices=("caller", "prefix", "dedicated"))
    p.add_argument("--disable-lookahead", action="store_true")
    p.add_argument("--reverse", action="store_true")
    args = p.parse_args()
    if "/" in args.name or args.name.startswith("."):
        raise ValueError("Use a new simple output name")
    receipt = HERE / (args.name + ".execution.json")
    output = HERE / args.name
    if receipt.exists() or output.exists():
        raise RuntimeError("Existing run must be inspected; never overwrite or repeat blindly")
    plan = json.loads((ROOT / "results/idea2/batch_preparation_20261031/experiment_plan/plan.json").read_text())
    job = next(row for row in plan["jobs"] if row["id"] == args.job)
    command = list(job["argv"])
    command[command.index("--output") + 1] = str(output)
    command[command.index("--gpu-uuid") + 1] = args.gpu_uuid
    if args.ae_sm_min is not None:
        command[command.index("--ae-sm-min") + 1] = str(args.ae_sm_min)
    if args.lead is not None:
        command[command.index("--lead") + 1] = str(args.lead)
    for option, value in (("--pairs", args.pairs), ("--length", args.length), ("--warmup", args.warmup)):
        if value is not None:
            command[command.index(option) + 1] = str(value)
    if args.reverse:
        command.append("--reverse")
    if args.diagnostic_batch_drift:
        command.append("--diagnostic-batch-drift")
    if args.prepare_stream is not None:
        command += ["--prepare-stream", args.prepare_stream]
    if args.disable_lookahead:
        command.append("--disable-lookahead")
    if job["stage"] == "profile":
        for i, value in enumerate(command):
            if value.startswith("--output="):
                command[i] = "--output=" + str(HERE / (args.name + "_trace"))
    environment = dict(os.environ, **job["environment"])
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    affinity = set(AFFINITIES[args.gpu_index]) & os.sched_getaffinity(0)
    if affinity:
        os.sched_setaffinity(0, affinity)
    state = dict(status="starting", pid=os.getpid(), start_utc=now(),
                 job=args.job, gpu_uuid=args.gpu_uuid, output=str(output),
                 command=command, source_before=source_hashes(),
                 cpu_affinity=sorted(os.sched_getaffinity(0)))
    save(receipt, state)
    with (HERE / (args.name + ".console.log")).open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
        state.update(status="running", child_pid=child.pid)
        save(receipt, state)
        print("STARTED", args.name, child.pid, flush=True)
        child.wait()
    state.update(status="passed" if child.returncode == 0 else "failed",
                 exit_code=child.returncode, end_utc=now(), source_after=source_hashes())
    state["source_unchanged"] = state["source_before"] == state["source_after"]
    result = output / "result.json"
    if result.is_file():
        value = json.loads(result.read_text())
        state.update(benchmark_status=value.get("status"), error=value.get("error"),
                     summary=value.get("summary"))
    save(receipt, state)
    print("FINISHED", args.name, state["status"], flush=True)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
