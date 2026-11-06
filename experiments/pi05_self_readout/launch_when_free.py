"""Launch the bounded self-readout diagnostic when an authorized GPU is free."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UUIDS = {"GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25",
         "GPU-555cde36-5e41-bc44-0c69-ea47358fd703"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / "resource_status.json"
    start = time.monotonic()
    attempt = 0
    while attempt < 3:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        available = []
        observations = []
        for line in raw.splitlines():
            index, uuid, used, free, util = [s.strip() for s in line.split(",")]
            if uuid not in UUIDS:
                continue
            observations.append({"index": int(index), "uuid": uuid, "memory_used_mib": int(used),
                                 "memory_free_mib": int(free), "utilization_percent": int(util)})
            # Do not share a busy GPU with an unrelated job even if it has a
            # little memory left. Preserve small pre-existing keeper contexts.
            if int(used) < 1000 and int(free) > 11000 and int(util) <= 5:
                available.append((int(index), uuid))
        if not available:
            status.write_text(json.dumps({"phase": "waiting", "poll_seconds": 10,
                                          "wait_seconds": time.monotonic() - start, "gpus": observations}, indent=2))
            time.sleep(10)
            continue
        index, uuid = available[0]
        attempt += 1
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
        command = [sys.executable, "-u", "-m", "experiments.pi05_self_readout.run", "--output", str(args.output),
                   "--max-groups", str(args.max_groups), "--device", "cuda:0", "--resume"]
        status.write_text(json.dumps({"phase": "running", "gpu_index": index, "gpu_uuid": uuid,
                                      "attempt": attempt, "command": command}, indent=2))
        print("launch", index, uuid, "attempt", attempt, flush=True)
        with (args.output / f"attempt{attempt:02d}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            status.write_text(json.dumps({"phase": "complete", "gpu_index": index, "gpu_uuid": uuid,
                                          "attempt": attempt, "returncode": 0}, indent=2))
            print("complete", flush=True)
            return
        text = (args.output / f"attempt{attempt:02d}.log").read_text()
        if "out of memory" not in text.lower():
            raise RuntimeError(f"Scientific/implementation failure; see attempt{attempt:02d}.log")
        print("technical_oom", "attempt", attempt, flush=True)
        time.sleep(10)
    raise RuntimeError("Repeated external contention: bounded retries exhausted")


if __name__ == "__main__":
    main()
