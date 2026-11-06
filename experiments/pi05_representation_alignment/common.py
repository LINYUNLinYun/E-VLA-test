import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.pi05_hidden_probe.common import digest, observation_hash, write_json
from .prepare import OUT

GPU_UUIDS = {"GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25", "GPU-555cde36-5e41-bc44-0c69-ea47358fd703"}


def plan():
    value = json.loads((OUT / "plan.json").read_text())
    if digest({k: v for k, v in value.items() if k != "identity"}) != value["identity"]:
        raise ValueError("Frozen plan changed")
    return value


def source_states(item):
    with np.load(item["source_cache"], allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        if meta["spec"]["observation"] != item["observation_hash"]:
            raise ValueError("Source observation/cache mismatch")
        if not meta["spec"]["identity"]["audit"].get("pi05"):
            raise ValueError("Source cache is not the actual pi0.5 policy")
        value = data["layer_3"][item["source_offsets"]].copy()
    if value.shape != (256, 2048) or not np.isfinite(value).all():
        raise ValueError("Unexpected Source states")
    return value


def pair_observation(item):
    # The full manifest was validated before labels/questions were deliberately
    # removed from alignment inputs. Keep this loader label-free as well.
    keys = ("observation/image", "observation/wrist_image", "observation/state")
    with np.load(item["observation_path"], allow_pickle=False) as data:
        obs = {k: data[k].copy() for k in keys}
    for k in keys[:2]:
        if obs[k].dtype != np.uint8 or obs[k].ndim != 3 or obs[k].shape[-1] != 3:
            raise ValueError("Invalid paired camera image")
    if obs[keys[2]].shape != (8,) or not np.isfinite(obs[keys[2]]).all():
        raise ValueError("Invalid original robot state")
    obs["prompt"] = item["original_task_instruction"]
    if observation_hash(obs) != item["observation_hash"]:
        raise ValueError("Alignment observation content changed")
    return obs


def acquire_gpu(status_path):
    """One shared claim per authorized GPU; never use an occupied foreign GPU."""
    start = time.monotonic()
    while True:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        observed = []
        for line in raw.splitlines():
            index, uuid, used, free, util = [x.strip() for x in line.split(",")]
            if uuid not in GPU_UUIDS:
                continue
            info = {"index": int(index), "uuid": uuid, "used_mib": int(used), "free_mib": int(free), "utilization": int(util)}
            observed.append(info)
            if int(used) >= 1000 or int(free) <= 11000 or int(util) > 5:
                continue
            lease = open(f"/tmp/pi05_alignment_{uuid}.lock", "a+")
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            write_json(status_path, {"phase": "running", "pid": os.getpid(), "gpu": info})
            print("gpu_acquired", info, flush=True)
            return info, lease
        write_json(status_path, {"phase": "waiting", "pid": os.getpid(), "poll_seconds": 10,
                                 "wait_seconds": time.monotonic() - start, "gpus": observed})
        time.sleep(10)
