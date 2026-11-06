"""Extract Target pairs; Source states reuse the verified robot-policy cache."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.pi05_hidden_probe.backend import select_positions
from experiments.pi05_hidden_probe.common import write_json
from experiments.pi05_hidden_probe.hf_backend import HFBackend
from experiments.pi05_self_readout.run import robot_prompt, sha
from .common import acquire_gpu, pair_observation, plan, source_states
from .prepare import OUT


def main():
    config = plan()
    out = OUT / "pairs"
    out.mkdir(exist_ok=True)
    items = config["source_pairs"]
    if (out / "complete.json").exists():
        if json.loads((out / "complete.json").read_text())["identity"] != config["identity"]:
            raise ValueError("Existing extraction belongs to another plan")
        subprocess.run([sys.executable, "-u", "-m", "experiments.pi05_representation_alignment.fit"], check=True)
        return
    # A live worker waits on the exact authorized GPU UUIDs before loading CUDA.
    gpu, lease = acquire_gpu(out / "status.json")
    backend = HFBackend(config["target_checkpoint"])
    write_json(out / "model_audit.json", backend.audit())
    index = []
    for item in items:
        target_path = out / (item["observation_hash"] + ".npz")
        if target_path.exists():
            with np.load(target_path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"].item()))
                if meta["identity"] != config["identity"] or meta["observation_hash"] != item["observation_hash"]:
                    raise ValueError("Paired cache belongs to another experiment")
        else:
            prefix = backend.prepare_target(pair_observation(item), robot_prompt(item))
            positions = select_positions(prefix, "image:base_0_rgb")
            y = backend.forward(prefix, capture_layers=[3], capture_positions=positions).hidden_states[3]
            x = source_states(item)
            if x.shape != y.shape:
                raise ValueError("Corresponding Source/Target visual-token shape differs")
            meta = {"identity": config["identity"], "observation_hash": item["observation_hash"],
                    "sample_id": item["sample_id"], "partition": item["partition"],
                    "source_context": prefix.prompt_text, "source_cache": item["source_cache"]}
            temp = target_path.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.savez(stream, x=x, y=y, metadata=np.array(json.dumps(meta)))
            temp.replace(target_path)
        index.append({"path": str(target_path), "observation_hash": item["observation_hash"],
                      "partition": item["partition"], "sha256": sha(target_path)})
        write_json(out / "status.json", {"phase": "running", "gpu": gpu, "completed": len(index), "expected": len(items)})
        print("paired", len(index), len(items), item["sample_id"], flush=True)
    write_json(out / "index.json", {"identity": config["identity"], "items": index})
    write_json(out / "complete.json", {"complete": True, "identity": config["identity"], "pairs": len(index), "gpu": gpu})
    write_json(out / "status.json", {"phase": "complete", "pairs": len(index), "gpu": gpu})
    # Release model memory and our own GPU lease before the CPU fitting stage.
    import gc
    torch = backend.torch
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    lease.close()
    subprocess.run([sys.executable, "-u", "-m", "experiments.pi05_representation_alignment.fit"], check=True)


if __name__ == "__main__":
    main()
