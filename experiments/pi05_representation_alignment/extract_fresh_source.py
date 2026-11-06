"""Extract new-test robot-policy states in a process separate from the reader."""
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.pi05_hidden_probe.backend import Backend, select_positions
from experiments.pi05_hidden_probe.common import model_identity, observation_hash, write_json
from experiments.pi05_hidden_probe.data import load_observation, read_manifest
from experiments.pi05_hidden_probe.workflow import difference
from experiments.pi05_self_readout.run import sha
from .common import acquire_gpu, plan
from .prepare import OUT


def main():
    config = plan()
    out = OUT / "fresh_source"
    out.mkdir(exist_ok=True)
    if (out / "complete.json").exists():
        print("existing_fresh_source_preserved", flush=True)
        return
    while not (OUT / "fresh_data/complete.json").exists():
        write_json(out / "status.json", {"phase": "waiting_for_fresh_data", "pid": os.getpid()})
        time.sleep(10)
    rows = read_manifest(OUT / "fresh_data/manifest.jsonl")
    with np.load(config["source_pairs"][0]["source_cache"], allow_pickle=False) as data:
        old = json.loads(str(data["metadata"].item()))["spec"]["identity"]
    gpu, lease = acquire_gpu(out / "status.json")
    backend = Backend(**old["model"])
    identity = model_identity(old["model"], backend)
    for field in ("checkpoint_files_digest", "openpi_code_digest", "probe_code_digest"):
        if identity[field] != old[field]:
            raise ValueError(f"Source changed since alignment training features were extracted: {field}")
    write_json(out / "model_audit.json", identity)
    index = []
    for row in rows:
        obs = load_observation(row)
        content = observation_hash(obs)
        path = out / (content + ".npz")
        if not path.exists():
            prefix = backend.prepare_source(obs)
            positions = select_positions(prefix, "image:base_0_rgb")
            result = backend.forward(prefix, capture_layers=[3], capture_positions=positions)
            if not index:
                reference = backend.reference_forward(prefix)
                checks = {"normalized": difference(result.normalized, reference.normalized, 1e-5, 1e-5),
                          "logits": difference(result.logits, reference.logits, 1e-5, 1e-5)}
                write_json(out / "integrity.json", checks)
                if not all(r["passed"] for r in checks.values()):
                    raise ValueError("Native Source prefix integrity failed")
            with path.open("wb") as stream:
                np.savez(stream, x=result.hidden_states[3],
                         metadata=np.array(json.dumps({"identity": config["identity"], "observation_hash": content})))
        else:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                if metadata["identity"] != config["identity"] or metadata["observation_hash"] != content:
                    raise ValueError("Unexpected fresh Source cache")
        index.append({"sample_id": row["sample_id"], "observation_hash": content,
                      "path": str(path), "sha256": sha(path)})
        write_json(out / "status.json", {"phase": "running", "gpu": gpu, "completed": len(index), "expected": len(rows)})
        print("fresh_source", len(index), len(rows), row["sample_id"], flush=True)
    write_json(out / "index.json", {"identity": config["identity"], "items": index})
    write_json(out / "complete.json", {"complete": True, "identity": config["identity"], "images": len(index), "gpu": gpu})
    write_json(out / "status.json", {"phase": "complete", "images": len(index), "gpu": gpu})
    lease.close()


if __name__ == "__main__":
    main()
