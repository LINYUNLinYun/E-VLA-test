"""One model load: all-suite E0 gate, E1, all-suite E2, optional E3 service."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle

import torch

from .experiments import e0, e1, e2, load_backend, native_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank-root", required=True, help="Directory containing <suite>/task00/init07/real_predict_inputs_first25.pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--observations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--serve-port", type=int)
    args = parser.parse_args()
    backend = load_backend(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    suites = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
    banks = {}
    weight_hash = hashlib.sha256()
    with (Path(args.checkpoint)/"model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            weight_hash.update(block)
    for suite in suites:
        path = Path(args.bank_root)/suite/"task00/init07/real_predict_inputs_first25.pkl"
        raw = path.read_bytes()
        bank = pickle.loads(raw)
        banks[suite] = [backend.prepare(native_raw(record)) for record in bank[:args.observations]]
        destination = output/suite
        destination.mkdir(parents=True, exist_ok=True)
        identity = dict(backend.identity, checkpoint=args.checkpoint, bank=str(path), bank_sha256=hashlib.sha256(raw).hexdigest(),
                        gpu_uuid=str(getattr(torch.cuda.get_device_properties(backend.device), "uuid", None)),
                        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        weights_sha256=weight_hash.hexdigest(), args=vars(args),
                        implementation_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
        identity_path = destination/"identity.json"
        if identity_path.exists():
            previous = json.loads(identity_path.read_text())
            if previous.get("bank_sha256") != identity["bank_sha256"] or previous.get("weights_sha256") != identity["weights_sha256"]:
                raise ValueError("Resume identity mismatch: existing bank or checkpoint differs")
            for filename in ("experiments.py", "openpi_backend.py"):
                old = {Path(key).name:value for key,value in previous.get("implementation_sha256", {}).items()}
                new = {Path(key).name:value for key,value in identity["implementation_sha256"].items()}
                if old.get(filename) != new.get(filename):
                    raise ValueError(f"Resume implementation mismatch: {filename}")
            if previous.get("config") != identity["config"] or previous.get("num_steps") != identity["num_steps"]:
                raise ValueError("Resume model configuration mismatch")
            (destination/"resume_identity.json").write_text(json.dumps(identity, indent=2))
        else:
            identity_path.write_text(json.dumps(identity, indent=2))
        e0_path = destination/"e0.json"
        result = json.loads(e0_path.read_text()) if e0_path.exists() else e0(backend, banks[suite][0], atol=1e-5, rtol=1e-5)
        if not e0_path.exists():
            e0_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(stage="e0", suite=suite, passed=result["passed"])), flush=True)
        if not result["passed"]:
            raise RuntimeError("All-suite E0 gate failed")
    gc.collect()
    torch.cuda.empty_cache()
    e1_path = output/"libero_spatial/e1.json"
    result = json.loads(e1_path.read_text()) if e1_path.exists() else e1(backend, banks["libero_spatial"], args.warmup, args.samples, output/"libero_spatial/trace.json")
    if len(result["rows"]) != args.samples * 2:
        raise ValueError("Existing E1 does not match frozen sample count")
    if any(row["output_error"]["max_abs"] > 1e-5 for row in result["rows"]):
        raise RuntimeError("E1 paired output mismatch")
    if not e1_path.exists():
        e1_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(stage="e1", summary=result["summary"], trace_summary=result["trace_summary"])), flush=True)
    for suite in suites:
        e2_path = output/suite/"e2.json"
        result = json.loads(e2_path.read_text()) if e2_path.exists() else e2(backend, banks[suite])
        if len(result["rows"]) != args.observations * 3:
            raise ValueError("Existing E2 does not match frozen observation/interval count")
        if not e2_path.exists():
            e2_path.write_text(json.dumps(result, indent=2))
        if any(row["serial_async_error"]["max_abs"] > 1e-5 for row in result["rows"]):
            raise RuntimeError("E2 fixed-version serial/async mismatch")
        print(json.dumps(dict(stage="e2", suite=suite, rows=len(result["rows"]))), flush=True)
    banks.clear()
    gc.collect()
    torch.cuda.empty_cache()
    (output/"complete.json").write_text(json.dumps(dict(valid=True, stages=["e0", "e1", "e2"], suites=suites)))
    if args.serve_port:
        from .policy import PipelinePolicy
        from .server import serve
        serve(PipelinePolicy(backend), args.serve_port)


if __name__ == "__main__":
    main()
