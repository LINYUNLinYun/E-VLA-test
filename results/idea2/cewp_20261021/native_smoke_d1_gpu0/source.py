"""Native JAX fresh-KV reference on the CEWP observation/noise sequence.

This is a cross-framework performance reference, not an old-KV correctness
oracle. CUDA Torch generates explicit noise before timing; original JAX
Policy.infer handles preprocessing, noise upload and CPU output completion.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import threading
import time

import numpy as np


def statistics(values):
    return dict(n=len(values), mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)))


class MemorySampler:
    """Sample assigned-device usage; container and driver PIDs may differ."""
    def __init__(self, gpu_uuid):
        self.gpu_uuid = gpu_uuid
        self.rows, self.errors = [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        try:
            output = subprocess.check_output([
                "nvidia-smi", "--id="+self.gpu_uuid,
                "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=5)
            self.rows.append(dict(monotonic_s=time.perf_counter(), device_used_mib=float(output.strip())))
        except Exception as error:
            self.errors.append(str(error))

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=6)
        self.sample()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    parser.add_argument("--checkpoint", default="shared/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, choices=(1, 5, 10), default=10)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--instruction-index", type=int, default=0)
    parser.add_argument("--sequence", choices=("continuous", "paired"), default="continuous",
                        help="continuous uses observation i; paired matches CEWP measured observation 2*i+1")
    parser.add_argument("--smoke", action="store_true", help="Allow fewer samples; never label as steady benchmark")
    args = parser.parse_args()
    if args.count < 1 or args.warmup < 1:
        parser.error("Positive count and warmup required")
    if not args.smoke and (args.count < 100 or args.warmup < 10):
        parser.error("Steady reference requires count>=100 and warmup>=10; use --smoke for entrypoint checks")
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    # Must precede the first JAX import; caller may have inherited CPU-only JAX
    # settings from the Torch experiment environment.
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import torch
    import jax
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training import config
    from .experiments import native_raw
    from preparations.idea1_v2_1.runtime_client import input_digest

    source = Path(__file__).read_bytes()
    (dest/"source.py").write_bytes(source)
    bank_bytes = Path(args.bank).read_bytes()
    original = pickle.loads(bank_bytes)
    instruction = original[args.instruction_index]["instruction"]
    selected = [i for i, record in enumerate(original) if record["instruction"] == instruction]
    if len(selected) < 2:
        raise ValueError("Need at least two observations of the selected instruction")
    records = [original[i] for i in selected]
    model_config = config.get_config("pi05_libero")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
    noise_shape = (1, model_config.model.action_horizon, model_config.model.action_dim)
    noise = {}
    noise_start = time.perf_counter()
    for seed in [*(900+i for i in range(args.warmup)), *(1701+i for i in range(args.count))]:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise[seed] = torch.randn(noise_shape, generator=generator, device=device,
                                  dtype=torch.float32)[0].cpu().numpy()
    noise_setup_ms = (time.perf_counter()-noise_start)*1000
    np.savez(dest/"noise.npz", **{str(seed): value for seed, value in noise.items()})
    result = dict(args=vars(args), rows=[], bank_sha256=hashlib.sha256(bank_bytes).hexdigest(),
                  selected_bank_indices=selected, instruction=instruction, gpu_uuid=gpu_uuid,
                  source_sha256=hashlib.sha256(source).hexdigest(),
                  backend="original OpenPI JAX pi05_libero", model_config=repr(model_config.model),
                  num_steps=args.steps, state_path="none", checkpoint=str(Path(args.checkpoint).resolve()),
                  freshness="same-observation full fresh KV for every request; differs from old-KV pipeline",
                  boundary="native_raw + original Policy.infer preprocessing/noise H2D/inference/output D2H and transforms; explicit CUDA noise generation excluded",
                  noise_setup_ms=noise_setup_ms, noise_method="torch CUDA Generator.manual_seed, torch.randn FP32, identical shape to OpenPIBackend.noise",
                  comparison="No cross-framework exact-output equivalence is assumed; D1/5/10 are independent matched-step strata",
                  memory_definition="500 ms nvidia-smi memory.used samples of the assigned device, including contexts/driver; sampled maximum may miss transients, not Torch allocated or process-only memory",
                  mode="smoke" if args.smoke else "steady")
    def save():
        (dest/"result.json").write_text(json.dumps(result, indent=2))
    save()
    loading = time.perf_counter()
    policy = create_trained_policy(model_config, args.checkpoint, sample_kwargs={"num_steps":args.steps})
    if policy._is_pytorch_model or not policy._model.pi05:
        raise ValueError("This reference requires original JAX pi05 checkpoint")
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Native reference must execute on the assigned GPU")
    result["model_load_ms"] = (time.perf_counter()-loading)*1000
    result["jax_devices"] = [str(d) for d in jax.devices()]
    result["versions"] = dict(jax=jax.__version__, torch=torch.__version__)

    def infer(index, seed):
        record = records[index % len(records)]
        begin = time.perf_counter()
        response = policy.infer(native_raw(record), noise=noise[seed])
        actions = np.asarray(response["actions"])
        elapsed = (time.perf_counter()-begin)*1000
        if actions.shape != (model_config.model.action_horizon, 7) or not np.isfinite(actions).all():
            raise AssertionError("Invalid native action output")
        return actions.copy(), elapsed, record

    cold = []
    for i in range(args.warmup):
        _, elapsed, _ = infer(i, 900+i)
        cold.append(elapsed)
    result["warmup_cpu_cpu_ms"] = cold
    outputs = []
    with MemorySampler(gpu_uuid) as memory:
        for i in range(args.count):
            index = i if args.sequence == "continuous" else 2*i+1
            seed = 1701+i
            actions, elapsed, record = infer(index, seed)
            outputs.append(actions)
            result["rows"].append(dict(index=index % len(records), bank_index=selected[index % len(records)],
                seed=seed, cpu_cpu_ms=elapsed, input_hash=input_digest(record["obs"], record["instruction"]),
                noise_sha256=hashlib.sha256(noise[seed].tobytes()).hexdigest(), observation_age=0))
            if i % 20 == 19:
                print("Native completed", i+1, flush=True)
    result["memory_samples"] = memory.rows
    result["memory_errors"] = memory.errors
    result["sampled_peak_device_used_mib"] = max((r["device_used_mib"] for r in memory.rows), default=None)
    result["summary"] = statistics([row["cpu_cpu_ms"] for row in result["rows"]])
    np.save(dest/"actions.npy", np.stack(outputs))
    repeat, _, _ = infer(0 if args.sequence == "continuous" else 1, 1701)
    delta = np.abs(repeat.astype(np.float64)-outputs[0].astype(np.float64))
    result["repeatability"] = dict(max_abs=float(delta.max()), mae=float(delta.mean()),
                                   atol=1e-5, passed=bool(delta.max() <= 1e-5))
    save()
    with (dest/"timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    if not result["repeatability"]["passed"]:
        raise AssertionError("Same-input native repeatability failed")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
