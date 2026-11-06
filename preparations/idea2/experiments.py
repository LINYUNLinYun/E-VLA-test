"""E0/E1/E2 runner. Run with the OpenPI virtualenv and repository on PYTHONPATH."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import torch

from .openpi_backend import OpenPIBackend, cache_tensors, error


def load_backend(checkpoint, device="cuda:0", *, model_config="pi0_libero", num_steps=10):
    from openpi.training import config
    from openpi.policies.policy_config import create_trained_policy
    cfg = config.get_config(model_config)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = create_trained_policy(cfg, checkpoint, pytorch_device=device, sample_kwargs={"num_steps": num_steps})
    return OpenPIBackend(policy, num_steps=num_steps)


def native_raw(record):
    from preparations.idea1_v2_1.runtime_models import native_observation
    images, state = native_observation(record["obs"])
    return {"observation/image": np.asarray(images[0]), "observation/wrist_image": np.asarray(images[1]),
            "observation/state": state, "prompt": record["instruction"]}


def event():
    return torch.cuda.Event(enable_timing=True)


def e0(backend, prepared, seed=0, atol=1e-5, rtol=1e-5):
    noise = backend.noise(seed)
    prefix = backend.build_prefix(prepared)
    before = [x.clone() for x in cache_tensors(prefix.cache)]
    original_denoise = backend.model.denoise_step
    original_times = []
    original_velocities = []
    def audited_denoise(*args, **kwargs):
        original_times.append(args[-1].clone())
        value = original_denoise(*args, **kwargs)
        if len(original_times) == 1:
            original_velocities.append(value.clone())
        return value
    backend.model.denoise_step = audited_denoise
    try:
        reference = backend.original(prepared, noise)
    finally:
        backend.model.denoise_step = original_denoise
    repeat = backend.original(prepared, noise)
    output = backend.generate(prefix, prepared, noise)
    t = torch.ones(noise.shape[0], device=backend.device)
    v1 = backend.velocity(prefix, prepared, noise, t)
    p2 = backend.build_prefix(prepared)
    v2 = backend.velocity(p2, prepared, noise, t)
    # Exactly two full snapshots during E0. Original temporaries are gone.
    checks = dict(normalized=error(reference.cpu(), output.cpu()),
                  original_repeat=error(reference.cpu(), repeat.cpu()),
                  decoded=error(backend.output(prepared, reference), backend.output(prepared, output)),
                  velocity=error(original_velocities[0].cpu(), v1.cpu()),
                  rebuilt_prefix_velocity=error(v1.cpu(), v2.cpu()),
                  prefix_masks_equal=bool(torch.equal(prefix.pad_masks, p2.pad_masks)),
                  positions_equal=bool(torch.equal(prefix.position_ids, p2.position_ids)),
                  kv_unchanged=all(torch.equal(a, b) for a, b in zip(before, cache_tensors(prefix.cache), strict=True)),
                  original_denoise_calls=len(original_times),
                  original_time_nodes=[float(x[0].cpu()) for x in original_times],
                  kv_shapes=[list(x.shape) for x in cache_tensors(prefix.cache)], atol=atol, rtol=rtol)
    checks["passed"] = (bool(torch.allclose(reference, output, atol=atol, rtol=rtol))
                        and bool(torch.allclose(original_velocities[0], v1, atol=atol, rtol=rtol))
                        and bool(np.allclose(backend.output(prepared, reference), backend.output(prepared, output), atol=atol, rtol=rtol))
                        and checks["original_denoise_calls"] == backend.num_steps
                        and checks["kv_unchanged"] and checks["prefix_masks_equal"] and checks["positions_equal"])
    return checks


def pair(backend, prefix, current, next_observation, noise, concurrent=False):
    """Same two tasks in each arm; AE first in serial, background first in async.

    Enqueueing background first avoids host dispatch delaying its launch until
    after all ten AE steps. Both streams depend on the same start event.
    """
    foreground = torch.cuda.current_stream(backend.device)
    background = torch.cuda.Stream(device=backend.device) if concurrent else foreground
    start, ae_start, ae_done, pre_start, pre_done = [event() for _ in range(5)]
    start.record(foreground)
    background.wait_event(start)
    wall = time.perf_counter()
    if concurrent:
        with torch.cuda.stream(background):
            pre_start.record()
            fresh = backend.build_prefix(next_observation)
            pre_done.record()
    ae_start.record(foreground)
    actions = backend.generate(prefix, current, noise)
    ae_done.record(foreground)
    ae_done.synchronize()
    # Controller availability includes actual D2H and final output transform.
    decoded = backend.output(current, actions)
    ready_ms = (time.perf_counter() - wall) * 1000
    if not concurrent:
        # Decode before placing unrelated work on the foreground stream: a D2H
        # copy queued after prefill would otherwise report false AE latency.
        pre_start.record(foreground)
        fresh = backend.build_prefix(next_observation)
        pre_done.record(foreground)
    pre_done.synchronize()
    total_ms = (time.perf_counter() - wall) * 1000
    return decoded, dict(action_ready_host_ms=ready_ms, all_done_host_ms=total_ms,
                         ae_gpu_ms=ae_start.elapsed_time(ae_done), prefix_gpu_ms=pre_start.elapsed_time(pre_done),
                         ae_end_gpu_ms=start.elapsed_time(ae_done), prefix_end_gpu_ms=start.elapsed_time(pre_done)), fresh


def e1(backend, observations, warmup=5, samples=20, trace=None):
    prefix = backend.build_prefix(observations[0])
    torch.cuda.current_stream().synchronize()
    rows = []
    for index in range(warmup + samples):
        noise = backend.noise(index)
        torch.cuda.current_stream().synchronize()  # Input preparation outside timed pair.
        # Reset peaks only after both tasks of the previous arm finished.
        arm_rows = []
        arm_outputs = {}
        for concurrent in ((False, True) if index % 2 == 0 else (True, False)):
            torch.cuda.reset_peak_memory_stats(backend.device)
            decoded, timings, fresh = pair(backend, prefix, observations[0], observations[1], noise, concurrent)
            del fresh
            arm_outputs[concurrent] = decoded.copy()
            timings.update(mode="two_stream" if concurrent else "serial", iteration=index - warmup,
                           peak_allocated=torch.cuda.max_memory_allocated(backend.device),
                           peak_reserved=torch.cuda.max_memory_reserved(backend.device))
            arm_rows.append(timings)
        for timings in arm_rows:
            timings["output_error"] = error(arm_outputs[False], arm_outputs[timings["mode"] == "two_stream"])
        if index >= warmup:
            rows.extend(arm_rows)
    if trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            _, _, fresh = pair(backend, prefix, observations[0], observations[1], backend.noise(0), True)
            del fresh
        profiler.export_chrome_trace(str(trace))
        from .trace_analysis import summarize
        trace_summary = summarize(trace)
    else:
        trace_summary = None
    summary = {}
    for mode in ("serial", "two_stream"):
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {key: dict(zip(("p50", "p95", "p99"), np.percentile([r[key] for r in selected], [50, 95, 99]).tolist()))
                         for key in ("action_ready_host_ms", "all_done_host_ms", "ae_gpu_ms", "prefix_gpu_ms")}
    return dict(rows=rows, summary=summary, trace_summary=trace_summary)


def e2(backend, observations, intervals=(1, 2, 4)):
    rows = []
    # Keep exactly two caches: current fresh and selected stale, never precompute
    # a replay sequence of caches. Same noise and current suffix/reference state.
    for interval in intervals:
        stale = None
        source = None
        for index, current in enumerate(observations):
            noise = backend.noise(index)
            if index % interval == 0:
                stale = None
                stale = backend.build_prefix(current)
                source = index
                fresh = stale
            else:
                fresh = backend.build_prefix(current)
            reference = backend.generate(fresh, current, noise)
            selected = backend.generate(stale, current, noise)
            fresh = None
            decoded_ref = backend.output(current, reference)
            decoded = backend.output(current, selected)
            # Fixed-version paired two-task replay; next prefill uses the next
            # recorded input but cannot change this request's selected version.
            torch.cuda.current_stream().synchronize()
            serial, _, temporary = pair(backend, stale, current, observations[(index + 1) % len(observations)], noise, False)
            del temporary
            parallel, _, temporary = pair(backend, stale, current, observations[(index + 1) % len(observations)], noise, True)
            del temporary
            components = {}
            for label, selection in (("first5", slice(0, 5)), ("full50", slice(None))):
                a, b = decoded_ref[selection], decoded[selection]
                components[label] = dict(position=error(a[..., :3], b[..., :3]),
                                         rotation=error(a[..., 3:6], b[..., 3:6]),
                                         gripper=error(a[..., 6:], b[..., 6:]),
                                         gripper_sign_mismatches=int(np.count_nonzero((a[..., 6] < 0) != (b[..., 6] < 0))))
            rows.append(dict(interval=interval, request=index, prefix_source=source,
                             observation_request_age=index-source, noise_seed=index,
                             suffix_state_source=index, reference_state_source=index,
                             normalized_error=error(reference.cpu(), selected.cpu()), decoded_error=error(decoded_ref, decoded),
                             position_error=error(decoded_ref[..., :3], decoded[..., :3]),
                             rotation_error=error(decoded_ref[..., 3:6], decoded[..., 3:6]),
                             gripper_error=error(decoded_ref[..., 6:], decoded[..., 6:]),
                             by_horizon=components, serial_async_error=error(serial, parallel)))
    return dict(rows=rows, age_unit="generation requests; source recording has no synchronized camera timestamps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True, help="Trusted local real_predict_inputs_first25.pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", choices=("e0", "e1", "e2", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1e-5, help="Frozen before E0, never automatically relaxed")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    backend = load_backend(args.checkpoint, args.device)
    raw = Path(args.bank).read_bytes()
    bank = pickle.loads(raw)
    observations = [backend.prepare(native_raw(row)) for row in bank[:args.observations]]
    if len(observations) < 2:
        raise ValueError("Need at least two observations")
    identity = dict(backend.identity, checkpoint=str(Path(args.checkpoint).resolve()), bank=str(Path(args.bank).resolve()),
                    bank_sha256=hashlib.sha256(raw).hexdigest(), args=vars(args),
                    openpi_commit=subprocess.check_output(["git", "-C", "policies/openpi", "rev-parse", "HEAD"], text=True).strip())
    weight_hash = hashlib.sha256()
    with (Path(args.checkpoint) / "model.safetensors").open("rb") as weights:
        for block in iter(lambda: weights.read(8 * 1024 * 1024), b""):
            weight_hash.update(block)
    identity["weights_sha256"] = weight_hash.hexdigest()
    identity["implementation_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in Path(__file__).parent.glob("*.py")}
    (output / "identity.json").write_text(json.dumps(identity, indent=2))
    for name in ("e0", "e1", "e2"):
        if args.experiment not in (name, "all"):
            continue
        result = e0(backend, observations[0], atol=args.atol) if name == "e0" else (
            e1(backend, observations, args.warmup, args.samples, output / "trace.json" if args.trace else None)
            if name == "e1" else e2(backend, observations))
        (output / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(experiment=name, output=str(output / f"{name}.json"), passed=result.get("passed"))), flush=True)
        if name == "e0" and not result["passed"]:
            raise RuntimeError("E0 failed; stop before concurrency")


if __name__ == "__main__":
    main()
