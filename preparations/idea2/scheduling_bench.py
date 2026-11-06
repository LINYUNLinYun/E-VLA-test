"""Same-card scheduling comparison, plus a separate bounded nsys capture.

Original pair() is retained as the unmodified reference. Fixed KV microbenchmark
and real PipelinePolicy replay are reported separately.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import functools
import importlib.util
import os
import subprocess
import threading
import time

import numpy as np
import torch

from .experiments import event, load_backend, native_raw, pair
from .graph_bench import cache_digest
from .openpi_backend import cache_tensors, error
from .submission import HostWorkers, submit


@contextmanager
def span(label):
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def scheduled_pair(backend, prefix, current, next_observation, noise, order, streams, workers=None):
    ae_stream, pre_stream = streams
    ready = event()
    ready.record()
    begin = time.perf_counter()
    jobs = {}
    values = {}
    futures = {}
    for name in (("ae", "prefill") if order == "ae_first" else ("prefill", "ae")):
        stream, fn, args = ((ae_stream, backend.generate, (prefix, current, noise)) if name == "ae"
                            else (pre_stream, backend.build_prefix, (next_observation,)))
        if workers is None:
            values[name], jobs[name] = submit(stream, name, fn, *args, dependency=ready)
        else:
            futures[name] = workers.enqueue(name, stream, fn, *args, dependency=ready)
    if workers is not None:
        for name, (value, job) in workers.collect(futures, streams).items():
            values[name], jobs[name] = value, job
    after_submit = {name: job.done.query() for name, job in jobs.items()}
    jobs["ae"].done.synchronize()
    # Output copies must use AE's stream; unrelated default-stream work must not
    # delay the measured action availability.
    with torch.cuda.stream(ae_stream):
        decoded = backend.output(current, values["ae"])
    action_ms = (time.perf_counter()-begin)*1000
    jobs["prefill"].done.synchronize()
    total_ms = (time.perf_counter()-begin)*1000
    return decoded, dict(action_ready_host_ms=action_ms, all_done_host_ms=total_ms,
                        ae_gpu_ms=jobs["ae"].start.elapsed_time(jobs["ae"].done),
                        prefix_gpu_ms=jobs["prefill"].start.elapsed_time(jobs["prefill"].done),
                        jobs={name:job.metrics() for name,job in jobs.items()},
                        done_after_both_submissions=after_submit), values["prefill"]


def summary(rows):
    return {mode: {key: dict(zip(("p50", "p95"), np.percentile(
                [r[key] for r in rows if r["mode"] == mode], [50, 95]).tolist()))
            for key in ("action_ready_host_ms", "all_done_host_ms", "ae_gpu_ms", "prefix_gpu_ms")}
            for mode in dict.fromkeys(r["mode"] for r in rows)}


def runtime_replay(backend, records, args, submissions, raw_outputs, capturing):
    from .policy import PipelinePolicy
    spec = importlib.util.spec_from_file_location("preparations.idea2._original_policy", args.original_policy)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    modes = ("original", "ae_first", "prefill_first")
    if args.threads:
        modes += ("threaded_ae_first", "threaded_prefill_first")
    policies = {mode:(original.PipelinePolicy(backend) if mode == "original" else
                      PipelinePolicy(backend, submission_order=mode.removeprefix("threaded_"),
                                     host_workers=mode.startswith("threaded_"), diagnostics=True))
                for mode in modes}
    rows = []
    try:
        for policy in policies.values():
            for i in range(args.warmup):
                r = records[i % len(records)]
                policy.predict(r["obs"], r["instruction"], "libero_spatial", 1701+i)
            policy.reset()
        if capturing:
            torch.cuda.profiler.start()
        try:
            # The old-KV protocol remains exactly the real policy protocol.
            for mode, policy in policies.items():
                for i in range(args.runtime_requests):
                    r = records[i % len(records)]
                    submissions.clear()
                    raw_outputs.clear()
                    with span(f"runtime/{mode}/{i}"):
                        response = policy.predict(r["obs"], r["instruction"], "libero_spatial", 1701+i)
                    rows.append(dict(mode=mode, response=response,
                                     cpu_submissions=list(submissions), raw=raw_outputs[-1]))
                # Drains only owned events, after all measured requests.
                policy.reset()
        finally:
            if capturing:
                torch.cuda.profiler.stop()
        # Validate each response against its actual old-KV source after capture.
        # No extra complete KV snapshot exists alongside a live policy pool.
        for row in rows:
            response = row["response"]
            i, source = response["request"], response["prefix_observation_id"]
            assert 0 <= source <= i
            assert i == response["suffix_state_id"] == response["reference_state_id"]
            if i > 0:
                assert source < i, "Steady-state AE must use an old prefix"
            selected = backend.prepare(native_raw(records[source % len(records)]))
            current = backend.prepare(native_raw(records[i % len(records)]))
            prefix = backend.build_prefix(selected)
            before = cache_digest(prefix)
            noise = backend.noise(response["seed"])
            reference = backend.generate(prefix, current, noise)
            row["normalized_same_source_error"] = error(reference.cpu(), row.pop("raw").cpu())
            row["same_source_output_error"] = error(backend.output(current, reference), response.pop("actions"))
            assert row["same_source_output_error"]["max_abs"] <= 1e-5
            assert row["normalized_same_source_error"]["max_abs"] <= 1e-5
            assert cache_digest(prefix) == before
            assert response["pool"]["live_snapshots"] <= 2
            del prefix, reference, noise
    finally:
        for policy in policies.values():
            if hasattr(policy, "close"):
                policy.close()
            else:
                policy.reset()
    return dict(rows=rows, summary={mode:{key:dict(zip(("p50","p95"),np.percentile(
                    [r["response"]["timing"][key] for r in rows if r["mode"]==mode and r["response"]["request"]>0], [50,95]).tolist()))
                for key in ("action_ready_host_ms", "response_host_ms", "ae_gpu_ms")}
                for mode in modes}, scope="Real observations replay; includes preparation/D2H/decode, excludes RPC/robot motion. Request zero excluded from steady-state summary.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi0_libero_pytorch_idea2")
    parser.add_argument("--bank", default="policies/realtime-vla-flash/results/idea1_v3_supplement/e1_expand_init7_10_20261020T0431Z/episodes/E1/supp_E1_D2_tau0.1_P2_Gconfirmed_disagreement_rejectD2/A/libero_spatial/task00/init07/real_predict_inputs_first25.pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--threads", action="store_true")
    parser.add_argument("--runtime-requests", type=int, default=0)
    parser.add_argument("--original-policy", default="results/idea2/scheduling_fix_20261021/baseline/policy.py")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = {p.name:p.read_bytes() for p in Path(__file__).parent.glob("*.py")}
    (output/"source").mkdir()
    for name, payload in sources.items():
        (output/"source"/name).write_bytes(payload)
    source_hashes = {name:hashlib.sha256(payload).hexdigest() for name,payload in sources.items()}
    gpu_identity = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    backend = load_backend(args.checkpoint)
    records = pickle.loads(Path(args.bank).read_bytes())
    current, next_observation = [backend.prepare(native_raw(r)) for r in records[:2]]
    noise = backend.noise(1701)
    prefix = backend.build_prefix(current)
    torch.cuda.current_stream().synchronize()
    old_hash = cache_digest(prefix)
    reference_raw = backend.generate(prefix, current, noise)
    reference = backend.output(current, reference_raw)
    reference_normalized = reference_raw.cpu().numpy()
    native = backend.original(current, noise)
    native_error = error(native.cpu(), reference_raw.cpu())
    assert native_error["max_abs"] <= 1e-5
    del reference_raw, native
    streams = (torch.cuda.current_stream(), torch.cuda.Stream())
    modes = ("serial", "original", "ae_first", "prefill_first")
    workers = HostWorkers() if args.threads else None
    if workers is not None:
        modes += ("threaded_ae_first", "threaded_prefill_first")
    rows = []
    submissions = []
    raw_outputs = []
    # CPU-only observation around the unchanged original pair, as well as the
    # candidates. NVTX is thread-local and records actual module submissions.
    for method, label in (("build_prefix", "prefill"), ("generate", "ae")):
        original = getattr(backend, method)
        def wrap(fn, label):
            @functools.wraps(fn)
            def call(*values, **kwargs):
                begin = time.perf_counter()
                with span(label):
                    result = fn(*values, **kwargs)
                submissions.append(dict(module=label, cpu_submit_ms=(time.perf_counter()-begin)*1000,
                                        thread=threading.get_ident()))
                if label == "ae":
                    raw_outputs.append(result)
                return result
            return call
        setattr(backend, method, wrap(original, label))
    def run(mode, iteration):
        submissions.clear()
        raw_outputs.clear()
        with span(f"pair/{mode}/{iteration}"):
            if mode in ("serial", "original"):
                decoded, row, fresh = pair(backend, prefix, current, next_observation, noise, mode == "original")
            else:
                threaded = mode.startswith("threaded_")
                order = mode.removeprefix("threaded_")
                decoded, row, fresh = scheduled_pair(backend, prefix, current, next_observation, noise, order, streams,
                                                     workers if threaded else None)
        row.update(mode=mode, iteration=iteration, output_error=error(reference, decoded))
        row["cpu_submissions"] = list(submissions)
        row["normalized_error"] = error(reference_normalized, raw_outputs[-1].cpu())
        assert row["output_error"]["max_abs"] <= 1e-5
        assert row["normalized_error"]["max_abs"] <= 1e-5
        del fresh
        return row
    for i in range(args.warmup):
        for mode in modes:
            run(mode, -1)
    if args.profile:
        torch.cuda.profiler.start()
    try:
        for i in range(args.samples):
            for mode in modes[i % len(modes):]+modes[:i % len(modes)]:
                rows.append(run(mode, i))
    finally:
        if args.profile:
            torch.cuda.profiler.stop()
    expected_fresh = backend.build_prefix(next_observation)
    fresh_hash = cache_digest(expected_fresh)
    del expected_fresh
    fresh_checks = {}
    try:
        for mode in modes:
            if mode in ("serial", "original"):
                _, _, fresh = pair(backend, prefix, current, next_observation, noise, mode == "original")
            else:
                _, _, fresh = scheduled_pair(backend, prefix, current, next_observation, noise,
                                             mode.removeprefix("threaded_"), streams,
                                             workers if mode.startswith("threaded_") else None)
            fresh_checks[mode] = dict(kv_equal=cache_digest(fresh)==fresh_hash,
                                     old_new_storage_disjoint=not bool(
                                         {t.data_ptr() for t in cache_tensors(prefix.cache)} &
                                         {t.data_ptr() for t in cache_tensors(fresh.cache)}))
            assert all(fresh_checks[mode].values())
            del fresh
    finally:
        if workers is not None:
            workers.close()
    result = dict(identity=backend.identity, args=vars(args),
                  bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
                  implementation_sha256=source_hashes, gpu_inventory=gpu_identity,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  native_error=native_error, old_kv_unchanged=cache_digest(prefix)==old_hash,
                  rows=rows, summary=summary(rows), fresh_kv_checks=fresh_checks)
    assert result["old_kv_unchanged"]
    (output/"pair.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)
    del prefix, noise
    if args.runtime_requests:
        result["runtime"] = runtime_replay(backend, records, args, submissions, raw_outputs, args.profile)
    assert source_hashes == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}, "Source changed during benchmark"
    (output/"result.json").write_text(json.dumps(result, indent=2))
    if args.runtime_requests:
        print(json.dumps(result["runtime"]["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
