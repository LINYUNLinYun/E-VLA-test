"""Optional E1b: full AE/prefill CUDA Graph dispatch follow-up, never E3 policy.

Each module has a distinct private graph pool. Fixed addresses are intentional:
AE always reads the same eager old prefix, and prefill rewrites its own captured
cache, which no AE reader consumes. There are exactly two full KV snapshots.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import torch

from .experiments import event, load_backend, native_raw, pair as eager_pair
from .openpi_backend import cache_tensors, error


def address_signature(prefix):
    return [dict(address=x.data_ptr(), shape=list(x.shape), dtype=str(x.dtype))
            for x in cache_tensors(prefix.cache)]


def cache_digest(prefix):
    digest = hashlib.sha256()
    for tensor in cache_tensors(prefix.cache):
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def summarize_rows(rows):
    return {mode:{key:dict(zip(("p50","p95","p99"),np.percentile(
                [row[key] for row in rows if row["mode"]==mode],[50,95,99]).tolist()))
                  for key in ("action_ready_host_ms","all_done_host_ms","ae_gpu_ms","prefix_gpu_ms")}
            for mode in ("serial","two_stream")}


def eager_reference(backend,current,next_observation,warmup,samples):
    """Same-GPU reference before any captured cache exists; fixed noise seed0."""
    prefix = backend.build_prefix(current)
    noise = backend.noise(0)
    torch.cuda.current_stream(backend.device).synchronize()
    rows = []
    output_hash = None
    try:
        for index in range(warmup+samples):
            outputs,iteration = {},[]
            for concurrent in ((False,True) if index%2==0 else (True,False)):
                torch.cuda.reset_peak_memory_stats(backend.device)
                decoded,row,fresh = eager_pair(backend,prefix,current,next_observation,noise,concurrent)
                # pair waits for both readers and producer; release the second
                # owned cache before the next arm can allocate another one.
                del fresh
                outputs[concurrent] = decoded.copy()
                row.update(mode="two_stream" if concurrent else "serial",iteration=index-warmup,
                           peak_allocated=torch.cuda.max_memory_allocated(backend.device),
                           peak_reserved=torch.cuda.max_memory_reserved(backend.device))
                iteration.append(row)
            difference = error(outputs[False],outputs[True])
            if difference["max_abs"]>1e-5:
                raise RuntimeError("Same fixed-input eager serial/async mismatch")
            signature = hashlib.sha256(np.ascontiguousarray(outputs[False]).tobytes()).hexdigest()
            if output_hash is not None and signature!=output_hash:
                raise RuntimeError("Fixed-noise eager reference changed between iterations")
            output_hash = signature
            for row in iteration:
                row["paired_output_error"] = difference
            if index>=warmup:
                rows.extend(iteration)
            del decoded,outputs,iteration
    finally:
        # No eager reference KV/noise/output remains when graph capture begins.
        del prefix,noise
        gc.collect()
    return dict(rows=rows,summary=summarize_rows(rows),fixed_noise_seed=0,fixed_output_sha256=output_hash,
                scope="Same loaded model, GPU, current/next observation and seed0 noise as the following graphs; all tasks complete per arm")


class GraphPair:
    def __init__(self, backend, current, next_observation, noise, capture_warmup=3):
        if capture_warmup < 1:
            raise ValueError("Capture needs at least one full warmup")
        self.backend, self.current, self.next = backend, current, next_observation
        self.noise = noise
        self.ae_stream = torch.cuda.Stream(device=backend.device)
        self.prefix_stream = torch.cuda.Stream(device=backend.device)
        self.prefix = backend.build_prefix(current)
        # Owned references keep every graph input address live throughout replay.
        torch.cuda.current_stream(backend.device).synchronize()
        self.old_signature = address_signature(self.prefix)
        self.old_digest = cache_digest(self.prefix)
        for _ in range(capture_warmup):
            with torch.cuda.stream(self.ae_stream):
                temporary_action = backend.generate(self.prefix, current, noise)
            self.ae_stream.synchronize()
            del temporary_action
            with torch.cuda.stream(self.prefix_stream):
                temporary_prefix = backend.build_prefix(next_observation)
            self.prefix_stream.synchronize()
            self.expected_fresh_digest = cache_digest(temporary_prefix)
            del temporary_prefix
        gc.collect()
        # No warmup snapshot survives into either capture. Default pool=None
        # gives each independently captured graph a distinct private pool.
        self.ae_graph, self.prefix_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.ae_graph, stream=self.ae_stream):
            self.action_buffer = backend.generate(self.prefix, current, noise)
        with torch.cuda.graph(self.prefix_graph, stream=self.prefix_stream):
            self.fresh = backend.build_prefix(next_observation)
        # Explicit first replay initializes captured output buffers before any
        # digest/parity inspection; capture alone is not an output-ready lease.
        with torch.cuda.stream(self.ae_stream):
            self.ae_graph.replay()
        with torch.cuda.stream(self.prefix_stream):
            self.prefix_graph.replay()
        self.ae_stream.synchronize()
        self.prefix_stream.synchronize()
        self.fresh_signature = address_signature(self.fresh)
        if cache_digest(self.fresh) != self.expected_fresh_digest:
            raise RuntimeError("Captured full prefill differs from eager full prefill")
        if self.ae_graph.pool() == self.prefix_graph.pool():
            raise RuntimeError("Independent captures unexpectedly share a graph pool")
        # Eager baseline creates no third cache: it reads the retained old KV.
        eager = backend.generate(self.prefix, current, noise)
        torch.cuda.current_stream(backend.device).synchronize()
        self.eager_actions = backend.output(current, eager)
        self.captured_error = error(self.eager_actions, backend.output(current, self.action_buffer))
        if not np.allclose(self.eager_actions, backend.output(current, self.action_buffer), atol=1e-5, rtol=1e-5):
            raise RuntimeError("CUDA Graph AE differs from eager full AE")

    def pair(self, concurrent):
        start, ae_start, ae_done, prefix_start, prefix_done = [event() for _ in range(5)]
        start.record(self.ae_stream)
        self.prefix_stream.wait_event(start)
        begin = time.perf_counter()
        if concurrent:
            with torch.cuda.stream(self.prefix_stream):
                prefix_start.record()
                self.prefix_graph.replay()
                prefix_done.record()
        with torch.cuda.stream(self.ae_stream):
            ae_start.record()
            self.ae_graph.replay()
            # Every returned result owns its address, so later graph replay
            # cannot overwrite a controller result retained by the caller.
            owned = self.action_buffer.clone()
            ae_done.record()
        ae_done.synchronize()
        # D2H is issued on the same AE stream after its completed owned copy.
        with torch.cuda.stream(self.ae_stream):
            decoded = self.backend.output(self.current, owned)
        action_ready = (time.perf_counter()-begin)*1000
        if not concurrent:
            with torch.cuda.stream(self.prefix_stream):
                prefix_start.record()
                self.prefix_graph.replay()
                prefix_done.record()
        prefix_done.synchronize()
        all_done = (time.perf_counter()-begin)*1000
        row = dict(mode="two_stream" if concurrent else "serial", action_ready_host_ms=action_ready,
                   all_done_host_ms=all_done, ae_gpu_ms=ae_start.elapsed_time(ae_done),
                   prefix_gpu_ms=prefix_start.elapsed_time(prefix_done),
                   ae_end_gpu_ms=start.elapsed_time(ae_done), prefix_end_gpu_ms=start.elapsed_time(prefix_done),
                   peak_allocated=torch.cuda.max_memory_allocated(self.backend.device),
                   peak_reserved=torch.cuda.max_memory_reserved(self.backend.device))
        return decoded, owned, row

    def audit(self):
        if address_signature(self.prefix) != self.old_signature or address_signature(self.fresh) != self.fresh_signature:
            raise RuntimeError("Captured KV addresses or layouts changed")
        if cache_digest(self.prefix) != self.old_digest:
            raise RuntimeError("Graph replay mutated the read-only old KV")
        if cache_digest(self.fresh) != self.expected_fresh_digest:
            raise RuntimeError("Graph replay prefill differs from eager full prefill")
        first, retained, _ = self.pair(False)
        reference = retained.clone()
        _, second, _ = self.pair(True)
        same_storage = retained.data_ptr() == second.data_ptr()
        stable = bool(torch.equal(retained, reference))
        if same_storage or not stable:
            raise RuntimeError("Returned graph output does not have stable owned storage")
        if not np.allclose(first, self.eager_actions, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Graph replay changed eager output")
        return dict(stable_output=stable, distinct_output_addresses=not same_storage,
                    old_kv_addresses_stable=True, captured_kv_addresses_stable=True,
                    graph_pools_distinct=True, logical_snapshots=2, eager_error=self.captured_error,
                    old_kv_content_unchanged=True, captured_prefill_equals_eager=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--eager-reference",action="store_true",help="Run same-GPU fixed-input eager20/100 pairs before graph capture")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    backend = load_backend(args.checkpoint)
    raw = Path(args.bank).read_bytes()
    bank = pickle.loads(raw)
    current, next_observation = [backend.prepare(native_raw(row)) for row in bank[:2]]
    original_eager = eager_reference(backend,current,next_observation,args.warmup,args.samples) if args.eager_reference else None
    if original_eager is not None:
        (destination/"original_eager_reference.json").write_text(json.dumps(original_eager,indent=2))
    from .graph_sampler import GraphSafeBackend,validate
    original_backend = backend
    backend = GraphSafeBackend(original_backend,current,next_observation)
    validation = validate(original_backend,backend,current,next_observation,backend.noise(0))
    (destination/"graph_safe_e0.json").write_text(json.dumps(validation,indent=2))
    if not validation["passed"]:
        raise RuntimeError("Hoisted-metadata wrapper differs from the original full backend")
    gc.collect()
    eager = eager_reference(backend,current,next_observation,args.warmup,args.samples) if args.eager_reference else None
    if eager is not None:
        (destination/"eager_reference.json").write_text(json.dumps(eager,indent=2))
    setup_start = time.perf_counter()
    try:
        pair = GraphPair(backend, current, next_observation, backend.noise(0))
    except Exception:
        failure = dict(stage="cold_graph_setup",error=traceback.format_exc(),args=vars(args),
                       setup_wall_ms=(time.perf_counter()-setup_start)*1000,
                       identity=backend.identity,
                       gpu_uuid=str(getattr(torch.cuda.get_device_properties(backend.device),"uuid",None)),
                       eager_reference_artifact="eager_reference.json" if eager is not None else None,
                       existing_trace_preserved=(destination/"trace.json").exists())
        (destination/"capture_failure.json").write_text(json.dumps(failure,indent=2))
        gc.collect()
        raise
    setup_wall_ms = (time.perf_counter()-setup_start)*1000
    if eager is not None and hashlib.sha256(np.ascontiguousarray(pair.eager_actions).tobytes()).hexdigest()!=eager["fixed_output_sha256"]:
        raise RuntimeError("Fixed eager reference output changed across graph setup")
    audit = pair.audit()
    rows = []
    for index in range(args.warmup+args.samples):
        outputs = {}
        iteration = []
        for concurrent in ((False, True) if index % 2 == 0 else (True, False)):
            torch.cuda.reset_peak_memory_stats(backend.device)
            decoded, owned, row = pair.pair(concurrent)
            outputs[concurrent] = decoded.copy()
            del owned
            row["iteration"] = index-args.warmup
            iteration.append(row)
        difference = error(outputs[False], outputs[True])
        if not np.allclose(outputs[False], outputs[True], atol=1e-5, rtol=1e-5):
            raise RuntimeError("Same-input graph serial/async mismatch")
        for row in iteration:
            row["paired_output_error"] = difference
        if index >= args.warmup:
            rows.extend(iteration)
    summary = summarize_rows(rows)
    trace_summary = None
    if args.trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            _, owned, _ = pair.pair(True)
            del owned
        path = destination/"trace.json"
        profiler.export_chrome_trace(str(path))
        from .trace_analysis import summarize
        trace_summary = summarize(path)
    audit = pair.audit()  # Recheck immutable old KV and stable graph outputs after all measured replays.
    identity = dict(backend.identity,
                    gpu_uuid=str(getattr(torch.cuda.get_device_properties(backend.device), "uuid", None)),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    graph_execution=True, ae_private_pool=list(pair.ae_graph.pool()),
                    prefix_private_pool=list(pair.prefix_graph.pool()),
                    graph_sampler_source=str(Path(__file__).with_name("graph_sampler.py").resolve()),
                    graph_sampler_sha256=hashlib.sha256(Path(__file__).with_name("graph_sampler.py").read_bytes()).hexdigest())
    report = dict(identity=identity, checkpoint=args.checkpoint, bank=args.bank,
                  bank_sha256=hashlib.sha256(raw).hexdigest(), implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  graph_sampler_sha256=hashlib.sha256(Path(__file__).with_name("graph_sampler.py").read_bytes()).hexdigest(),
                  args=vars(args), audit=audit, rows=rows, summary=summary, trace_summary=trace_summary,
                  eager_rows=eager["rows"] if eager is not None else None,
                  eager_summary=eager["summary"] if eager is not None else None,
                  eager_reference_kind="graph-safe static-metadata wrapper, same NN and original CUDA N10 recurrence",
                  original_eager_rows=original_eager["rows"] if original_eager is not None else None,
                  original_eager_summary=original_eager["summary"] if original_eager is not None else None,
                  graph_safe_e0=validation,
                  eager_fixed_output_sha256=eager["fixed_output_sha256"] if eager is not None else None,
                  graph_setup_wall_ms=setup_wall_ms,
                  graph_over_eager_speedups={mode:{key:eager["summary"][mode][key]["p50"]/summary[mode][key]["p50"]
                                                   for key in ("action_ready_host_ms","all_done_host_ms")}
                                            for mode in ("serial","two_stream")} if eager is not None else None,
                  scope="Six same-GPU arms: original eager serial/parallel, static-metadata eager serial/parallel, same static wrapper graph serial/parallel; graph speedup uses the matched static eager reference, not the original eager path; no E3 modification")
    (destination/"e1b.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(audit=audit, summary=summary, trace_summary=trace_summary)), flush=True)


if __name__ == "__main__":
    main()
