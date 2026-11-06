"""Fixed-input GPU concurrency diagnosis; does not change the runtime policy.

Reuse the validated GraphSafeBackend and independent GraphPair pools. Only the
replay order/stream priority changes. AE reads the fixed old KV and prefill owns
the other cache; each iteration drains both events before any replay.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .experiments import event, load_backend, native_raw
from .graph_bench import GraphPair
from .graph_sampler import GraphSafeBackend, validate
from .openpi_backend import error


@contextmanager
def span(label):
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def linear_shapes(backend, current, nxt, noise):
    rows, hooks = [], []
    def hook(name):
        def record(module, inputs):
            x = inputs[0]
            m = x.numel() // x.shape[-1]
            n, k = module.weight.shape
            byte_count = (m*k+k*n+m*n)*module.weight.element_size()
            rows.append(dict(name=name, input_shape=list(x.shape), weight_shape=list(module.weight.shape),
                             weight_dtype=str(module.weight.dtype), m=m,n=n,k=k,
                             flops=2*m*n*k, ideal_operand_bytes=byte_count,
                             ideal_cold_operand_flops_per_byte=2*m*n*k/byte_count))
        return record
    old = backend.build_prefix(current)
    for name, module in backend.model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_pre_hook(hook(name)))
    try:
        actions = backend.generate(old, current, noise)
        fresh = backend.build_prefix(nxt)
        torch.cuda.current_stream().synchronize()
    finally:
        for handle in hooks:
            handle.remove()
    del old, fresh, actions
    return rows


def run_pair(pair, mode, priority_stream):
    ae_stream = priority_stream if mode.endswith("_high_ae") else pair.ae_stream
    pre_stream = pair.prefix_stream
    ae_begin, ae_end, pre_begin, pre_end = [event() for _ in range(4)]
    epoch = event()
    epoch.record()
    ae_stream.wait_event(epoch)
    pre_stream.wait_event(epoch)
    jobs = {}
    begin = time.perf_counter()
    def enqueue(name):
        stream, graph, start, done = ((ae_stream,pair.ae_graph,ae_begin,ae_end) if name=="ae"
                                      else (pre_stream,pair.prefix_graph,pre_begin,pre_end))
        with torch.cuda.stream(stream), span(name):
            start.record()
            cpu_start = time.perf_counter()
            graph.replay()
            cpu_end = time.perf_counter()
            done.record()
        jobs[name] = dict(cpu_launch_ms=(cpu_end-cpu_start)*1000, done_after_launch=done.query())
    order = ("prefill","ae") if "prefill_first" in mode else ("ae","prefill")
    if mode.endswith("_only"):
        order = (mode.removesuffix("_only"),)
    for name in order:
        enqueue(name)
        if mode.startswith("serial_"):
            (ae_end if name=="ae" else pre_end).synchronize()
    if "ae" in jobs:
        ae_end.synchronize()
        with torch.cuda.stream(ae_stream):
            owned = pair.action_buffer.clone()
            decoded = pair.backend.output(pair.current, owned)
        action_ready = (time.perf_counter()-begin)*1000
        difference = error(pair.eager_actions, decoded)
        assert difference["max_abs"] <= 1e-5
    else:
        action_ready, difference = None, None
    if "prefill" in jobs:
        pre_end.synchronize()
    total = (time.perf_counter()-begin)*1000
    return dict(mode=mode, all_done_host_ms=total, action_ready_host_ms=action_ready, jobs=jobs,
                ae_gpu_ms=ae_begin.elapsed_time(ae_end) if "ae" in jobs else None,
                prefill_gpu_ms=pre_begin.elapsed_time(pre_end) if "prefill" in jobs else None,
                ae_start_ms=epoch.elapsed_time(ae_begin) if "ae" in jobs else None,
                prefill_start_ms=epoch.elapsed_time(pre_begin) if "prefill" in jobs else None,
                ae_end_ms=epoch.elapsed_time(ae_end) if "ae" in jobs else None,
                prefill_end_ms=epoch.elapsed_time(pre_end) if "prefill" in jobs else None,
                output_error=difference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi0_libero_pytorch_idea2")
    parser.add_argument("--bank", default="policies/realtime-vla-flash/results/idea1_v3_supplement/e1_expand_init7_10_20261020T0431Z/episodes/E1/supp_E1_D2_tau0.1_P2_Gconfirmed_disagreement_rejectD2/A/libero_spatial/task00/init07/real_predict_inputs_first25.pkl")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    print("Loading original full model", flush=True)
    backend = load_backend(args.checkpoint)
    payload = Path(args.bank).read_bytes()
    records = pickle.loads(payload)
    current, nxt = [backend.prepare(native_raw(r)) for r in records[:2]]
    noise = backend.noise(1701)
    shapes = linear_shapes(backend,current,nxt,noise)
    (output/"linear_shapes.json").write_text(json.dumps(shapes,indent=2))
    safe = GraphSafeBackend(backend,current,nxt)
    validation = validate(backend,safe,current,nxt,noise)
    assert validation["passed"]
    gc.collect()
    pair = GraphPair(safe,current,nxt,noise)
    high = torch.cuda.Stream(device=backend.device, priority=-1)
    modes = ("ae_only","prefill_only","serial_ae_first","serial_prefill_first",
             "concurrent_ae_first","concurrent_prefill_first",
             "concurrent_ae_first_high_ae","concurrent_prefill_first_high_ae")
    print("Captures validated; warming eight modes",flush=True)
    for _ in range(args.warmup):
        for mode in modes:
            run_pair(pair,mode,high)
    rows = []
    if args.profile:
        torch.cuda.profiler.start()
    try:
        for i in range(args.samples):
            for mode in modes[i%len(modes):]+modes[:i%len(modes)]:
                with span(f"pair/{mode}/{i}"):
                    row = run_pair(pair,mode,high)
                row["iteration"] = i
                rows.append(row)
    finally:
        if args.profile:
            torch.cuda.profiler.stop()
    audit = pair.audit()
    summary = {}
    for mode in modes:
        selected = [r for r in rows if r["mode"]==mode]
        summary[mode] = {key:dict(zip(("p50","p95"),np.percentile([r[key] for r in selected],[50,95]).tolist()))
                         for key in ("all_done_host_ms","action_ready_host_ms","ae_gpu_ms","prefill_gpu_ms")
                         if selected[0][key] is not None}
    props = torch.cuda.get_device_properties(backend.device)
    report = dict(args=vars(args),identity=backend.identity,gpu_uuid=str(props.uuid),sm_count=props.multi_processor_count,
                  validation=validation,audit=audit,rows=rows,summary=summary,
                  bank_sha256=hashlib.sha256(payload).hexdigest(),
                  implementation_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                         for name in ("graph_overlap_diagnosis.py","graph_bench.py","graph_sampler.py","openpi_backend.py")},
                  scope="Fixed-input diagnostic only; old/new KV disjoint; no runtime Policy or model algorithm changes")
    (output/"result.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
