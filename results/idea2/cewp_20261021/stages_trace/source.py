"""Real-input, private-Graph phase pairing screen; not an end-to-end benchmark.

Attention includes QKV, RoPE, attention and output projection. AE attention reads
only the selected original OLD KV layer; no complete additional KV bank exists.
All nine phase pairs use alternating serial/concurrent order after warmup.
"""
import argparse
import csv
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_map

from .experiments import event, native_raw
from .openpi_backend import OpenPIBackend

SOURCE_BYTES = Path(__file__).read_bytes()
SOURCE_SHA256 = hashlib.sha256(SOURCE_BYTES).hexdigest()


def clone_tree(value):
    return tree_map(lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x, value)


def output_tensors(value):
    return [x for x in tree_flatten(value)[0] if isinstance(x, torch.Tensor)]


def cpu_outputs(value):
    return [x.detach().float().cpu().numpy().copy() for x in output_tensors(value)]


def difference(expected, actual):
    diffs = [np.abs(a.astype(np.float64)-b.astype(np.float64)).ravel()
             for a, b in zip(expected, actual, strict=True)]
    return dict(max_abs=max(float(d.max()) for d in diffs),
                mae=sum(float(d.sum()) for d in diffs)/sum(d.size for d in diffs))


class OneLayerKV:
    """Only the selected layer, retaining original old-KV tensor references."""
    def __init__(self, layer, pair):
        self.layer, self.pair = layer, tuple(pair)

    def __getitem__(self, layer):
        if layer != self.layer:
            raise IndexError("Stage screen may access only its selected OLD KV layer")
        return self.pair


def extract_phases(backend, current, nxt, noise, layer_index):
    from transformers.models.gemma import modeling_gemma
    model = backend.model.paligemma_with_expert
    layers = {"vlm": model.paligemma.language_model.layers[layer_index],
              "ae": model.gemma_expert.model.layers[layer_index]}
    saved, hooks = {}, []
    old = backend.build_prefix(current)
    old_layer = OneLayerKV(layer_index, old.cache[layer_index])
    old_before = cpu_outputs(old_layer.pair)
    original_attention = modeling_gemma.eager_attention_forward

    def capture_pure_attention(module, *args, **kwargs):
        for branch, layer in layers.items():
            name = branch+"_pure_attention"
            if module is layer.self_attn and name not in saved:
                # One layer's real Q/K/V, after RoPE and readonly prefix concat.
                saved[name] = (clone_tree(args), clone_tree(kwargs))
        return original_attention(module, *args, **kwargs)

    def attention_hook(branch):
        def record(module, args, kwargs):
            key = branch+"_attention"
            if key in saved:
                return
            # Never retain the full Cache object supplied to the forward hook.
            kw = {k: v for k, v in kwargs.items() if k != "past_key_value"}
            kw = clone_tree(kw)
            kw.update(use_cache=False, past_key_value=old_layer if branch == "ae" else None)
            saved[key] = (clone_tree(args), kw)
        return record

    def ffn_hook(branch):
        def record(module, args):
            key = branch+"_ffn"
            if key not in saved:
                saved[key] = clone_tree(args)
        return record

    for branch, layer in layers.items():
        hooks.append(layer.self_attn.register_forward_pre_hook(attention_hook(branch), with_kwargs=True))
        hooks.append(layer.mlp.register_forward_pre_hook(ffn_hook(branch)))
    try:
        modeling_gemma.eager_attention_forward = capture_pure_attention
        actions = backend.generate(old, current, noise)
        fresh = backend.build_prefix(nxt)
        torch.cuda.current_stream(backend.device).synchronize()
    finally:
        modeling_gemma.eager_attention_forward = original_attention
        for hook in hooks:
            hook.remove()
    del old, fresh, actions
    gc.collect()
    # Both complete forward caches are now gone. Only ONE original layer pair
    # remains for read-only AE attention, never a third complete KV snapshot.
    functions, metadata = {}, {}
    for branch, layer in layers.items():
        args, kwargs = saved[branch+"_attention"]
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        attn, mlp = layer.self_attn, layer.mlp

        def qkv(attn=attn, hidden=hidden):
            return attn.q_proj(hidden), attn.k_proj(hidden), attn.v_proj(hidden)

        def attention(attn=attn, args=args, kwargs=kwargs):
            return attn(*args, **kwargs)

        def ffn(mlp=mlp, args=saved[branch+"_ffn"]):
            return mlp(*args)

        def pure_attention(attn=attn, captured=saved[branch+"_pure_attention"]):
            return original_attention(attn, *captured[0], **captured[1])

        functions[branch] = dict(qkv=qkv, self_attention=attention, pure_attention=pure_attention, ffn=ffn)
        metadata[branch] = dict(hidden_shape=list(hidden.shape), dtype=str(hidden.dtype),
            qkv_widths=[attn.q_proj.out_features, attn.k_proj.out_features, attn.v_proj.out_features],
            ffn_input_shape=list(saved[branch+"_ffn"][0].shape),
            pure_attention_shapes=[list(x.shape) for x in output_tensors(saved[branch+"_pure_attention"])],
            attention_scope="QKV + RoPE + readonly KV concat (AE only) + eager attention + output projection")
    return functions, metadata, old_layer, old_before


class PhaseGraph:
    def __init__(self, function, stream, repetitions):
        self.stream = stream
        self.graph = torch.cuda.CUDAGraph()
        ready = event()
        ready.record()
        stream.wait_event(ready)
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                result = function()
            stream.synchronize()
            self.reference = cpu_outputs(result)
            del result
            with torch.cuda.graph(self.graph, stream=stream):
                for _ in range(repetitions):
                    self.output = function()
            self.graph.replay()
        stream.synchronize()
        self.error = difference(self.reference, cpu_outputs(self.output))
        if self.error["max_abs"] != 0:
            raise AssertionError(self.error)


def run(graphs, mode, order, release_delay_cycles=0):
    epoch = event()
    if release_delay_cycles:
        # Controlled capacity screen only: allow both host launches to queue
        # before the common GPU release. Excluded from epoch-based GPU times;
        # included in CPU completion. Never used by the inference runtime.
        torch.cuda._sleep(release_delay_cycles)
    epoch.record()
    events, jobs = {}, {}
    begin = time.perf_counter()
    selected = (order if mode in ("serial", "concurrent") else (mode.removesuffix("_only"),))
    last_done = None
    for branch in selected:
        phase = graphs[branch]
        start, done = event(), event()
        with torch.cuda.stream(phase.stream):
            phase.stream.wait_event(epoch)
            if mode == "serial" and last_done is not None:
                phase.stream.wait_event(last_done)
            torch.cuda.nvtx.range_push(branch)
            start.record()
            cpu = time.perf_counter()
            phase.graph.replay()
            cpu_ms = (time.perf_counter()-cpu)*1000
            done.record()
            torch.cuda.nvtx.range_pop()
        events[branch] = (start, done)
        jobs[branch] = dict(cpu_submit_ms=cpu_ms, done_at_submit=done.query())
        last_done = done
    submitted = time.perf_counter()
    for start, done in events.values():
        done.synchronize()
    end = time.perf_counter()
    row = dict(mode=mode, cpu_completion_ms=(end-begin)*1000,
               release_delay_cycles=release_delay_cycles,
               cpu_submit_total_ms=(submitted-begin)*1000, cpu_wait_ms=(end-submitted)*1000)
    for branch, (start, done) in events.items():
        row.update({branch+"_gpu_ms": start.elapsed_time(done),
                    branch+"_end_ms": epoch.elapsed_time(done),
                    branch+"_cpu_submit_ms": jobs[branch]["cpu_submit_ms"],
                    branch+"_done_at_submit": jobs[branch]["done_at_submit"]})
    row["both_gpu_complete_ms"] = max(epoch.elapsed_time(done) for start, done in events.values())
    return row


def stats(values):
    return dict(mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--bank", default="policies/realtime-vla-flash/results/idea1_v3_supplement/e1_expand_init7_10_20261020T0431Z/episodes/E1/supp_E1_D2_tau0.1_P2_Gconfirmed_disagreement_rejectD2/A/libero_spatial/task00/init07/real_predict_inputs_first25.pkl")
    parser.add_argument("--steps", type=int, choices=(1, 5, 10), default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Calls per graph; 1 preserves actual single-phase shape and work")
    parser.add_argument("--order", choices=("ae_first", "prefill_first"), default="ae_first")
    parser.add_argument("--ae-priority", type=int, choices=(0, -1), default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--phases", nargs="+", default=["qkv", "pure_attention", "ffn"],
                        choices=("qkv", "pure_attention", "ffn", "self_attention"))
    parser.add_argument("--pair-filter", help="Only pairs containing this phase (e.g. pure_attention)")
    parser.add_argument("--release-delay-cycles", type=int, default=0,
                        help="Controlled GPU release gate for phase-capacity diagnosis; CPU completion includes artificial delay")
    args = parser.parse_args()
    if min(args.pairs, args.repetitions) < 1:
        parser.error("pairs and repetitions must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"source.py").write_bytes(SOURCE_BYTES)
    from openpi.training import config
    from openpi.policies.policy_config import create_trained_policy
    cfg = config.get_config(args.config)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = create_trained_policy(cfg, args.checkpoint, pytorch_device="cuda:0",
                                   sample_kwargs={"num_steps": args.steps})
    backend = OpenPIBackend(policy, num_steps=args.steps)
    payload = Path(args.bank).read_bytes()
    records = pickle.loads(payload)
    current, nxt = [backend.prepare(native_raw(r)) for r in records[:2]]
    functions, shapes, old_layer, old_before = extract_phases(backend, current, nxt, backend.noise(1701), args.layer)
    streams = dict(vlm=torch.cuda.Stream(), ae=torch.cuda.Stream(priority=args.ae_priority))
    phases = {b: {name: PhaseGraph(fn, streams[b], args.repetitions) for name, fn in fns.items()
                  if name in args.phases}
              for b, fns in functions.items()}
    pools = [g.graph.pool() for branch in phases.values() for g in branch.values()]
    assert len(set(pools)) == 2*len(args.phases), "Every captured phase must own a private pool"
    rows, summaries = [], {}
    order = ("ae", "vlm") if args.order == "ae_first" else ("vlm", "ae")
    torch.cuda.reset_peak_memory_stats()
    for vlm_phase in phases["vlm"]:
        for ae_phase in phases["ae"]:
            if args.pair_filter and args.pair_filter not in (vlm_phase, ae_phase):
                continue
            name = vlm_phase+"__"+ae_phase
            pair = dict(vlm=phases["vlm"][vlm_phase], ae=phases["ae"][ae_phase])
            print("Stage pair", name, flush=True)
            for index in range(-args.warmup, args.pairs):
                modes = ("serial", "concurrent") if index % 2 == 0 else ("concurrent", "serial")
                for mode in modes:
                    row = run(pair, mode, order, args.release_delay_cycles)
                    row.update(pair=name, vlm_phase=vlm_phase, ae_phase=ae_phase, iteration=index)
                    if index >= 0:
                        rows.append(row)
            for mode in ("ae_only", "vlm_only"):
                for index in range(-args.warmup, args.pairs):
                    row = run(pair, mode, order, args.release_delay_cycles)
                    if index >= 0:
                        row.update(pair=name, vlm_phase=vlm_phase, ae_phase=ae_phase, iteration=index)
                        rows.append(row)
            errors = {b: difference(g.reference, cpu_outputs(g.output)) for b, g in pair.items()}
            assert all(e["max_abs"] == 0 for e in errors.values()), errors
            subset = [r for r in rows if r["pair"] == name]
            by_mode = {m: {key: stats([r[key] for r in subset if r["mode"] == m and key in r])
                          for key in ("both_gpu_complete_ms", "cpu_completion_ms", "ae_gpu_ms", "vlm_gpu_ms")
                          if any(r["mode"] == m and key in r for r in subset)}
                       for m in ("serial", "concurrent", "ae_only", "vlm_only")}
            summaries[name] = dict(metrics=by_mode, errors=errors,
                gpu_pair_speedup=by_mode["serial"]["both_gpu_complete_ms"]["p50"]/by_mode["concurrent"]["both_gpu_complete_ms"]["p50"],
                ae_concurrent_slowdown=by_mode["concurrent"]["ae_gpu_ms"]["p50"]/by_mode["ae_only"]["ae_gpu_ms"]["p50"],
                vlm_concurrent_slowdown=by_mode["concurrent"]["vlm_gpu_ms"]["p50"]/by_mode["vlm_only"]["vlm_gpu_ms"]["p50"])
            if args.profile:
                torch.cuda.profiler.start()
                for mode in ("serial", "concurrent"):
                    torch.cuda.nvtx.range_push("stage/"+name+"/"+mode)
                    run(pair, mode, order, args.release_delay_cycles)
                    torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
            (output/"progress.json").write_text(json.dumps(summaries, indent=2)+"\n")
    old_error = difference(old_before, cpu_outputs(old_layer.pair))
    assert old_error["max_abs"] == 0
    fields = sorted({k for row in rows for k in row})
    with (output/"measurements.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = dict(args=vars(args), backend=backend.identity, shapes=shapes, pairs=summaries,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        old_layer_error=old_error, numerical_tolerance=dict(max_abs=0, mae=0),
        graph_captures=len(pools), steady_recaptures=0, private_graph_pools=len(set(pools)),
        bank_sha256=hashlib.sha256(payload).hexdigest(),
        source_sha256=SOURCE_SHA256,
        limitation="Phase screen only; CUDA Event spans do not establish kernel overlap. CPU timings end at both GPU events; they omit real preprocessing/output copies. Whole-request and continuous-run validation is required.")
    (output/"result.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
