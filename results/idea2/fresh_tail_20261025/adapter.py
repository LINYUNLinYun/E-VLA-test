"""Opt-in D10 fresh-KV tail, preserving the frozen progressive runtime.

The final K Euler rounds wait for the last layer's KV-ready Event BEFORE
snapshotting their frontier. This is all-fresh KV, not a wait for VLM FFN.
The Pool continues to use the independent full-prefix completion Event.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shlex
import sys

_ORIGINAL_GRAPHS = None
_ORIGINAL_RUNTIME = None
_INSTALLATIONS = []
_STATE = {}


def install(ae_sm_min=16, fresh_tail_steps=0, encoder_after_step=1):
    global _ORIGINAL_GRAPHS, _ORIGINAL_RUNTIME
    if ae_sm_min not in (0, 16, 32) or fresh_tail_steps not in (0, 1, 2):
        raise ValueError("Expected AE SM 0/16/32 and fresh tail 0/1/2")
    import torch
    from preparations.idea2 import progressive_kv as runtime
    from results.idea2.progressive_kv_20261025 import resource_adapter as resources
    if _ORIGINAL_GRAPHS is None:
        _ORIGINAL_GRAPHS, _ORIGINAL_RUNTIME = runtime.ProgressiveGraphs, runtime.ProgressiveRuntime
    runtime.ProgressiveGraphs, runtime.ProgressiveRuntime = _ORIGINAL_GRAPHS, _ORIGINAL_RUNTIME
    layouts, pending = resources.install(ae_sm_min, encoder_after_step)
    ResourceGraphs = runtime.ProgressiveGraphs

    class FreshTailGraphs(ResourceGraphs):
        def _capture_ae(self, index):
            if not fresh_tail_steps:
                return super()._capture_ae(index)
            slot = self.slots[index]
            if not 0 < fresh_tail_steps < self.steps:
                raise ValueError("Fresh tail must leave at least one mixed round")
            with torch.cuda.stream(self.ae_stream):
                slot.tail_wait_complete, = runtime.initialized_events(self.ae_stream, 1)
            original_snapshot = runtime.snapshot

            def tail_snapshot(frontier, history, step, *, override):
                if step == self.steps - fresh_tail_steps:
                    # All 18 new K/V pairs are immutable after this Event.
                    # The full VLM producer may still execute its final FFN.
                    with torch.cuda.nvtx.range(f"AE/fresh_tail_{fresh_tail_steps}/all_KV_ready_wait"):
                        if torch.cuda.is_current_stream_capturing():
                            self.external.wait(slot.layer_ready[-1], self.ae_stream)
                            self.external.mark(slot.tail_wait_complete, self.ae_stream)
                        else:
                            self.ae_stream.wait_event(slot.layer_ready[-1])
                            slot.tail_wait_complete.record(self.ae_stream)
                return original_snapshot(frontier, history, step, override=override)

            # Only capture-time Python is patched; runtime submission uses the
            # resulting immutable graph. Construction occurs on one Host.
            runtime.snapshot = tail_snapshot
            try:
                return super()._capture_ae(index)
            finally:
                runtime.snapshot = original_snapshot

    class FreshTailRuntime(_ORIGINAL_RUNTIME):
        def request(self, prepared, noise, **kwargs):
            if fresh_tail_steps and kwargs.get("force_old", False):
                raise ValueError("force_old conflicts with mandatory fresh tail")
            output, row, nxt = super().request(prepared, noise, **kwargs)
            row["fresh_tail_steps"] = fresh_tail_steps
            row["fresh_tail_wait_ms"] = 0.0
            if fresh_tail_steps:
                if row["frontiers"][-fresh_tail_steps:] != [self.graphs.num_layers] * fresh_tail_steps:
                    self.requires_reset = True
                    raise RuntimeError("Fresh-tail rounds consumed incomplete current KV")
                slot = self.graphs.slots[row["slot"]]
                row["fresh_tail_wait_ms"] = slot.iteration_marks[self.graphs.steps-fresh_tail_steps].elapsed_time(slot.tail_wait_complete)
            return output, row, nxt

    runtime.ProgressiveGraphs, runtime.ProgressiveRuntime = FreshTailGraphs, FreshTailRuntime
    _INSTALLATIONS.append((layouts, pending))
    _STATE.update(ae_sm_min=ae_sm_min, fresh_tail_steps=fresh_tail_steps,
                  encoder_after_step=encoder_after_step, actual_resources=layouts)
    return layouts, pending


def finish_contexts():
    """Call only AFTER the driver's frame and caller-owned tensors are gone."""
    import torch
    gc.collect()
    pairs = [pair for _, pending in _INSTALLATIONS for pair in pending]
    for pair in pairs:
        for stream in pair.streams:
            stream.synchronize()
    if pairs:
        with torch.cuda.device(pairs[0].device):
            torch.cuda.empty_cache()
    gc.collect()
    for pair in pairs:
        pair.close()
    for _, pending in _INSTALLATIONS:
        pending.clear()
    gc.collect()
    return dict(passed=True, contexts_closed=len(pairs),
                pairs_created=sum(len(layouts) for layouts, _ in _INSTALLATIONS))


def install_policy():
    from preparations.idea2.progressive_kv_policy import ProgressivePolicy
    original_protocol = ProgressivePolicy.protocol
    original_configure = ProgressivePolicy.configure

    def protocol(self):
        return dict(original_protocol.fget(self), fresh_tail_steps=_STATE["fresh_tail_steps"],
                    resource_adapter=dict(ae_sm_min=_STATE["ae_sm_min"], encoder_after_step=_STATE["encoder_after_step"],
                                          actual_resources=_STATE["actual_resources"]))

    def configure(self, mode, interval=1, ae_sm_min=None, fresh_tail_steps=None, **compatibility):
        sms = _STATE["ae_sm_min"] if ae_sm_min is None else ae_sm_min
        tail = _STATE["fresh_tail_steps"] if fresh_tail_steps is None else fresh_tail_steps
        if compatibility or sms not in (0, 16, 32) or tail not in (0, 1, 2) or interval != 1:
            raise ValueError("Invalid fresh-tail condition")
        if sms != _STATE["ae_sm_min"] or tail != _STATE["fresh_tail_steps"]:
            self.close()
            install(sms, tail)
        return original_configure(self, mode, interval=interval)

    ProgressivePolicy.protocol = property(protocol)
    ProgressivePolicy.configure = configure


def main():
    original_argv = list(sys.argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ae-sm-min", type=int, choices=(0, 16, 32), default=16)
    parser.add_argument("--fresh-tail-steps", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--encoder-after-step", type=int, choices=(1, 4, 7, 10), default=1)
    parser.add_argument("--serve", action="store_true")
    args, rest = parser.parse_known_args()
    layouts, _ = install(args.ae_sm_min, args.fresh_tail_steps, args.encoder_after_step)
    sys.argv = [sys.argv[0], *rest]
    cleanup = dict(passed=False)
    try:
        if args.serve:
            install_policy()
            from preparations.idea2.progressive_kv_policy import main as driver
        else:
            from results.idea2.progressive_kv_20261025.performance_bench import main as driver
        driver()
    except BaseException:
        cleanup["retained_due_to_exception"] = True
        raise
    else:
        cleanup = finish_contexts()
    finally:
        if "--output" in rest:
            path = Path(rest[rest.index("--output") + 1]) / "result.json"
            if path.exists():
                data = json.loads(path.read_text())
                data["fresh_tail_adapter"] = dict(ae_sm_min=args.ae_sm_min, fresh_tail_steps=args.fresh_tail_steps,
                    encoder_after_step=args.encoder_after_step, actual_resources=layouts, cleanup=cleanup,
                    ready_semantics="all current layer K/V ready; final VLM FFN need not be complete",
                    sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                data["command"] = shlex.join([sys.executable, "-m", "results.idea2.fresh_tail_20261025.adapter", *original_argv[1:]])
                path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
