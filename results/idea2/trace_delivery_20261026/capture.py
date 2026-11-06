"""Instrumentation-only wrappers for three requested pi0.5 D10 traces.

All model, sampler, KV, and control implementations are imported unchanged.
Profiles exclude loading, compilation, graph capture, and ten warm requests.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import sys


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def native(args, rest):
    os.environ.update(JAX_PLATFORMS="cuda", JAX_PLATFORM_NAME="gpu")
    import torch
    from openpi.policies.policy import Policy
    from preparations.idea2 import cewp_native_baseline

    original = Policy.infer
    state = dict(path="original_OpenPI_JAX_pi05_D10", calls=0, captured=0,
                 warm_requests=args.warmup, requested_capture=args.count)

    @functools.wraps(original)
    def infer(self, *a, **kw):
        state["calls"] += 1
        index = state["calls"] - args.warmup - 1
        active = 0 <= index < args.count
        if index == 0:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            with torch.cuda.nvtx.range(f"NativeJAX/request_{index}/raw_observation_to_CPU_action"):
                result = original(self, *a, **kw)
            if active:
                state["captured"] += 1
            return result
        finally:
            if index == args.count - 1:
                torch.cuda.cudart().cudaProfilerStop()
                save(args.receipt, state)

    Policy.infer = infer
    sys.argv = [sys.argv[0], *rest]
    cewp_native_baseline.main()
    if state["captured"] != args.count:
        raise RuntimeError("Native capture window incomplete")


def server(args, rest):
    import torch
    from preparations.idea2.openpi_backend import OpenPIBackend
    from results.idea2.continuous_encoder_20261026.policy import ContinuousPolicy

    original_init = ContinuousPolicy.__init__
    original_predict = ContinuousPolicy.predict_ahead
    original_prepare = OpenPIBackend.prepare
    state = dict(path=args.kind, calls=0, captured=0, warm_requests=args.warmup,
                 requested_capture=args.count, current_request=None,
                 production_model_and_control_unmodified=True)

    @functools.wraps(original_init)
    def init(self, *a, **kw):
        original_init(self, *a, **kw)
        def factory(backend, prepared, noise):
            from preparations.idea2.progressive_kv import ProgressiveRuntime
            return ProgressiveRuntime(backend, prepared, noise, profile=True)
        # profile=True only adds the existing module NVTX labels at capture.
        self.runtime_factory = factory

    @functools.wraps(original_prepare)
    def prepare(self, *a, **kw):
        label = ("Input/Prepare[next_observation_for_request_"
                 + str(state["current_request"]) + "+1]")
        with torch.cuda.nvtx.range(label):
            return original_prepare(self, *a, **kw)

    @functools.wraps(original_predict)
    def predict(self, *a, **kw):
        state["calls"] += 1
        index = state["calls"] - args.warmup - 1
        active = 0 <= index < args.count
        state["current_request"] = kw.get("request_id")
        if index == 0:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            name = (f"{args.kind}/request_{kw.get('request_id')}/"
                    f"input_control_step_{kw.get('input_step')}")
            with torch.cuda.nvtx.range(name):
                result = original_predict(self, *a, **kw)
            if active:
                state["captured"] += 1
                state.setdefault("requests", []).append(dict(
                    request_id=kw.get("request_id"), input_step=kw.get("input_step"),
                    arm=self.arm, protocol=self.protocol, online=result["online"],
                    frontiers=result["frontiers"], timing=result["timing"],
                    actual_next_encoder=result["actual_next_encoder"]))
            return result
        finally:
            if index == args.count - 1:
                # Include any real next-Encoder tail. Only the last captured
                # RPC is delayed by this explicitly labelled diagnostic drain.
                with torch.cuda.nvtx.range("CaptureDrain/last_request_only"):
                    self.engine.drain()
                torch.cuda.cudart().cudaProfilerStop()
                save(args.receipt, state)

    ContinuousPolicy.__init__ = init
    ContinuousPolicy.predict_ahead = predict
    OpenPIBackend.prepare = prepare
    sys.argv = [sys.argv[0], *rest]
    if args.kind == "fresh_prepared":
        from results.idea2.continuous_encoder_20261026.server import main
    else:
        from results.idea2.control_alignment_20261026.server import main
    main()
    if state["captured"] != args.count:
        raise RuntimeError("Server capture window incomplete")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("native_jax", "fresh_prepared", "pipeline"), required=True)
    p.add_argument("--receipt", required=True)
    p.add_argument("--capture-warmup", dest="warmup", type=int, default=10)
    p.add_argument("--capture-count", dest="count", type=int, default=8)
    args, rest = p.parse_known_args()
    (native if args.kind == "native_jax" else server)(args, rest)


if __name__ == "__main__":
    main()
