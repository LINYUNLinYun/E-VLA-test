"""Only four application NVTX names: preprocess, encoder, vlm, AE.

Torch paths retain the existing computation and scheduling. Native JAX uses
explicit diagnostic JIT boundaries; it must not replace the whole-JIT baseline.
"""
import argparse
from contextlib import contextmanager
import functools
import json
from pathlib import Path
import sys


def simple_names(torch):
    original = torch.cuda.nvtx.range

    @contextmanager
    def simple(name, *args, **kwargs):
        if name in ("preprocess", "encoder", "vlm", "AE"):
            label = name
        elif name == "Prefill/VisionEncoder+Projector+Metadata":
            label = "encoder"
        elif name == "Prefill/VLM[release_KV_frontier]":
            label = "vlm"
        elif name == "AE/D10[per_round_frozen_frontier]":
            label = "AE"
        elif name.startswith("Input/Prepare["):
            label = "preprocess"
        else:
            label = None
        if label:
            with original(label):
                yield
        else:
            yield

    torch.cuda.nvtx.range = simple


def native(args, rest):
    import os
    os.environ.update(JAX_PLATFORMS="cuda", JAX_PLATFORM_NAME="gpu")
    import time
    import types
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from openpi.models import model as model_module
    from openpi.policies.policy import Policy
    from openpi.shared.nnx_utils import module_jit
    from preparations.idea1_v3.pi05_bench import image_stage, vlm_stage, ae_stage
    from preparations.idea2 import cewp_native_baseline

    simple_names(torch)
    original = Policy.infer
    state = dict(path="native_JAX_split_stage_diagnostic", calls=0, captured=0,
        original_whole_JIT_retained_elsewhere=True, separate_JIT_and_stage_fences=True,
        allowed_labels=["preprocess", "encoder", "vlm", "AE"],
        numerical_tolerance=dict(max_abs=1e-5, mae=1e-6), checks=[], stages=[])
    functions = {}

    def staged(self, obs, noise):
        if noise is None:
            raise ValueError("Explicit matched noise is required")
        timings = {}
        with torch.cuda.nvtx.range("preprocess"):
            start = time.perf_counter()
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self._input_transform(inputs)
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
            observation = model_module.preprocess_observation(None,
                model_module.Observation.from_dict(inputs), train=False)
            noisy = jnp.asarray(noise)
            if noisy.ndim == 2:
                noisy = noisy[None, ...]
            jax.block_until_ready((observation, noisy))
            timings["preprocess"] = (time.perf_counter()-start)*1000
        with torch.cuda.nvtx.range("encoder"):
            start = time.perf_counter()
            prefix = functions["encoder"](observation)
            jax.block_until_ready(prefix)
            timings["encoder"] = (time.perf_counter()-start)*1000
        with torch.cuda.nvtx.range("vlm"):
            start = time.perf_counter()
            cache = functions["vlm"](*prefix)
            jax.block_until_ready(cache)
            timings["vlm"] = (time.perf_counter()-start)*1000
        with torch.cuda.nvtx.range("AE"):
            start = time.perf_counter()
            raw = functions["AE"](observation, prefix[1], cache, noisy)
            jax.block_until_ready(raw)
            result = self._output_transform(dict(state=np.asarray(inputs["state"][0]),
                                                actions=np.asarray(raw[0])))
            timings["AE"] = (time.perf_counter()-start)*1000
        result["policy_timing"] = dict(infer_ms=sum(timings.values()))
        return result, timings

    @functools.wraps(original)
    def infer(self, obs, *, noise=None):
        if self._is_pytorch_model:
            raise ValueError("Native diagnostic requires the original JAX model")
        if not functions:
            if self._model.action_horizon != 10 or not self._model.pi05:
                raise ValueError("Expected complete pi0.5 H10")
            functions.update(encoder=module_jit(types.MethodType(image_stage, self._model)),
                             vlm=module_jit(types.MethodType(vlm_stage, self._model)),
                             AE=module_jit(types.MethodType(ae_stage, self._model)))
        state["calls"] += 1
        index = state["calls"]-args.warmup-1
        if index < 0:
            reference = original(self, obs, noise=noise)
            result, timing = staged(self, obs, noise)
            delta = np.abs(np.asarray(reference["actions"], np.float64)-np.asarray(result["actions"], np.float64))
            check = dict(max_abs=float(delta.max()), mae=float(delta.mean()))
            state["checks"].append(check)
            if check["max_abs"]>1e-5 or check["mae"]>1e-6:
                Path(args.receipt).write_text(json.dumps(state, indent=2)+"\n")
                raise ValueError("Separate-JIT diagnostic failed the predefined numerical check")
            return result
        if index == 0:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            result, timing = staged(self, obs, noise)
            if index < args.count:
                state["captured"] += 1
                state["stages"].append(timing)
            return result
        finally:
            if index == args.count-1:
                torch.cuda.cudart().cudaProfilerStop()
                Path(args.receipt).write_text(json.dumps(state, indent=2)+"\n")

    Policy.infer = infer
    sys.argv = [sys.argv[0], *rest]
    cewp_native_baseline.main()
    if state["captured"] != args.count:
        raise ValueError("Native diagnostic capture incomplete")


def server(args, rest):
    import torch
    from preparations.idea2.progressive_kv import ProgressiveGraphs
    from results.idea2.trace_delivery_20261026.capture import server as original_server
    simple_names(torch)
    original_init = ProgressiveGraphs.__init__

    @functools.wraps(original_init)
    def init(self, *a, **kw):
        # Only the existing three Graph-level names remain. No layer hooks.
        kw["profile"] = False
        return original_init(self, *a, **kw)

    ProgressiveGraphs.__init__ = init
    original_server(args, rest)


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
